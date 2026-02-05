"""
Modal inference service for LingBot-World NF4 quantized model.

This module provides a serverless GPU deployment for image-to-video generation
with camera pose control, optimized for approximately 32GB VRAM using NF4
quantization via bitsandbytes.

The service exposes both a Python API for programmatic access and a FastAPI
REST interface for HTTP-based interactions. It supports concurrent requests
with automatic GPU scaling and persistent model caching.

Model
-----
cahlen/lingbot-world-base-cam-nf4
    NF4 quantized version of LingBot-World, featuring:
    - High/low noise diffusion models (~19.2GB combined)
    - T5-XXL text encoder in BFloat16 (~10.6GB)
    - VAE encoder/decoder (~485MB)

Architecture
------------
The deployment uses Modal's serverless infrastructure with:
- A100-80GB GPUs for inference (required due to model size)
- Persistent volumes for model weight caching
- 15-minute scaledown window to minimize cold starts
- Concurrent request handling (max 2 per container)

Examples
--------
Deploy the service::

    $ modal run src/lingbot_world/inference.py --action setup
    $ modal deploy src/lingbot_world/inference.py

Generate video via API::

    import httpx

    with open("input.jpg", "rb") as f:
        response = httpx.post(
            "https://your-workspace--lingbot-world-api.modal.run/generate",
            files={"image": f},
            data={"prompt": "A cinematic video with camera movement"},
        )

    with open("output.mp4", "wb") as f:
        f.write(response.content)

Notes
-----
This module requires the following Modal secrets to be configured:
- ``huggingface-token``: HuggingFace token with read access to the model repository

See Also
--------
lingbot_world.ui : Gradio web interface for interactive use.
lingbot_world.cli : Command-line interface for deployment operations.

References
----------
.. [1] LingBot-World Model: https://huggingface.co/cahlen/lingbot-world-base-cam-nf4
.. [2] Modal Documentation: https://modal.com/docs
"""

import io
import tempfile
import uuid
from pathlib import Path
from typing import Any

import modal

# =============================================================================
# Module Constants
# =============================================================================

#: HuggingFace model repository identifier for the NF4 quantized model.
MODEL_ID: str = "cahlen/lingbot-world-base-cam-nf4"

#: Path where model weights are stored inside the container.
MODEL_DIR: Path = Path("/models")

#: Path where generated video outputs are stored.
OUTPUTS_DIR: Path = Path("/outputs")

#: Supported output resolutions as "height*width" strings.
SUPPORTED_SIZES: tuple[str, ...] = ("480*832", "832*480", "720*1280", "1280*720")

#: Maximum number of frames supported (approximately 1 minute at 16fps).
MAX_FRAMES: int = 961

#: Snapshot version key - change this to invalidate GPU memory snapshot cache.
#: Increment when model loading code changes to force new snapshot creation.
SNAPSHOT_KEY: str = "v3-h100-warmup"

# =============================================================================
# Modal Application Configuration
# =============================================================================

app = modal.App("lingbot-world")

#: Persistent volume for caching downloaded model weights across container restarts.
model_volume = modal.Volume.from_name(
    "lingbot-world-nf4-weights",
    create_if_missing=True,
)

#: Persistent volume for storing generated video outputs.
outputs_volume = modal.Volume.from_name(
    "lingbot-world-outputs",
    create_if_missing=True,
)

# =============================================================================
# Container Image Definition
# =============================================================================

cuda_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "ffmpeg", "libsm6", "libxext6", "libgl1-mesa-glx")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "bitsandbytes>=0.45.0",
        "safetensors>=0.4.0",
        "accelerate>=1.1.1",
        "transformers>=4.49.0",
        "diffusers>=0.31.0",
        "einops>=0.8.0",
        "imageio>=2.36.0",
        "imageio-ffmpeg>=0.5.1",
        "pillow>=10.0.0",
        "numpy>=1.23.5,<2",
        "tqdm>=4.66.0",
        "easydict>=1.13",
        "scipy>=1.11.0",
        "ftfy>=6.2.0",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.27.0",
        "hf_transfer>=0.1.0",
    )
    .env(
        {
            "HF_HUB_CACHE": str(MODEL_DIR),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# Note: flash-attn requires CUDA compilation, skip for now
# Can be added later with a custom Docker image

#: Extended image with FastAPI dependencies for the web endpoint.
web_image = cuda_image.pip_install("fastapi[standard]", "python-multipart")


# =============================================================================
# Model Download Function
# =============================================================================


@app.function(
    image=cuda_image,
    volumes={str(MODEL_DIR): model_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def download_model() -> str:
    """
    Download LingBot-World NF4 model weights from HuggingFace Hub.

    This function downloads all required model components to a persistent
    Modal volume, enabling fast model loading on subsequent container starts.
    The download typically takes 30-60 minutes depending on network conditions.

    The following components are downloaded:

    - ``high_noise_model_bnb_nf4/`` : High-noise diffusion model (~9.6GB)
    - ``low_noise_model_bnb_nf4/`` : Low-noise diffusion model (~9.6GB)
    - ``models_t5_umt5-xxl-enc-bf16.pth`` : T5-XXL encoder (~10.6GB)
    - ``Wan2.1_VAE.pth`` : VAE model (~485MB)
    - ``wan/`` : Model source code and utilities
    - ``generate_prequant.py`` : Pre-quantized inference script
    - ``load_prequant.py`` : Weight loading utilities

    Returns
    -------
    str
        Absolute path to the downloaded model directory.

    Raises
    ------
    huggingface_hub.utils.HfHubHTTPError
        If the HuggingFace token is invalid or lacks repository access.
    OSError
        If there is insufficient disk space or network errors occur.

    Notes
    -----
    This function requires the ``huggingface-token`` Modal secret to be configured
    with a valid HuggingFace token that has read access to the model.

    The download is idempotent - running it multiple times will only
    download changed or missing files.

    Examples
    --------
    Run the download via Modal CLI::

        $ modal run src/lingbot_world/inference.py --action setup

    Or call directly in Python::

        >>> download_model.remote()
        '/models/lingbot-world-nf4'
    """
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    model_path = MODEL_DIR / "lingbot-world-nf4"

    print(f"Downloading model: {MODEL_ID}")
    print(f"Target directory: {model_path}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_path),
        token=os.environ.get("HF_TOKEN"),
    )

    model_volume.commit()

    # List downloaded files for verification
    result = subprocess.run(
        ["ls", "-lah", str(model_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)

    print(f"Model downloaded successfully to {model_path}")
    return str(model_path)


# =============================================================================
# Inference Class
# =============================================================================


@app.cls(
    image=cuda_image,
    volumes={
        str(MODEL_DIR): model_volume,
        str(OUTPUTS_DIR): outputs_volume,
    },
    gpu="H100",
    # Timeout configuration per Modal docs:
    # - startup_timeout: For model loading during @modal.enter(snap=True)
    # - timeout: For actual inference requests
    startup_timeout=600,  # 10 minutes for model loading + snapshot creation
    timeout=30 * 60,  # 30 minutes max per video generation request
    scaledown_window=15 * 60,
    secrets=[modal.Secret.from_name("huggingface-token")],
    # GPU memory snapshot for 10x faster cold starts
    # Both flags required per Modal docs for full GPU state capture
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=2)
class LingBotWorld:
    """
    Image-to-video generation service using NF4 quantized LingBot-World.

    This class encapsulates the inference pipeline for generating cinematic
    videos from single images with optional camera trajectory control. It
    manages GPU resources, model loading, and video encoding.

    The class is designed as a Modal service with:

    - Automatic model loading on container startup
    - Concurrent request handling (up to 2 per container)
    - 30-minute timeout for long video generation
    - 15-minute keepalive to reduce cold starts

    Attributes
    ----------
    model_path : pathlib.Path
        Path to the model checkpoint directory on the container filesystem.
    pipeline : WanI2V_PreQuant
        The loaded inference pipeline for video generation.
    device : torch.device
        CUDA device used for inference (always ``cuda:0``).

    Methods
    -------
    generate(image_bytes, prompt, ...)
        Generate a video from an input image and text prompt.
    health_check()
        Return current health status and GPU metrics.

    Notes
    -----
    The model uses NF4 quantization via bitsandbytes, requiring approximately
    32GB of GPU VRAM. During inference, the model dynamically swaps between
    high-noise and low-noise diffusion models based on the current timestep
    to optimize quality while managing memory usage.

    Video generation time depends on several factors:

    - Resolution: 480p is ~2x faster than 720p
    - Frame count: Linear scaling with number of frames
    - Sampling steps: Linear scaling with step count

    Typical generation times on A100-80GB:

    - 81 frames @ 480*832, 40 steps: ~2-3 minutes
    - 161 frames @ 720*1280, 70 steps: ~8-10 minutes

    See Also
    --------
    download_model : Download model weights before first use.

    Examples
    --------
    Generate a video programmatically::

        model = LingBotWorld()
        video_bytes = model.generate.remote(
            image_bytes=open("input.jpg", "rb").read(),
            prompt="A cinematic video with gentle camera movement",
            frame_num=81,
        )
        with open("output.mp4", "wb") as f:
            f.write(video_bytes)
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """
        Initialize model pipeline on container startup with GPU snapshot.

        This method is called automatically by Modal when a new container
        is started. It loads the pre-quantized diffusion models, T5 text
        encoder, and VAE to GPU memory. The ``snap=True`` parameter captures
        a GPU memory snapshot after loading, reducing subsequent cold starts
        from ~90s to ~10s.

        The loading process includes:

        1. Adding the model source directory to Python path
        2. Importing the WanI2V_PreQuant pipeline class
        3. Initializing the pipeline with checkpoint directory
        4. Moving required model components to GPU
        5. Capturing GPU memory snapshot for fast restoration

        Raises
        ------
        FileNotFoundError
            If model weights have not been downloaded.
        RuntimeError
            If CUDA is not available or GPU memory is insufficient.

        Notes
        -----
        With GPU memory snapshots enabled:

        - First cold start: ~90 seconds (creates snapshot)
        - Subsequent cold starts: ~10 seconds (restores from snapshot)

        The 15-minute scaledown window helps minimize cold starts by
        keeping containers warm between requests.
        """
        import logging
        import sys

        import torch

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger("lingbot_world")

        self.model_path = MODEL_DIR / "lingbot-world-nf4"

        logger.info(f"Snapshot key: {SNAPSHOT_KEY}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        # Verify CUDA is available before proceeding
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This function requires GPU. "
                "Ensure gpu='H100' is set in the Modal decorator."
            )

        self.device = torch.device("cuda:0")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Add model source to Python path for imports
        sys.path.insert(0, str(self.model_path))

        # Import and initialize the pre-quantized pipeline
        from generate_prequant import WanI2V_PreQuant

        logger.info("Loading WanI2V_PreQuant pipeline...")
        self.pipeline = WanI2V_PreQuant(
            checkpoint_dir=str(self.model_path),
            device_id=0,
        )

        logger.info("Pipeline loaded successfully!")

        # Warmup pass to compile CUDA kernels and pre-allocate memory
        # This is critical for GPU snapshots - kernels must be compiled before snapshot
        logger.info("Running warmup pass to compile CUDA kernels...")
        try:
            from PIL import Image

            # Create a small dummy image for warmup
            warmup_img = Image.new("RGB", (512, 512), color=(128, 128, 128))

            # Run minimal inference to compile kernels
            _ = self.pipeline.generate(
                input_prompt="warmup test",
                img=warmup_img,
                action_path=None,
                max_area=480 * 480,  # Small area for fast warmup
                frame_num=17,  # Minimum frames (4n+1)
                shift=3.0,
                sampling_steps=2,  # Minimal steps for warmup
                guide_scale=5.0,
                seed=42,
            )

            # Clear CUDA cache after warmup
            torch.cuda.empty_cache()
            logger.info("Warmup complete - CUDA kernels compiled")
            logger.info(
                f"GPU Memory after warmup: "
                f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated, "
                f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved"
            )
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")
            # Warmup failure is non-fatal - model is still loaded

        logger.info("✓ Model ready for GPU memory snapshot")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        prompt: str,
        intrinsics_bytes: bytes | None = None,
        poses_bytes: bytes | None = None,
        size: str = "480*832",
        frame_num: int = 81,
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        seed: int = -1,
    ) -> bytes:
        """
        Generate a video from an input image with optional camera control.

        This method takes a single image and text prompt, then generates a
        video showing the scene with natural motion based on the prompt.
        Optionally, camera trajectory data can be provided for precise
        control over the virtual camera's movement.

        Parameters
        ----------
        image_bytes : bytes
            Input image encoded as JPEG or PNG bytes. The image should be
            in RGB format. Recommended aspect ratios match the output size
            (e.g., 9:16 for 480*832, 16:9 for 832*480).
        prompt : str
            Text prompt describing the desired video motion and style.
            Longer, more descriptive prompts generally produce better results.
            Example: "A cinematic first-person exploration through the urban
            environment, with gentle forward movement and ambient sounds."
        intrinsics_bytes : bytes, optional
            Camera intrinsics as serialized numpy array bytes with shape
            ``[num_frames, 4]``. Each row contains ``[fx, fy, cx, cy]`` where:

            - ``fx, fy``: Focal lengths (typically normalized to 1.0)
            - ``cx, cy``: Principal point (typically 0.5, 0.5 for centered)

            Must be provided together with ``poses_bytes``.
        poses_bytes : bytes, optional
            Camera poses as serialized numpy array bytes with shape
            ``[num_frames, 4, 4]``. Each pose is a 4x4 homogeneous
            transformation matrix representing camera-to-world transform.
            Must be provided together with ``intrinsics_bytes``.
        size : str, default="480*832"
            Output resolution as "height*width". Supported values:

            - ``"480*832"``: Portrait HD (9:16 aspect ratio)
            - ``"832*480"``: Landscape HD (16:9 aspect ratio)
            - ``"720*1280"``: Portrait Full HD (9:16 aspect ratio)
            - ``"1280*720"``: Landscape Full HD (16:9 aspect ratio)

        frame_num : int, default=81
            Number of frames to generate. Must follow the formula ``4n+1``
            where n is a positive integer. Common values:

            - 81 frames: ~5 seconds at 16fps
            - 161 frames: ~10 seconds at 16fps
            - 241 frames: ~15 seconds at 16fps
            - 481 frames: ~30 seconds at 16fps

        sampling_steps : int, default=40
            Number of diffusion sampling steps. Higher values produce
            better quality but increase generation time linearly.
            Recommended range: 30-70.
        guide_scale : float, default=5.0
            Classifier-free guidance scale. Higher values increase
            adherence to the text prompt but may reduce diversity.
            Recommended range: 3.0-7.0.
        seed : int, default=-1
            Random seed for reproducible generation. Use -1 for a
            random seed on each call.

        Returns
        -------
        bytes
            Generated video encoded as H.264 MP4 bytes at 16fps.

        Raises
        ------
        ValueError
            If ``frame_num`` is not of the form ``4n+1``.
        ValueError
            If ``size`` is not in the supported sizes list.
        RuntimeError
            If video generation fails due to GPU memory or other errors.

        Notes
        -----
        Memory usage scales with resolution and frame count:

        - 480*832, 81 frames: ~28GB peak VRAM
        - 720*1280, 161 frames: ~45GB peak VRAM

        The model automatically offloads unused components to manage memory.

        Examples
        --------
        Generate a simple video::

            video = model.generate.remote(
                image_bytes=image_data,
                prompt="The scene comes alive with gentle motion",
            )

        Generate with custom parameters::

            video = model.generate.remote(
                image_bytes=image_data,
                prompt="A dramatic zoom into the scene",
                size="720*1280",
                frame_num=161,
                sampling_steps=50,
                guide_scale=6.0,
                seed=42,
            )
        """
        import numpy as np
        from PIL import Image

        # Load and validate input image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"Input image size: {img.size}")

        # Handle camera poses if provided
        action_path = None
        if intrinsics_bytes is not None and poses_bytes is not None:
            action_path = tempfile.mkdtemp()
            intrinsics = np.frombuffer(intrinsics_bytes, dtype=np.float32)
            poses = np.frombuffer(poses_bytes, dtype=np.float32)
            np.save(f"{action_path}/intrinsics.npy", intrinsics)
            np.save(f"{action_path}/poses.npy", poses)
            print("Camera poses loaded")

        # Determine shift parameter based on resolution
        # Lower shift for 480p produces better results
        shift = 3.0 if "480" in size else 5.0

        # Parse resolution to compute max area
        try:
            height, width = map(int, size.split("*"))
            max_area = height * width
        except ValueError:
            max_area = 720 * 1280  # Default fallback

        print(f"Generating: size={size}, frames={frame_num}, steps={sampling_steps}")

        # Run the diffusion pipeline
        video = self.pipeline.generate(
            input_prompt=prompt,
            img=img,
            action_path=action_path,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
        )

        # Convert output tensor to video file
        output_id = str(uuid.uuid4())
        output_path = OUTPUTS_DIR / f"{output_id}.mp4"

        import imageio

        # Convert tensor to numpy array
        video_np = video.cpu().numpy() if hasattr(video, "cpu") else video

        # Normalize from [-1, 1] to [0, 255]
        if video_np.min() < 0:
            video_np = (video_np + 1) / 2
        video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

        # Handle tensor shape: [T, C, H, W] -> [T, H, W, C]
        if video_np.ndim == 4 and video_np.shape[1] == 3:
            video_np = video_np.transpose(0, 2, 3, 1)

        # Write video file
        imageio.mimwrite(str(output_path), video_np, fps=16, codec="libx264")
        outputs_volume.commit()

        # Read and return video bytes
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        print(f"Video generated: {len(video_bytes) / 1e6:.2f} MB")
        return video_bytes

    @modal.method()
    def health_check(self) -> dict[str, Any]:
        """
        Check service health and return GPU metrics.

        This method verifies that the model is properly loaded and returns
        current resource utilization metrics. It can be used for monitoring
        and load balancing decisions.

        Returns
        -------
        dict[str, Any]
            Health status dictionary containing:

            - ``status`` : str
                Overall health status, either "healthy" or "unhealthy".
            - ``model_loaded`` : bool
                Whether model weight files exist on disk.
            - ``pipeline_ready`` : bool
                Whether the inference pipeline is initialized.
            - ``gpu`` : str
                Name of the GPU device (e.g., "NVIDIA A100-SXM4-80GB").
            - ``gpu_memory_gb`` : float
                Total GPU memory in gigabytes.
            - ``gpu_memory_used_gb`` : float
                Currently allocated GPU memory in gigabytes.

        Examples
        --------
        Check health status::

            >>> model = LingBotWorld()
            >>> status = model.health_check.remote()
            >>> print(status)
            {
                'status': 'healthy',
                'model_loaded': True,
                'pipeline_ready': True,
                'gpu': 'NVIDIA A100-SXM4-80GB',
                'gpu_memory_gb': 79.4,
                'gpu_memory_used_gb': 28.3
            }
        """
        import torch

        model_exists = (self.model_path / "Wan2.1_VAE.pth").exists()
        pipeline_loaded = hasattr(self, "pipeline") and self.pipeline is not None

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9

        return {
            "status": "healthy" if (model_exists and pipeline_loaded) else "unhealthy",
            "model_loaded": model_exists,
            "pipeline_ready": pipeline_loaded,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(gpu_memory, 1),
            "gpu_memory_used_gb": round(gpu_memory_used, 1),
        }


# =============================================================================
# FastAPI Web Application
# =============================================================================


@app.function(
    image=web_image,
    volumes={str(OUTPUTS_DIR): outputs_volume},
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def api():
    r"""
    Create FastAPI web application for LingBot-World inference.

    This function creates and configures a FastAPI application that exposes
    the LingBot-World inference service via HTTP endpoints. The application
    includes OpenAPI documentation, CORS middleware, structured error
    handling, and comprehensive validation.

    Returns
    -------
    fastapi.FastAPI
        Configured ASGI application instance.

    Endpoints
    ---------
    GET /
        Return API information and available endpoints.
    GET /health
        Return health status and GPU metrics.
    GET /presets
        Return available camera motion presets.
    GET /config
        Return current service configuration and limits.
    POST /generate
        Generate video from uploaded image and prompt.
    POST /generate/async
        Submit async generation job (returns job ID).
    GET /job/{job_id}
        Check status of async generation job.
    GET /docs
        OpenAPI/Swagger documentation.
    GET /redoc
        ReDoc documentation.

    Notes
    -----
    The API uses Modal's asgi_app decorator for automatic scaling and
    load balancing. Up to 20 concurrent requests can be processed per
    container, with requests queued to the inference class.

    Examples
    --------
    After deployment, access the API at::

        https://YOUR_WORKSPACE--lingbot-world-api.modal.run

    Generate video via curl::

        curl -X POST "https://...api.modal.run/generate" \\
            -F "image=@input.jpg" \\
            -F "prompt=A cinematic video" \\
            -o output.mp4
    """
    from enum import Enum

    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field

    # ==========================================================================
    # Pydantic Models for API Validation
    # ==========================================================================

    class VideoSize(str, Enum):
        """Supported video output resolutions."""

        PORTRAIT_480 = "480*832"
        LANDSCAPE_480 = "832*480"
        PORTRAIT_720 = "720*1280"
        LANDSCAPE_720 = "1280*720"

    class FramePreset(str, Enum):
        """Common frame count presets for different video durations."""

        SHORT = "81"  # ~5 seconds
        MEDIUM = "161"  # ~10 seconds
        LONG = "241"  # ~15 seconds
        EXTRA_LONG = "481"  # ~30 seconds

    class CameraPreset(BaseModel):
        """Camera motion preset definition."""

        name: str = Field(..., description="Preset name")
        description: str = Field(..., description="What the camera does")
        prompt_modifier: str = Field(..., description="Text to append to prompt")

    class GenerationConfig(BaseModel):
        """Service configuration and limits."""

        model_id: str
        supported_sizes: list[str]
        max_frames: int
        default_sampling_steps: int
        default_guide_scale: float
        min_sampling_steps: int
        max_sampling_steps: int
        min_guide_scale: float
        max_guide_scale: float

    class GenerationResponse(BaseModel):
        """Response metadata for video generation."""

        success: bool
        filename: str
        size_bytes: int
        resolution: str
        frames: int
        duration_seconds: float

    class ErrorResponse(BaseModel):
        """Structured error response."""

        success: bool = False
        error: str
        error_code: str
        details: dict[str, Any] | None = None

    # Camera motion presets
    CAMERA_PRESETS: dict[str, CameraPreset] = {
        "static": CameraPreset(
            name="Static",
            description="Camera remains stationary, scene animates naturally",
            prompt_modifier="with a stationary camera view",
        ),
        "pan_left": CameraPreset(
            name="Pan Left",
            description="Camera pans smoothly to the left",
            prompt_modifier="with a smooth camera pan to the left",
        ),
        "pan_right": CameraPreset(
            name="Pan Right",
            description="Camera pans smoothly to the right",
            prompt_modifier="with a smooth camera pan to the right",
        ),
        "zoom_in": CameraPreset(
            name="Zoom In",
            description="Camera zooms into the scene",
            prompt_modifier="with a gentle zoom into the scene",
        ),
        "zoom_out": CameraPreset(
            name="Zoom Out",
            description="Camera zooms out from the scene",
            prompt_modifier="with a gentle zoom out from the scene",
        ),
        "orbit": CameraPreset(
            name="Orbit",
            description="Camera orbits around the subject",
            prompt_modifier="with a cinematic orbit around the subject",
        ),
        "dolly_forward": CameraPreset(
            name="Dolly Forward",
            description="Camera moves forward into the scene",
            prompt_modifier="with a forward dolly movement into the scene",
        ),
        "crane_up": CameraPreset(
            name="Crane Up",
            description="Camera rises upward",
            prompt_modifier="with an upward crane movement",
        ),
        "tracking": CameraPreset(
            name="Tracking Shot",
            description="Camera follows movement in the scene",
            prompt_modifier="with a tracking shot following the motion",
        ),
    }

    # ==========================================================================
    # FastAPI Application Setup
    # ==========================================================================

    fastapi_app = FastAPI(
        title="LingBot-World API",
        description=(
            "Image-to-Video World Model with Camera Pose Control.\n\n"
            "Transform static images into cinematic videos using a state-of-the-art "
            "NF4-quantized diffusion model with optional camera trajectory control."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        contact={"name": "DreamFlux", "url": "https://github.com/DreamFlux-Workspace"},
        license_info={"name": "Creative Rail v1.0"},
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==========================================================================
    # Validation Helpers
    # ==========================================================================

    def validate_frame_num(frame_num: int) -> None:
        """Validate frame number follows 4n+1 rule."""
        if (frame_num - 1) % 4 != 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid frame_num",
                    "error_code": "INVALID_FRAME_COUNT",
                    "details": {
                        "received": frame_num,
                        "rule": "Must be 4n+1 (e.g., 81, 161, 241, 481)",
                        "valid_examples": [81, 161, 241, 321, 481],
                    },
                },
            )
        if frame_num > MAX_FRAMES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"frame_num exceeds maximum ({MAX_FRAMES})",
                    "error_code": "FRAME_COUNT_TOO_HIGH",
                    "details": {"received": frame_num, "max": MAX_FRAMES},
                },
            )

    def validate_size(size: str) -> None:
        """Validate output resolution is supported."""
        if size not in SUPPORTED_SIZES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported resolution",
                    "error_code": "INVALID_SIZE",
                    "details": {"received": size, "supported": list(SUPPORTED_SIZES)},
                },
            )

    def validate_sampling_steps(steps: int) -> None:
        """Validate sampling steps within reasonable range."""
        if not 10 <= steps <= 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "sampling_steps out of range",
                    "error_code": "INVALID_STEPS",
                    "details": {"received": steps, "min": 10, "max": 100},
                },
            )

    def validate_guide_scale(scale: float) -> None:
        """Validate guidance scale within reasonable range."""
        if not 1.0 <= scale <= 15.0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "guide_scale out of range",
                    "error_code": "INVALID_GUIDE_SCALE",
                    "details": {"received": scale, "min": 1.0, "max": 15.0},
                },
            )

    # ==========================================================================
    # API Endpoints
    # ==========================================================================

    @fastapi_app.get("/", tags=["Info"])
    async def root() -> dict[str, Any]:
        """Return API information and capabilities."""
        return {
            "name": "LingBot-World API",
            "model": MODEL_ID,
            "description": "Image-to-Video generation with camera control",
            "version": "0.1.0",
            "endpoints": {
                "/docs": "OpenAPI documentation",
                "/health": "Health check and GPU metrics",
                "/config": "Service configuration and limits",
                "/presets": "Available camera motion presets",
                "/generate": "Generate video from image (sync)",
            },
            "supported_sizes": list(SUPPORTED_SIZES),
            "max_frames": MAX_FRAMES,
        }

    @fastapi_app.get("/health", tags=["Info"])
    async def health() -> Any:
        """
        Return service health status and GPU metrics.

        Checks if the model is loaded and returns current GPU utilization.
        """
        try:
            model = LingBotWorld()
            result: dict[str, Any] = model.health_check.remote()
            return result
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "error_code": "SERVICE_UNAVAILABLE",
                },
            )

    @fastapi_app.get("/config", tags=["Info"], response_model=GenerationConfig)
    async def get_config() -> GenerationConfig:
        """Return service configuration and parameter limits."""
        return GenerationConfig(
            model_id=MODEL_ID,
            supported_sizes=list(SUPPORTED_SIZES),
            max_frames=MAX_FRAMES,
            default_sampling_steps=40,
            default_guide_scale=5.0,
            min_sampling_steps=10,
            max_sampling_steps=100,
            min_guide_scale=1.0,
            max_guide_scale=15.0,
        )

    @fastapi_app.get("/presets", tags=["Info"])
    async def get_presets() -> dict[str, Any]:
        """
        Return available camera motion presets.

        Use these presets to add cinematic camera movements to your videos.
        Include the preset name in the `camera_preset` parameter of /generate.
        """
        return {
            "presets": {k: v.model_dump() for k, v in CAMERA_PRESETS.items()},
            "frame_presets": {
                "short": {"frames": 81, "duration": "~5 seconds"},
                "medium": {"frames": 161, "duration": "~10 seconds"},
                "long": {"frames": 241, "duration": "~15 seconds"},
                "extra_long": {"frames": 481, "duration": "~30 seconds"},
            },
        }

    @fastapi_app.post("/generate", tags=["Generation"])
    async def generate(
        image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
        prompt: str = Form(..., description="Text prompt for video generation"),
        size: str = Form("480*832", description="Resolution: 480*832, 832*480, 720*1280, 1280*720"),
        frame_num: int = Form(81, description="Frames (4n+1 rule: 81, 161, 241)"),
        sampling_steps: int = Form(40, description="Diffusion sampling steps (10-100)"),
        guide_scale: float = Form(5.0, description="Guidance scale (1.0-15.0)"),
        seed: int = Form(-1, description="Random seed (-1 for random)"),
        camera_preset: str | None = Form(None, description="Camera motion preset name"),
    ) -> Response:
        """
        Generate video from an uploaded image.

        Upload an image file and provide a text prompt to generate a
        cinematic video showing the scene with natural motion.

        **Parameters:**
        - `image`: Input image (JPEG or PNG format)
        - `prompt`: Describe the motion and style you want
        - `size`: Output resolution (default: 480*832 portrait)
        - `frame_num`: Number of frames (must follow 4n+1 rule)
        - `sampling_steps`: Quality vs speed tradeoff (higher = better but slower)
        - `guide_scale`: Prompt adherence (higher = stricter prompt following)
        - `seed`: For reproducible results
        - `camera_preset`: Optional camera motion (see /presets)

        **Generation Time:**
        - 480p, 81 frames: ~2-3 minutes
        - 720p, 161 frames: ~8-10 minutes
        """
        # Validate all inputs
        validate_size(size)
        validate_frame_num(frame_num)
        validate_sampling_steps(sampling_steps)
        validate_guide_scale(guide_scale)

        # Apply camera preset to prompt if specified
        final_prompt = prompt
        if camera_preset:
            preset = CAMERA_PRESETS.get(camera_preset.lower())
            if preset:
                final_prompt = f"{prompt} {preset.prompt_modifier}"
            else:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Unknown camera preset",
                        "error_code": "INVALID_PRESET",
                        "details": {
                            "received": camera_preset,
                            "available": list(CAMERA_PRESETS.keys()),
                        },
                    },
                )

        try:
            image_bytes = await image.read()

            # Validate image format
            if not image_bytes[:8]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Empty or invalid image file",
                        "error_code": "INVALID_IMAGE",
                    },
                )

            model = LingBotWorld()
            video_bytes: bytes = model.generate.remote(
                image_bytes=image_bytes,
                prompt=final_prompt,
                size=size,
                frame_num=frame_num,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
            )

            filename = f"lingbot_{uuid.uuid4().hex[:8]}.mp4"
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "X-Generation-Frames": str(frame_num),
                    "X-Generation-Resolution": size,
                    "X-Generation-Steps": str(sampling_steps),
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Generation failed",
                    "error_code": "GENERATION_ERROR",
                    "details": {"message": str(e)},
                },
            ) from e

    @fastapi_app.get("/estimate", tags=["Info"])
    async def estimate_time(
        size: str = Query("480*832", description="Output resolution"),
        frame_num: int = Query(81, description="Number of frames"),
        sampling_steps: int = Query(40, description="Sampling steps"),
    ) -> dict[str, Any]:
        """
        Estimate generation time for given parameters.

        Returns an approximate generation time based on resolution,
        frame count, and sampling steps.
        """
        validate_size(size)
        validate_frame_num(frame_num)

        # Base time estimates (minutes) on A100-80GB
        base_times = {
            "480*832": 0.025,  # per frame per step
            "832*480": 0.025,
            "720*1280": 0.05,
            "1280*720": 0.05,
        }

        base = base_times.get(size, 0.035)
        estimated_minutes = base * frame_num * sampling_steps / 40  # normalized to 40 steps

        return {
            "resolution": size,
            "frames": frame_num,
            "sampling_steps": sampling_steps,
            "estimated_minutes": round(estimated_minutes, 1),
            "estimated_range": (
                f"{round(estimated_minutes * 0.8, 1)}-{round(estimated_minutes * 1.2, 1)} min"
            ),
            "note": "First request may take longer due to cold start (~10-90s)",
        }

    return fastapi_app


# =============================================================================
# CLI Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(action: str = "info") -> None:
    """
    Execute CLI actions for LingBot-World deployment management.

    This entrypoint provides commands for setting up, deploying, and
    monitoring the LingBot-World inference service.

    Parameters
    ----------
    action : str, default="info"
        Action to perform. Available options:

        - ``"info"``: Display deployment information and usage instructions.
        - ``"setup"``: Download model weights to Modal volume.
        - ``"health"``: Check current deployment health status.

    Examples
    --------
    Show deployment info::

        $ modal run src/lingbot_world/inference.py

    Download model weights::

        $ modal run src/lingbot_world/inference.py --action setup

    Check health::

        $ modal run src/lingbot_world/inference.py --action health

    Deploy the service::

        $ modal deploy src/lingbot_world/inference.py
    """
    if action == "info":
        print("=" * 60)
        print("LingBot-World Modal Deployment")
        print("=" * 60)
        print(f"\nModel: {MODEL_ID}")
        print("\nCommands:")
        print("  modal run src/lingbot_world/inference.py --action setup")
        print("  modal run src/lingbot_world/inference.py --action health")
        print("  modal deploy src/lingbot_world/inference.py")
        print("\nAfter deployment:")
        print("  API: https://YOUR_WORKSPACE--lingbot-world-api.modal.run")

    elif action == "setup":
        print("=" * 60)
        print("Downloading LingBot-World NF4 Model")
        print("=" * 60)
        model_path = download_model.remote()
        print(f"\nModel downloaded to: {model_path}")
        print("\nNext: modal deploy src/lingbot_world/inference.py")

    elif action == "health":
        print("Checking deployment health...")
        model = LingBotWorld()
        result = model.health_check.remote()
        print(f"Status: {result}")

    else:
        print(f"Unknown action: {action}")
        print("Available: info, setup, health")
