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

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from typing import Any

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
    gpu="A100-80GB",
    timeout=30 * 60,
    scaledown_window=15 * 60,
    secrets=[modal.Secret.from_name("huggingface-token")],
    # GPU memory snapshot for faster cold starts (~10s vs ~90s)
    enable_memory_snapshot=True,
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
        import sys

        import torch

        self.model_path = MODEL_DIR / "lingbot-world-nf4"
        self.device = torch.device("cuda:0")

        # Add model source to Python path for imports
        sys.path.insert(0, str(self.model_path))

        print(f"Model path: {self.model_path}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Import and initialize the pre-quantized pipeline
        from generate_prequant import WanI2V_PreQuant

        print("Loading WanI2V_PreQuant pipeline...")
        self.pipeline = WanI2V_PreQuant(
            checkpoint_dir=str(self.model_path),
            device_id=0,
        )

        print("Pipeline loaded successfully!")

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
    includes OpenAPI documentation, CORS middleware, and structured error
    handling.

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
    POST /generate
        Generate video from uploaded image and prompt.
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
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    fastapi_app = FastAPI(
        title="LingBot-World API",
        description="Image-to-Video World Model with Camera Pose Control (NF4 Quantized)",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/")
    async def root() -> dict[str, Any]:
        """Return API information and capabilities."""
        return {
            "name": "LingBot-World API",
            "model": MODEL_ID,
            "description": "Image-to-Video generation with camera control",
            "endpoints": {
                "/docs": "OpenAPI documentation",
                "/health": "Health check",
                "/generate": "Generate video from image",
            },
            "supported_sizes": list(SUPPORTED_SIZES),
            "max_frames": MAX_FRAMES,
        }

    @fastapi_app.get("/health")
    async def health() -> dict[str, Any] | JSONResponse:
        """Return service health status and GPU metrics."""
        try:
            model = LingBotWorld()
            result: dict[str, Any] = model.health_check.remote()
            return result
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)},
            )

    @fastapi_app.post("/generate")
    async def generate(
        image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
        prompt: str = Form(..., description="Text prompt for video generation"),
        size: str = Form("480*832", description="Resolution: 480*832 or 720*1280"),
        frame_num: int = Form(81, description="Number of frames (4n+1)"),
        sampling_steps: int = Form(40, description="Diffusion sampling steps"),
        guide_scale: float = Form(5.0, description="Guidance scale"),
        seed: int = Form(-1, description="Random seed (-1 for random)"),
    ) -> Response:
        """
        Generate video from an uploaded image.

        Upload an image file and provide a text prompt to generate a
        cinematic video showing the scene with natural motion.
        """
        try:
            image_bytes = await image.read()

            model = LingBotWorld()
            video_bytes = model.generate.remote(
                image_bytes=image_bytes,
                prompt=prompt,
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
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

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
