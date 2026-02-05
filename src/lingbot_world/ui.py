"""
Gradio UI for LingBot-World image-to-video generation.

This module provides a web interface for interacting with the LingBot-World
model, allowing users to upload images and generate cinematic videos with
optional camera trajectory control.
"""

import os

import gradio as gr
import httpx
import numpy as np
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

# API endpoint - set via environment variable or use default
API_URL = os.environ.get(
    "LINGBOT_API_URL",
    "https://your-workspace--lingbot-world-api.modal.run",
)

# Default generation parameters
DEFAULT_SIZE = "480*832"
DEFAULT_FRAME_NUM = 81
DEFAULT_SAMPLING_STEPS = 40
DEFAULT_GUIDE_SCALE = 5.0
DEFAULT_SEED = -1

# Supported resolutions
SUPPORTED_SIZES = ["480*832", "832*480", "720*1280", "1280*720"]

# Frame number presets (must be 4n+1)
FRAME_PRESETS = {
    "Short (~5 sec)": 81,
    "Medium (~10 sec)": 161,
    "Long (~15 sec)": 241,
    "Very Long (~30 sec)": 481,
}


# =============================================================================
# Camera Trajectory Presets
# =============================================================================


def generate_camera_trajectory(
    trajectory_type: str,
    num_frames: int,
    intensity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate camera trajectory poses and intrinsics.

    Parameters
    ----------
    trajectory_type : str
        Type of camera motion: "static", "pan_left", "pan_right",
        "zoom_in", "zoom_out", "orbit", "dolly_forward".
    num_frames : int
        Number of frames in the trajectory.
    intensity : float, default=1.0
        Motion intensity multiplier.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (intrinsics, poses) arrays.
        intrinsics: shape [num_frames, 4] with [fx, fy, cx, cy]
        poses: shape [num_frames, 4, 4] transformation matrices
    """
    # Default intrinsics (normalized)
    fx, fy = 1.0, 1.0
    cx, cy = 0.5, 0.5

    intrinsics = np.tile([fx, fy, cx, cy], (num_frames, 1)).astype(np.float32)
    poses = np.zeros((num_frames, 4, 4), dtype=np.float32)

    # Initialize with identity matrices
    for i in range(num_frames):
        poses[i] = np.eye(4)

    t = np.linspace(0, 1, num_frames)

    if trajectory_type == "static":
        pass  # Keep identity matrices

    elif trajectory_type == "pan_left":
        for i, ti in enumerate(t):
            angle = -ti * np.pi / 6 * intensity
            poses[i, 0, 0] = np.cos(angle)
            poses[i, 0, 2] = np.sin(angle)
            poses[i, 2, 0] = -np.sin(angle)
            poses[i, 2, 2] = np.cos(angle)

    elif trajectory_type == "pan_right":
        for i, ti in enumerate(t):
            angle = ti * np.pi / 6 * intensity
            poses[i, 0, 0] = np.cos(angle)
            poses[i, 0, 2] = np.sin(angle)
            poses[i, 2, 0] = -np.sin(angle)
            poses[i, 2, 2] = np.cos(angle)

    elif trajectory_type == "zoom_in":
        for i, ti in enumerate(t):
            poses[i, 2, 3] = -ti * 0.5 * intensity

    elif trajectory_type == "zoom_out":
        for i, ti in enumerate(t):
            poses[i, 2, 3] = ti * 0.5 * intensity

    elif trajectory_type == "orbit":
        for i, ti in enumerate(t):
            angle = ti * np.pi / 4 * intensity
            poses[i, 0, 0] = np.cos(angle)
            poses[i, 0, 2] = np.sin(angle)
            poses[i, 2, 0] = -np.sin(angle)
            poses[i, 2, 2] = np.cos(angle)
            # Add slight translation for orbit effect
            poses[i, 0, 3] = np.sin(angle) * 0.1
            poses[i, 2, 3] = (1 - np.cos(angle)) * 0.1

    elif trajectory_type == "dolly_forward":
        for i, ti in enumerate(t):
            poses[i, 2, 3] = -ti * 0.3 * intensity

    return intrinsics, poses


# =============================================================================
# API Client
# =============================================================================


def generate_video(
    image: Image.Image,
    prompt: str,
    size: str,
    frame_num: int,
    sampling_steps: int,
    guide_scale: float,
    seed: int,
    camera_trajectory: str,
    camera_intensity: float,
    progress: gr.Progress = gr.Progress(),
) -> str | None:
    """
    Generate video by calling the Modal API.

    Parameters
    ----------
    image : Image.Image
        Input image for video generation.
    prompt : str
        Text prompt describing the desired video.
    size : str
        Output resolution.
    frame_num : int
        Number of frames to generate.
    sampling_steps : int
        Number of diffusion steps.
    guide_scale : float
        Classifier-free guidance scale.
    seed : int
        Random seed.
    camera_trajectory : str
        Type of camera motion.
    camera_intensity : float
        Intensity of camera motion.
    progress : gr.Progress
        Gradio progress tracker.

    Returns
    -------
    str or None
        Path to the generated video file, or None if generation failed.
    """
    import io
    import tempfile

    if image is None:
        raise gr.Error("Please upload an image first!")

    if not prompt.strip():
        raise gr.Error("Please enter a prompt!")

    progress(0.1, desc="Preparing request...")

    # Convert image to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG", quality=95)
    img_bytes = img_buffer.getvalue()

    # Prepare form data
    form_data = {
        "prompt": prompt,
        "size": size,
        "frame_num": str(frame_num),
        "sampling_steps": str(sampling_steps),
        "guide_scale": str(guide_scale),
        "seed": str(seed),
    }

    files = {"image": ("input.jpg", img_bytes, "image/jpeg")}

    # Add camera trajectory if not static
    if camera_trajectory != "static":
        intrinsics, poses = generate_camera_trajectory(
            camera_trajectory, frame_num, camera_intensity
        )
        # Note: Camera data handling would need backend support
        # For now, include in the description
        form_data["prompt"] = f"{prompt} [Camera: {camera_trajectory}]"

    progress(0.2, desc="Sending to API...")

    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{API_URL}/generate",
                data=form_data,
                files=files,
            )

            if response.status_code != 200:
                error_detail = response.text
                raise gr.Error(f"API error ({response.status_code}): {error_detail}")

            progress(0.9, desc="Saving video...")

            # Save video to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                video_path = f.name

            progress(1.0, desc="Done!")
            return video_path

    except httpx.TimeoutException as e:
        raise gr.Error("Request timed out. Video generation can take several minutes.") from e
    except httpx.RequestError as e:
        raise gr.Error(f"Connection error: {e}") from e


def check_api_health() -> str:
    """
    Check API health status.

    Returns
    -------
    str
        Health status message.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                return f"✅ API Online | GPU: {data.get('gpu', 'Unknown')} | Memory: {data.get('gpu_memory_used_gb', 0):.1f}/{data.get('gpu_memory_gb', 0):.1f} GB"
            else:
                return f"⚠️ API returned status {response.status_code}"
    except Exception as e:
        return f"❌ API Offline: {e}"


# =============================================================================
# Gradio Interface
# =============================================================================


def create_ui() -> gr.Blocks:
    """
    Create the Gradio UI interface.

    Returns
    -------
    gr.Blocks
        The Gradio Blocks application.
    """
    with gr.Blocks(
        title="LingBot-World",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .status-bar { font-size: 0.9em; padding: 10px; border-radius: 5px; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🎮 LingBot-World
            ### Image-to-Video World Model with Camera Control

            Transform any image into a cinematic video with AI-powered motion generation.
            Upload an image, describe the motion you want, and let the model bring it to life!
            """,
            elem_classes=["main-header"],
        )

        # API Status
        with gr.Row():
            status_text = gr.Textbox(
                label="API Status",
                value="Checking...",
                interactive=False,
                elem_classes=["status-bar"],
            )
            refresh_btn = gr.Button("🔄 Refresh", size="sm")

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📷 Input Image",
                    type="pil",
                    height=400,
                )

                prompt = gr.Textbox(
                    label="📝 Prompt",
                    placeholder="Describe the motion and style you want...\nExample: A cinematic video with gentle camera movement through the scene",
                    lines=3,
                )

                with gr.Accordion("⚙️ Generation Settings", open=False):
                    size = gr.Dropdown(
                        label="Resolution",
                        choices=SUPPORTED_SIZES,
                        value=DEFAULT_SIZE,
                    )

                    frame_preset = gr.Dropdown(
                        label="Duration",
                        choices=list(FRAME_PRESETS.keys()),
                        value="Short (~5 sec)",
                    )

                    frame_num = gr.Slider(
                        label="Frame Count",
                        minimum=17,
                        maximum=481,
                        step=4,
                        value=DEFAULT_FRAME_NUM,
                        info="Must be 4n+1 (17, 21, 25, ...)",
                    )

                    sampling_steps = gr.Slider(
                        label="Sampling Steps",
                        minimum=20,
                        maximum=100,
                        step=5,
                        value=DEFAULT_SAMPLING_STEPS,
                        info="More steps = better quality but slower",
                    )

                    guide_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        step=0.5,
                        value=DEFAULT_GUIDE_SCALE,
                        info="Higher = more prompt adherence",
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=DEFAULT_SEED,
                        precision=0,
                        info="-1 for random",
                    )

                with gr.Accordion("🎬 Camera Control", open=False):
                    camera_trajectory = gr.Dropdown(
                        label="Camera Motion",
                        choices=[
                            "static",
                            "pan_left",
                            "pan_right",
                            "zoom_in",
                            "zoom_out",
                            "orbit",
                            "dolly_forward",
                        ],
                        value="static",
                    )

                    camera_intensity = gr.Slider(
                        label="Motion Intensity",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                    )

                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg",
                )

            # Right column - Output
            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="🎥 Generated Video",
                    height=400,
                )

                gr.Markdown(
                    """
                    ### Tips
                    - Use descriptive prompts for better results
                    - Start with lower resolution for faster testing
                    - Camera controls add cinematic motion to the scene
                    - Generation typically takes 2-5 minutes
                    """
                )

        # Examples
        gr.Examples(
            examples=[
                [
                    "examples/cityscape.jpg",
                    "A cinematic first-person exploration through the urban environment, with gentle forward movement and ambient city sounds",
                ],
                [
                    "examples/nature.jpg",
                    "The camera slowly pans across the serene landscape, capturing the gentle sway of trees and flowing water",
                ],
                [
                    "examples/interior.jpg",
                    "A smooth tracking shot through the room, revealing architectural details and warm lighting",
                ],
            ],
            inputs=[input_image, prompt],
            label="Example Prompts",
        )

        # Event handlers
        def update_frame_num(preset):
            return FRAME_PRESETS.get(preset, DEFAULT_FRAME_NUM)

        frame_preset.change(
            fn=update_frame_num,
            inputs=[frame_preset],
            outputs=[frame_num],
        )

        refresh_btn.click(
            fn=check_api_health,
            outputs=[status_text],
        )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                input_image,
                prompt,
                size,
                frame_num,
                sampling_steps,
                guide_scale,
                seed,
                camera_trajectory,
                camera_intensity,
            ],
            outputs=[output_video],
        )

        # Check health on load
        demo.load(
            fn=check_api_health,
            outputs=[status_text],
        )

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Launch the Gradio UI."""
    demo = create_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,
    )


if __name__ == "__main__":
    main()
