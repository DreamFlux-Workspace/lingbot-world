"""
CLI entrypoint for LingBot-World.

Provides command-line interface for deploying, managing, and interacting
with the LingBot-World image-to-video generation service.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_modal_command(args: list[str]) -> int:
    """
    Run a Modal CLI command.

    Parameters
    ----------
    args : list[str]
        Arguments to pass to the modal command.

    Returns
    -------
    int
        Return code from the command.
    """
    cmd = ["uv", "run", "modal"] + args
    return subprocess.call(cmd)


def cmd_setup(_args: argparse.Namespace) -> int:
    """
    Download model weights to Modal volume.

    Parameters
    ----------
    _args : argparse.Namespace
        Parsed command arguments (unused).

    Returns
    -------
    int

        Exit code.
    """
    print("=" * 60)
    print("LingBot-World Setup")
    print("=" * 60)
    print("\nDownloading model weights (this may take 30+ minutes)...")
    print("Model: cahlen/lingbot-world-base-cam-nf4 (~30GB)")
    print()

    inference_path = Path(__file__).parent / "inference.py"
    return run_modal_command(
        [
            "run",
            str(inference_path),
            "--action",
            "setup",
        ]
    )


def cmd_deploy(_args: argparse.Namespace) -> int:
    """
    Deploy the inference service to Modal.

    Parameters
    ----------
    _args : argparse.Namespace
        Parsed command arguments (unused).

    Returns
    -------
    int
        Exit code.
    """
    print("=" * 60)
    print("Deploying LingBot-World to Modal")
    print("=" * 60)
    print()

    inference_path = Path(__file__).parent / "inference.py"
    return run_modal_command(["deploy", str(inference_path)])


def cmd_serve(_args: argparse.Namespace) -> int:
    """
    Run the service locally with Modal.

    Parameters
    ----------
    _args : argparse.Namespace
        Parsed command arguments (unused).

    Returns
    -------
    int
        Exit code.
    """
    print("=" * 60)
    print("Running LingBot-World locally")
    print("=" * 60)
    print()

    inference_path = Path(__file__).parent / "inference.py"
    return run_modal_command(["serve", str(inference_path)])


def cmd_ui(args: argparse.Namespace) -> int:
    """
    Launch the Gradio UI.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments with api_url.

    Returns
    -------
    int
        Exit code.
    """
    import os

    if args.api_url:
        os.environ["LINGBOT_API_URL"] = args.api_url

    print("=" * 60)
    print("Launching LingBot-World UI")
    print("=" * 60)
    print()

    from lingbot_world.ui import main as ui_main

    ui_main()
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """
    Check API health status.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments with api_url.

    Returns
    -------
    int
        Exit code.
    """
    import httpx

    api_url = args.api_url or "https://your-workspace--lingbot-world-api.modal.run"

    print(f"Checking health: {api_url}/health")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{api_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("\n✅ API is healthy!")
                print(f"   Status: {data.get('status')}")
                print(f"   GPU: {data.get('gpu')}")
                print(
                    f"   Memory: {data.get('gpu_memory_used_gb', 0):.1f}/{data.get('gpu_memory_gb', 0):.1f} GB"
                )
                return 0
            else:
                print(f"\n⚠️ API returned status {response.status_code}")
                print(response.text)
                return 1
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    """
    Generate a video from an image.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    import httpx
    from PIL import Image

    api_url = args.api_url or "https://your-workspace--lingbot-world-api.modal.run"

    print(f"Generating video from: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"API: {api_url}")
    print()

    # Load and prepare image
    img = Image.open(args.image).convert("RGB")
    import io

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="JPEG", quality=95)
    img_bytes = img_buffer.getvalue()

    form_data = {
        "prompt": args.prompt,
        "size": args.size,
        "frame_num": str(args.frames),
        "sampling_steps": str(args.steps),
        "guide_scale": str(args.guidance),
        "seed": str(args.seed),
    }

    files = {"image": ("input.jpg", img_bytes, "image/jpeg")}

    try:
        print("Sending request (this may take several minutes)...")
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{api_url}/generate",
                data=form_data,
                files=files,
            )

            if response.status_code != 200:
                print(f"❌ Error ({response.status_code}): {response.text}")
                return 1

            output_path = Path(args.output)
            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"\n✅ Video saved to: {output_path}")
            print(f"   Size: {len(response.content) / 1e6:.2f} MB")
            return 0

    except httpx.TimeoutException:
        print("❌ Request timed out")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main() -> int:
    """
    Main CLI entry point.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="lingbot",
        description="LingBot-World: Image-to-Video Generation with Camera Control",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    subparsers.add_parser(
        "setup",
        help="Download model weights to Modal volume",
    )

    # Deploy command
    subparsers.add_parser(
        "deploy",
        help="Deploy the inference service to Modal",
    )

    # Serve command
    subparsers.add_parser(
        "serve",
        help="Run the service locally with Modal",
    )

    # UI command
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Gradio web interface",
    )
    ui_parser.add_argument(
        "--api-url",
        type=str,
        help="API URL for the Modal deployment",
    )

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Check API health status",
    )
    health_parser.add_argument(
        "--api-url",
        type=str,
        help="API URL to check",
    )

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a video from an image",
    )
    gen_parser.add_argument(
        "image",
        type=str,
        help="Path to input image",
    )
    gen_parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt for generation",
    )
    gen_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path",
    )
    gen_parser.add_argument(
        "--api-url",
        type=str,
        help="API URL for the Modal deployment",
    )
    gen_parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        choices=["480*832", "832*480", "720*1280", "1280*720"],
        help="Output resolution",
    )
    gen_parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of frames (4n+1)",
    )
    gen_parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Sampling steps",
    )
    gen_parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Guidance scale",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "setup": cmd_setup,
        "deploy": cmd_deploy,
        "serve": cmd_serve,
        "ui": cmd_ui,
        "health": cmd_health,
        "generate": cmd_generate,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
