#!/usr/bin/env python3
"""Test script for LingBot-World inference API."""

import sys
from pathlib import Path

import httpx

API_URL = "https://smolagentsworkspace--lingbot-world-api.modal.run"


def test_inference():
    """Run a sample inference request."""
    image_path = Path("lingbot-world-ref/examples/00/image.jpg")

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Testing inference with image: {image_path}")
    print(f"API URL: {API_URL}/generate")
    print()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Use httpx with long timeout and redirect following
    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        print("Sending request... (this may take 3-5 minutes)")
        response = client.post(
            f"{API_URL}/generate",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={
                "prompt": "A cinematic video with gentle natural motion",
                "size": "480*832",
                "frame_num": "81",
                "sampling_steps": "40",
            },
        )

        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code == 200:
            output_path = Path("sample_output.mp4")
            output_path.write_bytes(response.content)
            print(f"\n✅ Video saved to: {output_path}")
            print(f"   Size: {len(response.content) / 1e6:.2f} MB")
        else:
            print(f"\n❌ Error: {response.text}")


if __name__ == "__main__":
    test_inference()
