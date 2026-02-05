"""Pydantic data models for API requests and responses."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


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


class GenerationRequest(BaseModel):
    """Video generation request parameters."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt")
    size: VideoSize = Field(default=VideoSize.PORTRAIT_480, description="Output resolution")
    frame_num: int = Field(default=81, ge=17, le=961, description="Number of frames")
    sampling_steps: int = Field(default=40, ge=10, le=100, description="Sampling steps")
    guide_scale: float = Field(default=5.0, ge=1.0, le=15.0, description="Guidance scale")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    camera_preset: str | None = Field(default=None, description="Camera motion preset")

    @field_validator("frame_num")
    @classmethod
    def validate_frame_num(cls, v: int) -> int:
        """Validate frame number follows 4n+1 rule."""
        if (v - 1) % 4 != 0:
            msg = f"frame_num must be 4n+1 (e.g., 81, 161, 241). Got {v}"
            raise ValueError(msg)
        return v


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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model weights exist")
    pipeline_ready: bool = Field(..., description="Whether pipeline is initialized")
    gpu: str | None = Field(None, description="GPU name")
    gpu_memory_gb: float | None = Field(None, description="Total GPU memory")
    gpu_memory_used_gb: float | None = Field(None, description="Used GPU memory")


class ErrorResponse(BaseModel):
    """Structured error response."""

    success: bool = False
    error: str
    error_code: str
    details: dict[str, Any] | None = None


class TimeEstimate(BaseModel):
    """Generation time estimate."""

    resolution: str
    frames: int
    sampling_steps: int
    estimated_minutes: float
    estimated_range: str
    note: str
