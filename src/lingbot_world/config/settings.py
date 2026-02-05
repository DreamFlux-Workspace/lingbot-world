"""Application settings using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes
    ----------
    model_id : str
        HuggingFace model repository identifier.
    model_dir : str
        Path where model weights are stored in the container.
    outputs_dir : str
        Path where generated videos are stored.
    gpu_type : str
        GPU type for Modal deployment (H100, A100-80GB, A100-40GB).
    timeout_seconds : int
        Maximum execution time for video generation.
    scaledown_window : int
        Time in seconds to keep containers warm between requests.
    max_concurrent_inputs : int
        Maximum concurrent requests per GPU container.
    default_sampling_steps : int
        Default number of diffusion sampling steps.
    default_guide_scale : float
        Default classifier-free guidance scale.
    default_frame_num : int
        Default number of frames to generate.
    default_size : str
        Default output resolution.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    hf_secret_name : str
        Name of the Modal secret containing HuggingFace token.
    """

    model_config = SettingsConfigDict(
        env_prefix="LINGBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model Configuration
    model_id: str = Field(
        default="cahlen/lingbot-world-base-cam-nf4",
        description="HuggingFace model repository identifier",
    )
    model_dir: str = Field(
        default="/models",
        description="Path where model weights are stored",
    )
    outputs_dir: str = Field(
        default="/outputs",
        description="Path where generated videos are stored",
    )

    # GPU Configuration
    gpu_type: Literal["H100", "A100-80GB", "A100-40GB", "L40S"] = Field(
        default="H100",
        description="GPU type for Modal deployment",
    )
    timeout_seconds: int = Field(
        default=3600,  # 60 minutes
        ge=300,
        le=7200,
        description="Maximum execution time in seconds",
    )
    scaledown_window: int = Field(
        default=900,  # 15 minutes
        ge=60,
        le=3600,
        description="Container keepalive time in seconds",
    )
    max_concurrent_inputs: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum concurrent requests per container",
    )

    # Generation Defaults
    default_sampling_steps: int = Field(
        default=40,
        ge=10,
        le=100,
        description="Default diffusion sampling steps",
    )
    default_guide_scale: float = Field(
        default=5.0,
        ge=1.0,
        le=15.0,
        description="Default guidance scale",
    )
    default_frame_num: int = Field(
        default=81,
        ge=17,
        le=961,
        description="Default number of frames",
    )
    default_size: str = Field(
        default="480*832",
        description="Default output resolution",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Secrets
    hf_secret_name: str = Field(
        default="huggingface-token",
        description="Modal secret name for HuggingFace token",
    )

    # Supported configurations
    supported_sizes: tuple[str, ...] = Field(
        default=("480*832", "832*480", "720*1280", "1280*720"),
        description="Supported output resolutions",
    )
    max_frames: int = Field(
        default=961,
        description="Maximum number of frames",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns
    -------
    Settings
        Application configuration instance.
    """
    return Settings()
