"""Configuration module for Medical Speech-to-Text."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelType(Enum):
    """Available medical Whisper models."""

    WHISPER_LARGE_V3_GERMAN_MEDICAL = "primeLine/whisper-large-v3-german-medical"
    WHISPER_SMALL_MEDICAL = "ayoubkirouane/whisper-small-medical"
    WHISPER_LARGE_V2_MEDICAL = "MohamedRashad/whisper-large-v2-medical"

    @classmethod
    def get_default(cls) -> "ModelType":
        """Get the default model."""
        return cls.WHISPER_SMALL_MEDICAL

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """Create ModelType from string value."""
        for model in cls:
            if model.value == value or model.name == value:
                return model
        raise ValueError(f"Unknown model: {value}")


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 1000  # 1 second chunks for streaming
    dtype: str = "float32"

    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in samples."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


@dataclass
class StreamingConfig:
    """Streaming transcription configuration."""

    buffer_size_seconds: float = 5.0  # Buffer for context
    overlap_seconds: float = 0.5  # Overlap between chunks
    min_audio_length_seconds: float = 0.5  # Minimum audio to process
    max_silence_duration_seconds: float = 2.0  # Silence detection threshold
    vad_enabled: bool = True  # Voice Activity Detection


@dataclass
class Config:
    """Main configuration for Medical STT."""

    model_type: ModelType = field(default_factory=ModelType.get_default)
    audio: AudioConfig = field(default_factory=AudioConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    language: Optional[str] = None  # None for auto-detection
    task: str = "transcribe"  # "transcribe" or "translate"
    hf_token: Optional[str] = None  # HuggingFace API token for gated models

    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.model_type, str):
            self.model_type = ModelType.from_string(self.model_type)
        if isinstance(self.audio, dict):
            self.audio = AudioConfig(**self.audio)
        if isinstance(self.streaming, dict):
            self.streaming = StreamingConfig(**self.streaming)

    @property
    def model_id(self) -> str:
        """Get the HuggingFace model ID."""
        return self.model_type.value

    def get_device(self) -> str:
        """Determine the device to use."""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def for_german_medical(cls) -> "Config":
        """Create config optimized for German medical transcription."""
        return cls(
            model_type=ModelType.WHISPER_LARGE_V3_GERMAN_MEDICAL,
            language="de",
        )

    @classmethod
    def for_english_medical(cls) -> "Config":
        """Create config optimized for English medical transcription."""
        return cls(
            model_type=ModelType.WHISPER_LARGE_V2_MEDICAL,
            language="en",
        )

    @classmethod
    def for_lightweight(cls) -> "Config":
        """Create config for lightweight/fast transcription."""
        return cls(
            model_type=ModelType.WHISPER_SMALL_MEDICAL,
            audio=AudioConfig(chunk_duration_ms=500),
            streaming=StreamingConfig(buffer_size_seconds=3.0),
        )
