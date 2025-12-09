"""Tests for configuration module."""

import pytest
from src.medical_stt.config import Config, ModelType, AudioConfig, StreamingConfig


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_values(self):
        """Test model type values."""
        assert ModelType.WHISPER_LARGE_V3_GERMAN_MEDICAL.value == "primeLine/whisper-large-v3-german-medical"
        assert ModelType.WHISPER_SMALL_MEDICAL.value == "ayoubkirouane/whisper-small-medical"
        assert ModelType.WHISPER_LARGE_V2_MEDICAL.value == "MohamedRashad/whisper-large-v2-medical"

    def test_get_default(self):
        """Test default model."""
        default = ModelType.get_default()
        assert default == ModelType.WHISPER_SMALL_MEDICAL

    def test_from_string_by_value(self):
        """Test creating from string value."""
        model = ModelType.from_string("ayoubkirouane/whisper-small-medical")
        assert model == ModelType.WHISPER_SMALL_MEDICAL

    def test_from_string_by_name(self):
        """Test creating from enum name."""
        model = ModelType.from_string("WHISPER_SMALL_MEDICAL")
        assert model == ModelType.WHISPER_SMALL_MEDICAL

    def test_from_string_invalid(self):
        """Test error on invalid model string."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelType.from_string("invalid-model")


class TestAudioConfig:
    """Tests for AudioConfig."""

    def test_defaults(self):
        """Test default values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_duration_ms == 1000
        assert config.dtype == "float32"

    def test_chunk_size(self):
        """Test chunk size calculation."""
        config = AudioConfig(sample_rate=16000, chunk_duration_ms=1000)
        assert config.chunk_size == 16000

        config = AudioConfig(sample_rate=16000, chunk_duration_ms=500)
        assert config.chunk_size == 8000

    def test_custom_values(self):
        """Test custom configuration."""
        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            chunk_duration_ms=500,
        )
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_size == 22050


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = StreamingConfig()
        assert config.buffer_size_seconds == 5.0
        assert config.overlap_seconds == 0.5
        assert config.min_audio_length_seconds == 0.5
        assert config.max_silence_duration_seconds == 2.0
        assert config.vad_enabled is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = StreamingConfig(
            buffer_size_seconds=10.0,
            overlap_seconds=1.0,
            vad_enabled=False,
        )
        assert config.buffer_size_seconds == 10.0
        assert config.overlap_seconds == 1.0
        assert config.vad_enabled is False


class TestConfig:
    """Tests for main Config class."""

    def test_defaults(self):
        """Test default configuration."""
        config = Config()
        assert config.model_type == ModelType.WHISPER_SMALL_MEDICAL
        assert config.device == "auto"
        assert config.language is None
        assert config.task == "transcribe"

    def test_model_id_property(self):
        """Test model_id property."""
        config = Config(model_type=ModelType.WHISPER_LARGE_V2_MEDICAL)
        assert config.model_id == "MohamedRashad/whisper-large-v2-medical"

    def test_post_init_string_conversion(self):
        """Test string to enum conversion in post_init."""
        config = Config(model_type="WHISPER_SMALL_MEDICAL")
        assert config.model_type == ModelType.WHISPER_SMALL_MEDICAL

    def test_post_init_dict_conversion(self):
        """Test dict to dataclass conversion."""
        config = Config(
            audio={"sample_rate": 22050, "channels": 2},
            streaming={"buffer_size_seconds": 10.0},
        )
        assert config.audio.sample_rate == 22050
        assert config.audio.channels == 2
        assert config.streaming.buffer_size_seconds == 10.0

    def test_for_german_medical(self):
        """Test German medical preset."""
        config = Config.for_german_medical()
        assert config.model_type == ModelType.WHISPER_LARGE_V3_GERMAN_MEDICAL
        assert config.language == "de"

    def test_for_english_medical(self):
        """Test English medical preset."""
        config = Config.for_english_medical()
        assert config.model_type == ModelType.WHISPER_LARGE_V2_MEDICAL
        assert config.language == "en"

    def test_for_lightweight(self):
        """Test lightweight preset."""
        config = Config.for_lightweight()
        assert config.model_type == ModelType.WHISPER_SMALL_MEDICAL
        assert config.streaming.buffer_size_seconds == 3.0

    def test_get_device_explicit(self):
        """Test explicit device setting."""
        config = Config(device="cpu")
        assert config.get_device() == "cpu"

        config = Config(device="cuda")
        assert config.get_device() == "cuda"
