"""Pytest configuration and fixtures for medical STT tests."""

import pytest
import numpy as np

from src.medical_stt.config import Config, ModelType, AudioConfig, StreamingConfig
from src.medical_stt.transcriber import (
    MedicalTranscriber,
    MockBackend,
    TranscriptionResult,
    AudioProcessor,
)


@pytest.fixture
def sample_rate():
    """Default sample rate."""
    return 16000


@pytest.fixture
def audio_config():
    """Default audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=1000,
    )


@pytest.fixture
def streaming_config():
    """Default streaming configuration."""
    return StreamingConfig(
        buffer_size_seconds=5.0,
        overlap_seconds=0.5,
        min_audio_length_seconds=0.5,
    )


@pytest.fixture
def default_config(audio_config, streaming_config):
    """Default configuration."""
    return Config(
        model_type=ModelType.WHISPER_SMALL_MEDICAL,
        audio=audio_config,
        streaming=streaming_config,
        device="cpu",
        language="en",
    )


@pytest.fixture
def mock_backend(default_config):
    """Mock backend for testing."""
    backend = MockBackend(default_config)
    backend.load()
    return backend


@pytest.fixture
def mock_transcriber(default_config):
    """Transcriber with mock backend."""
    transcriber = MedicalTranscriber(config=default_config, use_mock=True)
    transcriber.load_model()
    return transcriber


@pytest.fixture
def test_audio(sample_rate):
    """Generate test audio (1 second sine wave)."""
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


@pytest.fixture
def test_audio_chunks(sample_rate):
    """Generate multiple test audio chunks."""
    chunks = []
    chunk_size = sample_rate // 2  # 0.5 second chunks
    for i in range(5):
        frequency = 440.0 + i * 100
        t = np.linspace(0, 0.5, chunk_size)
        chunk = 0.5 * np.sin(2 * np.pi * frequency * t)
        chunks.append(chunk.astype(np.float32))
    return chunks


@pytest.fixture
def silence_audio(sample_rate):
    """Generate silence audio."""
    return np.zeros(sample_rate, dtype=np.float32)


@pytest.fixture
def audio_processor(default_config):
    """Audio processor instance."""
    return AudioProcessor(default_config)
