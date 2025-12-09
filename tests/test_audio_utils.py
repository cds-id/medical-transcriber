"""Tests for audio utility functions."""

import pytest
import numpy as np
from src.medical_stt.audio_utils import (
    chunk_audio,
    normalize_audio,
    detect_silence,
    generate_test_audio,
    generate_silence,
)


class TestChunkAudio:
    """Tests for chunk_audio function."""

    def test_basic_chunking(self):
        """Test basic audio chunking."""
        audio = np.arange(1000, dtype=np.float32)
        chunks = list(chunk_audio(audio, chunk_size=100))

        assert len(chunks) == 10
        for chunk in chunks:
            assert len(chunk) == 100

    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        audio = np.arange(1000, dtype=np.float32)
        chunks = list(chunk_audio(audio, chunk_size=100, overlap=50))

        # With 50% overlap (step=50), we get 20 chunks: ceil(1000/50) = 20
        assert len(chunks) == 20

    def test_partial_last_chunk(self):
        """Test handling of partial last chunk."""
        audio = np.arange(250, dtype=np.float32)
        chunks = list(chunk_audio(audio, chunk_size=100))

        assert len(chunks) == 3
        assert len(chunks[-1]) == 50

    def test_empty_audio(self):
        """Test empty audio input."""
        audio = np.array([], dtype=np.float32)
        chunks = list(chunk_audio(audio, chunk_size=100))
        assert len(chunks) == 0

    def test_chunk_smaller_than_audio(self):
        """Test when chunk size is larger than audio."""
        audio = np.arange(50, dtype=np.float32)
        chunks = list(chunk_audio(audio, chunk_size=100))

        assert len(chunks) == 1
        assert len(chunks[0]) == 50


class TestNormalizeAudio:
    """Tests for normalize_audio function."""

    def test_normalization(self):
        """Test basic normalization."""
        audio = np.array([0.5, 1.0, -0.5, 2.0], dtype=np.float32)
        normalized = normalize_audio(audio)

        assert np.abs(normalized).max() == 1.0
        np.testing.assert_array_almost_equal(
            normalized,
            np.array([0.25, 0.5, -0.25, 1.0]),
        )

    def test_already_normalized(self):
        """Test audio already in [-1, 1] range."""
        audio = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        normalized = normalize_audio(audio)

        assert np.abs(normalized).max() == 1.0

    def test_zero_audio(self):
        """Test zero/silence audio."""
        audio = np.zeros(100, dtype=np.float32)
        normalized = normalize_audio(audio)

        np.testing.assert_array_equal(normalized, audio)

    def test_negative_peak(self):
        """Test when negative peak is largest."""
        audio = np.array([0.5, -2.0, 0.3], dtype=np.float32)
        normalized = normalize_audio(audio)

        assert normalized.min() == -1.0
        assert np.abs(normalized).max() == 1.0


class TestDetectSilence:
    """Tests for detect_silence function."""

    def test_silence_detected(self):
        """Test silence detection."""
        silence = np.zeros(16000, dtype=np.float32)
        assert detect_silence(silence, threshold=0.01) == True

    def test_audio_not_silence(self):
        """Test non-silent audio."""
        audio = np.sin(np.linspace(0, 10 * np.pi, 16000)).astype(np.float32)
        assert detect_silence(audio, threshold=0.01) == False

    def test_low_level_audio(self):
        """Test very low level audio."""
        audio = np.random.randn(16000).astype(np.float32) * 0.001
        assert detect_silence(audio, threshold=0.01) == True

    def test_short_audio(self):
        """Test audio shorter than min_silence_duration."""
        short_silence = np.zeros(500, dtype=np.float32)
        assert detect_silence(short_silence, min_silence_duration=1000) == False

    def test_custom_threshold(self):
        """Test with custom threshold."""
        audio = np.random.randn(16000).astype(np.float32) * 0.05
        assert detect_silence(audio, threshold=0.01) == False
        assert detect_silence(audio, threshold=0.1) == True


class TestGenerateTestAudio:
    """Tests for generate_test_audio function."""

    def test_default_generation(self):
        """Test default audio generation."""
        audio = generate_test_audio()

        assert len(audio) == 16000  # 1 second at 16kHz
        assert audio.dtype == np.float32
        assert np.abs(audio).max() <= 1.0

    def test_custom_duration(self):
        """Test custom duration."""
        audio = generate_test_audio(duration_seconds=2.5)
        assert len(audio) == 40000  # 2.5 seconds at 16kHz

    def test_custom_sample_rate(self):
        """Test custom sample rate."""
        audio = generate_test_audio(
            duration_seconds=1.0,
            sample_rate=44100,
        )
        assert len(audio) == 44100

    def test_frequency(self):
        """Test generated frequency is approximately correct."""
        sample_rate = 16000
        frequency = 440.0
        audio = generate_test_audio(
            duration_seconds=1.0,
            sample_rate=sample_rate,
            frequency=frequency,
        )

        # FFT to verify frequency
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1 / sample_rate)
        peak_freq = np.abs(freqs[np.argmax(np.abs(fft[:len(fft) // 2]))])

        assert abs(peak_freq - frequency) < 10  # Within 10 Hz


class TestGenerateSilence:
    """Tests for generate_silence function."""

    def test_default_silence(self):
        """Test default silence generation."""
        silence = generate_silence()

        assert len(silence) == 16000
        assert silence.dtype == np.float32
        assert silence.sum() == 0

    def test_custom_duration(self):
        """Test custom duration silence."""
        silence = generate_silence(duration_seconds=0.5)
        assert len(silence) == 8000

    def test_custom_sample_rate(self):
        """Test custom sample rate silence."""
        silence = generate_silence(
            duration_seconds=1.0,
            sample_rate=44100,
        )
        assert len(silence) == 44100
