"""Tests for transcriber module."""

import pytest
import numpy as np
from src.medical_stt.transcriber import (
    MedicalTranscriber,
    TranscriptionResult,
    AudioProcessor,
    MockBackend,
)
from src.medical_stt.config import Config, ModelType


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = TranscriptionResult(text="Test")
        assert result.text == "Test"
        assert result.is_final is False
        assert result.confidence == 0.0
        assert result.language is None
        assert result.medical_terms == []

    def test_str_representation(self):
        """Test string representation."""
        result = TranscriptionResult(text="Patient diagnosed")
        assert str(result) == "Patient diagnosed"

    def test_with_all_fields(self):
        """Test with all fields populated."""
        result = TranscriptionResult(
            text="Blood pressure 120/80",
            is_final=True,
            confidence=0.95,
            start_time=0.0,
            end_time=2.5,
            language="en",
            medical_terms=["blood pressure"],
        )
        assert result.is_final is True
        assert result.confidence == 0.95
        assert result.end_time == 2.5
        assert "blood pressure" in result.medical_terms


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    def test_add_chunk(self, audio_processor, sample_rate):
        """Test adding audio chunks."""
        chunk = np.random.randn(sample_rate).astype(np.float32)
        audio_processor.add_chunk(chunk)
        assert audio_processor.get_duration() == 1.0

    def test_add_multiple_chunks(self, audio_processor, sample_rate):
        """Test adding multiple chunks."""
        for _ in range(3):
            chunk = np.random.randn(sample_rate // 2).astype(np.float32)
            audio_processor.add_chunk(chunk)
        assert audio_processor.get_duration() == 1.5

    def test_get_buffer(self, audio_processor, sample_rate):
        """Test getting concatenated buffer."""
        chunk1 = np.ones(sample_rate // 2, dtype=np.float32)
        chunk2 = np.zeros(sample_rate // 2, dtype=np.float32)

        audio_processor.add_chunk(chunk1)
        audio_processor.add_chunk(chunk2)

        buffer = audio_processor.get_buffer()
        assert len(buffer) == sample_rate
        assert buffer[:sample_rate // 2].sum() == sample_rate // 2
        assert buffer[sample_rate // 2:].sum() == 0

    def test_clear(self, audio_processor, sample_rate):
        """Test clearing buffer."""
        chunk = np.random.randn(sample_rate).astype(np.float32)
        audio_processor.add_chunk(chunk)
        audio_processor.clear()

        assert audio_processor.get_duration() == 0
        assert len(audio_processor.get_buffer()) == 0

    def test_trim_buffer(self, audio_processor, sample_rate):
        """Test trimming buffer."""
        # Add 3 seconds of audio
        for _ in range(3):
            chunk = np.random.randn(sample_rate).astype(np.float32)
            audio_processor.add_chunk(chunk)

        assert audio_processor.get_duration() == 3.0

        # Trim to keep only 1 second
        audio_processor.trim_buffer(1.0)
        assert audio_processor.get_duration() == 1.0

    def test_stereo_to_mono_conversion(self, audio_processor, sample_rate):
        """Test stereo to mono conversion."""
        stereo_chunk = np.random.randn(sample_rate, 2).astype(np.float32)
        audio_processor.add_chunk(stereo_chunk)

        buffer = audio_processor.get_buffer()
        assert buffer.ndim == 1
        assert len(buffer) == sample_rate


class TestMockBackend:
    """Tests for MockBackend."""

    def test_load(self, default_config):
        """Test loading mock backend."""
        backend = MockBackend(default_config)
        assert not backend.is_loaded()

        backend.load()
        assert backend.is_loaded()

    def test_transcribe_not_loaded(self, default_config, test_audio, sample_rate):
        """Test transcribe raises when not loaded."""
        backend = MockBackend(default_config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.transcribe(test_audio, sample_rate)

    def test_transcribe(self, mock_backend, test_audio, sample_rate):
        """Test transcription."""
        result = mock_backend.transcribe(test_audio, sample_rate)

        assert isinstance(result, TranscriptionResult)
        assert result.text != ""
        assert result.is_final is True
        assert result.confidence > 0

    def test_transcribe_multiple_calls(self, mock_backend, test_audio, sample_rate):
        """Test multiple transcriptions cycle through responses."""
        results = []
        for _ in range(5):
            result = mock_backend.transcribe(test_audio, sample_rate)
            results.append(result.text)

        # Should have at least some different responses
        assert len(set(results)) > 1


class TestMedicalTranscriber:
    """Tests for MedicalTranscriber class."""

    def test_init_default(self):
        """Test default initialization."""
        transcriber = MedicalTranscriber()
        assert transcriber.config is not None
        assert not transcriber.is_ready()

    def test_init_with_config(self, default_config):
        """Test initialization with config."""
        transcriber = MedicalTranscriber(config=default_config)
        assert transcriber.config == default_config

    def test_init_with_mock(self, default_config):
        """Test initialization with mock backend."""
        transcriber = MedicalTranscriber(config=default_config, use_mock=True)
        transcriber.load_model()
        assert transcriber.is_ready()

    def test_load_model(self, mock_transcriber):
        """Test model loading."""
        assert mock_transcriber.is_ready()

    def test_transcribe(self, mock_transcriber, test_audio):
        """Test transcription."""
        result = mock_transcriber.transcribe(test_audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text != ""

    def test_transcribe_with_bytes(self, mock_transcriber, test_audio):
        """Test transcription with bytes input."""
        audio_bytes = test_audio.tobytes()
        result = mock_transcriber.transcribe(audio_bytes)

        assert isinstance(result, TranscriptionResult)
        assert result.text != ""

    def test_callback_management(self, mock_transcriber):
        """Test adding and removing callbacks."""
        results = []

        def callback(result):
            results.append(result)

        mock_transcriber.add_callback(callback)
        assert len(mock_transcriber._callbacks) == 1

        mock_transcriber.remove_callback(callback)
        assert len(mock_transcriber._callbacks) == 0

    def test_stream_transcribe(self, mock_transcriber, test_audio_chunks):
        """Test streaming transcription."""
        results = list(mock_transcriber.stream_transcribe(iter(test_audio_chunks)))

        assert len(results) > 0
        assert results[-1].is_final is True

        # All but last should be non-final
        for result in results[:-1]:
            assert result.is_final is False

    def test_stream_transcribe_with_callback(self, mock_transcriber, test_audio_chunks):
        """Test streaming with callback."""
        callback_results = []

        def callback(result):
            callback_results.append(result)

        mock_transcriber.add_callback(callback)
        list(mock_transcriber.stream_transcribe(iter(test_audio_chunks)))

        assert len(callback_results) > 0

    def test_stop_streaming(self, mock_transcriber, test_audio_chunks):
        """Test stopping streaming."""
        results = []

        def audio_gen():
            for i, chunk in enumerate(test_audio_chunks):
                if i == 2:
                    mock_transcriber.stop_streaming()
                yield chunk

        for result in mock_transcriber.stream_transcribe(audio_gen()):
            results.append(result)

        # Should have stopped early
        assert not mock_transcriber.is_streaming

    def test_is_streaming_property(self, mock_transcriber, test_audio_chunks):
        """Test is_streaming property."""
        assert not mock_transcriber.is_streaming

        # Start streaming in a controlled way
        gen = mock_transcriber.stream_transcribe(iter(test_audio_chunks[:1]))
        next(gen)  # Get first result
        # Note: is_streaming becomes False after iteration completes

        # After full iteration
        list(gen)
        assert not mock_transcriber.is_streaming


class TestMedicalTranscriberPresets:
    """Tests for transcriber preset configurations."""

    def test_german_medical_config(self):
        """Test German medical configuration."""
        config = Config.for_german_medical()
        transcriber = MedicalTranscriber(config=config, use_mock=True)

        assert transcriber.config.model_type == ModelType.WHISPER_LARGE_V3_GERMAN_MEDICAL
        assert transcriber.config.language == "de"

    def test_english_medical_config(self):
        """Test English medical configuration."""
        config = Config.for_english_medical()
        transcriber = MedicalTranscriber(config=config, use_mock=True)

        assert transcriber.config.model_type == ModelType.WHISPER_LARGE_V2_MEDICAL
        assert transcriber.config.language == "en"

    def test_lightweight_config(self):
        """Test lightweight configuration."""
        config = Config.for_lightweight()
        transcriber = MedicalTranscriber(config=config, use_mock=True)

        assert transcriber.config.model_type == ModelType.WHISPER_SMALL_MEDICAL
        assert transcriber.config.streaming.buffer_size_seconds == 3.0
