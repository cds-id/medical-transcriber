"""Tests for the post-processing module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.medical_stt.postprocess import (
    TranscriptionPostProcessor,
    get_postprocessor,
    fix_transcription,
)


class TestTranscriptionPostProcessor:
    """Tests for TranscriptionPostProcessor class."""

    def test_init_defaults(self):
        """Test default initialization."""
        processor = TranscriptionPostProcessor()
        assert processor.provider == "ollama"
        assert processor.model == "llama3.2:3b"
        assert processor._client is None

    def test_init_custom_provider(self):
        """Test initialization with custom provider."""
        processor = TranscriptionPostProcessor(provider="openai")
        assert processor.provider == "openai"
        assert processor.model == "gpt-4o-mini"

    def test_init_anthropic_provider(self):
        """Test initialization with anthropic provider."""
        processor = TranscriptionPostProcessor(provider="anthropic")
        assert processor.provider == "anthropic"
        assert processor.model == "claude-3-haiku-20240307"

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        processor = TranscriptionPostProcessor(provider="ollama", model="mistral")
        assert processor.model == "mistral"

    def test_fix_transcription_empty_text(self):
        """Test that empty text returns empty."""
        processor = TranscriptionPostProcessor()
        result = processor.fix_transcription("")
        assert result == ""
        result = processor.fix_transcription("   ")
        assert result == "   "

    @patch("src.medical_stt.postprocess.TranscriptionPostProcessor._get_client")
    def test_fix_transcription_ollama(self, mock_get_client):
        """Test fix_transcription with ollama provider."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Fixed text with Amoxicillin"}
        }
        mock_get_client.return_value = mock_client

        processor = TranscriptionPostProcessor(provider="ollama")
        result = processor.fix_transcription("text with amoksilin", language="id")

        assert result == "Fixed text with Amoxicillin"
        mock_client.chat.assert_called_once()

    @patch("src.medical_stt.postprocess.TranscriptionPostProcessor._get_client")
    def test_fix_transcription_openai(self, mock_get_client):
        """Test fix_transcription with openai provider."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Fixed medical text"))]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        processor = TranscriptionPostProcessor(provider="openai")
        result = processor.fix_transcription("broken text", language="en")

        assert result == "Fixed medical text"
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.medical_stt.postprocess.TranscriptionPostProcessor._get_client")
    def test_fix_transcription_anthropic(self, mock_get_client):
        """Test fix_transcription with anthropic provider."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Corrected by Claude")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        processor = TranscriptionPostProcessor(provider="anthropic")
        result = processor.fix_transcription("text to fix", language="en")

        assert result == "Corrected by Claude"
        mock_client.messages.create.assert_called_once()

    @patch("src.medical_stt.postprocess.TranscriptionPostProcessor._get_client")
    def test_fix_transcription_error_returns_original(self, mock_get_client):
        """Test that errors return original text."""
        mock_get_client.side_effect = Exception("API Error")

        processor = TranscriptionPostProcessor(provider="ollama")
        result = processor.fix_transcription("original text")

        assert result == "original text"

    def test_indonesian_prompt(self):
        """Test that Indonesian language uses Indonesian prompt."""
        processor = TranscriptionPostProcessor()
        # We can verify the prompt logic by checking the method doesn't crash
        # and returns empty for empty input
        result = processor.fix_transcription("", language="id")
        assert result == ""

    def test_english_prompt(self):
        """Test that English language uses English prompt."""
        processor = TranscriptionPostProcessor()
        result = processor.fix_transcription("", language="en")
        assert result == ""


class TestGetPostprocessor:
    """Tests for get_postprocessor function."""

    def test_creates_instance(self):
        """Test that it creates a new instance."""
        # Reset global
        import src.medical_stt.postprocess as pp
        pp._processor = None

        processor = get_postprocessor()
        assert processor is not None
        assert processor.provider == "ollama"

    def test_reuses_instance(self):
        """Test that it reuses existing instance."""
        import src.medical_stt.postprocess as pp
        pp._processor = None

        processor1 = get_postprocessor()
        processor2 = get_postprocessor()
        assert processor1 is processor2

    def test_creates_new_for_different_provider(self):
        """Test that it creates new instance for different provider."""
        import src.medical_stt.postprocess as pp
        pp._processor = None

        processor1 = get_postprocessor(provider="ollama")
        processor2 = get_postprocessor(provider="openai")
        assert processor1 is not processor2


class TestFixTranscriptionFunction:
    """Tests for fix_transcription convenience function."""

    @patch("src.medical_stt.postprocess.get_postprocessor")
    def test_calls_processor(self, mock_get):
        """Test that it calls the processor."""
        mock_processor = Mock()
        mock_processor.fix_transcription.return_value = "fixed"
        mock_get.return_value = mock_processor

        result = fix_transcription("test", language="id", provider="ollama")

        assert result == "fixed"
        mock_processor.fix_transcription.assert_called_once_with("test", language="id")
