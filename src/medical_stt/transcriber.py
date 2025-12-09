"""Medical Speech-to-Text Transcriber with streaming support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterator, List, Optional, Protocol, Union
import asyncio
import logging
import time

import numpy as np

from .config import Config


logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    is_final: bool = False
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    language: Optional[str] = None
    medical_terms: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


class AudioProcessor:
    """Process audio chunks for transcription."""

    def __init__(self, config: Config):
        self.config = config
        self._buffer: List[np.ndarray] = []
        self._total_samples = 0

    def add_chunk(self, audio: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        self._buffer.append(audio)
        self._total_samples += len(audio)

    def get_buffer(self) -> np.ndarray:
        """Get concatenated buffer."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer).astype(np.float32)

    def get_duration(self) -> float:
        """Get buffer duration in seconds."""
        return self._total_samples / self.config.audio.sample_rate

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_samples = 0

    def trim_buffer(self, keep_seconds: float) -> None:
        """Keep only the last N seconds of audio."""
        samples_to_keep = int(keep_seconds * self.config.audio.sample_rate)
        if self._total_samples <= samples_to_keep:
            return

        full_audio = self.get_buffer()
        trimmed = full_audio[-samples_to_keep:]
        self._buffer = [trimmed]
        self._total_samples = len(trimmed)


class ModelLoader(Protocol):
    """Protocol for model loading."""

    def load_model(self, model_id: str, device: str) -> object:
        """Load the model."""
        ...

    def load_processor(self, model_id: str) -> object:
        """Load the processor."""
        ...


class TranscriberBackend(ABC):
    """Abstract backend for transcription."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...


class HuggingFaceBackend(TranscriberBackend):
    """HuggingFace Transformers backend for transcription."""

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load the model and processor from HuggingFace."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        self._device = self.config.get_device()
        self._torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info(f"Loading model {self.config.model_id} on {self._device}")

        # Prepare kwargs for HuggingFace token
        hf_kwargs = {}
        if self.config.hf_token:
            hf_kwargs["token"] = self.config.hf_token

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            **hf_kwargs,
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            **hf_kwargs,
        ).to(self._device)

        logger.info("Model loaded successfully")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        # Ensure input features match model dtype (fix float32/float16 mismatch on CUDA)
        input_features = inputs.input_features.to(self._device, dtype=self._torch_dtype)

        # Generate
        generate_kwargs = {"task": self.config.task}
        if language:
            generate_kwargs["language"] = language

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                **generate_kwargs,
            )

        # Decode
        transcription = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return TranscriptionResult(
            text=transcription.strip(),
            is_final=True,
            language=language or self.config.language,
        )

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._processor is not None


class MockBackend(TranscriberBackend):
    """Mock backend for testing without loading actual models."""

    def __init__(self, config: Config):
        self.config = config
        self._loaded = False
        self._responses: List[str] = [
            "Patient presents with acute symptoms.",
            "Blood pressure measured at 120/80 mmHg.",
            "No signs of cardiovascular abnormalities.",
            "Prescription: Amoxicillin 500mg three times daily.",
        ]
        self._call_count = 0

    def load(self) -> None:
        """Simulate model loading."""
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Return mock transcription."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Simulate processing time based on audio length
        duration = len(audio) / sample_rate
        time.sleep(min(0.1, duration * 0.01))

        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1

        return TranscriptionResult(
            text=response,
            is_final=True,
            confidence=0.95,
            language=language or "en",
            medical_terms=["patient", "blood pressure", "mmHg"],
        )

    def is_loaded(self) -> bool:
        """Check if mock is loaded."""
        return self._loaded


class MedicalTranscriber:
    """Main transcriber class for medical speech-to-text."""

    def __init__(
        self,
        config: Optional[Config] = None,
        backend: Optional[TranscriberBackend] = None,
        use_mock: bool = False,
    ):
        self.config = config or Config()
        self._audio_processor = AudioProcessor(self.config)

        if backend:
            self._backend = backend
        elif use_mock:
            self._backend = MockBackend(self.config)
        else:
            self._backend = HuggingFaceBackend(self.config)

        self._is_streaming = False
        self._callbacks: List[Callable[[TranscriptionResult], None]] = []

    def load_model(self) -> None:
        """Load the transcription model."""
        if hasattr(self._backend, "load"):
            self._backend.load()

    def is_ready(self) -> bool:
        """Check if transcriber is ready."""
        return self._backend.is_loaded()

    def transcribe(
        self,
        audio: Union[np.ndarray, bytes],
        sample_rate: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe a complete audio segment."""
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.float32)

        sr = sample_rate or self.config.audio.sample_rate

        return self._backend.transcribe(
            audio=audio,
            sample_rate=sr,
            language=self.config.language,
        )

    def add_callback(
        self, callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """Add callback for streaming results."""
        self._callbacks.append(callback)

    def remove_callback(
        self, callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, result: TranscriptionResult) -> None:
        """Notify all callbacks with a result."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def stream_transcribe(
        self,
        audio_stream: Iterator[np.ndarray],
    ) -> Iterator[TranscriptionResult]:
        """Stream transcription from an audio iterator."""
        self._is_streaming = True
        self._audio_processor.clear()

        try:
            for chunk in audio_stream:
                if not self._is_streaming:
                    break

                self._audio_processor.add_chunk(chunk)

                # Check if we have enough audio to process
                if self._audio_processor.get_duration() >= self.config.streaming.min_audio_length_seconds:
                    audio = self._audio_processor.get_buffer()
                    result = self._backend.transcribe(
                        audio=audio,
                        sample_rate=self.config.audio.sample_rate,
                        language=self.config.language,
                    )
                    result.is_final = False

                    self._notify_callbacks(result)
                    yield result

                    # Trim buffer but keep overlap
                    self._audio_processor.trim_buffer(
                        self.config.streaming.overlap_seconds
                    )

            # Final transcription
            if self._audio_processor.get_duration() > 0:
                audio = self._audio_processor.get_buffer()
                result = self._backend.transcribe(
                    audio=audio,
                    sample_rate=self.config.audio.sample_rate,
                    language=self.config.language,
                )
                result.is_final = True
                self._notify_callbacks(result)
                yield result

        finally:
            self._is_streaming = False
            self._audio_processor.clear()

    async def async_stream_transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[TranscriptionResult]:
        """Async stream transcription."""
        self._is_streaming = True
        self._audio_processor.clear()

        try:
            async for chunk in audio_stream:
                if not self._is_streaming:
                    break

                self._audio_processor.add_chunk(chunk)

                if self._audio_processor.get_duration() >= self.config.streaming.min_audio_length_seconds:
                    audio = self._audio_processor.get_buffer()

                    # Run transcription in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._backend.transcribe(
                            audio=audio,
                            sample_rate=self.config.audio.sample_rate,
                            language=self.config.language,
                        ),
                    )
                    result.is_final = False

                    self._notify_callbacks(result)
                    yield result

                    self._audio_processor.trim_buffer(
                        self.config.streaming.overlap_seconds
                    )

            # Final transcription
            if self._audio_processor.get_duration() > 0:
                audio = self._audio_processor.get_buffer()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._backend.transcribe(
                        audio=audio,
                        sample_rate=self.config.audio.sample_rate,
                        language=self.config.language,
                    ),
                )
                result.is_final = True
                self._notify_callbacks(result)
                yield result

        finally:
            self._is_streaming = False
            self._audio_processor.clear()

    def stop_streaming(self) -> None:
        """Stop the current streaming session."""
        self._is_streaming = False

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
