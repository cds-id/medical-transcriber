"""Streaming components for real-time audio transcription."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Optional
import asyncio
import logging
import queue
import threading

import numpy as np

from .config import Config
from .transcriber import MedicalTranscriber, TranscriptionResult


logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """Abstract base class for audio sources."""

    @abstractmethod
    def start(self) -> None:
        """Start the audio source."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the audio source."""
        ...

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read audio chunk."""
        ...

    @abstractmethod
    def is_active(self) -> bool:
        """Check if source is active."""
        ...


class MicrophoneSource(AudioSource):
    """Microphone audio source using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 1000,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.device = device

        self._stream = None
        self._queue: queue.Queue = queue.Queue()
        self._active = False

    def _callback(self, indata, frames, time, status):
        """Sounddevice callback."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self._queue.put(indata.copy())

    def start(self) -> None:
        """Start microphone recording."""
        import sounddevice as sd

        self._active = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("Microphone started")

    def stop(self) -> None:
        """Stop microphone recording."""
        self._active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Microphone stopped")

    def read(self) -> Optional[np.ndarray]:
        """Read audio chunk from queue."""
        try:
            return self._queue.get(timeout=1.0).flatten()
        except queue.Empty:
            return None

    def is_active(self) -> bool:
        """Check if microphone is active."""
        return self._active and self._stream is not None


class QueueAudioSource(AudioSource):
    """Audio source from a queue (for testing or WebSocket input)."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._active = False

    def start(self) -> None:
        """Start the source."""
        self._active = True

    def stop(self) -> None:
        """Stop the source."""
        self._active = False

    def put(self, audio: np.ndarray) -> None:
        """Add audio to the queue."""
        self._queue.put(audio)

    def read(self) -> Optional[np.ndarray]:
        """Read audio from queue."""
        try:
            return self._queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def is_active(self) -> bool:
        """Check if source is active."""
        return self._active


class StreamingSession:
    """Manages a streaming transcription session."""

    def __init__(
        self,
        transcriber: MedicalTranscriber,
        audio_source: AudioSource,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        self.transcriber = transcriber
        self.audio_source = audio_source
        self.on_result = on_result

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._results: list = []

    def start(self) -> None:
        """Start the streaming session."""
        if self._running:
            return

        self._running = True
        self.audio_source.start()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Streaming session started")

    def stop(self) -> None:
        """Stop the streaming session."""
        self._running = False
        self.audio_source.stop()
        self.transcriber.stop_streaming()

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("Streaming session stopped")

    def _run(self) -> None:
        """Main streaming loop."""

        def audio_generator():
            while self._running and self.audio_source.is_active():
                chunk = self.audio_source.read()
                if chunk is not None:
                    yield chunk

        try:
            for result in self.transcriber.stream_transcribe(audio_generator()):
                self._results.append(result)
                if self.on_result:
                    self.on_result(result)
        except Exception as e:
            logger.error(f"Streaming error: {e}")

    @property
    def results(self) -> list:
        """Get all results."""
        return self._results

    @property
    def is_running(self) -> bool:
        """Check if session is running."""
        return self._running


class AsyncStreamingSession:
    """Async streaming session for WebSocket/HTTP streaming."""

    def __init__(
        self,
        transcriber: MedicalTranscriber,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        self.transcriber = transcriber
        self.on_result = on_result

        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._results: list = []

    async def add_audio(self, audio: np.ndarray) -> None:
        """Add audio chunk to processing queue."""
        await self._queue.put(audio)

    async def stop(self) -> None:
        """Signal end of audio stream."""
        self._running = False
        await self._queue.put(None)  # Sentinel

    async def _audio_generator(self) -> AsyncIterator[np.ndarray]:
        """Generate audio chunks from queue."""
        self._running = True
        while self._running:
            chunk = await self._queue.get()
            if chunk is None:  # Sentinel
                break
            yield chunk

    async def run(self) -> AsyncIterator[TranscriptionResult]:
        """Run the streaming session."""
        async for result in self.transcriber.async_stream_transcribe(
            self._audio_generator()
        ):
            self._results.append(result)
            if self.on_result:
                self.on_result(result)
            yield result

    @property
    def results(self) -> list:
        """Get all results."""
        return self._results
