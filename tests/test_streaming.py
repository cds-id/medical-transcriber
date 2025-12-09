"""Tests for streaming module."""

import pytest
import numpy as np
import asyncio
from src.medical_stt.streaming import (
    QueueAudioSource,
    StreamingSession,
    AsyncStreamingSession,
)
from src.medical_stt.transcriber import MedicalTranscriber, TranscriptionResult
from src.medical_stt.config import Config


class TestQueueAudioSource:
    """Tests for QueueAudioSource."""

    def test_init(self):
        """Test initialization."""
        source = QueueAudioSource(sample_rate=16000)
        assert source.sample_rate == 16000
        assert not source.is_active()

    def test_start_stop(self):
        """Test starting and stopping source."""
        source = QueueAudioSource()
        assert not source.is_active()

        source.start()
        assert source.is_active()

        source.stop()
        assert not source.is_active()

    def test_put_and_read(self):
        """Test putting and reading audio."""
        source = QueueAudioSource()
        source.start()

        audio = np.random.randn(16000).astype(np.float32)
        source.put(audio)

        read_audio = source.read()
        np.testing.assert_array_equal(read_audio, audio)

    def test_read_empty(self):
        """Test reading from empty queue."""
        source = QueueAudioSource()
        source.start()

        result = source.read()
        assert result is None

    def test_multiple_chunks(self):
        """Test putting and reading multiple chunks."""
        source = QueueAudioSource()
        source.start()

        chunks = [np.random.randn(8000).astype(np.float32) for _ in range(3)]
        for chunk in chunks:
            source.put(chunk)

        for expected in chunks:
            received = source.read()
            np.testing.assert_array_equal(received, expected)


class TestStreamingSession:
    """Tests for StreamingSession."""

    @pytest.fixture
    def transcriber(self):
        """Create mock transcriber."""
        config = Config()
        return MedicalTranscriber(config=config, use_mock=True)

    @pytest.fixture
    def audio_source(self):
        """Create queue audio source."""
        return QueueAudioSource(sample_rate=16000)

    def test_init(self, transcriber, audio_source):
        """Test session initialization."""
        session = StreamingSession(
            transcriber=transcriber,
            audio_source=audio_source,
        )
        assert not session.is_running
        assert session.results == []

    def test_callback_collection(self, transcriber, audio_source):
        """Test results collected via callback."""
        results = []

        def callback(result):
            results.append(result)

        session = StreamingSession(
            transcriber=transcriber,
            audio_source=audio_source,
            on_result=callback,
        )

        # Just verify callback is set
        assert session.on_result == callback


class TestAsyncStreamingSession:
    """Tests for AsyncStreamingSession."""

    @pytest.fixture
    def transcriber(self):
        """Create mock transcriber."""
        config = Config()
        transcriber = MedicalTranscriber(config=config, use_mock=True)
        transcriber.load_model()
        return transcriber

    def test_init(self, transcriber):
        """Test async session initialization."""
        session = AsyncStreamingSession(transcriber=transcriber)
        assert session.results == []

    @pytest.mark.asyncio
    async def test_add_audio(self, transcriber):
        """Test adding audio to async session."""
        session = AsyncStreamingSession(transcriber=transcriber)

        audio = np.random.randn(16000).astype(np.float32)
        await session.add_audio(audio)

        # Audio should be in queue
        assert not session._queue.empty()

    @pytest.mark.asyncio
    async def test_stop(self, transcriber):
        """Test stopping async session."""
        session = AsyncStreamingSession(transcriber=transcriber)

        await session.stop()

        # Sentinel should be in queue
        item = await session._queue.get()
        assert item is None

    @pytest.mark.asyncio
    async def test_async_streaming(self, transcriber):
        """Test async streaming transcription."""
        session = AsyncStreamingSession(transcriber=transcriber)

        # Add audio chunks
        chunks = [np.random.randn(8000).astype(np.float32) for _ in range(3)]

        async def add_chunks():
            for chunk in chunks:
                await session.add_audio(chunk)
                await asyncio.sleep(0.01)
            await session.stop()

        results = []

        async def collect_results():
            async for result in session.run():
                results.append(result)

        # Run both tasks
        await asyncio.gather(add_chunks(), collect_results())

        assert len(results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in results)

    @pytest.mark.asyncio
    async def test_callback_invocation(self, transcriber):
        """Test callback is invoked during async streaming."""
        callback_results = []

        def callback(result):
            callback_results.append(result)

        session = AsyncStreamingSession(
            transcriber=transcriber,
            on_result=callback,
        )

        # Add some audio
        audio = np.random.randn(16000).astype(np.float32)

        async def run_session():
            await session.add_audio(audio)
            await session.stop()
            async for _ in session.run():
                pass

        await run_session()

        assert len(callback_results) > 0
