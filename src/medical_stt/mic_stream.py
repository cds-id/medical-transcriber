#!/usr/bin/env python3
"""Microphone streaming for Medical STT."""

import sys
import signal
import threading

from .config import Config, ModelType
from .transcriber import MedicalTranscriber
from .streaming import MicrophoneSource, StreamingSession


def run_mic_stream():
    """Run real-time microphone transcription."""
    print("=" * 50)
    print("  Medical STT - Microphone Streaming")
    print("=" * 50)
    print()

    # Use whisper-large-v3
    model_type = ModelType.WHISPER_LARGE_V3
    print(f"Model: {model_type.value}")

    # Create config (Indonesian language)
    config = Config(
        model_type=model_type,
        device="auto",
        language="id",  # Indonesian
    )

    print(f"Device: {config.get_device()}")
    print(f"Sample rate: {config.audio.sample_rate} Hz")
    print()

    # Load model
    print("Loading model...")
    transcriber = MedicalTranscriber(config=config, use_mock=False)

    try:
        transcriber.load_model()
        print("Model loaded!")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Run: ./scripts/download_model.sh")
        sys.exit(1)

    # Create microphone source
    try:
        mic = MicrophoneSource(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            chunk_duration_ms=config.audio.chunk_duration_ms,
        )
    except Exception as e:
        print(f"Error initializing microphone: {e}")
        print("Make sure you have a microphone connected and sounddevice installed.")
        sys.exit(1)

    # Result callback
    def on_result(result):
        status = "✓" if result.is_final else "..."
        print(f"  {status} {result.text}")

    # Create streaming session
    session = StreamingSession(
        transcriber=transcriber,
        audio_source=mic,
        on_result=on_result,
    )

    # Handle Ctrl+C
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n\nStopping...")
        stop_event.set()
        session.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Start streaming
    print("=" * 50)
    print("  Speak into your microphone (Ctrl+C to stop)")
    print("=" * 50)
    print()

    session.start()

    # Wait for stop signal
    stop_event.wait()

    print()
    print("Session ended.")
    print(f"Total transcriptions: {len(session.results)}")
    print()


if __name__ == "__main__":
    run_mic_stream()
