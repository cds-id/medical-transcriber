#!/usr/bin/env python3
"""Demo script for Medical STT transcription."""

import sys
import numpy as np

from .config import Config, ModelType
from .transcriber import MedicalTranscriber
from .audio_utils import generate_test_audio


def run_demo():
    """Run a demo transcription."""
    print("=" * 50)
    print("  Medical Speech-to-Text Demo")
    print("=" * 50)
    print()

    # Use whisper-large-v3
    model_type = ModelType.WHISPER_LARGE_V3
    print(f"Model: {model_type.value}")
    print()

    # Create config (Indonesian language)
    config = Config(
        model_type=model_type,
        device="auto",
        language="id",  # Indonesian
    )

    print(f"Device: {config.get_device()}")
    print()

    # Create transcriber
    print("Loading model (this may take a moment on first run)...")
    transcriber = MedicalTranscriber(config=config, use_mock=False)

    try:
        transcriber.load_model()
        print("Model loaded successfully!")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        print()
        print("Have you downloaded the model?")
        print("Run: ./scripts/download_model.sh")
        sys.exit(1)

    # Generate test audio (or load from file if provided)
    if len(sys.argv) > 2:
        audio_file = sys.argv[2]
    else:
        # Check for default test file
        import os
        default_files = ["tests.wav", "test.wav", "samples/sample_english.wav"]
        audio_file = None
        for f in default_files:
            if os.path.exists(f):
                audio_file = f
                break

    if audio_file:
        print(f"Loading audio from: {audio_file}")
        from .audio_utils import load_audio_file
        audio, sr = load_audio_file(audio_file, target_sr=config.audio.sample_rate)
        duration = len(audio) / config.audio.sample_rate
        print(f"Audio duration: {duration:.1f} seconds")
    else:
        print("Using generated test audio (silence - for real test, provide audio file)")
        print("Usage: python -m src.medical_stt.demo [model] [audio_file.wav]")
        print()
        # Generate 3 seconds of test audio
        audio = generate_test_audio(duration_seconds=3.0)

    # Transcribe
    print("Transcribing... (this may take a while for long audio)")
    print("-" * 50)

    import time
    start_time = time.time()

    result = transcriber.transcribe(audio)

    elapsed = time.time() - start_time
    print(f"\nTranscription ({elapsed:.1f}s):")
    print("-" * 50)
    print(result.text)
    print("-" * 50)
    print()

    if result.text.strip():
        print(f"✓ Transcription complete! ({elapsed:.1f} seconds)")
    else:
        print("(No speech detected in audio)")

    print()


def run_streaming_demo():
    """Run a streaming demo with chunked audio."""
    print("=" * 50)
    print("  Streaming Demo")
    print("=" * 50)
    print()

    config = Config.for_lightweight()
    transcriber = MedicalTranscriber(config=config, use_mock=False)

    try:
        transcriber.load_model()
    except Exception as e:
        print(f"Error: {e}")
        print("Run: ./scripts/download_model.sh")
        sys.exit(1)

    # Generate chunks of audio
    print("Simulating streaming with audio chunks...")
    print()

    def audio_generator():
        for i in range(5):
            chunk = generate_test_audio(duration_seconds=1.0)
            print(f"  Chunk {i+1}/5 sent...")
            yield chunk

    print("Transcription results:")
    print("-" * 50)

    for result in transcriber.stream_transcribe(audio_generator()):
        status = "[FINAL]" if result.is_final else "[partial]"
        print(f"  {status} {result.text}")

    print("-" * 50)
    print()


if __name__ == "__main__":
    run_demo()
