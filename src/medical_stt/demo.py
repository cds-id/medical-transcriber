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

    # Check for model argument
    model_map = {
        "small": ModelType.WHISPER_SMALL_MEDICAL,
        "large-en": ModelType.WHISPER_LARGE_V2_MEDICAL,
        "large-de": ModelType.WHISPER_LARGE_V3_GERMAN_MEDICAL,
    }

    model_key = sys.argv[1] if len(sys.argv) > 1 else "small"

    if model_key not in model_map:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(model_map.keys())}")
        sys.exit(1)

    model_type = model_map[model_key]
    print(f"Model: {model_type.value}")
    print()

    # Create config
    config = Config(
        model_type=model_type,
        device="auto",
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
        print(f"Loading audio from: {audio_file}")
        from .audio_utils import load_audio_file
        audio, sr = load_audio_file(audio_file, target_sr=config.audio.sample_rate)
    else:
        print("Using generated test audio (silence - for real test, provide audio file)")
        print("Usage: python -m src.medical_stt.demo [model] [audio_file.wav]")
        print()
        # Generate 3 seconds of test audio
        audio = generate_test_audio(duration_seconds=3.0)

    # Transcribe
    print("Transcribing...")
    print("-" * 50)

    result = transcriber.transcribe(audio)

    print(f"Transcription: {result.text}")
    print("-" * 50)
    print()

    if result.text.strip():
        print("✓ Transcription complete!")
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
