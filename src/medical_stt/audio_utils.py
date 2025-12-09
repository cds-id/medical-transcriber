"""Audio utility functions for Medical STT."""

from typing import Generator, Optional, Tuple
import io

import numpy as np


def load_audio_file(
    file_path: str,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample if needed.

    Args:
        file_path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        Tuple of (audio array, sample rate).
    """
    import soundfile as sf
    import librosa

    audio, sr = sf.read(file_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def load_audio_bytes(
    audio_bytes: bytes,
    target_sr: int = 16000,
    format: str = "wav",
) -> Tuple[np.ndarray, int]:
    """Load audio from bytes.

    Args:
        audio_bytes: Raw audio bytes.
        target_sr: Target sample rate.
        format: Audio format (wav, mp3, etc.).

    Returns:
        Tuple of (audio array, sample rate).
    """
    import soundfile as sf
    import librosa

    buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buffer)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def chunk_audio(
    audio: np.ndarray,
    chunk_size: int,
    overlap: int = 0,
) -> Generator[np.ndarray, None, None]:
    """Split audio into chunks.

    Args:
        audio: Audio array.
        chunk_size: Size of each chunk in samples.
        overlap: Overlap between chunks in samples.

    Yields:
        Audio chunks.
    """
    start = 0
    step = chunk_size - overlap

    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        yield audio[start:end]
        start += step


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.

    Args:
        audio: Audio array.

    Returns:
        Normalized audio array.
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_silence_duration: int = 1000,
    sample_rate: int = 16000,
) -> bool:
    """Detect if audio chunk is silence.

    Args:
        audio: Audio array.
        threshold: RMS threshold for silence.
        min_silence_duration: Minimum samples to consider silence.
        sample_rate: Sample rate.

    Returns:
        True if audio is silence.
    """
    if len(audio) < min_silence_duration:
        return False

    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold


def generate_test_audio(
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> np.ndarray:
    """Generate test audio signal (sine wave).

    Args:
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate.
        frequency: Frequency in Hz.

    Returns:
        Audio array.
    """
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def generate_silence(
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate silence.

    Args:
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate.

    Returns:
        Silent audio array.
    """
    return np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
