"""Medical Speech-to-Text Streaming Module.

A POC for streaming speech-to-text transcription optimized for medical terminology.
"""

from .config import Config, ModelType
from .transcriber import MedicalTranscriber, TranscriptionResult

__version__ = "0.1.0"
__all__ = ["Config", "ModelType", "MedicalTranscriber", "TranscriptionResult"]
