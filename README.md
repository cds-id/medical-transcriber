# Medical Speech-to-Text (STT) Streaming

A proof-of-concept for streaming speech-to-text transcription optimized for medical terminology using HuggingFace Whisper models.

## Features

- **Streaming transcription** - Real-time audio processing with sync and async support
- **Medical-optimized models** - Pre-configured for medical terminology recognition
- **Multiple model support** - Choose between speed and accuracy
- **Microphone streaming** - Live transcription from microphone input
- **Mock backend** - Test without downloading models

## Supported Models

| Model | Size | Language | Use Case |
|-------|------|----------|----------|
| `ayoubkirouane/whisper-small-medical` | ~500MB | Multi | Fast, lightweight |
| `MohamedRashad/whisper-large-v2-medical` | ~3GB | English | High accuracy |
| `primeLine/whisper-large-v3-german-medical` | ~3GB | German | German medical |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd medical-analytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Download Models

```bash
# Interactive model selection
./scripts/download_model.sh

# Or download specific model
./scripts/download_model.sh 1  # small (fast)
./scripts/download_model.sh 2  # large-v2 (English)
./scripts/download_model.sh 3  # large-v3 (German)
./scripts/download_model.sh 4  # all models
```

## Usage

### Run Demo

```bash
# With default test file (tests.wav)
./scripts/run.sh demo

# Or manually
source venv/bin/activate
python -m src.medical_stt.demo small

# With specific audio file
python -m src.medical_stt.demo small /path/to/audio.wav
```

### Microphone Streaming

```bash
./scripts/run.sh mic

# Or with specific model
source venv/bin/activate
python -m src.medical_stt.mic_stream small
```

### Python API

```python
from src.medical_stt import MedicalTranscriber, Config, ModelType

# Create configuration
config = Config(
    model_type=ModelType.WHISPER_SMALL_MEDICAL,
    device="auto",  # auto, cpu, cuda, mps
    language="en",
)

# Or use presets
config = Config.for_english_medical()  # Large v2
config = Config.for_german_medical()   # Large v3 German
config = Config.for_lightweight()      # Small, fast

# Initialize transcriber
transcriber = MedicalTranscriber(config=config)
transcriber.load_model()

# Transcribe audio file
from src.medical_stt.audio_utils import load_audio_file
audio, sr = load_audio_file("recording.wav", target_sr=16000)
result = transcriber.transcribe(audio)
print(result.text)
```

### Streaming API

```python
import numpy as np
from src.medical_stt import MedicalTranscriber, Config

config = Config.for_lightweight()
transcriber = MedicalTranscriber(config=config)
transcriber.load_model()

# Sync streaming
def audio_generator():
    # Yield audio chunks (numpy arrays)
    for chunk in get_audio_chunks():
        yield chunk

for result in transcriber.stream_transcribe(audio_generator()):
    print(f"[{'FINAL' if result.is_final else 'partial'}] {result.text}")
```

### Async Streaming

```python
import asyncio
from src.medical_stt import MedicalTranscriber, Config
from src.medical_stt.streaming import AsyncStreamingSession

async def main():
    config = Config.for_lightweight()
    transcriber = MedicalTranscriber(config=config)
    transcriber.load_model()

    session = AsyncStreamingSession(transcriber=transcriber)

    # Add audio chunks
    await session.add_audio(audio_chunk)

    # Process results
    async for result in session.run():
        print(result.text)

    await session.stop()

asyncio.run(main())
```

## Testing

```bash
# Run all tests
./scripts/run.sh test

# Or with pytest directly
source venv/bin/activate
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/medical_stt --cov-report=html
```

## Project Structure

```
medical-analytics/
├── src/
│   └── medical_stt/
│       ├── __init__.py       # Package exports
│       ├── config.py         # Configuration classes
│       ├── transcriber.py    # Main transcriber
│       ├── streaming.py      # Streaming components
│       ├── audio_utils.py    # Audio utilities
│       ├── demo.py           # Demo script
│       └── mic_stream.py     # Microphone streaming
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_config.py
│   ├── test_transcriber.py
│   ├── test_streaming.py
│   └── test_audio_utils.py
├── scripts/
│   ├── download_model.sh     # Model downloader
│   ├── run.sh                # App runner
│   └── download_sample_audio.sh
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Configuration Options

### AudioConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate in Hz |
| `channels` | 1 | Number of audio channels |
| `chunk_duration_ms` | 1000 | Chunk size for streaming |

### StreamingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size_seconds` | 5.0 | Audio buffer size |
| `overlap_seconds` | 0.5 | Overlap between chunks |
| `min_audio_length_seconds` | 0.5 | Minimum audio to process |
| `vad_enabled` | True | Voice Activity Detection |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers 4.36+
- sounddevice (for microphone)
- CUDA (optional, for GPU acceleration)

## License

MIT

## Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- Medical model fine-tuners:
  - [ayoubkirouane](https://huggingface.co/ayoubkirouane)
  - [MohamedRashad](https://huggingface.co/MohamedRashad)
  - [primeLine](https://huggingface.co/primeLine)
