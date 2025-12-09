# Medical Speech-to-Text (STT) Streaming

A proof-of-concept for streaming speech-to-text transcription optimized for medical terminology using OpenAI Whisper Large V3.

## Features

- **Streaming transcription** - Real-time audio processing with sync and async support
- **OpenAI Whisper Large V3** - State-of-the-art speech recognition
- **Indonesian language support** - Optimized for Indonesian medical transcription
- **Microphone streaming** - Live transcription from microphone input
- **Mock backend** - Test without downloading models

## Model

| Model | Size | Description |
|-------|------|-------------|
| `openai/whisper-large-v3` | ~3GB | OpenAI Whisper Large V3 - Best accuracy |

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

## Download Model

```bash
# Download whisper-large-v3
./scripts/download_model.sh

# With HuggingFace token (for gated models)
export HF_TOKEN=hf_xxxxxxxxxxxx
./scripts/download_model.sh
```

## Usage

### Run Demo

```bash
# With default test file (tests.wav)
./scripts/run.sh demo

# Or manually
source venv/bin/activate
python -m src.medical_stt.demo

# With specific audio file
python -m src.medical_stt.demo audio_file.wav
```

### Microphone Streaming

```bash
./scripts/run.sh mic

# Or manually
source venv/bin/activate
python -m src.medical_stt.mic_stream
```

### Python API

```python
from src.medical_stt import MedicalTranscriber, Config, ModelType

# Create configuration (Indonesian by default)
config = Config(
    model_type=ModelType.WHISPER_LARGE_V3,
    device="auto",  # auto, cpu, cuda, mps
    language="id",  # Indonesian
)

# Or use presets
config = Config.for_indonesian()  # Indonesian
config = Config.for_english()     # English
config = Config.for_german()      # German
config = Config.for_medical()     # Medical (Indonesian default)

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

config = Config.for_indonesian()
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
    config = Config.for_indonesian()
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

### With HuggingFace Token

```python
from src.medical_stt import MedicalTranscriber, Config

config = Config(
    hf_token="hf_xxxxxxxxxxxx",  # For gated models
    language="id",
)

transcriber = MedicalTranscriber(config=config)
transcriber.load_model()
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

### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | WHISPER_LARGE_V3 | Model to use |
| `device` | auto | Device (auto, cpu, cuda, mps) |
| `language` | None | Language code (id, en, de, etc.) |
| `hf_token` | None | HuggingFace API token |

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

## Supported Languages

Whisper Large V3 supports 99+ languages including:
- Indonesian (id)
- English (en)
- German (de)
- And many more...

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers 4.36+
- sounddevice (for microphone)
- CUDA (optional, for GPU acceleration)

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
