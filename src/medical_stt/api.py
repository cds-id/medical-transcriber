#!/usr/bin/env python3
"""FastAPI server for Medical STT with streaming and upload endpoints."""

import io
import os
import tempfile
import numpy as np
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import Config, ModelType
from .transcriber import MedicalTranscriber
from .postprocess import TranscriptionPostProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Medical Speech-to-Text API",
    description="API for transcribing medical audio using Whisper",
    version="1.0.0",
)

# Get static directory path
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global transcriber instance
transcriber: Optional[MedicalTranscriber] = None
postprocessor: Optional[TranscriptionPostProcessor] = None


def get_transcriber() -> MedicalTranscriber:
    """Get or initialize the transcriber."""
    global transcriber
    if transcriber is None:
        config = Config(
            model_type=ModelType.WHISPER_LARGE_V3,
            device="auto",
            language=None,  # Auto-detect language
        )
        transcriber = MedicalTranscriber(config=config, use_mock=False)
        transcriber.load_model()
    return transcriber


def get_postprocessor(
    provider: str = "ollama",
    model: Optional[str] = None,
) -> TranscriptionPostProcessor:
    """Get or initialize the post-processor."""
    global postprocessor
    if postprocessor is None or postprocessor.provider != provider:
        postprocessor = TranscriptionPostProcessor(provider=provider, model=model)
    return postprocessor


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("Loading model...")
    get_transcriber()
    print("Model loaded!")


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Medical STT API",
        "endpoints": {
            "upload": "/api/transcribe/upload",
            "stream": "/api/transcribe/stream (WebSocket)",
        }
    }


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format using ffmpeg."""
    import subprocess
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", output_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        # ffmpeg not installed, try pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            return True
        except:
            return False


@app.post("/api/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    language: Optional[str] = Query(default=None, description="Language code (id, en, de, etc.) or None for auto-detect"),
    postprocess: bool = Query(default=False, description="Use LLM to fix medical terms"),
    llm_provider: str = Query(default="ollama", description="LLM provider: ollama, openai, anthropic"),
    llm_model: Optional[str] = Query(default=None, description="LLM model name (optional)"),
):
    """
    Upload audio file and get transcription.

    Supports: wav, mp3, flac, ogg, m4a, webm
    """
    try:
        # Read uploaded file
        content = await file.read()

        # Save to temp file for processing
        suffix = f".{file.filename.split('.')[-1]}" if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        wav_path = tmp_path

        # Convert non-wav formats (especially webm from browser)
        if suffix.lower() in ['.webm', '.ogg', '.m4a', '.mp4', '.mp3', '.flac']:
            wav_path = tmp_path + ".wav"
            if not convert_to_wav(tmp_path, wav_path):
                os.unlink(tmp_path)
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": f"Could not convert {suffix} format. Please install ffmpeg.",
                    }
                )

        # Load audio
        from .audio_utils import load_audio_file
        audio, sr = load_audio_file(wav_path, target_sr=16000)

        # Clean up temp files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            os.unlink(wav_path)

        # Get transcriber and transcribe
        trans = get_transcriber()
        result = trans.transcribe(audio, sample_rate=sr, language=language)

        duration = len(audio) / sr

        # Get detected/used language
        detected_language = result.language or language or "auto"

        raw_transcription = result.text
        final_transcription = raw_transcription

        # Post-process with LLM if requested
        if postprocess:
            try:
                processor = get_postprocessor(provider=llm_provider, model=llm_model)
                # Use detected language for post-processing, default to "id" if unknown
                pp_language = detected_language if detected_language != "auto" else "id"
                final_transcription = processor.fix_transcription(
                    raw_transcription,
                    language=pp_language,
                    context="medical",
                )
            except Exception as e:
                # Log error but return raw transcription
                import logging
                logging.error(f"Post-processing failed: {e}")

        return JSONResponse(content={
            "success": True,
            "transcription": final_transcription,
            "raw_transcription": raw_transcription if postprocess else None,
            "postprocessed": postprocess,
            "language": detected_language,
            "duration_seconds": round(duration, 2),
            "filename": file.filename,
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            }
        )


@app.websocket("/api/transcribe/stream")
async def transcribe_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio transcription.

    Client sends audio chunks (bytes), server responds with transcriptions.

    Protocol:
    1. Client connects
    2. Client sends JSON: {"action": "start", "language": "id", "sample_rate": 16000}
    3. Client sends binary audio chunks
    4. Server responds with JSON: {"transcription": "...", "is_final": false}
    5. Client sends JSON: {"action": "stop"} to finalize
    6. Server responds with final transcription
    """
    await websocket.accept()

    audio_buffer = []
    sample_rate = 16000
    language = "id"

    try:
        while True:
            data = await websocket.receive()

            # Handle text messages (control commands)
            if "text" in data:
                import json
                msg = json.loads(data["text"])
                action = msg.get("action", "")

                if action == "start":
                    # Reset buffer and configure
                    audio_buffer = []
                    language = msg.get("language", "id")
                    sample_rate = msg.get("sample_rate", 16000)
                    await websocket.send_json({
                        "status": "started",
                        "language": language,
                        "sample_rate": sample_rate,
                    })

                elif action == "stop":
                    # Final transcription
                    if audio_buffer:
                        audio = np.concatenate(audio_buffer).astype(np.float32)
                        trans = get_transcriber()
                        result = trans.transcribe(audio, sample_rate=sample_rate, language=language)

                        await websocket.send_json({
                            "transcription": result.text,
                            "is_final": True,
                            "duration_seconds": round(len(audio) / sample_rate, 2),
                        })
                    else:
                        await websocket.send_json({
                            "transcription": "",
                            "is_final": True,
                            "duration_seconds": 0,
                        })

                    audio_buffer = []

                elif action == "transcribe_now":
                    # Transcribe current buffer without clearing
                    if audio_buffer:
                        audio = np.concatenate(audio_buffer).astype(np.float32)
                        trans = get_transcriber()
                        result = trans.transcribe(audio, sample_rate=sample_rate, language=language)

                        await websocket.send_json({
                            "transcription": result.text,
                            "is_final": False,
                            "duration_seconds": round(len(audio) / sample_rate, 2),
                        })

            # Handle binary messages (audio data)
            elif "bytes" in data:
                audio_bytes = data["bytes"]
                # Convert bytes to numpy array (assuming float32 or int16)
                try:
                    # Try float32 first
                    chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                except:
                    # Fall back to int16 and convert
                    chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                audio_buffer.append(chunk)

                # Send acknowledgment
                await websocket.send_json({
                    "status": "chunk_received",
                    "buffer_duration": round(sum(len(c) for c in audio_buffer) / sample_rate, 2),
                })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "error": str(e),
        })


@app.get("/api/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"id": "large", "name": "openai/whisper-large-v3", "size": "~3GB", "description": "Best accuracy"},
            {"id": "turbo", "name": "openai/whisper-large-v3-turbo", "size": "~1.6GB", "description": "Fast & accurate"},
            {"id": "medium", "name": "openai/whisper-medium", "size": "~1.5GB", "description": "Balanced"},
            {"id": "small", "name": "openai/whisper-small", "size": "~500MB", "description": "Fastest"},
        ],
        "current": get_transcriber().config.model_id,
    }


def main():
    """Run the API server."""
    uvicorn.run(
        "src.medical_stt.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
