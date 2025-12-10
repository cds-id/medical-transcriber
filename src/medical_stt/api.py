#!/usr/bin/env python3
"""FastAPI server for Medical STT with streaming and upload endpoints."""

import io
import os
import gc
import secrets
import subprocess
import tempfile
import numpy as np
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn

from .config import Config, ModelType
from .transcriber import MedicalTranscriber
from .postprocess import TranscriptionPostProcessor
from . import image_gen
from . import video_gen

# Initialize FastAPI app
app = FastAPI(
    title="Medical Speech-to-Text API",
    description="API for transcribing medical audio using Whisper",
    version="1.0.0",
)

# Basic Auth configuration
security = HTTPBasic()

# Get credentials from environment variables (with defaults for development)
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "medical123")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify basic auth credentials."""
    correct_username = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


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
async def root(username: str = Depends(verify_credentials)):
    """Serve the web UI (protected)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/image")
async def image_page(username: str = Depends(verify_credentials)):
    """Serve the image generation page (protected)."""
    return FileResponse(STATIC_DIR / "image.html")


@app.get("/health")
async def health():
    """Health check endpoint (public)."""
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


@app.post("/api/unload")
async def unload_models(
    unload_whisper: bool = Query(default=True, description="Unload Whisper model from GPU"),
    stop_ollama: bool = Query(default=True, description="Stop Ollama service"),
):
    """
    Unload models to free GPU memory for other tasks (e.g., image generation).

    This will:
    - Unload Whisper model from GPU memory
    - Stop Ollama service (optional)
    - Run garbage collection and clear CUDA cache
    """
    global transcriber, postprocessor

    results = {
        "whisper_unloaded": False,
        "ollama_stopped": False,
        "gpu_memory_cleared": False,
    }

    # Unload Whisper model
    if unload_whisper and transcriber is not None:
        try:
            # Delete the transcriber and its backend
            del transcriber
            transcriber = None
            results["whisper_unloaded"] = True
        except Exception as e:
            results["whisper_error"] = str(e)

    # Clear postprocessor
    if postprocessor is not None:
        try:
            del postprocessor
            postprocessor = None
        except:
            pass

    # Stop Ollama service
    if stop_ollama:
        try:
            # Try systemctl first
            result = subprocess.run(
                ["sudo", "systemctl", "stop", "ollama"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                results["ollama_stopped"] = True
            else:
                # Try pkill as fallback
                subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)
                results["ollama_stopped"] = True
        except Exception as e:
            results["ollama_error"] = str(e)

    # Clear GPU memory
    try:
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                results["gpu_memory_cleared"] = True
        except ImportError:
            pass
    except Exception as e:
        results["gc_error"] = str(e)

    # Get current GPU memory status
    try:
        import torch
        if torch.cuda.is_available():
            results["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
            results["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
    except:
        pass

    return JSONResponse(content={
        "success": True,
        "message": "Models unloaded. GPU memory freed for image generation.",
        "details": results,
    })


@app.post("/api/reload")
async def reload_models():
    """
    Reload Whisper model and restart Ollama after image generation.
    """
    results = {
        "whisper_loaded": False,
        "ollama_started": False,
    }

    # Start Ollama service
    try:
        # Try systemctl first
        result = subprocess.run(
            ["sudo", "systemctl", "start", "ollama"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            results["ollama_started"] = True
        else:
            # Try starting ollama directly as fallback
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            results["ollama_started"] = True
    except Exception as e:
        results["ollama_error"] = str(e)

    # Reload Whisper model
    try:
        get_transcriber()
        results["whisper_loaded"] = True
    except Exception as e:
        results["whisper_error"] = str(e)

    # Get current GPU memory status
    try:
        import torch
        if torch.cuda.is_available():
            results["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
            results["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
    except:
        pass

    return JSONResponse(content={
        "success": True,
        "message": "Models reloaded. Ready for transcription.",
        "details": results,
    })


@app.get("/api/gpu/status")
async def gpu_status():
    """Get current GPU memory status."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
                "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2),
                "whisper_loaded": transcriber is not None,
                "ollama_running": _check_ollama_running(),
            }
        else:
            return {"gpu_available": False, "device": "cpu"}
    except ImportError:
        return {"gpu_available": False, "error": "torch not installed"}


def _check_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-x", "ollama"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except:
        return False


# ============== Image Generation Endpoints ==============

@app.get("/api/image/models")
async def list_image_models():
    """List available image generation models."""
    models = image_gen.get_available_models()
    current = image_gen.get_current_model()
    return {
        "models": [
            {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "default_steps": config["default_steps"],
                "default_guidance": config["default_guidance"],
                "supports_negative": config["supports_negative"],
            }
            for model_id, config in models.items()
        ],
        "current_model": current,
    }


@app.post("/api/image/load")
async def load_image_model(
    model_id: str = Query(default="realvisxl-v4", description="Model ID to load"),
):
    """
    Load an image generation model.

    Available models: realvisxl-v4, dreamshaper-xl, juggernaut-xl, playground-v2.5, sdxl-turbo

    IMPORTANT: Call /api/unload first to free GPU memory!
    """
    try:
        models = image_gen.get_available_models()
        if model_id not in models:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Unknown model: {model_id}",
                    "available_models": list(models.keys()),
                },
            )

        current = image_gen.get_current_model()
        if current == model_id and image_gen.is_loaded():
            return JSONResponse(content={
                "success": True,
                "message": f"{models[model_id]['name']} already loaded",
                "model_id": model_id,
                "memory": image_gen.get_memory_usage(),
            })

        image_gen.load_model(model_id)

        return JSONResponse(content={
            "success": True,
            "message": f"{models[model_id]['name']} loaded successfully",
            "model_id": model_id,
            "model_config": {
                "default_steps": models[model_id]["default_steps"],
                "default_guidance": models[model_id]["default_guidance"],
                "supports_negative": models[model_id]["supports_negative"],
            },
            "memory": image_gen.get_memory_usage(),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/image/unload")
async def unload_image_model():
    """Unload current image model to free GPU memory."""
    try:
        current = image_gen.get_current_model()
        unloaded = image_gen.unload_model()
        return JSONResponse(content={
            "success": True,
            "unloaded": unloaded,
            "message": f"Model unloaded" if unloaded else "No model was loaded",
            "previous_model": current,
            "memory": image_gen.get_memory_usage(),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/image/generate")
async def generate_image(
    prompt: str = Query(..., description="Text description of the image to generate"),
    negative_prompt: str = Query(default="low quality, blurry, distorted, deformed, ugly, bad anatomy", description="What to avoid in the image"),
    width: int = Query(default=1024, ge=512, le=1024, description="Image width"),
    height: int = Query(default=1024, ge=512, le=1024, description="Image height"),
    steps: int = Query(default=25, ge=10, le=50, description="Number of inference steps (20-30 recommended)"),
    seed: Optional[int] = Query(default=None, description="Random seed for reproducibility"),
):
    """
    Generate an image from a text prompt using RealVisXL V4.

    Returns base64 encoded PNG image.
    """
    try:
        # Auto-load model if not loaded
        if not image_gen.is_loaded():
            # Check if Whisper is still loaded
            if transcriber is not None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "Please unload Whisper first using /api/unload to free GPU memory",
                    },
                )
            image_gen.load_model()

        result = image_gen.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=seed,
        )

        return JSONResponse(content={
            "success": True,
            "image_base64": result["image_base64"],
            "width": result["width"],
            "height": result["height"],
            "prompt": result["prompt"],
            "seed": result["seed"],
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/image/status")
async def image_model_status():
    """Check current image model status."""
    current = image_gen.get_current_model()
    models = image_gen.get_available_models()
    model_config = models.get(current, {}) if current else {}
    return {
        "loaded": image_gen.is_loaded(),
        "model_id": current,
        "model_name": model_config.get("name") if current else None,
        "model_config": {
            "default_steps": model_config.get("default_steps"),
            "default_guidance": model_config.get("default_guidance"),
            "supports_negative": model_config.get("supports_negative"),
        } if current else None,
        "memory": image_gen.get_memory_usage(),
    }


# ============== Video Generation Endpoints ==============

@app.get("/video")
async def video_page(username: str = Depends(verify_credentials)):
    """Serve the video generation page (protected)."""
    return FileResponse(STATIC_DIR / "video.html")


@app.get("/api/video/models")
async def list_video_models():
    """List available video generation models."""
    models = video_gen.get_available_models()
    current = video_gen.get_current_model()
    return {
        "models": [
            {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "default_steps": config["default_steps"],
                "default_guidance": config["default_guidance"],
                "default_frames": config["default_frames"],
                "default_fps": config["default_fps"],
                "supports_negative": config["supports_negative"],
            }
            for model_id, config in models.items()
        ],
        "current_model": current,
    }


@app.post("/api/video/load")
async def load_video_model(
    model_id: str = Query(default="wan2.1-t2v-1.3b", description="Model ID to load"),
):
    """Load a video generation model."""
    try:
        video_gen.load_model(model_id)
        model_config = video_gen.get_available_models().get(model_id, {})
        return {
            "success": True,
            "message": f"{model_config.get('name', model_id)} loaded successfully",
            "model_id": model_id,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/video/unload")
async def unload_video_model():
    """Unload current video model to free GPU memory."""
    try:
        video_gen.unload_model()
        return {
            "success": True,
            "message": "Video model unloaded and GPU memory cleared",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/video/generate")
async def generate_video(
    prompt: str = Query(..., description="Text description of the video to generate"),
    negative_prompt: str = Query(default="low quality, blurry, distorted", description="What to avoid in the video"),
    num_frames: int = Query(default=33, ge=9, le=81, description="Number of frames (33 = ~2 sec)"),
    height: int = Query(default=480, description="Video height"),
    width: int = Query(default=848, description="Video width"),
    steps: int = Query(default=25, ge=1, le=50, description="Number of inference steps"),
    guidance: float = Query(default=5.0, ge=0, le=20, description="Guidance scale"),
    seed: Optional[int] = Query(default=None, description="Random seed for reproducibility"),
):
    """Generate a video from text prompt."""
    try:
        if not video_gen.is_loaded():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Video model not loaded. Please load a model first."},
            )

        output_path = video_gen.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=Path(output_path).name,
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/video/status")
async def video_model_status():
    """Check current video model status."""
    current = video_gen.get_current_model()
    models = video_gen.get_available_models()
    model_config = models.get(current, {}) if current else {}
    return {
        "loaded": video_gen.is_loaded(),
        "model_id": current,
        "model_name": model_config.get("name") if current else None,
        "model_config": {
            "default_steps": model_config.get("default_steps"),
            "default_guidance": model_config.get("default_guidance"),
            "default_frames": model_config.get("default_frames"),
            "supports_negative": model_config.get("supports_negative"),
        } if current else None,
        "memory": video_gen.get_memory_usage(),
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
