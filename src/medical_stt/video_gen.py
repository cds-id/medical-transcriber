#!/usr/bin/env python3
"""Video generation module using Wan2.1-T2V-1.3B."""

import os
import gc
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Available video models configuration
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "wan2.1-t2v-1.3b": {
        "name": "Wan2.1 T2V 1.3B",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "description": "Text-to-video, 480P, ~8GB VRAM",
        "default_steps": 25,
        "default_guidance": 5.0,
        "default_frames": 33,  # ~2 seconds at 16fps
        "default_fps": 16,
        "supports_negative": True,
    },
}

# Global model instance
_pipe = None
_device = None
_current_model_id = None


def get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available video models."""
    return AVAILABLE_MODELS


def get_current_model() -> Optional[str]:
    """Get currently loaded model ID."""
    return _current_model_id


def is_loaded() -> bool:
    """Check if a model is currently loaded."""
    return _pipe is not None


def get_memory_usage() -> Dict[str, Any]:
    """Get current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            }
    except Exception:
        pass
    return {"allocated_mb": 0, "reserved_mb": 0}


def load_model(model_id: str = "wan2.1-t2v-1.3b", device: Optional[str] = None):
    """
    Load a video generation model.

    Args:
        model_id: Model ID from AVAILABLE_MODELS
        device: Device to load on (auto-detected if None)
    """
    global _pipe, _device, _current_model_id

    # Check if same model already loaded
    if _pipe is not None and _current_model_id == model_id:
        logger.info(f"{AVAILABLE_MODELS[model_id]['name']} already loaded")
        return _pipe

    # Unload current model if different
    if _pipe is not None:
        logger.info(f"Unloading {_current_model_id} to load {model_id}")
        unload_model()

    # Validate model_id
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")

    model_config = AVAILABLE_MODELS[model_id]
    _device = device or get_device()

    logger.info(f"Loading {model_config['name']} on {_device}...")

    try:
        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as e:
        raise ImportError(
            "Please install diffusers from source: pip install git+https://github.com/huggingface/diffusers"
        ) from e

    # Use float16 for GPU (bfloat16 not supported on T4)
    torch_dtype = torch.float16 if _device == "cuda" else torch.float32

    logger.info(f"Using dtype: {torch_dtype}")

    # Load model
    _pipe = WanPipeline.from_pretrained(
        model_config["repo"],
        torch_dtype=torch_dtype,
    )

    # Enable CPU offload for T4 (16GB) - keeps model on CPU, moves to GPU as needed
    if _device == "cuda":
        logger.info("Enabling sequential CPU offload for memory optimization...")
        _pipe.enable_sequential_cpu_offload()
        # Also enable memory efficient attention
        try:
            _pipe.enable_attention_slicing("auto")
        except Exception:
            pass
        try:
            _pipe.enable_vae_slicing()
        except Exception:
            pass
    else:
        _pipe = _pipe.to(_device)

    _current_model_id = model_id
    logger.info(f"{model_config['name']} loaded successfully")

    return _pipe


def unload_model():
    """Unload the current model to free GPU memory."""
    global _pipe, _current_model_id

    if _pipe is not None:
        logger.info(f"Unloading video model {_current_model_id}...")
        del _pipe
        _pipe = None
        _current_model_id = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        logger.info("Video model unloaded and GPU memory cleared")
    else:
        logger.info("No video model loaded")


def generate_video(
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted",
    num_frames: int = 33,
    height: int = 480,
    width: int = 848,
    num_inference_steps: int = 25,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a video from text prompt.

    Args:
        prompt: Text description of the video to generate
        negative_prompt: What to avoid in the video
        num_frames: Number of frames (33 = ~2 sec at 16fps)
        height: Video height (480 recommended for 1.3B)
        width: Video width (848 for 16:9 aspect at 480p)
        num_inference_steps: Denoising steps
        guidance_scale: How closely to follow prompt
        seed: Random seed for reproducibility
        output_path: Path to save video (auto-generated if None)

    Returns:
        Path to generated video file
    """
    global _pipe

    if _pipe is None:
        raise RuntimeError("Video model not loaded. Call load_model() first.")

    import torch
    from diffusers.utils import export_to_video

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    logger.info(f"Generating video: {prompt[:50]}...")
    logger.info(f"Settings: {num_frames} frames, {width}x{height}, {num_inference_steps} steps")

    model_config = AVAILABLE_MODELS.get(_current_model_id, {})

    # Generate video
    output = _pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if model_config.get("supports_negative", True) else None,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Get frames from output
    frames = output.frames[0]

    # Generate output path if not provided
    if output_path is None:
        import uuid
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"video_{uuid.uuid4().hex[:8]}.mp4")

    # Export to video file
    export_to_video(frames, output_path, fps=16)

    logger.info(f"Video saved to: {output_path}")
    return output_path
