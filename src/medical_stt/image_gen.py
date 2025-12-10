#!/usr/bin/env python3
"""Image generation module with multiple model support."""

import os
import gc
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Available models configuration
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "realvisxl-v4": {
        "name": "RealVisXL V4",
        "repo": "SG161222/RealVisXL_V4.0",
        "description": "Photorealistic images, high quality",
        "default_steps": 25,
        "default_guidance": 7.0,
        "supports_negative": True,
    },
    "dreamshaper-xl": {
        "name": "DreamShaper XL",
        "repo": "Lykon/dreamshaper-xl-1-0",
        "description": "Artistic and creative style",
        "default_steps": 25,
        "default_guidance": 7.5,
        "supports_negative": True,
    },
    "juggernaut-xl": {
        "name": "Juggernaut XL",
        "repo": "RunDiffusion/Juggernaut-XL-v9",
        "description": "High detail, versatile",
        "default_steps": 30,
        "default_guidance": 7.0,
        "supports_negative": True,
    },
    "playground-v2.5": {
        "name": "Playground v2.5",
        "repo": "playgroundai/playground-v2.5-1024px-aesthetic",
        "description": "Best overall quality, aesthetic focus",
        "default_steps": 30,
        "default_guidance": 3.0,
        "supports_negative": True,
    },
    "sdxl-turbo": {
        "name": "SDXL Turbo",
        "repo": "stabilityai/sdxl-turbo",
        "description": "Fast generation (1-4 steps)",
        "default_steps": 4,
        "default_guidance": 0.0,
        "supports_negative": False,
    },
    "z-image-turbo": {
        "name": "Z-Image Turbo",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "description": "Fast photorealistic, supports Chinese/English text",
        "default_steps": 8,
        "default_guidance": 3.5,
        "supports_negative": True,
        "pipeline_class": "ZiPiPipeline",
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
    """Get list of available models."""
    return AVAILABLE_MODELS


def get_current_model() -> Optional[str]:
    """Get currently loaded model ID."""
    return _current_model_id


def load_model(model_id: str = "realvisxl-v4", device: Optional[str] = None):
    """
    Load an image generation model.

    Args:
        model_id: Model ID from AVAILABLE_MODELS
        device: Device to load on (auto-detected if None)

    Note: Call unload_model() first if switching models!
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

    # Check imports first
    try:
        import torch
        from diffusers import DiffusionPipeline, AutoPipelineForText2Image
    except ImportError as e:
        raise ImportError(
            "Please install diffusers: pip install diffusers transformers accelerate"
        ) from e

    # Use float16 for GPU
    torch_dtype = torch.float16 if _device == "cuda" else torch.float32

    logger.info(f"Using dtype: {torch_dtype}")

    # Load model
    try:
        if model_id == "sdxl-turbo":
            # SDXL Turbo uses AutoPipeline
            _pipe = AutoPipelineForText2Image.from_pretrained(
                model_config["repo"],
                torch_dtype=torch_dtype,
                variant="fp16" if _device == "cuda" else None,
            )
        elif model_id == "z-image-turbo":
            # Z-Image-Turbo uses ZiPiPipeline
            from diffusers import ZiPiPipeline
            _pipe = ZiPiPipeline.from_pretrained(
                model_config["repo"],
                torch_dtype=torch_dtype,
            )
        else:
            # Standard diffusion pipeline for others
            _pipe = DiffusionPipeline.from_pretrained(
                model_config["repo"],
                torch_dtype=torch_dtype,
                variant="fp16" if _device == "cuda" else None,
                use_safetensors=True,
            )
    except Exception as e:
        # Try without variant if fp16 not available
        logger.warning(f"Failed with fp16 variant, trying without: {e}")
        if model_id == "sdxl-turbo":
            _pipe = AutoPipelineForText2Image.from_pretrained(
                model_config["repo"],
                torch_dtype=torch_dtype,
            )
        elif model_id == "z-image-turbo":
            from diffusers import ZiPiPipeline
            _pipe = ZiPiPipeline.from_pretrained(
                model_config["repo"],
                torch_dtype=torch_dtype,
            )
        else:
            _pipe = DiffusionPipeline.from_pretrained(
                model_config["repo"],
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )

    _pipe = _pipe.to(_device)

    # Enable memory optimizations
    if _device == "cuda":
        _pipe.enable_attention_slicing()
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    _current_model_id = model_id
    logger.info(f"{model_config['name']} loaded successfully")
    return _pipe


def unload_model():
    """Unload current model to free GPU memory."""
    global _pipe, _device, _current_model_id

    if _pipe is not None:
        model_name = AVAILABLE_MODELS.get(_current_model_id, {}).get("name", "Unknown")
        del _pipe
        _pipe = None
        _device = None
        _current_model_id = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        logger.info(f"{model_name} model unloaded")
        return True

    return False


def is_loaded() -> bool:
    """Check if any model is loaded."""
    return _pipe is not None


def generate_image(
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, deformed, ugly, bad anatomy",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        negative_prompt: What to avoid in the image
        width: Image width (default 1024)
        height: Image height (default 1024)
        num_inference_steps: Number of denoising steps (uses model default if None)
        guidance_scale: Guidance scale (uses model default if None)
        seed: Random seed for reproducibility
        output_path: Optional path to save the image

    Returns:
        dict with image data and metadata
    """
    global _pipe, _current_model_id

    if _pipe is None:
        raise RuntimeError("No model loaded. Call load_model() first.")

    import torch
    from io import BytesIO
    import base64

    # Get model config for defaults
    model_config = AVAILABLE_MODELS.get(_current_model_id, {})

    # Use model defaults if not specified
    if num_inference_steps is None:
        num_inference_steps = model_config.get("default_steps", 25)
    if guidance_scale is None:
        guidance_scale = model_config.get("default_guidance", 7.0)

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    logger.info(f"Generating image with {_current_model_id}: {prompt[:50]}...")

    # Build generation kwargs
    gen_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    # Only add negative_prompt if model supports it
    if model_config.get("supports_negative", True) and negative_prompt:
        gen_kwargs["negative_prompt"] = negative_prompt

    # Generate image
    result = _pipe(**gen_kwargs)

    image = result.images[0]

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")

    # Convert to base64 for API response
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": image_base64,
        "width": width,
        "height": height,
        "prompt": prompt,
        "seed": seed,
        "output_path": output_path,
        "model_id": _current_model_id,
        "model_name": model_config.get("name", "Unknown"),
        "steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }


def get_memory_usage() -> dict:
    """Get current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2),
            }
    except:
        pass
    return {}
