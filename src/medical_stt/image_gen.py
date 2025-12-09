#!/usr/bin/env python3
"""Image generation module using FLUX.1 Schnell."""

import os
import gc
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model instance
_pipe = None
_device = None


def get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_model(device: Optional[str] = None):
    """
    Load FLUX.1 Schnell model.

    Note: Call /api/unload first to free GPU memory!
    """
    global _pipe, _device

    if _pipe is not None:
        logger.info("FLUX model already loaded")
        return _pipe

    _device = device or get_device()

    logger.info(f"Loading FLUX.1 Schnell on {_device}...")

    try:
        import torch
        from diffusers import FluxPipeline

        # Load with optimizations for T4
        _pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
        )

        _pipe = _pipe.to(_device)

        # Enable memory optimizations
        if _device == "cuda":
            _pipe.enable_attention_slicing()
            # Try to enable xformers if available
            try:
                _pipe.enable_xformers_memory_efficient_attention()
            except:
                pass

        logger.info("FLUX.1 Schnell loaded successfully")
        return _pipe

    except ImportError as e:
        raise ImportError(
            "Please install diffusers: pip install diffusers transformers accelerate"
        ) from e


def unload_model():
    """Unload FLUX model to free GPU memory."""
    global _pipe, _device

    if _pipe is not None:
        del _pipe
        _pipe = None
        _device = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        logger.info("FLUX model unloaded")
        return True

    return False


def is_loaded() -> bool:
    """Check if FLUX model is loaded."""
    return _pipe is not None


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,  # FLUX Schnell is fast, 4 steps is good
    guidance_scale: float = 0.0,  # Schnell doesn't use guidance
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        width: Image width (default 1024)
        height: Image height (default 1024)
        num_inference_steps: Number of denoising steps (default 4 for Schnell)
        guidance_scale: Guidance scale (0.0 for Schnell)
        seed: Random seed for reproducibility
        output_path: Optional path to save the image

    Returns:
        dict with image data and metadata
    """
    global _pipe

    if _pipe is None:
        raise RuntimeError("FLUX model not loaded. Call load_model() first.")

    import torch
    from io import BytesIO
    import base64

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)

    logger.info(f"Generating image: {prompt[:50]}...")

    # Generate image
    result = _pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

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
