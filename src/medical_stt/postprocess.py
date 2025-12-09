#!/usr/bin/env python3
"""Post-processing module using LLM to fix transcription errors."""

import os
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TranscriptionPostProcessor:
    """Post-process transcriptions using LLM to fix errors."""

    def __init__(
        self,
        provider: str = "ollama",  # "ollama", "openai", "anthropic"
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url

        # Default models per provider
        default_models = {
            "ollama": "llama3.2:3b",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
        }
        self.model = model or default_models.get(provider, "llama3.2:3b")

        self._client = None

    def _get_client(self):
        """Get or initialize the LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "ollama":
            try:
                import ollama
                self._client = ollama
            except ImportError:
                raise ImportError("Please install ollama: pip install ollama")

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")

        return self._client

    def fix_transcription(
        self,
        text: str,
        language: str = "id",
        context: str = "medical",
    ) -> str:
        """
        Fix transcription errors using LLM.

        Args:
            text: Raw transcription text
            language: Language code (id, en, etc.)
            context: Context for better corrections (medical, general)

        Returns:
            Corrected transcription
        """
        if not text.strip():
            return text

        # Build prompt based on language and context
        if language == "id":
            system_prompt = """Anda adalah asisten yang memperbaiki hasil transkripsi audio medis dalam Bahasa Indonesia.

Tugas Anda:
1. Perbaiki kesalahan ejaan dan tata bahasa
2. Perbaiki istilah medis yang salah transkripsi
3. Tambahkan tanda baca yang tepat
4. Pertahankan makna asli, jangan tambah atau kurangi informasi
5. Format paragraf dengan baik

Balas HANYA dengan teks yang sudah diperbaiki, tanpa penjelasan."""

        else:
            system_prompt = """You are an assistant that fixes medical audio transcription errors.

Your tasks:
1. Fix spelling and grammar errors
2. Correct medical terminology that was mistranscribed
3. Add proper punctuation
4. Preserve the original meaning, don't add or remove information
5. Format paragraphs properly

Reply ONLY with the corrected text, no explanations."""

        user_prompt = f"Fix this transcription:\n\n{text}"

        try:
            client = self._get_client()

            if self.provider == "ollama":
                response = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response["message"]["content"].strip()

            elif self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"LLM post-processing failed: {e}")
            return text  # Return original if LLM fails

        return text


# Singleton instance for easy access
_processor: Optional[TranscriptionPostProcessor] = None


def get_postprocessor(
    provider: str = "ollama",
    model: Optional[str] = None,
) -> TranscriptionPostProcessor:
    """Get or create the post-processor instance."""
    global _processor
    if _processor is None or _processor.provider != provider:
        _processor = TranscriptionPostProcessor(provider=provider, model=model)
    return _processor


def fix_transcription(
    text: str,
    language: str = "id",
    provider: str = "ollama",
    model: Optional[str] = None,
) -> str:
    """
    Convenience function to fix transcription.

    Args:
        text: Raw transcription
        language: Language code
        provider: LLM provider (ollama, openai, anthropic)
        model: Model name (optional)

    Returns:
        Fixed transcription
    """
    processor = get_postprocessor(provider=provider, model=model)
    return processor.fix_transcription(text, language=language)
