"""Project-specific exceptions."""

from __future__ import annotations


class TaigiASRError(Exception):
    """Base class for all taigi_asr errors."""


class InsufficientVRAMError(TaigiASRError):
    """Raised when the requested engine cannot fit into the available VRAM."""


class ModelLoadError(TaigiASRError):
    """Raised when a model fails to download / deserialize / push to device."""


class TranscriptionError(TaigiASRError):
    """Raised when the inference pipeline itself fails mid-transcribe."""
