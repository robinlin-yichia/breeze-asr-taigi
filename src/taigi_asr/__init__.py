"""Taiwanese Hokkien ASR — powered by MediaTek Breeze-ASR-26, tuned for low-VRAM GPUs."""

__version__ = "0.1.0"

from taigi_asr.segments import TimestampedSegment

__all__ = ["TimestampedSegment", "__version__"]
