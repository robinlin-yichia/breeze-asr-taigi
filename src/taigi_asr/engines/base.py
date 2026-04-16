"""Engine contract shared by every inference backend.

All engines — HuggingFace pipeline, faster-whisper, the test fake — implement the
same structural Protocol so callers (UI, CLI, batch scripts) stay backend-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from taigi_asr.segments import TimestampedSegment


@runtime_checkable
class ASREngine(Protocol):
    """Any ASR backend recognized by the router."""

    def load(self) -> None:
        """Download (if missing) and warm the model. Idempotent."""
        ...

    def is_loaded(self) -> bool:
        """True once :meth:`load` has finished successfully."""
        ...

    def transcribe(
        self,
        wav_path: str | Path,
        *,
        word_timestamps: bool = False,
    ) -> list[TimestampedSegment]:
        """Run ASR on a normalized 16 kHz mono WAV and return ordered segments."""
        ...

    def unload(self) -> None:
        """Release GPU / CPU resources. Safe to call on an already-unloaded engine."""
        ...
