"""In-memory test double that satisfies :class:`taigi_asr.engines.base.ASREngine`."""

from __future__ import annotations

from pathlib import Path

from taigi_asr.segments import TimestampedSegment


class FakeEngine:
    """Replay a fixed transcript script — no GPU, no network, no model.

    Used by unit tests and by UI smoke tests that exercise routing / formatting
    without spending seconds loading Whisper.
    """

    def __init__(self, script: list[TimestampedSegment]) -> None:
        self._script = list(script)
        self._loaded = False
        self.call_count = 0

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(
        self,
        wav_path: str | Path,
        *,
        word_timestamps: bool = False,
        **_engine_kwargs: object,
    ) -> list[TimestampedSegment]:
        # Concrete engines take extra knobs beyond the Protocol (beam_size,
        # hotwords, ...). Swallow them so tests substituting this double for a
        # FASTER_WHISPER spec don't explode on kwargs the caller legitimately
        # passes to the real engine.
        if not self._loaded:
            raise RuntimeError("FakeEngine.transcribe called before load().")
        self.call_count += 1
        return list(self._script)

    def unload(self) -> None:
        self._loaded = False
