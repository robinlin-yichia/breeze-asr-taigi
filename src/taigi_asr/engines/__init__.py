"""ASR engine implementations."""

from __future__ import annotations

from taigi_asr.engines.base import ASREngine
from taigi_asr.engines.fake import FakeEngine
from taigi_asr.errors import InsufficientVRAMError, ModelLoadError
from taigi_asr.router import EngineKind, EngineSpec


def build_engine(spec: EngineSpec) -> ASREngine:
    """Instantiate the engine specified by the router without loading it yet."""
    if spec.kind is EngineKind.FASTER_WHISPER:
        from taigi_asr.engines.faster_whisper import FasterWhisperEngine

        return FasterWhisperEngine(
            device=spec.device,
            compute_type=spec.compute_type,
            batch_size=spec.batch_size,
        )
    if spec.kind is EngineKind.HUGGINGFACE:
        from taigi_asr.engines.huggingface import HuggingFaceEngine

        return HuggingFaceEngine(
            device=spec.device,
            compute_type=spec.compute_type,
            batch_size=spec.batch_size,
        )
    raise ValueError(f"Unknown engine kind: {spec.kind}")


__all__ = [
    "ASREngine",
    "FakeEngine",
    "InsufficientVRAMError",
    "ModelLoadError",
    "build_engine",
]
