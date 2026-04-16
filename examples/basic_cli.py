"""Simplest possible usage: one file, default settings.

Run:
    python examples/basic_cli.py path/to/audio.m4a
"""

from __future__ import annotations

import sys
from pathlib import Path

from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.formatters import to_srt
from taigi_asr.router import EngineRouter, GPUProfiler


def main(audio_path: str) -> None:
    src = Path(audio_path)
    if not src.exists():
        print(f"Audio not found: {src}")
        sys.exit(1)

    info = GPUProfiler.detect()
    spec = EngineRouter.select(info)
    print(f"GPU: {info.name} ({info.vram_gb:.1f} GB)")
    print(f"Engine: {spec.kind.value} / {spec.compute_type} / batch={spec.batch_size}")

    wav, duration = AudioConverter.convert(src)
    print(f"Audio: {duration:.1f}s")

    engine = build_engine(spec)
    try:
        engine.load()
        segments = engine.transcribe(wav)
    finally:
        AudioConverter.cleanup(wav)
        engine.unload()

    out = src.with_suffix(".srt")
    out.write_text(to_srt(segments), encoding="utf-8")
    print(f"Saved: {out}  ({len(segments)} segments)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/basic_cli.py <audio>")
        sys.exit(2)
    main(sys.argv[1])
