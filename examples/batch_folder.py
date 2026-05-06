"""Batch-transcribe every audio file in a folder, reusing one loaded engine.

For typical use, prefer the CLI: ``taigi-asr --input-dir /path/to/folder``
does the same thing with auto-routing, multi-format output, and per-file
xRT reporting. This script stays as a programmatic-API reference: it
illustrates the load-once / transcribe-many pattern that any custom
application built on top of taigi_asr should follow.

Run:
    python examples/batch_folder.py /path/to/folder
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.formatters import to_srt
from taigi_asr.router import EngineRouter, GPUProfiler

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".mov", ".mkv", ".flac", ".ogg", ".webm"}


def main(folder: str) -> None:
    root = Path(folder)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio files in {root}")
        sys.exit(1)

    info = GPUProfiler.detect()
    spec = EngineRouter.select(info)
    engine = build_engine(spec)
    engine.load()
    print(f"Engine loaded: {spec.kind.value} / {spec.compute_type} / batch={spec.batch_size}")

    try:
        for i, audio in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {audio.name} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            wav, duration = AudioConverter.convert(audio)
            try:
                segments = engine.transcribe(wav)
            finally:
                AudioConverter.cleanup(wav)
            elapsed = time.perf_counter() - t0

            out = audio.with_suffix(".srt")
            out.write_text(to_srt(segments), encoding="utf-8")
            xrt = duration / elapsed if elapsed else 0
            print(f"{duration:.1f}s audio -> {elapsed:.1f}s ({xrt:.2f}x) -> {out.name}")
    finally:
        engine.unload()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/batch_folder.py <folder>")
        sys.exit(2)
    main(sys.argv[1])
