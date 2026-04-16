"""Command-line entry point: ``taigi-asr <audio> [options]``.

Uses only stdlib argparse to keep the dependency tree flat.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from taigi_asr import __version__
from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.errors import InsufficientVRAMError, TaigiASRError
from taigi_asr.formatters import to_json, to_srt, to_txt, to_vtt
from taigi_asr.router import EngineKind, EngineRouter, GPUProfiler


def _parse_engine(raw: str) -> EngineKind | None:
    mapping = {
        "auto": None,
        "fw": EngineKind.FASTER_WHISPER,
        "faster-whisper": EngineKind.FASTER_WHISPER,
        "hf": EngineKind.HUGGINGFACE,
        "huggingface": EngineKind.HUGGINGFACE,
    }
    if raw not in mapping:
        raise argparse.ArgumentTypeError(f"Unknown engine: {raw}")
    return mapping[raw]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taigi-asr",
        description="Taiwanese Hokkien ASR powered by MediaTek Breeze-ASR-26.",
    )
    parser.add_argument("audio", type=Path, help="Path to audio/video file")
    parser.add_argument(
        "--engine",
        default="auto",
        type=_parse_engine,
        help="auto | fw (faster-whisper) | hf (huggingface)",
    )
    parser.add_argument(
        "--format",
        default="srt",
        help="Output format(s). Single: srt|txt|vtt|json. Multiple: "
             "comma-separated (e.g. 'srt,txt,json') - each written alongside input.",
    )
    parser.add_argument("--out", type=Path, help="Output path (single format only)")
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Emit word-level timestamps (slower)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam search width (default 5; RTX 3050 4GB safe ceiling: 12)",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Number of candidates for temperature fallback sampling (default 5)",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"taigi-asr {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        return 2

    info = GPUProfiler.detect()
    try:
        spec = EngineRouter.select(info, prefer=args.engine)
    except InsufficientVRAMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(
        f"Device: {info.name} | {info.vram_gb:.1f} GB | "
        f"Engine: {spec.kind.value} ({spec.compute_type}, batch={spec.batch_size})",
        file=sys.stderr,
    )

    wav_path = None
    engine = None
    try:
        wav_path, duration = AudioConverter.convert(args.audio)
        print(f"Audio duration: {duration:.1f} s", file=sys.stderr)

        engine = build_engine(spec)
        engine.load()
        extra_kwargs: dict = {"word_timestamps": args.word_timestamps}
        if args.beam_size is not None:
            extra_kwargs["beam_size"] = args.beam_size
        if args.best_of is not None:
            extra_kwargs["best_of"] = args.best_of
        segments = engine.transcribe(wav_path, **extra_kwargs)
    except TaigiASRError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    finally:
        if wav_path is not None:
            AudioConverter.cleanup(wav_path)
        if engine is not None:
            try:
                engine.unload()
            except Exception:  # pragma: no cover
                pass

    if not segments:
        print("WARNING: empty transcript", file=sys.stderr)
        return 5

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    valid = {"srt", "txt", "vtt", "json"}
    bad = [f for f in formats if f not in valid]
    if bad:
        print(f"ERROR: unknown format(s): {bad}. Choose from {sorted(valid)}", file=sys.stderr)
        return 6

    meta = {
        "engine": spec.kind.value,
        "compute_type": spec.compute_type,
        "duration_sec": round(duration, 2),
    }

    if len(formats) == 1:
        out_path = args.out or args.audio.with_suffix(f".{formats[0]}")
        out_path.write_text(_render(segments, formats[0], meta), encoding="utf-8")
        print(f"[OK] Saved: {out_path}", file=sys.stderr)
    else:
        if args.out is not None:
            print("WARNING: --out ignored when multiple formats requested", file=sys.stderr)
        for fmt in formats:
            out_path = args.audio.with_suffix(f".{fmt}")
            out_path.write_text(_render(segments, fmt, meta), encoding="utf-8")
            print(f"[OK] Saved: {out_path}", file=sys.stderr)
    return 0


def _render(segments, fmt: str, meta: dict) -> str:
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "txt":
        return to_txt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "json":
        return to_json(segments, meta=meta)
    raise ValueError(fmt)


if __name__ == "__main__":
    sys.exit(main())
