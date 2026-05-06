"""Command-line entry point: ``taigi-asr <audio> [<audio> ...] [options]``.

Uses only stdlib argparse to keep the dependency tree flat.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from taigi_asr import __version__
from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.errors import InsufficientVRAMError, TaigiASRError
from taigi_asr.formatters import to_json, to_srt, to_txt, to_vtt
from taigi_asr.router import EngineKind, EngineRouter, GPUProfiler

# Extensions ffmpeg can decode that we'll auto-pick from --input-dir.
# Limited to common audio/video containers; users can still pass arbitrary
# files explicitly as positional args.
_AUDIO_EXTS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".webm",
    ".mp4",
    ".mkv",
    ".aac",
    ".opus",
    ".wma",
}


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
    # nargs="*" so users can rely solely on --input-dir without supplying
    # positional paths. Validation of "at least one resolved input" happens
    # in main() after the directory glob runs.
    parser.add_argument(
        "audio",
        type=Path,
        nargs="*",
        help="One or more audio/video files. Combine with --input-dir to add a directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Add every supported audio/video file in this directory to the batch "
            f"(extensions: {', '.join(sorted(_AUDIO_EXTS))}). Non-recursive."
        ),
    )
    # TAIGI_ASR_DEFAULT_ENGINE lets the user flip the default engine per
    # machine without changing the router (e.g. set to "hf" on an Optimus
    # laptop where the 4 GB fp16 path is preferred over int8_float16).
    env_default = os.environ.get("TAIGI_ASR_DEFAULT_ENGINE", "auto").strip() or "auto"
    parser.add_argument(
        "--engine",
        default=env_default,
        type=_parse_engine,
        help="auto | fw (faster-whisper) | hf (huggingface). "
        "Default can be overridden via TAIGI_ASR_DEFAULT_ENGINE env var.",
    )
    parser.add_argument(
        "--format",
        default="srt",
        help="Output format(s). Single: srt|txt|vtt|json. Multiple: "
        "comma-separated (e.g. 'srt,txt,json') - each written alongside input.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path. Honored only when one input + one format are requested; "
        "otherwise outputs land alongside each input file.",
    )
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


def _resolve_inputs(positional: list[Path], input_dir: Path | None) -> list[Path]:
    """Merge positional file args with --input-dir glob.

    Order: positional first (preserved), then directory entries in sorted
    order. Duplicates collapsed by resolved absolute path so the same file
    isn't transcribed twice when both modes hit it.
    """
    resolved: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            key = p.resolve()
        except OSError:
            key = p.absolute()
        if key in seen:
            return
        seen.add(key)
        resolved.append(p)

    for p in positional:
        _add(p)

    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"--input-dir not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")
        for child in sorted(input_dir.iterdir()):
            if child.is_file() and child.suffix.lower() in _AUDIO_EXTS:
                _add(child)

    return resolved


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


def _transcribe_one(
    engine,
    spec,
    audio: Path,
    formats: list[str],
    out_override: Path | None,
    *,
    word_timestamps: bool,
    beam_size: int | None,
    best_of: int | None,
    write_alongside_only: bool,
) -> tuple[bool, float, float]:
    """Run convert + transcribe + render for one input.

    Returns ``(ok, duration_sec, elapsed_sec)``. Caller decides how to
    aggregate failures across the batch — we only print and return False.
    """
    t0 = time.monotonic()
    wav_path: Path | None = None
    try:
        wav_path, duration = AudioConverter.convert(audio)
        print(f"Audio duration: {duration:.1f} s", file=sys.stderr)

        extra: dict = {"word_timestamps": word_timestamps}
        if spec.kind is EngineKind.FASTER_WHISPER:
            if beam_size is not None:
                extra["beam_size"] = beam_size
            if best_of is not None:
                extra["best_of"] = best_of
        elif beam_size is not None or best_of is not None:
            print(
                "WARNING: --beam-size / --best-of are ignored on the HuggingFace engine.",
                file=sys.stderr,
            )

        segments = engine.transcribe(wav_path, **extra)
    except TaigiASRError as exc:
        print(f"ERROR [{audio.name}]: {exc}", file=sys.stderr)
        return False, 0.0, time.monotonic() - t0
    finally:
        if wav_path is not None:
            AudioConverter.cleanup(wav_path)

    if not segments:
        print(f"WARNING [{audio.name}]: empty transcript", file=sys.stderr)
        return False, duration, time.monotonic() - t0

    meta = {
        "engine": spec.kind.value,
        "compute_type": spec.compute_type,
        "duration_sec": round(duration, 2),
    }

    if (not write_alongside_only) and out_override is not None and len(formats) == 1:
        out_path = out_override
        out_path.write_text(_render(segments, formats[0], meta), encoding="utf-8")
        print(f"[OK] Saved: {out_path}", file=sys.stderr)
    else:
        for fmt in formats:
            out_path = audio.with_suffix(f".{fmt}")
            out_path.write_text(_render(segments, fmt, meta), encoding="utf-8")
            print(f"[OK] Saved: {out_path}", file=sys.stderr)

    return True, duration, time.monotonic() - t0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        inputs = _resolve_inputs(list(args.audio), args.input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print(
            "ERROR: no audio inputs provided (pass paths positionally or use --input-dir).",
            file=sys.stderr,
        )
        return 2

    missing = [p for p in inputs if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: audio not found: {p}", file=sys.stderr)
        inputs = [p for p in inputs if p.exists()]
        if not inputs:
            return 2

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    valid = {"srt", "txt", "vtt", "json"}
    bad = [f for f in formats if f not in valid]
    if bad:
        print(f"ERROR: unknown format(s): {bad}. Choose from {sorted(valid)}", file=sys.stderr)
        return 6

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

    write_alongside_only = len(inputs) > 1 or len(formats) > 1
    if args.out is not None and write_alongside_only:
        print(
            "WARNING: --out ignored when multiple inputs or formats are requested; "
            "outputs will be written alongside each input.",
            file=sys.stderr,
        )

    engine = build_engine(spec)
    try:
        engine.load()
    except InsufficientVRAMError as exc:
        # Auto-downgrade only when the user asked for `auto` AND the
        # original choice wasn't already Faster-Whisper.
        if args.engine is not None or spec.kind == EngineKind.FASTER_WHISPER:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 4
        print(f"WARNING: {exc}. Retrying with faster_whisper.", file=sys.stderr)
        try:
            engine.unload()
        except Exception:  # pragma: no cover
            pass
        spec = EngineRouter.select(info, prefer=EngineKind.FASTER_WHISPER)
        print(
            f"Device: {info.name} | {info.vram_gb:.1f} GB | "
            f"Engine: {spec.kind.value} ({spec.compute_type}, batch={spec.batch_size})",
            file=sys.stderr,
        )
        engine = build_engine(spec)
        try:
            engine.load()
        except TaigiASRError as exc2:
            print(f"ERROR: {exc2}", file=sys.stderr)
            return 4
    except TaigiASRError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    failed: list[Path] = []
    total_duration = 0.0
    total_elapsed = 0.0
    try:
        for idx, audio in enumerate(inputs, 1):
            if len(inputs) > 1:
                print(f"\n[{idx}/{len(inputs)}] {audio}", file=sys.stderr)
            ok, duration, elapsed = _transcribe_one(
                engine,
                spec,
                audio,
                formats,
                args.out,
                word_timestamps=args.word_timestamps,
                beam_size=args.beam_size,
                best_of=args.best_of,
                write_alongside_only=write_alongside_only,
            )
            total_duration += duration
            total_elapsed += elapsed
            if len(inputs) > 1 and ok:
                # xRT here is per-file convert+transcribe wall-clock vs audio
                # duration. Excludes the one-time model load.
                xrt = duration / max(elapsed, 1e-3)
                print(f"  -> {elapsed:.1f}s (xRT {xrt:.1f})", file=sys.stderr)
            if not ok:
                failed.append(audio)
    finally:
        try:
            engine.unload()
        except Exception:  # pragma: no cover
            pass

    if len(inputs) > 1:
        agg_xrt = total_duration / max(total_elapsed, 1e-3)
        print(
            f"\nBatch summary: {len(inputs) - len(failed)}/{len(inputs)} OK | "
            f"audio {total_duration:.0f}s | wall {total_elapsed:.0f}s | "
            f"xRT {agg_xrt:.1f} (excl. model load)",
            file=sys.stderr,
        )

    if failed:
        print(
            f"FAILED: {len(failed)} file(s): {', '.join(p.name for p in failed)}",
            file=sys.stderr,
        )
        return 4 if len(failed) == len(inputs) else 7

    return 0


if __name__ == "__main__":
    sys.exit(main())
