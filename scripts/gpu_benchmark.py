"""GPU benchmark for Faster-Whisper on RTX 3050 4GB.

Runs the full pipeline against data/test.mp3 (~54 min Hokkien) and reports:
  - wall-clock time per stage (preprocess, load, transcribe)
  - real-time factor (xRT)
  - peak GPU memory usage
  - accuracy anchor hits against a reference transcript set

Run with: python scripts/gpu_benchmark.py --beam-size 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set allocator hint BEFORE importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402

from taigi_asr.audio import AudioConverter  # noqa: E402
from taigi_asr.engines.faster_whisper import FasterWhisperEngine  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "data" / "test.mp3"

# Reference anchors likely to appear in a 54-min Hokkien recording. Used as a
# rough accuracy smoke; for rigorous WER you'd need a full reference transcript.
ACCURACY_ANCHORS = [
    "小時候",
    "跌倒",
    "長大",
    "傻傻",
    "這裡",
    "玲瓏",
    "醫沒好",
    "這",
    "不",
    "是",
]


def _fmt_time(s: float) -> str:
    return f"{s:.1f} s" if s < 60 else f"{int(s // 60)}m {s % 60:.1f}s"


def _torch_peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _nvidia_smi_used_mib() -> int:
    """CTranslate2 doesn't use torch's allocator, so torch.cuda.max_memory_allocated
    reports 0. Fall back to nvidia-smi for the real VRAM figure."""
    import subprocess

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--best-of", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--compute-type", default="int8_float16", choices=["int8_float16", "float16", "int8"]
    )
    ap.add_argument("--audio", type=Path, default=AUDIO)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; benchmark requires GPU.")
        return 2

    dev = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("=" * 72)
    print(f"Device: {dev} ({vram:.2f} GB)")
    print(
        f"Config: compute_type={args.compute_type} batch={args.batch_size} "
        f"beam={args.beam_size} best_of={args.best_of}"
    )
    print(f"Audio:  {args.audio}")
    print("=" * 72)

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}")
        return 2

    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    wav_path, duration = AudioConverter.convert(args.audio)
    t_pre = time.perf_counter() - t0
    print(f"[1/3] Preprocess: {_fmt_time(t_pre)} | audio duration {_fmt_time(duration)}")

    try:
        engine = FasterWhisperEngine(
            device="cuda",
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            best_of=args.best_of,
        )
        vram_before = _nvidia_smi_used_mib()
        t0 = time.perf_counter()
        engine.load()
        t_load = time.perf_counter() - t0
        vram_after_load = _nvidia_smi_used_mib()
        print(
            f"[2/3] Model load: {_fmt_time(t_load)} | "
            f"VRAM after load {vram_after_load} MiB (delta +{vram_after_load - vram_before} MiB)"
        )

        t0 = time.perf_counter()
        segments = engine.transcribe(wav_path, word_timestamps=False)
        t_tr = time.perf_counter() - t0
        vram_peak = _nvidia_smi_used_mib()
        peak = vram_peak / 1024  # GB for display

        full = "".join(s.text for s in segments)
        char_count = len(full)
        seg_count = len(segments)

        xrt = duration / t_tr if t_tr > 0 else 0
        hits = [a for a in ACCURACY_ANCHORS if a in full]

        print(
            f"[3/3] Transcribe: {_fmt_time(t_tr)} | "
            f"xRT {xrt:.2f}x | peak VRAM {vram_peak} MiB ({peak:.2f} GB)"
        )
        print("-" * 72)
        print(f"Segments: {seg_count} | Characters: {char_count}")
        print(f"Anchors hit: {len(hits)}/{len(ACCURACY_ANCHORS)} -> {hits}")
        print("-" * 72)
        print("First 500 chars of transcript:")
        print(full[:500])
        print("-" * 72)
        print("Last 300 chars of transcript:")
        print(full[-300:])
        print("=" * 72)

        total = t_pre + t_load + t_tr
        print(
            f"TOTAL: {_fmt_time(total)} for {_fmt_time(duration)} of audio "
            f"(overall {duration / total:.2f}x real-time)"
        )
    finally:
        AudioConverter.cleanup(wav_path)
        try:
            engine.unload()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
