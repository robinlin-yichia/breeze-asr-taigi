"""Integration test: real Faster-Whisper engine transcribing data/test.m4a.

Needs:
  - faster-whisper installed
  - ``paulpengtw/faster-whisper-Breeze-ASR-26`` model in HF cache (~2.9 GB)
  - ffmpeg on PATH

Marked ``slow`` so CI (no GPU / no model) skips it. Run locally with::

    pytest -m slow tests/integration -v -s
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from taigi_asr.audio import AudioConverter
from taigi_asr.router import EngineKind, EngineRouter, GPUProfiler
from taigi_asr.segments import TimestampedSegment

# Reference transcript: "小時候跌倒醫沒好，長大傻傻在這裡喊玲瓏。"
# Model output varies significantly based on compute_type (int8 vs fp16) and
# beam_size. We assert on a set of reliable *anchor* substrings and require a
# majority to be present, rather than demanding exact-match.
RELIABLE_ANCHORS = ["小時候", "跌倒", "沒好", "長大", "傻傻", "這裡"]
MIN_ANCHORS = 4  # out of 6; tolerates int8-CPU quality drift


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _faster_whisper_available(), reason="faster-whisper not installed"),
    pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH"),
]


def test_transcribe_hokkien_sample(sample_audio: Path, tmp_path: Path) -> None:
    """End-to-end: preprocess audio -> route -> load -> transcribe -> assert content."""
    from taigi_asr.engines.faster_whisper import FasterWhisperEngine

    info = GPUProfiler.detect()
    spec = EngineRouter.select(info, prefer=EngineKind.FASTER_WHISPER)

    # When running on a machine without CUDA (e.g. CI / dev laptop via remote
    # shell), fall back to CPU int8 — same path as the router's CPU branch.
    if not info.cuda_available:
        engine = FasterWhisperEngine(device="cpu", compute_type="int8", batch_size=1)
    else:
        engine = FasterWhisperEngine(
            device=spec.device,
            compute_type=spec.compute_type,
            batch_size=spec.batch_size,
        )

    wav_path, duration = AudioConverter.convert(sample_audio, out_dir=tmp_path)
    try:
        assert duration > 0, "Expected non-zero audio duration"
        segments = engine.transcribe(wav_path, word_timestamps=False)
    finally:
        AudioConverter.cleanup(wav_path)
        engine.unload()

    assert segments, "Engine returned zero segments — inference failed silently"
    full_text = "".join(seg.text for seg in segments)

    print(f"\nRecognized: {full_text}\n")

    hits = [a for a in RELIABLE_ANCHORS if a in full_text]
    assert len(hits) >= MIN_ANCHORS, (
        f"Only matched {len(hits)}/{len(RELIABLE_ANCHORS)} anchors "
        f"({hits}); expected >= {MIN_ANCHORS}. Full output: {full_text!r}"
    )


def test_timestamps_are_monotonic_and_within_duration(
    sample_audio: Path, tmp_path: Path
) -> None:
    """Segments must be time-ordered and bounded by audio duration."""
    from taigi_asr.engines.faster_whisper import FasterWhisperEngine

    info = GPUProfiler.detect()
    engine = FasterWhisperEngine(
        device="cpu" if not info.cuda_available else "cuda",
        compute_type="int8" if not info.cuda_available else "int8_float16",
        batch_size=1,
    )

    wav_path, duration = AudioConverter.convert(sample_audio, out_dir=tmp_path)
    try:
        segments = engine.transcribe(wav_path)
    finally:
        AudioConverter.cleanup(wav_path)
        engine.unload()

    assert all(isinstance(s, TimestampedSegment) for s in segments)
    assert all(s.start_time <= s.end_time for s in segments)
    for prev, curr in zip(segments, segments[1:]):
        assert prev.start_time <= curr.start_time, "Segments must be in chronological order"
    # Tolerance: +0.5s for VAD padding
    assert segments[-1].end_time <= duration + 0.5
