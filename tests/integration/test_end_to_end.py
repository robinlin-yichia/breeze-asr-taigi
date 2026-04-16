"""End-to-end pipeline test: audio file -> CLI -> SRT file on disk.

Uses :class:`FakeEngine` for a fast path (no GPU / no model download), then a
separate test actually exercises the Faster-Whisper path on ``data/test.m4a``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from taigi_asr.audio import AudioConverter
from taigi_asr.engines.fake import FakeEngine
from taigi_asr.formatters import to_srt
from taigi_asr.segments import TimestampedSegment


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_fake_engine_end_to_end(sample_audio: Path, tmp_path: Path) -> None:
    """Full pipeline with a fake engine — fast sanity check, no model needed."""
    scripted = [
        TimestampedSegment(0.0, 3.0, "小時候跌倒醫沒好"),
        TimestampedSegment(3.0, 6.5, "長大傻傻在這裡喊玲瓏"),
    ]
    engine = FakeEngine(script=scripted)

    wav_path, duration = AudioConverter.convert(sample_audio, out_dir=tmp_path)
    try:
        engine.load()
        segments = engine.transcribe(wav_path)
    finally:
        AudioConverter.cleanup(wav_path)
        engine.unload()

    srt_text = to_srt(segments)
    out_file = tmp_path / "out.srt"
    out_file.write_text(srt_text, encoding="utf-8")

    content = out_file.read_text(encoding="utf-8")
    assert "小時候跌倒醫沒好" in content
    assert "長大傻傻在這裡喊玲瓏" in content
    # Structural check: each block starts with an integer index
    assert "1\n00:00:00,000 --> 00:00:03,000" in content
    assert "2\n00:00:03,000 --> 00:00:06,500" in content
    assert duration > 0


@pytest.mark.slow
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_cli_transcribes_sample_audio(sample_audio: Path, tmp_path: Path) -> None:
    """Invoke the CLI as a subprocess — true smoke of the whole installed entry point."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        pytest.skip("faster-whisper not installed")

    out_srt = tmp_path / "out.srt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "taigi_asr.cli",
            str(sample_audio),
            "--engine",
            "fw",
            "--format",
            "srt",
            "--out",
            str(out_srt),
            "-v",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert out_srt.exists(), "SRT file not created"
    content = out_srt.read_text(encoding="utf-8")
    # int8/CPU inference introduces small character-level variance; check
    # robust anchors instead of exact reference transcript.
    anchors = ["小時候", "跌倒", "長大", "傻傻", "這裡"]
    hits = [a for a in anchors if a in content]
    assert len(hits) >= 4, f"Only matched {hits} / {anchors}; SRT={content!r}"
    # SRT structural sanity
    assert "-->" in content
    assert content.strip().startswith("1")
