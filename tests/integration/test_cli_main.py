"""End-to-end-shaped tests for ``taigi_asr.cli.main``.

These exercise the real CLI flow — argparse → input resolution → engine
load → per-file transcribe → output write — but stub out the two heavy
external dependencies (ffmpeg via ``AudioConverter`` and the GPU/model
via ``build_engine``) so they run in <1 s without ffmpeg or CUDA.

Why the integration/ folder rather than unit/: these touch multiple
modules wired together (cli + audio + engine + formatters + router) and
assert end-to-end behavior (exit codes, output files on disk, stderr
content). The ``slow`` mark is intentionally NOT applied because no
network or GPU is involved — they should run in CI's default bucket.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taigi_asr import cli
from taigi_asr.engines.fake import FakeEngine
from taigi_asr.errors import TranscriptionError
from taigi_asr.router import EngineKind, EngineSpec, GPUInfo
from taigi_asr.segments import TimestampedSegment

# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class CountingFakeEngine(FakeEngine):
    """FakeEngine that also tracks how many times load() was invoked.

    ``call_count`` (inherited) tracks transcribe() calls; ``load_count``
    here lets multi-file tests prove that the CLI loads the model exactly
    once for the entire batch.
    """

    def __init__(self, script: list[TimestampedSegment]) -> None:
        super().__init__(script=script)
        self.load_count = 0

    def load(self) -> None:
        self.load_count += 1
        super().load()


class SelectiveFakeEngine(CountingFakeEngine):
    """FakeEngine that fails on a configurable subset of transcribe calls.

    Used to prove the multi-file flow keeps going past a per-file failure
    and reports the right exit code (7 = partial, 4 = total).
    """

    def __init__(
        self,
        script: list[TimestampedSegment],
        fail_indices: set[int] | None = None,
    ) -> None:
        super().__init__(script=script)
        self._fail_at = set(fail_indices or ())

    def transcribe(self, wav_path, *, word_timestamps: bool = False):
        idx = self.call_count
        self.call_count += 1
        if idx in self._fail_at:
            raise TranscriptionError(f"simulated failure on call {idx}")
        return list(self._script)


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Wire up the cli module with stubbed GPU + ffmpeg + engine.

    Returns a callable ``patch(engine, spec=None, info=None)`` so each test
    can supply its own engine flavor while reusing the boilerplate.
    """

    def patch(
        engine,
        *,
        spec: EngineSpec | None = None,
        info: GPUInfo | None = None,
        convert_duration: float = 5.0,
    ) -> None:
        spec = spec or EngineSpec(
            kind=EngineKind.FASTER_WHISPER,
            device="cuda",
            compute_type="int8_float16",
            batch_size=4,
        )
        info = info or GPUInfo(
            name="FakeGPU",
            vram_gb=4.0,
            cuda_available=True,
            bf16_supported=False,
        )

        monkeypatch.setattr("taigi_asr.cli.GPUProfiler.detect", staticmethod(lambda: info))
        monkeypatch.setattr(
            "taigi_asr.cli.EngineRouter.select",
            staticmethod(lambda _info, prefer=None: spec),
        )
        monkeypatch.setattr("taigi_asr.cli.build_engine", lambda _spec: engine)

        def fake_convert(src, out_dir=None):
            wav = tmp_path / "_taigi_asr_fake.wav"
            wav.write_bytes(b"")
            return wav, convert_duration

        monkeypatch.setattr(
            "taigi_asr.cli.AudioConverter.convert",
            staticmethod(fake_convert),
        )
        monkeypatch.setattr(
            "taigi_asr.cli.AudioConverter.cleanup",
            staticmethod(lambda _p: None),
        )

    return patch


def _default_engine() -> CountingFakeEngine:
    return CountingFakeEngine(
        script=[
            TimestampedSegment(0.0, 1.0, "小時候跌倒"),
            TimestampedSegment(1.0, 2.0, "醫沒好"),
        ]
    )


# --------------------------------------------------------------------------
# Single-file path
# --------------------------------------------------------------------------


def test_single_file_default_writes_alongside(fake_cli, tmp_path: Path) -> None:
    engine = _default_engine()
    fake_cli(engine)
    audio = _touch(tmp_path / "song.mp3")

    rc = cli.main([str(audio)])

    assert rc == 0
    assert (tmp_path / "song.srt").exists()
    assert engine.load_count == 1
    assert engine.call_count == 1


def test_single_file_with_out_honors_explicit_path(fake_cli, tmp_path: Path) -> None:
    engine = _default_engine()
    fake_cli(engine)
    audio = _touch(tmp_path / "in.mp3")
    out = tmp_path / "subdir" / "custom.srt"
    out.parent.mkdir()

    rc = cli.main([str(audio), "--out", str(out)])

    assert rc == 0
    assert out.exists()
    assert not (tmp_path / "in.srt").exists()


def test_single_file_multi_format_writes_each_alongside(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _default_engine()
    fake_cli(engine)
    audio = _touch(tmp_path / "in.mp3")
    out_should_be_ignored = tmp_path / "ignored.json"

    rc = cli.main([str(audio), "--format", "srt,json,txt", "--out", str(out_should_be_ignored)])

    assert rc == 0
    for suf in (".srt", ".json", ".txt"):
        assert (tmp_path / f"in{suf}").exists()
    assert not out_should_be_ignored.exists()
    assert "--out ignored" in capsys.readouterr().err


def test_empty_transcript_returns_4(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = CountingFakeEngine(script=[])
    fake_cli(engine)
    audio = _touch(tmp_path / "silence.mp3")

    rc = cli.main([str(audio)])

    assert rc == 4
    assert "empty transcript" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Multi-file path
# --------------------------------------------------------------------------


def test_multi_file_loads_engine_once(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _default_engine()
    fake_cli(engine, convert_duration=10.0)
    files = [_touch(tmp_path / n) for n in ("a.mp3", "b.mp3", "c.mp3")]

    rc = cli.main([str(f) for f in files])

    assert rc == 0
    assert engine.load_count == 1, "engine.load() must be called exactly once for the batch"
    assert engine.call_count == 3
    for f in files:
        assert f.with_suffix(".srt").exists()
    err = capsys.readouterr().err
    assert "Batch summary:" in err
    assert "3/3 OK" in err


def test_input_dir_picks_only_supported_extensions(fake_cli, tmp_path: Path) -> None:
    engine = _default_engine()
    fake_cli(engine)
    folder = tmp_path / "music"
    folder.mkdir()
    for n in ("track.mp3", "voice.m4a", "skip.txt", "cover.png"):
        _touch(folder / n)

    rc = cli.main(["--input-dir", str(folder)])

    assert rc == 0
    assert engine.call_count == 2
    assert (folder / "track.srt").exists()
    assert (folder / "voice.srt").exists()
    assert not (folder / "skip.srt").exists()


def test_partial_batch_failure_returns_7(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = SelectiveFakeEngine(
        script=[TimestampedSegment(0.0, 1.0, "ok")],
        fail_indices={1},  # second of three fails
    )
    fake_cli(engine)
    files = [_touch(tmp_path / n) for n in ("a.mp3", "b.mp3", "c.mp3")]

    rc = cli.main([str(f) for f in files])

    assert rc == 7
    err = capsys.readouterr().err
    assert "FAILED: 1 file(s)" in err
    assert "b.mp3" in err
    # Successful peers still produced output.
    assert (tmp_path / "a.srt").exists()
    assert (tmp_path / "c.srt").exists()
    assert not (tmp_path / "b.srt").exists()


def test_total_batch_failure_returns_4(fake_cli, tmp_path: Path) -> None:
    engine = SelectiveFakeEngine(
        script=[TimestampedSegment(0.0, 1.0, "ok")],
        fail_indices={0, 1},  # both fail
    )
    fake_cli(engine)
    files = [_touch(tmp_path / n) for n in ("a.mp3", "b.mp3")]

    rc = cli.main([str(f) for f in files])

    assert rc == 4


def test_aggregate_xrt_excludes_failed_files(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R2 fix: aggregate xRT counts successful files only."""
    engine = SelectiveFakeEngine(
        script=[TimestampedSegment(0.0, 1.0, "ok")],
        fail_indices={1},
    )
    fake_cli(engine, convert_duration=12.0)
    files = [_touch(tmp_path / n) for n in ("a.mp3", "b.mp3", "c.mp3")]

    cli.main([str(f) for f in files])

    err = capsys.readouterr().err
    # Two successful files × 12s convert_duration = 24s of audio counted,
    # not 36s (which would happen if we naively summed all three).
    assert "audio 24s" in err
    assert "2/3 OK" in err


# --------------------------------------------------------------------------
# Argparse + validation paths
# --------------------------------------------------------------------------


def test_no_inputs_returns_2(fake_cli, capsys: pytest.CaptureFixture[str]) -> None:
    fake_cli(_default_engine())
    rc = cli.main([])
    assert rc == 2
    assert "no audio inputs" in capsys.readouterr().err


def test_missing_file_returns_2(fake_cli, tmp_path: Path) -> None:
    fake_cli(_default_engine())
    rc = cli.main([str(tmp_path / "does_not_exist.mp3")])
    assert rc == 2


def test_bad_format_returns_6(fake_cli, tmp_path: Path) -> None:
    fake_cli(_default_engine())
    audio = _touch(tmp_path / "in.mp3")
    rc = cli.main([str(audio), "--format", "weird,srt"])
    assert rc == 6


def test_input_dir_missing_returns_2(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli(_default_engine())
    rc = cli.main(["--input-dir", str(tmp_path / "nope")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Engine-specific knob filtering (R1 regression guard)
# --------------------------------------------------------------------------


def test_hf_engine_warning_printed_once_in_batch(
    fake_cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1 regression: ``--beam-size`` warning on HF engine prints exactly
    once even with N input files. Previously printed once per file inside
    the batch loop."""
    engine = _default_engine()
    hf_spec = EngineSpec(
        kind=EngineKind.HUGGINGFACE,
        device="cuda",
        compute_type="float16",
        batch_size=1,
    )
    fake_cli(engine, spec=hf_spec)
    files = [_touch(tmp_path / n) for n in ("a.mp3", "b.mp3", "c.mp3", "d.mp3")]

    rc = cli.main([str(f) for f in files] + ["--beam-size", "10", "--best-of", "3"])

    assert rc == 0
    err = capsys.readouterr().err
    occurrences = err.count("--beam-size / --best-of are ignored on the HuggingFace engine")
    assert occurrences == 1, f"Expected exactly 1 HF warning, got {occurrences}\nstderr=\n{err}"


def test_fw_engine_passes_beam_size_to_transcribe(
    fake_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symmetric guard: on FW spec, beam_size flows through to the engine."""
    received_kwargs: dict = {}

    class CapturingEngine(CountingFakeEngine):
        def transcribe(self, wav_path, **kwargs):
            received_kwargs.update(kwargs)
            self.call_count += 1
            return list(self._script)

    engine = CapturingEngine(script=[TimestampedSegment(0.0, 1.0, "x")])
    fake_cli(engine)
    audio = _touch(tmp_path / "song.mp3")

    cli.main([str(audio), "--beam-size", "8", "--best-of", "3", "--word-timestamps"])

    assert received_kwargs == {"beam_size": 8, "best_of": 3, "word_timestamps": True}
