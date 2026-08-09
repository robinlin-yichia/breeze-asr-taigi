"""Speaker diarization — pyannote.audio wrapped for in-process use.

Used by the transcription UI to label speakers directly in-process.

Requires a HuggingFace token and accepted licences for
``pyannote/speaker-diarization-3.1``, ``pyannote/segmentation-3.0`` and
``pyannote/speaker-diarization-community-1`` — pyannote 4.x pulls the last one
in as the default checkpoint even when you ask for 3.1.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODEL_ID = "pyannote/speaker-diarization-3.1"

_pipeline: Any = None
_pipeline_device: str | None = None
_lock = threading.Lock()


class DiarizationError(RuntimeError):
    """Raised when speakers could not be determined."""


def available() -> bool:
    """True if pyannote.audio is importable. Cheap enough for UI wiring."""
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        return False
    return True


def token_from_env() -> str:
    # HUGGINGFACE_HUB_TOKEN is the name huggingface_hub itself honours; users
    # who already authenticated other tools shouldn't need to set a second var.
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )


def load_waveform(wav_path: str | Path) -> dict:
    """Decode a WAV into pyannote's ``{"waveform", "sample_rate"}`` form.

    Handing pyannote a *path* makes it decode via torchcodec, which on Windows
    needs FFmpeg's shared DLLs (versions 4-8 only) and fails against a static
    ffmpeg build. Decoding here sidesteps that path entirely — which is what
    pyannote's own warning recommends.
    """
    import numpy as np
    import torch

    try:
        import soundfile as sf

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except ImportError:
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        if width not in dtype_map:
            raise DiarizationError(f"不支援的 WAV 取樣寬度：{width} bytes") from None
        dtype = dtype_map[width]
        data = np.frombuffer(raw, dtype=dtype)
        if width == 1:  # 8-bit WAV is unsigned
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32) / float(np.iinfo(dtype).max)
        data = data.reshape(-1, channels)

    tensor = torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32))
    return {"waveform": tensor, "sample_rate": int(sr)}


def get_pipeline(token: str, device: str = "cuda"):
    """Load (once) and place the pipeline on ``device``."""
    global _pipeline, _pipeline_device
    with _lock:
        if _pipeline is None:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise DiarizationError(
                    "未安裝 pyannote.audio。請執行 pip install \"pyannote.audio>=4.0\""
                ) from exc

            # pyannote 4.x renamed use_auth_token -> token; 3.x only knows the old name.
            try:
                try:
                    _pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
                except TypeError:
                    _pipeline = Pipeline.from_pretrained(MODEL_ID, use_auth_token=token)
            except Exception as exc:
                # The concrete exception class varies across huggingface_hub
                # versions (GatedRepoError / HfHubHTTPError / HTTPError) —
                # matching the message text is the version-stable signal.
                # (Approach adopted from upstream's feat/speaker-diarization.)
                msg = str(exc)
                if "gated" in msg.lower() or "401" in msg or "403" in msg:
                    raise DiarizationError(
                        "模型授權未通過。請用持有 HF_TOKEN 的同一個帳號，到以下三頁"
                        "各按一次「Agree and access repository」後重試：\n"
                        "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                        "  https://huggingface.co/pyannote/segmentation-3.0\n"
                        "  https://huggingface.co/pyannote/speaker-diarization-community-1"
                        "（最常漏掉這個）\n"
                        f"（原始錯誤：{exc}）"
                    ) from exc
                raise

            if _pipeline is None:
                raise DiarizationError(
                    "Pipeline 載入失敗。多半是 HF_TOKEN 無效，或尚未在 HuggingFace "
                    "同意 speaker-diarization-3.1、segmentation-3.0 與 "
                    "speaker-diarization-community-1 的授權條款。"
                )

        if _pipeline_device != device:
            import torch

            _pipeline.to(torch.device(device))
            _pipeline_device = device
    return _pipeline


def unload() -> None:
    """Drop the pipeline and reclaim VRAM.

    pyannote's pipeline keeps strong refs to its segmentation/embedding
    sub-models via Inference wrappers whose hook cycles survive one GC pass —
    upstream measured ~600-900 MB left resident after a plain ``del``. Null
    the leaf models first, then double-collect. (Adopted from upstream's
    feat/speaker-diarization branch.)
    """
    global _pipeline, _pipeline_device
    with _lock:
        if _pipeline is not None:
            try:
                for attr_name in ("_segmentation", "_embedding"):
                    sub = getattr(_pipeline, attr_name, None)
                    if sub is None:
                        continue
                    for inner in ("model", "model_"):
                        if hasattr(sub, inner):
                            setattr(sub, inner, None)
            except Exception as exc:  # a pyannote rename must not brick unload
                log.debug("pyannote sub-model null-out skipped: %s", exc)
        _pipeline = None
        _pipeline_device = None
    try:
        import gc

        import torch

        gc.collect()
        gc.collect()  # second pass breaks the Inference hook cycles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def diarize(
    wav_path: str | Path,
    token: str | None = None,
    device: str = "cuda",
    num_speakers: int = 0,
    min_speakers: int = 0,
    max_speakers: int = 0,
) -> list[tuple[float, float, str]]:
    """Return ``[(start, end, speaker), ...]`` speaker turns."""
    token = (token or "").strip() or token_from_env()
    if not token:
        raise DiarizationError(
            "缺少 HF_TOKEN。請設定環境變數，或在 UI 欄位填入 HuggingFace token。"
        )

    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA unavailable; diarizing on CPU.")
            device = "cpu"
    except ImportError:
        device = "cpu"

    pipe = get_pipeline(token, device)

    kwargs: dict = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers and min_speakers > 0:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers and max_speakers > 0:
            kwargs["max_speakers"] = int(max_speakers)

    annotation = pipe(load_waveform(wav_path), **kwargs)
    # pyannote 4.x returns DiarizeOutput; 3.x returns the Annotation directly.
    annotation = getattr(annotation, "speaker_diarization", annotation)
    return [
        (turn.start, turn.end, label)
        for turn, _track, label in annotation.itertracks(yield_label=True)
    ]


def parse_name_map(raw: str) -> dict[str, str]:
    """Parse ``SPEAKER_00=王經理`` lines into a rename mapping.

    One pair per line; fullwidth ＝ is accepted, blank lines and lines
    without a separator are ignored. Values are stripped; an empty value
    drops the line rather than renaming to nothing.
    """
    mapping: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in ("=", "＝"):
            if sep in line:
                key, value = line.split(sep, 1)
                key, value = key.strip(), value.strip()
                if key and value:
                    mapping[key] = value
                break
    return mapping


def rename_speakers(speakers: Sequence[str], mapping: dict[str, str]) -> list[str]:
    """Apply a name map; labels without an entry pass through unchanged."""
    if not mapping:
        return list(speakers)
    return [mapping.get(s, s) if s else s for s in speakers]


def assign(
    spans: Sequence[tuple[float, float]],
    turns: Sequence[tuple[float, float, str]],
) -> list[str]:
    """Label each ``(start, end)`` span with the speaker it overlaps longest.

    Overlap is summed **per speaker**, not per turn (approach adopted from
    upstream's feat/speaker-diarization branch): VAD frequently splits one
    person's speech into several short turns, and taking the single largest
    turn would let the other side's one medium turn outvote them.

    Returns a list positionally parallel to ``spans``.
    """
    if not turns:
        return ["SPEAKER_UNK"] * len(spans)

    # Sorting enables the early break below; itertracks() usually yields in
    # time order already, but that is not a documented guarantee.
    turns = sorted(turns, key=lambda t: t[0])

    labels: list[str] = []
    for start, end in spans:
        overlaps: dict[str, float] = {}
        for t_start, t_end, label in turns:
            if t_end <= start:
                continue
            if t_start >= end:
                break  # sorted: every later turn is also past this span
            ov = min(end, t_end) - max(start, t_start)
            if ov > 0:
                overlaps[label] = overlaps.get(label, 0.0) + ov
        if overlaps:
            labels.append(max(overlaps.items(), key=lambda kv: kv[1])[0])
        else:
            # No overlap at all — common at VAD boundaries. Fall back to the
            # turn whose edge sits closest to this span's midpoint.
            mid = (start + end) / 2
            labels.append(min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2])
    return labels
