"""Pure output formatters: TXT / SRT / WebVTT / JSON.

No side effects, no I/O — easy to unit-test. Callers write the returned strings to disk.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from taigi_asr.segments import TimestampedSegment


def to_txt(segments: Sequence[TimestampedSegment]) -> str:
    """Plain-text transcript, one timestamped line per segment."""
    if not segments:
        return ""
    return "".join(seg.to_timestamp_line() + "\n" for seg in segments)


def to_srt(segments: Sequence[TimestampedSegment]) -> str:
    """SRT subtitle format. Cues separated by a blank line; file ends with a
    blank line as most SRT players expect."""
    if not segments:
        return ""
    blocks = [seg.to_srt_block(idx) for idx, seg in enumerate(segments, 1)]
    return "\n".join(blocks) + "\n"


def to_vtt(
    segments: Sequence[TimestampedSegment],
    speakers: Sequence[str] | None = None,
) -> str:
    """WebVTT subtitle format. Always emits the WEBVTT header + blank line.

    With ``speakers`` (positionally parallel to ``segments``), each cue uses
    WebVTT's native voice tag ``<v Speaker>`` — compliant players render
    per-speaker colours. The label is escaped so a hostile speaker name can't
    inject markup. (Voice-tag approach adopted from upstream.)
    """
    if not segments:
        return "WEBVTT\n\n"
    if not speakers:
        body = "\n".join(seg.to_vtt_block() for seg in segments)
        return f"WEBVTT\n\n{body}"

    blocks: list[str] = []
    for seg, spk in zip(segments, speakers, strict=True):
        block = seg.to_vtt_block()
        if spk:
            safe = spk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ts_line, text = block.split("\n", 1)
            block = f"{ts_line}\n<v {safe}>{text}"
        blocks.append(block)
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_json(
    segments: Sequence[TimestampedSegment],
    meta: Mapping[str, Any] | None = None,
    speakers: Sequence[str] | None = None,
) -> str:
    """JSON serialization with optional metadata (engine name, language, duration...).

    With ``speakers``, each segment dict gains a ``"speaker"`` key. A separate
    field, not a text prefix: JSON is consumed by code, and downstream tools
    should not have to parse a label back out of the transcript text.
    """
    seg_dicts = [seg.to_json_dict() for seg in segments]
    if speakers:
        for d, spk in zip(seg_dicts, speakers, strict=True):
            if spk:
                d["speaker"] = spk
    payload: dict[str, Any] = {"segments": seg_dicts}
    if meta is not None:
        payload["meta"] = dict(meta)
    return json.dumps(payload, ensure_ascii=False, indent=2)
