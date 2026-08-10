"""Gradio UI — the zero-friction front door.

End-user flow: **drop audio -> click transcribe -> download SRT/TXT**. Advanced
options are collapsed by default; 99% of users never open them.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from taigi_asr import diarize as diar_mod
from taigi_asr import terms as terms_mod
from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.engines.base import ASREngine
from taigi_asr.errors import InsufficientVRAMError, ModelLoadError, TranscriptionError
from taigi_asr.formatters import to_json, to_srt, to_txt, to_vtt
from taigi_asr.router import EngineKind, EngineRouter, EngineSpec, GPUInfo, GPUProfiler
from taigi_asr.segments import TimestampedSegment

log = logging.getLogger(__name__)


# --------------------------- state / helpers ------------------------------- #


_engine_cache: dict[str, ASREngine] = {}
_engine_lock = threading.Lock()


def _spec_key(spec: EngineSpec) -> str:
    return f"{spec.kind.value}:{spec.device}:{spec.compute_type}:{spec.batch_size}"


def _get_or_build_engine(spec: EngineSpec) -> ASREngine:
    """LRU-of-1 engine cache. Swapping to a new spec unloads the old engine so
    two Whisper-large instances never coexist in VRAM (the 4 GB budget fits one).
    """
    key = _spec_key(spec)
    with _engine_lock:
        if key not in _engine_cache:
            # Evict every cached engine that doesn't match the new spec.
            for stale_key in list(_engine_cache.keys()):
                if stale_key == key:
                    continue
                try:
                    _engine_cache[stale_key].unload()
                except Exception:
                    log.exception("Failed to unload stale engine %s", stale_key)
                del _engine_cache[stale_key]
            _engine_cache[key] = build_engine(spec)
        return _engine_cache[key]


def _gpu_panel_text(info: GPUInfo, spec: EngineSpec) -> str:
    lines = [
        f"**裝置 / Device**：{info.name}",
        f"**VRAM**：{info.vram_gb:.1f} GB" if info.cuda_available else "**VRAM**：CPU-only",
        f"**自動選擇引擎**：{spec.kind.value} ({spec.compute_type}, batch={spec.batch_size})",
    ]
    if not info.cuda_available:
        lines.append("警告：未偵測到 CUDA GPU，將使用 CPU（較慢）")
    elif info.vram_gb < 4:
        lines.append("警告：VRAM < 4 GB，建議關閉其他 GPU 程式釋放記憶體")
    else:
        lines.append("狀態：硬體配置正常")
    return "\n\n".join(lines)


# Radio labels live here, shared with build_ui, so the mapping below cannot
# drift out of sync with the widget. A mismatch fails silently — an unknown
# label maps to None, which behaves exactly like "Auto" — so the strings must
# not be written out twice.
ENGINE_FW = "Faster-Whisper"
ENGINE_HF = "HuggingFace"
ENGINE_AUTO = "Auto / 自動"
ENGINE_CHOICES = [ENGINE_FW, ENGINE_HF, ENGINE_AUTO]

_ENGINE_PREFER: dict[str, EngineKind | None] = {
    ENGINE_FW: EngineKind.FASTER_WHISPER,
    ENGINE_HF: EngineKind.HUGGINGFACE,
    ENGINE_AUTO: None,
}


def _resolve_spec(info: GPUInfo, engine_choice: str) -> EngineSpec:
    if engine_choice not in _ENGINE_PREFER:
        log.warning("Unknown engine choice %r; falling back to Auto.", engine_choice)
    return EngineRouter.select(info, prefer=_ENGINE_PREFER.get(engine_choice))


# --------------------------- term corrections ------------------------------ #


def _apply_corrections(segments: list[TimestampedSegment], dict_path: str) -> str:
    """Load the term dictionary and fix segment text in place.

    Reloaded on every run so edits to terms.json take effect without a restart.
    Never raises: a broken dictionary degrades to "no correction" plus a note.
    """
    path = Path(dict_path).expanduser() if dict_path else terms_mod.find_default_dict()
    if not path or not Path(path).exists():
        return "校正：略過（找不到詞典檔）"
    try:
        rules = terms_mod.load_rules(path)
    except Exception as exc:
        log.warning("Term dictionary failed to load: %s", exc)
        return f"校正：詞典載入失敗（{exc}）"
    if not rules:
        return "校正：詞典無有效規則"
    counter = terms_mod.apply_to_segments(segments, rules)
    if not counter:
        return f"校正：{len(rules)} 條規則，無命中"
    return terms_mod.summarize(counter)


# --------------------------- speaker diarization --------------------------- #


def _run_diarization(
    wav_path: str | Path,
    segments: list[TimestampedSegment],
    num_speakers: int = 0,
) -> tuple[list[str], str]:
    """Label each segment with a speaker. Returns ``(speakers, note)``.

    ``speakers`` is positionally parallel to ``segments``, or empty when
    diarization could not run — a failure here keeps the transcript rather
    than discarding a run that may have taken ten minutes.
    """
    if not diar_mod.available():
        return [], (
            "語者標註：**略過** — 未安裝 pyannote.audio。\n\n"
            '請執行 `pip install "pyannote.audio>=4.0"`。'
        )
    if not diar_mod.token_from_env():
        return [], (
            "語者標註：**略過** — 未設定 `HF_TOKEN` 環境變數。\n\n"
            "需要 HuggingFace token，並已同意 `pyannote/speaker-diarization-community-1` 的授權。"
        )

    try:
        turns = diar_mod.diarize(wav_path, num_speakers=num_speakers)
    except diar_mod.DiarizationError as exc:
        return [], f"語者標註：**失敗** — {exc}\n\n逐字稿已保留，只是沒有語者標記。"
    except Exception as exc:  # pragma: no cover
        log.exception("Diarization failed")
        return [], (
            f"語者標註：**失敗** — {type(exc).__name__}: {str(exc).splitlines()[0]}\n\n"
            "逐字稿已保留，只是沒有語者標記。"
        )

    speakers = diar_mod.assign([(s.start_time, s.end_time) for s in segments], turns)
    detected = sorted({s for s in speakers if s})
    lines = [f"**語者標註完成** — {len(detected)} 位語者 / {len(turns)} 個 speaker turns"]
    for spk in sorted(detected, key=lambda s: -speakers.count(s)):
        lines.append(f"- {spk}：{speakers.count(spk)} 句")
    return speakers, "\n".join(lines)


def _compose(
    segments: list[TimestampedSegment],
    speakers: list[str] | None,
    on_change_only: bool = True,
) -> list[TimestampedSegment]:
    """Build the SRT-bound segments, prefixing a speaker label at turn changes.

    Speakers are kept alongside the transcript rather than baked into it, so
    re-applying the dictionary never has to parse a label back out of the text.
    SRT is the only format that needs text prefixes — it has no speaker
    syntax; VTT uses native voice tags and JSON a separate field instead.

    ``on_change_only`` tags just the first line of each turn — a label on every
    one of ~1900 five-second lines is noise when the speaker has not changed.
    """
    if not speakers:
        return list(segments)
    out: list[TimestampedSegment] = []
    previous: str | None = None
    for seg, spk in zip(segments, speakers, strict=True):
        show = spk and (not on_change_only or spk != previous)
        text = f"[{spk}] {seg.text}" if show else seg.text
        out.append(TimestampedSegment(seg.start_time, seg.end_time, text))
        previous = spk
    return out


def _speaker_txt(segments: list[TimestampedSegment], speakers: list[str]) -> str:
    """Readable transcript: one block per speaker turn, timestamped at the turn.

    This is the meeting-minutes view — consecutive lines from the same person
    are merged into a paragraph and carry a single start time, rather than a
    timestamp on every five-second fragment.
    """
    blocks: list[str] = []
    buf: list[str] = []
    current: str | None = None
    start = 0.0

    for seg, spk in zip(segments, speakers, strict=True):
        if spk != current:
            if buf:
                stamp = TimestampedSegment.format_time(start)
                blocks.append(f"[{stamp}] {current}：{' '.join(buf)}")
            buf, current, start = [], spk, seg.start_time
        buf.append(seg.text)

    if buf:
        stamp = TimestampedSegment.format_time(start)
        blocks.append(f"[{stamp}] {current}：{' '.join(buf)}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _preview_text(segments: list[TimestampedSegment]) -> str:
    preview = "\n".join(seg.to_timestamp_line() for seg in segments[:20])
    if len(segments) > 20:
        preview += f"\n... (共 {len(segments)} 段)"
    return preview


def _preview(segments: list[TimestampedSegment], speakers: list[str] | None) -> str:
    """Show what the TXT download will look like, not the raw segment list."""
    if not speakers:
        return _preview_text(segments)
    blocks = [b for b in _speaker_txt(segments, speakers).split("\n\n") if b.strip()]
    preview = "\n\n".join(blocks[:10])
    if len(blocks) > 10:
        preview += f"\n\n... (共 {len(blocks)} 段發言 / {len(segments)} 句)"
    return preview


def _reapply_terms(
    state: dict | None,
    enable_fix: bool,
    dict_path: str,
    name_map_raw: str = "",
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    """Re-run the dictionary against the LAST transcription without re-decoding.

    Always restores the pristine ASR text first, so removing a rule reverts its
    effect instead of leaving half-corrected text behind. Speaker renames work
    the same way: the state keeps raw SPEAKER_NN labels and the current map is
    applied fresh each time, so editing or clearing the map is always undoable.
    """
    if not state or not state.get("segments"):
        return ("ERROR: 尚無轉錄結果，請先執行一次轉錄。", "", None, None, None, None)

    segments = state["segments"]
    for seg, raw in zip(segments, state["raw"], strict=True):
        try:
            seg.text = raw
        except Exception:
            object.__setattr__(seg, "text", raw)

    fix_note = _apply_corrections(segments, dict_path) if enable_fix else "校正：已停用"

    # Diarization is never re-run here — only the rename map is re-applied.
    speakers = state.get("speakers")
    if speakers:
        speakers = diar_mod.rename_speakers(speakers, diar_mod.parse_name_map(name_map_raw))
    outputs = _write_outputs(segments, state["formats"], meta=state["meta"], speakers=speakers)
    status = (
        f"[OK] 已重新套用詞典與語者名稱（未重新轉錄／未重跑語者標註）／{len(segments)} 段 "
        f"/ {sum(len(s.text) for s in segments)} 字"
    )
    if fix_note:
        status += "\n\n" + fix_note
    return (
        status,
        _preview(segments, speakers),
        outputs.get("srt"),
        outputs.get("txt"),
        outputs.get("vtt"),
        outputs.get("json"),
    )


def _load_vocab_text(dict_path: str) -> str:
    path = _resolve_dict_path(dict_path)
    return "\n".join(terms_mod.load_vocabulary(path)) if path.exists() else ""


def _save_vocab(raw: str, dict_path: str) -> tuple[str, str]:
    """Persist the vocabulary list. Returns (status, resolved path)."""
    path = _resolve_dict_path(dict_path)
    vocab = terms_mod.parse_vocabulary(raw)
    try:
        saved = terms_mod.save_vocabulary(path, vocab)
    except Exception as exc:
        log.warning("Failed to save vocabulary: %s", exc)
        return f"ERROR: 存檔失敗（{exc}）", str(path)
    if not vocab:
        return f"[OK] 詞彙表已清空 — `{saved}`", str(saved)
    preview = "、".join(vocab[:8]) + ("…" if len(vocab) > 8 else "")
    return f"[OK] 已存 {len(vocab)} 個詞彙 — {preview}", str(saved)


def _reload_dict(dict_path: str) -> str:
    path = Path(dict_path).expanduser() if dict_path else terms_mod.find_default_dict()
    if not path or not Path(path).exists():
        return (
            f"找不到詞典檔：`{dict_path or terms_mod.DEFAULT_NAME}`\n\n"
            "可按下方「產生詞典範本」建立一份。"
        )
    try:
        return terms_mod.describe(terms_mod.load_rules(path), path)
    except Exception as exc:
        return f"詞典載入失敗：{exc}"


def _make_template(dict_path: str) -> tuple[str, str]:
    path = Path(dict_path).expanduser() if dict_path else (
        terms_mod.PROJECT_ROOT / terms_mod.DEFAULT_NAME
    )
    if path.exists():
        return f"`{path}` 已存在，未覆蓋。", str(path)
    try:
        terms_mod.write_template(path)
    except Exception as exc:
        return f"建立失敗：{exc}", str(path)
    return terms_mod.describe(terms_mod.load_rules(path), path), str(path)


def _on_dict_upload(uploaded) -> tuple[str, str, list[list[Any]]]:
    if not uploaded:
        return gr.skip(), gr.skip(), gr.skip()
    path = uploaded if isinstance(uploaded, str) else getattr(uploaded, "name", "")
    return _reload_dict(path), path, _load_table(path)[0]


# --------------------------- dictionary editor ----------------------------- #


def _resolve_dict_path(dict_path: str) -> Path:
    """UI 路徑欄 -> 實際檔案路徑。空白時退回專案根目錄的預設檔名。"""
    if dict_path and dict_path.strip():
        return Path(dict_path.strip()).expanduser()
    found = terms_mod.find_default_dict()
    return Path(found) if found else terms_mod.PROJECT_ROOT / terms_mod.DEFAULT_NAME


# 最外層的非規則欄位（例如 _comment）在編輯期間寄放在這裡，
# 存檔時原樣寫回，避免被表格編輯洗掉。
_dict_extra: dict[str, dict[str, Any]] = {}


def _load_table(dict_path: str) -> tuple[list[list[Any]], str]:
    """把詞典讀進表格。回傳 (rows, 狀態訊息)。"""
    path = _resolve_dict_path(dict_path)
    if not path.exists():
        return [], f"找不到 `{path}`，表格是空的。編好後按「儲存」就會建立這個檔案。"
    try:
        specs, extra = terms_mod.load_specs(path)
    except Exception as exc:
        log.warning("Term dictionary failed to parse: %s", exc)
        return [], f"ERROR: 詞典解析失敗（{exc}）。修好 JSON 再載入，以免存檔覆蓋掉內容。"
    _dict_extra[str(path)] = extra
    return terms_mod.specs_to_rows(specs), f"已載入 {len(specs)} 條規則 — `{path}`"


def _load_table_only(dict_path: str) -> list[list[Any]]:
    return _load_table(dict_path)[0]


def _refresh_editor(dict_path: str) -> tuple[list[list[Any]], str]:
    return _load_table(dict_path)


def _add_row(rows: list[list[Any]] | None) -> list[list[Any]]:
    return list(rows or []) + [["", "", False, False, ""]]


def _quick_add(
    src: str, dst: str, note: str, rows: list[list[Any]] | None, dict_path: str
) -> tuple[list[list[Any]], str, str, str, str, str]:
    """Append one term and save immediately.

    Adding a single correction is the common case, and making people hunt for
    the right cell in a 14-row table to do it is friction. Returns updated
    (table, editor status, dictionary summary, path, cleared src, cleared dst).
    """
    src, dst, note = src.strip(), dst.strip(), note.strip()
    if not src or not dst:
        return (
            gr.skip(),
            "ERROR: 「錯誤寫法」和「正確寫法」都要填。",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
    if src == dst:
        return (
            gr.skip(),
            f"ERROR: `{src}` 前後相同，加了也不會替換。",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    current = list(rows or [])
    if any(str(r[0]).strip() == src for r in current if r and len(r) > 0):
        return (
            gr.skip(),
            f"ERROR: `{src}` 已經在詞典裡了。要改請直接編輯下方表格。",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    current.append([src, dst, False, False, note])
    editor_msg, summary, saved_path = _save_table(current, dict_path)
    if editor_msg.startswith("ERROR"):
        return current, editor_msg, summary, saved_path, gr.skip(), gr.skip()
    return (
        _load_table(saved_path)[0],
        f"[OK] 已新增並儲存：`{src}` → `{dst}`\n\n{editor_msg}",
        summary,
        saved_path,
        "",
        "",
    )


def _save_table(rows: list[list[Any]] | None, dict_path: str) -> tuple[str, str, str]:
    """存檔。回傳 (編輯器狀態, 詞典摘要, 路徑欄)。"""
    path = _resolve_dict_path(dict_path)
    specs, problems = terms_mod.rows_to_specs(rows or [])

    if not specs and problems:
        return (
            "ERROR: 沒有任何有效規則，未存檔。\n\n" + "\n".join(f"- {p}" for p in problems),
            gr.skip(),
            str(path),
        )

    try:
        saved = terms_mod.save_specs(path, specs, _dict_extra.get(str(path)))
    except Exception as exc:
        log.warning("Failed to save term dictionary: %s", exc)
        return f"ERROR: 存檔失敗（{exc}）", gr.skip(), str(path)

    msg = [f"[OK] 已存檔 {len(specs)} 條規則 — `{saved}`", "", f"備份：`{saved.name}.bak`"]
    if problems:
        msg += ["", "**注意**"] + [f"- {p}" for p in problems]
    return "\n".join(msg), _reload_dict(str(saved)), str(saved)


# --------------------------- transcription pipeline ------------------------ #


def _transcribe(
    audio_path: str | None,
    engine_choice: str,
    formats: list[str],
    diarize: bool,
    word_timestamps: bool,
    num_speakers: int,
    beam_size: int,
    enable_fix: bool = True,
    dict_path: str = "",
    name_map_raw: str = "",
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str | None, str | None, str | None, str | None, dict | None]:
    """Core callback. Returns (status, preview, srt, txt, vtt, json, state)."""
    if not audio_path:
        return "ERROR: 請先上傳音檔。", "", None, None, None, None, None

    # Speaker labelling needs fine-grained lines: faster-whisper's own segments
    # average ~25 s on meeting audio, long enough to span several speakers, and
    # each line can only be attributed to one. Word timestamps cut that to ~5 s
    # and cost nothing on this engine, so diarization always turns them on.
    if diarize:
        word_timestamps = True

    progress(0.02, desc="偵測 GPU...")
    info = GPUProfiler.detect()
    try:
        spec = _resolve_spec(info, engine_choice)
    except InsufficientVRAMError as exc:
        return f"ERROR: VRAM 不足：{exc}", "", None, None, None, None, None

    wav_path: Path | None = None
    try:
        progress(0.08, desc="音檔前處理 (16 kHz / mono)...")
        wav_path, duration = AudioConverter.convert(audio_path)

        progress(0.15, desc=f"載入引擎 ({spec.kind.value})...")
        try:
            engine = _get_or_build_engine(spec)
            engine.load()
        except InsufficientVRAMError as exc:
            # HF path ran out of VRAM -> auto-downgrade to faster-whisper.
            log.warning("Downgrading to Faster-Whisper: %s", exc)
            fw_spec = EngineRouter.select(info, prefer=EngineKind.FASTER_WHISPER)
            engine = _get_or_build_engine(fw_spec)
            engine.load()
            spec = fw_spec

        progress(0.35, desc=f"轉錄中 (音檔長度 {duration:.1f} 秒)...")
        # beam_size only applies to the Faster-Whisper engine; HF pipeline
        # ignores it via its own generate_kwargs path.
        transcribe_kwargs: dict = {"word_timestamps": word_timestamps}
        vocab: list[str] = []
        if spec.kind is EngineKind.FASTER_WHISPER:
            transcribe_kwargs["beam_size"] = int(beam_size)
            # Bias decoding toward known names/jargon before it goes wrong,
            # rather than only patching the output afterwards.
            vocab = terms_mod.load_vocabulary(_resolve_dict_path(dict_path))
            transcribe_kwargs["hotwords"] = terms_mod.as_hotwords(vocab)
        segments = engine.transcribe(wav_path, **transcribe_kwargs)

        if not segments:
            return (
                "WARN: 轉錄結果為空 (可能是靜音或音量太小)",
                "",
                None,
                None,
                None,
                None,
                None,
            )

        raw_texts = [seg.text for seg in segments]

        speakers: list[str] = []
        diar_note = ""
        if diarize:
            progress(0.6, desc=f"語者標註中 ({len(segments)} 段，長音檔請耐心等候)...")
            # Reuse the 16 kHz mono WAV the ASR stage already produced rather
            # than decoding the source a second time.
            speakers, diar_note = _run_diarization(wav_path, segments, num_speakers)

        # State keeps the RAW labels; renaming happens at output time so the
        # user can adjust the map afterwards and just re-apply.
        raw_speakers = list(speakers)
        name_map = diar_mod.parse_name_map(name_map_raw)
        if speakers and name_map:
            speakers = diar_mod.rename_speakers(speakers, name_map)
            for raw_label, renamed in name_map.items():
                diar_note = diar_note.replace(raw_label, renamed)

        fix_note = ""
        if enable_fix:
            progress(0.88, desc="套用專有名詞校正...")
            fix_note = _apply_corrections(segments, dict_path)

        progress(0.9, desc="輸出字幕檔...")
        meta = {
            "engine": spec.kind.value,
            "compute_type": spec.compute_type,
            "duration_sec": round(duration, 2),
        }
        if speakers:
            meta["speakers"] = len({s for s in speakers if s})

        outputs = _write_outputs(segments, formats, meta=meta, speakers=speakers)

        status = (
            f"[OK] 完成！{len(segments)} 段 / {sum(len(s.text) for s in segments)} 字 "
            f"/ 引擎：{spec.kind.value} ({spec.compute_type})"
        )
        if vocab:
            status += f" / 詞彙表 {len(vocab)} 個詞"
        for note in (diar_note, fix_note):
            if note:
                status += "\n\n" + note
        return (
            status,
            _preview(segments, speakers),
            outputs.get("srt"),
            outputs.get("txt"),
            outputs.get("vtt"),
            outputs.get("json"),
            {
                "segments": segments,
                "raw": raw_texts,
                "speakers": raw_speakers,
                "meta": meta,
                "formats": formats,
            },
        )

    except (ModelLoadError, TranscriptionError) as exc:
        return f"ERROR: 引擎錯誤：{exc}", "", None, None, None, None, None
    except Exception as exc:  # pragma: no cover
        log.exception("Unexpected UI error")
        return f"ERROR: 未預期錯誤：{exc}", "", None, None, None, None, None
    finally:
        if wav_path is not None:
            AudioConverter.cleanup(wav_path)


_OUTPUT_DIRS: list[Path] = []
_OUTPUT_DIRS_LOCK = threading.Lock()


def _cleanup_output_dirs() -> None:
    with _OUTPUT_DIRS_LOCK:
        while _OUTPUT_DIRS:
            d = _OUTPUT_DIRS.pop()
            shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_output_dirs)


def _write_outputs(
    segments: list[TimestampedSegment],
    formats: list[str],
    meta: dict[str, Any],
    speakers: list[str] | None = None,
) -> dict[str, str]:
    """Persist requested formats to a temp dir so Gradio can serve them.

    Directories are tracked and deleted on process exit via ``atexit``. Also
    trims to the last 5 sessions so a long-running server doesn't fill disk.
    Gradio can serve concurrent requests -- guard append/pop with a lock so
    racing ``len()``/``pop(0)`` calls don't raise or delete the wrong dir.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="taigi_asr_"))
    with _OUTPUT_DIRS_LOCK:
        _OUTPUT_DIRS.append(out_dir)
        while len(_OUTPUT_DIRS) > 5:
            old = _OUTPUT_DIRS.pop(0)
            shutil.rmtree(old, ignore_errors=True)
    # Per-format speaker treatment: SRT prefixes a label only where the
    # speaker changes (SRT has no speaker syntax); VTT uses its native
    # <v Speaker> voice tag on every cue; JSON carries a separate "speaker"
    # field so downstream code never parses labels out of transcript text.
    subtitle_segs = _compose(segments, speakers, on_change_only=True)

    paths: dict[str, str] = {}
    if "SRT" in formats:
        p = out_dir / "transcript.srt"
        p.write_text(to_srt(subtitle_segs), encoding="utf-8")
        paths["srt"] = str(p)
    if "TXT" in formats:
        p = out_dir / "transcript.txt"
        body = _speaker_txt(segments, speakers) if speakers else to_txt(segments)
        p.write_text(body, encoding="utf-8")
        paths["txt"] = str(p)
    if "VTT" in formats:
        p = out_dir / "transcript.vtt"
        p.write_text(to_vtt(segments, speakers=speakers), encoding="utf-8")
        paths["vtt"] = str(p)
    if "JSON" in formats:
        p = out_dir / "transcript.json"
        p.write_text(to_json(segments, meta=meta, speakers=speakers), encoding="utf-8")
        paths["json"] = str(p)
    return paths


# --------------------------- Blocks UI ------------------------------------- #


# --------------------------- visual design --------------------------------- #
#
# 「圖面」— an engineering-blueprint look for a factory QC tool: cool drafting
# paper, near-black linework, one ultramarine accent. Hairlines, square
# corners, mono numerals. The previous layout is frozen in
# gradio_app_classic.py (launcher --classic).

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans TC"), "Noto Sans TC", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Consolas", "monospace"],
)

# Theme manager, injected via Blocks(head=...) — a real <head> script, since
# Gradio 6 does not reliably execute the js= parameter. The user's choice is
# stored in localStorage and enforced with a MutationObserver so Gradio's own
# system-preference sync cannot override it. The toggle chip is created here
# in JS rather than as a gr.HTML button because gr.HTML content is inserted
# via innerHTML, where inline handlers and <script> do not run.
THEME_HEAD = """
<script>
(function () {
  var KEY = "taigi-theme";
  function chosenDark() {
    var stored = localStorage.getItem(KEY);
    if (stored !== null) return stored === "dark";
    // No saved preference: honour Gradio's ?__theme= URL parameter so a
    // shared link (or a headless screenshot run) can pick the palette.
    return new URLSearchParams(window.location.search).get("__theme") === "dark";
  }
  function apply() {
    var dark = chosenDark();
    [document.documentElement, document.body].forEach(function (el) {
      if (el) el.classList.toggle("dark", dark);
    });
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = dark ? "◑ 淺色" : "◐ 深色";
  }
  function arm() {
    apply();
    var obs = new MutationObserver(apply);
    [document.documentElement, document.body].forEach(function (el) {
      if (el) obs.observe(el, { attributes: true, attributeFilter: ["class"] });
    });
    var timer = setInterval(function () {
      var mast = document.querySelector(".masthead");
      if (!mast) return;
      clearInterval(timer);
      if (document.getElementById("theme-toggle")) return;
      var btn = document.createElement("button");
      btn.id = "theme-toggle";
      btn.type = "button";
      btn.onclick = function () {
        localStorage.setItem(KEY, chosenDark() ? "light" : "dark");
        apply();
      };
      mast.appendChild(btn);
      apply();
    }, 200);
  }
  if (document.body) { arm(); }
  else { window.addEventListener("DOMContentLoaded", arm); }
})();
</script>
"""

CSS = """
/* 淺色：日間製圖紙 */
:root {
  --paper:   #eef1f4;
  --card:    #f9fafc;
  --field:   #ffffff;
  --ink:     #15181d;
  --ink-2:   #5c6570;
  --line:    #cdd5de;
  --line-2:  #aab6c2;
  --acc:     #2743c7;
  --acc-2:   #1b2f96;
  --ok:      #1f6b4a;
  --radius:  2px;
}

/* 深色：夜間製圖桌 — 同一套線稿邏輯，墨底亮線 */
.dark {
  --paper:   #14171c;
  --card:    #1b1f26;
  --field:   #10141a;
  --ink:     #e5e9ee;
  --ink-2:   #98a2af;
  --line:    #2c333d;
  --line-2:  #3e4854;
  --acc:     #5b74ec;
  --acc-2:   #8296f2;
  --ok:      #4ca97b;
}

body, .gradio-container {
  background: var(--paper) !important;
  color: var(--ink);
}

/* Everything above is var-driven, so flipping the palette is all dark mode
   needs — plus remapping Gradio's own text variables onto the same vars so
   its internals follow whichever palette is active. */
.dark, .dark .gradio-container {
  --body-text-color: var(--ink) !important;
  --body-text-color-subdued: var(--ink-2) !important;
  --block-title-text-color: var(--ink) !important;
  --block-label-text-color: var(--ink-2) !important;
  --block-info-text-color: var(--ink-2) !important;
  --checkbox-label-text-color: var(--ink) !important;
  --input-placeholder-color: var(--ink-2) !important;
  --color-accent: var(--acc) !important;
  color: var(--ink);
}

/* ---- theme toggle chip (created by the head script) ---- */
.masthead { position: relative; }
#theme-toggle {
  position: absolute; top: 10px; right: 0;
  font-family: var(--font-mono); font-size: 12px; letter-spacing: .08em;
  color: var(--ink-2); background: transparent;
  border: 1px solid var(--line-2); border-radius: var(--radius);
  padding: 5px 12px; cursor: pointer; transition: color .15s, border-color .15s;
}
#theme-toggle:hover { color: var(--acc); border-color: var(--acc); }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
gradio-app { background: var(--paper) !important; }

/* Flatten Gradio chrome into hairlines and paper */
.block, .form, .panel, fieldset {
  background: var(--card) !important;
  border-color: var(--line) !important;
  border-radius: var(--radius) !important;
  box-shadow: none !important;
}
.gap { gap: 10px !important; }

/* ---- masthead ---- */
.masthead { padding: 6px 0 2px; border: none !important; background: transparent !important; }
.masthead .kanban {
  font-size: 30px; font-weight: 700; letter-spacing: .04em; line-height: 1.15;
  display: flex; align-items: center; gap: 12px;
}
.masthead .kanban::before {
  content: ""; width: 14px; height: 14px; background: transparent;
  border: 3.5px solid var(--acc); display: inline-block; flex: none;
}
.masthead .sub {
  margin-top: 6px; color: var(--ink-2);
  font-family: var(--font-mono); font-size: 12px; letter-spacing: .08em;
}
.hwline {
  border: none !important; background: transparent !important; padding: 0;
  color: var(--ink-2); font-family: var(--font-mono); font-size: 12px;
  letter-spacing: .05em;
}

/* ---- numbered step rules ---- */
.step { border: none !important; background: transparent !important; padding: 0; }
.step p {
  display: flex; align-items: baseline; gap: 10px; margin: 14px 0 2px !important;
  padding-bottom: 5px; border-bottom: 1px solid var(--line-2);
  font-weight: 600; font-size: 15px; letter-spacing: .03em;
}
.step p code {
  font-family: var(--font-mono); font-weight: 600; font-size: 12px;
  color: var(--acc); background: none !important; border: none !important;
  padding: 0 !important; letter-spacing: .1em;
}

/* ---- controls ---- */
button.primary, button#run-btn {
  background: var(--acc) !important; color: #fff !important;
  border: 1px solid var(--acc) !important; border-radius: var(--radius) !important;
  font-weight: 600; letter-spacing: .12em; box-shadow: none !important;
  /* No transition here: Gradio's theme sync and our observer can briefly
     tug-of-war over the dark class, and a background transition restarted on
     every flip freezes the button at its pre-toggle colour. */
  transition: none;
}
button.primary:hover, button#run-btn:hover {
  background: var(--ink) !important; border-color: var(--ink) !important;
  color: var(--paper) !important;  /* dark mode: ink flips light, paper dark */
}
button#run-btn { font-size: 17px; padding: 14px 0; }
button.secondary {
  background: transparent !important; color: var(--ink) !important;
  border: 1px solid var(--line-2) !important; border-radius: var(--radius) !important;
  box-shadow: none !important;
}
button.secondary:hover { border-color: var(--ink) !important; }

input[type="checkbox"], input[type="radio"] { accent-color: var(--acc); }
input[type="text"], input[type="number"], textarea, select {
  background: var(--field) !important; border: 1px solid var(--line) !important;
  color: var(--ink) !important;
  border-radius: var(--radius) !important; box-shadow: none !important;
}
input:focus, textarea:focus { border-color: var(--acc) !important; box-shadow: 0 0 0 1px var(--acc) !important; }

/* ---- result panel ---- */
#preview textarea {
  font-family: var(--font-mono) !important; font-size: 13px; line-height: 1.75;
  background: var(--field) !important;
}
#status-line { min-height: 20px; }
#status-line p, #status-line li { font-size: 13.5px; }

/* download slots: quiet until filled */
#dl-row .block { border-style: dashed !important; }
#dl-row .file-preview { border-style: solid !important; }

/* ---- accordions as filing-cabinet drawers ---- */
.accordion, details {
  border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
}
.label-wrap span { font-weight: 600; letter-spacing: .02em; }

/* small print */
span[data-testid="block-info"] { color: var(--ink-2) !important; }
.footer-note {
  border: none !important; background: transparent !important;
  color: var(--ink-2); font-family: var(--font-mono); font-size: 11.5px;
  letter-spacing: .06em; margin-top: 10px;
}
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    """Construct the Blocks layout. Separated so tests can import without launching."""
    info = GPUProfiler.detect()
    initial_spec = EngineRouter.select(info)

    hw = (
        f"{info.name} · {info.vram_gb:.1f} GB · "
        f"{initial_spec.kind.value} ({initial_spec.compute_type}, batch={initial_spec.batch_size})"
        if info.cuda_available
        else "CPU ONLY · 無 CUDA GPU，速度會慢很多"
    )

    with gr.Blocks(title="台灣華語逐字稿 · Taigi ASR", css=CSS, head=THEME_HEAD) as demo:
        gr.HTML(
            '<div class="masthead">'
            '<div class="kanban">台灣華語逐字稿</div>'
            '<div class="sub">BREEZE-ASR-26 · 國語台語英文夾雜 · 語者標註 · 詞典校正</div>'
            "</div>"
        )
        gr.HTML(f'<div class="hwline">{hw}</div>')

        with gr.Row(equal_height=False):
            # ---------------- 左欄：操作 ---------------- #
            with gr.Column(scale=5, min_width=380):
                gr.Markdown("`01` 音檔", elem_classes="step")
                audio_input = gr.Audio(
                    label="拖放或點擊上傳（m4a / mp3 / wav / mp4 / mov / mkv / flac / ogg / webm）",
                    type="filepath",
                    sources=["upload"],
                )

                gr.Markdown("`02` 轉錄模型", elem_classes="step")
                # HuggingFace stays selectable — colleagues run different
                # hardware, and CTranslate2 has no Metal backend, so a Mac is a
                # real case where the HF path is the only GPU option. It fails
                # fast with an actionable message where torchcodec is broken.
                engine_choice = gr.Radio(
                    choices=ENGINE_CHOICES,
                    value=ENGINE_FW,
                    label="引擎",
                    info=(
                        "Auto 等同 Faster-Whisper — 實測快約 17 倍、長音檔不會 OOM，"
                        "一般情況不用改。裝了語者標註後 HuggingFace 會失效（相依衝突）。"
                    ),
                )

                gr.Markdown("`03` 發言者", elem_classes="step")
                _diar_ready = diar_mod.available()
                with gr.Group():
                    diarize_cb = gr.Checkbox(
                        value=False,
                        label="標記發言者（Speaker Diarization）",
                        info=(
                            "轉錄完自動標註，輸出直接帶 [SPEAKER_00]。"
                            "需要 HF_TOKEN 與 pyannote 模型授權。"
                            if _diar_ready
                            else '未安裝 pyannote.audio — 請執行 pip install "pyannote.audio>=4.0"。'
                        ),
                        interactive=_diar_ready,
                    )
                    num_spk = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
                        label="已知語者人數（0 = 自動判斷）",
                        visible=False,
                    )
                    name_map_box = gr.Textbox(
                        lines=3,
                        label="語者改名（一行一組，轉錄完再填也行）",
                        placeholder="SPEAKER_00=主持人\nSPEAKER_01=提問者",
                        info="填好後按詞典區的「重新套用」即可換成真名，不必重跑轉錄或語者標註。",
                        visible=False,
                    )
                    word_ts = gr.Checkbox(
                        value=False,
                        label="詞級時間軸（句子從 ~25 秒切成 ~5 秒）",
                        info="Faster-Whisper 走這條路不用多花時間。勾語者標註時自動啟用。",
                    )

                with gr.Accordion("進階選項", open=False):
                    formats = gr.CheckboxGroup(
                        choices=["SRT", "TXT", "VTT", "JSON"],
                        value=["SRT", "TXT"],
                        label="輸出格式",
                    )
                    beam_slider = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=5,
                        step=1,
                        label="Beam size（大 = 準但慢，吃 VRAM）",
                        info="建議 5-10；僅影響 Faster-Whisper。",
                    )

                gr.Markdown("`04` 執行", elem_classes="step")
                run_btn = gr.Button(
                    "開始轉錄", variant="primary", size="lg", elem_id="run-btn"
                )

            # ---------------- 右欄：結果 ---------------- #
            with gr.Column(scale=7, min_width=420):
                gr.Markdown("`→` 結果", elem_classes="step")
                status_md = gr.Markdown("", elem_id="status-line")
                preview_md = gr.Textbox(
                    label="預覽（與 TXT 下載相同格式）",
                    lines=18,
                    interactive=False,
                    elem_id="preview",
                )
                with gr.Row(elem_id="dl-row"):
                    dl_srt = gr.File(label="SRT 字幕", interactive=False)
                    dl_txt = gr.File(label="TXT 逐字稿", interactive=False)
                    dl_vtt = gr.File(label="VTT 字幕", interactive=False)
                    dl_json = gr.File(label="JSON", interactive=False)

        def _on_diarize_toggle(on: bool):
            # Diarization forces word timestamps on — reflect that in the box
            # instead of letting it sit unchecked while the run enables it.
            # Unchecking must not leave the value forced on, so only the
            # enabling direction sets a value.
            word_update = (
                gr.update(value=True, interactive=False) if on else gr.update(interactive=True)
            )
            return word_update, gr.update(visible=on), gr.update(visible=on)

        diarize_cb.change(
            fn=_on_diarize_toggle, inputs=[diarize_cb], outputs=[word_ts, num_spk, name_map_box]
        )

        last_result = gr.State(None)

        _default_dict = terms_mod.find_default_dict()
        _initial_path = str(_default_dict) if _default_dict else ""
        _initial_rows, _initial_msg = _load_table(_initial_path)

        gr.Markdown("`05` 詞典", elem_classes="step")
        with gr.Accordion("專有名詞校正 — 詞彙表 · 修正規則", open=False):
            enable_fix = gr.Checkbox(
                value=True,
                label="轉錄後自動套用詞典校正",
                info="每次轉錄都會重新讀取詞典，改完不用重開 UI。",
            )

            gr.Markdown(
                "### 常用詞彙（人名／術語／廠商）\n"
                "**一行一個，只要打正確的寫法**——不必知道模型會聽錯成什麼。"
                "這些詞會在轉錄時餵給解碼器，讓它一開始就偏向這些寫法。"
            )
            with gr.Row():
                vocab_box = gr.Textbox(
                    value=_load_vocab_text(_initial_path),
                    lines=6,
                    label="詞彙表",
                    placeholder="一行一個，例如：\n講者姓名\n單位或公司名\n你這個領域的專有名詞",
                    scale=4,
                )
                with gr.Column(scale=1):
                    vocab_save_btn = gr.Button("儲存詞彙表", variant="primary")
                    vocab_reload_btn = gr.Button("重新載入")
            vocab_status = gr.Markdown("")

            gr.Markdown("---")

            # 快速新增：單筆修正是最常見的操作，不該逼人去表格裡找空格。
            gr.Markdown(
                "### 修正規則（知道錯成什麼時用這個）\n"
                "詞彙表管不到的、或模型固定錯成某個寫法時，用這裡做事後替換。"
            )
            with gr.Row():
                quick_src = gr.Textbox(
                    label="辨識錯的寫法", placeholder="模型聽成什麼", scale=3
                )
                quick_dst = gr.Textbox(
                    label="正確寫法", placeholder="應該要是什麼", scale=3
                )
                quick_note = gr.Textbox(label="備註（可空白）", scale=2)
                quick_btn = gr.Button("新增並儲存", variant="primary", scale=2)

            gr.Markdown("### 全部規則（可直接在表格裡改，改完按儲存）")
            dict_table = gr.Dataframe(
                value=_initial_rows,
                headers=terms_mod.ROW_HEADERS,
                datatype=terms_mod.ROW_DATATYPES,
                column_count=len(terms_mod.ROW_HEADERS),
                row_count=max(1, len(_initial_rows)),
                type="array",
                interactive=True,
                wrap=True,
                label="詞典規則",
            )
            with gr.Row():
                add_row_btn = gr.Button("新增空白列")
                revert_btn = gr.Button("放棄修改，重新載入")
                save_btn = gr.Button("儲存詞典檔", variant="primary")
            editor_status = gr.Markdown(_initial_msg)
            gr.Markdown(
                "**正則**打勾時，「錯誤寫法」會當成正規表示式"
                "（例如 `\\b縮寫\\b` 只比對完整字詞、`甲 ?乙` 容許中間有空格）；"
                "沒打勾就是一般文字比對。"
                "存檔前會自動備份成 `.bak`。"
            )

            with gr.Accordion("詞典檔位置 / 匯入（通常不用動）", open=False):
                with gr.Row():
                    dict_path_box = gr.Textbox(
                        value=_initial_path,
                        label="詞典路徑",
                        placeholder=str(terms_mod.PROJECT_ROOT / terms_mod.DEFAULT_NAME),
                        scale=4,
                    )
                    reload_btn = gr.Button("重新載入", scale=1)
                    template_btn = gr.Button("產生詞典範本", scale=1)
                dict_upload = gr.File(
                    label="或改用其他詞典檔（.json）",
                    file_types=[".json"],
                    type="filepath",
                )
                dict_status = gr.Markdown(_reload_dict(_initial_path))

            reapply_btn = gr.Button(
                "重新套用詞典與語者名稱到上次的轉錄結果（不必重跑轉錄／語者標註）",
                variant="secondary",
            )
            gr.Markdown(
                "*每次都會從原始辨識文字與原始語者標籤重新套用，所以刪掉規則、"
                "改掉或清空語者改名都會如實反映。*"
            )

        reload_btn.click(
            fn=_reload_dict, inputs=[dict_path_box], outputs=[dict_status]
        ).then(
            fn=_refresh_editor,
            inputs=[dict_path_box],
            outputs=[dict_table, editor_status],
        )
        template_btn.click(
            fn=_make_template, inputs=[dict_path_box], outputs=[dict_status, dict_path_box]
        ).then(
            fn=_refresh_editor,
            inputs=[dict_path_box],
            outputs=[dict_table, editor_status],
        )
        dict_upload.change(
            fn=_on_dict_upload,
            inputs=[dict_upload],
            outputs=[dict_status, dict_path_box, dict_table],
        )

        vocab_save_btn.click(
            fn=_save_vocab,
            inputs=[vocab_box, dict_path_box],
            outputs=[vocab_status, dict_path_box],
        )
        vocab_reload_btn.click(
            fn=_load_vocab_text, inputs=[dict_path_box], outputs=[vocab_box]
        )

        quick_btn.click(
            fn=_quick_add,
            inputs=[quick_src, quick_dst, quick_note, dict_table, dict_path_box],
            outputs=[
                dict_table,
                editor_status,
                dict_status,
                dict_path_box,
                quick_src,
                quick_dst,
            ],
        )
        quick_dst.submit(
            fn=_quick_add,
            inputs=[quick_src, quick_dst, quick_note, dict_table, dict_path_box],
            outputs=[
                dict_table,
                editor_status,
                dict_status,
                dict_path_box,
                quick_src,
                quick_dst,
            ],
        )

        add_row_btn.click(fn=_add_row, inputs=[dict_table], outputs=[dict_table])
        revert_btn.click(
            fn=_refresh_editor,
            inputs=[dict_path_box],
            outputs=[dict_table, editor_status],
        )
        save_btn.click(
            fn=_save_table,
            inputs=[dict_table, dict_path_box],
            outputs=[editor_status, dict_status, dict_path_box],
        )

        run_btn.click(
            fn=_transcribe,
            inputs=[
                audio_input,
                engine_choice,
                formats,
                diarize_cb,
                word_ts,
                num_spk,
                beam_slider,
                enable_fix,
                dict_path_box,
                name_map_box,
            ],
            outputs=[
                status_md,
                preview_md,
                dl_srt,
                dl_txt,
                dl_vtt,
                dl_json,
                last_result,
            ],
        )

        reapply_btn.click(
            fn=_reapply_terms,
            inputs=[last_result, enable_fix, dict_path_box, name_map_box],
            outputs=[status_md, preview_md, dl_srt, dl_txt, dl_vtt, dl_json],
        )

        gr.Markdown(
            "CODE · MIT — MODELS · Breeze-ASR-26 (Apache-2.0) · "
            "pyannote speaker-diarization (MIT / CC-BY-4.0, gated)",
            elem_classes="footer-note",
        )

    return demo
