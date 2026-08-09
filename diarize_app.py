"""
diarize_app.py — 第二階段：Speaker Diarization 後處理 UI

放置位置：breeze-asr-taigi 專案根目錄（與 start.bat / start.sh 同層）

用途：
    輸入「原始音檔」+「Breeze ASR 產出的 SRT 或 JSON」，
    以 pyannote/speaker-diarization-3.1 做語者分群，
    再依時間區間重疊度把 speaker 標籤貼回逐字稿，
    輸出帶語者標示的 SRT / TXT / JSON。

啟動：
    python diarize_app.py          # http://127.0.0.1:7861

前置：
    pip install "pyannote.audio>=3.1" soundfile
    並到 HuggingFace 同意以下兩個模型的授權條款：
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/segmentation-3.0
    設定環境變數 HF_TOKEN=hf_xxxx
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import gradio as gr
import torch

from taigi_asr import diarize as diar

DIAR_MODEL = "pyannote/speaker-diarization-3.1"
OUTPUT_DIR = Path("outputs_diarized")
OUTPUT_DIR.mkdir(exist_ok=True)

# 全域快取，避免每次點按鈕都重新載入模型（載入約 5-10 秒）
_PIPELINE = None
_PIPELINE_DEVICE = None


# --------------------------------------------------------------------------
# 資料結構
# --------------------------------------------------------------------------
@dataclass
class Seg:
    start: float
    end: float
    text: str
    speaker: str | None = None


# --------------------------------------------------------------------------
# 逐字稿解析（SRT / VTT / JSON）
# --------------------------------------------------------------------------
_TS = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_TS_LINE = re.compile(rf"{_TS}\s*-->\s*{_TS}")


def _hms_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(raw: str) -> list[Seg]:
    """同時吃得下 SRT 與 WebVTT。"""
    segs: list[Seg] = []
    raw = raw.replace("\r\n", "\n").replace("\ufeff", "")
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        ts_idx, m = None, None
        for i, ln in enumerate(lines):
            m = _TS_LINE.search(ln)
            if m:
                ts_idx = i
                break
        if m is None or ts_idx is None:
            continue
        g = m.groups()
        body = " ".join(lines[ts_idx + 1:]).strip()
        if body:
            segs.append(Seg(_hms_to_sec(*g[:4]), _hms_to_sec(*g[4:]), body))
    return segs


def parse_json(raw: str) -> list[Seg]:
    """容忍多種 JSON 結構：list、或 dict 底下的 segments / results / data。"""
    data = json.loads(raw)
    if isinstance(data, dict):
        for key in ("segments", "results", "data", "chunks"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("無法辨識的 JSON 結構，請改用 SRT。")

    segs: list[Seg] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # 支援 {"timestamp": [s, e]} 的 HF pipeline 格式
        if "timestamp" in item and isinstance(item["timestamp"], (list, tuple)):
            start, end = item["timestamp"][0], item["timestamp"][1]
        else:
            start = item.get("start", item.get("start_time"))
            end = item.get("end", item.get("end_time"))
        text = item.get("text", item.get("content", ""))
        if start is None or end is None:
            continue
        text = str(text).strip()
        if text:
            segs.append(Seg(float(start), float(end), text))
    return segs


def load_transcript(path: str) -> list[Seg]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return parse_json(raw)
    if suffix in (".srt", ".vtt"):
        return parse_srt(raw)
    # 未知副檔名：先試 JSON 再試 SRT
    try:
        return parse_json(raw)
    except Exception:
        return parse_srt(raw)


# --------------------------------------------------------------------------
# 音檔前處理：統一轉 16 kHz mono wav（pyannote 對 m4a/mp4 支援不穩）
# --------------------------------------------------------------------------
def to_wav16k(src: str) -> str:
    if Path(src).suffix.lower() == ".wav":
        return src
    # 優先用專案自帶的 AudioConverter，行為與 ASR 階段一致
    try:
        from taigi_asr.audio import AudioConverter  # type: ignore

        wav, _duration = AudioConverter.convert(src)
        return str(wav)
    except Exception:
        pass

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請先安裝或改用 .wav 檔。")
    dst = Path(tempfile.mkdtemp()) / (Path(src).stem + "_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return str(dst)


# --------------------------------------------------------------------------
# Diarization
# --------------------------------------------------------------------------
def get_pipeline(token: str, device: str):
    """Delegates to :mod:`taigi_asr.diarize` — one implementation, shared with
    the transcription UI so the two cannot drift apart."""
    return diar.get_pipeline(token, device)


load_waveform = diar.load_waveform


def run_diarization(
    wav: str, token: str, device: str, num_spk: int, min_spk: int, max_spk: int
) -> list[tuple[float, float, str]]:
    return diar.diarize(
        wav,
        token=token,
        device=device,
        num_speakers=num_spk,
        min_speakers=min_spk,
        max_speakers=max_spk,
    )


def assign_speakers(segs: list[Seg], turns: list[tuple[float, float, str]]) -> list[Seg]:
    """對每個 ASR 句子，取與其時間區間重疊最久的 speaker turn。"""
    labels = diar.assign([(s.start, s.end) for s in segs], turns)
    for seg, label in zip(segs, labels, strict=True):
        seg.speaker = label
    return segs


# --------------------------------------------------------------------------
# 輸出格式
# --------------------------------------------------------------------------
def sec_to_srt(t: float) -> str:
    t = max(t, 0.0)
    ms = int(round((t - int(t)) * 1000))
    total = int(t)
    if ms == 1000:
        total, ms = total + 1, 0
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d},{ms:03d}"


def parse_name_map(raw: str) -> dict[str, str]:
    """解析 'SPEAKER_00=主持人' 這種每行一組的對應表。"""
    mapping: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and v.strip():
            mapping[k.strip()] = v.strip()
    return mapping


def to_speaker_srt(segs: list[Seg], names: dict[str, str]) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        sp = names.get(s.speaker or "", s.speaker)
        out.append(
            f"{i}\n{sec_to_srt(s.start)} --> {sec_to_srt(s.end)}\n[{sp}] {s.text}\n"
        )
    return "\n".join(out)


def to_speaker_txt(segs: list[Seg], names: dict[str, str], merge: bool = True) -> str:
    lines: list[str] = []
    prev_sp: str | None = None
    buf: list[str] = []
    start = 0.0

    def flush():
        if prev_sp is not None:
            ts = sec_to_srt(start)[:8]
            lines.append(f"[{ts}] {prev_sp}：{''.join(buf)}")

    for s in segs:
        sp = names.get(s.speaker or "", s.speaker) or "SPEAKER_UNK"
        if merge and sp == prev_sp:
            buf.append(s.text)
        else:
            flush()
            prev_sp, buf, start = sp, [s.text], s.start
    flush()
    return "\n".join(lines)


def to_speaker_json(segs: list[Seg], names: dict[str, str]) -> str:
    payload = []
    for s in segs:
        d = asdict(s)
        d["speaker"] = names.get(s.speaker or "", s.speaker)
        d["speaker_id"] = s.speaker
        payload.append(d)
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Gradio 主流程
# --------------------------------------------------------------------------
def process(
    audio_path,
    transcript_path,
    token,
    device,
    num_spk,
    min_spk,
    max_spk,
    name_map_raw,
    merge_same,
    progress=gr.Progress(),
):
    if not audio_path:
        raise gr.Error("請上傳原始音檔（Diarization 需要音訊，不能只給字幕）。")
    if not transcript_path:
        raise gr.Error("請上傳 Breeze 產出的 SRT 或 JSON。")

    token = (token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise gr.Error("缺少 HF_TOKEN。請填入欄位或設定環境變數。")

    if device == "cuda" and not torch.cuda.is_available():
        gr.Warning("偵測不到 CUDA，自動改用 CPU。")
        device = "cpu"

    progress(0.05, desc="解析逐字稿…")
    segs = load_transcript(transcript_path)
    if not segs:
        raise gr.Error("逐字稿解析後為空，請確認檔案格式。")

    progress(0.15, desc="轉檔 16kHz mono…")
    wav = to_wav16k(audio_path)

    progress(0.25, desc="執行語者分群（長音檔請耐心等候）…")
    turns = run_diarization(wav, token, device, num_spk, min_spk, max_spk)

    progress(0.85, desc="對齊時間軸…")
    segs = assign_speakers(segs, turns)
    names = parse_name_map(name_map_raw)

    srt_text = to_speaker_srt(segs, names)
    txt_text = to_speaker_txt(segs, names, merge=merge_same)
    json_text = to_speaker_json(segs, names)

    stem = Path(transcript_path).stem
    srt_path = OUTPUT_DIR / f"{stem}.speaker.srt"
    txt_path = OUTPUT_DIR / f"{stem}.speaker.txt"
    json_path = OUTPUT_DIR / f"{stem}.speaker.json"
    srt_path.write_text(srt_text, encoding="utf-8")
    txt_path.write_text(txt_text, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")

    detected = sorted({s.speaker for s in segs if s.speaker})
    stats = "\n".join(
        f"  {sp} → {sum(1 for s in segs if s.speaker == sp)} 句" for sp in detected
    )
    info = (
        f"句數：{len(segs)}\n"
        f"Speaker turns：{len(turns)}\n"
        f"偵測語者數：{len(detected)}\n{stats}\n\n"
        f"提示：把上方對應表填成 `{detected[0] if detected else 'SPEAKER_00'}=主持人` 再跑一次即可換成真名。"
    )

    progress(1.0, desc="完成")
    return txt_text, srt_text, info, [str(txt_path), str(srt_path), str(json_path)]


def build_ui():
    with gr.Blocks(title="Speaker Diarization 後處理") as demo:
        gr.Markdown(
            "## 語者標註後處理\n"
            "把 Breeze ASR 轉好的 **SRT / JSON** 加上 **原始音檔**，"
            "產出帶 speaker 標示的逐字稿。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="原始音檔（必要）", type="filepath", sources=["upload"]
                )
                transcript_in = gr.File(
                    label="Breeze 輸出的 SRT / VTT / JSON",
                    file_types=[".srt", ".vtt", ".json"],
                    type="filepath",
                )
                token_in = gr.Textbox(
                    label="HF_TOKEN",
                    type="password",
                    value=os.environ.get("HF_TOKEN", ""),
                    placeholder="hf_xxxxxxxx（已設環境變數可留空）",
                )
                device_in = gr.Radio(
                    ["cuda", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    label="運算裝置",
                )
                with gr.Accordion("語者數設定", open=True):
                    num_spk = gr.Number(
                        label="固定語者數（已知人數時填，最準）", value=0, precision=0
                    )
                    with gr.Row():
                        min_spk = gr.Number(label="最少", value=2, precision=0)
                        max_spk = gr.Number(label="最多", value=6, precision=0)
                name_map = gr.Textbox(
                    label="語者名稱對應（每行一組）",
                    lines=4,
                    placeholder="SPEAKER_00=主持人\nSPEAKER_01=提問者",
                )
                merge_same = gr.Checkbox(
                    label="TXT 合併同一語者的連續發言", value=True
                )
                run_btn = gr.Button("開始語者標註", variant="primary")

            with gr.Column(scale=2):
                info_out = gr.Textbox(label="執行資訊", lines=8)
                with gr.Tabs():
                    with gr.Tab("逐字稿 TXT"):
                        txt_out = gr.Textbox(label="", lines=22, buttons=["copy"])
                    with gr.Tab("字幕 SRT"):
                        srt_out = gr.Textbox(label="", lines=22, buttons=["copy"])
                files_out = gr.File(label="下載檔案")

        run_btn.click(
            fn=process,
            inputs=[
                audio_in, transcript_in, token_in, device_in,
                num_spk, min_spk, max_spk, name_map, merge_same,
            ],
            outputs=[txt_out, srt_out, info_out, files_out],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
