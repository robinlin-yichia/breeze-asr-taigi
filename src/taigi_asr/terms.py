"""專有名詞校正詞典 — 轉錄後把 ASR 常錯的字詞批次修正。

詞典檔（預設 terms.json，UTF-8 無 BOM）：

{
  "vocabulary": ["正確寫法", "另一個專有名詞"],
  "corrections": [
    {"from": "錯的寫法", "to": "正確寫法"},
    {"from": "正則樣式", "to": "正確寫法", "regex": true, "ignore_case": true}
  ]
}

vocabulary 是解碼期的偏向（hotwords），corrections 是事後替換——兩種機制。

尋找順序：
  1. 環境變數 TAIGI_ASR_TERMS 指定的路徑
  2. 目前工作目錄的 terms.json
  3. 專案根目錄的 terms.json
  4. 都沒有 → 從 terms.example.json 複製一份
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV_VAR = "TAIGI_ASR_TERMS"
DEFAULT_NAME = "terms.json"
EXAMPLE_NAME = "terms.example.json"

# src/taigi_asr/terms.py -> 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# 規則
# --------------------------------------------------------------------------
class Rule:
    __slots__ = ("src", "dst", "note", "pattern")

    def __init__(self, spec: dict[str, Any]) -> None:
        self.src: str = spec["from"]
        self.dst: str = spec["to"]
        self.note: str = spec.get("note", "")
        flags = re.IGNORECASE if spec.get("ignore_case") else 0
        pattern = self.src if spec.get("regex") else re.escape(self.src)
        self.pattern = re.compile(pattern, flags)

    def apply(self, text: str) -> tuple[str, int]:
        return self.pattern.subn(self.dst, text)

    def __repr__(self) -> str:
        return f"{self.src} → {self.dst}"


# --------------------------------------------------------------------------
# 載入
# --------------------------------------------------------------------------
def find_default_dict() -> Path | None:
    """依序尋找詞典檔，找不到回傳 None。

    全新 clone 只有版控的 ``terms.example.json``。第一次呼叫時把它複製成
    ``terms.json``（後者不進版控），這樣同事一裝好就有共用詞彙表，各自的修改
    也不會變成 git 衝突。讀寫一律指向真正的 terms.json，範本不會被覆蓋。
    """
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    for candidate in (Path.cwd() / DEFAULT_NAME, PROJECT_ROOT / DEFAULT_NAME):
        if candidate.exists():
            return candidate

    example = PROJECT_ROOT / EXAMPLE_NAME
    if example.exists():
        target = PROJECT_ROOT / DEFAULT_NAME
        try:
            shutil.copy2(example, target)
            log.info("Seeded %s from %s", target.name, example.name)
            return target
        except OSError as exc:
            log.warning("Could not seed %s (%s); using the example read-only.", target, exc)
            return example
    return None


def load_rules(path: str | Path) -> list[Rule]:
    """讀入詞典。格式錯誤的單條規則會被略過，不會讓整份失效。"""
    p = Path(path).expanduser()
    raw = p.read_text(encoding="utf-8-sig")  # 容忍 BOM
    data = json.loads(raw)

    if isinstance(data, dict):
        items = data.get("corrections", [])
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("詞典最外層必須是物件或陣列")

    rules: list[Rule] = []
    for spec in items:
        if not isinstance(spec, dict):
            continue
        if "from" not in spec or "to" not in spec:
            continue
        if spec["from"] == spec["to"]:
            continue  # 僅作標記，不替換
        try:
            rules.append(Rule(spec))
        except re.error:
            continue  # 正則寫壞的規則直接跳過
    return rules


# --------------------------------------------------------------------------
# 套用
# --------------------------------------------------------------------------
def apply_rules(text: str, rules: Iterable[Rule]) -> tuple[str, Counter]:
    counter: Counter = Counter()
    for rule in rules:
        text, n = rule.apply(text)
        if n:
            counter[repr(rule)] += n
    return text, counter


def _set_text(seg: Any, value: str) -> None:
    """相容一般類別與 frozen dataclass。"""
    try:
        seg.text = value
    except Exception:  # frozen dataclass 會丟 FrozenInstanceError
        object.__setattr__(seg, "text", value)


def apply_to_segments(segments: Sequence[Any], rules: Sequence[Rule]) -> Counter:
    """就地修改 segment.text，回傳命中統計。"""
    total: Counter = Counter()
    if not rules:
        return total
    for seg in segments:
        original = getattr(seg, "text", None)
        if not isinstance(original, str) or not original:
            continue
        fixed, counter = apply_rules(original, rules)
        if counter:
            _set_text(seg, fixed)
            total.update(counter)
    return total


# --------------------------------------------------------------------------
# 顯示 / 範本
# --------------------------------------------------------------------------
def describe(rules: Sequence[Rule], path: str | Path, limit: int = 12) -> str:
    """產生 Markdown 供 UI 顯示。"""
    if not rules:
        return f"詞典 `{path}` 載入成功，但沒有任何有效規則。"
    lines = [f"**已載入 {len(rules)} 條規則** — `{path}`", ""]
    for rule in rules[:limit]:
        note = f"  ({rule.note})" if rule.note else ""
        lines.append(f"- `{rule.src}` → `{rule.dst}`{note}")
    if len(rules) > limit:
        lines.append(f"- ...另外 {len(rules) - limit} 條")
    return "\n".join(lines)


def summarize(counter: Counter, limit: int = 8) -> str:
    if not counter:
        return ""
    lines = [f"**校正 {sum(counter.values())} 處**"]
    for name, n in counter.most_common(limit):
        lines.append(f"- {n}x  {name}")
    if len(counter) > limit:
        lines.append(f"- ...另外 {len(counter) - limit} 種")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 詞彙表 — 只列「正確的詞」，不必知道模型會聽錯成什麼
#
# 這與 corrections 是兩種不同的機制：corrections 是事後找錯字替換，詞彙表則是
# 事前餵給解碼器（faster-whisper 的 hotwords），讓它一開始就偏向這些寫法。
# 人名、廠商、內部術語適合放這裡——你多半說不出模型會錯成什麼樣子。
# --------------------------------------------------------------------------
VOCAB_KEY = "vocabulary"


def load_vocabulary(path: str | Path) -> list[str]:
    """讀出詞彙表。找不到檔案或格式不符時回傳空清單，不丟例外。"""
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    items = data.get(VOCAB_KEY, [])
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        word = str(item).strip()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def parse_vocabulary(raw: str) -> list[str]:
    """把 UI 的多行文字轉成詞彙清單。同時接受換行、逗號、頓號分隔。"""
    seen: set[str] = set()
    out: list[str] = []
    for chunk in re.split(r"[\n,，、;；]+", raw or ""):
        word = chunk.strip()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def as_hotwords(vocab: Sequence[str]) -> str | None:
    """組成 faster-whisper 的 hotwords 字串。空清單回傳 None。"""
    words = [w.strip() for w in vocab if w and w.strip()]
    return "、".join(words) if words else None


def save_vocabulary(path: str | Path, vocab: Sequence[str]) -> Path:
    """只更新詞彙表，corrections 與其他欄位原樣保留。"""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(existing, dict):
                payload = existing
            elif isinstance(existing, list):  # 舊式：最外層就是規則陣列
                payload = {"corrections": existing}
        except Exception as exc:
            raise ValueError(f"詞典解析失敗，未存檔以免覆蓋內容：{exc}") from exc
        try:
            shutil.copy2(p, p.with_name(p.name + ".bak"))
        except OSError as exc:
            log.warning("Failed to back up %s: %s", p, exc)

    payload.setdefault("corrections", [])
    payload[VOCAB_KEY] = list(vocab)

    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------
# 表格編輯（UI）— 原始 spec 的讀寫與轉換
#
# 這裡刻意不經過 Rule：Rule 編譯完就丟掉 regex / ignore_case 旗標，
# 無法還原成表格。編輯路徑一律走未編譯的 spec dict。
# --------------------------------------------------------------------------
ROW_HEADERS = ["錯誤寫法 (from)", "正確寫法 (to)", "正則", "忽略大小寫", "備註"]
ROW_DATATYPES = ["str", "str", "bool", "bool", "str"]


def load_specs(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """讀出未編譯的規則 spec，連同其他最外層欄位（如 _comment）一起回傳。

    最外層欄位另外收好，存檔時才能原樣寫回去，不會被表格編輯洗掉。
    """
    p = Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8-sig"))

    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)], {}
    if isinstance(data, dict):
        specs = [s for s in data.get("corrections", []) if isinstance(s, dict)]
        extra = {k: v for k, v in data.items() if k != "corrections"}
        return specs, extra
    raise ValueError("詞典最外層必須是物件或陣列")


def specs_to_rows(specs: Sequence[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            str(s.get("from", "")),
            str(s.get("to", "")),
            bool(s.get("regex", False)),
            bool(s.get("ignore_case", False)),
            str(s.get("note", "")),
        ]
        for s in specs
    ]


def _truthy(value: Any) -> bool:
    """表格元件回傳的布林可能是 bool / "true" / 1 / ""，一律收斂成 bool。"""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "v", "是"}
    return bool(value)


def rows_to_specs(rows: Iterable[Sequence[Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """表格 -> spec 清單。回傳 (specs, 問題訊息)。

    整列空白直接跳過（使用者按「新增一列」留下的空列）。其餘問題只回報，
    不擋存檔——除了正則寫壞的那列會被丟掉，否則存進去也載不起來。
    """
    specs: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: set[tuple[str, bool, bool]] = set()

    for i, row in enumerate(rows or [], start=1):
        cells = list(row) + [""] * (5 - len(row))
        src = str(cells[0] if cells[0] is not None else "").strip()
        dst = str(cells[1] if cells[1] is not None else "").strip()
        is_regex = _truthy(cells[2])
        ignore_case = _truthy(cells[3])
        note = str(cells[4] if cells[4] is not None else "").strip()

        if not src and not dst:
            continue
        if not src:
            problems.append(f"第 {i} 列：「錯誤寫法」是空的，已略過。")
            continue
        if not dst:
            problems.append(f"第 {i} 列：「正確寫法」是空的，已略過。")
            continue

        if is_regex:
            try:
                re.compile(src)
            except re.error as exc:
                problems.append(f"第 {i} 列：正則語法錯誤（{exc}），已略過 — `{src}`")
                continue

        key = (src, is_regex, ignore_case)
        if key in seen:
            problems.append(f"第 {i} 列：`{src}` 重複，只會套用第一條。")
        seen.add(key)

        if src == dst:
            problems.append(f"第 {i} 列：`{src}` 前後相同，載入時會被忽略（等於停用）。")

        spec: dict[str, Any] = {"from": src, "to": dst}
        if is_regex:
            spec["regex"] = True
        if ignore_case:
            spec["ignore_case"] = True
        if note:
            spec["note"] = note
        specs.append(spec)

    return specs, problems


def save_specs(
    path: str | Path,
    specs: Sequence[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path:
    """寫回詞典檔。先備份成 <檔名>.bak，再以暫存檔 + 原子替換落地。

    直接覆寫的風險太高——使用者按下「儲存」時整張表就是唯一真相，寫壞了
    沒有第二份。備份只保留最近一次。
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    # 以「磁碟上的現況」為底，而不是呼叫端載入表格時快取的那份。否則
    # 「載入表格 -> 存詞彙表 -> 存表格」會用舊的快取把剛存的詞彙表洗掉。
    payload: dict[str, Any] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(existing, dict):
                payload = {k: v for k, v in existing.items() if k != "corrections"}
        except Exception as exc:
            log.warning("Could not re-read %s before saving (%s); using cached fields.", p, exc)
            payload = dict(extra or {})
        try:
            shutil.copy2(p, p.with_name(p.name + ".bak"))
        except OSError as exc:  # 備份失敗不該擋存檔，但要留下痕跡
            log.warning("Failed to back up %s: %s", p, exc)
    else:
        payload = dict(extra or {})

    payload["corrections"] = list(specs)

    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


# Deliberately empty: the dictionary is domain-specific, and a template
# pre-filled with one industry's jargon actively hurts every other user —
# vocabulary bias is global, so irrelevant terms skew decoding. Ship the
# schema and let people add their own.
TEMPLATE: dict[str, Any] = {  # noqa: E305
    "_comment": [
        "vocabulary：只填「正確寫法」，一行一個。人名、機構、專有名詞放這裡。",
        "  轉錄時會餵給解碼器做偏向，讓它一開始就別聽錯。",
        "  這是全域偏向，放不相干的詞會干擾辨識——只加你真正常講的。",
        "corrections：知道它固定錯成某個寫法時，用這個做事後替換。",
        '  {"from": "錯的", "to": "對的"}',
        '  可選欄位：regex（正規表示式）、ignore_case（忽略大小寫）、note（備註）',
        "存檔請用 UTF-8。",
    ],
    "vocabulary": [],
    "corrections": [],
}


def write_template(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
