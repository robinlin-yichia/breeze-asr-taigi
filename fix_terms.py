"""
fix_terms.py — 轉錄後專有名詞校正

依據一份詞典檔，把 ASR 常錯的字詞批次修正，
支援 .srt / .vtt / .txt / .json，時間軸與編號不受影響。

用法：
    python fix_terms.py output.srt
    python fix_terms.py output.srt --dict terms.json --out fixed.srt
    python fix_terms.py *.srt --in-place
    python fix_terms.py output.srt --dry-run      # 只看會改什麼，不寫檔

詞典格式（terms.json，UTF-8 無 BOM）：
{
  "corrections": [
    {"from": "錯的寫法", "to": "正確寫法"},
    {"from": "正則樣式", "to": "正確寫法", "regex": true, "ignore_case": true},
    {"from": "同一個詞", "to": "同一個詞", "note": "前後相同者會被略過，可當停用"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DICT = "terms.json"

# SRT/VTT 的時間軸列與純數字編號列，這些不能動
_TS_LINE = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")
_INDEX_LINE = re.compile(r"^\d+$")
_VTT_HEADER = re.compile(r"^(WEBVTT|NOTE|STYLE|REGION)\b")


# --------------------------------------------------------------------------
# 詞典
# --------------------------------------------------------------------------
class Rule:
    def __init__(self, spec: dict):
        self.src = spec["from"]
        self.dst = spec["to"]
        self.note = spec.get("note", "")
        flags = re.IGNORECASE if spec.get("ignore_case") else 0
        pattern = self.src if spec.get("regex") else re.escape(self.src)
        self.pattern = re.compile(pattern, flags)

    def apply(self, text: str) -> tuple[str, int]:
        new, n = self.pattern.subn(self.dst, text)
        return new, n

    def __repr__(self) -> str:
        return f"{self.src} -> {self.dst}"


def load_rules(path: Path) -> list[Rule]:
    if not path.exists():
        raise SystemExit(f"[ERROR] 找不到詞典檔：{path}\n     可用 --init 產生範本。")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("corrections", data if isinstance(data, list) else [])
    rules = []
    for i, spec in enumerate(items, 1):
        if not isinstance(spec, dict) or "from" not in spec or "to" not in spec:
            print(f"[WARN] 第 {i} 條規則格式不正確，已略過：{spec}", file=sys.stderr)
            continue
        if spec["from"] == spec["to"]:
            continue  # 標記用，不實際替換
        rules.append(Rule(spec))
    return rules


# Deliberately empty. A template pre-filled with one field's jargon is worse
# than no template: users assume it is a sensible default and inherit terms
# that do not apply to their recordings.
TEMPLATE = {
    "_comment": [
        "corrections: {\"from\": \"錯的\", \"to\": \"對的\"}",
        "可選欄位：regex（正規表示式）、ignore_case（忽略大小寫）、note（備註）",
    ],
    "corrections": [],
}


# --------------------------------------------------------------------------
# 套用
# --------------------------------------------------------------------------
def is_protected(line: str, suffix: str) -> bool:
    """時間軸、字幕編號、VTT 標頭不可替換。"""
    if suffix not in (".srt", ".vtt"):
        return False
    s = line.strip()
    return bool(
        _TS_LINE.search(s) or _INDEX_LINE.match(s) or _VTT_HEADER.match(s)
    )


def fix_text_block(text: str, rules: list[Rule], counter: Counter) -> str:
    for rule in rules:
        text, n = rule.apply(text)
        if n:
            counter[repr(rule)] += n
    return text


def fix_json(raw: str, rules: list[Rule], counter: Counter) -> str:
    data = json.loads(raw)

    def walk(node):
        if isinstance(node, dict):
            return {
                k: (fix_text_block(v, rules, counter)
                    if k in ("text", "content") and isinstance(v, str)
                    else walk(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return json.dumps(walk(data), ensure_ascii=False, indent=2)


def fix_file(path: Path, rules: list[Rule]) -> tuple[str, Counter]:
    counter: Counter = Counter()
    raw = path.read_text(encoding="utf-8", errors="strict")
    suffix = path.suffix.lower()

    if suffix == ".json":
        return fix_json(raw, rules, counter), counter

    out_lines = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        if is_protected(line, suffix):
            out_lines.append(line)
        else:
            out_lines.append(fix_text_block(line, rules, counter))
    return "\n".join(out_lines), counter


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="ASR 轉錄後專有名詞校正")
    ap.add_argument("files", nargs="*", help="要處理的 srt/vtt/txt/json")
    ap.add_argument("--dict", default=DEFAULT_DICT, help=f"詞典檔（預設 {DEFAULT_DICT}）")
    ap.add_argument("--out", help="輸出路徑（僅單檔時有效）")
    ap.add_argument("--in-place", action="store_true", help="直接覆寫原檔")
    ap.add_argument("--dry-run", action="store_true", help="只顯示會改什麼，不寫檔")
    ap.add_argument("--init", action="store_true", help="產生詞典範本後結束")
    args = ap.parse_args()

    dict_path = Path(args.dict)

    if args.init:
        if dict_path.exists():
            print(f"[ERROR] {dict_path} 已存在，不覆蓋。")
            return 1
        dict_path.write_text(
            json.dumps(TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] 已產生詞典範本：{dict_path}")
        return 0

    if not args.files:
        ap.error("請指定至少一個檔案，或使用 --init 產生詞典範本。")

    rules = load_rules(dict_path)
    print(f"[INFO] 載入 {len(rules)} 條規則\n")

    total: Counter = Counter()
    for name in args.files:
        path = Path(name)
        if not path.exists():
            print(f"[WARN] 找不到 {path}，略過")
            continue

        fixed, counter = fix_file(path, rules)
        total.update(counter)

        hits = sum(counter.values())
        print(f"── {path.name}：{hits} 處")
        for rule_repr, n in counter.most_common():
            print(f"     {n:>4}x  {rule_repr}")
        if not counter:
            print("       （無變更）")

        if args.dry_run:
            continue

        if args.in_place:
            dst = path
        elif args.out and len(args.files) == 1:
            dst = Path(args.out)
        else:
            dst = path.with_name(f"{path.stem}.fixed{path.suffix}")

        dst.write_text(fixed, encoding="utf-8")
        print(f"     -> {dst}")

    print(f"\n[DONE] 合計修正 {sum(total.values())} 處"
          + ("（dry-run，未寫檔）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
