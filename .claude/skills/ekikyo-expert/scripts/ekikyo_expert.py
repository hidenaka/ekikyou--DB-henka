#!/usr/bin/env python3
"""
Ekikyo Expert CLI (日本語テキスト対応版)

- analyze: 卦の概要と爻の日本語要約を表示
- predict: 変爻から之卦を算出し、日本語要約を表示
- search: 日本語要約にキーワード検索
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional


class DataStore:
    def __init__(self, root_dir: str):
        self.root = root_dir
        self.hex_master = self._load_json("data/hexagrams/hexagram_master.json")
        self.iching_ja = self._load_json("data/reference/iching_texts_ctext_legge_ja.json")
        self.transitions = self._load_json("data/mappings/yao_transitions.json")

    def _load_json(self, rel_path: str) -> Dict:
        path = os.path.join(self.root, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_hex(self, hex_id: int) -> Optional[Dict]:
        return self.hex_master.get(str(hex_id))

    def get_texts(self, hex_id: int) -> Optional[Dict]:
        return self.iching_ja.get("hexagrams", {}).get(str(hex_id))

    def get_resulting(self, hex_id: int, line_nums: List[int]) -> Optional[int]:
        if not line_nums:
            return None
        key = str(hex_id)
        if key not in self.transitions:
            return None
        # transitions json has per-line mapping under ["transitions"][line]
        current = self.transitions[key]
        # apply flips sequentially
        lines = current.get("transitions", {})
        # combine by applying flip rules using map: just take final hex after applying all lines
        result = hex_id
        for ln in line_nums:
            ln_str = str(ln)
            entry = self.transitions[str(result)]["transitions"][ln_str]
            result = entry["next_hexagram_id"]
        return result


def format_block(title: str, text: Optional[str]) -> str:
    if not text:
        return f"{title}: （情報なし）"
    return f"{title}: {text}"


def handle_analyze(store: DataStore, hex_id: int, lines: List[int]):
    hex_info = store.get_hex(hex_id)
    texts = store.get_texts(hex_id)
    if not hex_info or not texts:
        print(f"Hexagram {hex_id} not found.")
        return
    print(f"# {hex_info['name']} (ID: {hex_id})")
    print(format_block("卦の概要", texts["judgment"].get("modern_ja")))
    print(format_block("彖伝", texts.get("tuan", {}).get("modern_ja")))
    print(format_block("象伝", texts.get("xiang", {}).get("modern_ja")))
    if lines:
        print("\n## 指定爻")
        for ln in lines:
            line = texts["lines"].get(str(ln))
            if not line:
                print(f"- 第{ln}爻: 情報なし")
                continue
            print(f"- 第{ln}爻: {line.get('modern_ja','')}")
            if "xiang" in line:
                print(f"  象: {line['xiang'].get('modern_ja','')}")


def handle_predict(store: DataStore, hex_id: int, lines: List[int]):
    if not lines:
        print("変爻を指定してください (--lines 1,2 など)")
        return
    result = store.get_resulting(hex_id, lines)
    print(f"# 変化: {hex_id} -> {result if result else '不明'}  (変爻: {','.join(map(str, lines))})")
    handle_analyze(store, hex_id, lines)
    if result:
        print("\n## 之卦")
        handle_analyze(store, result, [])


def handle_search(store: DataStore, keyword: str, limit: int):
    kw = keyword.strip()
    matches = []
    for hex_id, hx in store.iching_ja.get("hexagrams", {}).items():
        # search in judgment + lines
        fields = []
        fields.append(hx.get("judgment", {}).get("modern_ja", ""))
        fields.append(hx.get("tuan", {}).get("modern_ja", ""))
        fields.append(hx.get("xiang", {}).get("modern_ja", ""))
        for line in hx.get("lines", {}).values():
            fields.append(line.get("modern_ja", ""))
            if "xiang" in line:
                fields.append(line["xiang"].get("modern_ja", ""))
        text = " ".join(fields)
        if kw in text:
            matches.append((hex_id, hx, text))
        if len(matches) >= limit:
            break
    if not matches:
        print("該当なし")
        return
    for idx, (hid, hx, _) in enumerate(matches, 1):
        name = hx.get("local_name") or store.get_hex(int(hid)).get("name")
        print(f"{idx}. {name} (ID: {hid})")
        print(f"   卦の概要: {hx.get('judgment',{}).get('modern_ja','')}")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    store = DataStore(root_dir)

    parser = argparse.ArgumentParser(description="Ekikyo Expert CLI (日本語)")
    sub = parser.add_subparsers(dest="cmd")

    p_an = sub.add_parser("analyze", help="卦を分析（日本語）")
    p_an.add_argument("hexagram", type=int, help="卦番号 (1-64)")
    p_an.add_argument("--lines", type=str, help="変爻（例: 1,5）")

    p_pr = sub.add_parser("predict", help="変爻から之卦を計算（日本語）")
    p_pr.add_argument("hexagram", type=int, help="卦番号 (1-64)")
    p_pr.add_argument("--lines", required=True, help="変爻（例: 1,5）")

    p_se = sub.add_parser("search", help="日本語要約にキーワード検索")
    p_se.add_argument("keyword", type=str, help="検索キーワード")
    p_se.add_argument("--limit", type=int, default=5, help="最大件数")

    args = parser.parse_args()
    if args.cmd == "analyze":
        lines = []
        if args.lines:
            lines = [int(x) for x in args.lines.split(",") if x.strip().isdigit()]
        handle_analyze(store, args.hexagram, lines)
    elif args.cmd == "predict":
        lines = [int(x) for x in args.lines.split(",") if x.strip().isdigit()]
        handle_predict(store, args.hexagram, lines)
    elif args.cmd == "search":
        handle_search(store, args.keyword, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
