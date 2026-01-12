#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
64卦マッピングスクリプト

目的: before_hex（下卦/内卦）+ trigger_hex（上卦/外卦）の組み合わせから
      64卦のhexagram_idを決定し、cases.jsonlを更新する

八卦の対応:
- 乾 (☰) = 天
- 坤 (☷) = 地
- 震 (☳) = 雷
- 巽 (☴) = 風
- 坎 (☵) = 水
- 離 (☲) = 火
- 艮 (☶) = 山
- 兌 (☱) = 沢

64卦構成: 下卦（内卦）+ 上卦（外卦）の組み合わせ
周易の伝統的順序に従った番号付け
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

# ==============================================================================
# 64卦マッピングテーブル
# キー: (下卦, 上卦) → 値: (hexagram_id, 卦名)
# 周易の伝統的順序に基づく
# ==============================================================================

TRIGRAM_TO_HEXAGRAM: Dict[Tuple[str, str], Tuple[int, str]] = {
    # 乾を下卦とする8卦
    ("乾", "乾"): (1, "乾為天"),
    ("乾", "坤"): (12, "天地否"),
    ("乾", "震"): (25, "天雷无妄"),
    ("乾", "巽"): (44, "天風姤"),
    ("乾", "坎"): (6, "天水訟"),
    ("乾", "離"): (13, "天火同人"),
    ("乾", "艮"): (33, "天山遯"),
    ("乾", "兌"): (10, "天沢履"),

    # 坤を下卦とする8卦
    ("坤", "乾"): (11, "地天泰"),
    ("坤", "坤"): (2, "坤為地"),
    ("坤", "震"): (24, "地雷復"),
    ("坤", "巽"): (46, "地風升"),
    ("坤", "坎"): (7, "地水師"),
    ("坤", "離"): (36, "地火明夷"),
    ("坤", "艮"): (15, "地山謙"),
    ("坤", "兌"): (19, "地沢臨"),

    # 震を下卦とする8卦
    ("震", "乾"): (34, "雷天大壮"),
    ("震", "坤"): (16, "雷地予"),
    ("震", "震"): (51, "震為雷"),
    ("震", "巽"): (32, "雷風恒"),
    ("震", "坎"): (3, "水雷屯"),
    ("震", "離"): (55, "雷火豊"),
    ("震", "艮"): (62, "雷山小過"),
    ("震", "兌"): (54, "雷沢帰妹"),

    # 巽を下卦とする8卦
    ("巽", "乾"): (9, "風天小畜"),
    ("巽", "坤"): (20, "風地観"),
    ("巽", "震"): (42, "風雷益"),
    ("巽", "巽"): (57, "巽為風"),
    ("巽", "坎"): (59, "風水渙"),
    ("巽", "離"): (37, "風火家人"),
    ("巽", "艮"): (53, "風山漸"),
    ("巽", "兌"): (61, "風沢中孚"),

    # 坎を下卦とする8卦
    ("坎", "乾"): (5, "水天需"),
    ("坎", "坤"): (8, "水地比"),
    ("坎", "震"): (40, "雷水解"),
    ("坎", "巽"): (48, "水風井"),
    ("坎", "坎"): (29, "坎為水"),
    ("坎", "離"): (63, "水火既済"),
    ("坎", "艮"): (39, "水山蹇"),
    ("坎", "兌"): (60, "水沢節"),

    # 離を下卦とする8卦
    ("離", "乾"): (14, "火天大有"),
    ("離", "坤"): (35, "火地晋"),
    ("離", "震"): (21, "火雷噬嗑"),
    ("離", "巽"): (50, "火風鼎"),
    ("離", "坎"): (64, "火水未済"),
    ("離", "離"): (30, "離為火"),
    ("離", "艮"): (56, "火山旅"),
    ("離", "兌"): (38, "火沢睽"),

    # 艮を下卦とする8卦
    ("艮", "乾"): (26, "山天大畜"),
    ("艮", "坤"): (23, "山地剥"),
    ("艮", "震"): (27, "山雷頤"),
    ("艮", "巽"): (18, "山風蠱"),
    ("艮", "坎"): (4, "山水蒙"),
    ("艮", "離"): (22, "山火賁"),
    ("艮", "艮"): (52, "艮為山"),
    ("艮", "兌"): (41, "山沢損"),

    # 兌を下卦とする8卦
    ("兌", "乾"): (43, "沢天夬"),
    ("兌", "坤"): (45, "沢地萃"),
    ("兌", "震"): (17, "沢雷随"),
    ("兌", "巽"): (28, "沢風大過"),
    ("兌", "坎"): (47, "沢水困"),
    ("兌", "離"): (49, "沢火革"),
    ("兌", "艮"): (31, "沢山咸"),
    ("兌", "兌"): (58, "兌為沢"),
}

# 有効な八卦リスト
VALID_TRIGRAMS = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}


def get_hexagram_id(lower_trigram: str, upper_trigram: str) -> Optional[Tuple[int, str]]:
    """
    下卦と上卦から64卦のIDと名前を取得する

    Args:
        lower_trigram: 下卦（内卦）= before_hex
        upper_trigram: 上卦（外卦）= trigger_hex

    Returns:
        (hexagram_id, hexagram_name) or None if invalid
    """
    if lower_trigram not in VALID_TRIGRAMS:
        return None
    if upper_trigram not in VALID_TRIGRAMS:
        return None

    key = (lower_trigram, upper_trigram)
    return TRIGRAM_TO_HEXAGRAM.get(key)


def get_hexagram_name(hexagram_id: int) -> Optional[str]:
    """hexagram_idから卦名を取得"""
    for (lower, upper), (hid, name) in TRIGRAM_TO_HEXAGRAM.items():
        if hid == hexagram_id:
            return name
    return None


def process_cases(input_path: Path, output_path: Path, backup: bool = True) -> dict:
    """
    cases.jsonlを読み込み、hexagram_idを再計算して保存する

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス
        backup: バックアップを作成するかどうか

    Returns:
        処理結果の統計情報
    """
    stats = {
        "total": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
        "missing_trigram": 0,
        "invalid_trigram": 0,
        "error_details": []
    }

    # バックアップ作成
    if backup and input_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.with_suffix(f".jsonl.bak_{timestamp}")
        shutil.copy2(input_path, backup_path)
        print(f"バックアップを作成: {backup_path}")

    cases = []

    # ファイル読み込み
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
                stats["total"] += 1

                # before_hexとtrigger_hexの取得
                before_hex = case.get("before_hex")
                trigger_hex = case.get("trigger_hex")

                if not before_hex or not trigger_hex:
                    stats["missing_trigram"] += 1
                    stats["error_details"].append({
                        "line": line_num,
                        "id": case.get("transition_id", "unknown"),
                        "error": f"before_hex={before_hex}, trigger_hex={trigger_hex}"
                    })
                    cases.append(case)
                    continue

                # hexagram_idを計算
                result = get_hexagram_id(before_hex, trigger_hex)

                if result is None:
                    stats["invalid_trigram"] += 1
                    stats["error_details"].append({
                        "line": line_num,
                        "id": case.get("transition_id", "unknown"),
                        "error": f"無効な八卦: before={before_hex}, trigger={trigger_hex}"
                    })
                    cases.append(case)
                    continue

                hexagram_id, hexagram_name = result
                old_hexagram_id = case.get("hexagram_id")

                # hexagram_idを更新
                case["hexagram_id"] = hexagram_id
                case["hexagram_name"] = hexagram_name

                if old_hexagram_id != hexagram_id:
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

                cases.append(case)

            except json.JSONDecodeError as e:
                stats["errors"] += 1
                stats["error_details"].append({
                    "line": line_num,
                    "error": f"JSON解析エラー: {e}"
                })

    # ファイル書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    return stats


def print_mapping_table():
    """64卦マッピングテーブルを整形して表示"""
    print("\n" + "=" * 80)
    print("64卦マッピングテーブル（下卦 + 上卦 → 卦番号・卦名）")
    print("=" * 80)

    # 上卦をヘッダーに
    trigrams = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
    print(f"\n{'下卦＼上卦':>10}", end="")
    for t in trigrams:
        print(f"{t:>10}", end="")
    print()
    print("-" * 90)

    for lower in trigrams:
        print(f"{lower:>10}", end="")
        for upper in trigrams:
            result = get_hexagram_id(lower, upper)
            if result:
                print(f"{result[0]:>10}", end="")
            else:
                print(f"{'--':>10}", end="")
        print()

    print()


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(
        description="64卦マッピングによるhexagram_id再計算"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/cases.jsonl",
        help="入力ファイルパス"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="出力ファイルパス（デフォルト: 入力と同じ）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="バックアップを作成しない"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には書き込まず、統計のみ表示"
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="64卦マッピングテーブルを表示"
    )

    args = parser.parse_args()

    # プロジェクトルートの検出
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.show_table:
        print_mapping_table()
        return

    input_path = project_root / args.input
    output_path = project_root / (args.output or args.input)

    print("=" * 60)
    print("64卦マッピング hexagram_id再計算スクリプト")
    print("=" * 60)
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"バックアップ: {'作成しない' if args.no_backup else '作成する'}")
    print(f"モード: {'ドライラン' if args.dry_run else '実行'}")
    print()

    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return

    if args.dry_run:
        # ドライランの場合は一時ファイルに出力
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            temp_output = Path(tmp.name)
        stats = process_cases(input_path, temp_output, backup=False)
        temp_output.unlink()
    else:
        stats = process_cases(input_path, output_path, backup=not args.no_backup)

    # 結果表示
    print("-" * 60)
    print("処理結果:")
    print(f"  総レコード数:       {stats['total']:,}")
    print(f"  更新されたレコード: {stats['updated']:,}")
    print(f"  変更なし:           {stats['unchanged']:,}")
    print(f"  欠損八卦:           {stats['missing_trigram']:,}")
    print(f"  無効な八卦:         {stats['invalid_trigram']:,}")
    print(f"  JSONエラー:         {stats['errors']:,}")
    print("-" * 60)

    if stats["error_details"]:
        print("\nエラー詳細（最初の10件）:")
        for detail in stats["error_details"][:10]:
            print(f"  行 {detail.get('line', '?')}: {detail.get('id', '?')} - {detail.get('error', '?')}")

    if not args.dry_run and stats["updated"] > 0:
        print(f"\n完了: {output_path} に保存しました")


if __name__ == "__main__":
    main()
