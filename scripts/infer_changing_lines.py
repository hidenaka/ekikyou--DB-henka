#!/usr/bin/env python3
"""
変爻推定ロジック - 八卦のペアから変化した爻を推定する

八卦（3爻）のペアを比較して、どの爻が変化したかを推定します。
例: 乾（陽陽陽）→ 兌（陰陽陽）= 上爻（3爻目）が変化
"""
from typing import List, Optional
from enum import Enum

class LineType(str, Enum):
    """爻の種類"""
    YANG = "陽"  # ─
    YIN = "陰"   # - -

# 八卦の爻構成マッピング（下から上へ: 初爻、二爻、三爻）
TRIGRAM_LINES = {
    "乾": [LineType.YANG, LineType.YANG, LineType.YANG],  # ☰ 天
    "坤": [LineType.YIN, LineType.YIN, LineType.YIN],      # ☷ 地
    "震": [LineType.YANG, LineType.YIN, LineType.YIN],     # ☳ 雷
    "巽": [LineType.YIN, LineType.YANG, LineType.YANG],    # ☴ 風
    "坎": [LineType.YIN, LineType.YANG, LineType.YIN],     # ☵ 水
    "離": [LineType.YANG, LineType.YIN, LineType.YANG],    # ☲ 火
    "艮": [LineType.YIN, LineType.YIN, LineType.YANG],     # ☶ 山
    "兌": [LineType.YANG, LineType.YANG, LineType.YIN],    # ☱ 沢
}

def infer_changing_lines(from_hex: str, to_hex: str) -> Optional[List[int]]:
    """
    2つの八卦を比較して、変化した爻の番号を返す

    Args:
        from_hex: 変化前の八卦（例: "乾"）
        to_hex: 変化後の八卦（例: "兌"）

    Returns:
        変化した爻の番号リスト [1-3]
        - 1 = 初爻（最下位）
        - 2 = 二爻（中位）
        - 3 = 三爻（最上位）

        同じ卦の場合は空リスト []
        どちらかの卦が不明な場合は None

    Example:
        >>> infer_changing_lines("乾", "兌")
        [3]  # 上爻が陽→陰に変化

        >>> infer_changing_lines("坎", "離")
        [1, 3]  # 初爻と上爻が変化
    """
    # 八卦が不明な場合
    if from_hex not in TRIGRAM_LINES or to_hex not in TRIGRAM_LINES:
        return None

    # 同じ卦の場合
    if from_hex == to_hex:
        return []

    from_lines = TRIGRAM_LINES[from_hex]
    to_lines = TRIGRAM_LINES[to_hex]

    changing = []
    for i in range(3):
        if from_lines[i] != to_lines[i]:
            changing.append(i + 1)  # 1-indexed (初爻=1, 二爻=2, 三爻=3)

    return changing

def print_trigram_table():
    """全八卦の爻構成を表示（デバッグ用）"""
    print("八卦の爻構成:")
    print("-" * 40)
    print("卦名  | 初爻 | 二爻 | 三爻 | 記号")
    print("-" * 40)

    symbols = {
        "乾": "☰", "坤": "☷", "震": "☳", "巽": "☴",
        "坎": "☵", "離": "☲", "艮": "☶", "兌": "☱"
    }

    for hex_name, lines in TRIGRAM_LINES.items():
        line_str = " | ".join(line.value for line in lines)
        symbol = symbols.get(hex_name, "")
        print(f"{hex_name}    | {line_str} | {symbol}")
    print("-" * 40)

def test_infer_changing_lines():
    """推定ロジックのテストケース"""
    print("\n変爻推定ロジックのテスト:")
    print("=" * 60)

    test_cases = [
        ("乾", "兌", [3], "乾→兌: 上爻が陽→陰"),
        ("乾", "離", [2], "乾→離: 二爻が陽→陰"),
        ("坤", "震", [1], "坤→震: 初爻が陰→陽"),
        ("坎", "離", [1, 2, 3], "坎→離: 全ての爻が変化（陰陽陰→陽陰陽）"),
        ("乾", "乾", [], "乾→乾: 変化なし"),
        ("震", "艮", [1, 3], "震→艮: 初爻と上爻が変化（陽陰陰→陰陰陽）"),
        ("乾", "坤", [1, 2, 3], "乾→坤: 全ての爻が変化（陽陽陽→陰陰陰）"),
        ("震", "巽", [1, 2, 3], "震→巽: 全ての爻が変化（陽陰陰→陰陽陽）"),
    ]

    all_passed = True
    for from_h, to_h, expected, description in test_cases:
        result = infer_changing_lines(from_h, to_h)
        passed = result == expected
        status = "✅" if passed else "❌"

        print(f"{status} {description}")
        print(f"   期待値: {expected}")
        print(f"   実際値: {result}")

        if not passed:
            all_passed = False
            from_lines = TRIGRAM_LINES[from_h]
            to_lines = TRIGRAM_LINES[to_h]
            print(f"   {from_h}: {[l.value for l in from_lines]}")
            print(f"   {to_h}: {[l.value for l in to_lines]}")
        print()

    if all_passed:
        print("=" * 60)
        print("全てのテストが成功しました！")
    else:
        print("=" * 60)
        print("一部のテストが失敗しました。")

    return all_passed

def interactive_test():
    """インタラクティブなテストモード"""
    print("\n" + "=" * 60)
    print("インタラクティブテスト（Ctrl+Cで終了）")
    print("=" * 60)

    hex_names = list(TRIGRAM_LINES.keys())
    print(f"\n利用可能な八卦: {', '.join(hex_names)}")

    try:
        while True:
            print("\n" + "-" * 60)
            from_hex = input("変化前の八卦を入力（例: 乾）: ").strip()
            to_hex = input("変化後の八卦を入力（例: 兌）: ").strip()

            result = infer_changing_lines(from_hex, to_hex)

            if result is None:
                print("⚠️ 不明な八卦が入力されました")
            elif result == []:
                print("→ 変爻なし（同じ卦）")
            else:
                line_names = {1: "初爻", 2: "二爻", 3: "三爻"}
                changed = [line_names[i] for i in result]
                print(f"→ 変爻: {result} ({', '.join(changed)})")

                # 詳細表示
                from_lines = TRIGRAM_LINES[from_hex]
                to_lines = TRIGRAM_LINES[to_hex]
                print(f"\n   {from_hex}: {[l.value for l in from_lines]}")
                print(f"   {to_hex}: {[l.value for l in to_lines]}")

    except KeyboardInterrupt:
        print("\n\n終了します。")

if __name__ == "__main__":
    import sys

    # 八卦の爻構成を表示
    print_trigram_table()

    # テスト実行
    test_infer_changing_lines()

    # コマンドライン引数で --interactive を指定した場合のみインタラクティブモード
    if "--interactive" in sys.argv:
        interactive_test()
