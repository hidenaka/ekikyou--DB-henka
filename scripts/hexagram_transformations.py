#!/usr/bin/env python3
"""
易経 卦変換ロジック

互卦 (Nuclear Hexagram):     元の卦の2,3,4爻を下卦、3,4,5爻を上卦として構成
覆卦 (Inverted Hexagram):    卦を上下反転（6爻の順序を逆にする）
錯卦 (Complementary Hexagram): 全ての爻を陰陽反転（0→1, 1→0）

使用法:
    python3 scripts/hexagram_transformations.py
    python3 scripts/hexagram_transformations.py --hexagram 51
    python3 scripts/hexagram_transformations.py --all
"""

from typing import List, Tuple, Dict, Optional
import argparse
import json

# 八卦の爻構成 (下から上: 1爻, 2爻, 3爻)
# 1 = 陽爻（実線）, 0 = 陰爻（破線）
TRIGRAM_LINES: Dict[str, List[int]] = {
    "乾": [1, 1, 1],  # ☰ 天
    "兌": [1, 1, 0],  # ☱ 沢
    "離": [1, 0, 1],  # ☲ 火
    "震": [1, 0, 0],  # ☳ 雷
    "巽": [0, 1, 1],  # ☴ 風
    "坎": [0, 1, 0],  # ☵ 水
    "艮": [0, 0, 1],  # ☶ 山
    "坤": [0, 0, 0],  # ☷ 地
}

# 爻構成から八卦名への逆引き
LINES_TO_TRIGRAM: Dict[Tuple[int, int, int], str] = {
    tuple(v): k for k, v in TRIGRAM_LINES.items()
}

# 64卦テーブル (下卦, 上卦) -> (卦番号, 卦名)
HEXAGRAM_TABLE: Dict[Tuple[str, str], Tuple[int, str]] = {
    ("乾", "乾"): (1, "乾為天"),
    ("乾", "坤"): (12, "天地否"),
    ("乾", "震"): (25, "天雷无妄"),
    ("乾", "巽"): (44, "天風姤"),
    ("乾", "坎"): (6, "天水訟"),
    ("乾", "離"): (13, "天火同人"),
    ("乾", "艮"): (33, "天山遯"),
    ("乾", "兌"): (10, "天沢履"),

    ("坤", "乾"): (11, "地天泰"),
    ("坤", "坤"): (2, "坤為地"),
    ("坤", "震"): (24, "地雷復"),
    ("坤", "巽"): (46, "地風升"),
    ("坤", "坎"): (7, "地水師"),
    ("坤", "離"): (36, "地火明夷"),
    ("坤", "艮"): (15, "地山謙"),
    ("坤", "兌"): (19, "地沢臨"),

    ("震", "乾"): (34, "雷天大壮"),
    ("震", "坤"): (16, "雷地豫"),
    ("震", "震"): (51, "震為雷"),
    ("震", "巽"): (32, "雷風恒"),
    ("震", "坎"): (40, "雷水解"),
    ("震", "離"): (55, "雷火豊"),
    ("震", "艮"): (62, "雷山小過"),
    ("震", "兌"): (54, "雷沢帰妹"),

    ("巽", "乾"): (9, "風天小畜"),
    ("巽", "坤"): (20, "風地観"),
    ("巽", "震"): (42, "風雷益"),
    ("巽", "巽"): (57, "巽為風"),
    ("巽", "坎"): (59, "風水渙"),
    ("巽", "離"): (37, "風火家人"),
    ("巽", "艮"): (53, "風山漸"),
    ("巽", "兌"): (61, "風沢中孚"),

    ("坎", "乾"): (5, "水天需"),
    ("坎", "坤"): (8, "水地比"),
    ("坎", "震"): (3, "水雷屯"),
    ("坎", "巽"): (48, "水風井"),
    ("坎", "坎"): (29, "坎為水"),
    ("坎", "離"): (63, "水火既済"),
    ("坎", "艮"): (39, "水山蹇"),
    ("坎", "兌"): (60, "水沢節"),

    ("離", "乾"): (14, "火天大有"),
    ("離", "坤"): (35, "火地晋"),
    ("離", "震"): (21, "火雷噬嗑"),
    ("離", "巽"): (50, "火風鼎"),
    ("離", "坎"): (64, "火水未済"),
    ("離", "離"): (30, "離為火"),
    ("離", "艮"): (56, "火山旅"),
    ("離", "兌"): (38, "火沢睽"),

    ("艮", "乾"): (26, "山天大畜"),
    ("艮", "坤"): (23, "山地剥"),
    ("艮", "震"): (27, "山雷頤"),
    ("艮", "巽"): (18, "山風蠱"),
    ("艮", "坎"): (4, "山水蒙"),
    ("艮", "離"): (22, "山火賁"),
    ("艮", "艮"): (52, "艮為山"),
    ("艮", "兌"): (41, "山沢損"),

    ("兌", "乾"): (43, "沢天夬"),
    ("兌", "坤"): (45, "沢地萃"),
    ("兌", "震"): (17, "沢雷随"),
    ("兌", "巽"): (28, "沢風大過"),
    ("兌", "坎"): (47, "沢水困"),
    ("兌", "離"): (49, "沢火革"),
    ("兌", "艮"): (31, "沢山咸"),
    ("兌", "兌"): (58, "兌為沢"),
}

# 卦番号 -> (下卦, 上卦, 卦名) の逆引きテーブル
HEXAGRAM_BY_ID: Dict[int, Tuple[str, str, str]] = {
    v[0]: (k[0], k[1], v[1]) for k, v in HEXAGRAM_TABLE.items()
}


def hexagram_to_lines(hexagram_id: int) -> List[int]:
    """
    卦IDから6爻のリストを返す

    Args:
        hexagram_id: 1-64の卦番号

    Returns:
        [line1, line2, line3, line4, line5, line6]
        (下から上への順序、1=陽爻、0=陰爻)

    Example:
        >>> hexagram_to_lines(51)  # 震為雷
        [1, 0, 0, 1, 0, 0]
    """
    if hexagram_id not in HEXAGRAM_BY_ID:
        raise ValueError(f"無効な卦番号: {hexagram_id} (1-64の範囲で指定)")

    lower, upper, _ = HEXAGRAM_BY_ID[hexagram_id]
    lower_lines = TRIGRAM_LINES[lower]  # 1,2,3爻
    upper_lines = TRIGRAM_LINES[upper]  # 4,5,6爻

    return lower_lines + upper_lines


def lines_to_hexagram(lines: List[int]) -> Tuple[int, str]:
    """
    6爻のリストから卦ID・卦名を返す

    Args:
        lines: [line1, line2, line3, line4, line5, line6]
               (下から上への順序、1=陽爻、0=陰爻)

    Returns:
        (卦番号, 卦名)

    Example:
        >>> lines_to_hexagram([1, 0, 0, 1, 0, 0])
        (51, '震為雷')
    """
    if len(lines) != 6:
        raise ValueError(f"6爻が必要です（現在: {len(lines)}爻）")

    if not all(line in (0, 1) for line in lines):
        raise ValueError("爻の値は0または1のみ有効です")

    # 下卦 (1,2,3爻) と 上卦 (4,5,6爻) に分割
    lower_tuple = tuple(lines[0:3])
    upper_tuple = tuple(lines[3:6])

    lower = LINES_TO_TRIGRAM.get(lower_tuple)
    upper = LINES_TO_TRIGRAM.get(upper_tuple)

    if lower is None:
        raise ValueError(f"無効な下卦: {lower_tuple}")
    if upper is None:
        raise ValueError(f"無効な上卦: {upper_tuple}")

    result = HEXAGRAM_TABLE.get((lower, upper))
    if result is None:
        raise ValueError(f"無効な卦の組み合わせ: 下卦={lower}, 上卦={upper}")

    return result


def get_nuclear_hexagram(hexagram_id: int) -> Tuple[int, str]:
    """
    互卦 (Nuclear Hexagram) を計算

    互卦の構成:
    - 新しい下卦 = 元の卦の2,3,4爻
    - 新しい上卦 = 元の卦の3,4,5爻

    Args:
        hexagram_id: 1-64の卦番号

    Returns:
        (互卦の番号, 互卦の名前)

    Example:
        >>> get_nuclear_hexagram(51)  # 震為雷
        (39, '水山蹇')

        震為雷 [1,0,0,1,0,0]:
        - 新下卦 = 2,3,4爻 = [0,0,1] = 艮(山)
        - 新上卦 = 3,4,5爻 = [0,1,0] = 坎(水)
        → 水山蹇
    """
    lines = hexagram_to_lines(hexagram_id)

    # 互卦の構成
    # 新しい下卦 = 2爻, 3爻, 4爻 (インデックス: 1, 2, 3)
    # 新しい上卦 = 3爻, 4爻, 5爻 (インデックス: 2, 3, 4)
    new_lower = lines[1:4]  # 2,3,4爻
    new_upper = lines[2:5]  # 3,4,5爻

    nuclear_lines = new_lower + new_upper
    return lines_to_hexagram(nuclear_lines)


def get_inverted_hexagram(hexagram_id: int) -> Tuple[int, str]:
    """
    覆卦 (Inverted Hexagram) を計算

    卦を上下反転（180度回転）する。
    6爻の順序を逆にする。

    Args:
        hexagram_id: 1-64の卦番号

    Returns:
        (覆卦の番号, 覆卦の名前)

    Example:
        >>> get_inverted_hexagram(51)  # 震為雷
        (57, '巽為風')

        震為雷 [1,0,0,1,0,0] → 反転 → [0,0,1,0,0,1]
        - 下卦 = [0,0,1] = 艮 → 反転後は上卦に
        - 上卦 = [1,0,0] = 震 → 反転後は下卦に
        → 巽為風 ([0,1,1,0,1,1])

        注: 震為雷を反転すると巽為風になる
            震 [1,0,0] を反転 → [0,0,1] = 艮
            しかし覆卦は卦全体を反転するので
            [1,0,0,1,0,0] → [0,0,1,0,0,1]
    """
    lines = hexagram_to_lines(hexagram_id)

    # 6爻を逆順に
    inverted_lines = lines[::-1]

    return lines_to_hexagram(inverted_lines)


def get_complementary_hexagram(hexagram_id: int) -> Tuple[int, str]:
    """
    錯卦 (Complementary Hexagram) を計算

    全ての爻の陰陽を反転する（0→1, 1→0）。

    Args:
        hexagram_id: 1-64の卦番号

    Returns:
        (錯卦の番号, 錯卦の名前)

    Example:
        >>> get_complementary_hexagram(51)  # 震為雷
        (57, '巽為風')

        震為雷 [1,0,0,1,0,0] → 反転 → [0,1,1,0,1,1] = 巽為風
    """
    lines = hexagram_to_lines(hexagram_id)

    # 各爻を反転
    complementary_lines = [1 - line for line in lines]

    return lines_to_hexagram(complementary_lines)


def get_all_transformations(hexagram_id: int) -> Dict:
    """
    指定した卦の互卦・覆卦・錯卦を全て取得

    Args:
        hexagram_id: 1-64の卦番号

    Returns:
        {
            'original': (id, name, lines),
            'nuclear': (id, name, lines),
            'inverted': (id, name, lines),
            'complementary': (id, name, lines)
        }
    """
    original_lines = hexagram_to_lines(hexagram_id)
    _, _, original_name = HEXAGRAM_BY_ID[hexagram_id]

    nuclear_id, nuclear_name = get_nuclear_hexagram(hexagram_id)
    nuclear_lines = hexagram_to_lines(nuclear_id)

    inverted_id, inverted_name = get_inverted_hexagram(hexagram_id)
    inverted_lines = hexagram_to_lines(inverted_id)

    comp_id, comp_name = get_complementary_hexagram(hexagram_id)
    comp_lines = hexagram_to_lines(comp_id)

    return {
        'original': {
            'id': hexagram_id,
            'name': original_name,
            'lines': original_lines
        },
        'nuclear': {
            'id': nuclear_id,
            'name': nuclear_name,
            'lines': nuclear_lines,
            'description': '互卦（内なる構造）'
        },
        'inverted': {
            'id': inverted_id,
            'name': inverted_name,
            'lines': inverted_lines,
            'description': '覆卦（上下反転）'
        },
        'complementary': {
            'id': comp_id,
            'name': comp_name,
            'lines': comp_lines,
            'description': '錯卦（陰陽反転）'
        }
    }


def format_lines_visual(lines: List[int]) -> str:
    """爻を視覚的に表示"""
    symbols = []
    for line in reversed(lines):  # 上から表示
        if line == 1:
            symbols.append("━━━━━")  # 陽爻
        else:
            symbols.append("━━ ━━")  # 陰爻
    return "\n".join(symbols)


def print_transformation_report(hexagram_id: int) -> None:
    """変換結果をレポート形式で表示"""
    result = get_all_transformations(hexagram_id)

    print("=" * 60)
    print(f"【卦変換レポート】 {result['original']['name']} (第{hexagram_id}卦)")
    print("=" * 60)

    # 元の卦
    print(f"\n■ 元卦: {result['original']['name']} (第{result['original']['id']}卦)")
    print(f"  爻: {result['original']['lines']}")
    print(format_lines_visual(result['original']['lines']))

    # 互卦
    nuclear = result['nuclear']
    print(f"\n■ 互卦 (Nuclear): {nuclear['name']} (第{nuclear['id']}卦)")
    print(f"  説明: {nuclear['description']}")
    print(f"  爻: {nuclear['lines']}")
    print(format_lines_visual(nuclear['lines']))

    # 覆卦
    inverted = result['inverted']
    print(f"\n■ 覆卦 (Inverted): {inverted['name']} (第{inverted['id']}卦)")
    print(f"  説明: {inverted['description']}")
    print(f"  爻: {inverted['lines']}")
    print(format_lines_visual(inverted['lines']))

    # 錯卦
    comp = result['complementary']
    print(f"\n■ 錯卦 (Complementary): {comp['name']} (第{comp['id']}卦)")
    print(f"  説明: {comp['description']}")
    print(f"  爻: {comp['lines']}")
    print(format_lines_visual(comp['lines']))

    print("\n" + "=" * 60)


def generate_all_transformations_table() -> List[Dict]:
    """全64卦の変換テーブルを生成"""
    results = []
    for hex_id in range(1, 65):
        trans = get_all_transformations(hex_id)
        results.append({
            'hexagram_id': hex_id,
            'hexagram_name': trans['original']['name'],
            'lines': trans['original']['lines'],
            'nuclear_id': trans['nuclear']['id'],
            'nuclear_name': trans['nuclear']['name'],
            'inverted_id': trans['inverted']['id'],
            'inverted_name': trans['inverted']['name'],
            'complementary_id': trans['complementary']['id'],
            'complementary_name': trans['complementary']['name'],
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="易経 卦変換ロジック（互卦・覆卦・錯卦）"
    )
    parser.add_argument(
        "--hexagram", "-H",
        type=int,
        help="計算する卦の番号 (1-64)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="全64卦の変換テーブルをJSON出力"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="テスト実行"
    )

    args = parser.parse_args()

    if args.all:
        table = generate_all_transformations_table()
        print(json.dumps(table, ensure_ascii=False, indent=2))
    elif args.hexagram:
        print_transformation_report(args.hexagram)
    elif args.test:
        run_tests()
    else:
        # デフォルト: サンプル実行
        print("=== 易経 卦変換ロジック ===\n")
        print("サンプル実行: 震為雷 (第51卦)\n")
        print_transformation_report(51)

        print("\n\nサンプル実行: 乾為天 (第1卦)\n")
        print_transformation_report(1)

        print("\n\nサンプル実行: 水火既済 (第63卦)\n")
        print_transformation_report(63)


def run_tests():
    """テストケースを実行"""
    print("=== テスト実行 ===\n")

    # テスト1: hexagram_to_lines
    print("テスト1: hexagram_to_lines")
    assert hexagram_to_lines(51) == [1, 0, 0, 1, 0, 0], "震為雷の爻が不正"
    assert hexagram_to_lines(1) == [1, 1, 1, 1, 1, 1], "乾為天の爻が不正"
    assert hexagram_to_lines(2) == [0, 0, 0, 0, 0, 0], "坤為地の爻が不正"
    print("  ✓ hexagram_to_lines 正常\n")

    # テスト2: lines_to_hexagram
    print("テスト2: lines_to_hexagram")
    assert lines_to_hexagram([1, 0, 0, 1, 0, 0]) == (51, "震為雷")
    assert lines_to_hexagram([1, 1, 1, 1, 1, 1]) == (1, "乾為天")
    assert lines_to_hexagram([0, 0, 0, 0, 0, 0]) == (2, "坤為地")
    print("  ✓ lines_to_hexagram 正常\n")

    # テスト3: 互卦 (nuclear)
    print("テスト3: get_nuclear_hexagram (互卦)")
    # 震為雷 [1,0,0,1,0,0]
    # 互卦: 下卦=2,3,4爻=[0,0,1]=艮, 上卦=3,4,5爻=[0,1,0]=坎
    # (艮, 坎) → 山水蒙(4)
    nuclear_id, nuclear_name = get_nuclear_hexagram(51)
    print(f"  震為雷(51) → 互卦: {nuclear_name}({nuclear_id})")
    assert nuclear_id == 4, f"震為雷の互卦は山水蒙(4)のはず、実際は{nuclear_id}"

    # 乾為天 [1,1,1,1,1,1]
    # 互卦: 下卦=2,3,4爻=[1,1,1]=乾, 上卦=3,4,5爻=[1,1,1]=乾 → 乾為天(1)
    nuclear_id, nuclear_name = get_nuclear_hexagram(1)
    print(f"  乾為天(1) → 互卦: {nuclear_name}({nuclear_id})")
    assert nuclear_id == 1, f"乾為天の互卦は乾為天(1)のはず、実際は{nuclear_id}"
    print("  ✓ get_nuclear_hexagram 正常\n")

    # テスト4: 覆卦 (inverted)
    print("テスト4: get_inverted_hexagram (覆卦)")
    # 震為雷 [1,0,0,1,0,0] → 反転 [0,0,1,0,0,1]
    # 下卦=[0,0,1]=艮, 上卦=[0,0,1]=艮 → 艮為山(52)
    inv_id, inv_name = get_inverted_hexagram(51)
    print(f"  震為雷(51) → 覆卦: {inv_name}({inv_id})")
    assert inv_id == 52, f"震為雷の覆卦は艮為山(52)のはず、実際は{inv_id}"

    # 乾為天 [1,1,1,1,1,1] → 反転 [1,1,1,1,1,1] = 乾為天
    inv_id, inv_name = get_inverted_hexagram(1)
    print(f"  乾為天(1) → 覆卦: {inv_name}({inv_id})")
    assert inv_id == 1, f"乾為天の覆卦は乾為天(1)のはず、実際は{inv_id}"
    print("  ✓ get_inverted_hexagram 正常\n")

    # テスト5: 錯卦 (complementary)
    print("テスト5: get_complementary_hexagram (錯卦)")
    # 震為雷 [1,0,0,1,0,0] → 反転 [0,1,1,0,1,1]
    # 下卦=[0,1,1]=巽, 上卦=[0,1,1]=巽 → 巽為風(57)
    comp_id, comp_name = get_complementary_hexagram(51)
    print(f"  震為雷(51) → 錯卦: {comp_name}({comp_id})")
    assert comp_id == 57, f"震為雷の錯卦は巽為風(57)のはず、実際は{comp_id}"

    # 乾為天 [1,1,1,1,1,1] → 反転 [0,0,0,0,0,0] = 坤為地(2)
    comp_id, comp_name = get_complementary_hexagram(1)
    print(f"  乾為天(1) → 錯卦: {comp_name}({comp_id})")
    assert comp_id == 2, f"乾為天の錯卦は坤為地(2)のはず、実際は{comp_id}"
    print("  ✓ get_complementary_hexagram 正常\n")

    # テスト6: 水火既済の変換
    print("テスト6: 水火既済(63)の変換")
    # 水火既済: 下卦=坎[0,1,0], 上卦=離[1,0,1] → [0,1,0,1,0,1]
    lines = hexagram_to_lines(63)
    print(f"  爻: {lines}")
    assert lines == [0, 1, 0, 1, 0, 1], f"水火既済の爻が不正: {lines}"

    # 互卦: 下卦=2,3,4爻=[1,0,1]=離, 上卦=3,4,5爻=[0,1,0]=坎 → 火水未済(64)
    nuclear_id, nuclear_name = get_nuclear_hexagram(63)
    print(f"  互卦: {nuclear_name}({nuclear_id})")
    assert nuclear_id == 64, f"水火既済の互卦は火水未済(64)のはず、実際は{nuclear_id}"

    # 覆卦: [0,1,0,1,0,1] → 反転 → [1,0,1,0,1,0]
    # 下卦=[1,0,1]=離, 上卦=[0,1,0]=坎 → 火水未済(64)
    inv_id, inv_name = get_inverted_hexagram(63)
    print(f"  覆卦: {inv_name}({inv_id})")
    assert inv_id == 64, f"水火既済の覆卦は火水未済(64)のはず、実際は{inv_id}"

    # 錯卦: [0,1,0,1,0,1] → 陰陽反転 → [1,0,1,0,1,0]
    # 下卦=[1,0,1]=離, 上卦=[0,1,0]=坎 → 火水未済(64)
    comp_id, comp_name = get_complementary_hexagram(63)
    print(f"  錯卦: {comp_name}({comp_id})")
    assert comp_id == 64, f"水火既済の錯卦は火水未済(64)のはず、実際は{comp_id}"
    print("  ✓ 水火既済の変換 正常\n")

    print("=== 全テスト完了 ===")


if __name__ == "__main__":
    main()
