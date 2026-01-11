#!/usr/bin/env python3
"""
64卦（六十四卦）と384爻の網羅性を分析するスクリプト
"""
import json
from pathlib import Path
from collections import defaultdict
from schema_v3 import Case, Hex

# 64卦の名称マッピング（上卦・下卦の組み合わせ）
HEXAGRAM_64_NAMES = {
    ("乾", "乾"): "01. 乾為天（けんいてん）",
    ("乾", "坤"): "11. 地天泰（ちてんたい）",
    ("乾", "震"): "34. 雷天大壮（らいてんたいそう）",
    ("乾", "巽"): "09. 風天小畜（ふうてんしょうちく）",
    ("乾", "坎"): "05. 水天需（すいてんじゅ）",
    ("乾", "離"): "13. 天火同人（てんかどうじん）",
    ("乾", "艮"): "26. 山天大畜（さんてんたいちく）",
    ("乾", "兌"): "10. 天沢履（てんたくり）",

    ("坤", "乾"): "12. 天地否（てんちひ）",
    ("坤", "坤"): "02. 坤為地（こんいち）",
    ("坤", "震"): "16. 雷地豫（らいちよ）",
    ("坤", "巽"): "20. 風地観（ふうちかん）",
    ("坤", "坎"): "08. 水地比（すいちひ）",
    ("坤", "離"): "36. 地火明夷（ちかめいい）",
    ("坤", "艮"): "23. 山地剥（さんちはく）",
    ("坤", "兌"): "19. 地沢臨（ちたくりん）",

    ("震", "乾"): "25. 天雷无妄（てんらいむぼう）",
    ("震", "坤"): "24. 地雷復（ちらいふく）",
    ("震", "震"): "51. 震為雷（しんいらい）",
    ("震", "巽"): "42. 風雷益（ふうらいえき）",
    ("震", "坎"): "03. 水雷屯（すいらいちゅん）",
    ("震", "離"): "21. 火雷噬嗑（からいぜいこう）",
    ("震", "艮"): "27. 山雷頤（さんらいい）",
    ("震", "兌"): "17. 沢雷随（たくらいずい）",

    ("巽", "乾"): "44. 天風姤（てんぷうこう）",
    ("巽", "坤"): "46. 地風升（ちふうしょう）",
    ("巽", "震"): "32. 雷風恒（らいふうこう）",
    ("巽", "巽"): "57. 巽為風（そんいふう）",
    ("巽", "坎"): "48. 水風井（すいふうせい）",
    ("巽", "離"): "37. 風火家人（ふうかかじん）",
    ("巽", "艮"): "18. 山風蠱（さんぷうこ）",
    ("巽", "兌"): "28. 沢風大過（たくふうたいか）",

    ("坎", "乾"): "06. 天水訟（てんすいしょう）",
    ("坎", "坤"): "07. 地水師（ちすいし）",
    ("坎", "震"): "40. 雷水解（らいすいかい）",
    ("坎", "巽"): "59. 風水渙（ふうすいかん）",
    ("坎", "坎"): "29. 坎為水（かんいすい）",
    ("坎", "離"): "64. 火水未済（かすいびせい）",
    ("坎", "艮"): "04. 山水蒙（さんすいもう）",
    ("坎", "兌"): "47. 沢水困（たくすいこん）",

    ("離", "乾"): "14. 天火大有（てんかたいゆう）",
    ("離", "坤"): "35. 地火明夷（ちかめいい）",
    ("離", "震"): "55. 雷火豊（らいかほう）",
    ("離", "巽"): "50. 火風鼎（かふうてい）",
    ("離", "坎"): "63. 水火既済（すいかきせい）",
    ("離", "離"): "30. 離為火（りいか）",
    ("離", "艮"): "22. 山火賁（さんかひ）",
    ("離", "兌"): "49. 沢火革（たくかかく）",

    ("艮", "乾"): "33. 天山遯（てんざんとん）",
    ("艮", "坤"): "15. 地山謙（ちざんけん）",
    ("艮", "震"): "62. 雷山小過（らいざんしょうか）",
    ("艮", "巽"): "53. 風山漸（ふうざんぜん）",
    ("艮", "坎"): "39. 水山蹇（すいざんけん）",
    ("艮", "離"): "56. 山火旅（さんかりょ）",
    ("艮", "艮"): "52. 艮為山（ごんいざん）",
    ("艮", "兌"): "31. 山沢損（さんたくそん）",

    ("兌", "乾"): "43. 天沢夬（てんたくかい）",
    ("兌", "坤"): "45. 地沢臨（ちたくりん）",
    ("兌", "震"): "54. 雷沢帰妹（らいたくきまい）",
    ("兌", "巽"): "61. 風沢中孚（ふうたくちゅうふ）",
    ("兌", "坎"): "60. 水沢節（すいたくせつ）",
    ("兌", "離"): "38. 火沢睽（かたくけい）",
    ("兌", "艮"): "41. 山沢損（さんたくそん）",
    ("兌", "兌"): "58. 兌為沢（だいたく）",
}

def get_hex_name(hex_enum):
    """Hex enumから八卦名を取得"""
    mapping = {
        Hex.QIAN: "乾",
        Hex.KUN: "坤",
        Hex.ZHEN: "震",
        Hex.XUN: "巽",
        Hex.KAN: "坎",
        Hex.LI: "離",
        Hex.GEN: "艮",
        Hex.DUI: "兌",
    }
    return mapping[hex_enum]

def main():
    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    # 64卦の出現回数を記録（3つの異なる組み合わせ位置で）
    hexagram_64_position1 = defaultdict(int)  # before + trigger
    hexagram_64_position2 = defaultdict(int)  # trigger + action
    hexagram_64_position3 = defaultdict(int)  # action + after
    hexagram_64_all = defaultdict(int)  # 全ポジションの合計

    total = 0

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            case = Case(**data)
            total += 1

            # 八卦名を取得
            before = get_hex_name(case.before_hex)
            trigger = get_hex_name(case.trigger_hex)
            action = get_hex_name(case.action_hex)
            after = get_hex_name(case.after_hex)

            # 3つの64卦を生成（上卦・下卦の組み合わせ）
            # 易経では上卦が先、下卦が後
            hex64_1 = (trigger, before)  # トリガーが上、初期状態が下
            hex64_2 = (action, trigger)  # 行動が上、トリガーが下
            hex64_3 = (after, action)   # 結果が上、行動が下

            hexagram_64_position1[hex64_1] += 1
            hexagram_64_position2[hex64_2] += 1
            hexagram_64_position3[hex64_3] += 1

            hexagram_64_all[hex64_1] += 1
            hexagram_64_all[hex64_2] += 1
            hexagram_64_all[hex64_3] += 1

    print("=== 64卦（六十四卦）の網羅性分析 ===\n")
    print(f"総事例数: {total}")
    print(f"理論上の64卦インスタンス数: {total * 3} (各事例が3つの64卦を生成)\n")

    # 網羅性チェック
    all_64_hexagrams = set(HEXAGRAM_64_NAMES.keys())
    covered_hexagrams = set(hexagram_64_all.keys())
    coverage_rate = len(covered_hexagrams) / 64 * 100

    print(f"【網羅性】")
    print(f"  使用されている64卦: {len(covered_hexagrams)}/64 ({coverage_rate:.1f}%)")

    uncovered = all_64_hexagrams - covered_hexagrams
    if uncovered:
        print(f"  ⚠️  未使用の64卦: {len(uncovered)}個\n")
        print("【未使用の64卦一覧】")
        for upper, lower in sorted(uncovered):
            hex_name = HEXAGRAM_64_NAMES[(upper, lower)]
            print(f"  {hex_name} (上卦: {upper}, 下卦: {lower})")
    else:
        print(f"  ✅ 全64卦を網羅！")

    # 使用頻度の分布
    print("\n【使用頻度の分布】")
    freq_distribution = defaultdict(int)
    for count in hexagram_64_all.values():
        freq_distribution[count] += 1

    print(f"  平均使用回数: {sum(hexagram_64_all.values()) / len(covered_hexagrams):.1f}回")
    print(f"  最多使用回数: {max(hexagram_64_all.values())}回")
    print(f"  最少使用回数: {min(hexagram_64_all.values())}回\n")

    # 使用が少ない64卦（5回未満）
    underused = [(pair, count) for pair, count in hexagram_64_all.items() if count < 5]
    if underused:
        print(f"【使用が少ない64卦（5回未満）】 {len(underused)}個")
        for (upper, lower), count in sorted(underused, key=lambda x: x[1]):
            hex_name = HEXAGRAM_64_NAMES.get((upper, lower), f"{upper}・{lower}")
            print(f"  {hex_name}: {count}回")

    # 頻出の64卦（上位20位）
    print("\n【頻出の64卦（上位20位）】")
    sorted_hexagrams = sorted(hexagram_64_all.items(), key=lambda x: x[1], reverse=True)
    for i, ((upper, lower), count) in enumerate(sorted_hexagrams[:20], 1):
        hex_name = HEXAGRAM_64_NAMES.get((upper, lower), f"{upper}・{lower}")
        pct = count / (total * 3) * 100
        print(f"{i:2d}. {hex_name}: {count}回 ({pct:.1f}%)")

    # 384爻についての考察
    print("\n" + "="*50)
    print("=== 384爻の網羅性について ===\n")

    print("【現状の分析】")
    print(f"  現在のスキーマでは、各事例に「爻の変化」の情報が含まれていません。")
    print(f"  384爻 = 64卦 × 6爻 の各爻が「変化する/しない」の情報が必要です。\n")

    print("【理論的な考察】")
    print(f"  ・64卦の網羅率: {coverage_rate:.1f}%")
    print(f"  ・各卦には6つの爻があります")
    print(f"  ・各爻は「変化する（陽→陰 or 陰→陽）」または「変化しない」")
    print(f"  ・理論上、各卦で最大64通りの変化パターン（2^6）があります\n")

    print("【384爻を網羅するために必要なこと】")
    print(f"  1. スキーマに「変爻（へんこう）」の情報を追加")
    print(f"     例: changing_lines: [1, 3, 5] (初爻、三爻、五爻が変化)")
    print(f"  2. 各64卦について、どの爻が変化したかを記録")
    print(f"  3. 現在{len(covered_hexagrams)}個の64卦をカバーしているので、")
    print(f"     各卦で6爻をカバーするには:")
    print(f"     {len(covered_hexagrams)} × 6 = {len(covered_hexagrams) * 6}爻")
    print(f"     (理論上の384爻の{len(covered_hexagrams) * 6 / 384 * 100:.1f}%)\n")

    print("【推奨事項】")
    print(f"  現段階では、64卦の網羅性を優先すべきです。")
    print(f"  ・未使用の64卦: {len(uncovered)}個を埋める")
    print(f"  ・使用が少ない64卦（5回未満）: {len(underused)}個を増やす")
    print(f"  ・384爻の実装は、64卦が十分にカバーされてから検討するのが良いでしょう。")

if __name__ == "__main__":
    main()
