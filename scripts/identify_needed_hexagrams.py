#!/usr/bin/env python3
"""
不足している64卦を増やすために必要な八卦の組み合わせを特定
"""

# 不足している5つの64卦
underused_hexagrams = {
    "45. 地沢臨": {
        "upper": "坤",  # 地
        "lower": "兌",  # 沢
        "current": 1,
        "target": 20,
        "needed": 19,
        "meaning": "地沢臨 - 上から臨む、上位者が下位者を導く",
        "situation": "権力者が民衆を導く、上司が部下を育てる、親が子を導く"
    },
    "58. 兌為沢": {
        "upper": "兌",  # 沢
        "lower": "兌",  # 沢
        "current": 1,
        "target": 20,
        "needed": 19,
        "meaning": "兌為沢 - 喜びが重なる、喜悦、交流",
        "situation": "喜びの連鎖、成功の連続、人々との和やかな交流、祝福"
    },
    "12. 天地否": {
        "upper": "坤",  # 地
        "lower": "乾",  # 天
        "current": 2,
        "target": 20,
        "needed": 18,
        "meaning": "天地否 - 天地が交わらない、停滞、閉塞",
        "situation": "意思疎通の断絶、上下の不調和、努力が報われない"
    },
    "51. 震為雷": {
        "upper": "震",  # 雷
        "lower": "震",  # 雷
        "current": 2,
        "target": 20,
        "needed": 18,
        "meaning": "震為雷 - 雷が重なる、大きな衝撃、震動",
        "situation": "連続する衝撃、大事件、震撼、驚愕の連続"
    },
    "31. 山沢損": {
        "upper": "艮",  # 山
        "lower": "兌",  # 沢
        "current": 4,
        "target": 20,
        "needed": 16,
        "meaning": "山沢損 - 下を削って上を満たす、犠牲、損失",
        "situation": "何かを捨てて何かを得る、犠牲を払う、断捨離"
    }
}

print("=== 不足している64卦を増やすための事例収集ガイド ===\n")

total_needed = sum(h["needed"] for h in underused_hexagrams.values())
print(f"合計で {total_needed} 件の事例が必要です\n")

print("各64卦は（上卦、下卦）のペアで構成され、各事例は3つのペアを生成します：")
print("  ペア1: (trigger_hex, before_hex)")
print("  ペア2: (action_hex, trigger_hex)")
print("  ペア3: (after_hex, action_hex)\n")

print("="*70)

for hex_name, info in underused_hexagrams.items():
    print(f"\n【{hex_name}】")
    print(f"  意味: {info['meaning']}")
    print(f"  現在: {info['current']}回 → 目標: {info['target']}回 (あと{info['needed']}件必要)")
    print(f"  上卦: {info['upper']}, 下卦: {info['lower']}")
    print(f"  典型的な状況: {info['situation']}")

    print(f"\n  この卦を生成するための八卦の組み合わせ:")
    print(f"    パターンA: trigger={info['upper']}, before={info['lower']}")
    print(f"    パターンB: action={info['upper']}, trigger={info['lower']}")
    print(f"    パターンC: after={info['upper']}, action={info['lower']}")

    # 各八卦の意味
    hex_meanings = {
        "乾": "天・創造・剛健・強さ",
        "坤": "地・受容・柔順・基盤",
        "震": "雷・動き・奮起・衝撃",
        "巽": "風・浸透・柔軟・対話",
        "坎": "水・危険・困難・試練",
        "離": "火・明知・分離・才能",
        "艮": "山・止まる・待機・蓄積",
        "兌": "沢・喜び・和悦・成功"
    }

    upper_meaning = hex_meanings[info['upper']]
    lower_meaning = hex_meanings[info['lower']]

    print(f"\n  八卦の意味:")
    print(f"    上卦 {info['upper']}（{upper_meaning}）")
    print(f"    下卦 {info['lower']}（{lower_meaning}）")

print("\n" + "="*70)
print("\n【収集戦略】")
print("\n最も効率的な収集方法:")
print("  1. 各卦について約20件の事例を収集")
print("  2. 上記のパターンA/B/Cのいずれかを満たす事例を探す")
print("  3. 合計で約20件の新規事例を追加すれば、複数の不足卦を同時にカバー可能")
print("\n収集の優先順位:")
print("  1位: 45. 地沢臨 (1回) - 坤→兌の組み合わせ")
print("  2位: 58. 兌為沢 (1回) - 兌→兌の組み合わせ")
print("  3位: 12. 天地否 (2回) - 坤→乾の組み合わせ")
print("  4位: 51. 震為雷 (2回) - 震→震の組み合わせ")
print("  5位: 31. 山沢損 (4回) - 艮→兌の組み合わせ")
