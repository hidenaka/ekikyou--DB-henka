#!/usr/bin/env python3
"""
64卦遷移マップ生成スクリプト

cases.jsonlから以下を抽出・集計:
- classical_before_hexagram → classical_after_hexagram の全遷移パターン
- 各遷移の件数、成功率、主要action_hex

卦名は hexagram_master.json を参照して正規化。
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# パス設定
BASE_DIR = Path(__file__).parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "hexagrams" / "transition_map.json"
HEXAGRAM_MASTER_FILE = BASE_DIR / "data" / "hexagrams" / "hexagram_master.json"


def load_hexagram_master():
    """hexagram_master.json を読み込み、正規化マップを構築"""
    with open(HEXAGRAM_MASTER_FILE, 'r', encoding='utf-8') as f:
        master = json.load(f)

    # 正規化マップ: 様々な表記 → 正式名称
    normalize_map = {}

    for hex_id, data in master.items():
        name = data['name']       # 例: "乾為天", "坎為水"
        chinese = data['chinese'] # 例: "乾", "坎"

        # 正式名称自身
        normalize_map[name] = name

        # "N_中文" 形式 (例: "1_乾" → "乾為天")
        normalize_map[f"{hex_id}_{chinese}"] = name

        # 数字だけ (例: "1" → "乾為天")
        normalize_map[hex_id] = name

        # 中文だけ (例: "乾" → "乾為天") - ただし重複に注意
        if chinese not in normalize_map:
            normalize_map[chinese] = name

    # 特殊ケース: "習坎" = "坎為水" (伝統的な別名)
    normalize_map["習坎"] = "坎為水"

    # 別表記のバリエーション
    normalize_map["天山遁"] = "天山遯"  # 遁→遯
    normalize_map["天雷無妄"] = "天雷无妄"  # 無→无

    return normalize_map


def normalize_hexagram_name(name, normalize_map):
    """卦名を正規化"""
    if not name:
        return None

    # マップにあればその値を返す
    if name in normalize_map:
        return normalize_map[name]

    # マップにない場合はそのまま返す（警告付き）
    return name


def load_cases():
    """cases.jsonlを読み込む"""
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def extract_transitions(cases, normalize_map):
    """
    遷移データを抽出（卦名を正規化）

    Returns:
        transitions: {before_hex: {after_hex: [case_data, ...]}}
    """
    transitions = defaultdict(lambda: defaultdict(list))
    unknown_names = set()

    for case in cases:
        before_hex_raw = case.get('classical_before_hexagram')
        after_hex_raw = case.get('classical_after_hexagram')
        action_hex = case.get('action_hex')
        outcome = case.get('outcome')

        # 必須フィールドがない場合はスキップ
        if not before_hex_raw or not after_hex_raw:
            continue

        # 正規化
        before_hex = normalize_hexagram_name(before_hex_raw, normalize_map)
        after_hex = normalize_hexagram_name(after_hex_raw, normalize_map)

        # 正規化できなかった名前を記録
        if before_hex_raw not in normalize_map:
            unknown_names.add(before_hex_raw)
        if after_hex_raw not in normalize_map:
            unknown_names.add(after_hex_raw)

        transitions[before_hex][after_hex].append({
            'action_hex': action_hex,
            'outcome': outcome,
            'transition_id': case.get('transition_id'),
            'target_name': case.get('target_name'),
        })

    # 未知の卦名があれば警告
    if unknown_names:
        print(f"警告: 正規化できなかった卦名: {unknown_names}")

    return transitions


def calculate_success_rate(case_list):
    """成功率を計算"""
    if not case_list:
        return 0.0

    success_count = sum(1 for c in case_list if c['outcome'] == 'Success')
    return round(success_count / len(case_list), 3)


def count_actions(case_list):
    """action_hexの出現回数を集計"""
    action_counts = defaultdict(int)
    for c in case_list:
        action = c.get('action_hex')
        if action:
            action_counts[action] += 1
    return dict(sorted(action_counts.items(), key=lambda x: -x[1]))


def get_main_action(action_counts):
    """最も多く使われたaction_hexを取得"""
    if not action_counts:
        return None
    return max(action_counts.keys(), key=lambda k: action_counts[k])


def build_transition_map(transitions):
    """
    遷移マップを構築

    Returns:
        {
            "transitions": {before_hex: {after_hex: stats, ...}, ...},
            "metadata": {...}
        }
    """
    result = {}
    total_cases = 0
    unique_before = set()
    unique_after = set()

    for before_hex, after_dict in transitions.items():
        unique_before.add(before_hex)
        result[before_hex] = {}

        for after_hex, case_list in after_dict.items():
            unique_after.add(after_hex)
            count = len(case_list)
            total_cases += count

            action_counts = count_actions(case_list)
            main_action = get_main_action(action_counts)
            success_rate = calculate_success_rate(case_list)

            result[before_hex][after_hex] = {
                'count': count,
                'success_rate': success_rate,
                'main_action': main_action,
                'actions': action_counts
            }

    # 結果をソート（件数順）
    sorted_result = {}
    for before_hex in sorted(result.keys()):
        sorted_after = dict(sorted(
            result[before_hex].items(),
            key=lambda x: -x[1]['count']
        ))
        sorted_result[before_hex] = sorted_after

    # ユニークな卦の数を計算
    unique_hexagrams = unique_before | unique_after

    return {
        'transitions': sorted_result,
        'metadata': {
            'total_cases': total_cases,
            'unique_hexagrams': len(unique_hexagrams),
            'unique_before_hexagrams': len(unique_before),
            'unique_after_hexagrams': len(unique_after),
            'total_transition_patterns': sum(len(v) for v in sorted_result.values()),
            'generated_at': datetime.now().strftime('%Y-%m-%d')
        }
    }


def generate_summary(transition_map):
    """サマリー統計を出力"""
    meta = transition_map['metadata']
    transitions = transition_map['transitions']

    print("\n=== 64卦遷移マップ生成完了 ===")
    print(f"総事例数: {meta['total_cases']:,}件")
    print(f"ユニーク卦数: {meta['unique_hexagrams']}")
    print(f"  - Before卦: {meta['unique_before_hexagrams']}")
    print(f"  - After卦: {meta['unique_after_hexagrams']}")
    print(f"遷移パターン数: {meta['total_transition_patterns']}")

    # 最も多い遷移パターンTop10
    all_transitions = []
    for before_hex, after_dict in transitions.items():
        for after_hex, stats in after_dict.items():
            all_transitions.append({
                'from': before_hex,
                'to': after_hex,
                'count': stats['count'],
                'success_rate': stats['success_rate'],
                'main_action': stats['main_action']
            })

    top_transitions = sorted(all_transitions, key=lambda x: -x['count'])[:10]

    print("\n--- 遷移パターン Top 10 ---")
    for i, t in enumerate(top_transitions, 1):
        print(f"{i:2}. {t['from']} → {t['to']}: "
              f"{t['count']}件 (成功率: {t['success_rate']:.1%}, "
              f"主要action: {t['main_action']})")

    # 成功率が高い遷移（10件以上）
    high_success = [t for t in all_transitions if t['count'] >= 10 and t['success_rate'] >= 0.8]
    high_success = sorted(high_success, key=lambda x: -x['success_rate'])[:10]

    print("\n--- 高成功率遷移 (10件以上, 80%以上) Top 10 ---")
    for i, t in enumerate(high_success, 1):
        print(f"{i:2}. {t['from']} → {t['to']}: "
              f"{t['count']}件 (成功率: {t['success_rate']:.1%}, "
              f"主要action: {t['main_action']})")


def main():
    print("Loading hexagram master for normalization...")
    normalize_map = load_hexagram_master()
    print(f"Loaded {len(normalize_map)} normalization mappings")

    print("Loading cases...")
    cases = load_cases()
    print(f"Loaded {len(cases):,} cases")

    print("Extracting transitions (with normalization)...")
    transitions = extract_transitions(cases, normalize_map)

    print("Building transition map...")
    transition_map = build_transition_map(transitions)

    # 出力ディレクトリ確認
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(transition_map, f, ensure_ascii=False, indent=2)

    generate_summary(transition_map)

    print(f"\n出力ファイル: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
