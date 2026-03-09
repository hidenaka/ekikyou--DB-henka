#!/usr/bin/env python3
"""
重複検出・標本独立性分析スクリプト

検出基準:
  a. 完全一致: entity_name + period が完全一致
  b. 類似重複: entity_name一致 + before_summary/after_summary のJaccard類似度 >= 0.5
  c. ニアミス: entity_name部分一致 + period重複

出力:
  - analysis/quality/duplicate_candidates.json
  - analysis/quality/independence_report.md
"""

import json
import os
import sys
from collections import defaultdict, Counter
from itertools import combinations
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_PATH = os.path.join(BASE_DIR, "data", "raw", "cases.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "quality")


# =============================================================================
# ユーティリティ
# =============================================================================

def char_ngrams(text, n=2):
    """文字レベルのn-gramの集合を返す（日本語対応、MeCab不要）"""
    if not text:
        return set()
    # 記号・空白を除去して正規化
    text = re.sub(r'[\s\u3000、。・「」『』（）\(\)\[\]【】,\.!?！？]', '', text)
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a, set_b):
    """Jaccard類似度"""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def parse_period(period_str):
    """periodからおおまかな年範囲を抽出（重複判定用）"""
    if not period_str:
        return None, None
    years = re.findall(r'(\d{4})', str(period_str))
    if len(years) >= 2:
        return int(years[0]), int(years[-1])
    elif len(years) == 1:
        return int(years[0]), int(years[0])
    return None, None


def periods_overlap(p1_start, p1_end, p2_start, p2_end):
    """2つの期間が重複するか"""
    if any(v is None for v in [p1_start, p1_end, p2_start, p2_end]):
        return False
    return p1_start <= p2_end and p2_start <= p1_end


def is_partial_match(name_a, name_b):
    """entity_nameの部分一致（片方がもう片方を含む）

    誤検出を減らすため:
    - 短い方の名前が3文字未満なら除外
    - 短い方が長い方の長さの30%未満なら除外（「日本」が「日本マクドナルド」に一致するのを防ぐ）
    """
    if not name_a or not name_b:
        return False
    if name_a == name_b:
        return False  # 完全一致は別カテゴリ
    # 正規化: 括弧内を除去、装飾的サフィックスを除去
    def normalize(n):
        n = re.sub(r'[（\(].+?[）\)]', '', n)
        # 「_」や「・」以降の説明的テキストで始まるものは基本名だけ抽出
        n = re.split(r'[_\s]', n)[0] if '_' in n else n
        return n.strip()
    na, nb = normalize(name_a), normalize(name_b)
    shorter = min(len(na), len(nb))
    longer = max(len(na), len(nb))
    if shorter < 3:
        return False
    # 短い方が長い方の30%未満は除外
    if shorter / longer < 0.3:
        return False
    return na in nb or nb in na


# =============================================================================
# メイン処理
# =============================================================================

def load_cases():
    cases = []
    with open(CASES_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                cases.append(case)
            except json.JSONDecodeError:
                print(f"[WARN] Line {i+1}: JSON parse error, skipping")
    return cases


def detect_duplicates(cases):
    """重複検出を実行"""

    # entity_nameでグルーピング
    by_entity = defaultdict(list)
    for idx, c in enumerate(cases):
        name = c.get('target_name', '') or ''
        by_entity[name].append((idx, c))

    exact_matches = []      # (a) 完全一致
    similar_matches = []    # (b) 類似重複
    near_misses = []        # (c) ニアミス

    print("=== Phase 1: 完全一致 & 類似重複の検出 (同一entity_name内) ===")
    entity_count = 0
    for entity_name, group in by_entity.items():
        if len(group) < 2:
            continue
        entity_count += 1
        for (idx_a, ca), (idx_b, cb) in combinations(group, 2):
            period_a = ca.get('period', '')
            period_b = cb.get('period', '')

            # (a) 完全一致: entity_name + period
            if period_a and period_b and str(period_a) == str(period_b):
                exact_matches.append({
                    'type': 'exact',
                    'entity_name': entity_name,
                    'period': str(period_a),
                    'transition_id_a': ca.get('transition_id', f'idx_{idx_a}'),
                    'transition_id_b': cb.get('transition_id', f'idx_{idx_b}'),
                    'summary_a': (ca.get('story_summary') or '')[:100],
                    'summary_b': (cb.get('story_summary') or '')[:100],
                })
                continue

            # (b) 類似重複: Jaccard >= 0.5
            summary_a = ca.get('story_summary', '') or ''
            summary_b = cb.get('story_summary', '') or ''
            ngrams_a = char_ngrams(summary_a)
            ngrams_b = char_ngrams(summary_b)
            sim = jaccard_similarity(ngrams_a, ngrams_b)

            if sim >= 0.5:
                similar_matches.append({
                    'type': 'similar',
                    'entity_name': entity_name,
                    'jaccard': round(sim, 3),
                    'period_a': str(period_a),
                    'period_b': str(period_b),
                    'transition_id_a': ca.get('transition_id', f'idx_{idx_a}'),
                    'transition_id_b': cb.get('transition_id', f'idx_{idx_b}'),
                    'summary_a': summary_a[:100],
                    'summary_b': summary_b[:100],
                })

    print(f"  同一entity_nameに2+事例: {entity_count} entities")
    print(f"  完全一致: {len(exact_matches)} ペア")
    print(f"  類似重複 (Jaccard>=0.5): {len(similar_matches)} ペア")

    # (c) ニアミス: entity_name部分一致 + period重複
    print("\n=== Phase 2: ニアミス検出 (entity_name部分一致 + period重複) ===")
    entity_names = list(by_entity.keys())
    # 部分一致候補を効率的に検出
    checked = 0
    for i in range(len(entity_names)):
        na = entity_names[i]
        if len(na) < 2:
            continue
        for j in range(i+1, len(entity_names)):
            nb = entity_names[j]
            if len(nb) < 2:
                continue
            if not is_partial_match(na, nb):
                continue
            # period重複チェック
            for idx_a, ca in by_entity[na]:
                pa_s, pa_e = parse_period(ca.get('period', ''))
                for idx_b, cb in by_entity[nb]:
                    pb_s, pb_e = parse_period(cb.get('period', ''))
                    if periods_overlap(pa_s, pa_e, pb_s, pb_e):
                        # summaryの類似度も計算してノイズを減らす
                        sa = ca.get('story_summary', '') or ''
                        sb = cb.get('story_summary', '') or ''
                        sim = jaccard_similarity(char_ngrams(sa), char_ngrams(sb))
                        near_misses.append({
                            'type': 'near_miss',
                            'entity_name_a': na,
                            'entity_name_b': nb,
                            'period_a': str(ca.get('period', '')),
                            'period_b': str(cb.get('period', '')),
                            'jaccard': round(sim, 3),
                            'transition_id_a': ca.get('transition_id', f'idx_{idx_a}'),
                            'transition_id_b': cb.get('transition_id', f'idx_{idx_b}'),
                            'summary_a': sa[:100],
                            'summary_b': sb[:100],
                        })
            checked += 1

    print(f"  部分一致ペア数: {checked}")
    print(f"  ニアミス (部分一致+period重複): {len(near_misses)} ペア")

    return exact_matches, similar_matches, near_misses


def analyze_independence(cases):
    """独立性分析: entity別の事例数分布"""

    entity_counts = Counter()
    for c in cases:
        name = c.get('target_name', '') or '(empty)'
        entity_counts[name] += 1

    # 分布統計
    count_dist = Counter(entity_counts.values())  # 事例数ごとのentity数
    top_20 = entity_counts.most_common(20)
    total = len(cases)
    top_10_sum = sum(cnt for _, cnt in entity_counts.most_common(10))
    top_10_pct = (top_10_sum / total * 100) if total > 0 else 0
    top_20_sum = sum(cnt for _, cnt in top_20)
    top_20_pct = (top_20_sum / total * 100) if total > 0 else 0

    # 1件のみのentity数
    single_entities = sum(1 for cnt in entity_counts.values() if cnt == 1)
    multi_entities = sum(1 for cnt in entity_counts.values() if cnt > 1)

    stats = {
        'total_cases': total,
        'unique_entities': len(entity_counts),
        'single_case_entities': single_entities,
        'multi_case_entities': multi_entities,
        'top_10_concentration_pct': round(top_10_pct, 2),
        'top_20_concentration_pct': round(top_20_pct, 2),
        'top_20_entities': [{'entity': name, 'count': cnt} for name, cnt in top_20],
        'count_distribution': {str(k): v for k, v in sorted(count_dist.items())},
    }

    return stats


def generate_report(exact, similar, near_miss, independence_stats):
    """Markdownレポート生成"""

    total_dup = len(exact) + len(similar) + len(near_miss)

    lines = []
    lines.append("# 重複排除・標本独立性分析レポート")
    lines.append("")
    lines.append(f"**分析日**: 2026-03-09")
    lines.append(f"**総事例数**: {independence_stats['total_cases']:,}")
    lines.append(f"**ユニークentity数**: {independence_stats['unique_entities']:,}")
    lines.append("")

    # 1. 重複候補サマリー
    lines.append("## 1. 重複候補サマリー")
    lines.append("")
    lines.append(f"| カテゴリ | 件数 | 説明 |")
    lines.append(f"|---------|------|------|")
    lines.append(f"| 完全一致 | {len(exact)} | entity_name + period が完全一致 |")
    lines.append(f"| 類似重複 | {len(similar)} | entity_name一致 + summary Jaccard >= 0.5 |")
    lines.append(f"| ニアミス | {len(near_miss)} | entity_name部分一致 + period重複 |")
    lines.append(f"| **合計** | **{total_dup}** | |")
    lines.append("")

    # 2. Entity集中度
    lines.append("## 2. Entity集中度")
    lines.append("")
    lines.append(f"- 1事例のみのentity: {independence_stats['single_case_entities']:,}")
    lines.append(f"- 複数事例を持つentity: {independence_stats['multi_case_entities']:,}")
    lines.append(f"- **上位10 entityが全体の {independence_stats['top_10_concentration_pct']}% を占有**")
    lines.append(f"- 上位20 entityが全体の {independence_stats['top_20_concentration_pct']}% を占有")
    lines.append("")

    lines.append("### 上位20 Entity")
    lines.append("")
    lines.append("| # | Entity | 事例数 |")
    lines.append("|---|--------|--------|")
    for i, item in enumerate(independence_stats['top_20_entities'], 1):
        lines.append(f"| {i} | {item['entity']} | {item['count']} |")
    lines.append("")

    # 3. 事例数分布
    lines.append("### 事例数分布 (1 entityあたりの事例数)")
    lines.append("")
    lines.append("| 事例数 | Entity数 |")
    lines.append("|--------|----------|")
    for k, v in sorted(independence_stats['count_distribution'].items(), key=lambda x: int(x[0])):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # 4. 標本独立性への影響評価
    lines.append("## 3. 標本独立性への影響評価")
    lines.append("")

    # 影響度を定量的に判定
    exact_severity = "高" if len(exact) > 50 else ("中" if len(exact) > 10 else "低")
    similar_severity = "高" if len(similar) > 100 else ("中" if len(similar) > 20 else "低")
    concentration_severity = "高" if independence_stats['top_10_concentration_pct'] > 10 else ("中" if independence_stats['top_10_concentration_pct'] > 5 else "低")

    lines.append("### 3.1 重複による影響")
    lines.append("")
    lines.append(f"- **完全一致ペアのリスク: {exact_severity}**")
    lines.append(f"  - 同一事象の多重記録の可能性が高い。検定の標本独立性前提を直接破壊する")
    lines.append(f"- **類似重複のリスク: {similar_severity}**")
    lines.append(f"  - 同一事象を別角度で記述した可能性。手動レビューが必要")
    lines.append(f"- **ニアミスのリスク: 要確認**")
    lines.append(f"  - 表記揺れによる同一entity（例: 'トヨタ' vs 'トヨタ自動車'）の可能性")
    lines.append("")

    lines.append("### 3.2 集中度による影響")
    lines.append("")
    lines.append(f"- **リスク: {concentration_severity}**")
    lines.append(f"- 上位10 entityが全体の {independence_stats['top_10_concentration_pct']}% を占める")
    if independence_stats['top_10_concentration_pct'] > 5:
        lines.append(f"- 同一entityの複数事例は、entity固有の特性（業界、規模、文化）が交絡因子となる")
        lines.append(f"- クラスタリングされたデータとして扱い、混合効果モデルや階層ベイズの適用を検討すべき")
    lines.append("")

    # 5. 独立性判定基準の提案
    lines.append("### 3.3 同一entityの複数事例が独立と見なせる条件（提案）")
    lines.append("")
    lines.append("以下の **全て** を満たす場合、同一entityの2事例は独立標本として扱える:")
    lines.append("")
    lines.append("1. **時間的分離**: 2事例のperiodが5年以上離れている")
    lines.append("2. **フェーズの相違**: event_phase / pattern_type が異なる")
    lines.append("3. **トリガーの相違**: trigger_type が異なる（同一トリガーの連鎖でない）")
    lines.append("4. **要約の非類似**: story_summary の Jaccard類似度が 0.3 未満")
    lines.append("")
    lines.append("上記を満たさない場合:")
    lines.append("- entity_idによるクラスタリング + cluster-robust standard errorsの適用")
    lines.append("- または、entityごとに最も情報量の多い1事例を代表として選択")
    lines.append("")

    # 6. 推奨アクション
    lines.append("## 4. 推奨アクション")
    lines.append("")
    lines.append("| 優先度 | アクション | 理由 |")
    lines.append("|--------|-----------|------|")
    if len(exact) > 0:
        lines.append(f"| P0 | 完全一致 {len(exact)}ペアの手動レビュー・統合 | 同一事象の二重カウント排除 |")
    if len(similar) > 0:
        lines.append(f"| P1 | 類似重複 {len(similar)}ペアの手動レビュー | 独立事象か同一事象かの判定 |")
    if len(near_miss) > 0:
        lines.append(f"| P1 | ニアミス {len(near_miss)}ペアのentity名正規化 | 表記揺れの統一 |")
    if independence_stats['top_10_concentration_pct'] > 5:
        lines.append(f"| P2 | 集中entity({independence_stats['top_10_concentration_pct']}%)への対策 | entity-levelクラスタリング or 代表事例選択 |")
    lines.append(f"| P2 | 検定実施時にcluster-robust SEの適用 | 残存する非独立性への対処 |")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Loading cases...")
    cases = load_cases()
    print(f"Loaded {len(cases):,} cases\n")

    # 重複検出
    exact, similar, near_miss = detect_duplicates(cases)

    # 独立性分析
    print("\n=== Phase 3: 独立性分析 ===")
    independence_stats = analyze_independence(cases)
    print(f"  ユニークentity数: {independence_stats['unique_entities']:,}")
    print(f"  1事例entity: {independence_stats['single_case_entities']:,}")
    print(f"  複数事例entity: {independence_stats['multi_case_entities']:,}")
    print(f"  Top10集中度: {independence_stats['top_10_concentration_pct']}%")

    # 重複候補をJSON保存
    dup_output = {
        'summary': {
            'total_cases': len(cases),
            'exact_match_pairs': len(exact),
            'similar_pairs': len(similar),
            'near_miss_pairs': len(near_miss),
            'total_duplicate_candidates': len(exact) + len(similar) + len(near_miss),
        },
        'exact_matches': exact,
        'similar_matches': similar,
        'near_misses': near_miss,
    }

    dup_path = os.path.join(OUTPUT_DIR, "duplicate_candidates.json")
    with open(dup_path, 'w', encoding='utf-8') as f:
        json.dump(dup_output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 重複候補: {dup_path}")

    # レポート生成
    report = generate_report(exact, similar, near_miss, independence_stats)
    report_path = os.path.join(OUTPUT_DIR, "independence_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] レポート: {report_path}")

    # サマリー出力
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  完全一致ペア:     {len(exact):>6}")
    print(f"  類似重複ペア:     {len(similar):>6}")
    print(f"  ニアミスペア:     {len(near_miss):>6}")
    print(f"  合計候補:         {len(exact) + len(similar) + len(near_miss):>6}")
    print(f"  Top10 entity集中: {independence_stats['top_10_concentration_pct']:>5}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
