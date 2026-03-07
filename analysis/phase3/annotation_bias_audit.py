#!/usr/bin/env python3
"""
Phase 3: アノテーションバイアス検証 — Δ_lower == Δ_upper の97.8%対角偏向は本物か？

背景:
  cases.jsonlのΔ = before XOR after (6bit) において、
  下3bit(下卦変化) == 上3bit(上卦変化) が97.8%を占める。
  LLMアノテーションバイアスか構造的特性かを検証する。

分析内容:
  Part 1: Δ_lower × Δ_upper 8×8分割表 + χ²検定
  Part 2: 例外事例（Δ_lower ≠ Δ_upper）の監査
  Part 3: 対角事例のΔパターン分析
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from scipy.stats import chi2_contingency

# isomorphism_test.py から共有ユーティリティをインポート
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isomorphism_test import (
    load_json,
    load_cases,
    build_name_to_kw,
    build_kw_to_bits,
    resolve_hexagram_field,
    TRIGRAM_BITS,
    REFERENCE_FILE,
    CASES_FILE,
)

# ---------- パス設定 ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PHASE3_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = PHASE3_DIR / "annotation_bias_audit.json"

# ---------- 八卦名マッピング ----------
TRIGRAM_NAMES = {
    (0, 0, 0): '坤(地)',
    (0, 0, 1): '艮(山)',
    (0, 1, 0): '坎(水)',
    (0, 1, 1): '巽(風)',
    (1, 0, 0): '震(雷)',
    (1, 0, 1): '離(火)',
    (1, 1, 0): '兌(沢)',
    (1, 1, 1): '乾(天)',
}

# Δパターンの変換意味（3bit）
DELTA_TRIGRAM_MEANING = {
    (0, 0, 0): '変化なし',
    (0, 0, 1): '1爻変 (初爻/四爻)',
    (0, 1, 0): '1爻変 (二爻/五爻)',
    (0, 1, 1): '2爻変',
    (1, 0, 0): '1爻変 (三爻/上爻)',
    (1, 0, 1): '2爻変',
    (1, 1, 0): '2爻変',
    (1, 1, 1): '全爻変 (錯卦)',
}


# ============================================================
# ユーティリティ
# ============================================================

def xor_bits(bits_a, bits_b):
    """2つの6bitタプルのXOR"""
    return tuple(a ^ b for a, b in zip(bits_a, bits_b))


def bits_to_str(bits):
    """bitタプルを文字列に変換"""
    return ''.join(str(b) for b in bits)


def extract_transitions_with_metadata(cases, name_to_kw, kw_to_bits):
    """cases からΔベクトルとメタデータを含む遷移リストを抽出"""
    transitions = []
    n_excluded = 0

    for idx, case in enumerate(cases):
        before_val = case.get('classical_before_hexagram', '')
        after_val = case.get('classical_after_hexagram', '')

        kw_from = resolve_hexagram_field(before_val, name_to_kw)
        kw_to = resolve_hexagram_field(after_val, name_to_kw)

        if kw_from is None or kw_to is None:
            n_excluded += 1
            continue
        if kw_from not in kw_to_bits or kw_to not in kw_to_bits:
            n_excluded += 1
            continue

        bits_from = kw_to_bits[kw_from]
        bits_to = kw_to_bits[kw_to]
        delta = xor_bits(bits_from, bits_to)
        delta_lower = delta[:3]
        delta_upper = delta[3:]

        transitions.append({
            'index': idx,
            'transition_id': case.get('transition_id', ''),
            'target_name': case.get('target_name', ''),
            'before_hexagram': before_val,
            'after_hexagram': after_val,
            'kw_from': kw_from,
            'kw_to': kw_to,
            'bits_from': bits_from,
            'bits_to': bits_to,
            'delta': delta,
            'delta_lower': delta_lower,
            'delta_upper': delta_upper,
            'delta_lower_int': int(bits_to_str(delta_lower), 2),
            'delta_upper_int': int(bits_to_str(delta_upper), 2),
            'is_diagonal': delta_lower == delta_upper,
            'main_domain': case.get('main_domain', ''),
            'scale': case.get('scale', ''),
            'source_type': case.get('source_type', ''),
        })

    return transitions, n_excluded


# ============================================================
# Part 1: Δ_lower × Δ_upper 8×8分割表
# ============================================================

def part1_contingency_table(transitions):
    """8×8分割表 + χ²検定"""
    print("\n" + "=" * 70)
    print("[Part 1] Δ_lower x Δ_upper 8x8分割表")
    print("=" * 70)

    total = len(transitions)

    # 8×8分割表構築
    table = np.zeros((8, 8), dtype=int)
    for t in transitions:
        table[t['delta_lower_int']][t['delta_upper_int']] += 1

    # 対角 vs 非対角
    diag_count = sum(table[i][i] for i in range(8))
    off_diag_count = total - diag_count
    diag_ratio = diag_count / total if total > 0 else 0

    print(f"\n  総遷移数: {total}")
    print(f"  対角要素 (Δ_lower == Δ_upper): {diag_count} ({diag_ratio:.4f} = {diag_ratio*100:.1f}%)")
    print(f"  非対角要素 (Δ_lower != Δ_upper): {off_diag_count} ({1-diag_ratio:.4f} = {(1-diag_ratio)*100:.1f}%)")

    # 8×8テーブル表示（Markdownテーブル）
    print(f"\n### 8x8分割表")
    header = "| Δ_lower\\Δ_upper |"
    for j in range(8):
        header += f" {j:>3}({bits_to_str((j>>2, (j>>1)&1, j&1))}) |"
    header += " Row合計 |"
    print(header)
    separator = "|" + "-" * 17 + "|"
    for j in range(8):
        separator += "-" * 14 + "|"
    separator += "-" * 9 + "|"
    print(separator)

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    for i in range(8):
        bits_i = (i >> 2, (i >> 1) & 1, i & 1)
        row = f"| {i:>1}({bits_to_str(bits_i):>3})          |"
        for j in range(8):
            cell = table[i][j]
            marker = " *" if i == j and cell > 0 else "  "
            row += f" {cell:>5}{marker:>5}   |"
            # Simpler formatting
        # Redo with simpler format
        row = f"| {i}({bits_to_str(bits_i)})           |"
        for j in range(8):
            cell = table[i][j]
            if i == j:
                row += f" **{cell:>5}** |"
            else:
                row += f"   {cell:>5}   |"
        row += f" {row_sums[i]:>7} |"
        print(row)

    # Col合計
    col_row = "| Col合計         |"
    for j in range(8):
        col_row += f"   {col_sums[j]:>5}   |"
    col_row += f" {total:>7} |"
    print(col_row)

    # 独立仮定時の期待値
    print(f"\n### 独立仮定時の期待値との比較")
    expected_diag = 0
    for i in range(8):
        expected_diag_i = row_sums[i] * col_sums[i] / total if total > 0 else 0
        expected_diag += expected_diag_i

    expected_diag_ratio = expected_diag / total if total > 0 else 0
    enrichment = diag_ratio / expected_diag_ratio if expected_diag_ratio > 0 else float('inf')

    print(f"  独立仮定時の対角期待値: {expected_diag:.1f} ({expected_diag_ratio:.4f} = {expected_diag_ratio*100:.1f}%)")
    print(f"  観測対角:               {diag_count} ({diag_ratio:.4f} = {diag_ratio*100:.1f}%)")
    print(f"  対角エンリッチメント:   {enrichment:.2f}x")

    # χ²検定（対角偏向の有意性）
    # 方法1: 8×8テーブル全体のχ²独立性検定
    chi2_stat, chi2_p, dof, expected_table = chi2_contingency(table)
    print(f"\n### χ²独立性検定（8x8テーブル全体）")
    print(f"  χ² = {chi2_stat:.2f}")
    print(f"  df = {dof}")
    print(f"  p = {chi2_p:.2e}")

    # Cramer's V
    n = total
    k = min(table.shape)
    cramers_v = np.sqrt(chi2_stat / (n * (k - 1))) if n > 0 and k > 1 else 0
    print(f"  Cramer's V = {cramers_v:.4f}")

    # 方法2: 2×2テーブル（対角 vs 非対角）の検定
    # observed: [diag, off_diag], expected under independence: [expected_diag, total - expected_diag]
    from scipy.stats import chisquare
    chi2_2x2, p_2x2 = chisquare(
        [diag_count, off_diag_count],
        f_exp=[expected_diag, total - expected_diag]
    )
    print(f"\n### χ²適合度検定（対角 vs 非対角、独立仮定期待値との比較）")
    print(f"  χ² = {chi2_2x2:.2f}")
    print(f"  p = {p_2x2:.2e}")

    result = {
        'total': int(total),
        'diagonal_count': int(diag_count),
        'off_diagonal_count': int(off_diag_count),
        'diagonal_ratio': round(float(diag_ratio), 6),
        'expected_diagonal_under_independence': round(float(expected_diag), 1),
        'expected_diagonal_ratio': round(float(expected_diag_ratio), 6),
        'diagonal_enrichment': round(float(enrichment), 2),
        'chi2_8x8': {
            'statistic': round(float(chi2_stat), 2),
            'p_value': float(chi2_p),
            'df': int(dof),
            'cramers_v': round(float(cramers_v), 4),
        },
        'chi2_diag_vs_offdiag': {
            'statistic': round(float(chi2_2x2), 2),
            'p_value': float(p_2x2),
        },
        'contingency_table': table.tolist(),
    }

    return result


# ============================================================
# Part 2: 例外事例（Δ_lower ≠ Δ_upper）の監査
# ============================================================

def part2_exception_audit(transitions, name_to_kw):
    """非対角事例の全件抽出とパターン分析"""
    print("\n" + "=" * 70)
    print("[Part 2] 例外事例の監査 (Δ_lower != Δ_upper)")
    print("=" * 70)

    total = len(transitions)
    exceptions = [t for t in transitions if not t['is_diagonal']]
    diag_cases = [t for t in transitions if t['is_diagonal']]
    n_exceptions = len(exceptions)

    print(f"\n  例外事例数: {n_exceptions} / {total} ({n_exceptions/total*100:.1f}%)")

    # --- main_domain別の例外率 ---
    print(f"\n### main_domain別の例外率")
    domain_total = Counter(t['main_domain'] for t in transitions if t['main_domain'])
    domain_exception = Counter(t['main_domain'] for t in exceptions if t['main_domain'])

    print(f"| main_domain | 総数 | 例外数 | 例外率 |")
    print(f"|-------------|------|--------|--------|")

    domain_stats = []
    for domain in sorted(domain_total.keys(), key=lambda d: domain_total[d], reverse=True):
        dt = domain_total[domain]
        de = domain_exception.get(domain, 0)
        rate = de / dt if dt > 0 else 0
        if dt >= 10:  # 10件以上のドメインのみ表示
            print(f"| {domain:<20} | {dt:>4} | {de:>6} | {rate:>6.1%} |")
        domain_stats.append({
            'domain': domain,
            'total': int(dt),
            'exceptions': int(de),
            'exception_rate': round(float(rate), 4),
        })

    # --- scale別の例外率 ---
    print(f"\n### scale別の例外率")
    scale_total = Counter(t['scale'] for t in transitions if t['scale'])
    scale_exception = Counter(t['scale'] for t in exceptions if t['scale'])

    print(f"| scale | 総数 | 例外数 | 例外率 |")
    print(f"|-------|------|--------|--------|")

    scale_stats = []
    for scale in sorted(scale_total.keys(), key=lambda s: scale_total[s], reverse=True):
        st = scale_total[scale]
        se = scale_exception.get(scale, 0)
        rate = se / st if st > 0 else 0
        print(f"| {scale:<20} | {st:>4} | {se:>6} | {rate:>6.1%} |")
        scale_stats.append({
            'scale': scale,
            'total': int(st),
            'exceptions': int(se),
            'exception_rate': round(float(rate), 4),
        })

    # --- source_type別の例外率 ---
    print(f"\n### source_type別の例外率")
    source_total = Counter(t['source_type'] for t in transitions if t['source_type'])
    source_exception = Counter(t['source_type'] for t in exceptions if t['source_type'])

    print(f"| source_type | 総数 | 例外数 | 例外率 |")
    print(f"|-------------|------|--------|--------|")

    source_stats = []
    for src in sorted(source_total.keys(), key=lambda s: source_total[s], reverse=True):
        st_total = source_total[src]
        se = source_exception.get(src, 0)
        rate = se / st_total if st_total > 0 else 0
        print(f"| {src:<20} | {st_total:>4} | {se:>6} | {rate:>6.1%} |")
        source_stats.append({
            'source_type': src,
            'total': int(st_total),
            'exceptions': int(se),
            'exception_rate': round(float(rate), 4),
        })

    # --- before_hexagram/after_hexagramの頻度 ---
    print(f"\n### 例外事例のbefore_hexagram頻度（上位15）")
    before_counter = Counter(t['before_hexagram'] for t in exceptions)
    print(f"| before_hexagram | 件数 |")
    print(f"|-----------------|------|")
    for hex_name, count in before_counter.most_common(15):
        print(f"| {hex_name:<20} | {count:>4} |")

    print(f"\n### 例外事例のafter_hexagram頻度（上位15）")
    after_counter = Counter(t['after_hexagram'] for t in exceptions)
    print(f"| after_hexagram | 件数 |")
    print(f"|----------------|------|")
    for hex_name, count in after_counter.most_common(15):
        print(f"| {hex_name:<20} | {count:>4} |")

    # --- Δパターンの分布 ---
    print(f"\n### 例外Δパターン分布（Δ_lower x Δ_upper）")
    pattern_counter = Counter(
        (t['delta_lower_int'], t['delta_upper_int']) for t in exceptions
    )
    print(f"| Δ_lower | Δ_upper | Δ_lower(bin) | Δ_upper(bin) | 件数 |")
    print(f"|---------|---------|--------------|--------------|------|")
    for (dl, du), count in pattern_counter.most_common(20):
        dl_bits = ((dl >> 2) & 1, (dl >> 1) & 1, dl & 1)
        du_bits = ((du >> 2) & 1, (du >> 1) & 1, du & 1)
        print(f"| {dl:>7} | {du:>7} | {bits_to_str(dl_bits):>12} | {bits_to_str(du_bits):>12} | {count:>4} |")

    # --- 例外事例の全件リスト（JSONに出力、コンソールは上位20件のみ） ---
    print(f"\n### 例外事例サンプル（上位20件）")
    print(f"| # | transition_id | target_name | before | after | Δ_lower | Δ_upper | domain | scale | source |")
    print(f"|---|---------------|-------------|--------|-------|---------|---------|--------|-------|--------|")

    exception_records = []
    for i, t in enumerate(exceptions):
        dl_bits = ((t['delta_lower_int'] >> 2) & 1, (t['delta_lower_int'] >> 1) & 1, t['delta_lower_int'] & 1)
        du_bits = ((t['delta_upper_int'] >> 2) & 1, (t['delta_upper_int'] >> 1) & 1, t['delta_upper_int'] & 1)

        record = {
            'transition_id': t['transition_id'],
            'target_name': t['target_name'],
            'before_hexagram': t['before_hexagram'],
            'after_hexagram': t['after_hexagram'],
            'delta_lower': bits_to_str(dl_bits),
            'delta_upper': bits_to_str(du_bits),
            'delta_lower_int': t['delta_lower_int'],
            'delta_upper_int': t['delta_upper_int'],
            'main_domain': t['main_domain'],
            'scale': t['scale'],
            'source_type': t['source_type'],
        }
        exception_records.append(record)

        if i < 20:
            tid = str(t['transition_id'])[:15] if t['transition_id'] else ''
            tname = str(t['target_name'])[:15] if t['target_name'] else ''
            print(f"| {i+1:>1} | {tid:<13} | {tname:<11} | {t['before_hexagram']:<6} | {t['after_hexagram']:<5} | {bits_to_str(dl_bits)} | {bits_to_str(du_bits)} | {t['main_domain'][:8] if t['main_domain'] else '':<8} | {t['scale'][:6] if t['scale'] else '':<5} | {t['source_type'][:8] if t['source_type'] else '':<6} |")

    result = {
        'n_exceptions': int(n_exceptions),
        'exception_rate': round(float(n_exceptions / total), 6),
        'by_main_domain': domain_stats,
        'by_scale': scale_stats,
        'by_source_type': source_stats,
        'before_hexagram_frequency': [
            {'hexagram': h, 'count': int(c)}
            for h, c in before_counter.most_common()
        ],
        'after_hexagram_frequency': [
            {'hexagram': h, 'count': int(c)}
            for h, c in after_counter.most_common()
        ],
        'delta_pattern_distribution': [
            {
                'delta_lower': int(dl),
                'delta_upper': int(du),
                'delta_lower_bits': bits_to_str(((dl >> 2) & 1, (dl >> 1) & 1, dl & 1)),
                'delta_upper_bits': bits_to_str(((du >> 2) & 1, (du >> 1) & 1, du & 1)),
                'count': int(c),
            }
            for (dl, du), c in pattern_counter.most_common()
        ],
        'exception_records': exception_records,
    }

    return result


# ============================================================
# Part 3: 対角事例のΔパターン分析
# ============================================================

def part3_diagonal_pattern_analysis(transitions):
    """対角事例のΔパターン（000-111）の頻度分布"""
    print("\n" + "=" * 70)
    print("[Part 3] 対角事例のΔパターン分析 (Δ_lower == Δ_upper)")
    print("=" * 70)

    diag_cases = [t for t in transitions if t['is_diagonal']]
    n_diag = len(diag_cases)

    print(f"\n  対角事例数: {n_diag}")

    # 8パターンの頻度
    pattern_counter = Counter(t['delta_lower_int'] for t in diag_cases)

    print(f"\n### Δパターン分布（Δ_lower == Δ_upper の8パターン）")
    print(f"| Δ値 | bit表現 | 件数 | 割合 | 変換の意味 | 対応八卦変換 |")
    print(f"|-----|---------|------|------|-----------|-------------|")

    pattern_records = []
    for d in range(8):
        d_bits = ((d >> 2) & 1, (d >> 1) & 1, d & 1)
        count = pattern_counter.get(d, 0)
        ratio = count / n_diag if n_diag > 0 else 0
        meaning = DELTA_TRIGRAM_MEANING.get(d_bits, '')

        # 対応する八卦変換の説明
        # Δ=d のとき、任意の八卦 T に対して T XOR d で変換先が決まる
        # 例: Δ=7(111) → 乾↔坤, 兌↔艮, 離↔坎, 震↔巽
        trigram_examples = []
        for t_int in range(8):
            t_bits = ((t_int >> 2) & 1, (t_int >> 1) & 1, t_int & 1)
            result_int = t_int ^ d
            result_bits = ((result_int >> 2) & 1, (result_int >> 1) & 1, result_int & 1)
            if t_int < result_int:  # 重複除去
                t_name = TRIGRAM_NAMES.get(t_bits, '?')
                r_name = TRIGRAM_NAMES.get(result_bits, '?')
                trigram_examples.append(f"{t_name}→{r_name}")
            elif t_int == result_int:
                t_name = TRIGRAM_NAMES.get(t_bits, '?')
                trigram_examples.append(f"{t_name}=不変")

        example_str = ', '.join(trigram_examples[:4])  # 上位4つ

        print(f"| {d:>3} | {bits_to_str(d_bits):>7} | {count:>4} | {ratio:>5.1%} | {meaning:<9} | {example_str} |")

        pattern_records.append({
            'delta_value': int(d),
            'delta_bits': bits_to_str(d_bits),
            'count': int(count),
            'ratio': round(float(ratio), 4),
            'meaning': meaning,
            'trigram_transform_examples': trigram_examples,
        })

    # Δ=0（同一卦遷移）の比率
    same_hex_count = pattern_counter.get(0, 0)
    print(f"\n  Δ=000（同一卦遷移 = before==after）: {same_hex_count} ({same_hex_count/n_diag*100:.1f}%)")

    # ハミング重み別の集計（対角パターンのみ）
    print(f"\n### 対角パターンのハミング重み別集計")
    hw_groups = defaultdict(int)
    for d in range(8):
        d_bits = ((d >> 2) & 1, (d >> 1) & 1, d & 1)
        hw = sum(d_bits)
        hw_groups[hw] += pattern_counter.get(d, 0)

    print(f"| ハミング重み | 件数 | 割合 | 対応Δ値 |")
    print(f"|-------------|------|------|---------|")
    for hw in range(4):  # 3bit なので最大3
        count = hw_groups[hw]
        ratio = count / n_diag if n_diag > 0 else 0
        deltas = [d for d in range(8) if sum(((d >> 2) & 1, (d >> 1) & 1, d & 1)) == hw]
        delta_strs = ','.join(str(d) for d in deltas)
        print(f"| {hw:>11} | {count:>4} | {ratio:>5.1%} | {delta_strs} |")

    # 全卦ハミング重みとの関係（対角の場合、6bitのハミング重み = 2 * 3bitのハミング重み）
    print(f"\n  注: 対角パターンでは 6bitハミング重み = 2 x 3bitハミング重み（常に偶数）")
    print(f"  → 対角偏向は偶数パリティ偏向(98.6%)の直接的原因")

    result = {
        'n_diagonal': int(n_diag),
        'patterns': pattern_records,
        'same_hexagram_count': int(same_hex_count),
        'same_hexagram_ratio': round(float(same_hex_count / n_diag), 4) if n_diag > 0 else 0,
        'hamming_weight_groups': {
            str(hw): int(hw_groups[hw]) for hw in range(4)
        },
    }

    return result


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 70)
    print("Phase 3: アノテーションバイアス検証")
    print("  Δ_lower == Δ_upper の対角偏向（97.8%）の構造分析")
    print("=" * 70)

    t_start = time.time()

    # --- データ読み込み ---
    print("\n[0] データ読み込み...")
    reference_data = load_json(REFERENCE_FILE)
    cases = load_cases()
    print(f"  事例数: {len(cases)}")

    # --- マッピング構築 ---
    name_to_kw = build_name_to_kw(reference_data)
    kw_to_bits, bits_to_kw = build_kw_to_bits(reference_data)
    print(f"  卦名→番号: {len(name_to_kw)}件")
    print(f"  番号→6bit: {len(kw_to_bits)}件")

    # --- 遷移抽出 ---
    transitions, n_excluded = extract_transitions_with_metadata(cases, name_to_kw, kw_to_bits)
    n_used = len(transitions)
    print(f"  有効遷移ペア: {n_used} (除外: {n_excluded})")

    if n_used == 0:
        print("ERROR: 有効な遷移ペアが0件。終了。")
        return

    # --- 分析実行 ---
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_cases_total': len(cases),
            'n_transitions_valid': int(n_used),
            'n_transitions_excluded': int(n_excluded),
            'description': 'Δ_lower == Δ_upper 対角偏向のアノテーションバイアス検証',
        }
    }

    # Part 1: 8×8分割表
    results['part1_contingency_table'] = part1_contingency_table(transitions)

    # Part 2: 例外事例の監査
    results['part2_exception_audit'] = part2_exception_audit(transitions, name_to_kw)

    # Part 3: 対角パターン分析
    results['part3_diagonal_patterns'] = part3_diagonal_pattern_analysis(transitions)

    # --- サマリー ---
    print("\n" + "=" * 70)
    print("[Summary]")
    print("=" * 70)
    p1 = results['part1_contingency_table']
    p2 = results['part2_exception_audit']
    p3 = results['part3_diagonal_patterns']
    print(f"  対角比率:         {p1['diagonal_ratio']:.4f} ({p1['diagonal_ratio']*100:.1f}%)")
    print(f"  独立期待値:       {p1['expected_diagonal_ratio']:.4f} ({p1['expected_diagonal_ratio']*100:.1f}%)")
    print(f"  エンリッチメント: {p1['diagonal_enrichment']:.2f}x")
    print(f"  χ²(8x8) p値:     {p1['chi2_8x8']['p_value']:.2e}")
    print(f"  Cramer's V:       {p1['chi2_8x8']['cramers_v']:.4f}")
    print(f"  例外事例数:       {p2['n_exceptions']}")
    print(f"  同一卦遷移:       {p3['same_hexagram_count']} ({p3['same_hexagram_ratio']*100:.1f}%)")

    # バイアス vs 構造の判定材料
    print(f"\n### バイアス判定材料")

    # source_type間の例外率のばらつき
    src_rates = [s['exception_rate'] for s in p2['by_source_type'] if s['total'] >= 50]
    if len(src_rates) >= 2:
        src_range = max(src_rates) - min(src_rates)
        print(f"  source_type間の例外率レンジ(N>=50): {src_range:.4f}")
        if src_range < 0.02:
            print(f"    → ソースタイプによる差が小さい = バイアスの一貫性を示唆")
        else:
            print(f"    → ソースタイプで例外率が異なる = 構造的要因の可能性")

    domain_rates = [s['exception_rate'] for s in p2['by_main_domain'] if s['total'] >= 50]
    if len(domain_rates) >= 2:
        domain_range = max(domain_rates) - min(domain_rates)
        print(f"  domain間の例外率レンジ(N>=50):      {domain_range:.4f}")

    # --- JSON出力 ---
    print(f"\n[出力] JSON保存: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {OUTPUT_JSON}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"アノテーションバイアス検証 完了 ({elapsed:.1f}秒)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
