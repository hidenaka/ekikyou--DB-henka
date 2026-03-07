#!/usr/bin/env python3
"""
純卦への崩壊仮説の検証 (Pure Hexagram Collapse Hypothesis)

Codex(GPT-5.4)の批判:
  「対角構造の主因は、LLMが64卦空間を8個の純卦（上下同卦）に
   プロトタイプ的に圧縮しているからだ」

純卦 = 上卦==下卦の8卦:
  1(乾為天), 2(坤為地), 29(坎為水), 30(離為火),
  51(震為雷), 52(艮為山), 57(巽為風), 58(兌為沢)

分析内容:
  1. 純卦率の算出（before/after別）
  2. state別の純卦率
  3. source_type別の純卦率
  4. 純卦 vs 非純卦での対角率
  5. 64卦の使用頻度分布（エントロピー含む）
  6. Codex仮説の直接検証（結論）
"""

import json
import sys
import math
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# isomorphism_test.py からユーティリティをインポート
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis" / "phase3"))
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

# ---------- 定数 ----------
PURE_HEXAGRAM_KW = {1, 2, 29, 30, 51, 52, 57, 58}
PURE_HEXAGRAM_NAMES = {
    1: "乾為天", 2: "坤為地", 29: "坎為水", 30: "離為火",
    51: "震為雷", 52: "艮為山", 57: "巽為風", 58: "兌為沢",
}

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis" / "phase3"
OUTPUT_JSON = OUTPUT_DIR / "pure_hexagram_analysis.json"


def is_pure(kw):
    """King Wen番号が純卦かどうか"""
    return kw in PURE_HEXAGRAM_KW


def resolve_cases(cases, name_to_kw):
    """全事例のbefore/after hexagramをKing Wen番号に解決"""
    resolved = []
    n_excluded = 0
    for case in cases:
        bv = case.get('classical_before_hexagram', '')
        av = case.get('classical_after_hexagram', '')
        kw_b = resolve_hexagram_field(bv, name_to_kw)
        kw_a = resolve_hexagram_field(av, name_to_kw)
        if kw_b is None or kw_a is None:
            n_excluded += 1
            continue
        resolved.append({
            'kw_before': kw_b,
            'kw_after': kw_a,
            'before_state': case.get('before_state', ''),
            'after_state': case.get('after_state', ''),
            'source_type': case.get('source_type', ''),
        })
    return resolved, n_excluded


def shannon_entropy(counter, total):
    """シャノンエントロピー（ビット単位）"""
    if total == 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


# ============================================================
# 1. 純卦率の算出
# ============================================================

def analyze_pure_rate(resolved):
    print("\n" + "=" * 60)
    print("[1] 純卦率の算出")
    print("=" * 60)

    total = len(resolved)
    before_pure = sum(1 for r in resolved if is_pure(r['kw_before']))
    after_pure = sum(1 for r in resolved if is_pure(r['kw_after']))

    bp = before_pure / total * 100
    ap = after_pure / total * 100
    expected = 8 / 64 * 100  # 12.5%

    print(f"\n  総事例数: {total}")
    print(f"  期待値（均等分布）: {expected:.1f}%")
    print(f"\n  | 位置   | 純卦数 | 純卦率    | 非純卦率  |")
    print(f"  |--------|--------|-----------|-----------|")
    print(f"  | before | {before_pure:>6} | {bp:>7.2f}% | {100-bp:>7.2f}% |")
    print(f"  | after  | {after_pure:>6} | {ap:>7.2f}% | {100-ap:>7.2f}% |")

    # 純卦別の内訳
    before_pure_detail = Counter(r['kw_before'] for r in resolved if is_pure(r['kw_before']))
    after_pure_detail = Counter(r['kw_after'] for r in resolved if is_pure(r['kw_after']))

    print(f"\n  純卦別の内訳:")
    print(f"  | KW | 卦名     | before | after |")
    print(f"  |----|----------|--------|-------|")
    for kw in sorted(PURE_HEXAGRAM_KW):
        name = PURE_HEXAGRAM_NAMES[kw]
        bc = before_pure_detail.get(kw, 0)
        ac = after_pure_detail.get(kw, 0)
        print(f"  | {kw:>2} | {name:<8} | {bc:>6} | {ac:>5} |")

    return {
        'total': total,
        'expected_pure_rate_pct': expected,
        'before_pure_count': before_pure,
        'before_pure_rate_pct': round(bp, 2),
        'after_pure_count': after_pure,
        'after_pure_rate_pct': round(ap, 2),
        'before_pure_detail': {str(k): v for k, v in sorted(before_pure_detail.items())},
        'after_pure_detail': {str(k): v for k, v in sorted(after_pure_detail.items())},
    }


# ============================================================
# 2. state別の純卦率
# ============================================================

def analyze_state_pure_rate(resolved):
    print("\n" + "=" * 60)
    print("[2] state別の純卦率")
    print("=" * 60)

    # before_state別
    by_before_state = defaultdict(list)
    for r in resolved:
        by_before_state[r['before_state']].append(r)

    print(f"\n  before_state別のbefore_hexagram純卦率:")
    print(f"  | before_state        | 総数  | 純卦数 | 純卦率    |")
    print(f"  |---------------------|-------|--------|-----------|")

    before_state_results = {}
    for state in sorted(by_before_state.keys()):
        items = by_before_state[state]
        total = len(items)
        pure_count = sum(1 for r in items if is_pure(r['kw_before']))
        rate = pure_count / total * 100 if total > 0 else 0
        print(f"  | {state:<19} | {total:>5} | {pure_count:>6} | {rate:>7.2f}% |")
        before_state_results[state] = {
            'total': total, 'pure_count': pure_count, 'pure_rate_pct': round(rate, 2)
        }

    # after_state別
    by_after_state = defaultdict(list)
    for r in resolved:
        by_after_state[r['after_state']].append(r)

    print(f"\n  after_state別のafter_hexagram純卦率:")
    print(f"  | after_state          | 総数  | 純卦数 | 純卦率    |")
    print(f"  |----------------------|-------|--------|-----------|")

    after_state_results = {}
    for state in sorted(by_after_state.keys()):
        items = by_after_state[state]
        total = len(items)
        pure_count = sum(1 for r in items if is_pure(r['kw_after']))
        rate = pure_count / total * 100 if total > 0 else 0
        print(f"  | {state:<20} | {total:>5} | {pure_count:>6} | {rate:>7.2f}% |")
        after_state_results[state] = {
            'total': total, 'pure_count': pure_count, 'pure_rate_pct': round(rate, 2)
        }

    return {
        'before_state_pure_rates': before_state_results,
        'after_state_pure_rates': after_state_results,
    }


# ============================================================
# 3. source_type別の純卦率
# ============================================================

def analyze_source_pure_rate(resolved):
    print("\n" + "=" * 60)
    print("[3] source_type別の純卦率")
    print("=" * 60)

    by_source = defaultdict(list)
    for r in resolved:
        by_source[r['source_type']].append(r)

    print(f"\n  | source_type | 総数  | before純卦率 | after純卦率 |")
    print(f"  |-------------|-------|-------------|------------|")

    source_results = {}
    for src in sorted(by_source.keys()):
        items = by_source[src]
        total = len(items)
        b_pure = sum(1 for r in items if is_pure(r['kw_before']))
        a_pure = sum(1 for r in items if is_pure(r['kw_after']))
        br = b_pure / total * 100 if total > 0 else 0
        ar = a_pure / total * 100 if total > 0 else 0
        print(f"  | {src:<11} | {total:>5} | {br:>9.2f}%  | {ar:>8.2f}%  |")
        source_results[src] = {
            'total': total,
            'before_pure_rate_pct': round(br, 2),
            'after_pure_rate_pct': round(ar, 2),
        }

    return {'source_type_pure_rates': source_results}


# ============================================================
# 4. 純卦 vs 非純卦での対角率
# ============================================================

def analyze_diagonal_by_purity(resolved, kw_to_bits):
    print("\n" + "=" * 60)
    print("[4] 純卦 vs 非純卦での対角率")
    print("=" * 60)

    def xor_bits(a, b):
        return tuple(x ^ y for x, y in zip(a, b))

    def delta_lower_eq_upper(bits_from, bits_to):
        """Δ_lower == Δ_upper (対角構造)"""
        delta = xor_bits(bits_from, bits_to)
        return delta[:3] == delta[3:]

    # 3カテゴリに分類
    both_pure = []
    one_pure = []
    both_nonpure = []

    for r in resolved:
        kb, ka = r['kw_before'], r['kw_after']
        if kb not in kw_to_bits or ka not in kw_to_bits:
            continue
        bp = is_pure(kb)
        ap = is_pure(ka)
        entry = (kw_to_bits[kb], kw_to_bits[ka])
        if bp and ap:
            both_pure.append(entry)
        elif bp or ap:
            one_pure.append(entry)
        else:
            both_nonpure.append(entry)

    categories = [
        ("both_pure (before&after共に純卦)", both_pure),
        ("one_pure (片方のみ純卦)", one_pure),
        ("both_nonpure (両方非純卦)", both_nonpure),
    ]

    print(f"\n  | カテゴリ                        | 件数  | 対角数 | 対角率    |")
    print(f"  |---------------------------------|-------|--------|-----------|")

    diagonal_results = {}
    for name, items in categories:
        total = len(items)
        diag_count = sum(1 for bf, bt in items if delta_lower_eq_upper(bf, bt))
        rate = diag_count / total * 100 if total > 0 else 0
        key = name.split(" ")[0]
        print(f"  | {name:<31} | {total:>5} | {diag_count:>6} | {rate:>7.2f}% |")
        diagonal_results[key] = {
            'total': total, 'diagonal_count': diag_count, 'diagonal_rate_pct': round(rate, 2)
        }

    # 理論値: 純卦ペアなら上卦=下卦だからΔ_lower=Δ_upper は自動的に成立
    print(f"\n  理論的注釈:")
    print(f"    - both_pure: before/afterが共に純卦 → 上卦=下卦 → Δ_lower=Δ_upper は自動成立(100%)")
    print(f"    - both_nonpure: 純卦崩壊では説明できない対角構造の直接検証")

    # 追加: both_nonpureの対角率が偶然(50%)と有意に異なるか
    if len(both_nonpure) > 0:
        from scipy.stats import binomtest
        diag_np = diagonal_results['both_nonpure']['diagonal_count']
        total_np = diagonal_results['both_nonpure']['total']
        # Δ_lower == Δ_upper の偶然確率: 8/64 = 1/8 = 12.5%
        # (下卦Δは8通り、上卦Δも8通り、一致する確率は 8/64)
        expected_p = 8 / 64  # = 0.125
        btest = binomtest(diag_np, total_np, expected_p, alternative='two-sided')
        print(f"\n  both_nonpureの二項検定 (H0: 対角率={expected_p:.3f}):")
        print(f"    観測対角率: {diag_np/total_np:.4f}")
        print(f"    p値: {btest.pvalue:.2e}")
        diagonal_results['both_nonpure_binomial_test'] = {
            'expected_p': expected_p,
            'observed_p': round(diag_np / total_np, 4),
            'p_value': float(btest.pvalue),
        }

    return {'diagonal_by_purity': diagonal_results}


# ============================================================
# 5. 64卦の使用頻度分布
# ============================================================

def analyze_frequency_distribution(resolved):
    print("\n" + "=" * 60)
    print("[5] 64卦の使用頻度分布")
    print("=" * 60)

    before_counter = Counter(r['kw_before'] for r in resolved)
    after_counter = Counter(r['kw_after'] for r in resolved)
    total = len(resolved)

    # 使用されている卦の数
    before_used = len(before_counter)
    after_used = len(after_counter)

    # エントロピー
    max_entropy = math.log2(64)  # 6.0 bits (均等分布)
    before_entropy = shannon_entropy(before_counter, total)
    after_entropy = shannon_entropy(after_counter, total)

    print(f"\n  | 指標                | before     | after      |")
    print(f"  |---------------------|------------|------------|")
    print(f"  | 使用卦数(/64)       | {before_used:>10} | {after_used:>10} |")
    print(f"  | エントロピー(bits)  | {before_entropy:>10.3f} | {after_entropy:>10.3f} |")
    print(f"  | 最大エントロピー    | {max_entropy:>10.3f} | {max_entropy:>10.3f} |")
    print(f"  | エントロピー比      | {before_entropy/max_entropy:>10.3f} | {after_entropy/max_entropy:>10.3f} |")

    # 上位10卦
    print(f"\n  before_hexagram 上位10卦:")
    print(f"  | 順位 | KW | 使用回数 | 割合    | 純卦? |")
    print(f"  |------|------|----------|---------|-------|")
    for rank, (kw, count) in enumerate(before_counter.most_common(10), 1):
        pct = count / total * 100
        pure_mark = "***" if is_pure(kw) else ""
        print(f"  | {rank:>4} | {kw:>4} | {count:>8} | {pct:>5.2f}% | {pure_mark:<5} |")

    print(f"\n  after_hexagram 上位10卦:")
    print(f"  | 順位 | KW | 使用回数 | 割合    | 純卦? |")
    print(f"  |------|------|----------|---------|-------|")
    for rank, (kw, count) in enumerate(after_counter.most_common(10), 1):
        pct = count / total * 100
        pure_mark = "***" if is_pure(kw) else ""
        print(f"  | {rank:>4} | {kw:>4} | {count:>8} | {pct:>5.2f}% | {pure_mark:<5} |")

    # 下位10卦
    print(f"\n  before_hexagram 下位10卦:")
    # 使用0の卦を含む
    all_before = {kw: before_counter.get(kw, 0) for kw in range(1, 65)}
    sorted_before = sorted(all_before.items(), key=lambda x: x[1])
    print(f"  | 順位 | KW | 使用回数 | 割合    | 純卦? |")
    print(f"  |------|------|----------|---------|-------|")
    for rank, (kw, count) in enumerate(sorted_before[:10], 1):
        pct = count / total * 100
        pure_mark = "***" if is_pure(kw) else ""
        print(f"  | {rank:>4} | {kw:>4} | {count:>8} | {pct:>5.2f}% | {pure_mark:<5} |")

    print(f"\n  after_hexagram 下位10卦:")
    all_after = {kw: after_counter.get(kw, 0) for kw in range(1, 65)}
    sorted_after = sorted(all_after.items(), key=lambda x: x[1])
    print(f"  | 順位 | KW | 使用回数 | 割合    | 純卦? |")
    print(f"  |------|------|----------|---------|-------|")
    for rank, (kw, count) in enumerate(sorted_after[:10], 1):
        pct = count / total * 100
        pure_mark = "***" if is_pure(kw) else ""
        print(f"  | {rank:>4} | {kw:>4} | {count:>8} | {pct:>5.2f}% | {pure_mark:<5} |")

    # 純卦8卦の合計シェア
    before_pure_share = sum(before_counter.get(kw, 0) for kw in PURE_HEXAGRAM_KW) / total * 100
    after_pure_share = sum(after_counter.get(kw, 0) for kw in PURE_HEXAGRAM_KW) / total * 100

    print(f"\n  純卦8卦の合計シェア:")
    print(f"    before: {before_pure_share:.2f}% (期待値 12.5%)")
    print(f"    after:  {after_pure_share:.2f}% (期待値 12.5%)")

    return {
        'before_hexagram': {
            'n_used': before_used,
            'entropy_bits': round(before_entropy, 3),
            'max_entropy_bits': round(max_entropy, 3),
            'entropy_ratio': round(before_entropy / max_entropy, 3),
            'top10': [{'kw': kw, 'count': c, 'pct': round(c/total*100, 2), 'is_pure': is_pure(kw)}
                      for kw, c in before_counter.most_common(10)],
            'bottom10': [{'kw': kw, 'count': c, 'pct': round(c/total*100, 2), 'is_pure': is_pure(kw)}
                         for kw, c in sorted_before[:10]],
            'pure_8_share_pct': round(before_pure_share, 2),
        },
        'after_hexagram': {
            'n_used': after_used,
            'entropy_bits': round(after_entropy, 3),
            'max_entropy_bits': round(max_entropy, 3),
            'entropy_ratio': round(after_entropy / max_entropy, 3),
            'top10': [{'kw': kw, 'count': c, 'pct': round(c/total*100, 2), 'is_pure': is_pure(kw)}
                      for kw, c in after_counter.most_common(10)],
            'bottom10': [{'kw': kw, 'count': c, 'pct': round(c/total*100, 2), 'is_pure': is_pure(kw)}
                         for kw, c in sorted_after[:10]],
            'pure_8_share_pct': round(after_pure_share, 2),
        },
        'full_distribution': {
            'before': {str(kw): before_counter.get(kw, 0) for kw in range(1, 65)},
            'after': {str(kw): after_counter.get(kw, 0) for kw in range(1, 65)},
        },
    }


# ============================================================
# 6. Codex仮説の直接検証
# ============================================================

def verdict(results):
    print("\n" + "=" * 60)
    print("[6] Codex仮説の直接検証: 結論")
    print("=" * 60)

    pure_rate_b = results['pure_rate']['before_pure_rate_pct']
    pure_rate_a = results['pure_rate']['after_pure_rate_pct']
    entropy_ratio_b = results['frequency_distribution']['before_hexagram']['entropy_ratio']
    entropy_ratio_a = results['frequency_distribution']['after_hexagram']['entropy_ratio']
    n_used_b = results['frequency_distribution']['before_hexagram']['n_used']
    n_used_a = results['frequency_distribution']['after_hexagram']['n_used']

    diag = results['diagonal_by_purity']['diagonal_by_purity']
    both_nonpure_diag = diag.get('both_nonpure', {}).get('diagonal_rate_pct', 0)
    both_pure_count = diag.get('both_pure', {}).get('total', 0)
    total = results['pure_rate']['total']
    both_pure_share = both_pure_count / total * 100 if total > 0 else 0

    collapse_threshold = 25.0  # 純卦率がこれ以上なら「崩壊」と判定

    print(f"\n  Codex仮説: 「LLMが64卦を8純卦に圧縮している」")
    print(f"\n  検証指標:")
    print(f"    1. before純卦率: {pure_rate_b:.2f}% (閾値: {collapse_threshold}%)")
    print(f"    2. after純卦率:  {pure_rate_a:.2f}% (閾値: {collapse_threshold}%)")
    print(f"    3. before使用卦数: {n_used_b}/64")
    print(f"    4. after使用卦数:  {n_used_a}/64")
    print(f"    5. beforeエントロピー比: {entropy_ratio_b:.3f} (1.0=均等)")
    print(f"    6. afterエントロピー比:  {entropy_ratio_a:.3f} (1.0=均等)")
    print(f"    7. both_pure遷移の割合:  {both_pure_share:.2f}%")
    print(f"    8. both_nonpureでの対角率: {both_nonpure_diag:.2f}%")

    # 判定
    is_collapsed = (pure_rate_b > collapse_threshold or pure_rate_a > collapse_threshold)
    is_diverse = (n_used_b >= 50 and n_used_a >= 50 and entropy_ratio_b > 0.85 and entropy_ratio_a > 0.85)

    if is_collapsed:
        verdict_text = "SUPPORTED: 純卦への崩壊が検出された"
        explains_diagonal = "対角構造は純卦崩壊の機械的帰結である可能性が高い"
    elif is_diverse and both_nonpure_diag > 20:
        verdict_text = "REJECTED: 64卦空間は豊かに使用されており、純卦崩壊は発生していない"
        explains_diagonal = "対角構造は純卦崩壊では説明できない（非純卦ペアでも対角率が高い）"
    elif is_diverse:
        verdict_text = "REJECTED: 64卦空間は活用されているが、非純卦での対角率は低い"
        explains_diagonal = "対角構造の別の原因を調査する必要がある"
    else:
        verdict_text = "INCONCLUSIVE: 部分的な崩壊の可能性"
        explains_diagonal = "追加検証が必要"

    print(f"\n  判定: {verdict_text}")
    print(f"  対角構造への含意: {explains_diagonal}")

    return {
        'verdict': verdict_text,
        'diagonal_implication': explains_diagonal,
        'is_collapsed': is_collapsed,
        'is_diverse': is_diverse,
        'collapse_threshold_pct': collapse_threshold,
        'key_metrics': {
            'before_pure_rate_pct': pure_rate_b,
            'after_pure_rate_pct': pure_rate_a,
            'before_n_used': n_used_b,
            'after_n_used': n_used_a,
            'before_entropy_ratio': entropy_ratio_b,
            'after_entropy_ratio': entropy_ratio_a,
            'both_pure_share_pct': round(both_pure_share, 2),
            'both_nonpure_diagonal_rate_pct': both_nonpure_diag,
        }
    }


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("純卦への崩壊仮説の検証")
    print("  Codex批判: LLMが64卦を8純卦に圧縮している?")
    print("=" * 60)

    # データ読み込み
    print("\n[0] データ読み込み...")
    reference_data = load_json(REFERENCE_FILE)
    cases = load_cases()
    name_to_kw = build_name_to_kw(reference_data)
    kw_to_bits, _ = build_kw_to_bits(reference_data)
    print(f"  事例数: {len(cases)}")

    # 事例の解決
    resolved, n_excluded = resolve_cases(cases, name_to_kw)
    print(f"  有効事例: {len(resolved)} (除外: {n_excluded})")

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(cases),
            'valid_cases': len(resolved),
            'excluded_cases': n_excluded,
        }
    }

    # 1. 純卦率
    results['pure_rate'] = analyze_pure_rate(resolved)

    # 2. state別の純卦率
    results['state_pure_rates'] = analyze_state_pure_rate(resolved)

    # 3. source_type別
    results['source_pure_rates'] = analyze_source_pure_rate(resolved)

    # 4. 対角率
    results['diagonal_by_purity'] = analyze_diagonal_by_purity(resolved, kw_to_bits)

    # 5. 頻度分布
    results['frequency_distribution'] = analyze_frequency_distribution(resolved)

    # 6. 結論
    results['verdict'] = verdict(results)

    # JSON出力
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")

    print(f"\n{'=' * 60}")
    print("分析完了")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
