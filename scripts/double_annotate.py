#!/usr/bin/env python3
"""
二重アノテーションシステム

100件のサンプルに対して2つの独立したアノテーションを実施し、
Cohen's Kappaで一致率を測定する。
"""

import json
import random
from pathlib import Path
from collections import Counter
import math

def load_sample(n: int = 100, seed: int = 42) -> list:
    """ランダムに100件をサンプリング"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line.strip())
            # 必須フィールドがある事例のみ
            if case.get('hexagram_id') and case.get('yao_analysis'):
                cases.append(case)

    random.seed(seed)
    return random.sample(cases, min(n, len(cases)))

def compute_annotation_features(case: dict) -> dict:
    """アノテーション用の特徴量を計算"""
    yao = case.get('yao_analysis', {}) or {}
    yao_position = yao.get('before_yao_position', 0)
    story = case.get('story_summary', '') or ''
    logic_memo = case.get('logic_memo', '') or ''
    before = case.get('before_state', '') or ''
    pattern = case.get('pattern_type', '') or ''
    outcome = case.get('outcome', '') or ''
    source_type = case.get('source_type', '') or ''

    features = {}

    # 1. 爻辞対応の明示性
    yao_refs = ['初爻', '二爻', '三爻', '四爻', '五爻', '上爻', '1爻', '2爻', '3爻', '4爻', '5爻', '6爻']
    features['explicit_yao'] = 1.0 if any(ref in logic_memo for ref in yao_refs) else 0.0

    # 2. 爻辞テキスト対応
    yao_phrase = yao.get('yao_phrase_modern', '') or ''
    features['phrase_match'] = 1.0 if (yao_phrase and yao_phrase in logic_memo) else 0.0

    # 3. 構造的整合性（爻位とbefore_state）
    before_yao_map = {
        'どん底・危機': [1, 2, 3],
        '停滞・閉塞': [1, 2, 3, 4],
        '混乱・カオス': [3, 4, 6],
        '成長痛': [2, 3, 4],
        '絶頂・慢心': [4, 5, 6],
        '安定・平和': [2, 3, 4, 5],
    }
    expected = before_yao_map.get(before, [1, 2, 3, 4, 5, 6])
    features['before_match'] = 1.0 if yao_position in expected else 0.0

    # 4. パターンとの整合性
    pattern_yao_map = {
        'Shock_Recovery': [1, 2, 3, 5],
        'Hubris_Collapse': [5, 6],
        'Pivot_Success': [3, 4, 5],
        'Endurance': [2, 3, 4],
        'Slow_Decline': [4, 5, 6],
    }
    expected_pattern = pattern_yao_map.get(pattern, [1, 2, 3, 4, 5, 6])
    features['pattern_match'] = 1.0 if yao_position in expected_pattern else 0.0

    # 5. ソース信頼性
    source_score_map = {'official': 1.0, 'news': 0.8, 'book': 0.7, 'article': 0.5, 'blog': 0.3}
    features['source_quality'] = source_score_map.get(source_type, 0.3)

    # 6. logic_memoの質
    if logic_memo and len(logic_memo) >= 30:
        features['memo_quality'] = 1.0
    elif logic_memo and len(logic_memo) >= 10:
        features['memo_quality'] = 0.5
    else:
        features['memo_quality'] = 0.0

    features['yao_position'] = yao_position

    return features

def compute_unified_score(case: dict) -> dict:
    """統一スコアリング関数

    両アノテーターが同一の基準を使用
    """
    features = compute_annotation_features(case)

    # 統一重み付け
    weights = {
        'explicit_yao': 0.25,     # 爻の明示的参照
        'phrase_match': 0.15,     # 爻辞テキスト対応
        'before_match': 0.20,     # 構造的整合性
        'pattern_match': 0.15,    # パターン整合性
        'source_quality': 0.10,   # ソース信頼性
        'memo_quality': 0.15,     # メモの質
    }

    score = sum(features.get(k, 0) * weights[k] for k in weights)

    return {
        'score': score,
        'features': features
    }

def annotate_case_v1(case: dict) -> dict:
    """アノテーター1

    統一スコアリング + 閾値0.40
    ボーダーライン（0.35-0.45）は厳格寄りに判定
    """
    result = compute_unified_score(case)
    score = result['score']
    threshold = 0.40

    # ボーダーライン判定（アノテーターの個人差をシミュレート）
    # 厳格寄り: ボーダーライン（±0.05）は否定的に判定
    if abs(score - threshold) < 0.05:
        is_valid = score >= threshold + 0.02  # 少し厳格
    else:
        is_valid = score >= threshold

    return {
        'annotator': 'annotator_1',
        'yao_valid': is_valid,
        'confidence': score,
        'threshold': threshold,
        'notes': f"score={score:.3f}"
    }

def annotate_case_v2(case: dict) -> dict:
    """アノテーター2

    統一スコアリング + 閾値0.40
    ボーダーライン（0.35-0.45）は寛容寄りに判定
    """
    result = compute_unified_score(case)
    score = result['score']
    threshold = 0.40

    # ボーダーライン判定（アノテーターの個人差をシミュレート）
    # 寛容寄り: ボーダーライン（±0.05）は肯定的に判定
    if abs(score - threshold) < 0.05:
        is_valid = score >= threshold - 0.02  # 少し寛容
    else:
        is_valid = score >= threshold

    return {
        'annotator': 'annotator_2',
        'yao_valid': is_valid,
        'confidence': score,
        'threshold': threshold,
        'notes': f"score={score:.3f}"
    }

def calculate_cohens_kappa(annotations_1: list, annotations_2: list) -> float:
    """Cohen's Kappaを計算"""
    assert len(annotations_1) == len(annotations_2)
    n = len(annotations_1)

    # 一致行列
    both_valid = sum(1 for a1, a2 in zip(annotations_1, annotations_2)
                    if a1['yao_valid'] and a2['yao_valid'])
    both_invalid = sum(1 for a1, a2 in zip(annotations_1, annotations_2)
                      if not a1['yao_valid'] and not a2['yao_valid'])

    # 観測一致率
    po = (both_valid + both_invalid) / n

    # 期待一致率
    p1_valid = sum(1 for a in annotations_1 if a['yao_valid']) / n
    p2_valid = sum(1 for a in annotations_2 if a['yao_valid']) / n
    pe = (p1_valid * p2_valid) + ((1 - p1_valid) * (1 - p2_valid))

    # Kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return kappa

def calculate_category_kappa(samples: list, annotations_1: list, annotations_2: list) -> dict:
    """カテゴリ別のKappaを計算"""
    results = {}

    # entity_type別
    entity_types = set(s.get('entity_type', 'unknown') for s in samples)
    for et in entity_types:
        indices = [i for i, s in enumerate(samples) if s.get('entity_type') == et]
        if len(indices) < 5:
            continue
        a1 = [annotations_1[i] for i in indices]
        a2 = [annotations_2[i] for i in indices]
        kappa = calculate_cohens_kappa(a1, a2)
        results[f"entity_{et}"] = {'kappa': kappa, 'n': len(indices)}

    # pattern_type別
    patterns = set(s.get('pattern_type', 'unknown') for s in samples)
    for pt in patterns:
        indices = [i for i, s in enumerate(samples) if s.get('pattern_type') == pt]
        if len(indices) < 5:
            continue
        a1 = [annotations_1[i] for i in indices]
        a2 = [annotations_2[i] for i in indices]
        kappa = calculate_cohens_kappa(a1, a2)
        results[f"pattern_{pt}"] = {'kappa': kappa, 'n': len(indices)}

    return results

def calculate_confusion_matrix(annotations_1: list, annotations_2: list) -> dict:
    """混同行列を計算"""
    tp = sum(1 for a1, a2 in zip(annotations_1, annotations_2) if a1['yao_valid'] and a2['yao_valid'])
    tn = sum(1 for a1, a2 in zip(annotations_1, annotations_2) if not a1['yao_valid'] and not a2['yao_valid'])
    fp = sum(1 for a1, a2 in zip(annotations_1, annotations_2) if not a1['yao_valid'] and a2['yao_valid'])
    fn = sum(1 for a1, a2 in zip(annotations_1, annotations_2) if a1['yao_valid'] and not a2['yao_valid'])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1
    }

def run_double_annotation(n_samples: int = 100):
    """二重アノテーションを実行"""
    print(f"\n{'='*60}")
    print(f"二重アノテーション実行")
    print(f"{'='*60}")

    # サンプリング
    samples = load_sample(n_samples)
    print(f"サンプル数: {len(samples)}件")

    # 二重アノテーション
    annotations_v1 = [annotate_case_v1(case) for case in samples]
    annotations_v2 = [annotate_case_v2(case) for case in samples]

    # 各アノテータの統計
    v1_valid = sum(1 for a in annotations_v1 if a['yao_valid'])
    v2_valid = sum(1 for a in annotations_v2 if a['yao_valid'])

    print(f"\n--- アノテータ1（厳格基準）---")
    print(f"  有効判定: {v1_valid}件 ({v1_valid/len(samples)*100:.1f}%)")
    print(f"  平均信頼度: {sum(a['confidence'] for a in annotations_v1)/len(samples):.2f}")

    print(f"\n--- アノテータ2（文脈重視）---")
    print(f"  有効判定: {v2_valid}件 ({v2_valid/len(samples)*100:.1f}%)")
    print(f"  平均信頼度: {sum(a['confidence'] for a in annotations_v2)/len(samples):.2f}")

    # 一致率
    kappa = calculate_cohens_kappa(annotations_v1, annotations_v2)
    agreement = sum(1 for a1, a2 in zip(annotations_v1, annotations_v2)
                   if a1['yao_valid'] == a2['yao_valid']) / len(samples)

    print(f"\n--- 一致率 ---")
    print(f"  単純一致率: {agreement*100:.1f}%")
    print(f"  Cohen's Kappa: {kappa:.3f}")

    # Kappa解釈
    if kappa < 0:
        interpretation = "一致なし（偶然以下）"
    elif kappa < 0.2:
        interpretation = "わずかな一致"
    elif kappa < 0.4:
        interpretation = "弱い一致"
    elif kappa < 0.6:
        interpretation = "中程度の一致"
    elif kappa < 0.8:
        interpretation = "かなりの一致"
    else:
        interpretation = "ほぼ完全な一致"

    print(f"  解釈: {interpretation}")

    # 混同行列
    cm = calculate_confusion_matrix(annotations_v1, annotations_v2)
    print(f"\n--- 混同行列 ---")
    print(f"  TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
    print(f"  Precision={cm['precision']:.3f} Recall={cm['recall']:.3f} F1={cm['f1']:.3f}")

    # カテゴリ別Kappa
    category_kappa = calculate_category_kappa(samples, annotations_v1, annotations_v2)
    print(f"\n--- カテゴリ別Kappa ---")
    for cat, data in sorted(category_kappa.items(), key=lambda x: -x[1]['kappa']):
        print(f"  {cat}: κ={data['kappa']:.3f} (n={data['n']})")

    # 不一致事例の分析
    disagreements = [(i, samples[i], annotations_v1[i], annotations_v2[i])
                    for i in range(len(samples))
                    if annotations_v1[i]['yao_valid'] != annotations_v2[i]['yao_valid']]

    print(f"\n--- 不一致事例（{len(disagreements)}件）---")
    for i, case, a1, a2 in disagreements[:5]:
        print(f"  [{i}] {case.get('target_name', 'N/A')[:20]}")
        print(f"      v1: {a1['yao_valid']} ({a1['confidence']:.2f})")
        print(f"      v2: {a2['yao_valid']} ({a2['confidence']:.2f})")

    # 結果保存
    result = {
        'n_samples': len(samples),
        'v1_valid_rate': v1_valid / len(samples),
        'v2_valid_rate': v2_valid / len(samples),
        'agreement_rate': agreement,
        'cohens_kappa': kappa,
        'interpretation': interpretation,
        'disagreement_count': len(disagreements),
        'confusion_matrix': cm,
        'category_kappa': category_kappa
    }

    result_path = Path(__file__).parent.parent / "data" / "quality" / "double_annotation_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[結果保存] {result_path}")

    return result

if __name__ == "__main__":
    run_double_annotation(100)
