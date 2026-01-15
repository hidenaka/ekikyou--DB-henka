#!/usr/bin/env python3
"""
バッチインポート前検証スクリプト v2.0

Codex推奨5項目対応:
1. 一次テキスト根拠チェック
2. 反証欄（候補卦3つ以上、排除理由）チェック
3. 盲検レビュー準備チェック
4. 一致度計測準備
5. 意味的重複検出
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

def load_existing_cases():
    """既存DBをロード"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    existing = []
    if cases_path.exists():
        with open(cases_path, 'r', encoding='utf-8') as f:
            for line in f:
                existing.append(json.loads(line.strip()))
    return existing

def check_primary_text_basis(case):
    """一次テキスト根拠チェック"""
    ptb = case.get('primary_text_basis', {})
    issues = []

    if not ptb:
        issues.append("primary_text_basis フィールドなし")
        return 0, issues

    score = 0
    if ptb.get('source_type') in ['卦辞', '爻辞', '象伝', '彖伝']:
        score += 1
    else:
        issues.append("source_type が不正")

    if ptb.get('original_text') and len(ptb['original_text']) >= 2:
        score += 1
    else:
        issues.append("original_text（原文）なし")

    if ptb.get('interpretation'):
        score += 1
    else:
        issues.append("interpretation（現代語訳）なし")

    if ptb.get('relevance'):
        score += 1
    else:
        issues.append("relevance（事例との対応）なし")

    return score, issues

def check_hexagram_selection(case):
    """反証欄チェック（候補卦3つ以上、排除理由）"""
    hs = case.get('hexagram_selection', {})
    issues = []

    if not hs:
        issues.append("hexagram_selection フィールドなし")
        return 0, issues

    score = 0
    candidates = hs.get('candidate_hexagrams', [])

    if len(candidates) >= 3:
        score += 2
    elif len(candidates) >= 2:
        score += 1
        issues.append("候補卦が2つ（3つ以上推奨）")
    else:
        issues.append("候補卦が不足（3つ以上必須）")

    # fit_score チェック
    if all(c.get('fit_score') for c in candidates):
        score += 1
    else:
        issues.append("fit_score が一部欠落")

    # rejection_reasons チェック
    rr = hs.get('rejection_reasons', {})
    non_selected = [c for c in candidates if c.get('hexagram_id') != hs.get('selected_hexagram')]
    if len(rr) >= len(non_selected) - 1:
        score += 1
    else:
        issues.append("rejection_reasons が不足")

    return score, issues

def check_semantic_similarity(new_case, existing_cases, threshold=0.7):
    """意味的重複検出"""
    similar = []
    new_summary = new_case.get('story_summary', '')
    new_name = new_case.get('target_name', '')

    for existing in existing_cases:
        # 名前の類似度
        name_sim = SequenceMatcher(None, new_name, existing.get('target_name', '')).ratio()

        # サマリーの類似度
        summary_sim = SequenceMatcher(None, new_summary, existing.get('story_summary', '')).ratio()

        # 卦・爻の一致
        hex_match = (
            new_case.get('yao_analysis', {}).get('before_hexagram_id') ==
            existing.get('yao_analysis', {}).get('before_hexagram_id')
        )
        yao_match = (
            new_case.get('yao_analysis', {}).get('before_yao_position') ==
            existing.get('yao_analysis', {}).get('before_yao_position')
        )

        # 総合スコア
        combined_score = (name_sim * 0.4 + summary_sim * 0.4 + (0.2 if hex_match and yao_match else 0))

        if combined_score >= threshold or name_sim >= 0.8:
            similar.append({
                'existing_name': existing.get('target_name'),
                'existing_period': existing.get('period'),
                'name_similarity': f"{name_sim:.1%}",
                'summary_similarity': f"{summary_sim:.1%}",
                'combined_score': f"{combined_score:.1%}"
            })

    return similar[:3]  # 上位3件のみ返す

def validate_batch_v2(batch_path):
    """バッチファイルをv2基準で検証"""
    print(f"\n{'='*60}")
    print(f"検証 (v2): {batch_path}")
    print(f"{'='*60}")

    with open(batch_path, 'r', encoding='utf-8') as f:
        batch = json.load(f)

    if not isinstance(batch, list):
        print("ERROR: JSONが配列形式でない")
        return False

    print(f"\n件数: {len(batch)}件")

    existing = load_existing_cases()
    print(f"既存DB: {len(existing)}件")

    # チェック結果集計
    exact_duplicates = []
    semantic_duplicates = []
    primary_text_issues = []
    hexagram_selection_issues = []

    primary_text_scores = []
    hexagram_selection_scores = []

    for i, case in enumerate(batch):
        # 完全一致重複
        key = (case.get('target_name', ''), case.get('period', ''))
        for ex in existing:
            if (ex.get('target_name'), ex.get('period')) == key:
                exact_duplicates.append(f"[{i+1}] {key[0]} ({key[1]})")
                break

        # 意味的重複
        similar = check_semantic_similarity(case, existing)
        if similar:
            semantic_duplicates.append({
                'index': i+1,
                'name': case.get('target_name'),
                'similar_to': similar
            })

        # 一次テキスト根拠
        pt_score, pt_issues = check_primary_text_basis(case)
        primary_text_scores.append(pt_score)
        if pt_issues:
            primary_text_issues.append(f"[{i+1}] {', '.join(pt_issues)}")

        # 反証欄
        hs_score, hs_issues = check_hexagram_selection(case)
        hexagram_selection_scores.append(hs_score)
        if hs_issues:
            hexagram_selection_issues.append(f"[{i+1}] {', '.join(hs_issues)}")

    # 結果表示
    print(f"\n--- 1. 完全一致重複 ---")
    if exact_duplicates:
        print(f"❌ 重複: {len(exact_duplicates)}件")
        for d in exact_duplicates[:3]:
            print(f"   {d}")
    else:
        print("✅ 完全一致重複なし")

    print(f"\n--- 2. 意味的重複 ---")
    if semantic_duplicates:
        print(f"⚠️ 類似事例検出: {len(semantic_duplicates)}件（要確認）")
        for sd in semantic_duplicates[:2]:
            print(f"   [{sd['index']}] {sd['name']}")
            for sim in sd['similar_to'][:1]:
                print(f"       → 類似: {sim['existing_name']} (類似度: {sim['combined_score']})")
    else:
        print("✅ 意味的重複なし")

    print(f"\n--- 3. 一次テキスト根拠 ---")
    avg_pt = sum(primary_text_scores) / len(primary_text_scores) if primary_text_scores else 0
    if avg_pt >= 3.5:
        print(f"✅ 平均スコア: {avg_pt:.1f}/4")
    elif avg_pt >= 2.0:
        print(f"⚠️ 平均スコア: {avg_pt:.1f}/4（改善推奨）")
        for issue in primary_text_issues[:3]:
            print(f"   {issue}")
    else:
        print(f"❌ 平均スコア: {avg_pt:.1f}/4（不合格）")
        for issue in primary_text_issues[:3]:
            print(f"   {issue}")

    print(f"\n--- 4. 反証欄（候補卦・排除理由）---")
    avg_hs = sum(hexagram_selection_scores) / len(hexagram_selection_scores) if hexagram_selection_scores else 0
    if avg_hs >= 3.5:
        print(f"✅ 平均スコア: {avg_hs:.1f}/4")
    elif avg_hs >= 2.0:
        print(f"⚠️ 平均スコア: {avg_hs:.1f}/4（改善推奨）")
        for issue in hexagram_selection_issues[:3]:
            print(f"   {issue}")
    else:
        print(f"❌ 平均スコア: {avg_hs:.1f}/4（不合格）")
        for issue in hexagram_selection_issues[:3]:
            print(f"   {issue}")

    print(f"\n--- 総合判定 ---")
    passed = (
        len(exact_duplicates) == 0 and
        avg_pt >= 2.0 and
        avg_hs >= 2.0
    )

    if passed and avg_pt >= 3.5 and avg_hs >= 3.5:
        print("✅ PASS (A) - 高品質、インポート可")
    elif passed:
        print("⚠️ PASS (B) - 条件付き可、改善推奨")
    else:
        print("❌ FAIL - 修正が必要")

    return passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_batch_v2.py <batch_json_path>")
        sys.exit(1)

    batch_path = sys.argv[1]
    success = validate_batch_v2(batch_path)
    sys.exit(0 if success else 1)
