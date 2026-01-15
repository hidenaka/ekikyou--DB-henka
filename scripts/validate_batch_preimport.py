#!/usr/bin/env python3
"""
バッチインポート前検証スクリプト

Codex指摘に基づく品質チェック:
1. 重複チェック（0件目標）
2. logic_memo品質チェック
3. ソース品質チェック
4. 爻分布チェック
"""

import json
import sys
from pathlib import Path
from collections import Counter

def load_existing_cases():
    """既存DBの(target_name, period)セットを取得"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    existing = set()
    if cases_path.exists():
        with open(cases_path, 'r', encoding='utf-8') as f:
            for line in f:
                case = json.loads(line.strip())
                key = (case.get('target_name', ''), case.get('period', ''))
                existing.add(key)
    return existing

def check_logic_memo_quality(memo):
    """logic_memoの品質スコア（5要素チェック）"""
    if not memo or len(memo) < 50:
        return 0, "短すぎる（50字未満）"

    score = 0
    missing = []

    # 八卦キーワードチェック
    bagua = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']
    if any(b in memo for b in bagua):
        score += 1
    else:
        missing.append("八卦言及なし")

    # 状態変化の記述
    if '（' in memo and '）' in memo:
        score += 1
    else:
        missing.append("八卦対応記述なし")

    # 複数の段階記述
    transition_words = ['から', 'へ', 'により', 'を経て', 'に至り', 'に転落', 'を断行']
    if any(w in memo for w in transition_words):
        score += 1
    else:
        missing.append("変化記述なし")

    # 具体性
    if len(memo) >= 100:
        score += 1
    else:
        missing.append("具体性不足（100字未満）")

    # 結果記述
    result_words = ['回復', '成功', '失敗', '達成', '崩壊', '安定', '成長', '衰退']
    if any(w in memo for w in result_words):
        score += 1
    else:
        missing.append("結果記述なし")

    return score, ", ".join(missing) if missing else "OK"

def validate_batch(batch_path):
    """バッチファイルを検証"""
    print(f"\n{'='*60}")
    print(f"検証: {batch_path}")
    print(f"{'='*60}")

    # ファイル読み込み
    with open(batch_path, 'r', encoding='utf-8') as f:
        batch = json.load(f)

    if not isinstance(batch, list):
        print("ERROR: JSONが配列形式でない")
        return False

    print(f"\n件数: {len(batch)}件")

    # 既存DBロード
    existing = load_existing_cases()
    print(f"既存DB: {len(existing)}件")

    # チェック結果
    duplicates = []
    low_quality_memos = []
    missing_urls = []
    yao_distribution = Counter()
    hex_distribution = Counter()

    for i, case in enumerate(batch):
        # 重複チェック
        key = (case.get('target_name', ''), case.get('period', ''))
        if key in existing:
            duplicates.append(f"[{i+1}] {key[0]} ({key[1]})")

        # logic_memoチェック
        memo = case.get('logic_memo', '')
        score, issue = check_logic_memo_quality(memo)
        if score < 3:
            low_quality_memos.append(f"[{i+1}] スコア{score}/5: {issue}")

        # URLチェック
        if not case.get('source_url'):
            missing_urls.append(f"[{i+1}] {case.get('target_name', 'Unknown')}")

        # 分布集計
        yao = case.get('yao_analysis', {}).get('assigned_yao')
        if yao:
            yao_distribution[yao] += 1

        hex_id = case.get('before_hexagram_id')
        if hex_id:
            hex_distribution[hex_id] += 1

    # 結果表示
    print(f"\n--- 重複チェック ---")
    if duplicates:
        print(f"❌ 重複: {len(duplicates)}件（目標: 0件）")
        for d in duplicates[:5]:
            print(f"   {d}")
        if len(duplicates) > 5:
            print(f"   ...他{len(duplicates)-5}件")
    else:
        print("✅ 重複なし")

    print(f"\n--- logic_memo品質 ---")
    if low_quality_memos:
        print(f"⚠️ 低品質: {len(low_quality_memos)}件")
        for m in low_quality_memos[:3]:
            print(f"   {m}")
    else:
        print("✅ 全件品質OK")

    print(f"\n--- ソースURL ---")
    url_rate = (len(batch) - len(missing_urls)) / len(batch) * 100 if batch else 0
    if url_rate < 80:
        print(f"⚠️ URL記載率: {url_rate:.1f}%（目標: 80%以上）")
    else:
        print(f"✅ URL記載率: {url_rate:.1f}%")

    print(f"\n--- 爻分布 ---")
    for yao in sorted(yao_distribution.keys()):
        print(f"   {yao}爻: {yao_distribution[yao]}件")

    print(f"\n--- 総合判定 ---")
    passed = len(duplicates) == 0 and len(low_quality_memos) <= len(batch) * 0.2
    if passed:
        print("✅ PASS - インポート可")
    else:
        print("❌ FAIL - 修正が必要")

    return passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_batch_preimport.py <batch_json_path>")
        sys.exit(1)

    batch_path = sys.argv[1]
    success = validate_batch(batch_path)
    sys.exit(0 if success else 1)
