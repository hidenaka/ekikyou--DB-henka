#!/usr/bin/env python3
"""
auto_verify_high_confidence.py
------------------------------
verification_confidence=high の事例を自動的にverified_correctに昇格させる

検証ルール:
1. verification_confidence = 'high'
2. coi_status != 'self' (自己報告でない)
3. outcome が明確 (Success/Failure/Mixed/PartialSuccess)
4. ソースURLが有効 (httpで始まる)

更新内容:
- outcome_status: 'verified_correct'
- trust_level: 'verified'
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime


# 有効なoutcome値
VALID_OUTCOMES = {'Success', 'Failure', 'Mixed', 'PartialSuccess'}


def has_valid_source_url(sources):
    """ソースURLがhttpで始まるかチェック"""
    if not sources or not isinstance(sources, list):
        return False
    for source in sources:
        if isinstance(source, str) and source.startswith('http'):
            return True
    return False


def check_eligibility(case):
    """昇格条件をチェックし、失敗理由を返す"""
    reasons = []

    # 条件1: verification_confidence = 'high'
    if case.get('verification_confidence') != 'high':
        reasons.append('confidence_not_high')
        return False, reasons

    # 既にverified_correctの場合はスキップ
    if case.get('outcome_status') == 'verified_correct':
        reasons.append('already_verified')
        return False, reasons

    # 条件2: coi_status != 'self'
    if case.get('coi_status') == 'self':
        reasons.append('coi_self_report')

    # 条件3: outcomeが明確
    outcome = case.get('outcome')
    if outcome not in VALID_OUTCOMES:
        reasons.append(f'invalid_outcome:{outcome}')

    # 条件4: ソースURLが有効
    if not has_valid_source_url(case.get('sources')):
        reasons.append('no_valid_source_url')

    return len(reasons) == 0, reasons


def main():
    # パス設定
    base_dir = Path(__file__).parent.parent.parent
    cases_file = base_dir / 'data' / 'raw' / 'cases.jsonl'

    print("=" * 60)
    print("High Confidence 事例自動検証ツール")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"対象ファイル: {cases_file}")
    print()

    # ファイル読み込み
    cases = []
    with open(cases_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    total_cases = len(cases)
    print(f"総事例数: {total_cases:,}件")

    # 統計カウンタ
    stats = {
        'high_confidence_total': 0,
        'already_verified': 0,
        'promoted': 0,
        'failed': 0,
    }
    failure_reasons = Counter()

    # 処理
    updated_cases = []
    for case in cases:
        if case.get('verification_confidence') == 'high':
            stats['high_confidence_total'] += 1

        eligible, reasons = check_eligibility(case)

        if eligible:
            # 昇格処理
            case['outcome_status'] = 'verified_correct'
            case['trust_level'] = 'verified'
            stats['promoted'] += 1
        elif 'already_verified' in reasons:
            stats['already_verified'] += 1
        elif case.get('verification_confidence') == 'high':
            stats['failed'] += 1
            for reason in reasons:
                failure_reasons[reason] += 1

        updated_cases.append(case)

    # ファイル書き込み
    with open(cases_file, 'w', encoding='utf-8') as f:
        for case in updated_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    # 新しい検証済み率を計算
    new_verified_count = sum(
        1 for c in updated_cases
        if c.get('outcome_status') == 'verified_correct'
    )
    verified_rate = (new_verified_count / total_cases) * 100

    # レポート出力
    print()
    print("=" * 60)
    print("処理結果レポート")
    print("=" * 60)
    print()
    print("[High Confidence事例の内訳]")
    print(f"  verification_confidence=high 総数: {stats['high_confidence_total']:,}件")
    print(f"  既にverified_correct:            {stats['already_verified']:,}件")
    print(f"  今回昇格:                         {stats['promoted']:,}件")
    print(f"  昇格不可:                         {stats['failed']:,}件")
    print()
    print("[全体の検証状況]")
    print(f"  総事例数:                         {total_cases:,}件")
    print(f"  verified_correct (昇格後):        {new_verified_count:,}件")
    print(f"  検証済み率:                       {verified_rate:.1f}%")
    print()

    if failure_reasons:
        print("[昇格できなかった理由の内訳]")
        for reason, count in failure_reasons.most_common():
            print(f"  {reason}: {count:,}件")
        print()

    print("=" * 60)
    print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return {
        'promoted': stats['promoted'],
        'new_verified_count': new_verified_count,
        'verified_rate': verified_rate,
        'failure_reasons': dict(failure_reasons)
    }


if __name__ == '__main__':
    result = main()
    print()
    print(f"[サマリー] {result['promoted']}件を昇格、検証済み率 {result['verified_rate']:.1f}%")
