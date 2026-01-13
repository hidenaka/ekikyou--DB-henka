#!/usr/bin/env python3
"""
パイロット検証セット抽出スクリプト
易経変化ロジックDBから30件の層化サンプルを抽出

層構成:
- verified: 10件（trust_level='verified'から無作為抽出）
- high_quality_unverified: 10件（一次/二次ソースあり、未検証）
- low_quality: 10件（ソース品質が低い、または不明）
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "pilot" / "pilot_verification_30.jsonl"

# ソース品質判定用パターン
HIGH_QUALITY_DOMAINS = [
    # 政府・公的機関
    r'\.go\.jp',
    r'\.gov',
    r'\.gov\.uk',
    r'\.gouv\.fr',
    # 学術機関
    r'\.ac\.jp',
    r'\.edu',
    # 主要メディア
    r'nikkei\.com',
    r'reuters\.com',
    r'bloomberg\.com',
    r'wsj\.com',
    r'ft\.com',
    r'nytimes\.com',
    r'nhk\.or\.jp',
    r'asahi\.com',
    r'yomiuri\.co\.jp',
    r'mainichi\.jp',
    # 企業公式（IR含む）
    r'/ir/',
    r'/investor',
    r'/press/',
    r'/news/',
    r'\.co\.jp',  # 日本企業ドメイン
]

LOW_QUALITY_PATTERNS = [
    r'wikipedia\.org',
    r'google\.com/search',
    r'^$',  # 空
]

def load_cases() -> list[dict]:
    """cases.jsonlを読み込む"""
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

def has_high_quality_source(sources: list[str]) -> bool:
    """一次/二次ソース（高品質ドメイン）を持つか判定"""
    if not sources:
        return False

    for source in sources:
        for pattern in HIGH_QUALITY_DOMAINS:
            if re.search(pattern, source, re.IGNORECASE):
                return True
    return False

def has_low_quality_source(sources: list[str]) -> bool:
    """低品質ソースのみか判定"""
    if not sources:
        return True

    # すべてのソースが低品質パターンに該当するか
    for source in sources:
        is_low_quality = False
        for pattern in LOW_QUALITY_PATTERNS:
            if re.search(pattern, source, re.IGNORECASE):
                is_low_quality = True
                break
        if not is_low_quality:
            # 高品質でもない場合は中程度として扱う
            if not any(re.search(p, source, re.IGNORECASE) for p in HIGH_QUALITY_DOMAINS):
                # 明確な低品質でなければFalse
                return False
    return True

def classify_case(case: dict) -> str:
    """事例を層に分類"""
    trust_level = case.get('trust_level', 'unverified')
    sources = case.get('sources', [])

    if trust_level == 'verified':
        return 'verified'
    elif has_high_quality_source(sources):
        return 'high_quality_unverified'
    elif has_low_quality_source(sources):
        return 'low_quality'
    else:
        # 中間品質はhigh_quality側に含める
        return 'high_quality_unverified'

def create_pilot_record(case: dict, pilot_id: str, stratum: str) -> dict:
    """パイロット検証用レコードを作成"""
    return {
        "pilot_id": pilot_id,
        "original_id": case.get('transition_id', 'UNKNOWN'),
        "stratum": stratum,
        "target_name": case.get('target_name', ''),
        "outcome": case.get('outcome', ''),
        "sources": case.get('sources', []),
        "current_trust_level": case.get('trust_level', 'unverified'),
        "verification_template": {
            "outcome_status": None,  # "confirmed" | "disputed" | "unknown"
            "verification_confidence": None,  # 0.0-1.0
            "coi_status": None,  # "none" | "potential" | "significant"
            "reviewer_notes": ""
        },
        # 追加情報（検証に役立つ）
        "_metadata": {
            "story_summary": case.get('story_summary', ''),
            "period": case.get('period', ''),
            "scale": case.get('scale', ''),
            "pattern_type": case.get('pattern_type', ''),
            "extraction_date": datetime.now().isoformat()
        }
    }

def extract_pilot_set(cases: list[dict], seed: int = 42) -> list[dict]:
    """層化抽出を実行"""
    random.seed(seed)

    # 層別に分類
    verified_cases = []
    high_quality_cases = []
    low_quality_cases = []

    for case in cases:
        stratum = classify_case(case)
        if stratum == 'verified':
            verified_cases.append(case)
        elif stratum == 'high_quality_unverified':
            high_quality_cases.append(case)
        else:
            low_quality_cases.append(case)

    print(f"\n=== 層別分類結果 ===")
    print(f"verified: {len(verified_cases)}件")
    print(f"high_quality_unverified: {len(high_quality_cases)}件")
    print(f"low_quality: {len(low_quality_cases)}件")

    # 各層から10件ずつ抽出
    pilot_records = []

    # verified層
    sampled_verified = random.sample(verified_cases, min(10, len(verified_cases)))
    for i, case in enumerate(sampled_verified, 1):
        pilot_id = f"P{i:03d}"
        pilot_records.append(create_pilot_record(case, pilot_id, 'verified'))

    # high_quality層
    sampled_high = random.sample(high_quality_cases, min(10, len(high_quality_cases)))
    for i, case in enumerate(sampled_high, 11):
        pilot_id = f"P{i:03d}"
        pilot_records.append(create_pilot_record(case, pilot_id, 'high_quality'))

    # low_quality層
    sampled_low = random.sample(low_quality_cases, min(10, len(low_quality_cases)))
    for i, case in enumerate(sampled_low, 21):
        pilot_id = f"P{i:03d}"
        pilot_records.append(create_pilot_record(case, pilot_id, 'low_quality'))

    return pilot_records

def save_pilot_set(pilot_records: list[dict]):
    """パイロットセットを保存"""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in pilot_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n=== 出力完了 ===")
    print(f"ファイル: {OUTPUT_FILE}")
    print(f"件数: {len(pilot_records)}件")

def display_summary(pilot_records: list[dict]):
    """サマリーを表示"""
    print("\n" + "=" * 60)
    print("パイロット検証セット サマリー")
    print("=" * 60)

    # 層別集計
    strata_counts = {}
    for record in pilot_records:
        stratum = record['stratum']
        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1

    print(f"\n【層別構成】")
    for stratum, count in sorted(strata_counts.items()):
        print(f"  {stratum}: {count}件")

    # 各層の事例リスト
    for stratum in ['verified', 'high_quality', 'low_quality']:
        print(f"\n【{stratum}層 - 抽出事例】")
        stratum_records = [r for r in pilot_records if r['stratum'] == stratum]
        for record in stratum_records:
            sources = record.get('sources') or []
            sources_str = ', '.join(sources[:2]) if sources else '(なし)'
            if len(sources) > 2:
                sources_str += f' (+{len(sources)-2})'
            print(f"  {record['pilot_id']}: {record['target_name']}")
            print(f"         outcome={record['outcome']}, sources={sources_str[:60]}...")

    print("\n" + "=" * 60)
    print("検証テンプレート説明:")
    print("  outcome_status: 'confirmed' | 'disputed' | 'unknown'")
    print("  verification_confidence: 0.0〜1.0 (検証の確信度)")
    print("  coi_status: 'none' | 'potential' | 'significant'")
    print("  reviewer_notes: 検証者のメモ")
    print("=" * 60)

def main():
    print("パイロット検証セット抽出を開始...")

    # データ読み込み
    cases = load_cases()
    print(f"総事例数: {len(cases)}件")

    # 層化抽出
    pilot_records = extract_pilot_set(cases)

    # 保存
    save_pilot_set(pilot_records)

    # サマリー表示
    display_summary(pilot_records)

    return pilot_records

if __name__ == '__main__':
    main()
