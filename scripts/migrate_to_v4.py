#!/usr/bin/env python3
"""
v4.1スキーマへのマイグレーション

既存データにentity_type, event_driver_type, event_phaseを自動付与
（Codex批判対応: 主体型と事象ドライバー型を分離）
"""

import json
import re
from pathlib import Path
from datetime import datetime

# キーワード定義
GOVERNMENT_ENTITIES = ['日銀', 'FRB', 'ECB', 'BOJ', '中央銀行', '政府', '内閣', '首相', '大統領',
                       '財務省', '金融庁', '経産省', '厚労省', '国交省', 'SEC', 'FDA']
ORGANIZATION_ENTITIES = ['協会', '連合会', '組合', '財団', 'NPO', 'NGO', '団体']

POLICY_KEYWORDS = ['規制', '法改正', '関税', '政策', '金利', '利上げ', '利下げ',
                   '金融政策', '制裁', '補助金', '税制', '法律']
MARKET_KEYWORDS = ['バブル', '金融危機', 'リーマン', '景気', '不況', '株価', '相場',
                   '業界再編', '市場縮小', '需要減', 'インフレ', 'デフレ']
DISASTER_KEYWORDS = ['震災', '地震', '津波', '台風', '洪水', '豪雨', '噴火', '災害']
PANDEMIC_KEYWORDS = ['パンデミック', 'コロナ', 'COVID', '感染症', 'ウイルス', '疫病']
TECHNOLOGY_KEYWORDS = ['AI', '人工知能', 'DX', 'デジタル', 'イノベーション', '技術革新',
                       '自動化', 'IoT', 'クラウド', 'ブロックチェーン']
COMPETITION_KEYWORDS = ['M&A', '買収', '合併', '新規参入', '競合', 'シェア奪取', '価格競争']

def classify_entity_type(case: dict) -> str:
    """主体の型（エンティティ型）を分類

    主体が何者かを判定（会社/個人/政府/組織）
    """
    target_name = case.get('target_name', '')
    scale = case.get('scale', '')

    # 政府・規制当局チェック
    for kw in GOVERNMENT_ENTITIES:
        if kw in target_name:
            return 'government'

    # 組織（NPO、協会等）チェック
    for kw in ORGANIZATION_ENTITIES:
        if kw in target_name:
            return 'organization'

    # スケールベースの分類
    if scale == 'individual':
        return 'individual'
    if scale == 'family':
        return 'individual'
    if scale == 'country':
        return 'government'

    # デフォルトは企業
    return 'company'

def classify_event_driver_type(case: dict) -> str:
    """事象ドライバーの型を分類

    変化のトリガー/背景が何かを判定
    """
    story_summary = case.get('story_summary', '')
    trigger_type = case.get('trigger_type', '')

    # trigger_typeからの推論
    trigger_map = {
        '外部ショック': None,  # 詳細判定が必要
        '内部崩壊': 'internal',
        '意図的決断': 'internal',
        '偶発・出会い': 'internal',
    }

    # 詳細キーワードで判定
    # 災害
    for kw in DISASTER_KEYWORDS:
        if kw in story_summary:
            return 'disaster'

    # パンデミック
    for kw in PANDEMIC_KEYWORDS:
        if kw in story_summary:
            return 'pandemic'

    # 政策要因
    for kw in POLICY_KEYWORDS:
        if kw in story_summary:
            return 'policy'

    # 市場要因
    for kw in MARKET_KEYWORDS:
        if kw in story_summary:
            return 'market'

    # 技術変化
    for kw in TECHNOLOGY_KEYWORDS:
        if kw in story_summary:
            return 'technology'

    # 競争要因
    for kw in COMPETITION_KEYWORDS:
        if kw in story_summary:
            return 'competition'

    # trigger_typeからのフォールバック
    if trigger_type in trigger_map and trigger_map[trigger_type]:
        return trigger_map[trigger_type]

    # デフォルトは内部要因
    return 'internal'

def classify_subject_type(case: dict) -> str:
    """事例のsubject_typeを自動分類（後方互換性のため維持）"""
    entity_type = classify_entity_type(case)

    # entity_typeからsubject_typeへのマッピング
    if entity_type == 'government':
        return 'policy'
    if entity_type == 'individual':
        return 'individual'
    return 'company'

def infer_event_phase(case: dict) -> str:
    """事例のevent_phaseを推論"""
    story = case.get('story_summary', '')
    outcome = case.get('outcome', '')

    # 結果確定を示すキーワード
    if any(kw in story for kw in ['成功', '失敗', '倒産', '消滅', '完了', '達成']):
        return 'outcome'

    # 実行を示すキーワード
    if any(kw in story for kw in ['実施', '開始', '着手', '導入', '発足']):
        return 'execution'

    # 発表を示すキーワード
    if any(kw in story for kw in ['発表', '宣言', '計画', '検討']):
        return 'announcement'

    # outcomeがある場合は結果確定
    if outcome in ['Success', 'Failure']:
        return 'outcome'

    # デフォルト
    return 'completion'

def generate_subject_id(case: dict) -> str:
    """主体IDを生成"""
    import hashlib
    target = case.get('target_name', '').split('（')[0].strip()
    scale = case.get('scale', 'other')
    country = case.get('country', 'JP')

    # プレフィックス
    prefix_map = {
        'company': 'CORP',
        'individual': 'INDV',
        'country': 'GOVT',
        'other': 'OTHR'
    }
    prefix = prefix_map.get(scale, 'OTHR')

    # 英数字を抽出、なければハッシュを使用
    name_ascii = re.sub(r'[^a-zA-Z0-9]', '', target.upper())[:20]
    if not name_ascii:
        # 日本語名の場合はハッシュの先頭8文字を使用
        name_ascii = hashlib.md5(target.encode('utf-8')).hexdigest()[:8].upper()

    return f"{prefix}_{country}_{name_ascii}"

def migrate_cases(dry_run: bool = True):
    """既存データをv4形式にマイグレーション"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line.strip()))

    print(f"\n{'='*60}")
    print(f"v4.1マイグレーション（entity_type/event_driver_type分離）")
    print(f"{'='*60}")
    print(f"総事例数: {len(cases):,}件")

    # 統計
    stats = {
        'entity_type': {'company': 0, 'individual': 0, 'government': 0, 'organization': 0},
        'event_driver_type': {'internal': 0, 'market': 0, 'policy': 0, 'disaster': 0,
                              'pandemic': 0, 'technology': 0, 'competition': 0},
        'subject_type': {'company': 0, 'policy': 0, 'individual': 0, 'market': 0, 'exogenous': 0},
        'event_phase': {'announcement': 0, 'execution': 0, 'completion': 0, 'outcome': 0}
    }

    for case in cases:
        # entity_type（新フィールド）
        if not case.get('entity_type'):
            case['entity_type'] = classify_entity_type(case)
        if case['entity_type'] in stats['entity_type']:
            stats['entity_type'][case['entity_type']] += 1

        # event_driver_type（新フィールド）
        if not case.get('event_driver_type'):
            case['event_driver_type'] = classify_event_driver_type(case)
        if case['event_driver_type'] in stats['event_driver_type']:
            stats['event_driver_type'][case['event_driver_type']] += 1

        # subject_type（後方互換性）
        if not case.get('subject_type'):
            case['subject_type'] = classify_subject_type(case)
        if case['subject_type'] in stats['subject_type']:
            stats['subject_type'][case['subject_type']] += 1

        # event_phase
        if not case.get('event_phase'):
            case['event_phase'] = infer_event_phase(case)
        stats['event_phase'][case['event_phase']] += 1

        # primary_subject_id
        if not case.get('primary_subject_id'):
            case['primary_subject_id'] = generate_subject_id(case)

        # annotation_status
        if not case.get('annotation_status'):
            case['annotation_status'] = 'single'

    print(f"\n--- entity_type分布（主体型）---")
    for k, v in stats['entity_type'].items():
        print(f"  {k}: {v:,}件 ({v/len(cases)*100:.1f}%)")

    print(f"\n--- event_driver_type分布（事象ドライバー型）---")
    for k, v in stats['event_driver_type'].items():
        print(f"  {k}: {v:,}件 ({v/len(cases)*100:.1f}%)")

    print(f"\n--- subject_type分布（後方互換）---")
    for k, v in stats['subject_type'].items():
        print(f"  {k}: {v:,}件 ({v/len(cases)*100:.1f}%)")

    print(f"\n--- event_phase分布 ---")
    for k, v in stats['event_phase'].items():
        print(f"  {k}: {v:,}件 ({v/len(cases)*100:.1f}%)")

    if dry_run:
        print(f"\n[DRY RUN] 変更は適用されません")
        # サンプル表示
        print(f"\n--- サンプル（政策に分類された事例）---")
        policy_cases = [c for c in cases if c.get('subject_type') == 'policy'][:5]
        for c in policy_cases:
            print(f"  - {c.get('target_name')}")
    else:
        # バックアップ
        backup_path = cases_path.parent / f"cases_backup_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"\n[バックアップ] {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(cases_path, 'r', encoding='utf-8') as src:
                f.write(src.read())

        # 保存
        print(f"[保存] {cases_path}")
        with open(cases_path, 'w', encoding='utf-8') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')

        print(f"\n✅ マイグレーション完了")

    return cases

if __name__ == "__main__":
    import sys
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    migrate_cases(dry_run=dry_run)
