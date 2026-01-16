#!/usr/bin/env python3
"""
event_id付与スクリプト

同一主体の複数フェーズ事例にevent_idを付与し、
イベント束ねを実現する。

Codex批判対応:
- event_phaseがあってもイベント束ねがないと機能しない
- 同一イベントの位相を束ね、位相間の整合性を機械検証可能にする
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def extract_year_from_period(period: str) -> int:
    """periodから開始年を抽出"""
    if not period:
        return 0
    match = re.search(r'(\d{4})', period)
    if match:
        return int(match.group(1))
    return 0

def normalize_target_name(name: str) -> str:
    """target_nameを正規化（表記ゆれ吸収）"""
    # 括弧内の説明を除去
    name = re.sub(r'[（\(].+?[）\)]', '', name)
    # 株式会社などを除去
    name = re.sub(r'(株式会社|株|有限会社|合同会社|Inc\.|Corp\.|Ltd\.)', '', name)
    # スペースを除去
    name = name.strip()
    return name

def generate_event_id(target_name: str, year: int, event_keywords: list) -> str:
    """event_idを生成

    フォーマット: EVT_{YEAR}_{TARGET_HASH}_{EVENT_HASH}
    """
    # ターゲット名のハッシュ（先頭6文字）
    target_normalized = normalize_target_name(target_name)
    target_hash = hashlib.md5(target_normalized.encode('utf-8')).hexdigest()[:6].upper()

    # イベントキーワードのハッシュ（先頭4文字）
    event_text = '_'.join(sorted(event_keywords)) if event_keywords else 'MAIN'
    event_hash = hashlib.md5(event_text.encode('utf-8')).hexdigest()[:4].upper()

    return f"EVT_{year}_{target_hash}_{event_hash}"

def extract_event_keywords(story: str) -> list:
    """ストーリーからイベントキーワードを抽出"""
    keywords = []

    keyword_patterns = [
        (r'買収|M&A|TOB', 'ACQ'),
        (r'合併|統合', 'MERGE'),
        (r'倒産|破産|清算', 'BANKRUPT'),
        (r'リストラ|人員削減|解雇', 'RESTRUCT'),
        (r'上場|IPO', 'IPO'),
        (r'撤退|売却', 'DIVEST'),
        (r'新規事業|参入', 'ENTRY'),
        (r'不正|不祥事|スキャンダル', 'SCANDAL'),
        (r'危機|ショック', 'CRISIS'),
        (r'回復|復活|再生', 'RECOVERY'),
    ]

    for pattern, keyword in keyword_patterns:
        if re.search(pattern, story):
            keywords.append(keyword)

    return keywords[:2]  # 最大2つ

def identify_event_groups(cases: list) -> dict:
    """同一イベントの事例をグループ化"""
    # target_name（正規化済み）でグループ化
    groups = defaultdict(list)

    for i, case in enumerate(cases):
        target = normalize_target_name(case.get('target_name', ''))
        if not target:
            continue
        groups[target].append((i, case))

    return groups

def assign_event_ids(dry_run: bool = True):
    """event_idを付与"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line.strip()))

    print(f"\n{'='*60}")
    print(f"event_id付与")
    print(f"{'='*60}")
    print(f"総事例数: {len(cases):,}件")

    # グループ化
    groups = identify_event_groups(cases)
    print(f"ユニーク主体数: {len(groups):,}")

    # 統計
    multi_phase_count = 0
    single_phase_count = 0
    event_ids_assigned = 0

    for target, case_list in groups.items():
        if len(case_list) == 1:
            # 単一事例の主体
            idx, case = case_list[0]
            year = extract_year_from_period(case.get('period', ''))
            keywords = extract_event_keywords(case.get('story_summary', ''))
            event_id = generate_event_id(case.get('target_name', ''), year, keywords)
            cases[idx]['event_id'] = event_id
            single_phase_count += 1
            event_ids_assigned += 1
        else:
            # 複数事例の主体 → 同一イベントの可能性
            multi_phase_count += 1

            # 年でソート
            case_list_sorted = sorted(case_list, key=lambda x: extract_year_from_period(x[1].get('period', '')))

            # 連続性をチェック（5年以内なら同一イベント）
            current_group = []
            last_year = 0

            for idx, case in case_list_sorted:
                year = extract_year_from_period(case.get('period', ''))
                keywords = extract_event_keywords(case.get('story_summary', ''))

                if not current_group or (year - last_year) <= 5:
                    current_group.append((idx, case, year, keywords))
                else:
                    # 新しいグループ開始
                    # 既存グループにevent_idを付与
                    if current_group:
                        first_year = current_group[0][2]
                        all_keywords = []
                        for _, c, _, kw in current_group:
                            all_keywords.extend(kw)
                        base_event_id = generate_event_id(
                            current_group[0][1].get('target_name', ''),
                            first_year,
                            list(set(all_keywords))[:2]
                        )
                        for g_idx, g_case, _, _ in current_group:
                            cases[g_idx]['event_id'] = base_event_id
                            event_ids_assigned += 1

                    current_group = [(idx, case, year, keywords)]

                last_year = year

            # 残りのグループを処理
            if current_group:
                first_year = current_group[0][2]
                all_keywords = []
                for _, c, _, kw in current_group:
                    all_keywords.extend(kw)
                base_event_id = generate_event_id(
                    current_group[0][1].get('target_name', ''),
                    first_year,
                    list(set(all_keywords))[:2]
                )
                for g_idx, g_case, _, _ in current_group:
                    cases[g_idx]['event_id'] = base_event_id
                    event_ids_assigned += 1

    print(f"\n--- 結果 ---")
    print(f"単一事例主体: {single_phase_count:,}")
    print(f"複数事例主体: {multi_phase_count:,}")
    print(f"event_id付与数: {event_ids_assigned:,}")

    # 同一event_idを持つ事例数の分布
    event_id_counts = defaultdict(int)
    for case in cases:
        eid = case.get('event_id')
        if eid:
            event_id_counts[eid] += 1

    multi_phase_events = [(eid, count) for eid, count in event_id_counts.items() if count > 1]
    print(f"複数フェーズイベント数: {len(multi_phase_events):,}")

    if dry_run:
        print(f"\n[DRY RUN] 変更は適用されません")
        # サンプル表示
        print(f"\n--- 複数フェーズイベントのサンプル ---")
        for eid, count in sorted(multi_phase_events, key=lambda x: -x[1])[:5]:
            related_cases = [c for c in cases if c.get('event_id') == eid]
            print(f"\n  {eid} ({count}件)")
            for c in related_cases[:3]:
                print(f"    - {c.get('target_name')[:30]} ({c.get('period')}) [{c.get('event_phase')}]")
    else:
        # バックアップ
        backup_path = cases_path.parent / f"cases_backup_eventid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"\n[バックアップ] {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(cases_path, 'r', encoding='utf-8') as src:
                f.write(src.read())

        # 保存
        print(f"[保存] {cases_path}")
        with open(cases_path, 'w', encoding='utf-8') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')

        print(f"\n✅ event_id付与完了")

    return cases

if __name__ == "__main__":
    import sys
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    assign_event_ids(dry_run=dry_run)
