#!/usr/bin/env python3
"""
enrich_yao_phrase.py - 事例のyao_analysisにyao_master.jsonの爻辞を追加

使用方法:
    python3 scripts/enrich_yao_phrase.py --dry-run   # 確認のみ
    python3 scripts/enrich_yao_phrase.py --apply     # 実際に適用
    python3 scripts/enrich_yao_phrase.py --apply --infer-position  # 爻位置がない場合は推定

要件:
- 事例には既に before_hexagram_id と before_yao_position がある
- yao_master.jsonには64卦×6爻の爻辞がある
- 既に爻辞がある事例はスキップ
- --infer-position: 爻位置がない場合、changing_lines_1から推定、またはデフォルト3爻を使用
"""

import json
import argparse
import sys
from pathlib import Path

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
YAO_MASTER_FILE = BASE_DIR / "data" / "hexagrams" / "yao_master.json"


def load_yao_master():
    """yao_master.jsonを読み込み、辞書として返す"""
    with open(YAO_MASTER_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_yao_phrase(yao_master, hexagram_id, yao_position):
    """
    指定された卦IDと爻位置から爻辞を取得

    Args:
        yao_master: yao_master.jsonのデータ
        hexagram_id: 卦ID (1-64)
        yao_position: 爻位置 (1-6)

    Returns:
        tuple: (classic, modern) または (None, None)
    """
    hex_key = str(hexagram_id)
    yao_key = str(yao_position)

    if hex_key not in yao_master:
        return None, None

    hexagram_data = yao_master[hex_key]
    if 'yao' not in hexagram_data:
        return None, None

    yao_data = hexagram_data['yao']
    if yao_key not in yao_data:
        return None, None

    yao_entry = yao_data[yao_key]
    return yao_entry.get('classic'), yao_entry.get('modern')


def infer_yao_position(case):
    """
    爻位置を推定する

    Args:
        case: 事例データ

    Returns:
        int: 推定された爻位置 (1-6)、推定できない場合は3（転換期）
    """
    # changing_lines_1から推定
    changing = case.get('changing_lines_1')
    if changing and len(changing) > 0:
        # 最初の変爻を使用
        return changing[0]

    # デフォルトは3爻（転換期・岐路）
    return 3


def process_cases(dry_run=True, infer_position=False):
    """
    cases.jsonlを処理し、爻辞を追加

    Args:
        dry_run: Trueの場合は変更を適用しない
        infer_position: Trueの場合、爻位置がない場合は推定する

    Returns:
        dict: 処理結果の統計
    """
    # yao_masterを読み込み
    print(f"Loading yao_master.json from {YAO_MASTER_FILE}...")
    yao_master = load_yao_master()
    print(f"  Loaded {len(yao_master)} hexagrams")

    # 統計
    stats = {
        'total': 0,
        'already_has_phrase': 0,
        'enriched': 0,
        'enriched_with_inferred': 0,
        'no_hexagram_id': 0,
        'no_yao_position': 0,
        'phrase_not_found': 0,
        'errors': []
    }

    # 処理結果を保持
    updated_cases = []

    print(f"\nProcessing cases from {CASES_FILE}...")

    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats['total'] += 1

            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: JSON parse error - {e}")
                updated_cases.append(line)
                continue

            yao_analysis = case.get('yao_analysis', {})

            # 既に爻辞がある場合はスキップ
            if yao_analysis.get('yao_phrase_classic') and yao_analysis.get('yao_phrase_modern'):
                stats['already_has_phrase'] += 1
                updated_cases.append(line)
                continue

            # before_hexagram_idとbefore_yao_positionを取得
            hexagram_id = yao_analysis.get('before_hexagram_id')
            yao_position = yao_analysis.get('before_yao_position')

            if hexagram_id is None:
                stats['no_hexagram_id'] += 1
                updated_cases.append(line)
                continue

            position_inferred = False
            if yao_position is None:
                if infer_position:
                    # 爻位置を推定
                    yao_position = infer_yao_position(case)
                    position_inferred = True
                    # 推定した爻位置をyao_analysisに設定
                    case['yao_analysis']['before_yao_position'] = yao_position
                else:
                    stats['no_yao_position'] += 1
                    updated_cases.append(line)
                    continue

            # 爻辞を取得
            classic, modern = get_yao_phrase(yao_master, hexagram_id, yao_position)

            if classic is None or modern is None:
                stats['phrase_not_found'] += 1
                stats['errors'].append(
                    f"Line {line_num} ({case.get('transition_id', 'unknown')}): "
                    f"Phrase not found for hexagram {hexagram_id}, yao {yao_position}"
                )
                updated_cases.append(line)
                continue

            # 爻辞を追加
            case['yao_analysis']['yao_phrase_classic'] = classic
            case['yao_analysis']['yao_phrase_modern'] = modern
            if position_inferred:
                stats['enriched_with_inferred'] += 1
            else:
                stats['enriched'] += 1

            # 更新されたJSONを保持
            updated_cases.append(json.dumps(case, ensure_ascii=False) + '\n')

    # 変更を適用
    total_enriched = stats['enriched'] + stats['enriched_with_inferred']
    if not dry_run and total_enriched > 0:
        print(f"\nApplying changes to {CASES_FILE}...")
        with open(CASES_FILE, 'w', encoding='utf-8') as f:
            for case_line in updated_cases:
                f.write(case_line if case_line.endswith('\n') else case_line + '\n')
        print("  Changes applied successfully!")

    return stats


def print_report(stats, dry_run, infer_position=False):
    """処理結果のレポートを出力"""
    print("\n" + "=" * 60)
    print("ENRICHMENT REPORT")
    print("=" * 60)

    print(f"\nTotal cases processed: {stats['total']}")
    print(f"  - Already has phrase:      {stats['already_has_phrase']}")
    print(f"  - Enriched:                {stats['enriched']}")
    if infer_position:
        print(f"  - Enriched (w/ inferred):  {stats['enriched_with_inferred']}")
    print(f"  - No hexagram_id:          {stats['no_hexagram_id']}")
    print(f"  - No yao_position:         {stats['no_yao_position']}")
    print(f"  - Phrase not found:        {stats['phrase_not_found']}")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # 最大10件表示
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN MODE - No changes were made")
        print("Run with --apply to apply changes")
    else:
        print("CHANGES APPLIED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='事例のyao_analysisにyao_master.jsonの爻辞を追加'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true',
                       help='確認のみ（変更を適用しない）')
    group.add_argument('--apply', action='store_true',
                       help='変更を適用する')

    parser.add_argument('--infer-position', action='store_true',
                        help='爻位置がない場合は推定する（changing_lines_1またはデフォルト3爻）')

    args = parser.parse_args()

    # ファイルの存在確認
    if not CASES_FILE.exists():
        print(f"Error: Cases file not found: {CASES_FILE}", file=sys.stderr)
        sys.exit(1)

    if not YAO_MASTER_FILE.exists():
        print(f"Error: Yao master file not found: {YAO_MASTER_FILE}", file=sys.stderr)
        sys.exit(1)

    # 処理実行
    dry_run = args.dry_run
    infer_position = args.infer_position
    stats = process_cases(dry_run=dry_run, infer_position=infer_position)

    # レポート出力
    print_report(stats, dry_run, infer_position)

    # 終了コード
    total_enrichable = stats['enriched'] + stats['enriched_with_inferred']
    if total_enrichable > 0 and dry_run:
        print(f"\n{total_enrichable} cases can be enriched.")

    sys.exit(0)


if __name__ == '__main__':
    main()
