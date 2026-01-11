#!/usr/bin/env python3
"""
データ品質改善ワークフロー 再開スクリプト

コンテキスト圧縮後にこのスクリプトを実行して、
現在のフェーズと次のアクションを確認します。

使用方法:
  python3 scripts/workflow_resume.py

出力:
  - 現在のフェーズとステータス
  - 次に実行すべきアクション
  - 各フェーズの進捗状況
"""

import json
from pathlib import Path
from datetime import datetime

def load_checkpoint(phase_name):
    """チェックポイントファイルを読み込み"""
    checkpoint_files = {
        'A': 'data/diagnostic/phase_a_progress.json',
        'B': 'data/diagnostic/phase_b_progress.json',
        'C': 'data/diagnostic/phase_c_progress.json',
        'D': 'data/diagnostic/phase_d_progress.json',
    }
    path = Path(checkpoint_files.get(phase_name, ''))
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

def analyze_db_status():
    """現在のDB状態を分析"""
    db_path = Path('data/raw/cases.jsonl')
    if not db_path.exists():
        return None

    with open(db_path, 'r') as f:
        cases = [json.loads(line.strip()) for line in f]

    # main_domain状況
    no_domain = sum(1 for c in cases if not c.get('main_domain') or c.get('main_domain') == 'unknown')
    domain_rate = (len(cases) - no_domain) / len(cases) * 100

    # country状況
    japan = sum(1 for c in cases if c.get('country') == '日本')
    unknown = sum(1 for c in cases if c.get('country', 'unknown') == 'unknown')
    international = len(cases) - japan - unknown

    # sources状況
    has_sources = sum(1 for c in cases if c.get('sources') and len(c.get('sources', [])) > 0)
    sources_rate = has_sources / len(cases) * 100

    # 時代分布
    pre_2010 = 0
    for c in cases:
        p = c.get('period', '')
        for year in range(1950, 2010):
            if str(year) in p:
                pre_2010 += 1
                break

    return {
        'total': len(cases),
        'domain_rate': domain_rate,
        'no_domain': no_domain,
        'japan': japan,
        'international': international,
        'unknown_country': unknown,
        'sources_rate': sources_rate,
        'no_sources': len(cases) - has_sources,
        'pre_2010': pre_2010,
    }

def determine_current_phase(status):
    """現在のフェーズを判定"""
    # Phase A: main_domain補完
    if status['domain_rate'] < 90:
        return 'A', 'main_domain補完', f"設定率{status['domain_rate']:.1f}%（目標90%）"

    # Phase B: 国際事例追加
    if status['international'] < 700:
        return 'B', '国際事例追加', f"海外事例{status['international']}件（目標700件）"

    # Phase C: ソースURL補完
    if status['sources_rate'] < 80:
        return 'C', 'ソースURL補完', f"設定率{status['sources_rate']:.1f}%（目標80%）"

    # Phase D: 歴史事例追加
    if status['pre_2010'] < 1600:
        return 'D', '歴史事例追加', f"2010年以前{status['pre_2010']}件（目標1600件）"

    return 'COMPLETED', 'ワークフロー完了', '全目標達成'

def main():
    print("=" * 60)
    print("データ品質改善ワークフロー - 状態確認")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # DB状態分析
    status = analyze_db_status()
    if not status:
        print("\nエラー: データベースが見つかりません")
        return

    print(f"\n【現在のDB状態】")
    print(f"  総ケース数: {status['total']:,}件")
    print(f"  main_domain設定率: {status['domain_rate']:.1f}%（未設定{status['no_domain']:,}件）")
    print(f"  海外事例: {status['international']:,}件（日本{status['japan']:,}件、不明{status['unknown_country']:,}件）")
    print(f"  ソースURL設定率: {status['sources_rate']:.1f}%（未設定{status['no_sources']:,}件）")
    print(f"  2010年以前の事例: {status['pre_2010']:,}件")

    # 現在のフェーズ判定
    phase, phase_name, progress = determine_current_phase(status)

    print(f"\n【現在のフェーズ】")
    print(f"  Phase {phase}: {phase_name}")
    print(f"  進捗: {progress}")

    # チェックポイント確認
    checkpoint = load_checkpoint(phase)
    if checkpoint:
        print(f"\n【チェックポイント】")
        print(f"  最終更新: {checkpoint.get('timestamp', 'N/A')}")
        print(f"  ステータス: {checkpoint.get('status', 'N/A')}")

    # 次のアクション
    print(f"\n【次のアクション】")
    if phase == 'A':
        print("  1. python3 scripts/enrich_main_domain.py --dry-run  # プレビュー")
        print("  2. python3 scripts/enrich_main_domain.py            # 実行")
    elif phase == 'B':
        print("  1. /add-batch-cases で海外事例を収集")
        print("  2. batch_international_XXX_NNN.json を作成")
        print("  3. python3 scripts/add_batch.py data/import/batch_international_*.json")
    elif phase == 'C':
        print("  1. python3 scripts/enrich_sources.py --dry-run  # プレビュー")
        print("  2. python3 scripts/enrich_sources.py            # 実行")
    elif phase == 'D':
        print("  1. 年代別の歴史事例を収集")
        print("  2. batch_historical_DECADE_NNN.json を作成")
        print("  3. python3 scripts/add_batch.py data/import/batch_historical_*.json")
    else:
        print("  全フェーズ完了！おめでとうございます！")

    print("\n" + "=" * 60)
    print("MCPメモリ確認コマンド:")
    print("  mcp__memory__open_nodes(['データ品質改善ワークフロー_2026-01'])")
    print("=" * 60)


if __name__ == '__main__':
    main()
