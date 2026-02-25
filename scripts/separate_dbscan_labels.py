#!/usr/bin/env python3
"""
Phase 2B 品質レビュー修正: DBSCAN ラベル配列の分離 (M2)
========================================================

cluster_results.json (2.3MB) から DBSCAN の labels 配列を
別ファイル (dbscan_labels.json) に分離し、JSONサイズを適正化する。

入力: analysis/phase2/cluster_results.json
出力:
  - analysis/phase2/dbscan_labels.json    — 分離されたラベル配列
  - analysis/phase2/cluster_results.json  — labels を参照パスに置換
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PHASE2_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'phase2')
CLUSTER_JSON = os.path.join(PHASE2_DIR, 'cluster_results.json')
LABELS_JSON = os.path.join(PHASE2_DIR, 'dbscan_labels.json')


def main():
    print("=" * 70)
    print("Phase 2B DBSCAN ラベル配列分離 (品質レビュー修正 M2)")
    print("=" * 70)

    # 元ファイルサイズ
    orig_size = os.path.getsize(CLUSTER_JSON)
    print(f"  元ファイルサイズ: {orig_size:,} bytes ({orig_size / 1024:.1f} KB)")

    # 読み込み
    with open(CLUSTER_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # DBSCAN parameter_search から labels を抽出
    dbscan = data.get('clustering_comparison', {}).get('dbscan', {})
    param_search = dbscan.get('parameter_search', [])

    labels_store = {}
    extracted_count = 0
    total_elements = 0

    for i, entry in enumerate(param_search):
        labels = entry.get('labels', [])
        if labels and len(labels) > 0:
            key = f"entry_{i}_eps{entry.get('eps', 'na')}_ms{entry.get('min_samples', 'na')}"
            labels_store[key] = labels
            total_elements += len(labels)
            extracted_count += 1
            # labels を参照に置換
            entry['labels'] = f"see dbscan_labels.json key={key}"
        else:
            # 空の配列はそのまま残す
            entry['labels'] = []

    print(f"  抽出したラベル配列: {extracted_count} エントリ")
    print(f"  合計ラベル要素数: {total_elements:,}")

    # ラベルファイルを保存
    with open(LABELS_JSON, 'w', encoding='utf-8') as f:
        json.dump(labels_store, f, ensure_ascii=False)
    labels_size = os.path.getsize(LABELS_JSON)
    print(f"  dbscan_labels.json: {labels_size:,} bytes ({labels_size / 1024:.1f} KB)")

    # cluster_results.json のDBSCANセクションにメタデータ追加
    dbscan['labels_file'] = 'dbscan_labels.json'
    dbscan['labels_separated'] = True
    dbscan['n_entries_with_labels'] = extracted_count

    # 更新されたcluster_results.jsonを保存
    with open(CLUSTER_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    new_size = os.path.getsize(CLUSTER_JSON)

    print(f"\n  cluster_results.json:")
    print(f"    元サイズ:  {orig_size:,} bytes ({orig_size / 1024:.1f} KB)")
    print(f"    新サイズ:  {new_size:,} bytes ({new_size / 1024:.1f} KB)")
    print(f"    削減率:    {(1 - new_size / orig_size) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("完了")
    print("=" * 70)


if __name__ == '__main__':
    main()
