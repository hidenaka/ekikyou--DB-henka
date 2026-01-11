#!/usr/bin/env python3
"""
バッチ収集パイプライン

Wikipedia収集 → AI構造化 → 検証 → DB追加
を一括で実行する。
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import subprocess

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

import sys
sys.path.insert(0, str(SCRIPTS_DIR))

from collectors.wikipedia_collector import collect_person_articles, TARGET_CATEGORIES
from validators.quality_scorer import validate_case, QualityLevel


def load_existing_ids() -> set:
    """既存のtransition_idを読み込み"""
    ids = set()
    cases_file = DATA_DIR / "raw" / "cases.jsonl"
    if cases_file.exists():
        with open(cases_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("transition_id"):
                            ids.add(data["transition_id"])
                    except:
                        pass
    return ids


def generate_transition_id(scale: str, existing_ids: set) -> str:
    """新しいtransition_idを生成"""
    prefix_map = {
        "company": "CORP_JP",
        "individual": "PERS_JP",
        "family": "FAM_JP",
        "country": "COUN_JP",
        "other": "OTHR_JP"
    }
    prefix = prefix_map.get(scale, "OTHR_JP")

    # 最大番号を探す
    max_num = 0
    for tid in existing_ids:
        if tid.startswith(prefix):
            try:
                num = int(tid.split("_")[-1])
                max_num = max(max_num, num)
            except:
                pass

    return f"{prefix}_{max_num + 1:03d}"


def run_batch_collection(
    categories: List[str],
    articles_per_category: int = 20,
    min_quality: float = 40.0,
    dry_run: bool = False
) -> Dict:
    """
    バッチ収集を実行

    Args:
        categories: 収集するカテゴリのリスト
        articles_per_category: カテゴリあたりの記事数
        min_quality: 最低品質スコア
        dry_run: True の場合、実際のDB追加は行わない

    Returns:
        実行結果のサマリー
    """
    results = {
        "start_time": datetime.now().isoformat(),
        "categories": categories,
        "articles_per_category": articles_per_category,
        "min_quality": min_quality,
        "collected": 0,
        "processed": 0,
        "added_to_db": 0,
        "errors": [],
        "articles": []
    }

    print("=" * 60)
    print("バッチ収集パイプライン")
    print("=" * 60)
    print(f"カテゴリ数: {len(categories)}")
    print(f"カテゴリあたり: {articles_per_category}件")
    print(f"最低品質スコア: {min_quality}")
    print(f"ドライラン: {dry_run}")
    print("=" * 60)

    all_articles = []

    # Step 1: Wikipedia収集
    print("\n[Step 1] Wikipedia記事を収集中...")
    for category in categories:
        print(f"\nカテゴリ: {category}")
        try:
            articles = collect_person_articles(
                category,
                limit=articles_per_category,
                min_quality_score=min_quality
            )
            all_articles.extend(articles)
            results["collected"] += len(articles)
        except Exception as e:
            results["errors"].append(f"収集エラー ({category}): {str(e)}")
            print(f"エラー: {e}")

    print(f"\n収集完了: {len(all_articles)}件")

    if not all_articles:
        print("収集された記事がありません。終了します。")
        results["end_time"] = datetime.now().isoformat()
        return results

    # Step 2: 構造化（AI API必要、ここではスキップ可能）
    print("\n[Step 2] 構造化...")
    print("注意: Claude API が設定されていない場合はスキップされます")

    # 構造化されたデータを保存するパス
    pending_dir = DATA_DIR / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    structured_file = pending_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # 現時点では、収集した記事のメタデータのみを保存
    # 実際の構造化はAI APIで行う必要がある
    with open(structured_file, "w", encoding="utf-8") as f:
        for article in all_articles:
            # 基本情報のみ（構造化はAI APIで後から行う）
            meta = {
                "source_title": article.title,
                "source_url": article.url,
                "source_category": article.category,
                "career_text": article.career_text[:2000],  # テキストを短縮
                "birth_year": article.birth_year,
                "quality_score": article.quality_score,
                "collected_at": article.collected_at,
                "needs_structuring": True
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"メタデータを保存: {structured_file}")
    results["processed"] = len(all_articles)

    # Step 3: 検証（構造化後に実行）
    print("\n[Step 3] 検証...")
    print("構造化されたデータがないため、スキップします")

    # Step 4: DB追加（構造化後に実行）
    print("\n[Step 4] DB追加...")
    if dry_run:
        print("ドライランのため、DB追加はスキップします")
    else:
        print("構造化されたデータがないため、スキップします")

    results["end_time"] = datetime.now().isoformat()

    # サマリー
    print("\n" + "=" * 60)
    print("実行完了")
    print("=" * 60)
    print(f"収集: {results['collected']}件")
    print(f"処理: {results['processed']}件")
    print(f"DB追加: {results['added_to_db']}件")
    if results["errors"]:
        print(f"エラー: {len(results['errors'])}件")

    print(f"\n次のステップ:")
    print(f"1. {structured_file} を確認")
    print(f"2. Claude API で構造化を実行:")
    print(f"   python scripts/structurer/text_to_transitions.py -i {structured_file}")
    print(f"3. 構造化されたデータを検証:")
    print(f"   python scripts/validators/quality_scorer.py -i <output.jsonl>")
    print(f"4. DBに追加:")
    print(f"   python scripts/add_batch.py <verified.json>")

    # 結果を保存
    results_file = pending_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="バッチ収集パイプライン")
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=["日本の実業家"],
        help="収集するカテゴリ（複数指定可）"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="カテゴリあたりの記事数"
    )
    parser.add_argument(
        "--min-quality", "-q",
        type=float,
        default=40.0,
        help="最低品質スコア"
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="すべての対象カテゴリを収集"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン（DBへの追加なし）"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="対象カテゴリ一覧を表示"
    )

    args = parser.parse_args()

    if args.list_categories:
        print("対象カテゴリ一覧:")
        for cat in TARGET_CATEGORIES:
            print(f"  - {cat}")
        return

    categories = TARGET_CATEGORIES if args.all_categories else args.categories

    run_batch_collection(
        categories=categories,
        articles_per_category=args.limit,
        min_quality=args.min_quality,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
