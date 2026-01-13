#!/usr/bin/env python3
"""
Phase 2: 六十四卦マッピング用 層化サンプリング
8つのtriggerカテゴリから各8件、計64件を抽出
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# triggerカテゴリと特徴キーワード定義
TRIGGER_CATEGORIES = {
    "T_EXTERNAL_FORCE": {
        "label": "外部強制力",
        "keywords": ["リーマン", "震災", "コロナ", "パンデミック", "規制", "法改正",
                    "金融危機", "オイルショック", "サブプライム", "バブル崩壊", "円高"]
    },
    "T_ENVIRONMENT": {
        "label": "環境変化",
        "keywords": ["市場縮小", "競合", "技術革新", "デジタル化", "需要変化",
                    "業界再編", "価格競争", "新興国", "シェア", "市場環境"]
    },
    "T_INTERNAL_CRISIS": {
        "label": "内部危機",
        "keywords": ["不祥事", "粉飾", "品質問題", "経営失敗", "赤字", "債務超過",
                    "リコール", "スキャンダル", "隠蔽", "偽装", "欠陥", "改ざん"]
    },
    "T_LEADERSHIP": {
        "label": "リーダー起因",
        "keywords": ["創業者", "社長交代", "CEO", "後継者", "カリスマ", "経営者",
                    "会長", "代表取締役", "オーナー", "創設者", "リーダー"]
    },
    "T_OPPORTUNITY": {
        "label": "機会発見",
        "keywords": ["新市場", "ブレイクスルー", "発見", "チャンス", "可能性", "新技術",
                    "革新", "発明", "開発成功", "特許", "新製品"]
    },
    "T_STAGNATION": {
        "label": "停滞",
        "keywords": ["停滞", "行き詰まり", "低迷", "伸び悩み", "成長鈍化",
                    "頭打ち", "マンネリ", "硬直", "沈滞"]
    },
    "T_GROWTH_MOMENTUM": {
        "label": "成長の勢い",
        "keywords": ["成長期", "拡大期", "好調", "急成長", "躍進",
                    "右肩上がり", "高成長", "絶好調", "ブーム", "拡大"]
    },
    "T_INTERACTION": {
        "label": "外部との相互作用",
        "keywords": ["提携", "M&A", "買収", "合併", "アライアンス", "JV",
                    "資本提携", "業務提携", "統合", "連携"]
    }
}

def classify_trigger(story_summary: str) -> Tuple[str, int]:
    """
    story_summaryからtriggerカテゴリを推定
    戻り値: (カテゴリID, マッチしたキーワード数)
    """
    if not story_summary:
        return ("T_UNKNOWN", 0)

    best_category = "T_UNKNOWN"
    best_score = 0

    for cat_id, cat_info in TRIGGER_CATEGORIES.items():
        score = 0
        for keyword in cat_info["keywords"]:
            if keyword in story_summary:
                score += 1
        if score > best_score:
            best_score = score
            best_category = cat_id

    return (best_category, best_score)


def load_cases(input_path: str) -> List[dict]:
    """JSONLファイルから事例を読み込み"""
    cases = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cases


def stratified_sample(cases: List[dict], samples_per_category: int = 8) -> List[dict]:
    """
    層化サンプリング: 各triggerカテゴリから指定件数を抽出
    """
    # カテゴリごとに事例を分類
    categorized: Dict[str, List[dict]] = {cat_id: [] for cat_id in TRIGGER_CATEGORIES.keys()}
    unclassified = []

    for case in cases:
        story = case.get('story_summary', '') or case.get('pre_outcome_text', '')
        cat_id, score = classify_trigger(story)

        if cat_id != "T_UNKNOWN" and score > 0:
            categorized[cat_id].append({
                "case": case,
                "score": score
            })
        else:
            unclassified.append(case)

    # 各カテゴリの状況を表示
    print("=" * 60)
    print("triggerカテゴリ別の事例数:")
    print("=" * 60)
    for cat_id, cat_info in TRIGGER_CATEGORIES.items():
        count = len(categorized[cat_id])
        print(f"  {cat_id:20s} ({cat_info['label']:12s}): {count:5d}件")
    print(f"  {'未分類':20s}: {len(unclassified):5d}件")
    print("=" * 60)

    # 各カテゴリからサンプリング
    sampled = []

    for cat_id in TRIGGER_CATEGORIES.keys():
        candidates = categorized[cat_id]

        if len(candidates) >= samples_per_category:
            # スコアが高い順にソートしてからランダム抽出
            # (まずスコア上位者を優先しつつ、その中からランダムに選ぶ)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            top_pool = candidates[:min(len(candidates), samples_per_category * 3)]
            selected = random.sample(top_pool, samples_per_category)
        else:
            # 候補が足りない場合は全て使用
            selected = candidates
            print(f"警告: {cat_id} の事例が不足 ({len(candidates)}件 < {samples_per_category}件)")

        for item in selected:
            case = item["case"]
            sampled.append({
                "case_id": case.get("transition_id", ""),
                "entity_name": case.get("target_name", ""),
                "pre_outcome_text": case.get("pre_outcome_text", "") or case.get("story_summary", "")[:200],
                "estimated_trigger": cat_id,
                "match_score": item["score"]
            })

    return sampled


def save_results(results: List[dict], output_path: str):
    """結果をJSONLファイルに保存"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            # match_scoreは出力から除外（内部情報）
            output_item = {k: v for k, v in item.items() if k != "match_score"}
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"\n出力完了: {output_path} ({len(results)}件)")


def display_sample_results(results: List[dict]):
    """サンプル結果を表示"""
    print("\n" + "=" * 80)
    print("層化サンプリング結果 (各カテゴリから8件)")
    print("=" * 80)

    # カテゴリごとにグループ化して表示
    by_category = {}
    for item in results:
        cat = item["estimated_trigger"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    for cat_id, cat_info in TRIGGER_CATEGORIES.items():
        items = by_category.get(cat_id, [])
        print(f"\n[{cat_id}] {cat_info['label']} ({len(items)}件)")
        print("-" * 60)
        for i, item in enumerate(items, 1):
            name = item["entity_name"][:20] if item["entity_name"] else "(不明)"
            text_preview = item["pre_outcome_text"][:50] + "..." if item["pre_outcome_text"] else ""
            print(f"  {i}. {name:22s} | {text_preview}")


def main():
    """メイン処理"""
    # パス設定
    base_path = Path(__file__).parent.parent.parent
    input_path = base_path / "data" / "raw" / "cases_with_pre_outcome.jsonl"
    output_path = base_path / "data" / "hexagram" / "pilot_64.jsonl"

    print(f"入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")

    # シード設定（再現性のため）
    random.seed(42)

    # 事例を読み込み
    print("\nデータ読み込み中...")
    cases = load_cases(str(input_path))
    print(f"読み込み完了: {len(cases)}件")

    # 層化サンプリング実行
    print("\n層化サンプリング実行中...")
    sampled = stratified_sample(cases, samples_per_category=8)

    # 結果保存
    save_results(sampled, str(output_path))

    # 結果表示
    display_sample_results(sampled)

    # 統計サマリー
    print("\n" + "=" * 60)
    print(f"抽出完了: 計{len(sampled)}件 (目標: 64件)")
    print("=" * 60)


if __name__ == '__main__':
    main()
