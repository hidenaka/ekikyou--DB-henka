#!/usr/bin/env python3
"""
動画制作用データエクスポートツール

易経DBから動画制作に必要なデータを抽出・整形して出力します。
動画環境での活用を想定したフォーマットで提供。
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from schema_v3 import Case

# 八卦の基本情報
HEXAGRAM_INFO = {
    "乾": {
        "symbol": "☰",
        "meaning": "天・創造・剛健・強さ",
        "keyword": "強さ",
        "color": "#FFD700",  # 金色
        "description": "創造的な力、リーダーシップ、強固な意志"
    },
    "坤": {
        "symbol": "☷",
        "meaning": "地・受容・柔順・基盤",
        "keyword": "基盤",
        "color": "#8B4513",  # 茶色
        "description": "受容性、地道な努力、安定した基盤"
    },
    "震": {
        "symbol": "☳",
        "meaning": "雷・動き・奮起・衝撃",
        "keyword": "衝撃",
        "color": "#FF4500",  # オレンジレッド
        "description": "突然の変化、奮起、行動の始まり"
    },
    "巽": {
        "symbol": "☴",
        "meaning": "風・浸透・柔軟・対話",
        "keyword": "対話",
        "color": "#87CEEB",  # スカイブルー
        "description": "柔軟性、浸透力、対話と調和"
    },
    "坎": {
        "symbol": "☵",
        "meaning": "水・危険・困難・試練",
        "keyword": "試練",
        "color": "#4169E1",  # ロイヤルブルー
        "description": "困難、試練、危機的状況"
    },
    "離": {
        "symbol": "☲",
        "meaning": "火・明知・分離・才能",
        "keyword": "才能",
        "color": "#FF6347",  # トマト
        "description": "明知、才能の発揮、光明"
    },
    "艮": {
        "symbol": "☶",
        "meaning": "山・止まる・待機・蓄積",
        "keyword": "待機",
        "color": "#708090",  # スレートグレー
        "description": "停止、蓄積、忍耐の時期"
    },
    "兌": {
        "symbol": "☱",
        "meaning": "沢・喜び・和悦・成功",
        "keyword": "喜び",
        "color": "#FFB6C1",  # ライトピンク
        "description": "喜び、成功、和やかな交流"
    }
}

def get_cases_by_pattern(db_path: Path, from_hex: str, to_hex: str) -> List[Case]:
    """
    特定の卦の変化パターンに該当する事例を取得

    Args:
        db_path: データベースのパス
        from_hex: 変化前の卦
        to_hex: 変化後の卦

    Returns:
        該当する事例のリスト
    """
    cases = []

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            case = Case(**data)

            # 3つのトランジションのいずれかに該当するか確認
            if ((case.before_hex.value == from_hex and case.trigger_hex.value == to_hex) or
                (case.trigger_hex.value == from_hex and case.action_hex.value == to_hex) or
                (case.action_hex.value == from_hex and case.after_hex.value == to_hex)):
                cases.append(case)

    return cases

def get_pattern_statistics(db_path: Path, from_hex: str, to_hex: str) -> Dict:
    """
    特定パターンの統計情報を取得

    Returns:
        {
            "total": 総件数,
            "success_rate": 成功率,
            "outcomes": {outcome: count},
            "top_examples": [事例リスト]
        }
    """
    cases = get_cases_by_pattern(db_path, from_hex, to_hex)

    if not cases:
        return {
            "total": 0,
            "success_rate": 0.0,
            "outcomes": {},
            "top_examples": []
        }

    # 結果の集計
    outcomes = defaultdict(int)
    for case in cases:
        outcomes[case.outcome.value] += 1

    # 成功率計算
    success_count = outcomes.get("Success", 0) + outcomes.get("PartialSuccess", 0) * 0.5
    success_rate = (success_count / len(cases)) * 100 if cases else 0.0

    # トップ事例（信頼性順、Success優先）
    top_examples = sorted(
        cases,
        key=lambda c: (
            c.outcome.value == "Success",
            c.credibility_rank.value == "S",
            c.credibility_rank.value == "A"
        ),
        reverse=True
    )[:5]

    return {
        "total": len(cases),
        "success_rate": round(success_rate, 1),
        "outcomes": dict(outcomes),
        "top_examples": [
            {
                "target_name": c.target_name,
                "scale": c.scale.value,
                "period": c.period,
                "story_summary": c.story_summary,
                "outcome": c.outcome.value,
                "credibility_rank": c.credibility_rank.value,
                "logic_memo": c.logic_memo
            }
            for c in top_examples
        ]
    }

def export_hexagram_reference(output_path: Path):
    """
    八卦の参照情報をJSONで出力（動画制作時に使用）
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(HEXAGRAM_INFO, f, ensure_ascii=False, indent=2)

    print(f"八卦参照情報を出力: {output_path}")

def export_pattern_guide(db_path: Path, output_path: Path):
    """
    主要な変化パターンのガイドを出力

    動画で使いやすいように、各パターンの意味・統計・事例をまとめる
    """
    # 主要パターン（成功・失敗の典型例）
    key_patterns = [
        # 成功パターン
        {"from": "坎", "to": "乾", "name": "危機から力へ", "type": "success"},
        {"from": "震", "to": "乾", "name": "混乱から強さへ", "type": "success"},
        {"from": "艮", "to": "乾", "name": "停滞から躍進へ", "type": "success"},
        {"from": "乾", "to": "離", "name": "力を才能へ", "type": "success"},
        {"from": "離", "to": "乾", "name": "才能が力に", "type": "success"},

        # 失敗パターン
        {"from": "乾", "to": "震", "name": "絶頂から転落", "type": "failure"},
        {"from": "震", "to": "離", "name": "混乱が続く", "type": "failure"},
        {"from": "離", "to": "坎", "name": "才能から危機へ", "type": "failure"},

        # 中立パターン
        {"from": "艮", "to": "艮", "name": "停滞の継続", "type": "neutral"},
        {"from": "坎", "to": "坎", "name": "危機の継続", "type": "neutral"},
        {"from": "離", "to": "離", "name": "才能の継続", "type": "neutral"},
    ]

    guide = {
        "patterns": []
    }

    for pattern in key_patterns:
        from_hex = pattern["from"]
        to_hex = pattern["to"]

        from_info = HEXAGRAM_INFO[from_hex]
        to_info = HEXAGRAM_INFO[to_hex]

        stats = get_pattern_statistics(db_path, from_hex, to_hex)

        guide["patterns"].append({
            "name": pattern["name"],
            "type": pattern["type"],
            "from_hexagram": {
                "name": from_hex,
                "symbol": from_info["symbol"],
                "meaning": from_info["meaning"],
                "keyword": from_info["keyword"]
            },
            "to_hexagram": {
                "name": to_hex,
                "symbol": to_info["symbol"],
                "meaning": to_info["meaning"],
                "keyword": to_info["keyword"]
            },
            "statistics": stats,
            "narration_template": f"{from_info['keyword']}から{to_info['keyword']}へ",
            "visual_suggestion": f"{from_info['symbol']} → {to_info['symbol']}",
            "color_scheme": {
                "from_color": from_info["color"],
                "to_color": to_info["color"]
            }
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(guide, f, ensure_ascii=False, indent=2)

    print(f"パターンガイドを出力: {output_path}")
    print(f"  主要パターン数: {len(guide['patterns'])}")

def export_timeline_template(output_path: Path):
    """
    タイムライン型動画用のテンプレート（YAML）を出力

    動画制作者が埋めやすい形式
    """
    template = """# タイムライン型動画用 易経マッピングテンプレート
#
# 使い方:
# 1. project情報を埋める
# 2. timeline の各時代に対して、該当する八卦を選択
# 3. transitions の logic を記述
# 4. scripts/export_narration.py で台本生成

project:
  theme: "[テーマ名を記入]"
  theme_id: "[3桁の数字]"
  scale: "other"  # individual/company/family/country/other

# 八卦の選択肢:
# 乾(☰): 天・創造・剛健・強さ
# 坤(☷): 地・受容・柔順・基盤
# 震(☳): 雷・動き・奮起・衝撃
# 巽(☴): 風・浸透・柔軟・対話
# 坎(☵): 水・危険・困難・試練
# 離(☲): 火・明知・分離・才能
# 艮(☶): 山・止まる・待機・蓄積
# 兌(☱): 沢・喜び・和悦・成功

timeline:
  # 時代1（Hook）
  - era: "Hook"
    year: "現在"
    hexagram: "坎"  # 例: 困難・危機
    state: "混乱・カオス"
    description: "[現在の状況を記述]"

  # 時代2（過去1）
  - era: "過去1"
    year: "[年代]"
    hexagram: "艮"  # 例: 停滞
    state: "停滞・閉塞"
    description: "[時代の特徴を記述]"

  # 時代3（過去2）
  - era: "過去2"
    year: "[年代]"
    hexagram: "兌"  # 例: 喜び・成功
    state: "変質・新生"
    description: "[時代の特徴を記述]"

  # 時代4（現在）
  - era: "現在"
    year: "[年代]"
    hexagram: "離"  # 例: 才能・個性
    state: "成長痛"
    description: "[時代の特徴を記述]"

  # 時代5（未来1）
  - era: "未来1"
    year: "[年代]"
    hexagram: "乾"  # 例: 強さ・効率
    state: "絶頂・慢心"
    description: "[予測される状況]"

  # 時代6（未来2）
  - era: "未来2"
    year: "[年代]"
    hexagram: "坤"  # 例: 基盤・原点回帰
    state: "変質・新生"
    description: "[予測される状況]"

transitions:
  # 過去1 → 過去2
  - from_era: "過去1"
    to_era: "過去2"
    from_hex: "艮"
    to_hex: "兌"
    trigger_type: "意図的決断"  # 外部ショック/内部崩壊/意図的決断/偶発・出会い
    changing_lines: [1, 2]
    logic: "[変化の理由を記述]"

  # 過去2 → 現在
  - from_era: "過去2"
    to_era: "現在"
    from_hex: "兌"
    to_hex: "離"
    trigger_type: "偶発・出会い"
    changing_lines: [2]
    logic: "[変化の理由を記述]"

  # 現在 → 未来1
  - from_era: "現在"
    to_era: "未来1"
    from_hex: "離"
    to_hex: "乾"
    trigger_type: "外部ショック"
    changing_lines: [1, 3]
    logic: "[変化の理由を記述]"
    pattern_warning: "失敗パターンの可能性"  # オプション

  # 未来1 → 未来2
  - from_era: "未来1"
    to_era: "未来2"
    from_hex: "乾"
    to_hex: "坤"
    trigger_type: "内部崩壊"
    changing_lines: [1, 2, 3]
    logic: "[変化の理由を記述]"
    pattern_type: "Shock_Recovery"  # オプション
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"タイムラインテンプレートを出力: {output_path}")

def main():
    """メイン処理"""
    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    output_dir = Path(__file__).parent.parent / "exports" / "video_production"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("動画制作用データエクスポート")
    print("=" * 80)

    # 1. 八卦参照情報
    export_hexagram_reference(output_dir / "hexagram_reference.json")

    # 2. パターンガイド（主要な変化パターン）
    export_pattern_guide(db_path, output_dir / "pattern_guide.json")

    # 3. タイムラインテンプレート
    export_timeline_template(output_dir / "timeline_template.yaml")

    print("\n" + "=" * 80)
    print("エクスポート完了")
    print("=" * 80)
    print(f"\n出力先: {output_dir}")
    print("\n次のステップ:")
    print("  1. timeline_template.yaml をコピーして新しいプロジェクトを作成")
    print("  2. hexagram_reference.json で八卦の意味を確認")
    print("  3. pattern_guide.json で類似パターンの事例を参照")

if __name__ == "__main__":
    main()
