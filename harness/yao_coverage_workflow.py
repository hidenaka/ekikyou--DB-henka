#!/usr/bin/env python3
"""
卦×爻カバレッジワークフロー

特定の卦×爻に対して事例を調査・追加するためのハーネス。
MCPメモリからワークフロー状況を参照し、補充対象の表示や
バッチJSONテンプレートの生成を行う。

Usage:
    # Phase1の補充対象を確認
    python3 harness/yao_coverage_workflow.py --phase 1

    # 特定の卦×爻の詳細を確認
    python3 harness/yao_coverage_workflow.py --hex 27 --yao 1

    # バッチテンプレート生成
    python3 harness/yao_coverage_workflow.py --hex 27 --yao 1 --template
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
HEXAGRAMS_DIR = DATA_DIR / "hexagrams"
ANALYSIS_DIR = DATA_DIR / "analysis"
IMPORT_DIR = DATA_DIR / "import"


# ========== 爻段階のマッピング ==========
YAO_STAGES = {
    1: {
        "name": "発芽期・始動期",
        "description": "物事の始まり。潜む龍のように、まだ表に出る時ではない。",
        "keywords": ["初期", "始動", "潜伏", "準備", "萌芽"],
        "case_hints": {
            "success": ["創業初期の成功", "初動の正しい判断", "基盤作りの成功"],
            "failure": ["初動ミス", "準備不足", "性急な行動", "出発点の失敗"],
        },
    },
    2: {
        "name": "成長期・基盤確立期",
        "description": "基盤を固める時期。地に現れた龍のように、人々に認められ始める。",
        "keywords": ["成長", "基盤", "確立", "安定", "認知"],
        "case_hints": {
            "success": ["基盤確立の成功", "成長軌道への乗り", "信頼構築"],
            "failure": ["基盤の脆弱さ", "成長の停滞", "認知不足"],
        },
    },
    3: {
        "name": "転換期・岐路",
        "description": "転機の時期。上卦と下卦の境目で、困難と選択に直面する。",
        "keywords": ["転換", "岐路", "選択", "危機", "決断"],
        "case_hints": {
            "success": ["ピボット成功", "危機からの脱出", "正しい選択"],
            "failure": ["選択ミス", "転換失敗", "方向性の誤り"],
        },
    },
    4: {
        "name": "成熟期・接近期",
        "description": "上位への接近。淵から躍り出ようとする龍のように、次の段階を窺う。",
        "keywords": ["成熟", "接近", "跳躍準備", "野心", "展開"],
        "case_hints": {
            "success": ["事業拡大", "市場展開", "次のステージへ"],
            "failure": ["過信", "早すぎる拡大", "準備なき挑戦"],
        },
    },
    5: {
        "name": "全盛期・リーダー期",
        "description": "最も良い位置。天を飛ぶ龍のように、力を発揮する時期。",
        "keywords": ["全盛", "リーダー", "絶頂", "成功", "影響力"],
        "case_hints": {
            "success": ["業界リーダー", "大成功", "社会的影響"],
            "failure": ["絶頂からの転落", "慢心", "後継者問題"],
        },
    },
    6: {
        "name": "衰退期・転換期・極み",
        "description": "極みの位置。高ぶる龍のように、行き過ぎに悔いが生じる。",
        "keywords": ["衰退", "極み", "過剰", "終焉", "次への準備"],
        "case_hints": {
            "success": ["優雅な撤退", "次世代への橋渡し", "レガシー形成"],
            "failure": ["凋落", "過信の代償", "後継失敗", "消滅"],
        },
    },
}


def load_hexagram_master() -> Dict:
    """hexagram_master.jsonを読み込む"""
    path = HEXAGRAMS_DIR / "hexagram_master.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_yao_master() -> Dict:
    """yao_master.jsonを読み込む"""
    path = HEXAGRAMS_DIR / "yao_master.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_gap_analysis() -> Dict:
    """gap_analysis.jsonを読み込む"""
    path = ANALYSIS_DIR / "gap_analysis.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_hexagram_info(hex_id: int, hex_master: Dict) -> Dict:
    """卦情報を取得"""
    return hex_master.get(str(hex_id), {})


def get_yao_info(hex_id: int, yao_pos: int, yao_master: Dict) -> Dict:
    """爻情報を取得"""
    hex_data = yao_master.get(str(hex_id), {})
    yao_data = hex_data.get("yao", {}).get(str(yao_pos), {})
    return yao_data


def generate_case_hints(hex_info: Dict, yao_pos: int, yao_info: Dict) -> Dict:
    """その卦×爻に適した事例タイプのヒントを生成"""
    stage = YAO_STAGES.get(yao_pos, {})
    stage_name = stage.get("name", "不明")
    stage_hints = stage.get("case_hints", {})
    stage_keywords = stage.get("keywords", [])

    hex_keyword = hex_info.get("keyword", "")
    hex_meaning = hex_info.get("meaning", "")
    hex_interpretations = hex_info.get("interpretations", {})

    # 卦の解釈を組み合わせ
    hints = {
        "yao_stage": stage_name,
        "stage_description": stage.get("description", ""),
        "suitable_case_types": [],
        "search_keywords": [],
    }

    # 卦のキーワードと爻段階を組み合わせた事例タイプ
    if hex_keyword:
        keywords = [k.strip() for k in hex_keyword.split("・")]
        for kw in keywords:
            for sh in stage_hints.get("success", []):
                hints["suitable_case_types"].append(f"{kw}の{sh}")
            for fh in stage_hints.get("failure", []):
                hints["suitable_case_types"].append(f"{kw}における{fh}")

    # 検索キーワード候補
    for kw in stage_keywords:
        hints["search_keywords"].append(f"{hex_keyword} {kw}")

    # スケール別のヒント
    for scale, interp in hex_interpretations.items():
        if interp:
            hints["search_keywords"].append(f"{scale} {hex_keyword}")

    return hints


def generate_batch_template(
    hex_id: int,
    yao_pos: int,
    hex_info: Dict,
    yao_info: Dict,
    num_cases: int = 3
) -> List[Dict]:
    """バッチJSONテンプレートを生成"""
    hex_name = hex_info.get("name", f"卦{hex_id}")
    stage = YAO_STAGES.get(yao_pos, {})
    stage_name = stage.get("name", "")

    # 爻位に対応するbefore_stateの推定
    yao_to_state = {
        1: "成長痛",
        2: "安定・平和",
        3: "停滞・閉塞",
        4: "成長痛",
        5: "絶頂・慢心",
        6: "どん底・危機",
    }
    before_state = yao_to_state.get(yao_pos, "停滞・閉塞")

    template = []
    for i in range(num_cases):
        case = {
            "target_name": f"[事例名{i+1}]",
            "scale": "company",
            "period": "[YYYY-YYYY]",
            "story_summary": f"[{hex_name}第{yao_pos}爻（{stage_name}）の事例。具体的なストーリーを記述]",
            "before_state": before_state,
            "trigger_type": "外部ショック",
            "action_type": "攻める・挑戦",
            "after_state": "V字回復・大成功",
            "before_hex": hex_info.get("lower_trigram", "乾"),
            "trigger_hex": "震",
            "action_hex": hex_info.get("upper_trigram", "乾"),
            "after_hex": "兌",
            "pattern_type": "Shock_Recovery",
            "outcome": "Success",
            "free_tags": [hex_name, f"第{yao_pos}爻", stage_name],
            "source_type": "news",
            "credibility_rank": "B",
            "hexagram_id": hex_id,
            "hexagram_name": hex_name,
            "yao_context": f"{hex_name}の第{yao_pos}爻（{stage_name}）に該当する事例。[爻辞との関連を記述]",
            "main_domain": "[分野名]",
            "country": "[国名]",
            "sources": ["[URL1]", "[URL2]"],
        }
        template.append(case)

    return template


def display_hex_yao_detail(
    hex_id: int,
    yao_pos: int,
    hex_master: Dict,
    yao_master: Dict,
    gap_analysis: Dict,
    generate_template: bool = False,
) -> None:
    """卦×爻の詳細を表示"""
    hex_info = get_hexagram_info(hex_id, hex_master)
    yao_info = get_yao_info(hex_id, yao_pos, yao_master)

    if not hex_info:
        print(f"Error: 卦番号 {hex_id} が見つかりません")
        return

    hex_name = hex_info.get("name", f"卦{hex_id}")
    hex_keyword = hex_info.get("keyword", "")
    hex_meaning = hex_info.get("meaning", "")

    stage = YAO_STAGES.get(yao_pos, {})
    stage_name = stage.get("name", "不明")

    print()
    print("=" * 60)
    print(f"=== {hex_id}. {hex_name} 第{yao_pos}爻 ===")
    print("=" * 60)

    # 卦の基本情報
    print(f"\n【卦の意味】")
    print(f"  キーワード: {hex_keyword}")
    print(f"  意味: {hex_meaning}")

    # 爻辞
    print(f"\n【爻辞】")
    classic = yao_info.get("classic", "不明")
    modern = yao_info.get("modern", "不明")
    print(f"  原文: {classic}")
    print(f"  現代訳: {modern}")

    # 爻段階
    print(f"\n【爻段階】")
    print(f"  {stage_name}")
    print(f"  {stage.get('description', '')}")

    # 適合事例タイプ
    hints = generate_case_hints(hex_info, yao_pos, yao_info)
    print(f"\n【適合事例タイプ】")
    for case_type in hints["suitable_case_types"][:6]:
        print(f"  - {case_type}")

    # 検索キーワード候補
    print(f"\n【検索キーワード候補】")
    for kw in hints["search_keywords"][:6]:
        print(f"  - {kw}")

    # スケール別の解釈
    interps = hex_info.get("interpretations", {})
    if interps:
        print(f"\n【スケール別の解釈】")
        for scale, interp in interps.items():
            if interp:
                print(f"  [{scale}] {interp[:60]}...")

    # ギャップ分析からの情報
    missing = [
        m for m in gap_analysis.get("missing_hex_yao", [])
        if m["hexagram_id"] == hex_id and m["yao_position"] == yao_pos
    ]
    low_data = [
        l for l in gap_analysis.get("low_data_hex_yao", [])
        if l["hexagram_id"] == hex_id and l["yao_position"] == yao_pos
    ]

    print(f"\n【現在の状況】")
    if missing:
        print(f"  状態: データなし（優先度: {missing[0].get('priority', 'N/A')}）")
    elif low_data:
        print(f"  状態: データ不足（現在{low_data[0].get('count', 0)}件）")
    else:
        print(f"  状態: データあり")

    # テンプレート生成
    if generate_template:
        print(f"\n" + "=" * 60)
        print("【バッチJSONテンプレート】")
        print("=" * 60)
        template = generate_batch_template(hex_id, yao_pos, hex_info, yao_info)
        print(json.dumps(template, ensure_ascii=False, indent=2))

        # ファイル保存のヒント
        filename = f"batch_hex_{hex_id:02d}_yao_{yao_pos}.json"
        filepath = IMPORT_DIR / filename
        print(f"\n保存先: {filepath}")
        print(f"追加コマンド: python3 scripts/add_batch.py {filepath}")


def display_phase_targets(phase: int, gap_analysis: Dict, hex_master: Dict) -> None:
    """指定Phaseの補充対象を表示"""
    # Phaseの定義（優先度に基づく）
    phase_definitions = {
        1: {"min_priority": 30, "max_priority": 100, "description": "高優先度（priority >= 30）"},
        2: {"min_priority": 20, "max_priority": 29, "description": "中優先度（priority 20-29）"},
        3: {"min_priority": 10, "max_priority": 19, "description": "低優先度（priority 10-19）"},
        4: {"min_priority": 0, "max_priority": 9, "description": "追加データ（priority < 10）"},
    }

    phase_def = phase_definitions.get(phase)
    if not phase_def:
        print(f"Error: Phase {phase} は存在しません（1-4）")
        return

    print()
    print("=" * 60)
    print(f"=== Phase {phase}: {phase_def['description']} ===")
    print("=" * 60)

    # missing_hex_yaoからフィルタ
    missing = gap_analysis.get("missing_hex_yao", [])
    low_data = gap_analysis.get("low_data_hex_yao", [])

    # 優先度でフィルタ
    min_p = phase_def["min_priority"]
    max_p = phase_def["max_priority"]

    phase_missing = [
        m for m in missing
        if min_p <= m.get("priority", 0) <= max_p
    ]

    phase_low = [
        l for l in low_data
        if min_p <= l.get("priority", 0) <= max_p
    ]

    # 卦ごとにグループ化
    hex_groups: Dict[int, Dict] = {}
    for m in phase_missing:
        hex_id = m["hexagram_id"]
        if hex_id not in hex_groups:
            hex_info = get_hexagram_info(hex_id, hex_master)
            hex_groups[hex_id] = {
                "name": hex_info.get("name", f"卦{hex_id}"),
                "keyword": hex_info.get("keyword", ""),
                "missing_yao": [],
                "low_data_yao": [],
            }
        hex_groups[hex_id]["missing_yao"].append(m["yao_position"])

    for l in phase_low:
        hex_id = l["hexagram_id"]
        if hex_id not in hex_groups:
            hex_info = get_hexagram_info(hex_id, hex_master)
            hex_groups[hex_id] = {
                "name": hex_info.get("name", f"卦{hex_id}"),
                "keyword": hex_info.get("keyword", ""),
                "missing_yao": [],
                "low_data_yao": [],
            }
        hex_groups[hex_id]["low_data_yao"].append({
            "yao": l["yao_position"],
            "count": l.get("count", 0),
        })

    # 表示
    print(f"\n【補充対象: {len(hex_groups)}卦】")
    print()

    for hex_id in sorted(hex_groups.keys()):
        group = hex_groups[hex_id]
        print(f"  {hex_id}. {group['name']} ({group['keyword']})")
        if group["missing_yao"]:
            yao_str = ", ".join([f"第{y}爻" for y in sorted(group["missing_yao"])])
            print(f"      [データなし] {yao_str}")
        if group["low_data_yao"]:
            for ld in group["low_data_yao"]:
                print(f"      [データ不足] 第{ld['yao']}爻 ({ld['count']}件)")
        print()

    # サマリー
    total_missing = len(phase_missing)
    total_low = len(phase_low)
    print(f"\n【サマリー】")
    print(f"  データなし: {total_missing}件")
    print(f"  データ不足: {total_low}件")
    print(f"  合計補充対象: {total_missing + total_low}件")

    # コマンドヒント
    print(f"\n【次のステップ】")
    if hex_groups:
        first_hex = list(hex_groups.keys())[0]
        first_yao = (
            hex_groups[first_hex]["missing_yao"][0]
            if hex_groups[first_hex]["missing_yao"]
            else hex_groups[first_hex]["low_data_yao"][0]["yao"]
        )
        print(f"  詳細確認: python3 harness/yao_coverage_workflow.py --hex {first_hex} --yao {first_yao}")
        print(f"  テンプレート: python3 harness/yao_coverage_workflow.py --hex {first_hex} --yao {first_yao} --template")


def display_overview(gap_analysis: Dict, hex_master: Dict) -> None:
    """全体概要を表示"""
    print()
    print("=" * 60)
    print("=== 64卦×6爻 カバレッジ概要 ===")
    print("=" * 60)

    summary = gap_analysis.get("summary", {})
    print(f"\n【基本統計】")
    print(f"  総事例数: {gap_analysis.get('total_cases', 0)}")
    print(f"  卦ID付与数: {gap_analysis.get('cases_with_hexagram_id', 0)}")
    print(f"  カバー卦数: {summary.get('hexagrams_present', 0)}/64")
    print(f"  未カバー卦数: {summary.get('hexagrams_missing', 0)}")
    print(f"  データなし卦×爻: {len(gap_analysis.get('missing_hex_yao', []))}/384")
    print(f"  データ不足卦×爻: {len(gap_analysis.get('low_data_hex_yao', []))}")

    # Phase別サマリー
    missing = gap_analysis.get("missing_hex_yao", [])
    low_data = gap_analysis.get("low_data_hex_yao", [])

    phase_counts = {
        1: {"missing": 0, "low": 0},
        2: {"missing": 0, "low": 0},
        3: {"missing": 0, "low": 0},
        4: {"missing": 0, "low": 0},
    }

    for m in missing:
        p = m.get("priority", 0)
        if p >= 30:
            phase_counts[1]["missing"] += 1
        elif p >= 20:
            phase_counts[2]["missing"] += 1
        elif p >= 10:
            phase_counts[3]["missing"] += 1
        else:
            phase_counts[4]["missing"] += 1

    for l in low_data:
        p = l.get("priority", 0)
        if p >= 30:
            phase_counts[1]["low"] += 1
        elif p >= 20:
            phase_counts[2]["low"] += 1
        elif p >= 10:
            phase_counts[3]["low"] += 1
        else:
            phase_counts[4]["low"] += 1

    print(f"\n【Phase別補充対象】")
    for phase_num, counts in phase_counts.items():
        total = counts["missing"] + counts["low"]
        print(f"  Phase {phase_num}: {total}件 (データなし{counts['missing']} + 不足{counts['low']})")

    # 未カバー卦
    missing_hexagrams = gap_analysis.get("missing_hexagrams", [])
    if missing_hexagrams:
        print(f"\n【未カバー卦（全6爻データなし）】")
        for mh in missing_hexagrams[:10]:
            print(f"  {mh['hexagram_id']}. {mh['name']} ({mh['keyword']})")
        if len(missing_hexagrams) > 10:
            print(f"  ... 他{len(missing_hexagrams) - 10}卦")

    # コマンドヒント
    print(f"\n【次のステップ】")
    print(f"  Phase1確認: python3 harness/yao_coverage_workflow.py --phase 1")
    print(f"  特定卦詳細: python3 harness/yao_coverage_workflow.py --hex 27 --yao 1")


def main():
    parser = argparse.ArgumentParser(
        description="卦×爻カバレッジワークフロー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全体概要
  python3 harness/yao_coverage_workflow.py

  # Phase1の補充対象を確認
  python3 harness/yao_coverage_workflow.py --phase 1

  # 特定の卦×爻の詳細を確認
  python3 harness/yao_coverage_workflow.py --hex 27 --yao 1

  # バッチテンプレート生成
  python3 harness/yao_coverage_workflow.py --hex 27 --yao 1 --template
        """,
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="表示するPhase番号（1-4）",
    )
    parser.add_argument(
        "--hex",
        type=int,
        metavar="ID",
        help="卦番号（1-64）",
    )
    parser.add_argument(
        "--yao",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="爻位（1-6）",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="バッチJSONテンプレートを生成",
    )

    args = parser.parse_args()

    # データ読み込み
    hex_master = load_hexagram_master()
    yao_master = load_yao_master()
    gap_analysis = load_gap_analysis()

    if not hex_master:
        print("Error: hexagram_master.json が見つかりません")
        sys.exit(1)

    if not gap_analysis:
        print("Warning: gap_analysis.json が見つかりません。最新の分析を実行してください。")
        print("  python3 scripts/analyze_gaps.py")

    # コマンド分岐
    if args.hex is not None and args.yao is not None:
        # 特定の卦×爻の詳細
        display_hex_yao_detail(
            args.hex,
            args.yao,
            hex_master,
            yao_master,
            gap_analysis,
            generate_template=args.template,
        )
    elif args.phase is not None:
        # Phase別の補充対象
        display_phase_targets(args.phase, gap_analysis, hex_master)
    else:
        # 全体概要
        display_overview(gap_analysis, hex_master)


if __name__ == "__main__":
    main()
