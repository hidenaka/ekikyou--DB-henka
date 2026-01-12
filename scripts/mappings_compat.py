#!/usr/bin/env python3
"""
互換レイヤー: 新384爻データから既存6爻形式を生成

新しい data/reference/yao_recommendations.json から
既存の mappings/yao_recommendations.json 形式を生成する。
"""

import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent

# 入出力パス
NEW_RECOMMENDATIONS = BASE_DIR / "data" / "reference" / "yao_recommendations.json"
OLD_MAPPINGS_BACKUP = BASE_DIR / "data" / "mappings" / "yao_recommendations_backup.json"
OUTPUT_MAPPINGS = BASE_DIR / "data" / "mappings" / "yao_recommendations.json"


def load_new_recommendations():
    """新しい384爻推奨データを読み込む"""
    with open(NEW_RECOMMENDATIONS, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("recommendations", [])


def aggregate_by_line_position(recommendations: list) -> dict:
    """
    384爻データを爻位置（1-6）でグループ化し、
    共通パターンを抽出する
    """
    grouped = defaultdict(list)
    
    for rec in recommendations:
        line_pos = rec["line_position"]
        grouped[line_pos].append(rec)
    
    return dict(grouped)


def extract_common_actions(recs_list: list, field: str, sub_field: str = "general") -> list:
    """
    複数の推奨データから共通アクションを抽出
    出現頻度の高い順に上位5件を返す
    """
    action_count = defaultdict(int)
    
    for rec in recs_list:
        field_data = rec.get(field, {})
        if isinstance(field_data, dict):
            actions = field_data.get(sub_field, [])
        else:
            actions = []
        
        for action in actions:
            action_count[action] += 1
    
    # 頻度順にソートし、上位5件を返す
    sorted_actions = sorted(action_count.items(), key=lambda x: -x[1])
    return [action for action, count in sorted_actions[:5]]


def generate_compat_format(grouped: dict) -> dict:
    """
    既存の mappings/yao_recommendations.json 形式で出力
    """
    # 既存形式のテンプレート（旧データから転用）
    stage_names = {
        1: "発芽期・始動期",
        2: "成長期・基盤確立期",
        3: "転換期・岐路",
        4: "成熟期・接近期",
        5: "全盛期・リーダー期",
        6: "衰退期・転換期・極み"
    }
    
    basic_stances = {
        1: "待機・潜伏",
        2: "着実・中庸",
        3: "慎重・熟考",
        4: "謙虚・調整",
        5: "決断・実行",
        6: "撤退・譲渡"
    }
    
    lifecycle_corporate = {
        1: "スタートアップ・構想段階",
        2: "成長軌道・第二創業期",
        3: "事業転換点・M&A検討期",
        4: "拡大期・業界上位進出",
        5: "業界リーダー・最盛期",
        6: "成熟産業・次の一手・事業承継"
    }
    
    lifecycle_individual = {
        1: "新人・見習い・修業中",
        2: "中堅・実務者・専門性確立",
        3: "管理職候補・キャリア転換点",
        4: "管理職・専門家・次期リーダー",
        5: "経営層・第一人者・トップ",
        6: "引退期・次世代育成・セカンドキャリア"
    }
    
    yao_names = {
        1: "初爻",
        2: "二爻",
        3: "三爻",
        4: "四爻",
        5: "五爻",
        6: "上爻"
    }
    
    result = {}
    
    for line_pos in range(1, 7):
        recs_list = grouped.get(line_pos, [])
        
        if recs_list:
            # 384爻データから共通アクションを抽出
            recommended_actions = extract_common_actions(recs_list, "recommendations", "general")
            avoid_actions = extract_common_actions(recs_list, "avoid", "general")
            
            # 代表的な stage を取得
            sample_rec = recs_list[0]
            stage = sample_rec.get("stage", stage_names[line_pos])
        else:
            # データがない場合はデフォルト値
            recommended_actions = []
            avoid_actions = []
            stage = stage_names[line_pos]
        
        result[str(line_pos)] = {
            "name": yao_names[line_pos],
            "stage": stage,
            "basic_stance": basic_stances[line_pos],
            "recommended_actions": recommended_actions[:4],  # 既存形式は4件
            "avoid_actions": avoid_actions[:3],  # 既存形式は3件
            "success_condition": get_success_condition(line_pos),
            "failure_pattern": get_failure_pattern(line_pos),
            "lifecycle_corporate": lifecycle_corporate[line_pos],
            "lifecycle_individual": lifecycle_individual[line_pos]
        }
    
    return result


def get_success_condition(line_pos: int) -> str:
    """爻位置に応じた成功条件"""
    conditions = {
        1: "時機を待ち、力を蓄えること",
        2: "地道な努力で実績を積むこと",
        3: "分岐点を正しく認識し、適切な選択をすること",
        4: "適切な関係者と連携し、機を待つこと",
        5: "最適なタイミングで決断し、リーダーシップを発揮すること",
        6: "適切なタイミングで引く、または次世代に託すこと"
    }
    return conditions.get(line_pos, "")


def get_failure_pattern(line_pos: int) -> str:
    """爻位置に応じた失敗パターン"""
    patterns = {
        1: "焦って動き、準備不足で失敗",
        2: "基盤が固まる前に拡大を急ぐ",
        3: "岐路での判断を誤る、または決断を先送り",
        4: "独走して孤立、または過度な慎重さで機を逃す",
        5: "機会を逃す、または決断を下せない",
        6: "行き過ぎて孤立、または引き際を誤る"
    }
    return patterns.get(line_pos, "")


def main():
    print("=== 互換レイヤー生成 ===\n")
    
    # 1. 新データ読み込み
    print("1. 新384爻データを読み込み...")
    new_recs = load_new_recommendations()
    print(f"   - {len(new_recs)} 件の爻データを読み込み")
    
    # 2. 爻位置でグループ化
    print("\n2. 爻位置でグループ化...")
    grouped = aggregate_by_line_position(new_recs)
    for pos in range(1, 7):
        print(f"   - {pos}爻: {len(grouped.get(pos, []))} 件")
    
    # 3. 既存ファイルをバックアップ
    print("\n3. 既存ファイルをバックアップ...")
    if OUTPUT_MAPPINGS.exists():
        with open(OUTPUT_MAPPINGS, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        with open(OLD_MAPPINGS_BACKUP, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False, indent=2)
        print(f"   - バックアップ: {OLD_MAPPINGS_BACKUP}")
    
    # 4. 互換形式を生成
    print("\n4. 互換形式を生成...")
    compat_data = generate_compat_format(grouped)
    
    # 5. 出力
    print("\n5. 出力...")
    with open(OUTPUT_MAPPINGS, "w", encoding="utf-8") as f:
        json.dump(compat_data, f, ensure_ascii=False, indent=2)
    print(f"   - 出力: {OUTPUT_MAPPINGS}")
    
    # 6. 検証
    print("\n6. 検証...")
    for pos in range(1, 7):
        yao_data = compat_data.get(str(pos), {})
        print(f"   {pos}爻: stage={yao_data.get('stage', 'N/A')}, "
              f"推奨={len(yao_data.get('recommended_actions', []))}件, "
              f"避ける={len(yao_data.get('avoid_actions', []))}件")
    
    print("\n✅ 互換レイヤー生成完了")


if __name__ == "__main__":
    main()
