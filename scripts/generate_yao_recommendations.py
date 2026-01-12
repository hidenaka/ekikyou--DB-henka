#!/usr/bin/env python3
"""
384爻の「推奨/避ける」リストを生成するスクリプト

プロンプトファイル: prompts/yao_recommendations_prompt.md に基づいて
yao_master.json から yao_recommendations.json を生成する
"""

import json
import os
from pathlib import Path

# プロジェクトルート
BASE_DIR = Path(__file__).parent.parent

# 爻の位置による共通パターン
LINE_PATTERNS = {
    1: {
        "stage": "始まり",
        "meaning": "基盤、準備、潜む",
        "general_recommend": ["準備を整える", "情報収集に努める", "基礎を固める", "慎重に観察する"],
        "general_avoid": ["早まって動く", "目立とうとする", "大きなリスクを取る", "急いで結論を出す"],
        "free_to_act": ["小さく始める機会を探す", "学習に時間を投資する"],
        "constrained": ["現状を維持しながら準備を進める", "内面的な成長に集中する"]
    },
    2: {
        "stage": "中位（内）",
        "meaning": "信頼構築、力を蓄える",
        "general_recommend": ["コツコツと着実に進める", "協力者を見つける", "信頼関係を築く", "内面を充実させる"],
        "general_avoid": ["焦って成果を求める", "孤立してしまう", "人間関係を軽視する"],
        "free_to_act": ["人脈を広げる活動に参加する", "チームワークを強化する"],
        "constrained": ["既存の関係を大切にする", "身近な人との絆を深める"]
    },
    3: {
        "stage": "過渡期",
        "meaning": "転換点、リスクあり",
        "general_recommend": ["慎重に判断する", "小さく試す", "リスクを見極める", "柔軟性を持つ"],
        "general_avoid": ["衝動的な決断をする", "油断する", "一か八かの賭けに出る"],
        "free_to_act": ["計画的な方向転換を検討する", "新しい選択肢を模索する"],
        "constrained": ["現状の安定を優先する", "無理のない範囲で変化を取り入れる"]
    },
    4: {
        "stage": "中位（外）",
        "meaning": "近くて遠い、慎重に",
        "general_recommend": ["上位者との関係を大切にする", "謙虚な姿勢を保つ", "礼儀を重んじる", "協調性を発揮する"],
        "general_avoid": ["出過ぎた振る舞いをする", "傲慢になる", "自己主張しすぎる"],
        "free_to_act": ["メンターや指導者を見つける", "学びの姿勢で関わる"],
        "constrained": ["組織内での役割を果たす", "求められた以上のことをしない"]
    },
    5: {
        "stage": "頂点",
        "meaning": "リーダー、責任、中正",
        "general_recommend": ["リーダーシップを発揮する", "決断を下す", "責任を持って行動する", "公正を心がける"],
        "general_avoid": ["優柔不断になる", "人任せにする", "責任を回避する"],
        "free_to_act": ["大きなプロジェクトを主導する", "ビジョンを示す"],
        "constrained": ["現在の立場で最善を尽くす", "影響力を適切に行使する"]
    },
    6: {
        "stage": "極まり",
        "meaning": "終わりの始まり、手放す",
        "general_recommend": ["次のフェーズへ準備する", "執着を手放す", "後進を育てる", "収束を意識する"],
        "general_avoid": ["しがみつく", "変化を恐れる", "過去の成功にこだわる"],
        "free_to_act": ["新しい始まりを計画する", "退き際を美しくする"],
        "constrained": ["穏やかに手を引く準備をする", "引き継ぎを丁寧に行う"]
    }
}

# 卦の性質グループ
HEXAGRAM_GROUPS = {
    "発展系": {
        "hexagrams": [1, 14, 25, 42, 46],  # 乾、大有、无妄、益、升
        "characteristic": "伸びる、拡大",
        "adjust": "積極的に動ける時。機会を逃さない",
        "caution": "調子に乗りすぎない"
    },
    "安定系": {
        "hexagrams": [2, 8, 11, 13, 15, 37],  # 坤、比、泰、同人、謙、家人
        "characteristic": "安定、協調",
        "adjust": "焦らず維持することが大切",
        "caution": "変化のタイミングを見逃さない"
    },
    "困難系": {
        "hexagrams": [3, 29, 39, 47, 36],  # 屯、坎、蹇、困、明夷
        "characteristic": "困難、試練",
        "adjust": "耐える、待つことが正解",
        "caution": "無理に打開しようとしない"
    },
    "変革系": {
        "hexagrams": [49, 50, 51, 52, 32],  # 革、鼎、震、艮、恒
        "characteristic": "変化、転換",
        "adjust": "変化を受け入れ、適応する",
        "caution": "変化を恐れて固執しない"
    },
    "停滞系": {
        "hexagrams": [12, 20, 23, 33],  # 否、観、剥、遯
        "characteristic": "後退、待機",
        "adjust": "無理に動かない、時を待つ",
        "caution": "焦って悪手を打たない"
    }
}

def get_hexagram_group(hexagram_id: int) -> dict:
    """卦グループを特定する"""
    for group_name, group_info in HEXAGRAM_GROUPS.items():
        if hexagram_id in group_info["hexagrams"]:
            return {
                "name": group_name,
                "characteristic": group_info["characteristic"],
                "adjust": group_info["adjust"],
                "caution": group_info["caution"]
            }
    # グループに属さない卦はバランス型として扱う
    return {
        "name": "バランス系",
        "characteristic": "状況に応じた対応",
        "adjust": "バランスを取りながら進む",
        "caution": "極端な行動を避ける"
    }

def generate_recommendations(hexagram_id: int, hexagram_name: str, line_position: int, yao_info: dict) -> dict:
    """各爻の推奨/避けるリストを生成"""
    
    line_pattern = LINE_PATTERNS[line_position]
    hexagram_group = get_hexagram_group(hexagram_id)
    
    # yao_idの生成 (例: 01_1, 64_6)
    yao_id = f"{hexagram_id:02d}_{line_position}"
    
    # 推奨アクションの構築
    recommendations = {
        "general": line_pattern["general_recommend"].copy(),
        "free_to_act": line_pattern["free_to_act"].copy(),
        "constrained": line_pattern["constrained"].copy()
    }
    
    # 卦グループに基づく調整を追加
    if hexagram_group["name"] != "バランス系":
        recommendations["general"].append(hexagram_group["adjust"])
    
    # 避けるべきことの構築
    avoid = {
        "general": line_pattern["general_avoid"].copy(),
        "reasons": [
            f"この爻は「{line_pattern['stage']}」の段階にある",
            f"爻辞の「{yao_info.get('modern', yao_info.get('classic', ''))}」が示すように、今は{line_pattern['meaning']}の時期"
        ]
    }
    
    # 卦グループに基づく注意点を追加
    avoid["general"].append(hexagram_group["caution"])
    
    return {
        "yao_id": yao_id,
        "hexagram_id": hexagram_id,
        "hexagram_name": hexagram_name,
        "line_position": line_position,
        "classic_text": yao_info.get("classic", ""),
        "modern_text": yao_info.get("modern", ""),
        "stage": line_pattern["stage"],
        "hexagram_group": hexagram_group["name"],
        "recommendations": recommendations,
        "avoid": avoid
    }

def main():
    # yao_master.json を読み込む
    input_path = BASE_DIR / "data" / "hexagrams" / "yao_master.json"
    output_path = BASE_DIR / "data" / "reference" / "yao_recommendations.json"
    
    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        yao_master = json.load(f)
    
    # 384爻の推奨リストを生成
    all_recommendations = []
    
    for hexagram_id_str, hexagram_data in yao_master.items():
        hexagram_id = int(hexagram_id_str)
        hexagram_name = hexagram_data["name"]
        yao_data = hexagram_data["yao"]
        
        for line_position_str, yao_info in yao_data.items():
            line_position = int(line_position_str)
            
            rec = generate_recommendations(
                hexagram_id=hexagram_id,
                hexagram_name=hexagram_name,
                line_position=line_position,
                yao_info=yao_info
            )
            all_recommendations.append(rec)
    
    # ソート（yao_id順）
    all_recommendations.sort(key=lambda x: (x["hexagram_id"], x["line_position"]))
    
    # JSON出力
    output_data = {
        "metadata": {
            "description": "384爻の推奨/避けるリスト",
            "generated_by": "generate_yao_recommendations.py",
            "total_count": len(all_recommendations)
        },
        "recommendations": all_recommendations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 生成完了: {output_path}")
    print(f"📊 総数: {len(all_recommendations)}爻")
    
    # 統計を表示
    groups = {}
    for rec in all_recommendations:
        group = rec["hexagram_group"]
        groups[group] = groups.get(group, 0) + 1
    
    print("\n📈 卦グループ別統計:")
    for group_name, count in sorted(groups.items()):
        print(f"  - {group_name}: {count}爻")

if __name__ == "__main__":
    main()
