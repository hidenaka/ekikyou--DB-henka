#!/usr/bin/env python3
"""
分割されたSNSフレーズJSONを統合してyao_master.jsonとyao_recommendations.jsonを更新
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "data" / "reference"
YAO_MASTER = BASE_DIR / "data" / "hexagrams" / "yao_master.json"
YAO_RECOMMENDATIONS = BASE_DIR / "data" / "reference" / "yao_recommendations.json"


def load_all_sns_phrases() -> dict:
    """分割されたSNSフレーズを統合"""
    all_phrases = {}
    
    phrase_files = [
        "sns_phrases_01_16.json",
        "sns_phrases_17_32.json",
        "sns_phrases_33_47.json",
        "sns_phrases_48_64.json",
    ]
    
    for filename in phrase_files:
        filepath = REFERENCE_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                phrases = json.load(f)
            all_phrases.update(phrases)
            print(f"  - {filename}: {len(phrases)}件")
    
    return all_phrases


def update_yao_master(phrases: dict):
    """yao_master.jsonを更新"""
    with open(YAO_MASTER, "r", encoding="utf-8") as f:
        yao_master = json.load(f)
    
    updated = 0
    for hex_id_str, hex_data in yao_master.items():
        for line_str, yao_data in hex_data.get("yao", {}).items():
            key = f"{hex_id_str}_{line_str}"
            if key in phrases:
                yao_data["sns_style"] = phrases[key]
                updated += 1
    
    with open(YAO_MASTER, "w", encoding="utf-8") as f:
        json.dump(yao_master, f, ensure_ascii=False, indent=2)
    
    print(f"  → yao_master.json: {updated}件更新")
    return yao_master


def update_yao_recommendations(phrases: dict):
    """yao_recommendations.jsonを更新"""
    with open(YAO_RECOMMENDATIONS, "r", encoding="utf-8") as f:
        recs_data = json.load(f)
    
    updated = 0
    for rec in recs_data.get("recommendations", []):
        hex_id = rec.get("hexagram_id")
        line = rec.get("line_position")
        if hex_id and line:
            key = f"{hex_id}_{line}"
            if key in phrases:
                rec["sns_style"] = phrases[key]
                updated += 1
    
    with open(YAO_RECOMMENDATIONS, "w", encoding="utf-8") as f:
        json.dump(recs_data, f, ensure_ascii=False, indent=2)
    
    print(f"  → yao_recommendations.json: {updated}件更新")


def main():
    print("=== 64卦SNSフレーズ統合 ===\n")
    
    print("1. 分割ファイルを読み込み...")
    phrases = load_all_sns_phrases()
    print(f"   合計: {len(phrases)}件\n")
    
    print("2. yao_master.json を更新...")
    update_yao_master(phrases)
    
    print("\n3. yao_recommendations.json を更新...")
    update_yao_recommendations(phrases)
    
    print("\n✅ 完了！")
    
    # サンプル表示
    print("\n=== サンプル ===")
    samples = ["1_1", "15_3", "29_3", "49_5", "64_6"]
    for key in samples:
        print(f"{key}: {phrases.get(key, 'なし')}")


if __name__ == "__main__":
    main()
