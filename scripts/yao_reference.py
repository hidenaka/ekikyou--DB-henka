#!/usr/bin/env python3
"""
384爻リファレンスツール

指定した卦番号と爻位置から、詳細な推奨/避けるアドバイスを取得する。

使用例:
  python3 scripts/yao_reference.py 1 1      # 乾為天の初爻
  python3 scripts/yao_reference.py 3 3      # 水雷屯の三爻
  python3 scripts/yao_reference.py --all    # 全384爻を表示
  python3 scripts/yao_reference.py --hexagram 15  # 地山謙の全6爻
"""

import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RECOMMENDATIONS_FILE = BASE_DIR / "data" / "reference" / "yao_recommendations.json"


def load_recommendations() -> dict:
    """384爻データを読み込み、yao_idをキーにした辞書で返す"""
    with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    result = {}
    for rec in data.get("recommendations", []):
        yao_id = rec.get("yao_id", "")
        if yao_id:
            result[yao_id] = rec
    return result


def format_yao_detail(rec: dict) -> str:
    """1つの爻データを読みやすく整形"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"【{rec.get('hexagram_name', '')}】 第{rec.get('line_position', '')}爻")
    lines.append(f"卦ID: {rec.get('hexagram_id', '')}  爻ID: {rec.get('yao_id', '')}")
    lines.append(f"卦グループ: {rec.get('hexagram_group', '')}")
    lines.append(f"段階: {rec.get('stage', '')}")
    lines.append("-" * 60)
    
    # 爻辞
    lines.append(f"【爻辞】")
    lines.append(f"  古典: {rec.get('classic_text', '')}")
    lines.append(f"  現代: {rec.get('modern_text', '')}")
    sns_style = rec.get('sns_style', '')
    if sns_style:
        lines.append(f"  SNS風: {sns_style}")
    lines.append("")
    
    # 推奨アクション
    recommendations = rec.get("recommendations", {})
    lines.append("【推奨アクション】")
    lines.append("  ▼ 一般:")
    for action in recommendations.get("general", []):
        lines.append(f"    • {action}")
    lines.append("  ▼ 自由に動ける人:")
    for action in recommendations.get("free_to_act", []):
        lines.append(f"    • {action}")
    lines.append("  ▼ 制約がある人:")
    for action in recommendations.get("constrained", []):
        lines.append(f"    • {action}")
    lines.append("")
    
    # 避けるべきこと
    avoid = rec.get("avoid", {})
    lines.append("【避けるべきこと】")
    for action in avoid.get("general", []):
        lines.append(f"    ✕ {action}")
    lines.append("")
    lines.append("  理由:")
    for reason in avoid.get("reasons", []):
        lines.append(f"    → {reason}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def get_yao(hexagram_id: int, line_position: int, data: dict) -> dict:
    """指定した卦と爻位置のデータを取得"""
    yao_id = f"{hexagram_id:02d}_{line_position}"
    return data.get(yao_id, {})


def main():
    parser = argparse.ArgumentParser(description="384爻リファレンスツール")
    parser.add_argument("hexagram", type=int, nargs="?", help="卦番号 (1-64)")
    parser.add_argument("line", type=int, nargs="?", help="爻位置 (1-6)")
    parser.add_argument("--all", action="store_true", help="全384爻を表示")
    parser.add_argument("--hexagram-all", type=int, metavar="N", help="指定卦の全6爻を表示")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    parser.add_argument("--group", type=str, help="卦グループで絞り込み (発展系/安定系/困難系/変革系/停滞系/バランス系)")
    
    args = parser.parse_args()
    
    data = load_recommendations()
    print(f"📚 384爻データを読み込みました ({len(data)}件)\n")
    
    if args.all:
        # 全384爻を表示
        for yao_id in sorted(data.keys()):
            print(format_yao_detail(data[yao_id]))
            print("")
    
    elif args.hexagram_all:
        # 特定の卦の全6爻を表示
        print(f"=== 卦 {args.hexagram_all} の全6爻 ===\n")
        for line in range(1, 7):
            rec = get_yao(args.hexagram_all, line, data)
            if rec:
                print(format_yao_detail(rec))
                print("")
            else:
                print(f"[卦{args.hexagram_all} 第{line}爻: データなし]")
    
    elif args.group:
        # 卦グループで絞り込み
        print(f"=== 卦グループ: {args.group} ===\n")
        count = 0
        for yao_id in sorted(data.keys()):
            rec = data[yao_id]
            if rec.get("hexagram_group") == args.group:
                print(format_yao_detail(rec))
                print("")
                count += 1
        print(f"\n合計: {count}件")
    
    elif args.hexagram and args.line:
        # 特定の1爻を表示
        rec = get_yao(args.hexagram, args.line, data)
        if rec:
            if args.json:
                print(json.dumps(rec, ensure_ascii=False, indent=2))
            else:
                print(format_yao_detail(rec))
        else:
            print(f"❌ 卦{args.hexagram} 第{args.line}爻のデータが見つかりません")
    
    else:
        # ヘルプを表示
        print("使用例:")
        print("  python3 scripts/yao_reference.py 1 1        # 乾為天の初爻")
        print("  python3 scripts/yao_reference.py 3 3        # 水雷屯の三爻")
        print("  python3 scripts/yao_reference.py --hexagram-all 15  # 地山謙の全6爻")
        print("  python3 scripts/yao_reference.py --group 困難系     # 困難系の爻のみ")
        print("  python3 scripts/yao_reference.py --all      # 全384爻")
        print("")
        print("卦グループ: 発展系, 安定系, 困難系, 変革系, 停滞系, バランス系")


if __name__ == "__main__":
    main()
