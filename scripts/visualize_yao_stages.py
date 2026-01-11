#!/usr/bin/env python3
"""
各卦の6爻の階段図を生成するスクリプト

出力例（乾為天）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
乾為天 - 創造・リーダーシップ
━━━━━━━━━━━━━━━━━━━━━━━━━━━
6爻 |##........| 12% 亢龍有悔（昇りすぎて後悔）
5爻 |##########| 89% 飛龍在天（天に飛ぶ龍）
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse


def load_hexagram_master(data_dir: Path) -> dict:
    """卦情報を読み込む"""
    path = data_dir / "hexagrams" / "hexagram_master.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yao_master(data_dir: Path) -> dict:
    """爻の伝統的な解釈を読み込む"""
    path = data_dir / "hexagrams" / "yao_master.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(data_dir: Path) -> List[dict]:
    """cases.jsonlから事例データを読み込む"""
    cases = []
    path = data_dir / "raw" / "cases.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def calculate_yao_success_rates(cases: List[dict]) -> Dict[int, Dict[int, dict]]:
    """
    各卦×爻の成功率を計算

    Returns:
        {hexagram_id: {yao_position: {"success": int, "total": int, "rate": float}}}
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))

    for case in cases:
        # yao_analysisがある場合のみ
        yao = case.get("yao_analysis")
        if not yao:
            continue

        hex_id = yao.get("before_hexagram_id")
        yao_pos = yao.get("before_yao_position")

        if hex_id is None or yao_pos is None:
            continue

        # 結果を判定
        outcome = case.get("outcome", "")
        actual = yao.get("actual_outcome", "")

        # 成功判定
        is_success = outcome in ["Success", "Partial_Success"] or \
                     actual in ["Success", "Partial_Success"]

        stats[hex_id][yao_pos]["total"] += 1
        if is_success:
            stats[hex_id][yao_pos]["success"] += 1

    # 成功率を計算
    for hex_id in stats:
        for yao_pos in stats[hex_id]:
            data = stats[hex_id][yao_pos]
            if data["total"] > 0:
                data["rate"] = data["success"] / data["total"] * 100
            else:
                data["rate"] = 0.0

    return stats


def generate_bar(rate: float, width: int = 10) -> str:
    """成功率を棒グラフに変換"""
    filled = int(rate / 100 * width)
    empty = width - filled
    return "#" * filled + "." * empty


def generate_text_diagram(
    hex_id: int,
    hex_info: dict,
    yao_info: dict,
    stats: Dict[int, dict]
) -> str:
    """テキスト形式の階段図を生成"""
    lines = []

    # ヘッダー
    name = hex_info.get("name", f"第{hex_id}卦")
    keyword = hex_info.get("keyword", "")

    separator = "=" * 55
    lines.append(separator)
    lines.append(f"{name} - {keyword}")
    lines.append(separator)

    # 各爻（6爻から1爻の順で表示）
    yao_names = ["", "初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]

    for pos in range(6, 0, -1):
        yao_data = stats.get(pos, {"success": 0, "total": 0, "rate": 0.0})
        rate = yao_data.get("rate", 0.0)
        total = yao_data.get("total", 0)

        # 爻辞を取得
        yao_classic = yao_info.get("yao", {}).get(str(pos), {}).get("classic", "")
        yao_modern = yao_info.get("yao", {}).get(str(pos), {}).get("modern", "")

        # バーを生成
        bar = generate_bar(rate)

        # サンプル数に応じて表示を調整
        if total == 0:
            rate_str = " --"
            note = "(データなし)"
        else:
            rate_str = f"{rate:3.0f}"
            note = f"(n={total})"

        # 爻辞の表示（短縮）
        if yao_classic:
            phrase = f"{yao_classic}"
            if yao_modern and len(yao_classic) < 8:
                phrase += f"（{yao_modern}）"
        else:
            phrase = ""

        lines.append(f"{yao_names[pos]} |{bar}| {rate_str}% {phrase} {note}")

    lines.append(separator)

    # 統計サマリー
    total_cases = sum(s.get("total", 0) for s in stats.values())
    total_success = sum(s.get("success", 0) for s in stats.values())
    if total_cases > 0:
        overall_rate = total_success / total_cases * 100
        lines.append(f"総事例数: {total_cases}件, 全体成功率: {overall_rate:.1f}%")
    else:
        lines.append("総事例数: 0件")

    return "\n".join(lines)


def generate_svg_diagram(
    hex_id: int,
    hex_info: dict,
    yao_info: dict,
    stats: Dict[int, dict]
) -> str:
    """SVG形式の階段図を生成"""

    name = hex_info.get("name", f"第{hex_id}卦")
    keyword = hex_info.get("keyword", "")

    # SVGサイズ
    width = 700
    height = 350
    margin = 20
    bar_height = 35
    bar_max_width = 200

    yao_names = ["", "初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '  .title { font: bold 18px sans-serif; fill: #333; }',
        '  .subtitle { font: 14px sans-serif; fill: #666; }',
        '  .yao-label { font: 14px sans-serif; fill: #333; }',
        '  .rate { font: bold 14px monospace; fill: #333; }',
        '  .phrase { font: 12px sans-serif; fill: #555; }',
        '  .note { font: 11px sans-serif; fill: #888; }',
        '  .bar-bg { fill: #e0e0e0; }',
        '  .bar-fill { fill: #4CAF50; }',
        '  .bar-empty { fill: #f5f5f5; stroke: #ccc; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{margin}" y="30" class="title">{name}</text>',
        f'<text x="{margin}" y="50" class="subtitle">{keyword}</text>',
        f'<line x1="{margin}" y1="60" x2="{width - margin}" y2="60" stroke="#ccc"/>',
    ]

    # 各爻（6爻から1爻の順）
    y_start = 80
    for i, pos in enumerate(range(6, 0, -1)):
        y = y_start + i * bar_height

        yao_data = stats.get(pos, {"success": 0, "total": 0, "rate": 0.0})
        rate = yao_data.get("rate", 0.0)
        total = yao_data.get("total", 0)

        # 爻辞
        yao_classic = yao_info.get("yao", {}).get(str(pos), {}).get("classic", "")
        yao_modern = yao_info.get("yao", {}).get(str(pos), {}).get("modern", "")

        # 爻名
        svg_lines.append(f'<text x="{margin}" y="{y + 18}" class="yao-label">{yao_names[pos]}</text>')

        # バー背景
        bar_x = 70
        bar_y = y + 3
        svg_lines.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_max_width}" height="20" class="bar-empty" rx="3"/>')

        # バー（成功率に応じた幅）
        fill_width = int(rate / 100 * bar_max_width)
        if fill_width > 0:
            # 色を成功率に応じて変化
            if rate >= 70:
                color = "#4CAF50"  # 緑
            elif rate >= 50:
                color = "#8BC34A"  # 黄緑
            elif rate >= 30:
                color = "#FFC107"  # 黄
            else:
                color = "#FF9800"  # オレンジ
            svg_lines.append(f'<rect x="{bar_x}" y="{bar_y}" width="{fill_width}" height="20" fill="{color}" rx="3"/>')

        # 成功率表示
        rate_x = bar_x + bar_max_width + 10
        if total == 0:
            rate_text = "---%"
        else:
            rate_text = f"{rate:3.0f}%"
        svg_lines.append(f'<text x="{rate_x}" y="{y + 18}" class="rate">{rate_text}</text>')

        # 爻辞表示
        phrase_x = rate_x + 55
        phrase_text = yao_classic if yao_classic else ""
        if len(phrase_text) > 10:
            phrase_text = phrase_text[:10] + "..."
        svg_lines.append(f'<text x="{phrase_x}" y="{y + 18}" class="phrase">{phrase_text}</text>')

        # 現代訳（短縮）
        modern_x = phrase_x + 120
        modern_text = yao_modern if yao_modern else ""
        if len(modern_text) > 12:
            modern_text = modern_text[:12] + "..."
        svg_lines.append(f'<text x="{modern_x}" y="{y + 18}" class="note">({modern_text})</text>')

        # サンプル数
        note_x = width - margin - 50
        note_text = f"n={total}" if total > 0 else "no data"
        svg_lines.append(f'<text x="{note_x}" y="{y + 18}" class="note">{note_text}</text>')

    # フッター
    total_cases = sum(s.get("total", 0) for s in stats.values())
    total_success = sum(s.get("success", 0) for s in stats.values())
    if total_cases > 0:
        overall_rate = total_success / total_cases * 100
        footer = f"総事例数: {total_cases}件, 全体成功率: {overall_rate:.1f}%"
    else:
        footer = "総事例数: 0件"

    svg_lines.append(f'<line x1="{margin}" y1="{height - 40}" x2="{width - margin}" y2="{height - 40}" stroke="#ccc"/>')
    svg_lines.append(f'<text x="{margin}" y="{height - 15}" class="note">{footer}</text>')

    svg_lines.append('</svg>')

    return "\n".join(svg_lines)


def main():
    parser = argparse.ArgumentParser(description="各卦の6爻の階段図を生成")
    parser.add_argument("--hex", type=int, help="特定の卦のみ生成（1-64）")
    parser.add_argument("--format", choices=["text", "svg", "both"], default="both",
                        help="出力形式 (default: both)")
    parser.add_argument("--output", type=str, help="出力ディレクトリ")
    args = parser.parse_args()

    # パス設定
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_dir / "assets" / "yao_stages"

    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("Loading hexagram master...")
    hex_master = load_hexagram_master(data_dir)

    print("Loading yao master...")
    yao_master = load_yao_master(data_dir)

    print("Loading cases...")
    cases = load_cases(data_dir)
    print(f"  Loaded {len(cases)} cases")

    print("Calculating success rates...")
    all_stats = calculate_yao_success_rates(cases)

    # 生成対象の卦を決定
    if args.hex:
        hex_ids = [args.hex]
    else:
        hex_ids = range(1, 65)

    # 各卦の階段図を生成
    print(f"\nGenerating diagrams for {len(list(hex_ids))} hexagrams...")

    generated_count = 0
    for hex_id in hex_ids:
        hex_key = str(hex_id)

        if hex_key not in hex_master:
            print(f"  Warning: Hexagram {hex_id} not found in master")
            continue

        hex_info = hex_master[hex_key]
        yao_info = yao_master.get(hex_key, {"name": "", "yao": {}})
        stats = all_stats.get(hex_id, {})

        # ファイル名
        hex_name = hex_info.get("name", f"hex_{hex_id:02d}")
        safe_name = f"{hex_id:02d}_{hex_name.replace('/', '_')}"

        # テキスト形式
        if args.format in ["text", "both"]:
            text_diagram = generate_text_diagram(hex_id, hex_info, yao_info, stats)
            text_path = output_dir / f"{safe_name}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_diagram)

        # SVG形式
        if args.format in ["svg", "both"]:
            svg_diagram = generate_svg_diagram(hex_id, hex_info, yao_info, stats)
            svg_path = output_dir / f"{safe_name}.svg"
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_diagram)

        generated_count += 1

        # プレビュー（最初の3卦のみ）
        if generated_count <= 3:
            print(f"\n--- {hex_info.get('name', '')} ---")
            if args.format in ["text", "both"]:
                print(generate_text_diagram(hex_id, hex_info, yao_info, stats))

    print(f"\nGenerated {generated_count} hexagram diagrams")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
