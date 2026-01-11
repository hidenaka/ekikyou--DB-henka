#!/usr/bin/env python3
"""
64卦×6爻の成功率ヒートマップ生成ツール

cases.jsonl から64卦×6爻の成功率を計算し、
インタラクティブなHTML版と静的なSVG版のヒートマップを生成します。
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import html

# 八卦の順序（伏羲八卦の順）
TRIGRAM_ORDER = ["乾", "兌", "離", "震", "巽", "坎", "艮", "坤"]

# 八卦の読みと意味
TRIGRAM_INFO = {
    "乾": {"reading": "けん", "meaning": "天・創造", "color": "#1a237e"},
    "兌": {"reading": "だ", "meaning": "沢・喜悦", "color": "#00838f"},
    "離": {"reading": "り", "meaning": "火・明晰", "color": "#c62828"},
    "震": {"reading": "しん", "meaning": "雷・動き", "color": "#558b2f"},
    "巽": {"reading": "そん", "meaning": "風・浸透", "color": "#6a1b9a"},
    "坎": {"reading": "かん", "meaning": "水・危険", "color": "#0d47a1"},
    "艮": {"reading": "ごん", "meaning": "山・停止", "color": "#4e342e"},
    "坤": {"reading": "こん", "meaning": "地・受容", "color": "#37474f"},
}

def load_hexagram_master(data_dir: Path) -> Dict:
    """64卦マスターデータを読み込み"""
    master_path = data_dir / "hexagrams" / "hexagram_master.json"
    if master_path.exists():
        with open(master_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_hexagram_trigrams(hexagram_id: int, master: Dict) -> Tuple[str, str]:
    """卦番号から上卦・下卦を取得"""
    hex_data = master.get(str(hexagram_id), {})
    upper = hex_data.get("upper_trigram", "")
    lower = hex_data.get("lower_trigram", "")
    return upper, lower

def analyze_hexagram_yao_success(db_path: Path) -> Dict:
    """
    64卦×6爻ごとの成功率を分析

    Returns:
        {hexagram_id: {yao_position: {"success": n, "total": n, "cases": [...]}}}
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "partial": 0, "failure": 0, "mixed": 0, "total": 0, "cases": []}))

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # hexagram_id と yao_analysis から爻位置を取得
            # yao_analysis.before_hexagram_id を優先、なければトップレベルの hexagram_id
            yao_analysis = data.get("yao_analysis", {})
            hexagram_id = None
            yao_position = None

            if yao_analysis:
                hexagram_id = yao_analysis.get("before_hexagram_id")
                yao_position = yao_analysis.get("before_yao_position")

            # フォールバック: トップレベルの hexagram_id
            if not hexagram_id:
                hexagram_id = data.get("hexagram_id")

            # outcome を取得
            outcome = data.get("outcome", "")
            target_name = data.get("target_name", "")

            if hexagram_id and yao_position and 1 <= yao_position <= 6:
                stat = stats[hexagram_id][yao_position]
                stat["total"] += 1

                if outcome == "Success":
                    stat["success"] += 1
                elif outcome == "PartialSuccess":
                    stat["partial"] += 1
                elif outcome == "Failure":
                    stat["failure"] += 1
                elif outcome == "Mixed":
                    stat["mixed"] += 1

                stat["cases"].append({
                    "name": target_name,
                    "outcome": outcome,
                    "period": data.get("period", ""),
                    "pattern": data.get("pattern_type", ""),
                })

    return stats

def calculate_success_rate(stat: Dict) -> Optional[float]:
    """成功率を計算（成功+部分成功を成功とみなす）"""
    if stat["total"] == 0:
        return None
    success_count = stat["success"] + stat["partial"]
    return success_count / stat["total"]

def get_color_for_rate(rate: Optional[float]) -> str:
    """成功率に応じた色を返す（緑→黄→赤のグラデーション）"""
    if rate is None:
        return "#e0e0e0"  # データなし：グレー

    # 成功率に基づいて色を決定
    if rate >= 0.7:
        # 高成功率：緑
        intensity = (rate - 0.7) / 0.3
        r = int(76 - 30 * intensity)
        g = int(175 + 30 * intensity)
        b = int(80 - 30 * intensity)
    elif rate >= 0.4:
        # 中間：黄色
        intensity = (rate - 0.4) / 0.3
        r = int(255 - 179 * intensity)
        g = int(235 - 60 * intensity)
        b = int(59 + 21 * intensity)
    else:
        # 低成功率：赤
        intensity = rate / 0.4
        r = int(244 + 11 * intensity)
        g = int(67 + 168 * intensity)
        b = int(54 + 5 * intensity)

    return f"#{r:02x}{g:02x}{b:02x}"

def generate_html_heatmap(stats: Dict, master: Dict, output_path: Path):
    """インタラクティブなHTMLヒートマップを生成"""

    # 64卦を八卦グループ順にソート
    hexagram_groups = defaultdict(list)
    for hex_id in range(1, 65):
        upper, lower = get_hexagram_trigrams(hex_id, master)
        if upper:
            hexagram_groups[upper].append(hex_id)

    html_content = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>64卦×6爻 成功率ヒートマップ</title>
<style>
:root {
    --bg-color: #1a1a2e;
    --text-color: #eaeaea;
    --card-bg: #16213e;
    --border-color: #0f3460;
}
body {
    font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif;
    background: var(--bg-color);
    color: var(--text-color);
    margin: 0;
    padding: 20px;
}
h1 {
    text-align: center;
    margin-bottom: 10px;
    color: #e94560;
}
.subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 30px;
}
.legend {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
}
.legend-color {
    width: 20px;
    height: 20px;
    border-radius: 3px;
}
.gradient-bar {
    display: flex;
    justify-content: center;
    margin-bottom: 30px;
}
.gradient-container {
    display: flex;
    align-items: center;
    gap: 10px;
}
.gradient {
    width: 300px;
    height: 20px;
    background: linear-gradient(to right, #f44336, #ffeb3b, #4caf50);
    border-radius: 3px;
}
.heatmap-container {
    overflow-x: auto;
    margin-bottom: 40px;
}
table {
    border-collapse: collapse;
    margin: 0 auto;
    background: var(--card-bg);
}
th, td {
    border: 1px solid var(--border-color);
    padding: 0;
    text-align: center;
}
th {
    background: var(--card-bg);
    padding: 8px 12px;
    font-weight: bold;
}
.yao-header {
    background: #0f3460;
}
.group-header {
    background: #e94560;
    color: white;
    padding: 10px;
    font-size: 1.1em;
}
.hexagram-name {
    text-align: left;
    padding: 5px 10px;
    font-size: 0.85em;
    white-space: nowrap;
    background: var(--card-bg);
    min-width: 120px;
}
.cell {
    width: 50px;
    height: 40px;
    cursor: pointer;
    position: relative;
    transition: transform 0.2s, box-shadow 0.2s;
}
.cell:hover {
    transform: scale(1.1);
    box-shadow: 0 0 10px rgba(233, 69, 96, 0.5);
    z-index: 10;
}
.cell-content {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75em;
    font-weight: bold;
}
.tooltip {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #16213e;
    border: 1px solid #e94560;
    border-radius: 8px;
    padding: 12px;
    min-width: 250px;
    z-index: 1000;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.cell:hover .tooltip {
    display: block;
}
.tooltip h4 {
    margin: 0 0 8px 0;
    color: #e94560;
    border-bottom: 1px solid #0f3460;
    padding-bottom: 5px;
}
.tooltip-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    font-size: 0.9em;
}
.tooltip-cases {
    margin-top: 10px;
    font-size: 0.8em;
    max-height: 100px;
    overflow-y: auto;
}
.tooltip-case {
    padding: 3px 0;
    border-bottom: 1px solid #0f3460;
}
.success { color: #4caf50; }
.partial { color: #8bc34a; }
.failure { color: #f44336; }
.mixed { color: #ff9800; }
.no-data {
    color: #666;
    font-style: italic;
}
.summary {
    background: var(--card-bg);
    padding: 20px;
    border-radius: 10px;
    margin-top: 30px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}
.summary h3 {
    color: #e94560;
    margin-top: 0;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}
.summary-item {
    background: #0f3460;
    padding: 15px;
    border-radius: 8px;
}
.summary-value {
    font-size: 2em;
    font-weight: bold;
    color: #e94560;
}
</style>
</head>
<body>
<h1>64卦 x 6爻 成功率ヒートマップ</h1>
<p class="subtitle">易経64卦の各爻における成功率を可視化</p>

<div class="gradient-bar">
    <div class="gradient-container">
        <span>0%（失敗）</span>
        <div class="gradient"></div>
        <span>100%（成功）</span>
    </div>
</div>

<div class="legend">
    <div class="legend-item">
        <div class="legend-color" style="background: #4caf50;"></div>
        <span>成功率 70%+</span>
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #ffeb3b;"></div>
        <span>成功率 40-70%</span>
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #f44336;"></div>
        <span>成功率 40%未満</span>
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #e0e0e0;"></div>
        <span>データなし</span>
    </div>
</div>

<div class="heatmap-container">
<table>
<thead>
<tr>
<th>卦</th>
<th class="yao-header">初爻</th>
<th class="yao-header">二爻</th>
<th class="yao-header">三爻</th>
<th class="yao-header">四爻</th>
<th class="yao-header">五爻</th>
<th class="yao-header">上爻</th>
</tr>
</thead>
<tbody>
"""

    total_cases = 0
    total_success = 0
    cells_with_data = 0
    best_cells = []
    worst_cells = []

    # 八卦グループ順に出力
    for trigram in TRIGRAM_ORDER:
        trigram_info = TRIGRAM_INFO[trigram]
        hex_ids = hexagram_groups.get(trigram, [])

        if not hex_ids:
            continue

        # グループヘッダー
        html_content += f'<tr><td colspan="7" class="group-header">{trigram}（{trigram_info["reading"]}）- {trigram_info["meaning"]}</td></tr>\n'

        for hex_id in sorted(hex_ids):
            hex_data = master.get(str(hex_id), {})
            hex_name = hex_data.get("name", f"第{hex_id}卦")
            hex_keyword = hex_data.get("keyword", "")

            html_content += f'<tr>\n'
            html_content += f'<td class="hexagram-name">{hex_id}. {hex_name}</td>\n'

            for yao in range(1, 7):
                stat = stats.get(hex_id, {}).get(yao, {"success": 0, "partial": 0, "failure": 0, "mixed": 0, "total": 0, "cases": []})
                rate = calculate_success_rate(stat)
                color = get_color_for_rate(rate)

                total_cases += stat["total"]
                total_success += stat["success"] + stat["partial"]

                if stat["total"] > 0:
                    cells_with_data += 1
                    cell_info = {
                        "hex_id": hex_id,
                        "hex_name": hex_name,
                        "yao": yao,
                        "rate": rate,
                        "total": stat["total"]
                    }
                    best_cells.append(cell_info)
                    worst_cells.append(cell_info)

                yao_names = {1: "初爻", 2: "二爻", 3: "三爻", 4: "四爻", 5: "五爻", 6: "上爻"}

                # セル内容
                if stat["total"] > 0:
                    rate_pct = int(rate * 100) if rate is not None else 0
                    cell_text = f"{rate_pct}%"
                else:
                    cell_text = "-"

                # ツールチップ内容
                tooltip_html = f"""
                <div class="tooltip">
                    <h4>{hex_name} - {yao_names[yao]}</h4>
                    <div class="tooltip-stats">
                        <div>成功率:</div><div><strong>{int(rate*100) if rate else 0}%</strong></div>
                        <div class="success">成功:</div><div>{stat['success']}件</div>
                        <div class="partial">部分成功:</div><div>{stat['partial']}件</div>
                        <div class="failure">失敗:</div><div>{stat['failure']}件</div>
                        <div class="mixed">混合:</div><div>{stat['mixed']}件</div>
                        <div>合計:</div><div><strong>{stat['total']}件</strong></div>
                    </div>
                """

                if stat["cases"]:
                    tooltip_html += '<div class="tooltip-cases"><strong>事例:</strong>'
                    for case in stat["cases"][:5]:  # 最大5件表示
                        outcome_class = case["outcome"].lower().replace("partialsuccess", "partial")
                        tooltip_html += f'<div class="tooltip-case"><span class="{outcome_class}">[{case["outcome"]}]</span> {html.escape(case["name"])}</div>'
                    if len(stat["cases"]) > 5:
                        tooltip_html += f'<div class="tooltip-case">...他{len(stat["cases"])-5}件</div>'
                    tooltip_html += '</div>'
                else:
                    tooltip_html += '<div class="no-data">データなし</div>'

                tooltip_html += '</div>'

                html_content += f'''<td class="cell" style="background-color: {color};">
                    <div class="cell-content">{cell_text}</div>
                    {tooltip_html}
                </td>\n'''

            html_content += '</tr>\n'

    # 統計サマリー
    overall_rate = (total_success / total_cases * 100) if total_cases > 0 else 0

    # ベスト・ワースト
    best_cells = sorted([c for c in best_cells if c["total"] >= 3], key=lambda x: -x["rate"])[:5]
    worst_cells = sorted([c for c in worst_cells if c["total"] >= 3], key=lambda x: x["rate"])[:5]

    html_content += f"""
</tbody>
</table>
</div>

<div class="summary">
    <h3>統計サマリー</h3>
    <div class="summary-grid">
        <div class="summary-item">
            <div>総事例数</div>
            <div class="summary-value">{total_cases:,}</div>
        </div>
        <div class="summary-item">
            <div>データあり</div>
            <div class="summary-value">{cells_with_data}/384</div>
        </div>
        <div class="summary-item">
            <div>全体成功率</div>
            <div class="summary-value">{overall_rate:.1f}%</div>
        </div>
    </div>

    <h4 style="margin-top: 20px;">成功率トップ5（3件以上）</h4>
    <ul>
"""

    for c in best_cells:
        yao_names = {1: "初爻", 2: "二爻", 3: "三爻", 4: "四爻", 5: "五爻", 6: "上爻"}
        html_content += f'<li class="success">{c["hex_name"]} {yao_names[c["yao"]]}: {int(c["rate"]*100)}% ({c["total"]}件)</li>\n'

    html_content += """
    </ul>

    <h4>成功率ワースト5（3件以上）</h4>
    <ul>
"""

    for c in worst_cells:
        yao_names = {1: "初爻", 2: "二爻", 3: "三爻", 4: "四爻", 5: "五爻", 6: "上爻"}
        html_content += f'<li class="failure">{c["hex_name"]} {yao_names[c["yao"]]}: {int(c["rate"]*100)}% ({c["total"]}件)</li>\n'

    html_content += """
    </ul>
</div>

<footer style="text-align: center; margin-top: 40px; color: #666;">
    <p>易経変化ロジックDB - Generated by visualize_heatmap.py</p>
</footer>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML heatmap generated: {output_path}")
    return {
        "total_cases": total_cases,
        "cells_with_data": cells_with_data,
        "overall_rate": overall_rate,
    }

def generate_svg_heatmap(stats: Dict, master: Dict, output_path: Path):
    """静的なSVGヒートマップを生成"""

    # SVGサイズ設定
    cell_width = 50
    cell_height = 30
    name_width = 150
    header_height = 40
    group_header_height = 25
    margin = 20

    # 八卦グループ順に整理
    hexagram_groups = defaultdict(list)
    for hex_id in range(1, 65):
        upper, lower = get_hexagram_trigrams(hex_id, master)
        if upper:
            hexagram_groups[upper].append(hex_id)

    # 高さ計算
    total_rows = 64 + len(TRIGRAM_ORDER)  # 64卦 + 8グループヘッダー
    svg_height = margin * 2 + header_height + total_rows * cell_height + 100  # 凡例用
    svg_width = margin * 2 + name_width + 6 * cell_width

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">
<defs>
    <linearGradient id="legendGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" style="stop-color:#f44336"/>
        <stop offset="50%" style="stop-color:#ffeb3b"/>
        <stop offset="100%" style="stop-color:#4caf50"/>
    </linearGradient>
    <style>
        .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; }}
        .header {{ font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #333; }}
        .group-header {{ font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; fill: white; }}
        .hex-name {{ font-family: Arial, sans-serif; font-size: 9px; fill: #333; }}
        .cell-text {{ font-family: Arial, sans-serif; font-size: 8px; fill: #333; text-anchor: middle; }}
        .legend-text {{ font-family: Arial, sans-serif; font-size: 10px; fill: #666; }}
    </style>
</defs>

<!-- Background -->
<rect width="100%" height="100%" fill="white"/>

<!-- Title -->
<text x="{svg_width/2}" y="25" text-anchor="middle" class="title">64卦 x 6爻 成功率ヒートマップ</text>

<!-- Headers -->
<text x="{margin + name_width/2}" y="{margin + header_height - 10}" text-anchor="middle" class="header">卦名</text>
'''

    yao_names = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    for i, name in enumerate(yao_names):
        x = margin + name_width + i * cell_width + cell_width / 2
        svg_content += f'<text x="{x}" y="{margin + header_height - 10}" text-anchor="middle" class="header">{name}</text>\n'

    current_y = margin + header_height

    # 八卦グループ順に描画
    for trigram in TRIGRAM_ORDER:
        trigram_info = TRIGRAM_INFO[trigram]
        hex_ids = hexagram_groups.get(trigram, [])

        if not hex_ids:
            continue

        # グループヘッダー
        svg_content += f'''<rect x="{margin}" y="{current_y}" width="{name_width + 6 * cell_width}" height="{group_header_height}" fill="{trigram_info['color']}"/>
<text x="{margin + 10}" y="{current_y + group_header_height - 7}" class="group-header">{trigram}（{trigram_info['reading']}）- {trigram_info['meaning']}</text>
'''
        current_y += group_header_height

        for hex_id in sorted(hex_ids):
            hex_data = master.get(str(hex_id), {})
            hex_name = hex_data.get("name", f"第{hex_id}卦")

            # 卦名
            svg_content += f'<text x="{margin + 5}" y="{current_y + cell_height - 10}" class="hex-name">{hex_id}. {hex_name}</text>\n'

            # 各爻のセル
            for yao in range(1, 7):
                stat = stats.get(hex_id, {}).get(yao, {"success": 0, "partial": 0, "failure": 0, "mixed": 0, "total": 0})
                rate = calculate_success_rate(stat)
                color = get_color_for_rate(rate)

                x = margin + name_width + (yao - 1) * cell_width

                svg_content += f'<rect x="{x}" y="{current_y}" width="{cell_width}" height="{cell_height}" fill="{color}" stroke="#ddd" stroke-width="0.5"/>\n'

                if stat["total"] > 0:
                    rate_pct = int(rate * 100) if rate is not None else 0
                    text_color = "#000" if rate and rate > 0.3 else "#fff"
                    svg_content += f'<text x="{x + cell_width/2}" y="{current_y + cell_height/2 + 3}" class="cell-text" fill="{text_color}">{rate_pct}%</text>\n'

            current_y += cell_height

    # 凡例
    legend_y = current_y + 20
    svg_content += f'''
<!-- Legend -->
<rect x="{margin}" y="{legend_y}" width="200" height="15" fill="url(#legendGrad)" stroke="#ccc"/>
<text x="{margin}" y="{legend_y + 30}" class="legend-text">0% (失敗)</text>
<text x="{margin + 170}" y="{legend_y + 30}" class="legend-text">100% (成功)</text>
<rect x="{margin + 250}" y="{legend_y}" width="15" height="15" fill="#e0e0e0" stroke="#ccc"/>
<text x="{margin + 270}" y="{legend_y + 12}" class="legend-text">データなし</text>
'''

    svg_content += '</svg>'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"SVG heatmap generated: {output_path}")

def main():
    """メイン処理"""
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "raw" / "cases.jsonl"
    assets_dir = base_dir / "assets"

    # assetsディレクトリがなければ作成
    assets_dir.mkdir(exist_ok=True)

    print("Loading hexagram master data...")
    master = load_hexagram_master(base_dir / "data")

    print("Analyzing hexagram x yao success rates...")
    stats = analyze_hexagram_yao_success(db_path)

    print("\nGenerating HTML heatmap...")
    html_path = assets_dir / "hexagram_heatmap.html"
    summary = generate_html_heatmap(stats, master, html_path)

    print("\nGenerating SVG heatmap...")
    svg_path = assets_dir / "hexagram_heatmap.svg"
    generate_svg_heatmap(stats, master, svg_path)

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total cases analyzed: {summary['total_cases']:,}")
    print(f"  Cells with data: {summary['cells_with_data']}/384 ({summary['cells_with_data']/384*100:.1f}%)")
    print(f"  Overall success rate: {summary['overall_rate']:.1f}%")
    print("=" * 50)
    print(f"\nOutput files:")
    print(f"  HTML (interactive): {html_path}")
    print(f"  SVG (static): {svg_path}")

if __name__ == "__main__":
    main()
