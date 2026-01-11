#!/usr/bin/env python3
"""
64卦遷移マップSVG可視化スクリプト

transition_map.json を読み込み、64卦を8×8グリッドに配置し、
遷移を矢印で表現するSVGを生成する。

- グリッド配置: 上卦×下卦（八卦ごとにグループ化）
- 矢印の太さ: 遷移件数に比例
- 矢印の色: 成功率（緑=高, 赤=低）
- 10件以上の遷移のみ表示
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

# パス設定
BASE_DIR = Path(__file__).parent.parent
TRANSITION_MAP_FILE = BASE_DIR / "data" / "hexagrams" / "transition_map.json"
HEXAGRAM_MASTER_FILE = BASE_DIR / "data" / "hexagrams" / "hexagram_master.json"
OUTPUT_FILE = BASE_DIR / "assets" / "hexagram_map.svg"

# 八卦の順序（伏羲先天八卦の順序をベースに、見やすく調整）
TRIGRAM_ORDER = ["乾", "兌", "離", "震", "巽", "坎", "艮", "坤"]

# 八卦の色（背景色、淡い色）
TRIGRAM_COLORS = {
    "乾": "#fff5e6",  # 淡いオレンジ（天）
    "兌": "#e6f7ff",  # 淡い水色（沢）
    "離": "#fff0f0",  # 淡いピンク（火）
    "震": "#f0ffe6",  # 淡い黄緑（雷）
    "巽": "#e6ffe6",  # 淡い緑（風）
    "坎": "#e6e6ff",  # 淡い青（水）
    "艮": "#f5f5dc",  # ベージュ（山）
    "坤": "#f0f0f0",  # 淡いグレー（地）
}

# SVG設定
CELL_SIZE = 80  # セルサイズ
GRID_PADDING = 60  # グリッド周囲のパディング
LABEL_OFFSET = 25  # ラベルオフセット
MIN_COUNT = 10  # 表示する最小遷移件数


def load_transition_map() -> Dict:
    """遷移マップを読み込む"""
    with open(TRANSITION_MAP_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_hexagram_master() -> Dict:
    """64卦マスターを読み込み、卦名から位置へのマッピングを作成"""
    with open(HEXAGRAM_MASTER_FILE, 'r', encoding='utf-8') as f:
        master = json.load(f)

    # 卦名 → (行, 列) のマッピング
    name_to_pos = {}
    pos_to_name = {}

    for hex_id, data in master.items():
        name = data['name']
        upper = data['upper_trigram']
        lower = data['lower_trigram']

        # 行=上卦のインデックス, 列=下卦のインデックス
        if upper in TRIGRAM_ORDER and lower in TRIGRAM_ORDER:
            row = TRIGRAM_ORDER.index(upper)
            col = TRIGRAM_ORDER.index(lower)
            name_to_pos[name] = (row, col)
            pos_to_name[(row, col)] = name

    return name_to_pos, pos_to_name, master


def get_cell_center(row: int, col: int) -> Tuple[float, float]:
    """グリッドセルの中心座標を取得"""
    x = GRID_PADDING + LABEL_OFFSET + col * CELL_SIZE + CELL_SIZE / 2
    y = GRID_PADDING + LABEL_OFFSET + row * CELL_SIZE + CELL_SIZE / 2
    return x, y


def success_rate_to_color(rate: float) -> str:
    """成功率を色に変換（赤→黄→緑のグラデーション）"""
    # 0.0 = 赤, 0.5 = 黄, 1.0 = 緑
    if rate <= 0.5:
        # 赤→黄
        r = 220
        g = int(180 * (rate / 0.5))
        b = 50
    else:
        # 黄→緑
        r = int(220 - 170 * ((rate - 0.5) / 0.5))
        g = 180
        b = 50
    return f"rgb({r},{g},{b})"


def count_to_stroke_width(count: int, max_count: int) -> float:
    """件数を線の太さに変換"""
    # 最小1.5px、最大8px
    min_width = 1.5
    max_width = 8

    # 対数スケールで調整
    if count <= MIN_COUNT:
        return min_width

    log_count = math.log(count)
    log_max = math.log(max_count)
    log_min = math.log(MIN_COUNT)

    ratio = (log_count - log_min) / (log_max - log_min) if log_max > log_min else 0
    return min_width + ratio * (max_width - min_width)


def create_curved_arrow_path(
    x1: float, y1: float,
    x2: float, y2: float,
    curve_offset: float = 0
) -> str:
    """曲線矢印のSVGパスを生成"""
    # 自己ループの場合
    if abs(x1 - x2) < 1 and abs(y1 - y2) < 1:
        # セルの右上に小さなループを描く
        return f"M {x1+15},{y1-15} C {x1+40},{y1-40} {x1+40},{y1+10} {x1+15},{y1+15}"

    # 通常の曲線
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)

    # 制御点のオフセット（曲がり具合）
    # 距離に応じて曲がり具合を調整
    curve_factor = min(dist * 0.3, 50) + curve_offset * 15

    # 垂直方向のオフセット
    nx = -dy / dist * curve_factor
    ny = dx / dist * curve_factor

    # 制御点
    cx = (x1 + x2) / 2 + nx
    cy = (y1 + y2) / 2 + ny

    # 矢印の先端を少し手前で止める（セルに重ならないように）
    end_offset = 25
    end_dx = dx / dist * end_offset
    end_dy = dy / dist * end_offset

    return f"M {x1},{y1} Q {cx},{cy} {x2-end_dx},{y2-end_dy}"


def generate_svg(
    transitions: Dict,
    name_to_pos: Dict,
    pos_to_name: Dict,
    hexagram_master: Dict
) -> str:
    """SVGを生成"""

    # SVGサイズ計算
    svg_width = GRID_PADDING * 2 + LABEL_OFFSET + CELL_SIZE * 8 + 100  # 凡例用に+100
    svg_height = GRID_PADDING * 2 + LABEL_OFFSET + CELL_SIZE * 8 + 150  # 凡例用に+150

    # 最大件数を取得（線の太さ計算用）
    max_count = 0
    for from_hex, to_dict in transitions.items():
        for to_hex, stats in to_dict.items():
            if stats['count'] >= MIN_COUNT:
                max_count = max(max_count, stats['count'])

    # SVGヘッダー
    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        f'<defs>',
        f'  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        f'    <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>',
        f'  </marker>',
        f'  <marker id="arrowhead-success" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        f'    <polygon points="0 0, 10 3.5, 0 7" fill="#2a2"/>',
        f'  </marker>',
        f'  <marker id="arrowhead-fail" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        f'    <polygon points="0 0, 10 3.5, 0 7" fill="#c44"/>',
        f'  </marker>',
        f'</defs>',
        f'<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    # タイトル
    svg_parts.append(
        f'<text x="{svg_width/2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20" font-weight="bold" fill="#333">'
        f'64卦遷移マップ（10件以上の遷移）</text>'
    )

    # グリッド背景（八卦ごとに色分け）
    for row in range(8):
        for col in range(8):
            x = GRID_PADDING + LABEL_OFFSET + col * CELL_SIZE
            y = GRID_PADDING + LABEL_OFFSET + row * CELL_SIZE

            # 上卦の色を使用
            upper_trigram = TRIGRAM_ORDER[row]
            bg_color = TRIGRAM_COLORS[upper_trigram]

            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" '
                f'fill="{bg_color}" stroke="#ccc" stroke-width="0.5"/>'
            )

    # グリッド枠線
    grid_x = GRID_PADDING + LABEL_OFFSET
    grid_y = GRID_PADDING + LABEL_OFFSET
    grid_size = CELL_SIZE * 8
    svg_parts.append(
        f'<rect x="{grid_x}" y="{grid_y}" width="{grid_size}" height="{grid_size}" '
        f'fill="none" stroke="#999" stroke-width="2"/>'
    )

    # 八卦の区切り線（2×2のブロック）
    for i in range(1, 4):
        # 縦線
        x = grid_x + i * 2 * CELL_SIZE
        svg_parts.append(
            f'<line x1="{x}" y1="{grid_y}" x2="{x}" y2="{grid_y + grid_size}" '
            f'stroke="#888" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )
        # 横線
        y = grid_y + i * 2 * CELL_SIZE
        svg_parts.append(
            f'<line x1="{grid_x}" y1="{y}" x2="{grid_x + grid_size}" y2="{y}" '
            f'stroke="#888" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )

    # 軸ラベル（八卦名）
    for i, trigram in enumerate(TRIGRAM_ORDER):
        # 上軸（下卦）
        x = GRID_PADDING + LABEL_OFFSET + i * CELL_SIZE + CELL_SIZE / 2
        svg_parts.append(
            f'<text x="{x}" y="{GRID_PADDING + 15}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" fill="#333">{trigram}</text>'
        )

        # 左軸（上卦）
        y = GRID_PADDING + LABEL_OFFSET + i * CELL_SIZE + CELL_SIZE / 2 + 5
        svg_parts.append(
            f'<text x="{GRID_PADDING + 5}" y="{y}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" fill="#333">{trigram}</text>'
        )

    # 軸タイトル
    svg_parts.append(
        f'<text x="{grid_x + grid_size/2}" y="{GRID_PADDING - 5}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="#666">下卦</text>'
    )
    svg_parts.append(
        f'<text x="{GRID_PADDING - 15}" y="{grid_y + grid_size/2}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="#666" '
        f'transform="rotate(-90,{GRID_PADDING - 15},{grid_y + grid_size/2})">上卦</text>'
    )

    # 卦名をセルに表示
    for (row, col), name in pos_to_name.items():
        cx, cy = get_cell_center(row, col)
        # 卦名から漢字部分を抽出（例: "乾為天" → "乾"）
        short_name = name[0] if len(name) > 0 else name

        # 全事例数を計算
        total_cases = 0
        if name in transitions:
            for to_hex, stats in transitions[name].items():
                total_cases += stats['count']

        svg_parts.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16" font-weight="bold" fill="#444">{short_name}</text>'
        )

        # 事例数を小さく表示
        if total_cases > 0:
            svg_parts.append(
                f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="9" fill="#888">{total_cases}</text>'
            )

    # 遷移矢印の描画（件数10件以上）
    arrow_data = []
    for from_hex, to_dict in transitions.items():
        if from_hex not in name_to_pos:
            continue

        for to_hex, stats in to_dict.items():
            if to_hex not in name_to_pos:
                continue
            if stats['count'] < MIN_COUNT:
                continue

            from_pos = name_to_pos[from_hex]
            to_pos = name_to_pos[to_hex]

            arrow_data.append({
                'from_pos': from_pos,
                'to_pos': to_pos,
                'from_hex': from_hex,
                'to_hex': to_hex,
                'count': stats['count'],
                'success_rate': stats['success_rate']
            })

    # 件数が少ない順に描画（多いものが上に来るように）
    arrow_data.sort(key=lambda x: x['count'])

    # 同じ遷移ペアをグループ化（双方向の場合にずらすため）
    pair_counts = {}
    for arrow in arrow_data:
        key = tuple(sorted([arrow['from_pos'], arrow['to_pos']]))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_offsets = {}
    for arrow in arrow_data:
        from_pos = arrow['from_pos']
        to_pos = arrow['to_pos']
        key = tuple(sorted([from_pos, to_pos]))

        if key not in pair_offsets:
            pair_offsets[key] = 0

        offset = pair_offsets[key]
        pair_offsets[key] += 1

        x1, y1 = get_cell_center(*from_pos)
        x2, y2 = get_cell_center(*to_pos)

        color = success_rate_to_color(arrow['success_rate'])
        width = count_to_stroke_width(arrow['count'], max_count)

        # 曲線パス
        path = create_curved_arrow_path(x1, y1, x2, y2, offset)

        # 透明度は件数に応じて調整
        opacity = min(0.4 + 0.4 * (arrow['count'] / max_count), 0.8)

        svg_parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="{width:.1f}" stroke-opacity="{opacity:.2f}" '
            f'marker-end="url(#arrowhead)"/>'
        )

    # 凡例
    legend_x = GRID_PADDING + LABEL_OFFSET
    legend_y = GRID_PADDING + LABEL_OFFSET + grid_size + 30

    svg_parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" '
        f'font-size="14" font-weight="bold" fill="#333">凡例</text>'
    )

    # 成功率の色凡例
    svg_parts.append(
        f'<text x="{legend_x}" y="{legend_y + 25}" font-family="sans-serif" '
        f'font-size="11" fill="#666">成功率:</text>'
    )

    for i, (rate, label) in enumerate([(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]):
        x = legend_x + 60 + i * 80
        color = success_rate_to_color(rate)
        svg_parts.append(
            f'<line x1="{x}" y1="{legend_y + 20}" x2="{x + 40}" y2="{legend_y + 20}" '
            f'stroke="{color}" stroke-width="4"/>'
        )
        svg_parts.append(
            f'<text x="{x + 45}" y="{legend_y + 25}" font-family="sans-serif" '
            f'font-size="10" fill="#666">{label}</text>'
        )

    # 件数の太さ凡例
    svg_parts.append(
        f'<text x="{legend_x + 350}" y="{legend_y + 25}" font-family="sans-serif" '
        f'font-size="11" fill="#666">件数:</text>'
    )

    for i, (count, label) in enumerate([(10, "10件"), (100, "100件"), (500, "500件")]):
        x = legend_x + 400 + i * 90
        width = count_to_stroke_width(count, max_count)
        svg_parts.append(
            f'<line x1="{x}" y1="{legend_y + 20}" x2="{x + 40}" y2="{legend_y + 20}" '
            f'stroke="#888" stroke-width="{width:.1f}"/>'
        )
        svg_parts.append(
            f'<text x="{x + 45}" y="{legend_y + 25}" font-family="sans-serif" '
            f'font-size="10" fill="#666">{label}</text>'
        )

    # 統計情報
    stats_y = legend_y + 55
    total_arrows = len(arrow_data)
    svg_parts.append(
        f'<text x="{legend_x}" y="{stats_y}" font-family="sans-serif" '
        f'font-size="11" fill="#666">表示遷移数: {total_arrows}パターン（10件以上）</text>'
    )

    # 八卦の色凡例
    svg_parts.append(
        f'<text x="{legend_x}" y="{stats_y + 30}" font-family="sans-serif" '
        f'font-size="11" fill="#666">八卦（上卦）:</text>'
    )

    for i, trigram in enumerate(TRIGRAM_ORDER):
        x = legend_x + 90 + i * 80
        color = TRIGRAM_COLORS[trigram]
        svg_parts.append(
            f'<rect x="{x}" y="{stats_y + 18}" width="20" height="15" '
            f'fill="{color}" stroke="#999" stroke-width="0.5"/>'
        )
        svg_parts.append(
            f'<text x="{x + 25}" y="{stats_y + 30}" font-family="sans-serif" '
            f'font-size="10" fill="#666">{trigram}</text>'
        )

    # SVGフッター
    svg_parts.append('</svg>')

    return '\n'.join(svg_parts)


def main():
    print("Loading transition map...")
    transition_data = load_transition_map()
    transitions = transition_data['transitions']
    metadata = transition_data['metadata']

    print(f"Total cases: {metadata['total_cases']:,}")
    print(f"Transition patterns: {metadata['total_transition_patterns']}")

    print("Loading hexagram master...")
    name_to_pos, pos_to_name, hexagram_master = load_hexagram_master()
    print(f"Mapped {len(name_to_pos)} hexagrams to grid positions")

    print("Generating SVG...")
    svg_content = generate_svg(transitions, name_to_pos, pos_to_name, hexagram_master)

    # 出力ディレクトリ確認
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"\nSVG generated: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == '__main__':
    main()
