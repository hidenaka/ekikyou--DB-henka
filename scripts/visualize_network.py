#!/usr/bin/env python3
"""
64卦遷移ネットワーク可視化スクリプト
D3.jsを使用したインタラクティブなHTMLファイルを生成
"""

import json
import os
from pathlib import Path

# パス設定
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

def load_data():
    """遷移マップと64卦マスターデータを読み込み"""
    with open(DATA_DIR / "hexagrams" / "transition_map.json", "r", encoding="utf-8") as f:
        transition_map = json.load(f)

    with open(DATA_DIR / "hexagrams" / "hexagram_master.json", "r", encoding="utf-8") as f:
        hexagram_master = json.load(f)

    return transition_map, hexagram_master

def get_trigram_color(trigram):
    """八卦ごとの色を返す"""
    colors = {
        "乾": "#FFD700",  # 金色 - 天・創造
        "坤": "#8B4513",  # 茶色 - 地・受容
        "震": "#FF4500",  # オレンジレッド - 雷・動き
        "巽": "#32CD32",  # ライムグリーン - 風・浸透
        "坎": "#1E90FF",  # ドジャーブルー - 水・危険
        "離": "#FF6347",  # トマト - 火・明るさ
        "艮": "#708090",  # スレートグレー - 山・停止
        "兌": "#FF69B4",  # ホットピンク - 沢・喜び
    }
    return colors.get(trigram, "#999999")

def get_success_rate_color(rate):
    """成功率に基づく色を返す（赤→黄→緑）"""
    if rate >= 0.8:
        return "#2ECC71"  # 緑
    elif rate >= 0.5:
        return "#F1C40F"  # 黄
    elif rate >= 0.2:
        return "#E67E22"  # オレンジ
    else:
        return "#E74C3C"  # 赤

def prepare_network_data(transition_map, hexagram_master):
    """D3.js用のノードとリンクデータを準備"""
    nodes = []
    links = []

    # 64卦すべてをノードとして追加
    for hex_id, hex_data in hexagram_master.items():
        upper_trigram = hex_data.get("upper_trigram", "")
        nodes.append({
            "id": hex_data["name"],
            "hexId": int(hex_id),
            "chinese": hex_data.get("chinese", ""),
            "reading": hex_data.get("reading", ""),
            "keyword": hex_data.get("keyword", ""),
            "meaning": hex_data.get("meaning", ""),
            "upperTrigram": upper_trigram,
            "lowerTrigram": hex_data.get("lower_trigram", ""),
            "color": get_trigram_color(upper_trigram),
            "totalFrom": 0,
            "totalTo": 0,
            "successRate": 0
        })

    # 遷移データからリンクを作成
    transitions = transition_map.get("transitions", {})

    # ノード名からインデックスへのマッピング
    node_index = {n["id"]: i for i, n in enumerate(nodes)}

    # 各卦の統計を計算
    from_counts = {}
    to_counts = {}
    success_counts = {}
    total_counts = {}

    for from_hex, to_hexes in transitions.items():
        if from_hex not in from_counts:
            from_counts[from_hex] = 0
            success_counts[from_hex] = 0
            total_counts[from_hex] = 0

        for to_hex, data in to_hexes.items():
            count = data.get("count", 0)
            success_rate = data.get("success_rate", 0)

            from_counts[from_hex] += count
            success_counts[from_hex] += count * success_rate
            total_counts[from_hex] += count

            if to_hex not in to_counts:
                to_counts[to_hex] = 0
            to_counts[to_hex] += count

            # リンクを追加（件数が1以上のもの）
            if count >= 1 and from_hex in node_index and to_hex in node_index:
                links.append({
                    "source": from_hex,
                    "target": to_hex,
                    "count": count,
                    "successRate": success_rate,
                    "mainAction": data.get("main_action", ""),
                    "actions": data.get("actions", {}),
                    "color": get_success_rate_color(success_rate)
                })

    # ノードの統計を更新
    for node in nodes:
        name = node["id"]
        node["totalFrom"] = from_counts.get(name, 0)
        node["totalTo"] = to_counts.get(name, 0)
        if total_counts.get(name, 0) > 0:
            node["successRate"] = round(success_counts.get(name, 0) / total_counts.get(name, 1), 3)

    return nodes, links

def generate_html(nodes, links, metadata):
    """D3.jsを使用したHTML生成"""

    # JSONデータをエスケープ
    nodes_json = json.dumps(nodes, ensure_ascii=False, indent=2)
    links_json = json.dumps(links, ensure_ascii=False, indent=2)

    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>64卦遷移ネットワーク</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
            overflow: hidden;
        }}

        #container {{
            display: flex;
            height: 100vh;
        }}

        #sidebar {{
            width: 320px;
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}

        #main {{
            flex: 1;
            position: relative;
        }}

        h1 {{
            font-size: 1.4em;
            margin-bottom: 10px;
            color: #00d4ff;
        }}

        h2 {{
            font-size: 1.1em;
            margin: 15px 0 10px 0;
            color: #ffd700;
            border-bottom: 1px solid rgba(255, 215, 0, 0.3);
            padding-bottom: 5px;
        }}

        .stats {{
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 15px;
        }}

        .filter-group {{
            margin-bottom: 15px;
        }}

        .filter-group label {{
            display: block;
            margin-bottom: 5px;
            font-size: 0.9em;
            color: #ccc;
        }}

        .filter-group input[type="range"] {{
            width: 100%;
            margin-bottom: 5px;
        }}

        .filter-group select {{
            width: 100%;
            padding: 8px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            border-radius: 4px;
        }}

        .filter-value {{
            text-align: right;
            font-size: 0.85em;
            color: #00d4ff;
        }}

        .legend {{
            margin-top: 20px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 0.85em;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        #detail-panel {{
            background: rgba(0, 0, 0, 0.5);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }}

        #detail-panel.active {{
            display: block;
        }}

        #detail-panel h3 {{
            font-size: 1.2em;
            color: #ffd700;
            margin-bottom: 10px;
        }}

        #detail-panel .hex-info {{
            font-size: 0.9em;
            line-height: 1.6;
        }}

        #detail-panel .hex-info dt {{
            color: #aaa;
            margin-top: 8px;
        }}

        #detail-panel .hex-info dd {{
            color: #fff;
            margin-left: 10px;
        }}

        .transitions-list {{
            max-height: 200px;
            overflow-y: auto;
            margin-top: 10px;
        }}

        .transition-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.85em;
        }}

        .transition-item:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}

        .success-badge {{
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
        }}

        .success-high {{ background: #2ecc71; color: #000; }}
        .success-mid {{ background: #f1c40f; color: #000; }}
        .success-low {{ background: #e74c3c; color: #fff; }}

        #network {{
            width: 100%;
            height: 100%;
        }}

        .node {{
            cursor: pointer;
        }}

        .node circle {{
            stroke: rgba(255, 255, 255, 0.5);
            stroke-width: 2px;
            transition: all 0.3s ease;
        }}

        .node:hover circle {{
            stroke: #fff;
            stroke-width: 3px;
            filter: brightness(1.3);
        }}

        .node.selected circle {{
            stroke: #00d4ff;
            stroke-width: 4px;
        }}

        .node text {{
            font-size: 10px;
            fill: #fff;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
        }}

        .link {{
            fill: none;
            opacity: 0.4;
            transition: opacity 0.3s ease;
        }}

        .link:hover {{
            opacity: 0.8;
        }}

        .link.highlighted {{
            opacity: 0.9;
            stroke-width: 3px !important;
        }}

        .link.dimmed {{
            opacity: 0.1;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            padding: 10px;
            font-size: 0.85em;
            pointer-events: none;
            z-index: 1000;
            max-width: 250px;
        }}

        .tooltip h4 {{
            color: #ffd700;
            margin-bottom: 5px;
        }}

        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
        }}

        .controls button {{
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }}

        .controls button:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}

        #layout-mode {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
        }}

        #layout-mode button {{
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
        }}

        #layout-mode button.active {{
            background: rgba(0, 212, 255, 0.3);
            border-color: #00d4ff;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>64卦遷移ネットワーク</h1>
            <div class="stats">
                総事例数: {metadata.get('total_cases', 0):,}件<br>
                遷移パターン: {metadata.get('total_transition_patterns', 0)}種類
            </div>

            <h2>フィルター</h2>

            <div class="filter-group">
                <label>最小成功率: <span id="success-rate-value">0%</span></label>
                <input type="range" id="success-rate-filter" min="0" max="100" value="0">
            </div>

            <div class="filter-group">
                <label>最小件数: <span id="count-value">1</span></label>
                <input type="range" id="count-filter" min="1" max="100" value="1">
            </div>

            <div class="filter-group">
                <label>八卦グループ</label>
                <select id="trigram-filter">
                    <option value="all">すべて表示</option>
                    <option value="乾">乾（天）- 創造</option>
                    <option value="坤">坤（地）- 受容</option>
                    <option value="震">震（雷）- 動き</option>
                    <option value="巽">巽（風）- 浸透</option>
                    <option value="坎">坎（水）- 危険</option>
                    <option value="離">離（火）- 明るさ</option>
                    <option value="艮">艮（山）- 停止</option>
                    <option value="兌">兌（沢）- 喜び</option>
                </select>
            </div>

            <h2>八卦カラー凡例</h2>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #FFD700"></div>乾（天）創造</div>
                <div class="legend-item"><div class="legend-color" style="background: #8B4513"></div>坤（地）受容</div>
                <div class="legend-item"><div class="legend-color" style="background: #FF4500"></div>震（雷）動き</div>
                <div class="legend-item"><div class="legend-color" style="background: #32CD32"></div>巽（風）浸透</div>
                <div class="legend-item"><div class="legend-color" style="background: #1E90FF"></div>坎（水）危険</div>
                <div class="legend-item"><div class="legend-color" style="background: #FF6347"></div>離（火）明るさ</div>
                <div class="legend-item"><div class="legend-color" style="background: #708090"></div>艮（山）停止</div>
                <div class="legend-item"><div class="legend-color" style="background: #FF69B4"></div>兌（沢）喜び</div>
            </div>

            <h2>エッジ色（成功率）</h2>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #2ECC71"></div>80%以上</div>
                <div class="legend-item"><div class="legend-color" style="background: #F1C40F"></div>50-79%</div>
                <div class="legend-item"><div class="legend-color" style="background: #E67E22"></div>20-49%</div>
                <div class="legend-item"><div class="legend-color" style="background: #E74C3C"></div>20%未満</div>
            </div>

            <div id="detail-panel">
                <h3 id="detail-title">卦の詳細</h3>
                <dl class="hex-info">
                    <dt>読み</dt>
                    <dd id="detail-reading">-</dd>
                    <dt>キーワード</dt>
                    <dd id="detail-keyword">-</dd>
                    <dt>意味</dt>
                    <dd id="detail-meaning">-</dd>
                    <dt>八卦構成</dt>
                    <dd id="detail-trigrams">-</dd>
                    <dt>遷移元件数</dt>
                    <dd id="detail-from">-</dd>
                    <dt>遷移先件数</dt>
                    <dd id="detail-to">-</dd>
                    <dt>平均成功率</dt>
                    <dd id="detail-success">-</dd>
                </dl>
                <h4 style="margin-top: 15px; font-size: 0.95em; color: #00d4ff;">主な遷移先</h4>
                <div id="transitions-list" class="transitions-list"></div>
            </div>
        </div>

        <div id="main">
            <svg id="network"></svg>
            <div class="controls">
                <button id="zoom-in">拡大 +</button>
                <button id="zoom-out">縮小 -</button>
                <button id="reset-zoom">リセット</button>
            </div>
            <div id="layout-mode">
                <button id="layout-circle" class="active">円形配置</button>
                <button id="layout-force">力学配置</button>
            </div>
        </div>
    </div>

    <div id="tooltip" class="tooltip" style="display: none;"></div>

    <script>
        // データ
        const nodes = {nodes_json};
        const links = {links_json};

        // グローバル変数
        let svg, g, node, link, simulation;
        let currentLayout = 'circle';
        let selectedNode = null;

        // SVG設定
        const width = document.getElementById('main').clientWidth;
        const height = document.getElementById('main').clientHeight;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.38;

        // ズーム設定
        const zoom = d3.zoom()
            .scaleExtent([0.3, 4])
            .on('zoom', (event) => {{
                g.attr('transform', event.transform);
            }});

        // 初期化
        function init() {{
            svg = d3.select('#network')
                .attr('width', width)
                .attr('height', height)
                .call(zoom);

            // 矢印マーカー定義
            const defs = svg.append('defs');

            // 成功率ごとの矢印マーカー
            ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C'].forEach(color => {{
                defs.append('marker')
                    .attr('id', 'arrow-' + color.replace('#', ''))
                    .attr('viewBox', '0 -5 10 10')
                    .attr('refX', 20)
                    .attr('refY', 0)
                    .attr('markerWidth', 6)
                    .attr('markerHeight', 6)
                    .attr('orient', 'auto')
                    .append('path')
                    .attr('d', 'M0,-5L10,0L0,5')
                    .attr('fill', color);
            }});

            g = svg.append('g');

            // 円形配置の初期位置設定
            nodes.forEach((d, i) => {{
                const angle = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
                d.x = centerX + radius * Math.cos(angle);
                d.y = centerY + radius * Math.sin(angle);
                d.fx = d.x;
                d.fy = d.y;
            }});

            // リンク描画
            link = g.append('g')
                .selectAll('path')
                .data(links)
                .join('path')
                .attr('class', 'link')
                .attr('stroke', d => d.color)
                .attr('stroke-width', d => Math.max(1, Math.log(d.count) * 0.8))
                .attr('marker-end', d => 'url(#arrow-' + d.color.replace('#', '') + ')');

            // ノード描画
            node = g.append('g')
                .selectAll('.node')
                .data(nodes)
                .join('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            node.append('circle')
                .attr('r', d => 8 + Math.log(d.totalFrom + 1) * 2)
                .attr('fill', d => d.color);

            node.append('text')
                .attr('dy', -12)
                .text(d => d.chinese);

            // イベント設定
            node.on('click', handleNodeClick)
                .on('mouseover', handleNodeMouseover)
                .on('mouseout', handleNodeMouseout);

            link.on('mouseover', handleLinkMouseover)
                .on('mouseout', handleLinkMouseout);

            // 力学シミュレーション
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(centerX, centerY))
                .force('collision', d3.forceCollide().radius(20))
                .on('tick', ticked)
                .stop();

            updatePositions();
            applyFilters();
        }}

        // 位置更新
        function updatePositions() {{
            link.attr('d', d => {{
                const sourceNode = nodes.find(n => n.id === d.source.id || n.id === d.source);
                const targetNode = nodes.find(n => n.id === d.target.id || n.id === d.target);
                if (!sourceNode || !targetNode) return '';

                const dx = targetNode.x - sourceNode.x;
                const dy = targetNode.y - sourceNode.y;
                const dr = Math.sqrt(dx * dx + dy * dy) * 2;

                return `M${{sourceNode.x}},${{sourceNode.y}}A${{dr}},${{dr}} 0 0,1 ${{targetNode.x}},${{targetNode.y}}`;
            }});

            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }}

        function ticked() {{
            updatePositions();
        }}

        // ドラッグ処理
        function dragstarted(event, d) {{
            if (currentLayout === 'force') {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
            if (currentLayout === 'circle') {{
                d.x = event.x;
                d.y = event.y;
                updatePositions();
            }}
        }}

        function dragended(event, d) {{
            if (currentLayout === 'force') {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
        }}

        // ノードクリック
        function handleNodeClick(event, d) {{
            event.stopPropagation();

            // 選択状態の切り替え
            if (selectedNode === d) {{
                selectedNode = null;
                node.classed('selected', false);
                link.classed('highlighted', false).classed('dimmed', false);
                document.getElementById('detail-panel').classList.remove('active');
            }} else {{
                selectedNode = d;
                node.classed('selected', n => n === d);

                // リンクのハイライト
                link.classed('highlighted', l =>
                    (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id
                ).classed('dimmed', l =>
                    (l.source.id || l.source) !== d.id && (l.target.id || l.target) !== d.id
                );

                // 詳細パネル更新
                showDetails(d);
            }}
        }}

        // 詳細表示
        function showDetails(d) {{
            const panel = document.getElementById('detail-panel');
            panel.classList.add('active');

            document.getElementById('detail-title').textContent = d.id + '（' + d.chinese + '）';
            document.getElementById('detail-reading').textContent = d.reading;
            document.getElementById('detail-keyword').textContent = d.keyword;
            document.getElementById('detail-meaning').textContent = d.meaning;
            document.getElementById('detail-trigrams').textContent =
                '上卦: ' + d.upperTrigram + ' / 下卦: ' + d.lowerTrigram;
            document.getElementById('detail-from').textContent = d.totalFrom.toLocaleString() + '件';
            document.getElementById('detail-to').textContent = d.totalTo.toLocaleString() + '件';
            document.getElementById('detail-success').textContent =
                (d.successRate * 100).toFixed(1) + '%';

            // 遷移先リスト
            const transitionsList = document.getElementById('transitions-list');
            const outLinks = links.filter(l => (l.source.id || l.source) === d.id)
                .sort((a, b) => b.count - a.count)
                .slice(0, 10);

            if (outLinks.length > 0) {{
                transitionsList.innerHTML = outLinks.map(l => {{
                    const targetId = l.target.id || l.target;
                    const successClass = l.successRate >= 0.8 ? 'success-high' :
                                        l.successRate >= 0.5 ? 'success-mid' : 'success-low';
                    return `<div class="transition-item">
                        <span>→ ${{targetId}}</span>
                        <span>
                            <span class="success-badge ${{successClass}}">${{(l.successRate * 100).toFixed(0)}}%</span>
                            <span style="color: #888; margin-left: 5px;">${{l.count}}件</span>
                        </span>
                    </div>`;
                }}).join('');
            }} else {{
                transitionsList.innerHTML = '<div style="color: #888;">遷移先データなし</div>';
            }}
        }}

        // ノードマウスオーバー
        function handleNodeMouseover(event, d) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `
                <h4>${{d.id}} (${{d.chinese}})</h4>
                <div>${{d.keyword}}</div>
                <div style="margin-top: 5px; color: #888;">
                    遷移元: ${{d.totalFrom}}件 / 遷移先: ${{d.totalTo}}件
                </div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function handleNodeMouseout() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        // リンクマウスオーバー
        function handleLinkMouseover(event, d) {{
            const sourceId = d.source.id || d.source;
            const targetId = d.target.id || d.target;
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `
                <h4>${{sourceId}} → ${{targetId}}</h4>
                <div>件数: ${{d.count}}</div>
                <div>成功率: ${{(d.successRate * 100).toFixed(1)}}%</div>
                <div>主要アクション: ${{d.mainAction}}</div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function handleLinkMouseout() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        // フィルター適用
        function applyFilters() {{
            const minSuccessRate = document.getElementById('success-rate-filter').value / 100;
            const minCount = parseInt(document.getElementById('count-filter').value);
            const trigramFilter = document.getElementById('trigram-filter').value;

            // リンクフィルタリング
            link.style('display', d => {{
                if (d.successRate < minSuccessRate) return 'none';
                if (d.count < minCount) return 'none';
                return 'block';
            }});

            // ノードフィルタリング
            node.style('opacity', d => {{
                if (trigramFilter !== 'all' && d.upperTrigram !== trigramFilter && d.lowerTrigram !== trigramFilter) {{
                    return 0.2;
                }}
                return 1;
            }});
        }}

        // レイアウト切り替え
        function setLayout(mode) {{
            currentLayout = mode;

            document.querySelectorAll('#layout-mode button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById('layout-' + mode).classList.add('active');

            if (mode === 'circle') {{
                simulation.stop();
                nodes.forEach((d, i) => {{
                    const angle = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
                    d.fx = centerX + radius * Math.cos(angle);
                    d.fy = centerY + radius * Math.sin(angle);
                    d.x = d.fx;
                    d.y = d.fy;
                }});
                updatePositions();
            }} else {{
                nodes.forEach(d => {{
                    d.fx = null;
                    d.fy = null;
                }});
                simulation.alpha(1).restart();
            }}
        }}

        // イベントリスナー設定
        document.getElementById('success-rate-filter').addEventListener('input', (e) => {{
            document.getElementById('success-rate-value').textContent = e.target.value + '%';
            applyFilters();
        }});

        document.getElementById('count-filter').addEventListener('input', (e) => {{
            document.getElementById('count-value').textContent = e.target.value;
            applyFilters();
        }});

        document.getElementById('trigram-filter').addEventListener('change', applyFilters);

        document.getElementById('zoom-in').addEventListener('click', () => {{
            svg.transition().call(zoom.scaleBy, 1.3);
        }});

        document.getElementById('zoom-out').addEventListener('click', () => {{
            svg.transition().call(zoom.scaleBy, 0.7);
        }});

        document.getElementById('reset-zoom').addEventListener('click', () => {{
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }});

        document.getElementById('layout-circle').addEventListener('click', () => setLayout('circle'));
        document.getElementById('layout-force').addEventListener('click', () => setLayout('force'));

        // 背景クリックで選択解除
        svg.on('click', () => {{
            selectedNode = null;
            node.classed('selected', false);
            link.classed('highlighted', false).classed('dimmed', false);
            document.getElementById('detail-panel').classList.remove('active');
        }});

        // 初期化実行
        init();
    </script>
</body>
</html>
'''

    return html_content

def main():
    """メイン処理"""
    print("64卦遷移ネットワーク可視化を開始...")

    # assetsディレクトリ作成
    ASSETS_DIR.mkdir(exist_ok=True)

    # データ読み込み
    print("データを読み込み中...")
    transition_map, hexagram_master = load_data()

    # ネットワークデータ準備
    print("ネットワークデータを準備中...")
    nodes, links = prepare_network_data(transition_map, hexagram_master)

    metadata = transition_map.get("metadata", {})

    print(f"  ノード数: {len(nodes)}")
    print(f"  リンク数: {len(links)}")
    print(f"  総事例数: {metadata.get('total_cases', 0)}")

    # HTML生成
    print("HTMLを生成中...")
    html_content = generate_html(nodes, links, metadata)

    # ファイル出力
    output_path = ASSETS_DIR / "hexagram_network.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n完了！出力先: {output_path}")
    print(f"ブラウザで開いてください: file://{output_path}")

if __name__ == "__main__":
    main()
