#!/usr/bin/env python3
"""
64卦 変化ルートナビゲーター

現在の卦から目標の卦への最適な変化ルートを探索する。
- 最短ルート（ステップ数最小）
- 高成功率ルート（成功確率最大化）

使用例:
    python3 scripts/route_navigator.py --from "29_坎" --to "14_大有"
    python3 scripts/route_navigator.py --from "坎為水" --to "火天大有"
"""

import json
import heapq
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# パス設定
BASE_DIR = Path(__file__).parent.parent
TRANSITION_MAP_PATH = BASE_DIR / "data" / "hexagrams" / "transition_map.json"
HEXAGRAM_MASTER_PATH = BASE_DIR / "data" / "hexagrams" / "hexagram_master.json"


# 八卦アクション名と意味
ACTION_MEANINGS = {
    "乾": "積極的にリードする",
    "坤": "受容し支える",
    "震": "積極的に動く・衝撃を与える",
    "巽": "柔軟に浸透する・交渉する",
    "坎": "困難に耐える・深く掘り下げる",
    "離": "明確化する・可視化する",
    "艮": "止まる・守る・蓄積する",
    "兌": "喜びを与える・対話する",
}


@dataclass
class TransitionStep:
    """遷移ステップの情報"""
    from_hex: str
    to_hex: str
    action: str
    success_rate: float
    count: int

    def __repr__(self):
        return f"{self.from_hex} → {self.to_hex} (行動: {self.action}, 成功率: {self.success_rate:.1%})"


@dataclass
class Route:
    """ルート情報"""
    steps: list[TransitionStep]
    total_success_rate: float

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def __repr__(self):
        return f"Route({self.step_count}ステップ, 総合成功率: {self.total_success_rate:.1%})"


class HexagramNormalizer:
    """卦名の正規化を行うクラス"""

    def __init__(self, master_path: Path):
        self.master = self._load_master(master_path)
        self._build_lookup_tables()

    def _load_master(self, path: Path) -> dict:
        """マスターデータを読み込む"""
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_lookup_tables(self):
        """検索用テーブルを構築"""
        # 番号→名前
        self.id_to_name = {}
        # フルネーム→標準キー
        self.fullname_to_key = {}
        # 短縮名→標準キー
        self.shortname_to_key = {}
        # 八卦名→純卦キー
        self.trigram_to_pure = {}

        # 八卦の純卦マッピング
        pure_hexagrams = {
            "乾": "1_乾", "坤": "2_坤", "震": "51_震", "巽": "57_巽",
            "坎": "29_坎", "離": "30_離", "艮": "52_艮", "兌": "58_兌"
        }
        self.trigram_to_pure = pure_hexagrams

        for id_str, info in self.master.items():
            id_num = int(id_str)
            name = info.get("name", "")
            chinese = info.get("chinese", "")

            # 標準キー形式: "番号_八卦" (例: "1_乾", "29_坎")
            # 遷移マップでは "1_乾", "29_坎" 形式で格納されている
            if chinese:
                standard_key = f"{id_num}_{chinese}"
                self.id_to_name[id_num] = name
                self.fullname_to_key[name] = standard_key
                self.shortname_to_key[chinese] = standard_key

                # 番号付き形式
                self.shortname_to_key[f"{id_num}_{chinese}"] = standard_key
                self.shortname_to_key[f"{id_num}_{name}"] = standard_key

    def normalize(self, hex_input: str) -> Optional[str]:
        """
        入力された卦名を標準形式に正規化

        入力例:
            "29_坎" → "29_坎"
            "坎為水" → "29_坎"
            "坎" → "29_坎"
            "14_大有" → "火天大有"（フルネーム）
        """
        hex_input = hex_input.strip()

        # 既に標準キー形式の場合
        if hex_input in self.shortname_to_key:
            return self.shortname_to_key[hex_input]

        # フルネームの場合
        if hex_input in self.fullname_to_key:
            return self.fullname_to_key[hex_input]

        # 八卦名のみの場合（純卦）
        if hex_input in self.trigram_to_pure:
            return self.trigram_to_pure[hex_input]

        # 番号_名前形式 (例: "14_大有")
        if "_" in hex_input:
            parts = hex_input.split("_", 1)
            if parts[0].isdigit():
                id_num = int(parts[0])
                if id_num in self.id_to_name:
                    # フルネームを返す
                    return self.id_to_name[id_num]

        # 番号のみの場合
        if hex_input.isdigit():
            id_num = int(hex_input)
            if id_num in self.id_to_name:
                return self.id_to_name[id_num]

        return None

    def get_display_name(self, hex_key: str) -> str:
        """表示用の卦名を取得"""
        # フルネームを検索
        for name, key in self.fullname_to_key.items():
            if key == hex_key:
                return name

        # 既にフルネームの場合
        if hex_key in self.fullname_to_key:
            return hex_key

        return hex_key


class RouteNavigator:
    """変化ルートナビゲーター"""

    def __init__(self, transition_path: Path = TRANSITION_MAP_PATH,
                 master_path: Path = HEXAGRAM_MASTER_PATH):
        self.normalizer = HexagramNormalizer(master_path)
        self.transitions = self._load_transitions(transition_path)
        self.graph = self._build_graph()

    def _load_transitions(self, path: Path) -> dict:
        """遷移マップを読み込む"""
        if not path.exists():
            raise FileNotFoundError(f"遷移マップが見つかりません: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("transitions", {})

    def _build_graph(self) -> dict:
        """遷移データからグラフを構築"""
        graph = {}

        for from_hex, targets in self.transitions.items():
            if from_hex not in graph:
                graph[from_hex] = {}

            for to_hex, data in targets.items():
                # 自己遷移は除外（同じ卦に留まる場合は進まない）
                if from_hex == to_hex:
                    continue

                success_rate = data.get("success_rate", 0)
                main_action = data.get("main_action", "")
                count = data.get("count", 0)

                graph[from_hex][to_hex] = {
                    "success_rate": success_rate,
                    "action": main_action,
                    "count": count
                }

        return graph

    def _find_matching_keys(self, hex_input: str) -> list[str]:
        """入力に一致するグラフのキーを検索"""
        # まず正規化を試みる
        normalized = self.normalizer.normalize(hex_input)

        matching_keys = []

        # 完全一致
        if hex_input in self.graph:
            matching_keys.append(hex_input)

        # 正規化後の形式で検索
        if normalized and normalized in self.graph:
            if normalized not in matching_keys:
                matching_keys.append(normalized)

        # 部分一致（入力を含むキーを検索）
        for key in self.graph.keys():
            if hex_input in key or (normalized and normalized in key):
                if key not in matching_keys:
                    matching_keys.append(key)

        # フルネームでの検索
        display_name = self.normalizer.get_display_name(normalized) if normalized else hex_input
        for key in self.graph.keys():
            if display_name == key or display_name in key:
                if key not in matching_keys:
                    matching_keys.append(key)

        return matching_keys

    def find_shortest_route(self, from_hex: str, to_hex: str, max_steps: int = 5) -> Optional[Route]:
        """
        BFSで最短ルートを探索

        Args:
            from_hex: 出発点の卦
            to_hex: 目標の卦
            max_steps: 最大ステップ数

        Returns:
            最短ルート（見つからない場合はNone）
        """
        # キーの検索
        from_keys = self._find_matching_keys(from_hex)
        to_keys = self._find_matching_keys(to_hex)

        if not from_keys:
            return None
        if not to_keys:
            return None

        # 最初に見つかったキーを使用
        start = from_keys[0]
        goals = set(to_keys)

        # BFS
        queue = [(start, [])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if current in goals:
                return self._build_route(path)

            if len(path) >= max_steps:
                continue

            for next_hex, data in self.graph.get(current, {}).items():
                if next_hex not in visited:
                    visited.add(next_hex)
                    new_path = path + [TransitionStep(
                        from_hex=current,
                        to_hex=next_hex,
                        action=data["action"],
                        success_rate=data["success_rate"],
                        count=data["count"]
                    )]
                    queue.append((next_hex, new_path))

        return None

    def find_high_success_route(self, from_hex: str, to_hex: str,
                                 max_steps: int = 5, min_success_rate: float = 0.5) -> Optional[Route]:
        """
        ダイクストラ法で高成功率ルートを探索

        成功率の積を最大化するルートを探索（-log(成功率)の和を最小化）

        Args:
            from_hex: 出発点の卦
            to_hex: 目標の卦
            max_steps: 最大ステップ数
            min_success_rate: 各ステップの最低成功率

        Returns:
            高成功率ルート（見つからない場合はNone）
        """
        import math

        from_keys = self._find_matching_keys(from_hex)
        to_keys = self._find_matching_keys(to_hex)

        if not from_keys or not to_keys:
            return None

        start = from_keys[0]
        goals = set(to_keys)

        # (コスト, ステップ数, 現在地, パス)
        # コスト = -log(成功率)の和 → 小さいほど成功率が高い
        heap = [(0, 0, start, [])]
        best_cost = {start: 0}

        while heap:
            cost, steps, current, path = heapq.heappop(heap)

            if current in goals:
                return self._build_route(path)

            if steps >= max_steps:
                continue

            if cost > best_cost.get(current, float('inf')):
                continue

            for next_hex, data in self.graph.get(current, {}).items():
                success_rate = data["success_rate"]

                # 最低成功率を満たさない遷移はスキップ
                if success_rate < min_success_rate:
                    continue

                # コスト計算（成功率が高いほどコストが低い）
                if success_rate > 0:
                    edge_cost = -math.log(success_rate)
                else:
                    edge_cost = float('inf')

                new_cost = cost + edge_cost

                if new_cost < best_cost.get(next_hex, float('inf')):
                    best_cost[next_hex] = new_cost
                    new_path = path + [TransitionStep(
                        from_hex=current,
                        to_hex=next_hex,
                        action=data["action"],
                        success_rate=success_rate,
                        count=data["count"]
                    )]
                    heapq.heappush(heap, (new_cost, steps + 1, next_hex, new_path))

        return None

    def _build_route(self, steps: list[TransitionStep]) -> Route:
        """ステップリストからルートオブジェクトを構築"""
        if not steps:
            return Route(steps=[], total_success_rate=1.0)

        total_success = 1.0
        for step in steps:
            total_success *= step.success_rate

        return Route(steps=steps, total_success_rate=total_success)

    def find_alternative_routes(self, from_hex: str, to_hex: str,
                                  max_routes: int = 3) -> list[Route]:
        """
        複数の代替ルートを探索
        """
        routes = []

        # 最短ルート
        shortest = self.find_shortest_route(from_hex, to_hex)
        if shortest and shortest.steps:
            routes.append(("最短ルート", shortest))

        # 高成功率ルート（閾値を変えて複数探索）
        for min_rate in [0.7, 0.5, 0.3]:
            high_success = self.find_high_success_route(from_hex, to_hex,
                                                         min_success_rate=min_rate)
            if high_success and high_success.steps:
                # 重複チェック
                is_duplicate = False
                for _, existing in routes:
                    if len(existing.steps) == len(high_success.steps):
                        if all(e.to_hex == h.to_hex for e, h in zip(existing.steps, high_success.steps)):
                            is_duplicate = True
                            break

                if not is_duplicate:
                    routes.append((f"高成功率ルート (閾値{min_rate:.0%})", high_success))

        return routes[:max_routes]


def format_route_output(navigator: RouteNavigator, route: Route, title: str) -> str:
    """ルートを整形して出力"""
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"=== {title} ({route.step_count}ステップ) ===")
    lines.append(f"{'='*50}")

    for i, step in enumerate(route.steps, 1):
        from_name = navigator.normalizer.get_display_name(step.from_hex)
        to_name = navigator.normalizer.get_display_name(step.to_hex)
        action_meaning = ACTION_MEANINGS.get(step.action, "")

        lines.append(f"\nステップ{i}: {from_name} → {to_name}")
        lines.append(f"  行動: {step.action}（{action_meaning}）")
        lines.append(f"  成功率: {step.success_rate:.1%}")
        lines.append(f"  実績件数: {step.count}件")

    lines.append(f"\n{'-'*50}")
    lines.append(f"総合成功確率: {route.total_success_rate:.1%}")
    lines.append(f"{'='*50}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="64卦 変化ルートナビゲーター",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python3 scripts/route_navigator.py --from "29_坎" --to "14_大有"
  python3 scripts/route_navigator.py --from "坎為水" --to "火天大有"
  python3 scripts/route_navigator.py --from "坎" --to "乾" --max-steps 3
        """
    )

    parser.add_argument("--from", dest="from_hex", required=True,
                        help="現在の卦（例: 29_坎, 坎為水, 坎）")
    parser.add_argument("--to", dest="to_hex", required=True,
                        help="目標の卦（例: 14_大有, 火天大有）")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="最大ステップ数（デフォルト: 5）")
    parser.add_argument("--json", action="store_true",
                        help="JSON形式で出力")

    args = parser.parse_args()

    try:
        navigator = RouteNavigator()
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return 1

    print(f"\n探索開始: {args.from_hex} → {args.to_hex}")
    print(f"最大ステップ数: {args.max_steps}")

    # ルート探索
    routes = navigator.find_alternative_routes(args.from_hex, args.to_hex)

    if not routes:
        print(f"\nルートが見つかりませんでした。")
        print(f"ヒント: 入力形式を確認してください（例: 29_坎, 坎為水, 坎）")
        return 1

    if args.json:
        # JSON出力
        result = {
            "from": args.from_hex,
            "to": args.to_hex,
            "routes": []
        }
        for title, route in routes:
            route_data = {
                "title": title,
                "step_count": route.step_count,
                "total_success_rate": route.total_success_rate,
                "steps": [
                    {
                        "from": step.from_hex,
                        "to": step.to_hex,
                        "action": step.action,
                        "action_meaning": ACTION_MEANINGS.get(step.action, ""),
                        "success_rate": step.success_rate,
                        "count": step.count
                    }
                    for step in route.steps
                ]
            }
            result["routes"].append(route_data)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # テキスト出力
        for title, route in routes:
            print(format_route_output(navigator, route, title))

    return 0


if __name__ == "__main__":
    exit(main())
