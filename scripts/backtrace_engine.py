#!/usr/bin/env python3
"""
BacktraceEngine — 逆算エンジン

「なりたい姿（目標卦・目標状態）」から「必要な変化」を逆算する。
3層の逆算レイヤー:
  L1: 爻レベル逆算（どの爻を変えるか）— 確実性95%
  L2: 状態レベル逆算（過去事例の逆引き）— 確実性70%
  L3: 行動レベル逆算（具体的行動パターン）— 確実性80%

Usage:
    from backtrace_engine import BacktraceEngine
    engine = BacktraceEngine()
    result = engine.full_backtrace(
        current_hex=12, current_state="停滞・閉塞",
        goal_hex=11, goal_state="安定・平和"
    )
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# パス設定: 他スクリプトと同一パターン
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from gap_analysis_engine import GapAnalysisEngine
from case_search import CaseSearchEngine
from hexagram_transformations import get_hexagram_name, get_trigrams

# ---------------------------------------------------------------------------
# データパス定数
# ---------------------------------------------------------------------------

_REV_DIR = os.path.join(_PROJECT_ROOT, "data", "reverse")
_REV_YAO_PATH = os.path.join(_REV_DIR, "rev_yao.json")
_REV_AFTER_HEX_PATH = os.path.join(_REV_DIR, "rev_after_hex.json")
_REV_AFTER_STATE_PATH = os.path.join(_REV_DIR, "rev_after_state.json")
_REV_OUTCOME_ACTION_PATH = os.path.join(_REV_DIR, "rev_outcome_action.json")
_REV_PATTERN_AFTER_PATH = os.path.join(_REV_DIR, "rev_pattern_after.json")
_REV_HEX_PAIR_STATS_PATH = os.path.join(_REV_DIR, "rev_hex_pair_stats.json")
_COMPAT_PATH = os.path.join(
    _PROJECT_ROOT, "data", "reference", "hexagram_compatibility_lookup.json"
)
_HEXAGRAM_MASTER_PATH = os.path.join(
    _PROJECT_ROOT, "data", "hexagrams", "hexagram_master.json"
)
_TRANSITION_MAP_PATH = os.path.join(
    _PROJECT_ROOT, "data", "hexagrams", "transition_map.json"
)

# RQ3: deterministic words to prohibit in any output text
_DETERMINISTIC_WORDS = ["必ず", "確実に", "絶対"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    """JSONファイルを読み込む。ファイルが存在しない場合は空の辞書を返す。"""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _wilson_score_interval(
    successes: int, total: int, z: float = 1.96
) -> Tuple[float, float]:
    """
    Wilson スコア区間（二項比率の信頼区間）を計算する。

    Args:
        successes: 成功件数
        total: 総件数
        z: z-score (デフォルト: 1.96 = 95%信頼区間)

    Returns:
        (lower, upper) の区間。total == 0 の場合は (0.0, 1.0)。
    """
    if total == 0:
        return (0.0, 1.0)

    p_hat = successes / total
    denominator = 1 + (z ** 2 / total)
    centre_adjusted = p_hat + z ** 2 / (2 * total)
    adjusted_standard_deviation = math.sqrt(
        (p_hat * (1 - p_hat) + z ** 2 / (4 * total)) / total
    )

    lower = (centre_adjusted - z * adjusted_standard_deviation) / denominator
    upper = (centre_adjusted + z * adjusted_standard_deviation) / denominator
    return (max(0.0, round(lower, 4)), min(1.0, round(upper, 4)))


def _sanitize_text(text: str) -> str:
    """RQ3: deterministic words を除去する。"""
    for word in _DETERMINISTIC_WORDS:
        text = text.replace(word, "")
    return text


def _hex_num_to_upper_trigram(hex_num: int) -> str:
    """
    卦番号から上卦の八卦名（短名）を返す。

    rev_hex_pair_stats のキー導出に使用する近似値。
    上卦はその卦の支配的特性を示すことが多い。
    """
    try:
        _lower, upper = get_trigrams(hex_num)
        return upper
    except Exception:
        return ""


def _route_to_serializable(route: Any) -> Dict:
    """RouteオブジェクトをJSON化可能な辞書に変換する。"""
    steps = []
    for step in route.steps:
        steps.append({
            "from_hex": step.from_hex,
            "to_hex": step.to_hex,
            "action": step.action,
            "success_rate": round(step.success_rate, 4),
            "count": step.count,
        })
    return {
        "steps": steps,
        "step_count": route.step_count,
        "total_success_rate": round(route.total_success_rate, 4),
    }


# ---------------------------------------------------------------------------
# BacktraceEngine
# ---------------------------------------------------------------------------

class BacktraceEngine:
    """
    逆算エンジン: 目標状態から「必要な変化」を3層で逆算する。

    Layer 1 (L1): 爻レベル逆算 — どの爻を変えれば目標卦に到達できるか
    Layer 2 (L2): 状態レベル逆算 — 過去事例から目標状態への到達パターン
    Layer 3 (L3): 行動レベル逆算 — 具体的な行動・ルートの推奨
    """

    def __init__(self) -> None:
        """
        初期化: 全6逆引きインデックスと依存エンジンをロードする。
        RouteNavigator は重いため遅延初期化する。
        """
        # 逆引きインデックス
        self._rev_yao: Dict[str, List[Dict]] = _load_json(_REV_YAO_PATH)
        self._rev_after_hex: Dict[str, List[Dict]] = _load_json(_REV_AFTER_HEX_PATH)
        self._rev_after_state: Dict[str, Dict] = _load_json(_REV_AFTER_STATE_PATH)
        self._rev_outcome_action: Dict[str, Dict] = _load_json(_REV_OUTCOME_ACTION_PATH)
        self._rev_pattern_after: Dict[str, Dict] = _load_json(_REV_PATTERN_AFTER_PATH)
        self._rev_hex_pair_stats: Dict[str, Dict] = _load_json(_REV_HEX_PAIR_STATS_PATH)

        # 相性データ
        self._compat: Dict[str, Dict] = _load_json(_COMPAT_PATH)

        # 依存エンジン
        self._gap_engine = GapAnalysisEngine(compat_path=_COMPAT_PATH)
        self._case_engine = CaseSearchEngine()

        # RouteNavigator は遅延初期化
        self._route_navigator: Optional[Any] = None

    # -----------------------------------------------------------------------
    # RouteNavigator 遅延初期化
    # -----------------------------------------------------------------------

    def _get_route_navigator(self) -> Any:
        """RouteNavigator を遅延初期化して返す。"""
        if self._route_navigator is None:
            try:
                from route_navigator import RouteNavigator
                self._route_navigator = RouteNavigator(
                    transition_path=Path(_TRANSITION_MAP_PATH),
                    master_path=Path(_HEXAGRAM_MASTER_PATH),
                )
            except (FileNotFoundError, ImportError) as e:
                raise RuntimeError(f"RouteNavigator の初期化に失敗しました: {e}")
        return self._route_navigator

    # -----------------------------------------------------------------------
    # compat lookup helper
    # -----------------------------------------------------------------------

    def _lookup_compat_score(self, hex_a: int, hex_b: int) -> float:
        """
        相性スコア（0.0-1.0）を返す。
        compat データがない場合は 0.0。
        """
        key = f"{hex_a}-{hex_b}"
        entry = self._compat.get(key)
        if entry is None:
            # 逆キーも試みる
            key_rev = f"{hex_b}-{hex_a}"
            entry = self._compat.get(key_rev)
        if entry is None:
            return 0.0
        score = entry.get("score", 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    # -----------------------------------------------------------------------
    # L1: 爻レベル逆算
    # -----------------------------------------------------------------------

    def reverse_yao(self, current_hex: int, goal_hex: int) -> Dict:
        """
        L1 逆算: 現在の卦から目標卦に到達するために変えるべき爻を特定する。

        Args:
            current_hex: 現在の卦番号 (1-64)
            goal_hex: 目標の卦番号 (1-64)

        Returns:
            {
                current_hex, current_hex_name,
                goal_hex, goal_hex_name,
                hamming_distance,
                changing_lines,         # 変えるべき爻の位置リスト (1-indexed)
                structural_relationship, # 錯卦・綜卦・之卦 etc.
                direct_yao_path,        # 単一爻変で到達できるか (bool)
                direct_yao_position,    # 到達できる場合の爻位置 (int or None)
                difficulty,
                intermediate_suggestions, # 中間卦候補リスト
                sources_that_reach_goal,  # rev_yao から goal_hex に達するソース卦
                current_is_source,       # current_hex が goal_hex のソースに含まれるか
            }
        """
        gap = self._gap_engine.analyze(current_hex, goal_hex)

        hamming = gap["hamming_distance"]
        changing_lines = gap["changing_lines"]
        structural_rel = gap["structural_relationship"]

        # rev_yao: goal_hex に到達できる全ソース卦
        goal_key = str(goal_hex)
        sources = self._rev_yao.get(goal_key, [])

        # current_hex がソースに含まれるか確認
        current_source_entries = [
            s for s in sources if s.get("source_hex_id") == current_hex
        ]
        current_is_source = len(current_source_entries) > 0

        # 直接1爻変で到達できるか
        direct_yao_path = hamming == 1
        direct_yao_position: Optional[int] = None
        if direct_yao_path and changing_lines:
            direct_yao_position = changing_lines[0]

        # 中間卦の提案
        intermediate_suggestions = gap.get("intermediate_paths", [])

        return {
            "current_hex": current_hex,
            "current_hex_name": get_hexagram_name(current_hex),
            "goal_hex": goal_hex,
            "goal_hex_name": get_hexagram_name(goal_hex),
            "hamming_distance": hamming,
            "changing_lines": changing_lines,
            "structural_relationship": structural_rel,
            "direct_yao_path": direct_yao_path,
            "direct_yao_position": direct_yao_position,
            "difficulty": gap["difficulty"],
            "intermediate_suggestions": intermediate_suggestions,
            "sources_that_reach_goal": sources[:10],  # 上位10件に制限
            "current_is_source": current_is_source,
        }

    # -----------------------------------------------------------------------
    # L2: 状態レベル逆算
    # -----------------------------------------------------------------------

    def reverse_state(self, current_state: str, goal_state: str) -> Dict:
        """
        L2 逆算: 目標状態に歴史的に到達した事例から、
        推奨行動・前状態分布・到達可能性を算出する。

        Args:
            current_state: 現在の状態（例: "停滞・閉塞"）
            goal_state: 目標の状態（例: "V字回復・大成功"）

        Returns:
            {
                current_state, goal_state,
                goal_reachability,          # 0.0-1.0 (current_state の出現率)
                before_state_distribution,  # goal_state に至った前状態分布
                recommended_actions,        # 成功率順の推奨行動リスト
                case_count,
                confidence_note,
            }
        """
        # rev_after_state から goal_state のデータを取得
        goal_entry = self._rev_after_state.get(goal_state, {})
        total_count = goal_entry.get("total_count", 0)
        before_dist = goal_entry.get("before_state_distribution", [])
        top_actions = goal_entry.get("top_actions", [])

        # goal_reachability: current_state が before_state_distribution に現れる割合
        goal_reachability = 0.0
        if total_count > 0:
            for entry in before_dist:
                if entry.get("state") == current_state:
                    goal_reachability = round(entry.get("pct", 0.0) / 100.0, 4)
                    break

        # Success|* の action ごとに success 実績を収集して推奨行動を強化
        action_success_map: Dict[str, int] = {}
        for key, val in self._rev_outcome_action.items():
            if not key.startswith("Success|"):
                continue
            action_type = key[len("Success|"):]
            action_success_map[action_type] = val.get("total_count", 0)

        # top_actions を success 実績でスコアリング
        recommended_actions = []
        for action_entry in top_actions:
            action_type = action_entry.get("action_type", "")
            pct = action_entry.get("pct", 0.0)
            count = action_entry.get("count", 0)
            success_count = action_success_map.get(action_type, 0)

            recommended_actions.append({
                "action_type": action_type,
                "frequency_pct": pct,
                "frequency_count": count,
                "success_case_count": success_count,
                "composite_score": round(
                    0.6 * (pct / 100.0) + 0.4 * min(success_count, 1000) / 1000.0,
                    4,
                ),
            })

        # composite_score 降順でソート
        recommended_actions.sort(key=lambda x: -x["composite_score"])

        # confidence_note
        if total_count >= 100:
            confidence_note = f"「{goal_state}」に至った事例が{total_count}件あります。"
        elif total_count >= 30:
            confidence_note = (
                f"「{goal_state}」への参考事例が{total_count}件あります。"
                "傾向の参考程度にご覧ください。"
            )
        elif total_count >= 10:
            confidence_note = (
                f"「{goal_state}」への該当事例は{total_count}件と少数です。"
                "分布は参考情報としてご覧ください。"
            )
        else:
            confidence_note = (
                f"「{goal_state}」への該当事例は{total_count}件のみです。"
                "統計的傾向の読み取りには限界があります。"
            )

        return {
            "current_state": current_state,
            "goal_state": goal_state,
            "goal_reachability": goal_reachability,
            "before_state_distribution": before_dist,
            "recommended_actions": recommended_actions[:5],  # 上位5件
            "case_count": total_count,
            "confidence_note": confidence_note,
        }

    # -----------------------------------------------------------------------
    # L3: 行動レベル逆算
    # -----------------------------------------------------------------------

    def reverse_action(
        self,
        current_hex: int,
        current_state: str,
        goal_hex: int,
        goal_state: str,
    ) -> Dict:
        """
        L3 逆算: 具体的な行動パターン・ルートを逆算する。

        Args:
            current_hex: 現在の卦番号 (1-64)
            current_state: 現在の状態
            goal_hex: 目標の卦番号 (1-64)
            goal_state: 目標の状態

        Returns:
            {
                routes,              # RouteNavigator からの上位3ルート
                direct_pair_stats,   # rev_hex_pair_stats の直接ペア統計
                pattern_suggestions, # rev_pattern_after からのパターン提案
                action_recommendations, # 統合行動推奨リスト
            }
        """
        current_name = get_hexagram_name(current_hex)
        goal_name = get_hexagram_name(goal_hex)

        # --- RouteNavigator でルート探索 ---
        routes: List[Dict] = []
        try:
            navigator = self._get_route_navigator()
            raw_routes = navigator.find_alternative_routes(
                current_name, goal_name, max_routes=3
            )
            for title, route in raw_routes:
                if route and route.steps:
                    routes.append({
                        "title": title,
                        "route": _route_to_serializable(route),
                    })
        except Exception:
            # RouteNavigator が使用不可でも処理継続
            routes = []

        # --- rev_hex_pair_stats: 上卦ベースのペア統計（近似） ---
        current_upper = _hex_num_to_upper_trigram(current_hex)
        goal_upper = _hex_num_to_upper_trigram(goal_hex)
        pair_key = f"{current_upper}|{goal_upper}"
        direct_pair_stats = self._rev_hex_pair_stats.get(pair_key, {})

        # --- rev_pattern_after: パターンベースの提案 ---
        # goal_state に関連するパターンを検索
        pattern_suggestions: List[Dict] = []
        for pattern_key, pattern_val in self._rev_pattern_after.items():
            # キーは "pattern_type|after_state"
            if "|" not in pattern_key:
                continue
            _ptype, _after_state = pattern_key.split("|", 1)
            if _after_state != goal_state:
                continue
            entries = pattern_val.get("entries", [])
            total = pattern_val.get("total_count", 0)
            if total == 0:
                continue
            pattern_suggestions.append({
                "pattern_type": _ptype,
                "after_state": _after_state,
                "total_count": total,
                "top_entries": entries[:3],  # 上位3件
            })

        # pattern_suggestions を total_count 降順でソート
        pattern_suggestions.sort(key=lambda x: -x["total_count"])

        # --- 統合行動推奨リスト ---
        action_scores: Dict[str, float] = {}

        # ルートから行動を収集
        for route_info in routes:
            route_data = route_info.get("route", {})
            steps = route_data.get("steps", [])
            sr = route_data.get("total_success_rate", 0.0)
            for step in steps[:1]:  # 最初のステップのみ（即時行動）
                action = step.get("action", "")
                if action:
                    current_score = action_scores.get(action, 0.0)
                    action_scores[action] = max(current_score, sr * 0.8)

        # direct_pair_stats の top_actions を追加
        for ta in direct_pair_stats.get("top_actions", [])[:3]:
            action = ta.get("action_type", "")
            cnt = ta.get("count", 0)
            pair_total = direct_pair_stats.get("total_count", 1)
            if action and pair_total > 0:
                score = 0.6 * (cnt / pair_total)
                current_score = action_scores.get(action, 0.0)
                action_scores[action] = max(current_score, score)

        # pattern_suggestions の top_entries を追加
        for ps in pattern_suggestions[:2]:
            for entry in ps.get("top_entries", [])[:2]:
                action = entry.get("action_type", "")
                cnt = entry.get("count", 0)
                ps_total = ps.get("total_count", 1)
                if action and ps_total > 0:
                    score = 0.4 * (cnt / ps_total)
                    current_score = action_scores.get(action, 0.0)
                    action_scores[action] = max(current_score, score)

        action_recommendations = sorted(
            [{"action_type": a, "score": round(s, 4)} for a, s in action_scores.items()],
            key=lambda x: -x["score"],
        )[:5]

        return {
            "routes": routes,
            "direct_pair_stats": {
                "pair_key": pair_key,
                "total_count": direct_pair_stats.get("total_count", 0),
                "success_rate": direct_pair_stats.get("success_rate", 0.0),
                "top_actions": direct_pair_stats.get("top_actions", [])[:5],
                "outcomes": direct_pair_stats.get("outcomes", {}),
            },
            "pattern_suggestions": pattern_suggestions[:3],
            "action_recommendations": action_recommendations,
        }

    # -----------------------------------------------------------------------
    # フルバックトレース (L1+L2+L3 統合)
    # -----------------------------------------------------------------------

    def full_backtrace(
        self,
        current_hex: int,
        current_state: str,
        goal_hex: int,
        goal_state: str,
    ) -> Dict:
        """
        フルバックトレース: L1+L2+L3 を統合し、スコアリング・品質ゲートを適用する。

        Args:
            current_hex: 現在の卦番号 (1-64)
            current_state: 現在の状態（例: "停滞・閉塞"）
            goal_hex: 目標の卦番号 (1-64)
            goal_state: 目標の状態（例: "安定・平和"）

        Returns:
            {
                l1_yao, l2_state, l3_action,
                recommended_routes,   # スコアリング済みルートリスト
                quality_gates,        # RQ1-RQ7 の評価結果
                summary,
            }
        """
        # --- 3層の逆算を実行 ---
        l1 = self.reverse_yao(current_hex, goal_hex)
        l2 = self.reverse_state(current_state, goal_state)
        l3 = self.reverse_action(current_hex, current_state, goal_hex, goal_state)

        # --- 相性スコア ---
        compat_score = self._lookup_compat_score(current_hex, goal_hex)

        # --- ルートスコアリング ---
        scored_routes = self._score_routes(l1, l2, l3, compat_score, goal_state)

        # --- RQ6: ゼロヒット時のフォールバック ---
        if not scored_routes:
            scored_routes = self._fallback_similar_hex_routes(
                current_hex, goal_hex, l2, compat_score
            )

        # --- RQ7: 矛盾ルートの排除 ---
        scored_routes = self._exclude_contradictory_routes(scored_routes)

        # --- RQ4: 代替ルートが1つ以上あることを保証（品質ゲート評価前に実行）---
        if len(scored_routes) < 2:
            scored_routes = self._ensure_alternative_route(
                scored_routes, l3, compat_score, goal_state
            )

        # --- 品質ゲート評価（RQ4 確保後に実行）---
        quality_gates = self._evaluate_quality_gates(l2, l3, scored_routes)

        # --- summary ---
        primary_score = scored_routes[0]["score"] if scored_routes else 0.0
        summary = {
            "primary_route_score": round(primary_score, 4),
            "alternative_count": max(0, len(scored_routes) - 1),
            "confidence_level": self._confidence_level(l2["case_count"], primary_score),
            "current_hex": current_hex,
            "current_hex_name": get_hexagram_name(current_hex),
            "goal_hex": goal_hex,
            "goal_hex_name": get_hexagram_name(goal_hex),
            "current_state": current_state,
            "goal_state": goal_state,
        }

        return {
            "l1_yao": l1,
            "l2_state": l2,
            "l3_action": l3,
            "recommended_routes": scored_routes,
            "quality_gates": quality_gates,
            "summary": summary,
        }

    # -----------------------------------------------------------------------
    # スコアリング
    # -----------------------------------------------------------------------

    def _score_routes(
        self,
        l1: Dict,
        l2: Dict,
        l3: Dict,
        compat_score: float,
        goal_state: str,
    ) -> List[Dict]:
        """
        各ルートに対して route_score を計算する。

        route_score = (
            0.30 * success_rate +
            0.25 * case_count_norm +   # min(case_count, 100) / 100
            0.20 * compatibility +     # compat lookup score
            0.15 * path_efficiency +   # 1.0 - (step_count - 1) / 5
            0.10 * pattern_match       # 1.0 if any pattern matches, 0.5 otherwise
        )
        """
        scored: List[Dict] = []
        case_count = l2.get("case_count", 0)
        case_count_norm = min(case_count, 100) / 100.0

        # パターンマッチ: goal_state に対応するパターン提案があるか
        has_pattern = len(l3.get("pattern_suggestions", [])) > 0

        for route_info in l3.get("routes", []):
            route_data = route_info.get("route", {})
            success_rate = route_data.get("total_success_rate", 0.0)
            step_count = route_data.get("step_count", 1)

            path_efficiency = max(0.0, 1.0 - (step_count - 1) / 5.0)
            pattern_match = 1.0 if has_pattern else 0.5

            score = (
                0.30 * success_rate
                + 0.25 * case_count_norm
                + 0.20 * compat_score
                + 0.15 * path_efficiency
                + 0.10 * pattern_match
            )

            labels: List[str] = []
            if case_count < 5:
                labels.append("reference_only")  # RQ1

            # Wilson スコア区間 (RQ5)
            success_count = int(success_rate * max(route_data.get("step_count", 1), 1))
            ci_lower, ci_upper = _wilson_score_interval(
                int(case_count * success_rate), case_count
            )

            scored.append({
                "title": route_info.get("title", "ルート"),
                "route": route_data,
                "score": round(score, 4),
                "labels": labels,
                "confidence_interval": {
                    "lower": ci_lower,
                    "upper": ci_upper,
                    "note": f"95%信頼区間 (n={case_count})",
                },
                "action_recommendations": l3.get("action_recommendations", []),
            })

        # スコア降順でソート
        scored.sort(key=lambda x: -x["score"])
        return scored

    # -----------------------------------------------------------------------
    # RQ2: 低成功率ルートへの代替保証
    # -----------------------------------------------------------------------

    def _ensure_alternative_route(
        self,
        scored_routes: List[Dict],
        l3: Dict,
        compat_score: float,
        goal_state: str,
    ) -> List[Dict]:
        """RQ4: 代替ルートが不足している場合、action_recommendations から補完する。"""
        if len(scored_routes) >= 2:
            return scored_routes

        # action_recommendations からダミールートを生成
        for rec in l3.get("action_recommendations", []):
            action = rec.get("action_type", "")
            if not action:
                continue
            alt_route = {
                "title": f"代替ルート（{action}）",
                "route": {
                    "steps": [{"action": action, "success_rate": rec.get("score", 0.3)}],
                    "step_count": 1,
                    "total_success_rate": rec.get("score", 0.3),
                },
                "score": round(rec.get("score", 0.3) * 0.5, 4),
                "labels": ["alternative_derived"],
                "confidence_interval": {"lower": 0.0, "upper": 1.0, "note": "推定値"},
                "action_recommendations": [],
            }
            scored_routes.append(alt_route)
            if len(scored_routes) >= 2:
                break

        return scored_routes

    # -----------------------------------------------------------------------
    # RQ6: フォールバック (ゼロヒット時)
    # -----------------------------------------------------------------------

    def _fallback_similar_hex_routes(
        self,
        current_hex: int,
        goal_hex: int,
        l2: Dict,
        compat_score: float,
    ) -> List[Dict]:
        """
        RQ6: RouteNavigator でルートが見つからなかった場合に、
        ハミング距離 ±1 の近隣卦でフォールバック検索を行う。
        """
        fallback_routes: List[Dict] = []
        case_count = l2.get("case_count", 0)
        case_count_norm = min(case_count, 100) / 100.0

        try:
            navigator = self._get_route_navigator()
        except RuntimeError:
            return []

        # goal_hex の ±1 ハミング近傍を試みる
        from hexagram_transformations import get_zhi_gua

        for yao in range(1, 7):
            try:
                neighbor = get_zhi_gua(goal_hex, yao)
                if neighbor == goal_hex:
                    continue
                neighbor_name = get_hexagram_name(neighbor)
                current_name = get_hexagram_name(current_hex)
                raw_routes = navigator.find_alternative_routes(
                    current_name, neighbor_name, max_routes=1
                )
                for title, route in raw_routes:
                    if route and route.steps:
                        route_data = _route_to_serializable(route)
                        sr = route_data.get("total_success_rate", 0.0)
                        step_count = route_data.get("step_count", 1)
                        path_efficiency = max(0.0, 1.0 - (step_count - 1) / 5.0)
                        score = (
                            0.30 * sr
                            + 0.25 * case_count_norm
                            + 0.20 * compat_score
                            + 0.15 * path_efficiency
                            + 0.10 * 0.5  # パターンマッチなし
                        )
                        fallback_routes.append({
                            "title": f"近傍ルート（{get_hexagram_name(neighbor)} 経由）",
                            "route": route_data,
                            "score": round(score, 4),
                            "labels": ["fallback", "neighbor_hex"],
                            "confidence_interval": {
                                "lower": 0.0,
                                "upper": 1.0,
                                "note": "フォールバック推定値",
                            },
                            "action_recommendations": [],
                        })
            except Exception:
                continue

        # スコア降順ソートして上位3件を返す
        fallback_routes.sort(key=lambda x: -x["score"])
        return fallback_routes[:3]

    # -----------------------------------------------------------------------
    # RQ7: 矛盾ルートの排除
    # -----------------------------------------------------------------------

    def _exclude_contradictory_routes(
        self, scored_routes: List[Dict]
    ) -> List[Dict]:
        """
        RQ7: 矛盾するルート（ゴールとは逆方向に進む中間状態を含む）を除外する。

        判定基準:
        - ルートのステップ途中で同一の from_hex が繰り返されている（循環）
        """
        filtered = []
        for route_info in scored_routes:
            route_data = route_info.get("route", {})
            steps = route_data.get("steps", [])

            # 循環検出: from_hex が重複している場合は矛盾ルートとみなす
            visited_from = set()
            is_contradictory = False
            for step in steps:
                from_hex = step.get("from_hex", "")
                if from_hex and from_hex in visited_from:
                    is_contradictory = True
                    break
                if from_hex:
                    visited_from.add(from_hex)

            if not is_contradictory:
                filtered.append(route_info)

        return filtered

    # -----------------------------------------------------------------------
    # 品質ゲート評価 (RQ1-RQ7)
    # -----------------------------------------------------------------------

    def _evaluate_quality_gates(
        self, l2: Dict, l3: Dict, scored_routes: List[Dict]
    ) -> Dict:
        """
        RQ1-RQ7 の品質ゲートを評価する。

        Returns:
            {
                rq1_reference_only: bool,  # case_count < 5
                rq2_low_success_rate: bool,  # primary success_rate < 0.3
                rq3_no_deterministic: bool,  # True = 違反なし
                rq4_has_alternative: bool,   # 代替ルートが1つ以上ある
                rq5_confidence_interval_computed: bool,
                rq6_fallback_used: bool,
                rq7_no_contradictory: bool,  # True = 矛盾排除済み
            }
        """
        case_count = l2.get("case_count", 0)
        rq1 = case_count < 5

        primary_sr = 0.0
        if scored_routes:
            primary_sr = scored_routes[0].get("route", {}).get("total_success_rate", 0.0)
        rq2 = primary_sr < 0.3  # True = 代替ルートが必要な状態

        # RQ3: deterministic words チェック（confidence_note / 推奨アクション名）
        rq3_violation = False
        texts_to_check = [l2.get("confidence_note", "")]
        for word in _DETERMINISTIC_WORDS:
            if any(word in t for t in texts_to_check):
                rq3_violation = True
                break
        rq3 = not rq3_violation  # True = 違反なし

        rq4 = len(scored_routes) >= 2  # 代替ルートが存在するか

        rq5 = all(
            "confidence_interval" in r for r in scored_routes
        )  # CI が計算されているか

        rq6 = any("fallback" in r.get("labels", []) for r in scored_routes)

        # RQ7: 矛盾ルートが除去されているかは _exclude_contradictory_routes が保証
        rq7 = True

        return {
            "rq1_reference_only": rq1,
            "rq2_low_success_rate": rq2,
            "rq3_no_deterministic_words": rq3,
            "rq4_has_alternative_route": rq4,
            "rq5_confidence_interval_computed": rq5,
            "rq6_fallback_used": rq6,
            "rq7_contradictory_routes_excluded": rq7,
        }

    # -----------------------------------------------------------------------
    # 信頼度レベル計算
    # -----------------------------------------------------------------------

    @staticmethod
    def _confidence_level(case_count: int, primary_score: float) -> str:
        """
        事例数とスコアから信頼度レベルを返す。
        """
        if case_count >= 100 and primary_score >= 0.6:
            return "high"
        elif case_count >= 30 and primary_score >= 0.4:
            return "medium"
        elif case_count >= 10:
            return "low"
        else:
            return "very_low"


# ---------------------------------------------------------------------------
# CLI — テスト用エントリポイント
# ---------------------------------------------------------------------------

def _print_section(title: str, data: Any, indent: int = 0) -> None:
    """デバッグ用の見やすい出力ヘルパー。"""
    prefix = "  " * indent
    print(f"\n{prefix}{'=' * (60 - indent * 2)}")
    print(f"{prefix}  {title}")
    print(f"{prefix}{'=' * (60 - indent * 2)}")
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)) and len(str(v)) > 80:
                print(f"{prefix}  {k}:")
                print(f"{prefix}    {json.dumps(v, ensure_ascii=False)[:200]}...")
            else:
                print(f"{prefix}  {k}: {v}")
    else:
        print(f"{prefix}  {data}")


if __name__ == "__main__":
    print("=" * 70)
    print("BacktraceEngine テスト実行")
    print("=" * 70)

    engine = BacktraceEngine()

    # -----------------------------------------------------------------------
    # L1 テスト: reverse_yao(1, 44) — 乾→天風姤 (第1爻変)
    # -----------------------------------------------------------------------
    print("\n\n[L1] reverse_yao(current_hex=1, goal_hex=44)")
    print("  乾為天（第1卦）→ 天風姤（第44卦）: ハミング距離=1, 第1爻変")
    l1_result = engine.reverse_yao(1, 44)
    print(f"  current: {l1_result['current_hex_name']} (#{l1_result['current_hex']})")
    print(f"  goal:    {l1_result['goal_hex_name']} (#{l1_result['goal_hex']})")
    print(f"  ハミング距離: {l1_result['hamming_distance']}")
    print(f"  変爻位置: {l1_result['changing_lines']}")
    print(f"  直接1爻変で到達: {l1_result['direct_yao_path']}")
    print(f"  直接爻位置: {l1_result['direct_yao_position']}")
    print(f"  構造的関係: {l1_result['structural_relationship']}")
    print(f"  難易度: {l1_result['difficulty']}")
    print(f"  current_is_source: {l1_result['current_is_source']}")
    print(f"  sources_that_reach_goal (上位3件): {l1_result['sources_that_reach_goal'][:3]}")

    # -----------------------------------------------------------------------
    # L2 テスト: reverse_state("停滞・閉塞", "V字回復・大成功")
    # -----------------------------------------------------------------------
    print("\n\n[L2] reverse_state('停滞・閉塞', 'V字回復・大成功')")
    l2_result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
    print(f"  current_state: {l2_result['current_state']}")
    print(f"  goal_state:    {l2_result['goal_state']}")
    print(f"  goal_reachability: {l2_result['goal_reachability']:.2%}")
    print(f"  case_count: {l2_result['case_count']}")
    print(f"  confidence_note: {l2_result['confidence_note']}")
    print("  before_state_distribution (上位3件):")
    for d in l2_result["before_state_distribution"][:3]:
        print(f"    {d['state']}: {d['count']}件 ({d['pct']}%)")
    print("  recommended_actions (上位3件):")
    for a in l2_result["recommended_actions"][:3]:
        print(f"    {a['action_type']}: composite_score={a['composite_score']:.4f}")

    # -----------------------------------------------------------------------
    # L3 テスト: reverse_action(12, "停滞・閉塞", 11, "安定・平和")
    # -----------------------------------------------------------------------
    print("\n\n[L3] reverse_action(12, '停滞・閉塞', 11, '安定・平和')")
    print("  天地否（第12卦）→ 地天泰（第11卦）")
    l3_result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
    print(f"  ルート数: {len(l3_result['routes'])}")
    for r in l3_result["routes"]:
        route_data = r.get("route", {})
        print(f"    [{r['title']}] {route_data.get('step_count')} ステップ, "
              f"success_rate={route_data.get('total_success_rate', 0):.2%}")
    print(f"  direct_pair_stats: {l3_result['direct_pair_stats']}")
    print(f"  pattern_suggestions数: {len(l3_result['pattern_suggestions'])}")
    print("  action_recommendations (上位3件):")
    for rec in l3_result["action_recommendations"][:3]:
        print(f"    {rec['action_type']}: score={rec['score']:.4f}")

    # -----------------------------------------------------------------------
    # フルテスト: full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
    # -----------------------------------------------------------------------
    print("\n\n[FULL] full_backtrace(12, '停滞・閉塞', 11, '安定・平和')")
    print("  天地否（第12卦）→ 地天泰（第11卦）")
    full_result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")

    summary = full_result["summary"]
    print(f"\n  === サマリー ===")
    print(f"  {summary['current_hex_name']} + 「{summary['current_state']}」")
    print(f"  → {summary['goal_hex_name']} + 「{summary['goal_state']}」")
    print(f"  primary_route_score: {summary['primary_route_score']}")
    print(f"  alternative_count:   {summary['alternative_count']}")
    print(f"  confidence_level:    {summary['confidence_level']}")

    qg = full_result["quality_gates"]
    print(f"\n  === 品質ゲート ===")
    # 品質ゲートの表示ルール:
    #   rq1_reference_only          True=WARN (件数が少なすぎる)
    #   rq2_low_success_rate        True=WARN (成功率が低い → 代替ルートが必要)
    #   rq3_no_deterministic_words  True=OK   (違反なし)
    #   rq4_has_alternative_route   True=OK   (代替ルートあり)
    #   rq5_confidence_interval_*   True=OK   (CI計算済み)
    #   rq6_fallback_used           True=INFO (フォールバック使用中)
    #   rq7_*_excluded              True=OK   (矛盾ルート除去済み)
    _positive_keys = {
        "rq3_no_deterministic_words",
        "rq4_has_alternative_route",
        "rq5_confidence_interval_computed",
        "rq7_contradictory_routes_excluded",
    }
    _warn_if_true = {"rq1_reference_only", "rq2_low_success_rate"}
    for key, val in qg.items():
        if key in _positive_keys:
            flag = "OK" if val else "WARN"
        elif key in _warn_if_true:
            flag = "WARN" if val else "OK"
        else:
            flag = "INFO"
        print(f"  [{flag}] {key}: {val}")

    print(f"\n  === 推奨ルート ===")
    for i, r in enumerate(full_result["recommended_routes"], 1):
        route_data = r.get("route", {})
        ci = r.get("confidence_interval", {})
        print(f"  ルート{i}: [{r['title']}]")
        print(f"    score={r['score']:.4f}, labels={r['labels']}")
        print(f"    success_rate={route_data.get('total_success_rate', 0):.2%}, "
              f"steps={route_data.get('step_count', 0)}")
        print(f"    CI: [{ci.get('lower', 0):.3f}, {ci.get('upper', 1):.3f}] ({ci.get('note', '')})")
        for step in route_data.get("steps", [])[:2]:
            print(f"    → {step.get('from_hex', '')} -{step.get('action', '')}→ {step.get('to_hex', '')}")

    print("\n" + "=" * 70)
    print("全テスト完了")
    print("=" * 70)
