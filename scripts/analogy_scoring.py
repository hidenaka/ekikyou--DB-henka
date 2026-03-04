#!/usr/bin/env python3
"""
AnalogyScoring — 事例類似度スコアリング

ハードフィルタリング（scaleでの厳密な分離）に加えて、
他scaleの事例も「どの程度類似しているか」を数値化する。

主な用途:
1. cross_scale_patterns の類似度ランキング改善
2. 事例検索時の「参考事例」のスコアリング
3. BacktraceEngineのルート推奨の精度向上

公式:
    TotalScore = BaseChangeSimilarity * ScaleMatchWeight * ExpertiseMatchWeight

BaseChangeSimilarity:
    卦変化パターンの類似度（Hamming距離の逆数 + action_type一致 + state一致 + outcome一致）

ScaleMatchWeight:
    スケール距離に応じた重み（同一scale=1.0, 隣接=0.7, 遠い=0.3）

ExpertiseMatchWeight:
    専門性レベルに応じた重み（未実装 — デフォルト1.0）
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# パス設定: backtrace_engine.py と同じパターン
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from gap_analysis_engine import GapAnalysisEngine

# ---------------------------------------------------------------------------
# スケール距離マトリクス (0=同一, 1=隣接, 2=遠い)
# ---------------------------------------------------------------------------
SCALE_DISTANCE: Dict[Tuple[str, str], int] = {
    ("company", "company"): 0,
    ("company", "individual"): 2,
    ("company", "family"): 1,       # 家族経営は企業と近い
    ("company", "country"): 2,
    ("company", "other"): 1,
    ("individual", "individual"): 0,
    ("individual", "family"): 1,    # 個人と家族は近い
    ("individual", "country"): 2,
    ("individual", "other"): 1,
    ("family", "family"): 0,
    ("family", "country"): 2,
    ("family", "other"): 1,
    ("country", "country"): 0,
    ("country", "other"): 1,
    ("other", "other"): 0,
}

SCALE_WEIGHT: Dict[int, float] = {0: 1.0, 1: 0.7, 2: 0.3}


class AnalogyScorer:
    """事例間の類似度を計算するスコアラー。

    cases.jsonl のインデックスは保持しない。
    入力として (source_case, target_case) の辞書ペアを受け取る。
    """

    def __init__(self) -> None:
        self._gap_engine = GapAnalysisEngine()

    # ------------------------------------------------------------------
    # BaseChangeSimilarity
    # ------------------------------------------------------------------

    def compute_base_similarity(self, source: Dict, target: Dict) -> float:
        """
        BaseChangeSimilarity を計算する。

        構成要素:
        1. hex_similarity: before_hex/after_hex のHamming距離の逆数 (0-1)
        2. action_similarity: action_type の一致度 (0 or 1)
        3. state_similarity: before_state/after_state の一致度 (0-1)
        4. outcome_similarity: outcome の一致度 (0 or 1)

        base_similarity = 0.35 * hex_similarity
                        + 0.25 * action_similarity
                        + 0.25 * state_similarity
                        + 0.15 * outcome_similarity

        Args:
            source: ソース事例の辞書
            target: ターゲット事例の辞書

        Returns:
            0.0 〜 1.0 の類似度スコア
        """
        hex_sim = self._hex_similarity(source, target)
        action_sim = self._action_similarity(source, target)
        state_sim = self._state_similarity(source, target)
        outcome_sim = self._outcome_similarity(source, target)

        return round(
            0.35 * hex_sim
            + 0.25 * action_sim
            + 0.25 * state_sim
            + 0.15 * outcome_sim,
            4,
        )

    # ------------------------------------------------------------------
    # ScaleMatchWeight
    # ------------------------------------------------------------------

    def compute_scale_weight(self, source_scale: str, target_scale: str) -> float:
        """ScaleMatchWeight を計算する。

        Args:
            source_scale: ソース事例のscale
            target_scale: ターゲット事例のscale

        Returns:
            1.0 (同一scale), 0.7 (隣接), 0.3 (遠い)
        """
        dist = SCALE_DISTANCE.get(
            (source_scale, target_scale),
            SCALE_DISTANCE.get((target_scale, source_scale), 2),
        )
        return SCALE_WEIGHT[dist]

    # ------------------------------------------------------------------
    # TotalScore
    # ------------------------------------------------------------------

    def compute_total_score(self, source: Dict, target: Dict) -> float:
        """TotalScore = BaseChangeSimilarity * ScaleMatchWeight

        ExpertiseMatchWeight は未実装のため 1.0 固定。

        Args:
            source: ソース事例の辞書
            target: ターゲット事例の辞書

        Returns:
            0.0 〜 1.0 の総合スコア
        """
        base = self.compute_base_similarity(source, target)
        scale_w = self.compute_scale_weight(
            source.get("scale", "other"),
            target.get("scale", "other"),
        )
        return round(base * scale_w, 4)

    # ------------------------------------------------------------------
    # ランキング
    # ------------------------------------------------------------------

    def rank_analogies(
        self,
        target_case: Dict,
        candidate_cases: List[Dict],
        top_n: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """
        target_case に対して candidate_cases を類似度順にランキングする。

        Args:
            target_case: 類似度比較の基準となる事例
            candidate_cases: ランキング対象の事例リスト
            top_n: 返却する上位件数

        Returns:
            [(score, case), ...] sorted by score desc
        """
        scored: List[Tuple[float, Dict]] = []
        for case in candidate_cases:
            score = self.compute_total_score(target_case, case)
            scored.append((score, case))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_n]

    # ------------------------------------------------------------------
    # 内部: 各類似度サブスコア
    # ------------------------------------------------------------------

    def _hex_similarity(self, source: Dict, target: Dict) -> float:
        """卦変化パターンのHamming距離ベース類似度。

        source と target の (before_hex_num, after_hex_num) ペアの
        Hamming距離を計算し、 1.0 - hamming/6.0 で正規化する。

        before_hex_num / after_hex_num が取れない場合は
        before_hex (八卦名) / after_hex (八卦名) から推定せず 0.0 を返す。
        """
        s_before = self._get_hex_num(source, "before")
        s_after = self._get_hex_num(source, "after")
        t_before = self._get_hex_num(target, "before")
        t_after = self._get_hex_num(target, "after")

        if None in (s_before, s_after, t_before, t_after):
            return 0.0

        # before同士、after同士のHamming距離を平均
        try:
            gap_before = self._gap_engine.analyze(s_before, t_before)
            gap_after = self._gap_engine.analyze(s_after, t_after)
        except (ValueError, Exception):
            return 0.0

        hamming_before = gap_before["hamming_distance"]
        hamming_after = gap_after["hamming_distance"]
        avg_hamming = (hamming_before + hamming_after) / 2.0

        return round(1.0 - avg_hamming / 6.0, 4)

    def _action_similarity(self, source: Dict, target: Dict) -> float:
        """action_type の一致度。完全一致 = 1.0, 不一致 = 0.0。"""
        s_action = source.get("action_type", "")
        t_action = target.get("action_type", "")
        if not s_action or not t_action:
            return 0.0
        return 1.0 if s_action == t_action else 0.0

    def _state_similarity(self, source: Dict, target: Dict) -> float:
        """before_state/after_state の一致度。

        before_state と after_state それぞれについて:
        - 完全一致 = 1.0
        - 「・」区切りの片方が一致 = 0.5
        - 不一致 = 0.0

        2つの平均を返す。
        """
        before_sim = self._single_state_similarity(
            source.get("before_state", ""),
            target.get("before_state", ""),
        )
        after_sim = self._single_state_similarity(
            source.get("after_state", ""),
            target.get("after_state", ""),
        )
        return round((before_sim + after_sim) / 2.0, 4)

    def _outcome_similarity(self, source: Dict, target: Dict) -> float:
        """outcome の一致度。完全一致 = 1.0, 不一致 = 0.0。"""
        s_outcome = source.get("outcome", "")
        t_outcome = target.get("outcome", "")
        if not s_outcome or not t_outcome:
            return 0.0
        return 1.0 if s_outcome == t_outcome else 0.0

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hex_num(case: Dict, prefix: str) -> Optional[int]:
        """事例から before/after の卦番号を取得する。

        優先順位:
        1. before_hex_num / after_hex_num (整数フィールド)
        2. hexagram_number (before の場合のみ)
        3. 取得不能 → None
        """
        key = f"{prefix}_hex_num"
        val = case.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

        # before の場合は hexagram_number もフォールバック
        if prefix == "before":
            hex_num = case.get("hexagram_number")
            if hex_num is not None:
                try:
                    return int(hex_num)
                except (TypeError, ValueError):
                    pass

        # after の場合: after_hex から番号を直接取れないため None
        return None

    @staticmethod
    def _single_state_similarity(state_a: str, state_b: str) -> float:
        """単一の状態ラベル同士の類似度。

        - 完全一致 = 1.0
        - 「・」区切りの片方が一致 = 0.5
        - 不一致 = 0.0
        """
        if not state_a or not state_b:
            return 0.0
        if state_a == state_b:
            return 1.0

        parts_a = set(state_a.split("・"))
        parts_b = set(state_b.split("・"))
        if parts_a & parts_b:
            return 0.5

        return 0.0
