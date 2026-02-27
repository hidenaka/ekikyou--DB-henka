#!/usr/bin/env python3
"""
日記モードのセッション全体を統括するオーケストレーター

DiaryExtractionEngine, GapAnalysisEngine, ChangeAdviceEngine, FeedbackEngine を
統合し、日記テキストの入力から最終的な変化アドバイスまでの全フローを管理する。

フロー:
    extract_diary() -> add_ideal_followup() -> confirm_dual() -> generate_change_advice()

Usage:
    from diary_session_orchestrator import DiarySessionOrchestrator

    orch = DiarySessionOrchestrator()
    session = orch.create_session()
    result = orch.extract_diary(session, "今日もまた同じ一日だった...")
    if result.get("assessment", {}).get("needs_ideal_followup"):
        result = orch.add_ideal_followup(session, "もっと自由に...")
    result = orch.confirm_dual(session)
    result = orch.generate_change_advice(session)
"""

import json
import os
import sys
from typing import Optional

# --- パス設定 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from diary_extraction_engine import DiaryExtractionEngine
from gap_analysis_engine import GapAnalysisEngine
from change_advice_engine import ChangeAdviceEngine
from feedback_engine import FeedbackEngine

# ---------------------------------------------------------------------------
# hex64_lookup (候補表示用): app.py と同一パターン
# ---------------------------------------------------------------------------
hex64_lookup = {}
_hex64_path = os.path.join(_PROJECT_ROOT, "data", "diagnostic", "hexagram_64.json")
if os.path.isfile(_hex64_path):
    with open(_hex64_path, encoding="utf-8") as _f:
        _hex64_raw = json.load(_f)
    for _name, _info in _hex64_raw["hexagrams"].items():
        hex64_lookup[_info["number"]] = _info

# ---------------------------------------------------------------------------
# compat_lookup (相性データ): app.py と同一パターン
# ---------------------------------------------------------------------------
_compat_path = os.path.join(
    _PROJECT_ROOT, "data", "reference", "hexagram_compatibility_lookup.json"
)
if os.path.exists(_compat_path):
    with open(_compat_path, encoding="utf-8") as _f:
        compat_lookup = json.load(_f)
else:
    compat_lookup = {}


class DiarySessionOrchestrator:
    """日記モードのセッション全体を統括するオーケストレーター"""

    def __init__(self):
        self._diary_engine = DiaryExtractionEngine()
        self._gap_engine = GapAnalysisEngine()
        self._advice_engine = ChangeAdviceEngine()
        self._feedback_engine = FeedbackEngine()

    # ------------------------------------------------------------------
    # セッション作成
    # ------------------------------------------------------------------

    def create_session(self) -> dict:
        """新しい日記セッションを作成して返す。

        Returns:
            セッション状態辞書。session_id は含まない（app.py 側で管理する）。
        """
        return {
            "mode": "diary",
            "phase": "input",
            "user_text": "",
            "dual_extraction": None,
            "assessment": None,
            "summaries": None,
            "hexagram_mapping": None,
            "selected_candidate_index": 0,
            "gap_analysis": None,
            "feedback": None,
            "change_advice": None,
            "followup_count": 0,
        }

    # ------------------------------------------------------------------
    # Phase 1: 日記テキストからの二重抽出
    # ------------------------------------------------------------------

    def extract_diary(self, session: dict, user_text: str) -> dict:
        """日記テキストから current + ideal を抽出する。

        DiaryExtractionEngine の extract_dual, assess_dual_confidence,
        summarize_dual を順に呼び出し、セッション状態を更新する。

        Args:
            session: create_session() で作成したセッション辞書
            user_text: ユーザーの日記テキスト

        Returns:
            {
                "dual_extraction": dict,
                "assessment": dict,
                "summaries": dict,
                "phase": "diary-reviewing",
                "demo_mode": bool,
            }
            エラー時は {"error": str}
        """
        try:
            # 1. 二重抽出
            dual = self._diary_engine.extract_dual(user_text)
            if dual is None:
                return {"error": "日記テキストからの抽出に失敗しました"}

            # 2. 確信度評価
            assessment = self._diary_engine.assess_dual_confidence(dual)

            # 3. 要約生成
            summaries = self._diary_engine.summarize_dual(dual)

            # 4. セッション更新
            session["user_text"] = user_text
            session["dual_extraction"] = dual
            session["assessment"] = assessment
            session["summaries"] = summaries
            session["phase"] = "diary-reviewing"

            # 5. デモモードフラグ
            demo_mode = not self._diary_engine.is_available()

            return {
                "dual_extraction": dual,
                "assessment": assessment,
                "summaries": {
                    "current_summary": summaries.get("current_summary", ""),
                    "ideal_summary": summaries.get("ideal_summary", ""),
                    "gap_summary": summaries.get("gap_summary", ""),
                },
                "phase": "diary-reviewing",
                "demo_mode": demo_mode,
            }

        except Exception as e:
            return {"error": f"抽出処理中にエラーが発生しました: {e}"}

    # ------------------------------------------------------------------
    # Phase 1.5: ideal フォローアップ（オプション）
    # ------------------------------------------------------------------

    def add_ideal_followup(self, session: dict, followup_answer: str) -> dict:
        """理想の状態に関するフォローアップ回答を統合する。

        gap_clarity が低い場合に呼ばれる。元の抽出結果にフォローアップ回答を
        マージし、ideal 側の確信度を改善する。最大1回のフォローアップ。

        Args:
            session: セッション辞書
            followup_answer: ユーザーの理想に関するフォローアップ回答

        Returns:
            extract_diary() と同じ構造の応答辞書。
            エラー時は {"error": str}
        """
        try:
            # フォローアップ回数チェック
            if session.get("followup_count", 0) >= 1:
                return {"error": "フォローアップの上限（1回）に達しました"}

            original_dual = session.get("dual_extraction")
            if original_dual is None:
                return {"error": "抽出結果がありません。先に extract_diary を実行してください"}

            original_text = session.get("user_text", "")

            # 1. マージ
            merged = self._diary_engine.merge_ideal_followup(
                original_dual, followup_answer, original_text
            )

            # 2. 再評価
            assessment = self._diary_engine.assess_dual_confidence(merged)

            # 3. 再要約
            summaries = self._diary_engine.summarize_dual(merged)

            # 4. セッション更新
            session["dual_extraction"] = merged
            session["assessment"] = assessment
            session["summaries"] = summaries
            session["followup_count"] = session.get("followup_count", 0) + 1

            # 5. デモモードフラグ
            demo_mode = not self._diary_engine.is_available()

            return {
                "dual_extraction": merged,
                "assessment": assessment,
                "summaries": {
                    "current_summary": summaries.get("current_summary", ""),
                    "ideal_summary": summaries.get("ideal_summary", ""),
                    "gap_summary": summaries.get("gap_summary", ""),
                },
                "phase": "diary-reviewing",
                "demo_mode": demo_mode,
            }

        except Exception as e:
            return {"error": f"フォローアップ処理中にエラーが発生しました: {e}"}

    # ------------------------------------------------------------------
    # Phase 2: 抽出確定 + 卦マッピング + ギャップ分析
    # ------------------------------------------------------------------

    def confirm_dual(self, session: dict, candidate_index: int = 0) -> dict:
        """抽出結果を確定し、卦マッピングとギャップ分析を実行する。

        DiaryExtractionEngine.map_to_hexagrams() で卦候補を取得し、
        GapAnalysisEngine.analyze() で構造的ギャップを分析する。
        候補に hex64_lookup のメタデータを付与する。

        Args:
            session: セッション辞書
            candidate_index: 選択する current 候補のインデックス (0-2)。
                             デフォルトは 0（最有力候補）。

        Returns:
            {
                "current_candidates": list,
                "goal_hexagram": dict,
                "gap_analysis": dict,
                "phase": "diary-confirmed",
            }
            エラー時は {"error": str}
        """
        try:
            dual = session.get("dual_extraction")
            if dual is None:
                return {
                    "error": "抽出結果がありません。先に extract_diary を実行してください"
                }

            # 1. 卦マッピング
            hexagram_mapping = self._diary_engine.map_to_hexagrams(dual)

            candidates = hexagram_mapping.get("current_candidates", [])
            goal_hexagram = hexagram_mapping.get("goal_hexagram")

            if not candidates:
                return {"error": "卦候補の取得に失敗しました"}

            # candidate_index のバリデーション
            if candidate_index < 0 or candidate_index >= len(candidates):
                candidate_index = 0

            # 2. 本卦と目標卦の番号を取得
            hexagram_a = candidates[candidate_index].get("hexagram_number")
            hexagram_g = goal_hexagram.get("hexagram_number") if goal_hexagram else None

            # 3. ギャップ分析
            gap_analysis = None
            if hexagram_a is not None and hexagram_g is not None:
                gap_analysis = self._gap_engine.analyze(hexagram_a, hexagram_g)

            # 4. 候補にメタデータを付与 (app.py confirm エンドポイントと同一パターン)
            for c in candidates:
                hex_info = hex64_lookup.get(c["hexagram_number"], {})
                c["meaning"] = hex_info.get("meaning", "")
                c["situation"] = hex_info.get("situation", "")
                c["keywords"] = hex_info.get("keywords", [])
                c["archetype"] = hex_info.get("archetype", "")
                c["modern_interpretation"] = hex_info.get(
                    "modern_interpretation", ""
                )

            # 5. セッション更新
            session["hexagram_mapping"] = hexagram_mapping
            session["selected_candidate_index"] = candidate_index
            session["gap_analysis"] = gap_analysis
            session["phase"] = "diary-confirmed"

            return {
                "current_candidates": candidates,
                "goal_hexagram": goal_hexagram,
                "gap_analysis": gap_analysis,
                "phase": "diary-confirmed",
            }

        except Exception as e:
            return {"error": f"確定処理中にエラーが発生しました: {e}"}

    # ------------------------------------------------------------------
    # Phase 3: 5層フィードバック + 変化アドバイス生成
    # ------------------------------------------------------------------

    def generate_change_advice(self, session: dict) -> dict:
        """5層フィードバックと変化アドバイス（Layer 6）を生成する。

        FeedbackEngine.generate() で5層フィードバックを生成し、
        compat_lookup から相性データを追加し、
        ChangeAdviceEngine.generate_advice() で変化アドバイスを生成する。

        Args:
            session: セッション辞書

        Returns:
            {
                "feedback": dict,
                "change_advice": dict,
                "gap_analysis_display": str,
                "selected_candidate": dict,
                "goal_hexagram": dict,
                "phase": "diary-result",
            }
            エラー時は {"error": str}
        """
        try:
            hexagram_mapping = session.get("hexagram_mapping")
            gap_analysis = session.get("gap_analysis")
            dual = session.get("dual_extraction")

            if hexagram_mapping is None:
                return {
                    "error": "卦マッピングがありません。先に confirm_dual を実行してください"
                }

            candidates = hexagram_mapping.get("current_candidates", [])
            goal_hexagram = hexagram_mapping.get("goal_hexagram")
            idx = session.get("selected_candidate_index", 0)

            if idx < 0 or idx >= len(candidates):
                idx = 0

            candidate = candidates[idx]
            hexagram_a = candidate.get("hexagram_number")
            yao = candidate.get("yao", 3)

            # DB ラベルの取得（current_db_labels から）
            current_db = hexagram_mapping.get("current_db_labels", {})
            db_state = current_db.get("db_state", "停滞・閉塞")
            db_action = current_db.get("db_action", "慎重・観察")
            confidence = current_db.get("overall_confidence", 0.7)

            # ----- 1. 5層フィードバック生成 -----
            fb = self._feedback_engine.generate(
                hexagram_a, yao, db_state, db_action, confidence
            )

            # ----- 2. 相性データ追加 (app.py feedback エンドポイントと同一パターン) -----
            zhi_id = (
                fb.get("layer2_direction", {})
                .get("resulting_hexagram", {})
                .get("id")
            )
            if zhi_id and compat_lookup:
                hex_id = candidate["hexagram_number"]
                compat_key = f"{hex_id}-{zhi_id}"
                compat_data = compat_lookup.get(compat_key)
                if compat_data:
                    fb["compatibility"] = {
                        "from_hexagram": hex_id,
                        "to_hexagram": zhi_id,
                        "type": compat_data.get("type", ""),
                        "score": compat_data.get("score", 0),
                        "summary": compat_data.get("summary", ""),
                    }

            # ----- 3. 変化アドバイス生成 (Layer 6) -----
            diary_meta = {}
            if dual:
                diary_meta = dual.get("diary_meta", {})

            hexagram_g = (
                goal_hexagram.get("hexagram_number") if goal_hexagram else None
            )

            change_advice = None
            gap_analysis_display = ""

            if hexagram_g is not None and gap_analysis is not None:
                change_advice = self._advice_engine.generate_advice(
                    hexagram_a=hexagram_a,
                    hexagram_g=hexagram_g,
                    gap_analysis=gap_analysis,
                    diary_meta=diary_meta,
                    yao_position=yao,
                )
                # ギャップ分析のフォーマット済み表示
                gap_analysis_display = self._advice_engine._format_gap_summary(
                    gap_analysis
                )

            # ----- 4. セッション更新 -----
            session["feedback"] = fb
            session["change_advice"] = change_advice
            session["phase"] = "diary-result"

            return {
                "feedback": fb,
                "change_advice": change_advice,
                "gap_analysis_display": gap_analysis_display,
                "selected_candidate": candidate,
                "goal_hexagram": goal_hexagram,
                "phase": "diary-result",
            }

        except Exception as e:
            return {"error": f"アドバイス生成中にエラーが発生しました: {e}"}
