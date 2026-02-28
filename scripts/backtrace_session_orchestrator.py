#!/usr/bin/env python3
"""
バックトレースモードのセッション全体を統括するオーケストレーター

BacktraceEngine, GapAnalysisEngine, FeedbackEngine, ReverseLookup を
統合し、目標卦の選択から逆算ロードマップ生成までの全フローを管理する。

フロー:
    create_session() -> set_goal() -> confirm_goal() -> describe_current()
    -> analyze() -> generate_roadmap()

Usage:
    from backtrace_session_orchestrator import BacktraceSessionOrchestrator

    orch = BacktraceSessionOrchestrator()
    session = orch.create_session()
    result = orch.set_goal(session, method="hexagram", value=11)
    result = orch.confirm_goal(session)
    result = orch.describe_current(session, current_hex=12,
                                   current_state="停滞・閉塞",
                                   action_type="慎重・観察")
    result = orch.analyze(session)
    result = orch.generate_roadmap(session)
"""

import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

# --- パス設定 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from backtrace_engine import BacktraceEngine
from gap_analysis_engine import GapAnalysisEngine
from feedback_engine import FeedbackEngine
from reverse_lookup import ReverseLookup
from llm_dialogue import extract_json_from_response

# ---------------------------------------------------------------------------
# hex64_lookup (候補表示用): diary_session_orchestrator.py と同一パターン
# ---------------------------------------------------------------------------
hex64_lookup: Dict[int, dict] = {}
_hex64_path = os.path.join(_PROJECT_ROOT, "data", "diagnostic", "hexagram_64.json")
if os.path.isfile(_hex64_path):
    with open(_hex64_path, encoding="utf-8") as _f:
        _hex64_raw = json.load(_f)
    for _name, _info in _hex64_raw["hexagrams"].items():
        # キー名（例: "乾為天"）を name フィールドとして保持
        _info_copy = dict(_info)
        if not _info_copy.get("name"):
            _info_copy["name"] = _name
        hex64_lookup[_info_copy["number"]] = _info_copy

# ---------------------------------------------------------------------------
# compat_lookup (相性データ): diary_session_orchestrator.py と同一パターン
# ---------------------------------------------------------------------------
_compat_path = os.path.join(
    _PROJECT_ROOT, "data", "reference", "hexagram_compatibility_lookup.json"
)
if os.path.exists(_compat_path):
    with open(_compat_path, encoding="utf-8") as _f:
        compat_lookup: dict = json.load(_f)
else:
    compat_lookup = {}

# ---------------------------------------------------------------------------
# LLM クライアント: change_advice_engine.py と同一パターン
# ---------------------------------------------------------------------------
try:
    import anthropic
    _client = anthropic.Anthropic()
    _LLM_AVAILABLE = True
except Exception:
    _client = None
    _LLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# ゴール状態マッピングヒューリスティック
# ---------------------------------------------------------------------------
_KEYWORD_TO_STATE: Dict[str, str] = {
    "安定": "安定・平和",
    "平和": "安定・平和",
    "成長": "安定成長・順調",
    "順調": "安定成長・順調",
    "成功": "V字回復・大成功",
    "回復": "V字回復・大成功",
    "飛躍": "V字回復・大成功",
    "変革": "再編成・変革",
    "改革": "再編成・変革",
    "刷新": "再編成・変革",
    "創造": "V字回復・大成功",
    "調和": "安定・平和",
    "拡大": "安定成長・順調",
    "発展": "安定成長・順調",
}
_DEFAULT_GOAL_STATE = "安定成長・順調"

# ---------------------------------------------------------------------------
# デモロードマップ（LLM不使用時フォールバック）
# ---------------------------------------------------------------------------
DEMO_ROADMAP: Dict[str, object] = {
    "overview": "現在の状態から目標に向かうルートが見えています。段階的に進むことが重要です",
    "phase_1": {
        "title": "準備フェーズ",
        "description": "現在の状態を正確に認識し、手放すべきものを特定する",
        "duration_hint": "1-2週間",
        "key_action": "今の状態で「維持したいもの」と「手放したいもの」を分ける",
    },
    "phase_2": {
        "title": "移行フェーズ",
        "description": "最も効果的な行動パターンを小さく試し始める",
        "duration_hint": "2-4週間",
        "key_action": "推奨された行動を1つ選び、小さく実験する",
    },
    "phase_3": {
        "title": "定着フェーズ",
        "description": "新しいパターンを日常に組み込み、安定させる",
        "duration_hint": "1-2ヶ月",
        "key_action": "変化を振り返り、次のステップを見定める",
    },
    "immediate_action": "今日の時点で、目標の状態をまず想像してみる。その感覚を体で感じる",
    "caution": "過去事例の分布に基づく参考情報です。あなたの状況は唯一無二です",
}

# ---------------------------------------------------------------------------
# ロードマップ必須キー
# ---------------------------------------------------------------------------
_ROADMAP_REQUIRED_KEYS = [
    "overview",
    "phase_1",
    "phase_2",
    "phase_3",
    "immediate_action",
    "caution",
]

# ---------------------------------------------------------------------------
# ロードマップ禁止ワード
# ---------------------------------------------------------------------------
_ROADMAP_PROHIBITED_WORDS = ["必ず", "絶対に", "間違いなく", "確実に"]


# ===========================================================================
# BacktraceSessionOrchestrator
# ===========================================================================


class BacktraceSessionOrchestrator:
    """バックトレースモードのセッション全体を統括するオーケストレーター"""

    def __init__(self) -> None:
        self._backtrace_engine = BacktraceEngine()
        self._gap_engine = GapAnalysisEngine()
        self._feedback_engine = FeedbackEngine()
        self._reverse_lookup = ReverseLookup()

    # ------------------------------------------------------------------
    # セッション作成
    # ------------------------------------------------------------------

    def create_session(self) -> dict:
        """新しいバックトレースセッションを作成して返す。

        Returns:
            セッション状態辞書。session_id は含まない（app.py 側で管理する）。
        """
        return {
            "mode": "backtrace",
            "phase": "goal-select",
            "goal_method": None,        # "theme" | "hexagram" | "text"
            "goal_hex": None,           # int (1-64)
            "goal_state": None,         # str
            "goal_summary": None,       # str (display text)
            "current_hex": None,        # int
            "current_state": None,      # str
            "current_action": None,     # str
            "current_text": None,       # user's situation text
            "backtrace_result": None,   # full_backtrace output
            "feedback": None,           # 5-layer feedback
            "roadmap": None,            # LLM-generated roadmap
            "gap_analysis": None,       # gap analysis result
        }

    # ------------------------------------------------------------------
    # ゴール状態推定ヘルパー
    # ------------------------------------------------------------------

    def _infer_goal_state(self, keywords: List[str]) -> str:
        """キーワードリストからゴール状態を推定する。

        Args:
            keywords: hex64_lookup の keywords フィールド (list of str)

        Returns:
            推定されたゴール状態文字列。マッチなしの場合はデフォルト値。
        """
        for kw in keywords:
            for key, state in _KEYWORD_TO_STATE.items():
                if key in kw:
                    return state
        return _DEFAULT_GOAL_STATE

    # ------------------------------------------------------------------
    # Phase 1: ゴール設定
    # ------------------------------------------------------------------

    def set_goal(self, session: dict, method: str, value) -> dict:
        """目標卦をメソッドに応じて設定する。

        Args:
            session: create_session() で作成したセッション辞書
            method: "hexagram" | "theme" | "text"
            value: method="hexagram" の場合は int (1-64)、
                   method="theme"/"text" の場合は str

        Returns:
            method="hexagram" の場合:
                {goal_hex, goal_name, goal_metadata, goal_state, phase: "goal-set"}
            method="theme"/"text" の場合:
                {candidates: [{hex_num, name, metadata, confidence}], phase: "goal-select"}
            エラー時は {"error": str}
        """
        session["goal_method"] = method

        # -------------------------------------------------------
        # method = "hexagram": 直接卦番号で設定
        # -------------------------------------------------------
        if method == "hexagram":
            try:
                hex_num = int(value)
            except (TypeError, ValueError):
                return {"error": f"卦番号が不正です: {value}"}

            if not (1 <= hex_num <= 64):
                return {"error": f"卦番号は1〜64の範囲で指定してください: {hex_num}"}

            hex_info = hex64_lookup.get(hex_num, {})
            keywords: List[str] = hex_info.get("keywords", [])
            goal_state = self._infer_goal_state(keywords)
            goal_name = hex_info.get("name", hex_info.get("japanese_name", str(hex_num)))
            goal_summary = f"{goal_name}（第{hex_num}卦）— {goal_state}"

            # セッション更新
            session["goal_hex"] = hex_num
            session["goal_state"] = goal_state
            session["goal_summary"] = goal_summary
            session["phase"] = "goal-set"

            return {
                "goal_hex": hex_num,
                "goal_name": goal_name,
                "goal_metadata": hex_info,
                "goal_state": goal_state,
                "phase": "goal-set",
            }

        # -------------------------------------------------------
        # method = "theme": テーマキーワードで候補を返す
        # -------------------------------------------------------
        if method == "theme":
            if not isinstance(value, str) or not value.strip():
                return {"error": "テーマ文字列が空です"}

            try:
                lookup_result = self._reverse_lookup.lookup_by_theme(value)
            except Exception as e:
                return {"error": f"テーマ検索中にエラーが発生しました: {e}"}

            candidates: List[dict] = []

            if lookup_result:
                # suggested_hexagrams は "乾為天 (1)" のような文字列リスト
                for hex_str in lookup_result.suggested_hexagrams[:3]:
                    hex_num = self._parse_hex_num_from_string(hex_str)
                    if hex_num is None:
                        continue
                    meta = hex64_lookup.get(hex_num, {})
                    candidates.append({
                        "hex_num": hex_num,
                        "name": meta.get(
                            "name", meta.get("japanese_name", str(hex_num))
                        ),
                        "metadata": meta,
                        "confidence": lookup_result.confidence,
                    })

            if not candidates:
                return {
                    "candidates": [],
                    "phase": "goal-select",
                    "message": f"「{value}」に対応する卦候補が見つかりませんでした。"
                               "別のキーワードをお試しください",
                }

            return {"candidates": candidates, "phase": "goal-select"}

        # -------------------------------------------------------
        # method = "text": 自然文フレーズから候補を返す
        # -------------------------------------------------------
        if method == "text":
            if not isinstance(value, str) or not value.strip():
                return {"error": "テキストが空です"}

            try:
                state, trigram, matches = self._reverse_lookup.lookup_by_phrase(value)
            except Exception as e:
                return {"error": f"フレーズ検索中にエラーが発生しました: {e}"}

            # trigram から候補卦を収集（hex64_lookup を全件スキャン）
            candidates = self._candidates_from_trigram(trigram, limit=3)

            if not candidates:
                return {
                    "candidates": [],
                    "phase": "goal-select",
                    "message": f"「{value}」に対応する卦候補が見つかりませんでした。"
                               "別の表現をお試しください",
                }

            return {"candidates": candidates, "phase": "goal-select"}

        return {"error": f"不明なメソッドです: {method}。'hexagram'/'theme'/'text' のいずれかを指定してください"}

    # ------------------------------------------------------------------
    # Phase 1 ヘルパー
    # ------------------------------------------------------------------

    def _parse_hex_num_from_string(self, hex_str: str) -> Optional[int]:
        """卦番号を含む文字列から卦番号を取り出す。

        対応形式:
            - '乾為天 (1)'  → 1  (括弧内の数字)
            - '29_坎為水'   → 29 (アンダースコア前の数字)
            - '1'           → 1  (数字のみ)
        """
        import re
        # 括弧内の数字を優先
        m = re.search(r"\((\d+)\)", hex_str)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 64:
                    return n
            except ValueError:
                pass
        # アンダースコア区切り形式: "29_坎為水"
        m = re.match(r"^(\d+)_", hex_str.strip())
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 64:
                    return n
            except ValueError:
                pass
        # 文字列全体が数字なら直接変換
        try:
            n = int(hex_str.strip())
            if 1 <= n <= 64:
                return n
        except ValueError:
            pass
        return None

    def _candidates_from_trigram(self, trigram: str, limit: int = 3) -> List[dict]:
        """八卦名（例: "乾"）から hex64_lookup を検索して候補リストを返す。"""
        results: List[dict] = []
        if not trigram or trigram == "不明":
            return results
        for hex_num, info in hex64_lookup.items():
            upper = info.get("upper_trigram", "")
            lower = info.get("lower_trigram", "")
            name = info.get("name", info.get("japanese_name", ""))
            if trigram in upper or trigram in lower or trigram in name:
                results.append({
                    "hex_num": hex_num,
                    "name": name,
                    "metadata": info,
                    "confidence": 0.5,
                })
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Phase 1.5: ゴール確定
    # ------------------------------------------------------------------

    def confirm_goal(self, session: dict, hex_num: Optional[int] = None) -> dict:
        """目標卦の選択を確定する。

        set_goal() で candidates が返された場合に、ユーザーが選択した卦番号を
        ここで渡す。hex_num が指定されない場合はセッションにすでに設定済みの
        goal_hex をそのまま使用する。

        Args:
            session: セッション辞書
            hex_num: 確定する卦番号 (1-64)。None の場合はセッション値を使用。

        Returns:
            {goal_hex, goal_name, goal_state, goal_summary, phase: "current-describe"}
            エラー時は {"error": str}
        """
        if hex_num is not None:
            # hex_num で上書き: set_goal(method="hexagram") と同処理
            result = self.set_goal(session, method="hexagram", value=hex_num)
            if "error" in result:
                return result

        goal_hex = session.get("goal_hex")
        if goal_hex is None:
            return {
                "error": "目標卦が設定されていません。先に set_goal を実行してください"
            }

        hex_info = hex64_lookup.get(goal_hex, {})
        goal_name = hex_info.get("name", hex_info.get("japanese_name", str(goal_hex)))
        goal_state = session.get("goal_state", _DEFAULT_GOAL_STATE)
        goal_summary = session.get(
            "goal_summary",
            f"{goal_name}（第{goal_hex}卦）— {goal_state}",
        )

        # フェーズ更新
        session["goal_summary"] = goal_summary
        session["phase"] = "current-describe"

        return {
            "goal_hex": goal_hex,
            "goal_name": goal_name,
            "goal_state": goal_state,
            "goal_summary": goal_summary,
            "phase": "current-describe",
        }

    # ------------------------------------------------------------------
    # Phase 2: 現在地の設定
    # ------------------------------------------------------------------

    def describe_current(
        self,
        session: dict,
        text: Optional[str] = None,
        current_hex: Optional[int] = None,
        current_state: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> dict:
        """現在地（本卦・状態・行動）を設定する。

        2つのモードがある:
        - Direct モード: current_hex + current_state + action_type を直接渡す
        - Text モード: text を渡し、LLM抽出で5軸を推定する（未実装時はエラー）

        Args:
            session: セッション辞書
            text: ユーザーの状況テキスト（Text モード用）
            current_hex: 現在の卦番号 1-64（Direct モード用）
            current_state: 現在の状態文字列（Direct モード用）
            action_type: 現在の行動タイプ文字列（Direct モード用）

        Returns:
            {current_hex, current_state, current_action, phase: "analyzing"}
            エラー時は {"error": str}
        """
        # -------------------------------------------------------
        # Direct モード
        # -------------------------------------------------------
        if current_hex is not None:
            try:
                current_hex = int(current_hex)
            except (TypeError, ValueError):
                return {"error": f"current_hex が不正です: {current_hex}"}

            if not (1 <= current_hex <= 64):
                return {
                    "error": f"current_hex は1〜64の範囲で指定してください: {current_hex}"
                }

            if not current_state:
                return {"error": "current_state が指定されていません"}

            action = action_type or "慎重・観察"

            session["current_hex"] = current_hex
            session["current_state"] = current_state
            session["current_action"] = action
            session["phase"] = "analyzing"

            return {
                "current_hex": current_hex,
                "current_state": current_state,
                "current_action": action,
                "phase": "analyzing",
            }

        # -------------------------------------------------------
        # Text モード
        # -------------------------------------------------------
        if text:
            session["current_text"] = text

            try:
                extracted = self._extract_from_text(text)
            except Exception as e:
                return {"error": f"テキスト抽出中にエラーが発生しました: {e}"}

            if extracted is None:
                return {
                    "error": "テキストから現在地を抽出できませんでした。"
                             "current_hex と current_state を直接指定してください"
                }

            ex_hex = extracted.get("hexagram_number")
            ex_state = extracted.get("before_state", "停滞・閉塞")
            ex_action = extracted.get("action_type", "慎重・観察")

            if ex_hex is None:
                return {
                    "error": "テキストから卦番号を特定できませんでした。"
                             "current_hex を直接指定してください"
                }

            session["current_hex"] = ex_hex
            session["current_state"] = ex_state
            session["current_action"] = ex_action
            session["phase"] = "analyzing"

            return {
                "current_hex": ex_hex,
                "current_state": ex_state,
                "current_action": ex_action,
                "phase": "analyzing",
            }

        return {
            "error": "text か current_hex のどちらかを指定してください"
        }

    # ------------------------------------------------------------------
    # Phase 2 ヘルパー: テキストからの抽出
    # ------------------------------------------------------------------

    def _extract_from_text(self, text: str) -> Optional[dict]:
        """LLM を使ってテキストから現在地を抽出する。

        LLM が利用不可の場合は None を返す。
        """
        if not _LLM_AVAILABLE or _client is None:
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        prompt = (
            "以下のテキストから、ユーザーの現在の状況を易経の言語で抽出してください。\n\n"
            f"テキスト:\n{text}\n\n"
            "以下のJSON形式で回答してください（コードブロック不要、JSONのみ）:\n"
            "{\n"
            '  "hexagram_number": <1-64の整数>,\n'
            '  "before_state": "<状態文字列>",\n'
            '  "action_type": "<行動タイプ文字列>"\n'
            "}\n\n"
            "before_state は次のいずれかから選んでください: "
            "どん底・危機, 停滞・閉塞, 不安定・混乱, 安定・順調, 成長・発展, "
            "頂点・過剰, 衰退・下降, 転換期, 萌芽・準備\n"
            "hexagram_number はテキストの状況に最も近い卦番号（1〜64）を選んでください。"
        )

        try:
            message = _client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
        except Exception as e:
            warnings.warn(f"LLM呼び出しエラー（extract_from_text）: {e}")
            return None

        return extract_json_from_response(response_text)

    # ------------------------------------------------------------------
    # Phase 3: 分析実行
    # ------------------------------------------------------------------

    def analyze(self, session: dict) -> dict:
        """フルバックトレース分析を実行する。

        BacktraceEngine.full_backtrace, GapAnalysisEngine.analyze,
        FeedbackEngine.generate を統合し、R1-R5の逆算フィードバック層を生成する。

        Args:
            session: セッション辞書

        Returns:
            {
                backtrace_result, gap_analysis,
                feedback_layers: {r1, r2, r3, r4, r5},
                quality_gates, summary, phase: "result"
            }
            エラー時は {"error": str}
        """
        current_hex = session.get("current_hex")
        current_state = session.get("current_state")
        goal_hex = session.get("goal_hex")
        goal_state = session.get("goal_state")

        if current_hex is None:
            return {
                "error": "現在の卦が設定されていません。先に describe_current を実行してください"
            }
        if goal_hex is None:
            return {
                "error": "目標卦が設定されていません。先に set_goal/confirm_goal を実行してください"
            }
        if not current_state:
            return {"error": "現在の状態が設定されていません"}
        if not goal_state:
            return {"error": "目標の状態が設定されていません"}

        try:
            # ----- 1. フルバックトレース -----
            backtrace_result = self._backtrace_engine.full_backtrace(
                current_hex=current_hex,
                current_state=current_state,
                goal_hex=goal_hex,
                goal_state=goal_state,
            )

            # ----- 2. ギャップ分析 -----
            gap_analysis = self._gap_engine.analyze(current_hex, goal_hex)

            # ----- 3. 現在卦の5層フィードバック -----
            current_action = session.get("current_action", "慎重・観察")
            feedback = self._feedback_engine.generate(
                hexagram_number=current_hex,
                yao_position=3,         # デフォルト中爻
                before_state=current_state,
                action_type=current_action,
                mapping_confidence=0.7,
            )

            # ----- 4. R1-R5 逆算フィードバック層の生成 -----
            feedback_layers = self._build_reverse_feedback_layers(
                backtrace_result=backtrace_result,
                gap_analysis=gap_analysis,
                goal_hex=goal_hex,
                current_hex=current_hex,
            )

            # ----- 5. 相性データを feedback に追加（diary_session_orchestrator と同一パターン） -----
            zhi_id = (
                feedback.get("layer2_direction", {})
                .get("resulting_hexagram", {})
                .get("id")
            )
            if zhi_id and compat_lookup:
                compat_key = f"{current_hex}-{zhi_id}"
                compat_data = compat_lookup.get(compat_key)
                if compat_data:
                    feedback["compatibility"] = {
                        "from_hexagram": current_hex,
                        "to_hexagram": zhi_id,
                        "type": compat_data.get("type", ""),
                        "score": compat_data.get("score", 0),
                        "summary": compat_data.get("summary", ""),
                    }

            # ----- 6. セッション更新 -----
            session["backtrace_result"] = backtrace_result
            session["gap_analysis"] = gap_analysis
            session["feedback"] = feedback
            session["phase"] = "result"

            return {
                "backtrace_result": backtrace_result,
                "gap_analysis": gap_analysis,
                "feedback_layers": feedback_layers,
                "quality_gates": backtrace_result.get("quality_gates", {}),
                "summary": backtrace_result.get("summary", {}),
                "phase": "result",
            }

        except Exception as e:
            return {"error": f"分析処理中にエラーが発生しました: {e}"}

    # ------------------------------------------------------------------
    # Phase 3 ヘルパー: R1-R5 逆算フィードバック層
    # ------------------------------------------------------------------

    def _build_reverse_feedback_layers(
        self,
        backtrace_result: dict,
        gap_analysis: dict,
        goal_hex: int,
        current_hex: int,
    ) -> dict:
        """バックトレース結果から R1-R5 の逆算フィードバック層を構築する。

        Args:
            backtrace_result: BacktraceEngine.full_backtrace() の出力
            gap_analysis: GapAnalysisEngine.analyze() の出力
            goal_hex: 目標卦番号
            current_hex: 現在卦番号

        Returns:
            {r1, r2, r3, r4, r5} の辞書
        """
        goal_info = hex64_lookup.get(goal_hex, {})
        goal_name = goal_info.get("name", goal_info.get("japanese_name", str(goal_hex)))

        l2 = backtrace_result.get("l2_state", {})
        l3 = backtrace_result.get("l3_action", {})
        recommended_routes = backtrace_result.get("recommended_routes", [])

        # ----- R1: 目標地点 -----
        # rev_after_hex から目標卦に到達した事例を取得
        l1 = backtrace_result.get("l1_yao", {})
        sources_reaching_goal = l1.get("sources_that_reach_goal", [])

        r1 = {
            "label": "目標地点",
            "goal_hex": goal_hex,
            "goal_name": goal_name,
            "goal_metadata": {
                "meaning": goal_info.get("meaning", ""),
                "situation": goal_info.get("situation", ""),
                "keywords": goal_info.get("keywords", []),
                "archetype": goal_info.get("archetype", ""),
                "modern_interpretation": goal_info.get("modern_interpretation", ""),
            },
            "cases_that_reached_goal": sources_reaching_goal[:5],
            "goal_state": backtrace_result.get("summary", {}).get("goal_state", ""),
        }

        # ----- R2: ギャップ構造 -----
        r2 = {
            "label": "ギャップ構造",
            "hamming_distance": gap_analysis.get("hamming_distance", 0),
            "changing_lines": gap_analysis.get("changing_lines", []),
            "difficulty": gap_analysis.get("difficulty", "medium"),
            "structural_relationship": gap_analysis.get("structural_relationship", ""),
            "trigram_changes": {
                "upper_changes": gap_analysis.get("upper_trigram_changes", ""),
                "lower_changes": gap_analysis.get("lower_trigram_changes", ""),
            },
            "intermediate_paths": gap_analysis.get("intermediate_paths", []),
        }

        # ----- R3: 推奨ルート（上位3件） -----
        top_routes = []
        for i, route in enumerate(recommended_routes[:3]):
            route_data = route.get("route", {})
            ci = route.get("confidence_interval", {})
            top_routes.append({
                "rank": i + 1,
                "title": route.get("title", f"ルート{i + 1}"),
                "score": route.get("score", 0.0),
                "labels": route.get("labels", []),
                "steps": route_data.get("steps", []),
                "step_count": route_data.get("step_count", 0),
                "total_success_rate": route_data.get("total_success_rate", 0.0),
                "confidence_interval": ci,
            })

        r3 = {
            "label": "推奨ルート",
            "routes": top_routes,
            "route_count": len(top_routes),
            "confidence_note": l2.get("confidence_note", ""),
        }

        # ----- R4: 行動パターン -----
        recommended_actions = l3.get("action_recommendations", [])
        state_actions = l2.get("recommended_actions", [])
        case_examples = l3.get("pattern_suggestions", [])

        r4 = {
            "label": "行動パターン",
            "recommended_actions_l3": recommended_actions[:5],
            "recommended_actions_l2": state_actions[:3],
            "case_examples": case_examples[:3],
            "direct_pair_stats": l3.get("direct_pair_stats", {}),
        }

        # ----- R5: 注意点・問い -----
        compat_warning = ""
        compat_key = f"{current_hex}-{goal_hex}"
        if compat_lookup:
            compat_entry = compat_lookup.get(compat_key)
            if compat_entry:
                compat_type = compat_entry.get("type", "")
                compat_score = compat_entry.get("score", 0)
                compat_summary = compat_entry.get("summary", "")
                if compat_score < 50:
                    compat_warning = (
                        f"現在卦と目標卦の相性は「{compat_type}」（スコア: {compat_score}）。"
                        f"{compat_summary}"
                    )

        confidence_level = backtrace_result.get("summary", {}).get("confidence_level", "low")
        confidence_notes = {
            "high": "事例が十分にあり、信頼度は高いです",
            "medium": "一定の事例数があります。傾向として参考にしてください",
            "low": "事例数が少ないため、参考情報としてご覧ください",
            "very_low": "事例が非常に少ないため、統計的傾向の読み取りには限界があります",
        }

        reflective_question = (
            f"「{goal_info.get('archetype', goal_name)}」の状態を既に体で感じているとしたら、"
            "今の状況は何を伝えていますか？"
        )

        r5 = {
            "label": "注意点・問い",
            "compatibility_warning": compat_warning,
            "confidence_level": confidence_level,
            "confidence_note": confidence_notes.get(confidence_level, ""),
            "rq1_reference_only": backtrace_result.get("quality_gates", {}).get(
                "rq1_reference_only", False
            ),
            "rq4_has_alternative_route": backtrace_result.get("quality_gates", {}).get(
                "rq4_has_alternative_route", True
            ),
            "reflective_question": reflective_question,
        }

        return {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5}

    # ------------------------------------------------------------------
    # Phase 4: ロードマップ生成（LLM）
    # ------------------------------------------------------------------

    def generate_roadmap(self, session: dict) -> dict:
        """LLM を使って自然言語のロードマップを生成する。

        prompts/backtrace_roadmap.txt テンプレートにバックトレースデータを
        注入し、LLM でロードマップ JSON を生成する。
        LLM が利用不可の場合はデモロードマップを返す。

        Args:
            session: セッション辞書（analyze() 実行済みであること）

        Returns:
            {roadmap: dict, phase: "result"}
            エラー時は {"error": str}
        """
        backtrace_result = session.get("backtrace_result")
        if backtrace_result is None:
            return {
                "error": "バックトレース結果がありません。先に analyze を実行してください"
            }

        # プロンプトテンプレートの読み込み
        prompt_path = os.path.join(_PROJECT_ROOT, "prompts", "backtrace_roadmap.txt")
        prompt_template: Optional[str] = None
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, encoding="utf-8") as _f:
                    prompt_template = _f.read()
            except IOError as e:
                warnings.warn(f"バックトレースロードマップテンプレートの読み込みに失敗: {e}")

        # LLM が利用不可 or テンプレートなし → デモロードマップ
        if not _LLM_AVAILABLE or _client is None or not os.environ.get("ANTHROPIC_API_KEY"):
            roadmap = DEMO_ROADMAP.copy()
            session["roadmap"] = roadmap
            return {"roadmap": roadmap, "phase": "result"}

        # テンプレートなし → デモロードマップ
        if prompt_template is None:
            roadmap = DEMO_ROADMAP.copy()
            session["roadmap"] = roadmap
            return {"roadmap": roadmap, "phase": "result"}

        # プロンプト構築
        try:
            prompt = self._build_roadmap_prompt(prompt_template, session, backtrace_result)
        except Exception as e:
            warnings.warn(f"ロードマッププロンプト構築中にエラー: {e}")
            roadmap = DEMO_ROADMAP.copy()
            session["roadmap"] = roadmap
            return {"roadmap": roadmap, "phase": "result"}

        # LLM 呼び出し
        try:
            message = _client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
        except Exception as e:
            warnings.warn(f"LLM呼び出しエラー（generate_roadmap）: {e}")
            roadmap = DEMO_ROADMAP.copy()
            session["roadmap"] = roadmap
            return {"roadmap": roadmap, "phase": "result"}

        # JSON 抽出
        roadmap = extract_json_from_response(response_text)
        if roadmap is None:
            warnings.warn("ロードマップ: LLMレスポンスからJSONを抽出できませんでした。デモにフォールバック")
            roadmap = DEMO_ROADMAP.copy()
            session["roadmap"] = roadmap
            return {"roadmap": roadmap, "phase": "result"}

        # バリデーション（必須キー + 禁止ワード）
        is_valid, issues = self._validate_roadmap(roadmap)
        if not is_valid:
            warnings.warn(
                f"ロードマップが品質ゲートを通過しませんでした: {'; '.join(issues)}。"
                "デモにフォールバック"
            )
            roadmap = DEMO_ROADMAP.copy()

        session["roadmap"] = roadmap
        return {"roadmap": roadmap, "phase": "result"}

    # ------------------------------------------------------------------
    # Phase 4 ヘルパー
    # ------------------------------------------------------------------

    def _build_roadmap_prompt(
        self,
        template: str,
        session: dict,
        backtrace_result: dict,
    ) -> str:
        """プロンプトテンプレートにデータを注入する。

        テンプレート中の {placeholder} をバックトレースデータで置換する。
        テンプレートが存在しない場合はデフォルトプロンプトを返す。
        """
        summary = backtrace_result.get("summary", {})
        l2 = backtrace_result.get("l2_state", {})
        l3 = backtrace_result.get("l3_action", {})
        recommended_routes = backtrace_result.get("recommended_routes", [])
        gap = session.get("gap_analysis", {})

        # 各プレースホルダー値を整形
        goal_hex = session.get("goal_hex", "")
        goal_state = session.get("goal_state", "")
        goal_info = hex64_lookup.get(int(goal_hex) if goal_hex else 0, {})
        goal_name = goal_info.get("name", goal_info.get("japanese_name", str(goal_hex)))

        current_hex = session.get("current_hex", "")
        current_state = session.get("current_state", "")
        current_info = hex64_lookup.get(int(current_hex) if current_hex else 0, {})
        current_name = current_info.get(
            "name", current_info.get("japanese_name", str(current_hex))
        )

        top_route = recommended_routes[0] if recommended_routes else {}
        top_route_data = top_route.get("route", {})
        top_actions = [
            a.get("action_type", "")
            for a in l3.get("action_recommendations", [])[:3]
        ]

        replacements = {
            "{current_hex}": str(current_hex),
            "{current_hex_name}": current_name,
            "{current_state}": str(current_state),
            "{goal_hex}": str(goal_hex),
            "{goal_hex_name}": goal_name,
            "{goal_state}": str(goal_state),
            "{hamming_distance}": str(gap.get("hamming_distance", "")),
            "{difficulty}": str(gap.get("difficulty", "")),
            "{primary_route_score}": str(summary.get("primary_route_score", "")),
            "{confidence_level}": str(summary.get("confidence_level", "")),
            "{top_route_steps}": str(top_route_data.get("steps", [])),
            "{top_route_success_rate}": str(
                top_route_data.get("total_success_rate", "")
            ),
            "{recommended_actions}": ", ".join(top_actions),
            "{confidence_note}": l2.get("confidence_note", ""),
            "{case_count}": str(l2.get("case_count", 0)),
        }

        prompt = template
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)

        return prompt

    def _validate_roadmap(self, roadmap: dict) -> Tuple[bool, List[str]]:
        """ロードマップの品質ゲートを検証する。

        Args:
            roadmap: LLM 生成のロードマップ辞書

        Returns:
            (is_valid, issues_list)
        """
        issues: List[str] = []

        # 必須キーチェック
        for key in _ROADMAP_REQUIRED_KEYS:
            if key not in roadmap:
                issues.append(f"必須キー '{key}' が欠落しています")

        # 禁止ワードチェック
        roadmap_text = json.dumps(roadmap, ensure_ascii=False)
        for word in _ROADMAP_PROHIBITED_WORDS:
            if word in roadmap_text:
                issues.append(f"禁止ワード '{word}' が含まれています")

        return (len(issues) == 0, issues)


# ===========================================================================
# CLI — テスト用エントリポイント
# ===========================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("BacktraceSessionOrchestrator テスト実行")
    print("=" * 70)

    orch = BacktraceSessionOrchestrator()

    # ------------------------------------------------------------------
    # Step 1: セッション作成
    # ------------------------------------------------------------------
    session = orch.create_session()
    print(f"\n[1] create_session() -> phase={session['phase']}")

    # ------------------------------------------------------------------
    # Step 2: ゴール設定（直接卦番号: 地天泰 = 第11卦）
    # ------------------------------------------------------------------
    print("\n[2] set_goal(method='hexagram', value=11)")
    result = orch.set_goal(session, method="hexagram", value=11)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  goal_hex={result['goal_hex']}, goal_name={result.get('goal_name')}")
        print(f"  goal_state={result['goal_state']}")
        print(f"  phase={result['phase']}")

    # ------------------------------------------------------------------
    # Step 3: ゴール確定
    # ------------------------------------------------------------------
    print("\n[3] confirm_goal()")
    result = orch.confirm_goal(session)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  goal_summary={result['goal_summary']}")
        print(f"  phase={result['phase']}")

    # ------------------------------------------------------------------
    # Step 4: 現在地設定（天地否 = 第12卦、停滞・閉塞）
    # ------------------------------------------------------------------
    print("\n[4] describe_current(current_hex=12, current_state='停滞・閉塞', action_type='慎重・観察')")
    result = orch.describe_current(
        session,
        current_hex=12,
        current_state="停滞・閉塞",
        action_type="慎重・観察",
    )
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  current_hex={result['current_hex']}, current_state={result['current_state']}")
        print(f"  current_action={result['current_action']}, phase={result['phase']}")

    # ------------------------------------------------------------------
    # Step 5: 分析実行
    # ------------------------------------------------------------------
    print("\n[5] analyze()")
    result = orch.analyze(session)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        summary = result.get("summary", {})
        print(
            f"  {summary.get('current_hex_name')} + 「{summary.get('current_state')}」"
            f" → {summary.get('goal_hex_name')} + 「{summary.get('goal_state')}」"
        )
        print(f"  primary_route_score: {summary.get('primary_route_score')}")
        print(f"  confidence_level: {summary.get('confidence_level')}")
        print(f"  alternative_count: {summary.get('alternative_count')}")

        qg = result.get("quality_gates", {})
        print(f"\n  == 品質ゲート ==")
        for k, v in qg.items():
            print(f"    {k}: {v}")

        fb_layers = result.get("feedback_layers", {})
        r1 = fb_layers.get("r1", {})
        r2 = fb_layers.get("r2", {})
        r3 = fb_layers.get("r3", {})
        r5 = fb_layers.get("r5", {})
        print(f"\n  == R1: 目標地点 ==")
        print(f"    goal_name={r1.get('goal_name')}, archetype={r1.get('goal_metadata', {}).get('archetype')}")
        print(f"\n  == R2: ギャップ構造 ==")
        print(
            f"    hamming={r2.get('hamming_distance')}, difficulty={r2.get('difficulty')}"
        )
        print(f"\n  == R3: 推奨ルート ==")
        for route in r3.get("routes", []):
            print(
                f"    ルート{route['rank']}: score={route['score']:.4f}, "
                f"success_rate={route['total_success_rate']:.2%}, steps={route['step_count']}"
            )
        print(f"\n  == R5: 注意点・問い ==")
        print(f"    confidence_level={r5.get('confidence_level')}")
        print(f"    reflective_question={r5.get('reflective_question')}")

    # ------------------------------------------------------------------
    # Step 6: ロードマップ生成（LLM なしはデモにフォールバック）
    # ------------------------------------------------------------------
    print("\n[6] generate_roadmap()")
    result = orch.generate_roadmap(session)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        roadmap = result.get("roadmap", {})
        print(f"  overview: {roadmap.get('overview', '')[:60]}...")
        ph1 = roadmap.get("phase_1", {})
        print(f"  phase_1.title: {ph1.get('title')}")
        print(f"  phase_1.duration_hint: {ph1.get('duration_hint')}")
        print(f"  immediate_action: {roadmap.get('immediate_action', '')[:60]}...")
        print(f"  phase={result['phase']}")

    # ------------------------------------------------------------------
    # テーマ検索のテスト
    # ------------------------------------------------------------------
    print("\n[THEME] set_goal(method='theme', value='成長')")
    session2 = orch.create_session()
    result2 = orch.set_goal(session2, method="theme", value="成長")
    if "error" in result2:
        print(f"  ERROR: {result2['error']}")
    else:
        candidates = result2.get("candidates", [])
        print(f"  候補数: {len(candidates)}")
        for c in candidates:
            print(f"    - 第{c['hex_num']}卦 {c.get('name')} (confidence={c.get('confidence')})")

    print("\n" + "=" * 70)
    print("全テスト完了")
    print("=" * 70)
