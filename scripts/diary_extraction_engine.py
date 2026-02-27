#!/usr/bin/env python3
"""
日記テキストからの二重抽出エンジン (DiaryExtractionEngine)

日記テキストから「現在の状態(current)」と「理想の状態(ideal)」を同時に抽出し、
ProbabilityMapper を通じて卦候補にマッピングする。

アーキテクチャ:
  1. LLM呼び出し: prompts/diary_dual_extraction.md を使って二重抽出
  2. 確信度評価: current / ideal 両方の確信度を個別評価
  3. フォローアップ: ideal の gap_clarity が低い場合に追加質問を生成
  4. 卦マッピング: current → 本卦候補3件、ideal → 目標卦1件

Usage:
    from diary_extraction_engine import DiaryExtractionEngine

    engine = DiaryExtractionEngine()
    if engine.is_available():
        dual = engine.extract_dual("今日もまた同じ一日だった...")
        if dual:
            hexagrams = engine.map_to_hexagrams(dual)
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

from llm_dialogue import (
    extract_json_from_response,
    _find_closest_key,
    LLMDialogueEngine,
    STATE_PHRASES,
    ACTION_PHRASES,
    TRIGGER_PHRASES,
    PHASE_PHRASES,
    ENERGY_PHRASES,
)
from probability_tables import ProbabilityMapper
from iching_cli import (
    STATE_TO_DB, ACTION_TO_DB, TRIGGER_TO_DB, PHASE_TO_DB, ENERGY_TO_DB,
    CURRENT_STATES, ENERGY_DIRECTIONS, INTENDED_ACTIONS, TRIGGER_NATURES, PHASE_STAGES,
)

__all__ = ["DiaryExtractionEngine"]


# ============================================================
# 5軸キー定義（LLMDialogueEngine と共通）
# ============================================================

_AXIS_KEYS = [
    "current_state", "energy_direction", "intended_action",
    "trigger_nature", "phase_stage",
]

_AXIS_LABELS = {
    "current_state": "現在の状況の核心的な感覚",
    "energy_direction": "エネルギーの方向",
    "intended_action": "実際にやっていること・やりたいこと",
    "trigger_nature": "変化のきっかけ",
    "phase_stage": "変化の進行度合い",
}


# ============================================================
# デモモード用サンプルデータ
# ============================================================

DIARY_DEMO_SCENARIOS = {
    "frustration": {
        "current": {
            "current_state": {"primary": "停滞・閉塞", "confidence": 0.85, "reasoning": "変化の兆しが見えない閉塞状態が記述されている"},
            "energy_direction": {"primary": "停止", "confidence": 0.70, "reasoning": "やろうとしていることが進まず、エネルギーが停滞している"},
            "intended_action": {"primary": "刷新・破壊", "confidence": 0.65, "reasoning": "現状を打破したいという意志が読み取れる"},
            "trigger_nature": {"primary": "漸進的変化", "confidence": 0.70, "reasoning": "急激な変化ではなく日々の不満の蓄積"},
            "phase_stage": {"primary": "展開中期", "confidence": 0.60, "reasoning": "不満が蓄積し分岐点にいる"},
            "domain": "個人",
        },
        "ideal": {
            "current_state": {"primary": "成長・発展", "confidence": 0.70, "reasoning": "停滞の裏返しとして成長を望んでいる"},
            "energy_direction": {"primary": "外向・拡散", "confidence": 0.65, "reasoning": "内に閉じた現状から外に向かいたい欲求"},
            "intended_action": {"primary": "挑戦・冒険", "confidence": 0.60, "reasoning": "現状打破は新しい挑戦への意志を示唆"},
            "trigger_nature": {"primary": "内発的衝動", "confidence": 0.55, "reasoning": "自ら変わりたいという内側からの欲求"},
            "phase_stage": {"primary": "始まり", "confidence": 0.60, "reasoning": "新しいスタートを切りたいという願望"},
            "domain": "個人",
        },
        "diary_meta": {
            "emotional_tone": "negative",
            "gap_clarity": 0.65,
            "key_tension": "現状の停滞と変化への渇望の間で身動きが取れない",
            "domain": "個人",
        },
    },
    "crossroads": {
        "current": {
            "current_state": {"primary": "安定・順調", "confidence": 0.80, "reasoning": "基盤は安定しているが成長の停滞を自覚"},
            "energy_direction": {"primary": "循環・往復", "confidence": 0.75, "reasoning": "2つの選択肢の間で揺れ動いている"},
            "intended_action": {"primary": "選択・判断", "confidence": 0.80, "reasoning": "重要な選択を迫られている最中"},
            "trigger_nature": {"primary": "偶発的機会", "confidence": 0.75, "reasoning": "予期しなかったチャンスがトリガー"},
            "phase_stage": {"primary": "展開中期", "confidence": 0.75, "reasoning": "決断を迫られる分岐点にいる"},
            "domain": "個人",
        },
        "ideal": {
            "current_state": {"primary": "成長・発展", "confidence": 0.75, "reasoning": "成長が見込めない不満の裏返し"},
            "energy_direction": {"primary": "上昇", "confidence": 0.70, "reasoning": "挑戦したい欲求は上昇志向を示す"},
            "intended_action": {"primary": "挑戦・冒険", "confidence": 0.70, "reasoning": "新しい環境での挑戦を望んでいる"},
            "trigger_nature": {"primary": "内発的衝動", "confidence": 0.60, "reasoning": "挑戦したいという内発的欲求"},
            "phase_stage": {"primary": "始まり", "confidence": 0.65, "reasoning": "新しいキャリアの始まりを望んでいる"},
            "domain": "個人",
        },
        "diary_meta": {
            "emotional_tone": "mixed",
            "gap_clarity": 0.80,
            "key_tension": "安定した現状と成長機会の間での選択",
            "domain": "個人",
        },
    },
    "emptiness": {
        "current": {
            "current_state": {"primary": "喪失・空虚", "confidence": 0.80, "reasoning": "目的を見失い時間だけが過ぎる空虚な状態"},
            "energy_direction": {"primary": "停止", "confidence": 0.75, "reasoning": "何も進まずエネルギーが停止している"},
            "intended_action": {"primary": "慎重・観察", "confidence": 0.60, "reasoning": "行動できず結果的に観望状態になっている"},
            "trigger_nature": {"primary": "漸進的変化", "confidence": 0.65, "reasoning": "日々の積み重ねによる緩やかな停滞"},
            "phase_stage": {"primary": "展開中期", "confidence": 0.60, "reasoning": "何かを始めようとしているが分岐点でもある"},
            "domain": "個人",
        },
        "ideal": {
            "current_state": {"primary": "成長・発展", "confidence": 0.60, "reasoning": "先に進みたいという成長欲求"},
            "energy_direction": {"primary": "上昇", "confidence": 0.55, "reasoning": "取り残される不安の裏返しとして上昇を望む"},
            "intended_action": {"primary": "表現・発信", "confidence": 0.55, "reasoning": "独自の価値を発揮したいという表現欲求"},
            "trigger_nature": {"primary": "内発的衝動", "confidence": 0.50, "reasoning": "自分を信じたいという内側からの微かな欲求"},
            "phase_stage": {"primary": "始まり", "confidence": 0.55, "reasoning": "何か新しいことを始めたいという漠然とした願望"},
            "domain": "個人",
        },
        "diary_meta": {
            "emotional_tone": "negative",
            "gap_clarity": 0.55,
            "key_tension": "行動できない自分と独自の価値を発揮したい理想の乖離",
            "domain": "個人",
        },
    },
}

DIARY_DEMO_KEYWORDS = {
    "停滞": "frustration", "動けない": "frustration", "変わらない": "frustration",
    "つまらない": "frustration", "退屈": "frustration", "同じ": "frustration",
    "転職": "crossroads", "選択": "crossroads", "迷": "crossroads",
    "オファー": "crossroads", "決断": "crossroads",
    "空虚": "emptiness", "何もできない": "emptiness", "嫌": "emptiness",
    "取り残され": "emptiness", "意味がない": "emptiness",
}


# ============================================================
# ヘルパー: 単一抽出セットの DB ラベル変換
# ============================================================

def _extraction_to_db_labels_internal(extraction: dict) -> dict:
    """
    5軸抽出結果のprimaryラベルをDB用ラベルに変換する。

    LLMDialogueEngine.extraction_to_db_labels と同等のロジックだが、
    current/ideal の個別の抽出結果（5軸 dict）をそのまま受け取る。

    Args:
        extraction: 5軸の抽出結果 dict
            (current_state, energy_direction, intended_action,
             trigger_nature, phase_stage, domain を含む)

    Returns:
        {
            "db_state": str,
            "db_energy": str or None,
            "db_action": str,
            "db_trigger": str,
            "db_phase": str,
            "ui_state": str,
            "ui_action": str,
            "overall_confidence": float,
            "domain": str,
        }
    """
    confidences = []
    penalty = 0.0

    # current_state
    state_primary = extraction.get("current_state", {}).get("primary", "停滞・閉塞")
    state_key, state_exact = _find_closest_key(state_primary, STATE_TO_DB)
    db_state = STATE_TO_DB[state_key]
    if not state_exact:
        penalty += 0.1
    confidences.append(extraction.get("current_state", {}).get("confidence", 0.5))

    # energy_direction
    energy_primary = extraction.get("energy_direction", {}).get("primary", "循環・往復")
    energy_key, energy_exact = _find_closest_key(energy_primary, ENERGY_TO_DB)
    db_energy = ENERGY_TO_DB[energy_key]
    if not energy_exact:
        penalty += 0.1
    confidences.append(extraction.get("energy_direction", {}).get("confidence", 0.5))

    # intended_action
    action_primary = extraction.get("intended_action", {}).get("primary", "慎重・観察")
    action_key, action_exact = _find_closest_key(action_primary, ACTION_TO_DB)
    db_action = ACTION_TO_DB[action_key]
    if not action_exact:
        penalty += 0.1
    confidences.append(extraction.get("intended_action", {}).get("confidence", 0.5))

    # trigger_nature
    trigger_primary = extraction.get("trigger_nature", {}).get("primary", "漸進的変化")
    trigger_key, trigger_exact = _find_closest_key(trigger_primary, TRIGGER_TO_DB)
    db_trigger = TRIGGER_TO_DB[trigger_key]
    if not trigger_exact:
        penalty += 0.1
    confidences.append(extraction.get("trigger_nature", {}).get("confidence", 0.5))

    # phase_stage
    phase_primary = extraction.get("phase_stage", {}).get("primary", "展開中期")
    phase_key, phase_exact = _find_closest_key(phase_primary, PHASE_TO_DB)
    db_phase = PHASE_TO_DB[phase_key]
    if not phase_exact:
        penalty += 0.1
    confidences.append(extraction.get("phase_stage", {}).get("confidence", 0.5))

    # 全体の確信度（平均 - ペナルティ）
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
    overall = max(0.0, round(avg_conf - penalty, 3))

    return {
        "db_state": db_state,
        "db_energy": db_energy,
        "db_action": db_action,
        "db_trigger": db_trigger,
        "db_phase": db_phase,
        "ui_state": state_key,
        "ui_action": action_key,
        "overall_confidence": overall,
        "domain": extraction.get("domain", "個人"),
    }


# ============================================================
# DiaryExtractionEngine クラス
# ============================================================

class DiaryExtractionEngine:
    """
    日記テキストからの二重抽出エンジン。

    日記テキストから「現在の状態(current)」と「理想の状態(ideal)」を
    同時にLLMで抽出し、ProbabilityMapperを使って卦候補にマッピングする。

    LLMが利用不可能な場合はキーワードマッチによるデモモードで動作する。
    """

    CONFIDENCE_THRESHOLD = 0.60
    GAP_CLARITY_THRESHOLD = 0.5

    def __init__(self):
        self.client = None  # Lazy init
        self._dual_prompt: Optional[str] = None
        self._dialogue_engine = LLMDialogueEngine()
        self._prob_mapper = ProbabilityMapper()

    # ----------------------------------------------------------
    # 初期化・チェック
    # ----------------------------------------------------------

    def is_available(self) -> bool:
        """ANTHROPIC_API_KEY が設定されているか確認する。"""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _ensure_client(self) -> bool:
        """
        Anthropic クライアントを遅延初期化する。

        Returns:
            True: クライアントの初期化に成功
            False: APIキー未設定またはSDK未インストール
        """
        if self.client is not None:
            return True

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return False

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            return True
        except ImportError:
            warnings.warn(
                "anthropic SDK がインストールされていません。"
                "pip install anthropic を実行してください。"
            )
            return False
        except Exception as e:
            warnings.warn(f"Anthropicクライアント初期化エラー: {e}")
            return False

    def _load_prompt(self) -> bool:
        """
        prompts/diary_dual_extraction.md を読み込む。

        Returns:
            True: ファイルの読み込みに成功
            False: ファイルが見つからない
        """
        if self._dual_prompt is not None:
            return True

        prompt_path = os.path.join(_PROJECT_ROOT, "prompts", "diary_dual_extraction.md")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self._dual_prompt = f.read()
            return True
        except FileNotFoundError:
            warnings.warn(f"プロンプトファイルが見つかりません: {prompt_path}")
            return False

    # ----------------------------------------------------------
    # デモモード: キーワードマッチでシナリオ選択
    # ----------------------------------------------------------

    @staticmethod
    def _select_demo_scenario(text: str) -> dict:
        """
        ユーザー入力テキストからキーワードマッチでデモシナリオを選択する。

        Args:
            text: ユーザーの日記テキスト

        Returns:
            DIARY_DEMO_SCENARIOS の1エントリ (current, ideal, diary_meta)
        """
        for keyword, scenario_key in DIARY_DEMO_KEYWORDS.items():
            if keyword in text:
                return DIARY_DEMO_SCENARIOS[scenario_key]
        return DIARY_DEMO_SCENARIOS["frustration"]

    # ----------------------------------------------------------
    # メイン抽出: extract_dual
    # ----------------------------------------------------------

    def extract_dual(self, user_text: str) -> Optional[dict]:
        """
        日記テキストから current + ideal + diary_meta を抽出する。

        LLMが利用可能な場合は diary_dual_extraction.md プロンプトを使って
        LLM呼び出しを行い、利用不可能な場合はデモモードにフォールバックする。

        Args:
            user_text: ユーザーの日記テキスト

        Returns:
            {
                "current": {5軸の抽出結果},
                "ideal": {5軸の抽出結果},
                "diary_meta": {
                    "emotional_tone": str,
                    "gap_clarity": float,
                    "key_tension": str,
                    "domain": str
                }
            }
            失敗時は None。
        """
        if not user_text or not user_text.strip():
            return None

        # LLM が利用不可能な場合はデモモード
        if not self.is_available():
            return self._select_demo_scenario(user_text)

        # LLM 呼び出し
        if not self._ensure_client():
            return self._select_demo_scenario(user_text)

        if not self._load_prompt():
            return self._select_demo_scenario(user_text)

        # プロンプト内の {user_text} プレースホルダーを置換
        prompt_with_input = self._dual_prompt.replace("{user_text}", user_text)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt_with_input},
                ],
            )
            response_text = message.content[0].text
        except Exception as e:
            warnings.warn(f"二重抽出LLM呼び出しエラー: {e}")
            return self._select_demo_scenario(user_text)

        # JSON 抽出
        result = extract_json_from_response(response_text)
        if result is None:
            warnings.warn("二重抽出LLMのレスポンスからJSONを抽出できませんでした。")
            return self._select_demo_scenario(user_text)

        # 必須キーの存在確認
        if "current" not in result or "ideal" not in result:
            warnings.warn("抽出結果に 'current' または 'ideal' が含まれていません。")
            return self._select_demo_scenario(user_text)

        # current と ideal の 5 軸キー存在確認
        for side in ("current", "ideal"):
            for key in _AXIS_KEYS:
                if key not in result[side]:
                    warnings.warn(f"抽出結果の '{side}' に '{key}' が含まれていません。")
                    return self._select_demo_scenario(user_text)

        # diary_meta がなければデフォルト生成
        if "diary_meta" not in result:
            result["diary_meta"] = {
                "emotional_tone": "mixed",
                "gap_clarity": 0.50,
                "key_tension": "",
                "domain": result["current"].get("domain", "個人"),
            }

        return result

    # ----------------------------------------------------------
    # 確信度評価
    # ----------------------------------------------------------

    def _assess_single_extraction(self, extraction: dict) -> dict:
        """
        単一の抽出結果(current または ideal)の確信度を評価する。

        Args:
            extraction: 5軸の抽出結果 dict

        Returns:
            {
                "action": "proceed" | "follow_up" | "broad_follow_up",
                "low_axes": [(axis_label, confidence, reasoning), ...],
                "overall_confidence": float
            }
        """
        low_axes: List[Tuple[str, float, str]] = []
        confidences: List[float] = []

        for key in _AXIS_KEYS:
            axis_data = extraction.get(key, {})
            if isinstance(axis_data, dict):
                conf = axis_data.get("confidence", 0.0)
                reasoning = axis_data.get("reasoning", "")
            else:
                conf = 0.0
                reasoning = ""

            confidences.append(conf)

            if conf < self.CONFIDENCE_THRESHOLD:
                label = _AXIS_LABELS.get(key, key)
                low_axes.append((label, conf, reasoning))

        overall = sum(confidences) / len(confidences) if confidences else 0.0

        if len(low_axes) == 0:
            action = "proceed"
        elif len(low_axes) >= 3:
            action = "broad_follow_up"
        else:
            action = "follow_up"

        return {
            "action": action,
            "low_axes": low_axes,
            "overall_confidence": round(overall, 3),
        }

    def assess_dual_confidence(self, dual_result: dict) -> dict:
        """
        current と ideal 両方の確信度を評価する。

        gap_clarity が GAP_CLARITY_THRESHOLD 未満の場合、
        ideal に関するフォローアップ質問が必要と判定する。

        Args:
            dual_result: extract_dual() の戻り値

        Returns:
            {
                "current_assessment": {action, low_axes, overall_confidence},
                "ideal_assessment": {action, low_axes, overall_confidence},
                "gap_clarity": float,
                "needs_ideal_followup": bool,
                "ideal_followup_question": str or None
            }
        """
        current_assessment = self._assess_single_extraction(
            dual_result.get("current", {})
        )
        ideal_assessment = self._assess_single_extraction(
            dual_result.get("ideal", {})
        )

        gap_clarity = dual_result.get("diary_meta", {}).get("gap_clarity", 0.0)
        needs_followup = gap_clarity < self.GAP_CLARITY_THRESHOLD

        # フォローアップ質問のプレ生成（必要な場合のみ）
        followup_question = None
        if needs_followup:
            followup_question = self._generate_default_ideal_question(dual_result)

        return {
            "current_assessment": current_assessment,
            "ideal_assessment": ideal_assessment,
            "gap_clarity": gap_clarity,
            "needs_ideal_followup": needs_followup,
            "ideal_followup_question": followup_question,
        }

    @staticmethod
    def _generate_default_ideal_question(dual_result: dict) -> str:
        """
        gap_clarity が低い場合のデフォルトフォローアップ質問を生成する。

        LLM を使わず、diary_meta の情報から質問を構築する。

        Args:
            dual_result: extract_dual() の戻り値

        Returns:
            フォローアップ質問文字列
        """
        key_tension = dual_result.get("diary_meta", {}).get("key_tension", "")
        emotional_tone = dual_result.get("diary_meta", {}).get("emotional_tone", "mixed")

        if key_tension:
            return (
                f"日記の中で「{key_tension}」という状況が見えてきました。"
                "もし理想通りになるとしたら、毎日はどんな感じになっていますか？"
            )
        elif emotional_tone == "negative":
            return (
                "今の状況がつらいことは伝わってきます。"
                "もし魔法で何でも変えられるとしたら、何を一番変えたいですか？"
            )
        else:
            return (
                "日記を読ませていただきました。"
                "この先、どんな状態になっていたら「うまくいっている」と感じますか？"
            )

    # ----------------------------------------------------------
    # ideal フォローアップ質問生成 (LLM使用)
    # ----------------------------------------------------------

    def generate_ideal_followup(self, dual_result: dict, user_text: str) -> Optional[str]:
        """
        理想の状態に関するフォローアップ質問をLLMで生成する。

        gap_clarity が GAP_CLARITY_THRESHOLD 未満の場合に呼び出す。
        日記の内容を参照しつつ、望む未来について温かく尋ねる質問を1つ生成する。

        LLM が利用不可能な場合はデフォルトの質問にフォールバックする。

        Args:
            dual_result: extract_dual() の戻り値
            user_text: 元の日記テキスト

        Returns:
            フォローアップ質問文字列。失敗時は None。
        """
        if not self._ensure_client():
            return self._generate_default_ideal_question(dual_result)

        key_tension = dual_result.get("diary_meta", {}).get("key_tension", "不明")
        emotional_tone = dual_result.get("diary_meta", {}).get("emotional_tone", "mixed")

        # current の状態ラベルを取得
        current_state = dual_result.get("current", {}).get(
            "current_state", {}
        ).get("primary", "")

        prompt = f"""以下の日記テキストを読んだ上で、書き手の「理想の状態」「望む未来」について質問を1つだけ生成してください。

## 日記テキスト
{user_text}

## 分析コンテキスト
- 現在の状態: {current_state}
- 感情トーン: {emotional_tone}
- 主要な緊張: {key_tension}

## 質問の条件
- 日記の内容に具体的に触れること
- 「もし理想通りになるとしたら」「もし何でも変えられるとしたら」のような仮定法を使うこと
- 温かく、非判断的であること
- 質問は1つだけ。複数投げないこと
- 日本語で。50-100文字程度
- 質問文のみを出力すること（説明や前置き不要）

## 出力
質問文のみ:"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            question = message.content[0].text.strip()
            # 余計な引用符やマークダウンを除去
            question = question.strip('"').strip("'").strip("`")
            return question
        except Exception as e:
            warnings.warn(f"idealフォローアップLLM呼び出しエラー: {e}")
            return self._generate_default_ideal_question(dual_result)

    # ----------------------------------------------------------
    # ideal フォローアップのマージ
    # ----------------------------------------------------------

    def merge_ideal_followup(
        self, original_dual: dict, followup_answer: str, original_text: str
    ) -> dict:
        """
        フォローアップ回答を追加して再抽出し、ideal 部分をマージする。

        元の日記テキストにフォローアップ回答を付加して再度 extract_dual を実行し、
        新しい ideal の方が確信度が高い軸を採用する。current は変更しない。

        Args:
            original_dual: 元の extract_dual() の結果
            followup_answer: ユーザーの理想に関するフォローアップ回答
            original_text: 元の日記テキスト

        Returns:
            マージされた dual_result（original_dual と同じ構造）
        """
        # 追加テキストを連結して再抽出
        augmented_text = (
            f"{original_text}\n\n"
            f"【理想の状態について】\n{followup_answer}"
        )

        new_dual = self.extract_dual(augmented_text)
        if new_dual is None:
            return original_dual

        # ideal のマージ: 確信度が高い方を採用
        merged_ideal = {}
        original_ideal = original_dual.get("ideal", {})
        new_ideal = new_dual.get("ideal", {})

        for key in _AXIS_KEYS:
            orig_data = original_ideal.get(key, {})
            new_data = new_ideal.get(key, {})

            orig_conf = orig_data.get("confidence", 0.0) if isinstance(orig_data, dict) else 0.0
            new_conf = new_data.get("confidence", 0.0) if isinstance(new_data, dict) else 0.0

            if new_conf > orig_conf:
                merged_ideal[key] = new_data
            else:
                merged_ideal[key] = orig_data

        # domain: 新しい方があればそちら
        merged_ideal["domain"] = (
            new_ideal.get("domain") or original_ideal.get("domain", "個人")
        )

        # 結果を組み立て
        merged = {
            "current": original_dual.get("current", {}),
            "ideal": merged_ideal,
            "diary_meta": {
                **original_dual.get("diary_meta", {}),
                # gap_clarity を更新: 新しい方が高ければ採用
                "gap_clarity": max(
                    original_dual.get("diary_meta", {}).get("gap_clarity", 0.0),
                    new_dual.get("diary_meta", {}).get("gap_clarity", 0.0),
                ),
                # key_tension: 新しいテキストがあればそちらを採用
                "key_tension": (
                    new_dual.get("diary_meta", {}).get("key_tension")
                    or original_dual.get("diary_meta", {}).get("key_tension", "")
                ),
            },
        }

        return merged

    # ----------------------------------------------------------
    # 卦マッピング
    # ----------------------------------------------------------

    def map_to_hexagrams(self, dual_result: dict) -> dict:
        """
        current と ideal の抽出結果を卦候補にマッピングする。

        ProbabilityMapper.get_top_candidates() を2回呼び出す:
        1. current 抽出 -> 本卦候補 3 件
        2. ideal 抽出 -> 目標卦 1 件（top-1 を自動採用）

        Args:
            dual_result: extract_dual() の戻り値

        Returns:
            {
                "current_candidates": [3候補 from ProbMapper],
                "goal_hexagram": {top-1候補 from ProbMapper for ideal},
                "current_db_labels": {db_state, db_action, ...},
                "ideal_db_labels": {db_state, db_action, ...}
            }
        """
        # current の DB ラベル変換
        current_extraction = dual_result.get("current", {})
        current_db = _extraction_to_db_labels_internal(current_extraction)

        # ideal の DB ラベル変換
        ideal_extraction = dual_result.get("ideal", {})
        ideal_db = _extraction_to_db_labels_internal(ideal_extraction)

        # current -> 本卦候補 3 件
        current_result = self._prob_mapper.get_top_candidates(
            current_state=current_db["db_state"],
            intended_action=current_db["db_action"],
            trigger_nature=current_db["db_trigger"],
            phase_stage=current_db["db_phase"],
            energy_direction=current_db["db_energy"],
            n=3,
        )
        current_candidates = current_result.get("candidates", [])

        # ideal -> 目標卦 1 件
        ideal_result = self._prob_mapper.get_top_candidates(
            current_state=ideal_db["db_state"],
            intended_action=ideal_db["db_action"],
            trigger_nature=ideal_db["db_trigger"],
            phase_stage=ideal_db["db_phase"],
            energy_direction=ideal_db["db_energy"],
            n=1,
        )
        ideal_candidates = ideal_result.get("candidates", [])
        goal_hexagram = ideal_candidates[0] if ideal_candidates else None

        return {
            "current_candidates": current_candidates,
            "goal_hexagram": goal_hexagram,
            "current_db_labels": current_db,
            "ideal_db_labels": ideal_db,
        }

    # ----------------------------------------------------------
    # 自然言語要約
    # ----------------------------------------------------------

    def summarize_dual(self, dual_result: dict) -> dict:
        """
        current と ideal の両方について自然言語の要約を生成する。

        LLMDialogueEngine.summarize_for_user と同様のテンプレートベースの
        要約を、current と ideal それぞれに対して生成する。

        Args:
            dual_result: extract_dual() の戻り値

        Returns:
            {
                "current_summary": str,
                "ideal_summary": str,
                "gap_summary": str
            }
        """
        # current の要約: LLMDialogueEngine.summarize_for_user を流用
        current_extraction = dual_result.get("current", {})
        current_summary = self._dialogue_engine.summarize_for_user(current_extraction)

        # ideal の要約: 同じフレーズ辞書を使うが文体を変える
        ideal_extraction = dual_result.get("ideal", {})
        ideal_summary = self._summarize_ideal(ideal_extraction)

        # gap の要約: diary_meta.key_tension を使用
        key_tension = dual_result.get("diary_meta", {}).get("key_tension", "")
        gap_summary = key_tension if key_tension else "現在と理想の間にギャップがあります"

        return {
            "current_summary": current_summary,
            "ideal_summary": ideal_summary,
            "gap_summary": gap_summary,
        }

    @staticmethod
    def _summarize_ideal(extraction: dict) -> str:
        """
        ideal の抽出結果から自然言語要約を生成する。

        current とは異なり「望んでいる」「向かいたい」などの表現を使う。

        Args:
            extraction: ideal の 5 軸抽出結果

        Returns:
            要約テキスト（複数行）
        """
        state_label = extraction.get("current_state", {}).get("primary", "")
        action_label = extraction.get("intended_action", {}).get("primary", "")
        energy_label = extraction.get("energy_direction", {}).get("primary", "")
        phase_label = extraction.get("phase_stage", {}).get("primary", "")

        # 状態フレーズ（ideal 用に表現を調整）
        state_ideal_phrases = {
            "どん底・危機": "厳しい試練を経て強くなること",
            "停滞・閉塞": "静かに力を蓄えること",
            "不安定・混乱": "変化の中で道を見つけること",
            "安定・順調": "安定した状態を手に入れること",
            "成長・発展": "成長し前に進んでいくこと",
            "頂点・過剰": "ピークに立つこと",
            "衰退・下降": "一度立ち止まり原点に帰ること",
            "転換期": "大きく方向を変えること",
            "萌芽・準備": "新しいことを始めること",
            "回復途上": "回復し元の力を取り戻すこと",
            "対立・分裂": "対立を解消し統合すること",
            "依存・束縛": "束縛から自由になること",
            "喪失・空虚": "新たな意味を見出すこと",
            "未知・探索": "未知の領域を切り開くこと",
            "忍耐・持久": "忍耐が実を結ぶこと",
        }

        state_p = state_ideal_phrases.get(state_label, "望む状態を手に入れること")
        action_p = ACTION_PHRASES.get(action_label, "何かをしたいと感じている")
        energy_p = ENERGY_PHRASES.get(energy_label, "エネルギーが動いている")
        phase_p = PHASE_PHRASES.get(phase_label, "変化の途中にいる")

        return (
            f"  あなたが望んでいるのは、{state_p}です。\n"
            f"  {action_p}という気持ちがあり、{energy_p}ようです。\n"
            f"  変化としては、{phase_p}段階を望んでいます。"
        )


# ============================================================
# スタンドアロン実行用
# ============================================================

def main():
    """スタンドアロン実行時のエントリポイント"""
    import textwrap

    engine = DiaryExtractionEngine()

    print()
    print("=" * 50)
    print("  日記テキスト二重抽出エンジン (Demo)")
    print("=" * 50)
    print()

    if not engine.is_available():
        print("  ! ANTHROPIC_API_KEY 未設定 -> デモモードで動作します")
        print()

    print("  日記テキストを入力してください。")
    print("  入力が終わったらEnterを2回押してください。")
    print()

    lines = []
    empty_count = 0
    try:
        while True:
            line = input()
            if line.strip() == "":
                empty_count += 1
                if empty_count >= 1 and lines:
                    break
            else:
                empty_count = 0
                lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print("\n  中断しました。")
        return

    user_text = "\n".join(lines).strip()
    if not user_text:
        print("  入力がありません。")
        return

    # --- 二重抽出 ---
    print()
    print("  抽出中...")

    dual = engine.extract_dual(user_text)
    if dual is None:
        print("  ! 抽出に失敗しました。")
        return

    # --- 確信度評価 ---
    assessment = engine.assess_dual_confidence(dual)

    # --- 要約表示 ---
    summaries = engine.summarize_dual(dual)

    print()
    print("  --- 現在の状態 ---")
    print(summaries["current_summary"])
    print()
    print("  --- 理想の状態 ---")
    print(summaries["ideal_summary"])
    print()
    print(f"  --- ギャップ ---")
    print(f"  {summaries['gap_summary']}")
    print()

    # 確信度
    cur_conf = assessment["current_assessment"]["overall_confidence"] * 100
    ideal_conf = assessment["ideal_assessment"]["overall_confidence"] * 100
    gap_cl = assessment["gap_clarity"] * 100
    print(f"  現在の確度: {cur_conf:.0f}%  |  理想の確度: {ideal_conf:.0f}%  |  ギャップ明確度: {gap_cl:.0f}%")

    # フォローアップが必要な場合
    if assessment["needs_ideal_followup"]:
        print()
        print(f"  [!] 理想の状態がまだ不明確です。")
        followup_q = assessment["ideal_followup_question"]
        if followup_q:
            print(f"  質問: {followup_q}")

    # --- 卦マッピング ---
    print()
    print("  卦マッピング中...")

    hexagrams = engine.map_to_hexagrams(dual)

    print()
    print("  --- 本卦候補（現在の状態から）---")
    for c in hexagrams["current_candidates"]:
        pct = c["probability"] * 100
        print(f"  {c['rank']}. {c['hexagram_name']}（{c['hexagram_number']}）"
              f" [確率: {pct:.1f}%]")

    goal = hexagrams["goal_hexagram"]
    if goal:
        print()
        print("  --- 目標卦（理想の状態から）---")
        pct = goal["probability"] * 100
        print(f"  {goal['hexagram_name']}（{goal['hexagram_number']}）"
              f" [確率: {pct:.1f}%]")

    print()
    print("  --- JSON出力 ---")
    output = {
        "dual_extraction": dual,
        "assessment": {
            "current": assessment["current_assessment"],
            "ideal": assessment["ideal_assessment"],
            "gap_clarity": assessment["gap_clarity"],
            "needs_ideal_followup": assessment["needs_ideal_followup"],
        },
        "hexagram_mapping": {
            "current_candidates": hexagrams["current_candidates"],
            "goal_hexagram": hexagrams["goal_hexagram"],
            "current_db_labels": hexagrams["current_db_labels"],
            "ideal_db_labels": hexagrams["ideal_db_labels"],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  中断しました。")
        sys.exit(0)
