#!/usr/bin/env python3
"""
LLMDialogueEngine のユニットテスト

全てのLLM呼び出し（Anthropic API）をモックし、
対話フロー・抽出・マージ・DB変換・要約のロジックをテストする。
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# --- パス設定 ---
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from llm_dialogue import (
    LLMDialogueEngine,
    extract_json_from_response,
    _find_closest_key,
    STATE_PHRASES, ACTION_PHRASES, TRIGGER_PHRASES, PHASE_PHRASES, ENERGY_PHRASES,
)

from iching_cli import (
    STATE_TO_DB, ACTION_TO_DB, TRIGGER_TO_DB, PHASE_TO_DB, ENERGY_TO_DB,
    CURRENT_STATES, ENERGY_DIRECTIONS, INTENDED_ACTIONS, TRIGGER_NATURES, PHASE_STAGES,
)


# ============================================================
# モックヘルパー
# ============================================================

def make_mock_response(content: str):
    """Anthropic Message オブジェクトのモックを作成"""
    mock_msg = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.text = content
    mock_msg.content = [mock_content_block]
    return mock_msg


def make_extraction_result(
    state="停滞・閉塞", state_conf=0.85,
    energy="外向・拡散", energy_conf=0.75,
    action="刷新・破壊", action_conf=0.90,
    trigger="内発的衝動", trigger_conf=0.80,
    phase="展開中期", phase_conf=0.75,
    domain="企業"
):
    """テスト用の抽出結果を生成"""
    return {
        "current_state": {"primary": state, "confidence": state_conf, "reasoning": "テスト"},
        "energy_direction": {"primary": energy, "confidence": energy_conf, "reasoning": "テスト"},
        "intended_action": {"primary": action, "confidence": action_conf, "reasoning": "テスト"},
        "trigger_nature": {"primary": trigger, "confidence": trigger_conf, "reasoning": "テスト"},
        "phase_stage": {"primary": phase, "confidence": phase_conf, "reasoning": "テスト"},
        "domain": domain,
    }


# ============================================================
# 1. extract_json_from_response テスト (5ケース)
# ============================================================

class TestExtractJson:
    def test_direct_json(self):
        """JSONがそのまま返された場合"""
        data = {"current_state": {"primary": "停滞・閉塞", "confidence": 0.85}}
        result = extract_json_from_response(json.dumps(data, ensure_ascii=False))
        assert result["current_state"]["primary"] == "停滞・閉塞"

    def test_code_block_json(self):
        """```json ... ``` で囲まれた場合"""
        raw = '以下が結果です:\n```json\n{"current_state": {"primary": "危機"}}\n```\n'
        result = extract_json_from_response(raw)
        assert result["current_state"]["primary"] == "危機"

    def test_embedded_json(self):
        """テキスト中に { } が埋め込まれた場合"""
        raw = '分析結果: {"current_state": {"primary": "成長"}} 以上です。'
        result = extract_json_from_response(raw)
        assert result["current_state"]["primary"] == "成長"

    def test_no_json(self):
        """JSONが含まれていない場合"""
        result = extract_json_from_response("これはJSONではありません")
        assert result is None

    def test_none_input(self):
        """None入力"""
        result = extract_json_from_response(None)
        assert result is None


# ============================================================
# 2. _find_closest_key テスト (4ケース)
# ============================================================

class TestFindClosestKey:
    def test_exact_match(self):
        """完全一致"""
        key, exact = _find_closest_key("停滞・閉塞", STATE_TO_DB)
        assert key == "停滞・閉塞"
        assert exact is True

    def test_partial_match(self):
        """部分一致"""
        key, exact = _find_closest_key("停滞", STATE_TO_DB)
        assert "停滞" in key
        assert exact is False

    def test_fuzzy_match(self):
        """difflibによる近似一致"""
        key, exact = _find_closest_key("どん底危機", STATE_TO_DB)
        assert key == "どん底・危機"
        assert exact is False

    def test_fallback(self):
        """全く一致しない場合"""
        key, exact = _find_closest_key("xyzabc", STATE_TO_DB)
        assert key in STATE_TO_DB  # 何かしらのキーが返る
        assert exact is False


# ============================================================
# 3. assess_confidence テスト (3ケース)
# ============================================================

class TestAssessConfidence:
    def test_all_high_confidence(self):
        """全軸が閾値以上 → proceed"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result(
            state_conf=0.85, energy_conf=0.75,
            action_conf=0.90, trigger_conf=0.80, phase_conf=0.75
        )
        result = engine.assess_confidence(extraction)
        assert result["action"] == "proceed"
        assert len(result["low_axes"]) == 0

    def test_one_low_confidence(self):
        """1軸が閾値未満 → follow_up"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result(
            state_conf=0.45, energy_conf=0.75,
            action_conf=0.90, trigger_conf=0.80, phase_conf=0.75
        )
        result = engine.assess_confidence(extraction)
        assert result["action"] == "follow_up"
        assert len(result["low_axes"]) == 1

    def test_many_low_confidence(self):
        """3軸以上が閾値未満 → broad_follow_up"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result(
            state_conf=0.30, energy_conf=0.40,
            action_conf=0.50, trigger_conf=0.80, phase_conf=0.75
        )
        result = engine.assess_confidence(extraction)
        assert result["action"] == "broad_follow_up"
        assert len(result["low_axes"]) == 3


# ============================================================
# 4. merge_extractions テスト (2ケース)
# ============================================================

class TestMergeExtractions:
    def test_takes_higher_confidence(self):
        """各軸で確信度が高い方を採用する"""
        engine = LLMDialogueEngine()
        original = make_extraction_result(state_conf=0.40, action_conf=0.90)
        new = make_extraction_result(
            state="どん底・危機", state_conf=0.85,
            action="守る・維持", action_conf=0.60
        )
        merged = engine.merge_extractions(original, new)
        # state: newの方が高い → "どん底・危機"
        assert merged["current_state"]["primary"] == "どん底・危機"
        assert merged["current_state"]["confidence"] == 0.85
        # action: originalの方が高い → "刷新・破壊"
        assert merged["intended_action"]["primary"] == "刷新・破壊"
        assert merged["intended_action"]["confidence"] == 0.90

    def test_preserves_domain(self):
        """domainは新しい方を優先"""
        engine = LLMDialogueEngine()
        original = make_extraction_result(domain="企業")
        new = make_extraction_result(domain="個人")
        merged = engine.merge_extractions(original, new)
        assert merged["domain"] == "個人"


# ============================================================
# 5. extraction_to_db_labels テスト (3ケース)
# ============================================================

class TestExtractionToDbLabels:
    def test_exact_match_all_axes(self):
        """全軸が完全一致する場合"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result()
        result = engine.extraction_to_db_labels(extraction)
        assert result["db_state"] == "停滞・閉塞"       # STATE_TO_DB["停滞・閉塞"]
        assert result["db_action"] == "刷新・破壊"       # ACTION_TO_DB["刷新・破壊"]
        assert result["db_trigger"] == "意図的決断"      # TRIGGER_TO_DB["内発的衝動"]
        assert result["db_energy"] == "expanding"         # ENERGY_TO_DB["外向・拡散"]
        assert result["db_phase"] == "危機・転換点"      # PHASE_TO_DB["展開中期"]
        assert result["ui_state"] == "停滞・閉塞"
        assert result["ui_action"] == "刷新・破壊"
        assert result["domain"] == "企業"

    def test_no_penalty_on_exact(self):
        """完全一致時はペナルティなし"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result(
            state_conf=0.80, energy_conf=0.80,
            action_conf=0.80, trigger_conf=0.80, phase_conf=0.80
        )
        result = engine.extraction_to_db_labels(extraction)
        assert result["overall_confidence"] == 0.8

    def test_all_valid_states_map(self):
        """CURRENT_STATESの全15個がSTATE_TO_DBに対応"""
        for state in CURRENT_STATES:
            assert state in STATE_TO_DB, f"'{state}' がSTATE_TO_DBに見つからない"


# ============================================================
# 6. summarize_for_user テスト (8ケース)
# ============================================================

class TestSummarizeForUser:
    def test_generates_natural_text(self):
        """自然言語のテキストが生成される"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result()
        summary = engine.summarize_for_user(extraction)
        assert "動きが止まって先が見えない" in summary  # STATE_PHRASES
        assert "一度壊して新しくしたい" in summary       # ACTION_PHRASES

    def test_no_category_labels_shown(self):
        """カテゴリラベルがそのまま表示されない"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result()
        summary = engine.summarize_for_user(extraction)
        # 内部キー名は表示されないはず
        assert "current_state" not in summary
        assert "energy_direction" not in summary

    def test_all_states_have_phrases(self):
        """全15の状態にフレーズが定義されている"""
        for state in CURRENT_STATES:
            assert state in STATE_PHRASES, f"'{state}' のフレーズが未定義"

    def test_all_actions_have_phrases(self):
        """全22のアクションにフレーズが定義されている"""
        for action in INTENDED_ACTIONS:
            assert action in ACTION_PHRASES, f"'{action}' のフレーズが未定義"

    def test_all_triggers_have_phrases(self):
        """全8のトリガーにフレーズが定義されている"""
        for trigger in TRIGGER_NATURES:
            assert trigger in TRIGGER_PHRASES, f"'{trigger}' のフレーズが未定義"

    def test_all_phases_have_phrases(self):
        """全6のフェーズにフレーズが定義されている"""
        for phase in PHASE_STAGES:
            assert phase in PHASE_PHRASES, f"'{phase}' のフレーズが未定義"

    def test_all_energies_have_phrases(self):
        """全6のエネルギー方向にフレーズが定義されている"""
        for energy in ENERGY_DIRECTIONS:
            assert energy in ENERGY_PHRASES, f"'{energy}' のフレーズが未定義"


# ============================================================
# 7. LLM呼び出しモックテスト (5ケース)
# ============================================================

class TestExtractAxesWithMock:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("llm_dialogue.LLMDialogueEngine._ensure_client")
    @patch("llm_dialogue.LLMDialogueEngine._load_prompts")
    def test_successful_extraction(self, mock_load, mock_client):
        """正常な抽出"""
        mock_load.return_value = True
        mock_client.return_value = True

        engine = LLMDialogueEngine()
        engine._extraction_prompt = "テスト用プロンプト {user_text}"
        engine.client = MagicMock()

        extraction_data = make_extraction_result()
        engine.client.messages.create.return_value = make_mock_response(
            json.dumps(extraction_data, ensure_ascii=False)
        )

        result = engine.extract_axes("会社が停滞しています")
        assert result is not None
        assert result["current_state"]["primary"] == "停滞・閉塞"

    def test_not_available_without_api_key(self):
        """APIキー未設定 → is_available() == False"""
        with patch.dict(os.environ, {}, clear=True):
            # ANTHROPIC_API_KEYを確実に消す
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                engine = LLMDialogueEngine()
                assert engine.is_available() is False

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_available_with_api_key(self):
        """APIキー設定済み → is_available() == True"""
        engine = LLMDialogueEngine()
        assert engine.is_available() is True


class TestGenerateFollowup:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_generates_question(self):
        """フォローアップ質問が生成される"""
        engine = LLMDialogueEngine()
        engine.client = MagicMock()
        engine._followup_prompt = "テスト {low_confidence_details} {user_text}"

        engine.client.messages.create.return_value = make_mock_response(
            "今の状況で一番つらいと感じていることは何ですか？"
        )

        low_axes = [("現在の状況の核心的な感覚", 0.45, "複数解釈可能")]
        question = engine.generate_followup_question(
            make_extraction_result(), low_axes, "会社が停滞"
        )
        assert question is not None
        assert "？" in question or "?" in question


# ============================================================
# 8. is_available / _load_prompts テスト (3ケース)
# ============================================================

class TestLoadPrompts:
    def test_loads_existing_prompts(self):
        """実在するプロンプトファイルが読み込める"""
        engine = LLMDialogueEngine()
        result = engine._load_prompts()
        assert result is True
        assert engine._extraction_prompt is not None
        assert engine._few_shot_examples is not None
        assert engine._followup_prompt is not None

    def test_few_shot_examples_are_list(self):
        """few_shot_examplesがリストで読み込まれる"""
        engine = LLMDialogueEngine()
        engine._load_prompts()
        assert isinstance(engine._few_shot_examples, list)
        assert len(engine._few_shot_examples) > 0

    def test_prompts_cached(self):
        """2回目の呼び出しはキャッシュを使う"""
        engine = LLMDialogueEngine()
        engine._load_prompts()
        prompt1 = engine._extraction_prompt
        engine._load_prompts()
        prompt2 = engine._extraction_prompt
        assert prompt1 is prompt2  # 同一オブジェクト


# ============================================================
# 9. E2Eパイプラインテスト (1ケース)
# ============================================================

class TestEndToEndPipeline:
    def test_extraction_to_feedback(self):
        """抽出結果 → DB変換 → ProbabilityMapper → FeedbackEngine の完全パイプライン"""
        engine = LLMDialogueEngine()
        extraction = make_extraction_result(
            state="停滞・閉塞",
            action="刷新・破壊",
        )

        db_labels = engine.extraction_to_db_labels(extraction)

        # ProbabilityMapper に渡せることを確認
        from probability_tables import ProbabilityMapper
        mapper = ProbabilityMapper()
        result = mapper.get_top_candidates(
            current_state=db_labels["db_state"],
            intended_action=db_labels["db_action"],
            trigger_nature=db_labels["db_trigger"],
            phase_stage=db_labels["db_phase"],
            energy_direction=db_labels["db_energy"],
            n=3,
        )
        assert "candidates" in result
        assert len(result["candidates"]) > 0

        # FeedbackEngine に渡せることを確認
        from feedback_engine import FeedbackEngine
        candidate = result["candidates"][0]
        fb_engine = FeedbackEngine()
        text = fb_engine.generate_text(
            candidate["hexagram_number"],
            candidate.get("yao", 3),
            db_labels["ui_state"],
            db_labels["ui_action"],
            mapping_confidence=db_labels["overall_confidence"],
        )
        assert len(text) > 100
        assert "現在地" in text or "LAYER" in text or "━" in text
