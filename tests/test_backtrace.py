#!/usr/bin/env python3
"""
逆算エンジン (BacktraceEngine) テスト

全テストは ANTHROPIC_API_KEY を unset した「デモモード」で動作する。
LLM呼び出しは一切行わず、モックも不要。
"""

import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtrace_engine import BacktraceEngine, VALID_SCALES, _wilson_score_interval, _sanitize_text, _build_number_definition, _VOCABULARY_MAP
from backtrace_session_orchestrator import (
    BacktraceSessionOrchestrator,
    DEMO_ROADMAP,
    _ROADMAP_REQUIRED_KEYS,
    _ROADMAP_PROHIBITED_WORDS,
)


# ============================================================
# 1. BacktraceEngine — 初期化テスト
# ============================================================

class TestBacktraceEngineInit:
    """BacktraceEngine の初期化を検証する。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_initialization(self, engine):
        """エンジンが正常に初期化でき、6つのインデックスと依存エンジンが存在すること。"""
        assert engine._rev_yao is not None
        assert engine._rev_after_hex is not None
        assert engine._rev_after_state is not None
        assert engine._rev_outcome_action is not None
        assert engine._rev_pattern_after is not None
        assert engine._rev_hex_pair_stats is not None
        assert engine._gap_engine is not None
        assert engine._case_engine is not None

    def test_reverse_indices_loaded(self, engine):
        """各逆引きインデックスが空でないこと。"""
        assert len(engine._rev_yao) > 0, "rev_yao is empty"
        assert len(engine._rev_after_hex) > 0, "rev_after_hex is empty"
        assert len(engine._rev_after_state) > 0, "rev_after_state is empty"
        assert len(engine._rev_outcome_action) > 0, "rev_outcome_action is empty"
        assert len(engine._rev_pattern_after) > 0, "rev_pattern_after is empty"
        assert len(engine._rev_hex_pair_stats) > 0, "rev_hex_pair_stats is empty"


# ============================================================
# 2. BacktraceEngine — L1 爻レベル逆算
# ============================================================

class TestReverseYao:
    """L1: reverse_yao のテスト。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_identical_hexagram(self, engine):
        """同一卦 -> hamming=0, changing_lines=[], direct_yao_path=False。"""
        result = engine.reverse_yao(1, 1)
        assert result["hamming_distance"] == 0
        assert result["changing_lines"] == []
        assert result["direct_yao_path"] is False

    def test_single_yao_change(self, engine):
        """乾(1)->天風姤(44) -> hamming=1, changing_lines=[1], direct_yao_path=True。"""
        result = engine.reverse_yao(1, 44)
        assert result["hamming_distance"] == 1
        assert result["changing_lines"] == [1]
        assert result["direct_yao_path"] is True
        assert result["direct_yao_position"] == 1

    def test_full_inversion(self, engine):
        """乾(1)->坤(2) -> hamming=6, structural_relationship='cuo_gua'。"""
        result = engine.reverse_yao(1, 2)
        assert result["hamming_distance"] == 6
        assert result["structural_relationship"] == "cuo_gua"

    def test_return_structure(self, engine):
        """戻り値に全ての期待キーが含まれること。"""
        result = engine.reverse_yao(1, 11)
        expected_keys = {
            "current_hex", "goal_hex",
            "hamming_distance", "changing_lines",
            "structural_relationship",
            "direct_yao_path", "direct_yao_position",
            "difficulty",
            "intermediate_suggestions",
            "sources_that_reach_goal",
            "current_is_source",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_sources_that_reach_goal(self, engine):
        """sources_that_reach_goal がリストで、各エントリに source_hex_id があること。"""
        result = engine.reverse_yao(1, 44)
        sources = result["sources_that_reach_goal"]
        assert isinstance(sources, list)
        for entry in sources:
            assert "source_hex_id" in entry

    def test_current_is_source(self, engine):
        """乾(1)->天風姤(44) で current_is_source=True。"""
        result = engine.reverse_yao(1, 44)
        assert result["current_is_source"] is True

    def test_boundary_values(self, engine):
        """卦番号1と64が正常に処理されること。"""
        result_1 = engine.reverse_yao(1, 64)
        assert result_1["current_hex"] == 1
        assert result_1["goal_hex"] == 64

        result_64 = engine.reverse_yao(64, 1)
        assert result_64["current_hex"] == 64
        assert result_64["goal_hex"] == 1


# ============================================================
# 3. BacktraceEngine — L2 状態レベル逆算
# ============================================================

class TestReverseState:
    """L2: reverse_state のテスト。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_return_structure(self, engine):
        """戻り値に全ての期待キーが含まれること。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        expected_keys = {
            "current_state", "goal_state",
            "goal_reachability",
            "before_state_distribution",
            "recommended_actions",
            "case_count",
            "confidence_note",
        }
        assert expected_keys.issubset(result.keys())

    def test_known_state_with_cases(self, engine):
        """'V字回復・大成功' のように事例がある状態で case_count > 0。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        assert result["case_count"] > 0

    def test_unknown_state_zero_cases(self, engine):
        """存在しない状態名 -> case_count=0, goal_reachability=0.0。"""
        result = engine.reverse_state("停滞・閉塞", "存在しない架空の状態XYZ")
        assert result["case_count"] == 0
        assert result["goal_reachability"] == 0.0

    def test_goal_reachability_range(self, engine):
        """0.0 <= goal_reachability <= 1.0。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        assert 0.0 <= result["goal_reachability"] <= 1.0

    def test_recommended_actions_sorted(self, engine):
        """composite_score 降順ソートされていること。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        actions = result["recommended_actions"]
        if len(actions) >= 2:
            scores = [a["composite_score"] for a in actions]
            assert scores == sorted(scores, reverse=True)

    def test_recommended_actions_limit(self, engine):
        """最大5件であること。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        assert len(result["recommended_actions"]) <= 5

    def test_confidence_note_varies_by_count(self, engine):
        """事例数が多い/少ないで異なる confidence_note が返ること。"""
        # 多い事例: V字回復・大成功 (721件)
        result_many = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        # 少ない事例: 存在しない状態
        result_few = engine.reverse_state("停滞・閉塞", "存在しない状態XYZ")
        assert result_many["confidence_note"] != result_few["confidence_note"]


# ============================================================
# 4. BacktraceEngine — L3 行動レベル逆算
# ============================================================

class TestReverseAction:
    """L3: reverse_action のテスト。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_return_structure(self, engine):
        """戻り値に routes, direct_pair_stats, pattern_suggestions, action_recommendations が含まれること。"""
        result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
        assert "routes" in result
        assert "direct_pair_stats" in result
        assert "pattern_suggestions" in result
        assert "action_recommendations" in result

    def test_routes_are_list(self, engine):
        """routes がリストであること。"""
        result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
        assert isinstance(result["routes"], list)

    def test_direct_pair_stats_fields(self, engine):
        """direct_pair_stats に pair_key, total_count, success_rate, top_actions, outcomes が含まれること。"""
        result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
        dps = result["direct_pair_stats"]
        assert "pair_key" in dps
        assert "total_count" in dps
        assert "success_rate" in dps
        assert "top_actions" in dps
        assert "outcomes" in dps

    def test_action_recommendations_sorted(self, engine):
        """score 降順ソートされていること。"""
        result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
        recs = result["action_recommendations"]
        if len(recs) >= 2:
            scores = [r["score"] for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_action_recommendations_limit(self, engine):
        """最大5件であること。"""
        result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
        assert len(result["action_recommendations"]) <= 5

    def test_pattern_suggestions_for_known_state(self, engine):
        """既知の状態 '安定・平和' でパターン提案が返ること。"""
        result = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和")
        # pattern_suggestions はゴール状態にマッチするパターンがある場合のみ返る
        # 安定・平和 は事例数1110件なのでパターンがある可能性が高い
        assert isinstance(result["pattern_suggestions"], list)


# ============================================================
# 5. BacktraceEngine — フルバックトレース
# ============================================================

class TestFullBacktrace:
    """full_backtrace の統合テスト。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_return_structure(self, engine):
        """戻り値に l1_yao, l2_state, l3_action, recommended_routes, quality_gates, summary が含まれること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert "l1_yao" in result
        assert "l2_state" in result
        assert "l3_action" in result
        assert "recommended_routes" in result
        assert "quality_gates" in result
        assert "summary" in result

    def test_summary_fields(self, engine):
        """summary に必要なフィールドが含まれること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        summary = result["summary"]
        expected_keys = {
            "primary_route_score", "alternative_count",
            "confidence_level",
            "current_hex", "goal_hex",
        }
        assert expected_keys.issubset(summary.keys())

    def test_quality_gates_all_present(self, engine):
        """rq1-rq8 の全8項目が存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        qg = result["quality_gates"]
        expected_rq_keys = {
            "rq1_reference_only",
            "rq2_low_success_rate",
            "rq3_no_deterministic_words",
            "rq4_has_alternative_route",
            "rq5_confidence_interval_computed",
            "rq6_fallback_used",
            "rq7_contradictory_routes_excluded",
            "rq8_scale_specified",
        }
        assert expected_rq_keys.issubset(qg.keys()), (
            f"Missing quality gates: {expected_rq_keys - qg.keys()}"
        )

    def test_rq3_no_deterministic_words(self, engine):
        """rq3_no_deterministic_words=True（決定論的表現排除）。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["quality_gates"]["rq3_no_deterministic_words"] is True

    def test_rq4_has_alternative(self, engine):
        """rq4_has_alternative_route=True（代替ルートが常に1つ以上）。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["quality_gates"]["rq4_has_alternative_route"] is True

    def test_rq5_confidence_interval(self, engine):
        """全ルートに confidence_interval が存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        for route in result["recommended_routes"]:
            assert "confidence_interval" in route

    def test_rq7_no_contradictory(self, engine):
        """rq7_contradictory_routes_excluded=True。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["quality_gates"]["rq7_contradictory_routes_excluded"] is True

    def test_recommended_routes_scored(self, engine):
        """各ルートに score, labels, confidence_interval が存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        for route in result["recommended_routes"]:
            assert "score" in route
            assert "labels" in route
            assert "confidence_interval" in route

    def test_confidence_level_values(self, engine):
        """confidence_level が 'high'/'medium'/'low'/'very_low' のいずれかであること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["summary"]["confidence_level"] in {
            "high", "medium", "low", "very_low"
        }

    def test_scenario_pi_to_tai(self, engine):
        """天地否(12)->地天泰(11) のシナリオが正常動作すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["l1_yao"]["current_hex"] == 12
        assert result["l1_yao"]["goal_hex"] == 11
        assert result["l2_state"]["current_state"] == "停滞・閉塞"
        assert result["l2_state"]["goal_state"] == "安定・平和"
        assert result["summary"]["current_hex"] == 12
        assert result["summary"]["goal_hex"] == 11


# ============================================================
# 6. Wilson スコア区間
# ============================================================

class TestWilsonScoreInterval:
    """_wilson_score_interval のユニットテスト。"""

    def test_zero_total(self):
        """total=0 -> (0.0, 1.0)。"""
        lower, upper = _wilson_score_interval(0, 0)
        assert lower == 0.0
        assert upper == 1.0

    def test_all_success(self):
        """100/100 -> lower > 0.9。"""
        lower, upper = _wilson_score_interval(100, 100)
        assert lower > 0.9

    def test_half_success(self):
        """50/100 -> lower < 0.5 < upper。"""
        lower, upper = _wilson_score_interval(50, 100)
        assert lower < 0.5
        assert upper > 0.5

    def test_range(self):
        """lower >= 0, upper <= 1.0。"""
        for s in range(0, 101, 10):
            lower, upper = _wilson_score_interval(s, 100)
            assert lower >= 0.0
            assert upper <= 1.0


# ============================================================
# 7. _sanitize_text
# ============================================================

class TestSanitizeText:
    """_sanitize_text のユニットテスト。"""

    def test_removes_deterministic_words(self):
        """'必ず成功する' -> '成功する'。"""
        result = _sanitize_text("必ず成功する")
        assert "必ず" not in result
        assert "成功する" in result

    def test_preserves_clean_text(self):
        """問題ない文章はそのまま返ること。"""
        text = "この結果は参考情報です"
        assert _sanitize_text(text) == text


# ============================================================
# 8. BacktraceSessionOrchestrator — セッション作成
# ============================================================

class TestBacktraceSessionOrchestrator:
    """BacktraceSessionOrchestrator のセッション作成テスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def test_create_session(self, orchestrator):
        """mode='backtrace', phase='goal-select', 全フィールドが None。"""
        session = orchestrator.create_session()
        assert session["mode"] == "backtrace"
        assert session["phase"] == "goal-select"
        assert session["goal_hex"] is None
        assert session["goal_state"] is None
        assert session["goal_summary"] is None
        assert session["current_hex"] is None
        assert session["current_state"] is None
        assert session["current_action"] is None
        assert session["backtrace_result"] is None
        assert session["feedback"] is None
        assert session["roadmap"] is None


# ============================================================
# 9. BacktraceSessionOrchestrator — set_goal
# ============================================================

class TestSetGoal:
    """set_goal のテスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def test_hexagram_method(self, orchestrator):
        """method='hexagram', value=11 -> goal_hex=11, goal_name='地天泰', phase='goal-set'。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="hexagram", value=11)
        assert "error" not in result
        assert result["goal_hex"] == 11
        assert result["goal_name"] == "地天泰"
        assert result["phase"] == "goal-set"

    def test_hexagram_invalid_number(self, orchestrator):
        """value=0 -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="hexagram", value=0)
        assert "error" in result

    def test_hexagram_out_of_range(self, orchestrator):
        """value=65 -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="hexagram", value=65)
        assert "error" in result

    def test_theme_method_with_results(self, orchestrator):
        """method='theme', value='孤独' -> candidates >= 1。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="theme", value="孤独")
        assert "error" not in result
        candidates = result.get("candidates", [])
        assert len(candidates) >= 1

    def test_theme_method_empty(self, orchestrator):
        """method='theme', value='' -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="theme", value="")
        assert "error" in result

    def test_text_method_empty(self, orchestrator):
        """method='text', value='' -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="text", value="")
        assert "error" in result

    def test_unknown_method(self, orchestrator):
        """method='unknown' -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="unknown", value="test")
        assert "error" in result

    def test_goal_name_is_kanji(self, orchestrator):
        """goal_name が数字だけでないこと（Bug Fix 1 検証）。"""
        session = orchestrator.create_session()
        result = orchestrator.set_goal(session, method="hexagram", value=11)
        goal_name = result.get("goal_name", "")
        # goal_name should contain kanji characters, not just digits
        assert not goal_name.isdigit(), f"goal_name should not be digits only: {goal_name}"


# ============================================================
# 10. _parse_hex_num_from_string
# ============================================================

class TestParseHexNumFromString:
    """_parse_hex_num_from_string のテスト。"""

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def test_parentheses_format(self, orchestrator):
        """'乾為天 (1)' -> 1。"""
        assert orchestrator._parse_hex_num_from_string("乾為天 (1)") == 1

    def test_underscore_format(self, orchestrator):
        """'29_坎為水' -> 29（Bug Fix 2 検証）。"""
        assert orchestrator._parse_hex_num_from_string("29_坎為水") == 29

    def test_plain_number(self, orchestrator):
        """'42' -> 42。"""
        assert orchestrator._parse_hex_num_from_string("42") == 42

    def test_out_of_range(self, orchestrator):
        """'65' -> None。"""
        assert orchestrator._parse_hex_num_from_string("65") is None

    def test_invalid_string(self, orchestrator):
        """'テスト' -> None。"""
        assert orchestrator._parse_hex_num_from_string("テスト") is None


# ============================================================
# 11. confirm_goal
# ============================================================

class TestConfirmGoal:
    """confirm_goal のテスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def test_confirm_with_hex_num(self, orchestrator):
        """hex_num=11 -> goal_name='地天泰', phase='current-describe'。"""
        session = orchestrator.create_session()
        result = orchestrator.confirm_goal(session, hex_num=11)
        assert "error" not in result
        assert result["goal_name"] == "地天泰"
        assert result["phase"] == "current-describe"

    def test_confirm_without_set_goal(self, orchestrator):
        """goal_hex 未設定 -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.confirm_goal(session)
        assert "error" in result

    def test_confirm_existing_goal(self, orchestrator):
        """set_goal の後に hex_num なしで confirm -> 成功。"""
        session = orchestrator.create_session()
        orchestrator.set_goal(session, method="hexagram", value=11)
        result = orchestrator.confirm_goal(session)
        assert "error" not in result
        assert result["goal_hex"] == 11
        assert result["phase"] == "current-describe"


# ============================================================
# 12. describe_current
# ============================================================

class TestDescribeCurrent:
    """describe_current のテスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def test_direct_mode(self, orchestrator):
        """current_hex=12, current_state='停滞・閉塞' -> phase='analyzing'。"""
        session = orchestrator.create_session()
        result = orchestrator.describe_current(
            session, current_hex=12, current_state="停滞・閉塞",
            scale="company"
        )
        assert "error" not in result
        assert result["current_hex"] == 12
        assert result["current_state"] == "停滞・閉塞"
        assert result["phase"] == "analyzing"
        assert result.get("scale") == "company"

    def test_missing_state(self, orchestrator):
        """current_hex=12, current_state=None -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.describe_current(session, current_hex=12, current_state=None)
        assert "error" in result

    def test_missing_hex(self, orchestrator):
        """text=None, current_hex=None -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.describe_current(session, text=None, current_hex=None)
        assert "error" in result

    def test_default_action(self, orchestrator):
        """action_type=None -> current_action='慎重・観察'。"""
        session = orchestrator.create_session()
        result = orchestrator.describe_current(
            session, current_hex=12, current_state="停滞・閉塞", action_type=None
        )
        assert result["current_action"] == "慎重・観察"

    def test_hex_out_of_range(self, orchestrator):
        """current_hex=0 -> error。"""
        session = orchestrator.create_session()
        result = orchestrator.describe_current(
            session, current_hex=0, current_state="テスト"
        )
        assert "error" in result


# ============================================================
# 13. analyze
# ============================================================

class TestAnalyze:
    """analyze のテスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def _setup_session(self, orchestrator):
        """ゴール設定 + 確定 + 現在地設定 済みのセッションを返す。"""
        session = orchestrator.create_session()
        orchestrator.set_goal(session, method="hexagram", value=11)
        orchestrator.confirm_goal(session)
        orchestrator.describe_current(
            session, current_hex=12, current_state="停滞・閉塞",
            action_type="慎重・観察", scale="company"
        )
        return session

    def test_full_flow(self, orchestrator):
        """set_goal -> confirm_goal -> describe_current -> analyze -> phase='result'。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        assert "error" not in result
        assert result["phase"] == "result"

    def test_analyze_without_current(self, orchestrator):
        """analyze 前に describe_current 未実行 -> error。"""
        session = orchestrator.create_session()
        orchestrator.set_goal(session, method="hexagram", value=11)
        orchestrator.confirm_goal(session)
        result = orchestrator.analyze(session)
        assert "error" in result

    def test_analyze_without_goal(self, orchestrator):
        """analyze 前に set_goal 未実行 -> error。"""
        session = orchestrator.create_session()
        session["current_hex"] = 12
        session["current_state"] = "停滞・閉塞"
        result = orchestrator.analyze(session)
        assert "error" in result

    def test_feedback_layers_structure(self, orchestrator):
        """feedback_layers に R1-R5 が全て存在すること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        assert "error" not in result
        fl = result["feedback_layers"]
        for key in ("R1", "R2", "R3", "R4", "R5"):
            assert key in fl, f"feedback_layers missing '{key}'"

    def test_r1_goal_metadata(self, orchestrator):
        """R1 に goal_name, goal_metadata (meaning, situation, keywords) が存在すること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r1 = result["feedback_layers"]["R1"]
        assert "goal_name" in r1
        assert "goal_metadata" in r1
        meta = r1["goal_metadata"]
        assert "meaning" in meta
        assert "situation" in meta
        assert "keywords" in meta

    def test_r2_gap_structure(self, orchestrator):
        """R2 に hamming_distance, changing_lines, difficulty が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r2 = result["feedback_layers"]["R2"]
        assert "hamming_distance" in r2
        assert "changing_lines" in r2
        assert "difficulty" in r2

    def test_r3_recommended_routes(self, orchestrator):
        """R3 に routes (リスト), route_count が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r3 = result["feedback_layers"]["R3"]
        assert "routes" in r3
        assert isinstance(r3["routes"], list)
        assert "route_count" in r3

    def test_r4_action_patterns(self, orchestrator):
        """R4 に recommended_actions が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r4 = result["feedback_layers"]["R4"]
        assert "recommended_actions" in r4

    def test_r5_reflective_question(self, orchestrator):
        """R5 に reflective_question (空でない文字列) が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r5 = result["feedback_layers"]["R5"]
        assert "reflective_question" in r5
        assert isinstance(r5["reflective_question"], str)
        assert len(r5["reflective_question"]) > 0


# ============================================================
# 14. generate_roadmap
# ============================================================

class TestGenerateRoadmap:
    """generate_roadmap のテスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def _setup_analyzed_session(self, orchestrator):
        """analyze 済みのセッションを返す。"""
        session = orchestrator.create_session()
        orchestrator.set_goal(session, method="hexagram", value=11)
        orchestrator.confirm_goal(session)
        orchestrator.describe_current(
            session, current_hex=12, current_state="停滞・閉塞",
            action_type="慎重・観察", scale="company"
        )
        orchestrator.analyze(session)
        return session

    def test_demo_roadmap_without_llm(self, orchestrator):
        """LLM 不使用 -> DEMO_ROADMAP が返ること。"""
        session = self._setup_analyzed_session(orchestrator)
        result = orchestrator.generate_roadmap(session)
        assert "error" not in result
        roadmap = result["roadmap"]
        assert roadmap["overview"] == DEMO_ROADMAP["overview"]

    def test_roadmap_required_keys(self, orchestrator):
        """roadmap に overview, phase_1-3, immediate_action, caution が含まれること。"""
        session = self._setup_analyzed_session(orchestrator)
        result = orchestrator.generate_roadmap(session)
        roadmap = result["roadmap"]
        for key in _ROADMAP_REQUIRED_KEYS:
            assert key in roadmap, f"roadmap missing required key: {key}"

    def test_roadmap_no_prohibited_words(self, orchestrator):
        """禁止ワード不含。"""
        session = self._setup_analyzed_session(orchestrator)
        result = orchestrator.generate_roadmap(session)
        roadmap_text = json.dumps(result["roadmap"], ensure_ascii=False)
        for word in _ROADMAP_PROHIBITED_WORDS:
            assert word not in roadmap_text, f"roadmap contains prohibited word: {word}"

    def test_roadmap_without_analyze(self, orchestrator):
        """analyze 未実行 -> error。"""
        session = orchestrator.create_session()
        orchestrator.set_goal(session, method="hexagram", value=11)
        orchestrator.confirm_goal(session)
        orchestrator.describe_current(
            session, current_hex=12, current_state="停滞・閉塞",
            scale="company"
        )
        # analyze を飛ばしてロードマップ生成
        result = orchestrator.generate_roadmap(session)
        assert "error" in result


# ============================================================
# 15. End-to-End テスト
# ============================================================

class TestEndToEnd:
    """完全フローの E2E テスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return BacktraceSessionOrchestrator()

    def test_pi_to_tai_scenario(self, orchestrator):
        """天地否(12, '停滞・閉塞') -> 地天泰(11, '安定成長・成功') の完全フロー。"""
        # Step 1: create_session
        session = orchestrator.create_session()
        assert session["phase"] == "goal-select"

        # Step 2: set_goal
        r1 = orchestrator.set_goal(session, method="hexagram", value=11)
        assert "error" not in r1
        assert r1["phase"] == "goal-set"

        # Step 3: confirm_goal
        r2 = orchestrator.confirm_goal(session)
        assert "error" not in r2
        assert r2["phase"] == "current-describe"

        # Step 4: describe_current
        r3 = orchestrator.describe_current(
            session, current_hex=12, current_state="停滞・閉塞",
            action_type="慎重・観察", scale="company"
        )
        assert "error" not in r3
        assert r3["phase"] == "analyzing"
        assert r3.get("scale") == "company"

        # Step 5: analyze
        r4 = orchestrator.analyze(session)
        assert "error" not in r4
        assert r4["phase"] == "result"

        # R1-R5 全層が存在
        fl = r4["feedback_layers"]
        for key in ("R1", "R2", "R3", "R4", "R5"):
            assert key in fl

        # quality_gates に rq1-rq8
        qg = r4["quality_gates"]
        for rq in ("rq1_reference_only", "rq2_low_success_rate",
                    "rq3_no_deterministic_words", "rq4_has_alternative_route",
                    "rq5_confidence_interval_computed", "rq6_fallback_used",
                    "rq7_contradictory_routes_excluded",
                    "rq8_scale_specified"):
            assert rq in qg

        # Step 6: generate_roadmap
        r5 = orchestrator.generate_roadmap(session)
        assert "error" not in r5
        roadmap = r5["roadmap"]
        for key in _ROADMAP_REQUIRED_KEYS:
            assert key in roadmap

    def test_kan_to_ken_scenario(self, orchestrator):
        """坎為水(29, 'どん底・危機') -> 乾為天(1, 'V字回復・大成功') のシナリオ。"""
        session = orchestrator.create_session()
        orchestrator.set_goal(session, method="hexagram", value=1)
        orchestrator.confirm_goal(session)
        orchestrator.describe_current(
            session, current_hex=29, current_state="どん底・危機",
            action_type="慎重・観察", scale="company"
        )
        result = orchestrator.analyze(session)
        assert "error" not in result

        # L1: hamming > 0
        l1 = result["backtrace_result"]["l1_yao"]
        assert l1["hamming_distance"] > 0

        # L3: routes or action_recommendations が空でないこと
        l3 = result["backtrace_result"]["l3_action"]
        has_routes = len(l3.get("routes", [])) > 0
        has_actions = len(l3.get("action_recommendations", [])) > 0
        assert has_routes or has_actions, (
            "L3 should have either routes or action_recommendations"
        )


# ============================================================
# 16. API エンドポイント テスト
# ============================================================

class TestBacktraceAPI:
    """Flask API のバックトレースモードエンドポイントを E2E で検証する。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture
    def client(self):
        from app import app, sessions
        app.config["TESTING"] = True
        sessions.clear()
        with app.test_client() as c:
            yield c

    def _create_backtrace_session(self, client):
        """バックトレースモードのセッションを作成し session_id を返すヘルパー。"""
        resp = client.post(
            "/api/session",
            data=json.dumps({"mode": "backtrace"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "session_id" in data
        assert data.get("mode") == "backtrace"
        return data["session_id"]

    def test_session_create_backtrace(self, client):
        """POST /api/session mode=backtrace -> session_id 返却。"""
        resp = client.post(
            "/api/session",
            data=json.dumps({"mode": "backtrace"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "session_id" in data
        assert data["mode"] == "backtrace"

    def test_set_goal_endpoint(self, client):
        """POST /api/backtrace/set-goal -> goal_hex, goal_name。"""
        sid = self._create_backtrace_session(client)
        resp = client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({
                "session_id": sid,
                "method": "hexagram",
                "value": 11,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["goal_hex"] == 11
        assert "goal_name" in data

    def test_confirm_goal_endpoint(self, client):
        """POST /api/backtrace/confirm-goal -> phase='current-describe'。"""
        sid = self._create_backtrace_session(client)
        # set-goal first
        client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({
                "session_id": sid,
                "method": "hexagram",
                "value": 11,
            }),
            content_type="application/json",
        )
        # confirm-goal
        resp = client.post(
            "/api/backtrace/confirm-goal",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["phase"] == "current-describe"

    def test_describe_current_endpoint(self, client):
        """POST /api/backtrace/describe-current -> phase='analyzing'。"""
        sid = self._create_backtrace_session(client)
        # set-goal
        client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({
                "session_id": sid,
                "method": "hexagram",
                "value": 11,
            }),
            content_type="application/json",
        )
        # confirm-goal
        client.post(
            "/api/backtrace/confirm-goal",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        # describe-current
        resp = client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
                "scale": "company",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["phase"] == "analyzing"

    def test_analyze_endpoint(self, client):
        """POST /api/backtrace/analyze -> feedback_layers, quality_gates。"""
        sid = self._create_backtrace_session(client)
        # set-goal
        client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({
                "session_id": sid,
                "method": "hexagram",
                "value": 11,
            }),
            content_type="application/json",
        )
        # confirm-goal
        client.post(
            "/api/backtrace/confirm-goal",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        # describe-current
        client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
                "scale": "company",
            }),
            content_type="application/json",
        )
        # analyze
        resp = client.post(
            "/api/backtrace/analyze",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "feedback_layers" in data
        assert "quality_gates" in data

    def test_roadmap_endpoint(self, client):
        """POST /api/backtrace/roadmap -> roadmap。"""
        sid = self._create_backtrace_session(client)
        # full flow setup
        client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({
                "session_id": sid,
                "method": "hexagram",
                "value": 11,
            }),
            content_type="application/json",
        )
        client.post(
            "/api/backtrace/confirm-goal",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
                "scale": "company",
            }),
            content_type="application/json",
        )
        client.post(
            "/api/backtrace/analyze",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        # roadmap
        resp = client.post(
            "/api/backtrace/roadmap",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "roadmap" in data

    def test_full_api_flow(self, client):
        """上記を順番に実行する完全フロー。"""
        # 1. session
        sid = self._create_backtrace_session(client)

        # 2. set-goal
        r1 = client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({
                "session_id": sid,
                "method": "hexagram",
                "value": 11,
            }),
            content_type="application/json",
        )
        assert r1.status_code == 200
        d1 = r1.get_json()
        assert d1["goal_hex"] == 11
        assert d1["phase"] == "goal-set"

        # 3. confirm-goal
        r2 = client.post(
            "/api/backtrace/confirm-goal",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert r2.status_code == 200
        d2 = r2.get_json()
        assert d2["phase"] == "current-describe"

        # 4. describe-current
        r3 = client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
                "scale": "company",
            }),
            content_type="application/json",
        )
        assert r3.status_code == 200
        d3 = r3.get_json()
        assert d3["phase"] == "analyzing"

        # 5. analyze
        r4 = client.post(
            "/api/backtrace/analyze",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert r4.status_code == 200
        d4 = r4.get_json()
        assert d4["phase"] == "result"
        assert "feedback_layers" in d4
        assert "quality_gates" in d4
        fl = d4["feedback_layers"]
        for key in ("R1", "R2", "R3", "R4", "R5"):
            assert key in fl

        # 6. roadmap
        r5 = client.post(
            "/api/backtrace/roadmap",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert r5.status_code == 200
        d5 = r5.get_json()
        assert "roadmap" in d5
        roadmap = d5["roadmap"]
        for key in _ROADMAP_REQUIRED_KEYS:
            assert key in roadmap


# ============================================================
# 17. Fix1: 状態ラベルのファジーマッチングテスト
# ============================================================

class TestFuzzyStateMatching:
    """Fix1: 状態ラベルのファジーマッチングテスト。

    rev_after_state.json の完全一致検索で 0 件になるケースに対し、
    エイリアス辞書 + 部分一致フォールバックで結果を返すことを検証する。
    """

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_exact_match_still_works(self, engine):
        """完全一致は引き続き動作する。"""
        # "安定成長・成功" は rev_after_state.json に実在するキー (1091件)
        result = engine.reverse_state("停滞・閉塞", "安定成長・成功")
        assert result.get("case_count", 0) > 0, (
            "Exact match for '安定成長・成功' should return cases"
        )

    def test_partial_match_juncho_to_seikou(self, engine):
        """'安定成長・順調' → '安定成長・成功' にマッチ。"""
        result = engine.reverse_state("停滞・閉塞", "安定成長・順調")
        assert result.get("total_cases", result.get("case_count", 0)) > 0, (
            "'安定成長・順調' should fuzzy-match to '安定成長・成功'"
        )

    def test_partial_match_suitai(self, engine):
        """'衰退・下降' → 類似ラベルにマッチ。"""
        result = engine.reverse_state("安定成長・成功", "衰退・下降")
        assert result.get("total_cases", result.get("case_count", 0)) > 0, (
            "'衰退・下降' should fuzzy-match to a similar label"
        )

    def test_fuzzy_match_returns_matched_label(self, engine):
        """ファジーマッチ時、実際にマッチしたラベルを返す。"""
        result = engine.reverse_state("停滞・閉塞", "安定成長・順調")
        assert "matched_goal_state" in result, (
            "Fuzzy match result should contain 'matched_goal_state'"
        )
        assert result["matched_goal_state"] == "安定成長・成功", (
            "matched_goal_state should be '安定成長・成功'"
        )

    def test_exact_match_matched_label_is_same(self, engine):
        """完全一致時、matched_goal_state は入力と同じ。"""
        result = engine.reverse_state("停滞・閉塞", "安定成長・成功")
        assert result.get("matched_goal_state") == "安定成長・成功"

    def test_no_match_returns_empty(self, engine):
        """完全に無関係な入力は空結果。"""
        result = engine.reverse_state("停滞・閉塞", "XYZXYZ完全無関係")
        assert result.get("case_count", 0) == 0, (
            "Completely unrelated input should return 0 cases"
        )

    def test_donzoko_alias(self, engine):
        """'どん底' → 'どん底・危機' にマッチ。"""
        result = engine.reverse_state("停滞・閉塞", "どん底")
        assert result.get("case_count", 0) > 0
        assert result.get("matched_goal_state") == "どん底・危機"

    def test_fuzzy_current_state_also_works(self, engine):
        """current_state のファジーマッチもテスト。
        goal_reachability の計算に current_state の一致が必要なので、
        current_state もファジーマッチされること。
        """
        # "停滞" → "停滞・閉塞" にマッチすべき
        result = engine.reverse_state("停滞", "V字回復・大成功")
        assert result.get("case_count", 0) > 0
        assert result.get("goal_reachability", 0.0) > 0.0, (
            "current_state '停滞' should fuzzy-match to '停滞・閉塞' "
            "and contribute to goal_reachability"
        )


# ============================================================
# 18. Fix2: CI計算バグ修正テスト
# ============================================================

import unittest
import re


class TestCICalculationFix(unittest.TestCase):
    """Fix2: CI計算バグ修正テスト — ルートのCIはステップの最小nを使うべき"""

    def test_wilson_score_basic(self):
        """Wilson Scoreが正しく計算される"""
        lower, upper = _wilson_score_interval(8, 10)
        # n=10, success=8 → 80% success rate
        self.assertGreater(lower, 0.4)
        self.assertLess(upper, 1.0)

    def test_wilson_score_small_n(self):
        """n=1の場合、CIは広くなるべき"""
        lower, upper = _wilson_score_interval(1, 1)
        # n=1で成功率100%でも、CIは広い（不確実性が高い）
        self.assertLess(lower, 0.5, "n=1でCI下限が0.5以上は詐欺的")

    def test_wilson_score_large_n(self):
        """n=100の場合、CIは狭い"""
        lower, upper = _wilson_score_interval(80, 100)
        ci_width = upper - lower
        self.assertLess(ci_width, 0.2)

    def test_full_backtrace_ci_uses_min_step_count(self):
        """full_backtraceのCI計算で、noteのnがルートステップの最小countを反映する"""
        engine = BacktraceEngine()
        # 47->1: step_counts=[1] で L2 case_count=2205。
        # 修正前: n=2205 (L2 case_count)。修正後: n=1 (min step count)
        result = engine.full_backtrace(47, "停滞・閉塞", 1, "持続成長・大成功", scale="company")

        routes = result.get("recommended_routes", [])
        found_route_with_steps = False
        for route in routes:
            route_data = route.get("route", {})
            steps = route_data.get("steps", [])
            # countフィールドを持つstepのみ対象
            step_counts = [s.get("count") for s in steps
                           if isinstance(s.get("count"), (int, float))]
            if not step_counts:
                continue
            found_route_with_steps = True
            min_step_count = min(step_counts)
            ci = route.get("confidence_interval", {})
            note = ci.get("note", "")

            # noteからnを抽出
            m = re.search(r"n=(\d+)", note)
            if m:
                n_in_ci = int(m.group(1))
                # CIのnはルートステップの最小countを使うべき
                self.assertLessEqual(
                    n_in_ci, min_step_count,
                    f"CI n={n_in_ci} がステップ最小count={min_step_count}より大きい。"
                    f"L2のcase_countをそのまま使っている可能性あり"
                )

            # min_step_count が小さいルートではCI幅は広くなるべき
            ci_lower = ci.get("lower", 0)
            ci_upper = ci.get("upper", 1)
            if min_step_count <= 5:
                ci_width = ci_upper - ci_lower
                self.assertGreater(
                    ci_width, 0.15,
                    f"min_step_count={min_step_count}でCI幅{ci_width:.3f}は"
                    f"狭すぎる（不確実性を過小評価）"
                )

        if not found_route_with_steps:
            self.skipTest("countフィールドを持つルートが見つからなかった")

    def test_full_backtrace_ci_n1_route_wide_ci(self):
        """n=1のステップを含むルートでCIが十分に広い"""
        engine = BacktraceEngine()
        # 29->11: step_counts=[247, 1] で min=1
        result = engine.full_backtrace(29, "どん底・危機", 11, "安定・平和", scale="company")

        routes = result.get("recommended_routes", [])
        for route in routes:
            route_data = route.get("route", {})
            steps = route_data.get("steps", [])
            step_counts = [s.get("count") for s in steps
                           if isinstance(s.get("count"), (int, float))]
            if not step_counts:
                continue
            min_step_count = min(step_counts)
            if min_step_count <= 1:
                ci = route.get("confidence_interval", {})
                ci_lower = ci.get("lower", 0)
                ci_upper = ci.get("upper", 1)
                ci_width = ci_upper - ci_lower
                self.assertGreater(
                    ci_width, 0.15,
                    f"min_step_count={min_step_count}でCI幅{ci_width:.3f}は"
                    f"狭すぎる。n=1なら不確実性は高いはず"
                )


# ============================================================
# 19. Fix3: 行動ラベル二重語彙統一テスト
# ============================================================


class TestActionVocabularyUnification(unittest.TestCase):
    """Fix3: 行動ラベル二重語彙統一テスト"""

    def setUp(self):
        self.engine = BacktraceEngine()

    def test_trigram_to_action_mapping_exists(self):
        """八卦→日本語行動タイプのマッピングが存在する"""
        self.assertTrue(
            hasattr(self.engine, '_trigram_to_action') or
            hasattr(BacktraceEngine, '_TRIGRAM_TO_ACTION'),
            "八卦→行動マッピングが存在しない"
        )

    def test_action_recommendations_no_raw_trigrams(self):
        """action_recommendationsに生の八卦名が含まれない"""
        result = self.engine.full_backtrace(29, "どん底・危機", 1, "持続成長・大成功", scale="company")
        actions = result.get("L3_action", result.get("l3_action", {})).get(
            "action_recommendations", []
        )
        raw_trigrams = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
        for action in actions:
            action_name = action.get("action_type", "") if isinstance(action, dict) else str(action)
            self.assertNotIn(
                action_name, raw_trigrams,
                f"生の八卦名 '{action_name}' がaction_recommendationsに含まれている"
            )

    def test_action_recommendations_use_japanese(self):
        """action_recommendationsが日本語行動タイプを使用"""
        result = self.engine.full_backtrace(29, "どん底・危機", 1, "持続成長・大成功", scale="company")
        actions = result.get("L3_action", result.get("l3_action", {})).get(
            "action_recommendations", []
        )
        if actions:
            for action in actions:
                action_name = action.get("action_type", "") if isinstance(action, dict) else str(action)
                # 日本語の行動タイプは「・」を含むか、漢字2文字以上
                self.assertTrue(
                    "・" in action_name or len(action_name) >= 4,
                    f"'{action_name}' は日本語行動タイプではない"
                )

    def test_mapping_covers_all_eight_trigrams(self):
        """8つの八卦全てにマッピングがある"""
        mapping = getattr(self.engine, '_trigram_to_action', None) or \
                  getattr(BacktraceEngine, '_TRIGRAM_TO_ACTION', {})
        trigrams = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
        for t in trigrams:
            self.assertIn(t, mapping, f"八卦 '{t}' のマッピングがない")

    def test_recommended_routes_action_recommendations_no_raw_trigrams(self):
        """recommended_routes内のaction_recommendationsにも生の八卦名が含まれない"""
        result = self.engine.full_backtrace(29, "どん底・危機", 1, "持続成長・大成功", scale="company")
        raw_trigrams = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
        for route in result.get("recommended_routes", []):
            for action in route.get("action_recommendations", []):
                action_name = action.get("action_type", "") if isinstance(action, dict) else str(action)
                self.assertNotIn(
                    action_name, raw_trigrams,
                    f"recommended_routes内に生の八卦名 '{action_name}' が含まれている"
                )

    def test_translate_trigram_method_exists(self):
        """_translate_trigram_to_action メソッドが存在する"""
        self.assertTrue(
            hasattr(self.engine, '_translate_trigram_to_action'),
            "_translate_trigram_to_action メソッドが存在しない"
        )

    def test_translate_trigram_returns_japanese(self):
        """_translate_trigram_to_action が八卦名を日本語行動タイプに変換する"""
        method = getattr(self.engine, '_translate_trigram_to_action', None)
        if method is None:
            self.fail("_translate_trigram_to_action メソッドが存在しない")
        raw_trigrams = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
        for tg in raw_trigrams:
            result = method(tg)
            self.assertNotIn(result, raw_trigrams,
                f"'{tg}' が変換後もそのまま '{result}' として返された")
            self.assertTrue(
                "・" in result or len(result) >= 4,
                f"'{tg}' の変換結果 '{result}' は日本語行動タイプではない"
            )

    def test_translate_non_trigram_passes_through(self):
        """_translate_trigram_to_action は八卦名以外をそのまま返す"""
        method = getattr(self.engine, '_translate_trigram_to_action', None)
        if method is None:
            self.fail("_translate_trigram_to_action メソッドが存在しない")
        # 日本語行動タイプはそのまま通過すべき
        self.assertEqual(method("攻める・挑戦"), "攻める・挑戦")
        self.assertEqual(method("守る・維持"), "守る・維持")


# ============================================================
# 20. Scale必須引数テスト
# ============================================================


class TestScaleRequired:
    """scale必須引数のテスト — full_backtrace, reverse_state, reverse_action のscale対応を検証。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    # --- RQ8: scale必須チェック ---

    def test_full_backtrace_scale_none_returns_error(self, engine):
        """scale=None で full_backtrace を呼ぶとエラー辞書を返すこと。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale=None)
        assert "error" in result, "scale=None should return error dict"
        assert "scale" in result["error"].lower() or "scale" in result["error"]

    def test_full_backtrace_invalid_scale_returns_error(self, engine):
        """不正なscale値で full_backtrace を呼ぶとエラー辞書を返すこと。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="invalid_scale")
        assert "error" in result, "Invalid scale should return error dict"

    def test_full_backtrace_all_valid_scales(self, engine):
        """全VALID_SCALES値でfull_backtraceが正常動作すること。"""
        for scale in VALID_SCALES:
            result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale=scale)
            assert "error" not in result, (
                f"scale='{scale}' でエラーが返された: {result.get('error', '')}"
            )
            assert "l1_yao" in result
            assert "l2_state" in result
            assert "l3_action" in result
            assert result.get("scale") == scale

    def test_rq8_scale_specified_true(self, engine):
        """scale指定時にrq8_scale_specified=Trueになること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["quality_gates"]["rq8_scale_specified"] is True

    # --- scaleインデックスの読み込み ---

    def test_scale_indices_loaded(self, engine):
        """エンジン初期化時にscale別インデックスが読み込まれること。"""
        assert hasattr(engine, "_scale_indices")
        for scale in list(VALID_SCALES) + ["all"]:
            assert scale in engine._scale_indices, (
                f"scale='{scale}' のインデックスが読み込まれていない"
            )
            idx = engine._scale_indices[scale]
            assert "rev_after_state" in idx
            assert "rev_outcome_action" in idx
            assert "rev_pattern_after" in idx
            assert "rev_hex_pair_stats" in idx

    # --- scale別データの検証 ---

    def test_company_scale_has_data(self, engine):
        """scale='company'のインデックスに事例が存在すること。"""
        idx = engine._scale_indices["company"]["rev_after_state"]
        total = sum(
            entry.get("total_count", 0) for entry in idx.values()
            if isinstance(entry, dict)
        )
        assert total > 0, "company scale should have cases"

    def test_scale_specific_result_contains_scale(self, engine):
        """full_backtrace結果のsummaryにscaleが含まれること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert result["summary"]["scale"] == "company"

    def test_scale_fallback_notes_in_summary(self, engine):
        """full_backtrace結果のsummaryにscale_fallback_notesリストが存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert "scale_fallback_notes" in result["summary"]
        assert isinstance(result["summary"]["scale_fallback_notes"], list)

    # --- フォールバックのテスト ---

    def test_get_scale_index_fallback_for_small_scale(self, engine):
        """事例数が閾値未満のscaleで"all"にフォールバックすること。"""
        # _get_scale_index を直接テスト
        # "family" は事例数が少ないインデックスがある可能性
        for index_name in ("rev_after_state", "rev_outcome_action",
                           "rev_pattern_after", "rev_hex_pair_stats"):
            idx, used_fallback = engine._get_scale_index(index_name, "company")
            # company は十分な事例数があるのでフォールバックしないはず
            assert isinstance(idx, dict)

    def test_get_scale_index_none_returns_legacy(self, engine):
        """scale=Noneでレガシーインデックスを返すこと。"""
        idx, used_fallback = engine._get_scale_index("rev_after_state", None)
        assert idx is engine._rev_after_state
        assert used_fallback is False

    # --- reverse_state / reverse_action のscale対応 ---

    def test_reverse_state_with_scale(self, engine):
        """reverse_stateがscale指定時にscale_fallback_noteを返すこと。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功", scale="company")
        # scale_fallback_note フィールドが存在すること（空文字でもOK）
        assert "scale_fallback_note" in result

    def test_reverse_action_with_scale(self, engine):
        """reverse_actionがscale指定時にscale_fallback_noteを返すこと。"""
        result = engine.reverse_action(
            12, "停滞・閉塞", 11, "安定・平和", scale="company"
        )
        assert "scale_fallback_note" in result

    def test_reverse_yao_accepts_scale(self, engine):
        """reverse_yaoがscale引数を受け入れること（フィルタリングはしない）。"""
        result = engine.reverse_yao(12, 11, scale="company")
        assert "current_hex" in result
        assert "goal_hex" in result

    def test_reverse_state_legacy_no_scale(self, engine):
        """reverse_stateがscale=Noneで後方互換動作すること。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功")
        assert result.get("case_count", 0) > 0


class TestScaleAPIEndpoint:
    """API エンドポイントのscale必須チェックテスト。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture
    def client(self):
        from app import app, sessions
        app.config["TESTING"] = True
        sessions.clear()
        with app.test_client() as c:
            yield c

    def _setup_to_describe_current(self, client):
        """set-goal + confirm-goal 済みのsession_idを返す。"""
        from app import sessions
        # create session
        resp = client.post(
            "/api/session",
            data=json.dumps({"mode": "backtrace"}),
            content_type="application/json",
        )
        sid = resp.get_json()["session_id"]
        # set-goal
        client.post(
            "/api/backtrace/set-goal",
            data=json.dumps({"session_id": sid, "method": "hexagram", "value": 11}),
            content_type="application/json",
        )
        # confirm-goal
        client.post(
            "/api/backtrace/confirm-goal",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        return sid

    def test_describe_current_without_scale_returns_400(self, client):
        """describe-currentでscale未指定時に400エラーを返すこと。"""
        sid = self._setup_to_describe_current(client)
        resp = client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_describe_current_with_invalid_scale_returns_error(self, client):
        """describe-currentで不正なscale値時にエラーを返すこと。"""
        sid = self._setup_to_describe_current(client)
        resp = client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
                "scale": "invalid_scale",
            }),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "error" in data

    def test_describe_current_with_valid_scale_succeeds(self, client):
        """describe-currentでscale='individual'が正常動作すること。"""
        sid = self._setup_to_describe_current(client)
        resp = client.post(
            "/api/backtrace/describe-current",
            data=json.dumps({
                "session_id": sid,
                "current_hex": 12,
                "current_state": "停滞・閉塞",
                "action_type": "慎重・観察",
                "scale": "individual",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["phase"] == "analyzing"
        assert data.get("scale") == "individual"


# ============================================================
# 多義性警告テスト (Polysemy Warnings)
# ============================================================

class TestPolysemyWarnings:
    """多義性メタデータと警告システムのテスト。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_polysemy_metadata_loaded(self, engine):
        """polysemy_metadata.json がロードされること。"""
        assert isinstance(engine._polysemy_metadata, dict)
        assert "labels" in engine._polysemy_metadata

    def test_polysemy_metadata_has_labels(self, engine):
        """メタデータに action_type ラベルが含まれること。"""
        labels = engine._polysemy_metadata.get("labels", {})
        assert len(labels) > 0, "labels は空であってはならない"

    def test_get_polysemy_warning_high(self, engine):
        """高多義性ラベル「捨てる・撤退」の警告が返ること。"""
        warning = engine._get_polysemy_warning("捨てる・撤退", "company")
        assert warning is not None
        assert warning["polysemy_level"] == "high"
        assert "warning" in warning

    def test_get_polysemy_warning_has_rates(self, engine):
        """高多義性ラベルの警告に成功率情報が含まれること。"""
        warning = engine._get_polysemy_warning("捨てる・撤退", "company")
        assert warning is not None
        assert "scale_success_rate" in warning
        assert "global_success_rate" in warning
        assert "divergence" in warning

    def test_get_polysemy_warning_returns_none_for_unknown(self, engine):
        """未知の action_type には None を返すこと。"""
        warning = engine._get_polysemy_warning("存在しないラベル", "company")
        assert warning is None

    def test_get_polysemy_warning_none_scale(self, engine):
        """scale=None でも高多義性ラベルは警告を返すこと。"""
        warning = engine._get_polysemy_warning("捨てる・撤退", None)
        # scale=None の場合、warning_message があれば返す
        if warning is not None:
            assert warning["polysemy_level"] == "high"

    def test_collect_polysemy_warnings(self, engine):
        """_collect_polysemy_warnings がリストを返すこと。"""
        l2 = engine.reverse_state("停滞・閉塞", "V字回復・大成功", scale="company")
        l3 = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        warnings = engine._collect_polysemy_warnings(l2, l3, "company")
        assert isinstance(warnings, list)

    def test_collect_polysemy_warnings_no_duplicates(self, engine):
        """_collect_polysemy_warnings に重複がないこと。"""
        l2 = engine.reverse_state("停滞・閉塞", "V字回復・大成功", scale="company")
        l3 = engine.reverse_action(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        warnings = engine._collect_polysemy_warnings(l2, l3, "company")
        action_types = [w["action_type"] for w in warnings]
        assert len(action_types) == len(set(action_types)), "重複があってはならない"

    def test_full_backtrace_includes_polysemy_warnings(self, engine):
        """full_backtrace の結果に polysemy_warnings が含まれること。"""
        result = engine.full_backtrace(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company",
        )
        assert "polysemy_warnings" in result
        assert isinstance(result["polysemy_warnings"], list)

    def test_full_backtrace_polysemy_warnings_structure(self, engine):
        """polysemy_warnings の各エントリに必要フィールドがあること。"""
        result = engine.full_backtrace(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company",
        )
        for warning in result["polysemy_warnings"]:
            assert "action_type" in warning
            assert "polysemy_level" in warning
            assert "warning" in warning

    def test_reverse_state_includes_polysemy_info(self, engine):
        """reverse_state の recommended_actions に polysemy_info が付与されること。"""
        result = engine.reverse_state("停滞・閉塞", "V字回復・大成功", scale="company")
        # 少なくとも一部の recommended_actions に polysemy_info があることを確認
        has_polysemy = any(
            "polysemy_info" in a for a in result.get("recommended_actions", [])
        )
        # 高多義性ラベルが推奨に含まれていれば polysemy_info が存在するはず
        # (含まれない場合もあるため、構造チェックのみ)
        for action in result.get("recommended_actions", []):
            if "polysemy_info" in action:
                pi = action["polysemy_info"]
                assert "polysemy_level" in pi
                assert "warning" in pi

    def test_full_backtrace_different_scales_different_warnings(self, engine):
        """異なる scale で polysemy_warnings の内容が変わりうること。"""
        r1 = engine.full_backtrace(
            current_hex=29, current_state="どん底・危機",
            goal_hex=1, goal_state="安定成長・成功",
            scale="company",
        )
        r2 = engine.full_backtrace(
            current_hex=29, current_state="どん底・危機",
            goal_hex=1, goal_state="安定成長・成功",
            scale="individual",
        )
        # 両方とも polysemy_warnings を含む
        assert "polysemy_warnings" in r1
        assert "polysemy_warnings" in r2


# ============================================================
# Analogy Scoring テスト
# ============================================================

class TestAnalogyScoring:
    """AnalogyScorer の単体テスト。"""

    @pytest.fixture(scope="class")
    def scorer(self):
        from analogy_scoring import AnalogyScorer
        return AnalogyScorer()

    def test_same_scale_weight(self, scorer):
        """同一スケールの重みが1.0であること。"""
        assert scorer.compute_scale_weight("company", "company") == 1.0

    def test_adjacent_scale_weight(self, scorer):
        """隣接スケールの重みが0.7であること。"""
        assert scorer.compute_scale_weight("company", "family") == 0.7

    def test_distant_scale_weight(self, scorer):
        """遠いスケールの重みが0.3であること。"""
        assert scorer.compute_scale_weight("company", "individual") == 0.3

    def test_scale_weight_symmetry(self, scorer):
        """スケール距離は対称であること。"""
        assert scorer.compute_scale_weight("individual", "family") == \
               scorer.compute_scale_weight("family", "individual")

    def test_base_similarity_identical(self, scorer):
        """同一事例の類似度が1.0に近いこと。"""
        case = {
            "before_hex_num": 12, "after_hex_num": 11,
            "action_type": "対話・融合",
            "before_state": "停滞・閉塞", "after_state": "安定・平和",
            "outcome": "Success", "scale": "company",
        }
        score = scorer.compute_base_similarity(case, case)
        assert score > 0.9, f"同一事例のbase_similarity={score}は0.9超のはず"

    def test_base_similarity_different(self, scorer):
        """完全に異なる事例の類似度が低いこと。"""
        case_a = {
            "before_hex_num": 1, "after_hex_num": 2,
            "action_type": "攻める・挑戦",
            "before_state": "安定成長・成功", "after_state": "拡大・繁栄",
            "outcome": "Success", "scale": "company",
        }
        case_b = {
            "before_hex_num": 63, "after_hex_num": 64,
            "action_type": "守る・維持",
            "before_state": "どん底・危機", "after_state": "崩壊・消滅",
            "outcome": "Failure", "scale": "individual",
        }
        score = scorer.compute_base_similarity(case_a, case_b)
        assert score < 0.5, f"異なる事例のbase_similarity={score}は0.5未満のはず"

    def test_total_score_penalizes_distant_scale(self, scorer):
        """遠いスケールではtotal_scoreがbase_similarityより低くなること。"""
        case = {
            "before_hex_num": 12, "after_hex_num": 11,
            "action_type": "対話・融合",
            "before_state": "停滞・閉塞", "after_state": "安定・平和",
            "outcome": "Success",
        }
        case_same = {**case, "scale": "company"}
        case_distant = {**case, "scale": "individual"}
        target = {**case, "scale": "company"}

        score_same = scorer.compute_total_score(target, case_same)
        score_distant = scorer.compute_total_score(target, case_distant)
        assert score_same > score_distant, \
            f"same_scale={score_same} should be > distant_scale={score_distant}"

    def test_rank_analogies_ordering(self, scorer):
        """rank_analogiesが類似度降順でソートされること。"""
        target = {
            "before_hex_num": 12, "after_hex_num": 11,
            "action_type": "対話・融合",
            "before_state": "停滞・閉塞", "after_state": "安定・平和",
            "outcome": "Success", "scale": "company",
        }
        candidates = [
            {
                "before_hex_num": 1, "after_hex_num": 2,
                "action_type": "攻める・挑戦",
                "before_state": "安定成長・成功", "after_state": "拡大・繁栄",
                "outcome": "Failure", "scale": "individual",
            },
            {
                "before_hex_num": 12, "after_hex_num": 11,
                "action_type": "対話・融合",
                "before_state": "停滞・閉塞", "after_state": "安定・平和",
                "outcome": "Success", "scale": "family",
            },
        ]
        ranked = scorer.rank_analogies(target, candidates, top_n=10)
        assert len(ranked) == 2
        # family (隣接) の事例はindividual (遠い) よりスコアが高いはず
        assert ranked[0][0] >= ranked[1][0], \
            f"ranked[0]={ranked[0][0]} should be >= ranked[1]={ranked[1][0]}"

    def test_state_similarity_partial_match(self, scorer):
        """「・」区切りの片方一致で0.5が返ること。"""
        case_a = {
            "before_hex_num": 12, "after_hex_num": 11,
            "action_type": "対話・融合",
            "before_state": "停滞・閉塞", "after_state": "安定・平和",
            "outcome": "Success", "scale": "company",
        }
        case_b = {
            "before_hex_num": 12, "after_hex_num": 11,
            "action_type": "対話・融合",
            "before_state": "停滞・混乱", "after_state": "安定・停止",
            "outcome": "Success", "scale": "company",
        }
        # before_state: 「停滞」が共通 → 0.5
        # after_state: 「安定」が共通 → 0.5
        # state_similarity = (0.5 + 0.5) / 2 = 0.5
        sim = scorer._state_similarity(case_a, case_b)
        assert sim == 0.5, f"state_similarity={sim}, expected 0.5"

    def test_rank_analogies_top_n(self, scorer):
        """rank_analogiesのtop_nが正しく機能すること。"""
        target = {
            "before_hex_num": 12, "after_hex_num": 11,
            "action_type": "対話・融合",
            "before_state": "停滞・閉塞", "after_state": "安定・平和",
            "outcome": "Success", "scale": "company",
        }
        candidates = [
            {
                "before_hex_num": i, "after_hex_num": i + 1,
                "action_type": "守る・維持",
                "before_state": "安定成長・成功", "after_state": "安定・平和",
                "outcome": "Success", "scale": "company",
            }
            for i in range(1, 20)
        ]
        ranked = scorer.rank_analogies(target, candidates, top_n=3)
        assert len(ranked) == 3


class TestAnalogyInBacktrace:
    """BacktraceEngine に統合された AnalogyScoring のテスト。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return BacktraceEngine()

    def test_analogy_scorer_exists(self, engine):
        """BacktraceEngineにAnalogyScorer属性があること。"""
        assert hasattr(engine, "_analogy_scorer")
        from analogy_scoring import AnalogyScorer
        assert isinstance(engine._analogy_scorer, AnalogyScorer)

    def test_full_backtrace_cross_scale_has_analogy_score(self, engine):
        """cross_scale_patternsにanalogy_scoreが含まれること。"""
        result = engine.full_backtrace(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", include_cross_scale=True,
        )
        patterns = result.get("cross_scale_patterns", [])
        if patterns:
            for p in patterns:
                assert "analogy_score" in p, \
                    f"pattern {p['pattern']} にanalogy_scoreがない"
                assert 0.0 <= p["analogy_score"] <= 1.0, \
                    f"analogy_score={p['analogy_score']}は0-1の範囲外"

    def test_cross_scale_analogy_score_range(self, engine):
        """cross_scale_patternsのanalogy_scoreが0.3〜0.7の範囲であること。
        (自分自身のscaleは除外されるため、1.0にはならない)"""
        result = engine.full_backtrace(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", include_cross_scale=True,
        )
        patterns = result.get("cross_scale_patterns", [])
        for p in patterns:
            # 自scale (company) は除外されるため、analogy_scoreは最大でも0.7(隣接)
            assert p["analogy_score"] <= 0.7 + 0.01, \
                f"analogy_score={p['analogy_score']}が0.7を超過"

    def test_find_analogous_cases_returns_list(self, engine):
        """find_analogous_casesがリストを返すこと。"""
        results = engine.find_analogous_cases(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", top_n=5,
        )
        assert isinstance(results, list)

    def test_find_analogous_cases_fields(self, engine):
        """find_analogous_casesの結果に必要なフィールドがあること。"""
        results = engine.find_analogous_cases(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", top_n=5,
        )
        if results:
            required_fields = {"score", "target_name", "scale",
                               "before_state", "action_type",
                               "after_state", "outcome"}
            for r in results:
                for field in required_fields:
                    assert field in r, f"結果に{field}フィールドがない"

    def test_find_analogous_cases_score_ordering(self, engine):
        """find_analogous_casesの結果がスコア降順であること。"""
        results = engine.find_analogous_cases(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", top_n=10,
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"], \
                    f"results[{i}].score={results[i]['score']} < " \
                    f"results[{i+1}].score={results[i+1]['score']}"

    def test_find_analogous_cases_top_n(self, engine):
        """find_analogous_casesのtop_n制限が機能すること。"""
        results = engine.find_analogous_cases(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", top_n=3,
        )
        assert len(results) <= 3

    def test_full_backtrace_without_cross_scale_no_analogy(self, engine):
        """include_cross_scale=Falseのときcross_scale_patternsが含まれないこと。"""
        result = engine.full_backtrace(
            current_hex=12, current_state="停滞・閉塞",
            goal_hex=11, goal_state="安定・平和",
            scale="company", include_cross_scale=False,
        )
        assert "cross_scale_patterns" not in result


# ============================================================
# P1: 推論ログ（reasoning_log）のテスト
# ============================================================

class TestReasoningLog:
    """P1: 推論ログ（reasoning_log）のテスト。"""

    @pytest.fixture
    def engine(self):
        return BacktraceEngine()

    def test_reasoning_log_exists_in_full_backtrace(self, engine):
        """reasoning_logキーがfull_backtrace()出力に存在する。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        assert "reasoning_log" in result

    def test_ekikyo_structure_analysis(self, engine):
        """易経構造分析に現在卦、目標卦、変爻、之卦が含まれる。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        rl = result["reasoning_log"]
        assert "易経構造分析" in rl
        ea = rl["易経構造分析"]
        assert "現在卦" in ea
        assert "目標卦" in ea
        assert "変爻" in ea
        assert "之卦" in ea
        # 現在卦の構造
        assert "id" in ea["現在卦"]
        assert "name" in ea["現在卦"]
        assert "upper_trigram" in ea["現在卦"]
        assert "lower_trigram" in ea["現在卦"]
        assert ea["現在卦"]["id"] == 12
        assert ea["目標卦"]["id"] == 11

    def test_case_data_analysis(self, engine):
        """事例データ分析にヒット件数、到達確率が含まれる。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        rl = result["reasoning_log"]
        assert "事例データ分析" in rl
        cda = rl["事例データ分析"]
        assert "ヒット件数" in cda
        assert "到達確率" in cda
        assert isinstance(cda["ヒット件数"], int)

    def test_integration_judgment(self, engine):
        """統合判断にスコア内訳が含まれる。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        rl = result["reasoning_log"]
        assert "統合判断" in rl
        ij = rl["統合判断"]
        assert "スコア内訳" in ij
        assert "ルート選択根拠" in ij

    def test_hu_gua_in_reasoning_log(self, engine):
        """reasoning_log.易経構造分析に互卦と目標互卦が含まれる。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        ea = result["reasoning_log"]["易経構造分析"]
        assert "互卦" in ea
        assert "目標互卦" in ea
        assert "hu_gua_id" in ea["互卦"]
        assert "hu_gua_name" in ea["互卦"]


# ============================================================
# P2: 語彙置換のテスト
# ============================================================

class TestVocabulary:
    """P2: 語彙置換のテスト。"""

    @pytest.fixture
    def engine(self):
        return BacktraceEngine()

    def test_display_title_format(self, engine):
        """display_titleが'選択肢A', '選択肢B'等の形式。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        routes = result.get("recommended_routes", [])
        if routes:
            assert routes[0].get("display_title", "").startswith("選択肢")

    def test_score_display_exists(self, engine):
        """score_display, success_rate_displayが存在し日本語ラベル付き。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        routes = result.get("recommended_routes", [])
        if routes:
            assert "score_display" in routes[0]
            assert "success_rate_display" in routes[0]
            assert "構造的適合度" in routes[0]["score_display"]
            assert "到達確率" in routes[0]["success_rate_display"]

    def test_existing_keys_preserved(self, engine):
        """既存キー（score, title等）が壊れていない（回帰テスト）。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        routes = result.get("recommended_routes", [])
        if routes:
            route = routes[0]
            assert "score" in route
            assert "title" in route
            assert "route" in route
            assert "confidence_interval" in route
            assert isinstance(route["score"], float)

    def test_vocabulary_map_exists(self):
        """_VOCABULARY_MAPが定義されている。"""
        from backtrace_engine import _VOCABULARY_MAP
        assert "成功率" in _VOCABULARY_MAP
        assert _VOCABULARY_MAP["成功率"] == "到達確率"
        assert "推奨ルート" in _VOCABULARY_MAP
        assert _VOCABULARY_MAP["推奨ルート"] == "選択肢"


# ============================================================
# P3: 数字の定義書のテスト
# ============================================================

class TestNumberDefinition:
    """P3: 数字の定義書のテスト。"""

    def test_build_number_definition_basic(self):
        """_build_number_definitionが正しい構造を返す。"""
        nd = _build_number_definition("到達確率", 0.204, 721, "テスト定義", successes=147)
        assert nd["label"] == "到達確率"
        assert nd["value"] == 0.204
        assert nd["n"] == 721
        assert nd["definition"] == "テスト定義"
        assert "confidence_interval" in nd
        assert "lower" in nd["confidence_interval"]
        assert "upper" in nd["confidence_interval"]
        assert "data_quality" in nd

    def test_data_quality_grades(self):
        """data_qualityがn>=100でA、30-99でB、<30でC。"""
        assert _build_number_definition("t", 0.5, 100, "d")["data_quality"] == "A"
        assert _build_number_definition("t", 0.5, 500, "d")["data_quality"] == "A"
        assert _build_number_definition("t", 0.5, 30, "d")["data_quality"] == "B"
        assert _build_number_definition("t", 0.5, 99, "d")["data_quality"] == "B"
        assert _build_number_definition("t", 0.5, 29, "d")["data_quality"] == "C"
        assert _build_number_definition("t", 0.5, 1, "d")["data_quality"] == "C"

    def test_display_format(self):
        """displayフォーマットが「類似N件中M件が到達（X%）」。"""
        nd = _build_number_definition("到達確率", 0.204, 721, "d", successes=147)
        assert "類似721件中147件が到達" in nd["display"]
        assert "20.4%" in nd["display"]

    def test_number_definitions_in_scored_routes(self):
        """各scored_routeにnumber_definitionsキーが存在する。"""
        engine = BacktraceEngine()
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        routes = result.get("recommended_routes", [])
        if routes:
            route = routes[0]
            assert "number_definitions" in route
            assert "score" in route["number_definitions"]
            assert "success_rate" in route["number_definitions"]
            nd_score = route["number_definitions"]["score"]
            assert "n" in nd_score
            assert "definition" in nd_score
            assert "confidence_interval" in nd_score
            assert "data_quality" in nd_score

    def test_summary_number_definitions(self):
        """summaryにnumber_definitionsが含まれる。"""
        engine = BacktraceEngine()
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和", scale="company")
        summary = result.get("summary", {})
        assert "number_definitions" in summary
        assert "primary_route_score" in summary["number_definitions"]

    def test_zero_n_handling(self):
        """n=0の場合にエラーにならない。"""
        nd = _build_number_definition("t", 0.0, 0, "d")
        assert nd["n"] == 0
        assert nd["data_quality"] == "C"
        assert nd["confidence_interval"]["lower"] == 0.0
        assert nd["confidence_interval"]["upper"] == 0.0


# ============================================================
# P4: 互卦による潜在要因抽出のテスト
# ============================================================

class TestHuGua:
    """P4: 互卦による潜在要因抽出のテスト。"""

    @pytest.fixture
    def engine(self):
        return BacktraceEngine()

    def test_analyze_hu_gua_returns_structure(self, engine):
        """_analyze_hu_guaが正しい構造を返す。"""
        result = engine._analyze_hu_gua(1)
        assert "hu_gua_id" in result
        assert "hu_gua_name" in result
        assert "潜在要因" in result
        assert "示唆" in result
        assert "対処の方向性" in result
        assert "爻構成" in result
        assert "下卦" in result["爻構成"]
        assert "上卦" in result["爻構成"]

    def test_hu_gua_name_not_empty(self, engine):
        """hu_gua_nameが空でない。"""
        for hex_num in [1, 2, 11, 12, 29, 63, 64]:
            result = engine._analyze_hu_gua(hex_num)
            assert result["hu_gua_name"], f"hu_gua_name is empty for hexagram {hex_num}"

    def test_hu_gua_id_is_valid(self, engine):
        """hu_gua_idが1-64の範囲。"""
        for hex_num in range(1, 65):
            result = engine._analyze_hu_gua(hex_num)
            assert 1 <= result["hu_gua_id"] <= 64, f"Invalid hu_gua_id for hexagram {hex_num}"
