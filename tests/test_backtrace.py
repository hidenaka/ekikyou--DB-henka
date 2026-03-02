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

from backtrace_engine import BacktraceEngine, _wilson_score_interval, _sanitize_text
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
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        assert "l1_yao" in result
        assert "l2_state" in result
        assert "l3_action" in result
        assert "recommended_routes" in result
        assert "quality_gates" in result
        assert "summary" in result

    def test_summary_fields(self, engine):
        """summary に必要なフィールドが含まれること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        summary = result["summary"]
        expected_keys = {
            "primary_route_score", "alternative_count",
            "confidence_level",
            "current_hex", "goal_hex",
        }
        assert expected_keys.issubset(summary.keys())

    def test_quality_gates_all_present(self, engine):
        """rq1-rq7 の全7項目が存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        qg = result["quality_gates"]
        expected_rq_keys = {
            "rq1_reference_only",
            "rq2_low_success_rate",
            "rq3_no_deterministic_words",
            "rq4_has_alternative_route",
            "rq5_confidence_interval_computed",
            "rq6_fallback_used",
            "rq7_contradictory_routes_excluded",
        }
        assert expected_rq_keys.issubset(qg.keys()), (
            f"Missing quality gates: {expected_rq_keys - qg.keys()}"
        )

    def test_rq3_no_deterministic_words(self, engine):
        """rq3_no_deterministic_words=True（決定論的表現排除）。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        assert result["quality_gates"]["rq3_no_deterministic_words"] is True

    def test_rq4_has_alternative(self, engine):
        """rq4_has_alternative_route=True（代替ルートが常に1つ以上）。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        assert result["quality_gates"]["rq4_has_alternative_route"] is True

    def test_rq5_confidence_interval(self, engine):
        """全ルートに confidence_interval が存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        for route in result["recommended_routes"]:
            assert "confidence_interval" in route

    def test_rq7_no_contradictory(self, engine):
        """rq7_contradictory_routes_excluded=True。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        assert result["quality_gates"]["rq7_contradictory_routes_excluded"] is True

    def test_recommended_routes_scored(self, engine):
        """各ルートに score, labels, confidence_interval が存在すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        for route in result["recommended_routes"]:
            assert "score" in route
            assert "labels" in route
            assert "confidence_interval" in route

    def test_confidence_level_values(self, engine):
        """confidence_level が 'high'/'medium'/'low'/'very_low' のいずれかであること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
        assert result["summary"]["confidence_level"] in {
            "high", "medium", "low", "very_low"
        }

    def test_scenario_pi_to_tai(self, engine):
        """天地否(12)->地天泰(11) のシナリオが正常動作すること。"""
        result = engine.full_backtrace(12, "停滞・閉塞", 11, "安定・平和")
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
            session, current_hex=12, current_state="停滞・閉塞"
        )
        assert "error" not in result
        assert result["current_hex"] == 12
        assert result["current_state"] == "停滞・閉塞"
        assert result["phase"] == "analyzing"

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
            action_type="慎重・観察"
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
        """feedback_layers に r1-r5 が全て存在すること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        assert "error" not in result
        fl = result["feedback_layers"]
        for key in ("r1", "r2", "r3", "r4", "r5"):
            assert key in fl, f"feedback_layers missing '{key}'"

    def test_r1_goal_metadata(self, orchestrator):
        """r1 に goal_name, goal_metadata (meaning, situation, keywords) が存在すること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r1 = result["feedback_layers"]["r1"]
        assert "goal_name" in r1
        assert "goal_metadata" in r1
        meta = r1["goal_metadata"]
        assert "meaning" in meta
        assert "situation" in meta
        assert "keywords" in meta

    def test_r2_gap_structure(self, orchestrator):
        """r2 に hamming_distance, changing_lines, difficulty が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r2 = result["feedback_layers"]["r2"]
        assert "hamming_distance" in r2
        assert "changing_lines" in r2
        assert "difficulty" in r2

    def test_r3_recommended_routes(self, orchestrator):
        """r3 に routes (リスト), route_count が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r3 = result["feedback_layers"]["r3"]
        assert "routes" in r3
        assert isinstance(r3["routes"], list)
        assert "route_count" in r3

    def test_r4_action_patterns(self, orchestrator):
        """r4 に recommended_actions_l3, recommended_actions_l2 が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r4 = result["feedback_layers"]["r4"]
        assert "recommended_actions_l3" in r4
        assert "recommended_actions_l2" in r4

    def test_r5_reflective_question(self, orchestrator):
        """r5 に reflective_question (空でない文字列) が含まれること。"""
        session = self._setup_session(orchestrator)
        result = orchestrator.analyze(session)
        r5 = result["feedback_layers"]["r5"]
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
            action_type="慎重・観察"
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
            session, current_hex=12, current_state="停滞・閉塞"
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
            action_type="慎重・観察"
        )
        assert "error" not in r3
        assert r3["phase"] == "analyzing"

        # Step 5: analyze
        r4 = orchestrator.analyze(session)
        assert "error" not in r4
        assert r4["phase"] == "result"

        # R1-R5 全層が存在
        fl = r4["feedback_layers"]
        for key in ("r1", "r2", "r3", "r4", "r5"):
            assert key in fl

        # quality_gates に rq1-rq7
        qg = r4["quality_gates"]
        for rq in ("rq1_reference_only", "rq2_low_success_rate",
                    "rq3_no_deterministic_words", "rq4_has_alternative_route",
                    "rq5_confidence_interval_computed", "rq6_fallback_used",
                    "rq7_contradictory_routes_excluded"):
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
            action_type="慎重・観察"
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
        for key in ("r1", "r2", "r3", "r4", "r5"):
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
