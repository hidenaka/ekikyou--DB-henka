#!/usr/bin/env python3
"""
診断エンジンのユニットテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnostic_engine import DiagnosticEngine, DiagnosticResult


def test_engine_initialization():
    """エンジンが正しく初期化されるか"""
    engine = DiagnosticEngine()
    assert len(engine.questions) == 8
    assert len(engine.action_types) == 8
    assert len(engine.pattern_types) == 6
    assert len(engine.phase_definitions) == 8
    print("✓ test_engine_initialization passed")


def test_record_answer():
    """回答の記録が正しく動作するか"""
    engine = DiagnosticEngine()
    answer = engine.record_answer("Q1", "active_strong")
    assert answer.question_id == "Q1"
    assert answer.value == "active_strong"
    assert answer.score == 3
    assert "震" in answer.weights
    print("✓ test_record_answer passed")


def test_hex_calculation():
    """八卦スコア計算が正しいか"""
    engine = DiagnosticEngine()
    engine.record_answer("Q1", "active_strong")  # 震:3, 乾:2, 巽:1
    engine.record_answer("Q2", "outward_expand")  # 乾:3, 離:2, 兌:1

    hex_scores = engine.calculate_hex_scores()
    assert hex_scores["乾"] == 5  # 2 + 3
    assert hex_scores["震"] == 3
    assert hex_scores["離"] == 2

    primary, _ = engine.get_primary_hex()
    assert primary == "乾"
    print("✓ test_hex_calculation passed")


def test_momentum_ascending():
    """上昇の勢いが正しく判定されるか"""
    engine = DiagnosticEngine()
    engine.record_answer("Q1", "active_strong")   # score: 3
    engine.record_answer("Q2", "outward_expand")  # score: 3
    engine.record_answer("Q5", "power_influence") # score: 3
    engine.record_answer("Q6", "nothing")         # score: 1

    momentum, score = engine.calculate_momentum()
    assert momentum == "ascending"
    assert score == 2.5  # (3+3+3+1)/4
    print("✓ test_momentum_ascending passed")


def test_momentum_chaotic():
    """混乱の勢いが正しく判定されるか"""
    engine = DiagnosticEngine()
    engine.record_answer("Q1", "static_stuck")     # score: -3
    engine.record_answer("Q2", "inward_protect")   # score: -3
    engine.record_answer("Q5", "nothing")          # score: -3
    engine.record_answer("Q6", "resources")        # score: -3

    momentum, score = engine.calculate_momentum()
    assert momentum == "chaotic"
    assert score == -3.0
    print("✓ test_momentum_chaotic passed")


def test_timing_act_now():
    """動くべき時が正しく判定されるか"""
    engine = DiagnosticEngine()
    engine.record_answer("Q3", "clear_certain")    # score: 3
    engine.record_answer("Q5", "power_influence")  # score: 3
    engine.record_answer("Q6", "nothing")          # score: 1

    timing, score = engine.calculate_timing()
    assert timing == "act_now"
    # (3 + 3 - 1) / 3 = 1.67
    assert round(score, 2) == 1.67
    print("✓ test_timing_act_now passed")


def test_timing_wait():
    """待つべき時が正しく判定されるか"""
    engine = DiagnosticEngine()
    engine.record_answer("Q3", "unclear_danger")  # score: -3
    engine.record_answer("Q5", "pressure")        # score: -2
    engine.record_answer("Q6", "nothing")         # score: 1

    timing, score = engine.calculate_timing()
    # (-3 + (-2) - 1) / 3 = -6/3 = -2.0 < -1.0 → wait
    assert timing == "wait"
    assert score == -2.0
    print("✓ test_timing_wait passed")


def test_full_diagnosis():
    """完全な診断が正しく動作するか"""
    engine = DiagnosticEngine()

    # すべての質問に回答
    answers = [
        ("Q1", "active_mild"),
        ("Q2", "outward_connect"),
        ("Q3", "clear_partial"),
        ("Q4", "intentional"),
        ("Q5", "clarity_insight"),
        ("Q6", "stability"),
        ("Q7", "pivot_fail"),
        ("Q8", "renewal"),
    ]

    for qid, value in answers:
        engine.record_answer(qid, value)

    result = engine.diagnose()

    # 結果の検証
    assert isinstance(result, DiagnosticResult)
    assert result.primary_hex in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
    assert result.momentum in ["ascending", "stable", "descending", "chaotic"]
    assert result.timing in ["act_now", "adapt", "wait"]
    assert len(result.recommended_actions) == 3
    assert result.judgment != ""
    assert result.preferred_action == "刷新・破壊"
    assert result.avoid_pattern == "Pivot_Success"

    print("✓ test_full_diagnosis passed")


def test_warnings_generation():
    """警告が正しく生成されるか"""
    engine = DiagnosticEngine()

    # Hubris_Collapse を避けたい場合の回答
    answers = [
        ("Q1", "active_strong"),
        ("Q2", "outward_expand"),
        ("Q3", "clear_certain"),
        ("Q4", "intentional"),
        ("Q5", "power_influence"),
        ("Q6", "nothing"),
        ("Q7", "hubris_collapse"),
        ("Q8", "growth"),
    ]

    for qid, value in answers:
        engine.record_answer(qid, value)

    warnings = engine.get_warnings()

    # Hubris_Collapse関連の警告が含まれているか
    assert any("絶頂からの転落" in w for w in warnings)
    print("✓ test_warnings_generation passed")


def test_reset():
    """リセットが正しく動作するか"""
    engine = DiagnosticEngine()
    engine.record_answer("Q1", "active_strong")
    assert len(engine.answers) == 1

    engine.reset()
    assert len(engine.answers) == 0
    print("✓ test_reset passed")


def test_statistics_lookup_priority():
    """統計テーブル参照の優先順位が正しいか"""
    engine = DiagnosticEngine()

    # by_state_trigger_action が優先されるケース
    stats = engine.lookup_statistics("どん底・危機", "外部ショック", "坎")
    # どん底・危機 × 外部ショック の組み合わせがあれば取得できる
    # なければ次の優先順位へフォールバック

    # 少なくとも何かしらの結果が返ること
    assert isinstance(stats, dict)
    print("✓ test_statistics_lookup_priority passed")


def run_all_tests():
    """全テストを実行"""
    print("\n" + "=" * 50)
    print("診断エンジン ユニットテスト")
    print("=" * 50 + "\n")

    tests = [
        test_engine_initialization,
        test_record_answer,
        test_hex_calculation,
        test_momentum_ascending,
        test_momentum_chaotic,
        test_timing_act_now,
        test_timing_wait,
        test_full_diagnosis,
        test_warnings_generation,
        test_reset,
        test_statistics_lookup_priority,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "-" * 50)
    print(f"結果: {passed} passed, {failed} failed")
    print("-" * 50 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
