#!/usr/bin/env python3
"""
日記モード テスト — GapAnalysisEngine / DiaryExtractionEngine /
ChangeAdviceEngine / DiarySessionOrchestrator / Flask API

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

from gap_analysis_engine import GapAnalysisEngine
from diary_extraction_engine import (
    DiaryExtractionEngine,
    DIARY_DEMO_SCENARIOS,
    DIARY_DEMO_KEYWORDS,
)
from change_advice_engine import ChangeAdviceEngine, DEMO_ADVICE, _REQUIRED_KEYS


# ============================================================
# 1. GapAnalysisEngine テスト（純粋計算、LLM/モック不要）
# ============================================================

class TestGapAnalysisEngine:
    """GapAnalysisEngine の全メソッドをテストする。"""

    @pytest.fixture(scope="class")
    def engine(self):
        return GapAnalysisEngine()

    def test_analyze_returns_valid_structure(self, engine):
        """基本分析の戻り値に全ての期待キーが含まれること。"""
        result = engine.analyze(1, 2)
        expected_keys = {
            "hexagram_a", "hexagram_g",
            "hamming_distance", "changing_lines",
            "difficulty", "difficulty_score",
            "compatibility", "structural_relationship",
            "intermediate_paths", "trigram_changes",
        }
        assert expected_keys.issubset(result.keys())
        # hexagram_a / hexagram_g の内部構造
        assert "number" in result["hexagram_a"]
        assert "name" in result["hexagram_a"]
        assert "lines" in result["hexagram_a"]

    def test_identical_hexagram(self, engine):
        """A == G のとき hamming=0, relationship=identical, 中間経路なし。"""
        result = engine.analyze(1, 1)
        assert result["hamming_distance"] == 0
        assert result["changing_lines"] == []
        assert result["structural_relationship"] == "identical"
        assert result["intermediate_paths"] == []
        assert result["difficulty"] == "easy"

    def test_known_cuo_gua(self, engine):
        """1 -> 2 は全爻反転なので cuo_gua（錯卦）と検出されること。"""
        result = engine.analyze(1, 2)
        assert result["structural_relationship"] == "cuo_gua"
        assert result["hamming_distance"] == 6

    def test_known_zong_gua(self, engine):
        """3 -> 4 は上下反転のみ（錯卦ではない）なので zong_gua（綜卦）と検出されること。
        注意: 11->12 は全爻反転でもあるため cuo_gua が先に検出される。
        """
        result = engine.analyze(3, 4)
        assert result["structural_relationship"] == "zong_gua"

    def test_difficulty_labels(self, engine):
        """ハミング距離と難易度ラベルの対応を検証する。
        hamming 0-1 = easy, 2-3 = moderate, 4-6 = hard
        """
        # hamming 0 -> easy (同一卦)
        r0 = engine.analyze(1, 1)
        assert r0["difficulty"] == "easy"

        # hamming 1 -> easy (之卦: 1爻変)
        # 卦1 の第5爻変 -> 卦14
        r1 = engine.analyze(1, 14)
        assert r1["hamming_distance"] == 1
        assert r1["difficulty"] == "easy"

        # hamming 6 -> hard (全爻反転: 1 -> 2)
        r6 = engine.analyze(1, 2)
        assert r6["hamming_distance"] == 6
        assert r6["difficulty"] == "hard"

    def test_all_64x64_no_error(self, engine):
        """全 64*64 = 4096 ペアで例外が発生しないことを検証する。"""
        errors = []
        for a in range(1, 65):
            for g in range(1, 65):
                try:
                    result = engine.analyze(a, g)
                    # 最低限の構造チェック
                    assert isinstance(result["hamming_distance"], int)
                    assert 0 <= result["hamming_distance"] <= 6
                except Exception as e:
                    errors.append(f"analyze({a}, {g}) failed: {e}")
        assert errors == [], f"{len(errors)} 件のエラー: {errors[:5]}"

    def test_intermediate_paths_count(self, engine):
        """中間経路は最大3件であること。"""
        for a in range(1, 65):
            for g in range(1, 65):
                result = engine.analyze(a, g)
                assert len(result["intermediate_paths"]) <= 3, (
                    f"analyze({a}, {g}): intermediate_paths が "
                    f"{len(result['intermediate_paths'])} 件"
                )

    def test_trigram_changes(self, engine):
        """trigram_changes の構造検証。"""
        result = engine.analyze(1, 2)
        tri = result["trigram_changes"]
        assert "lower" in tri
        assert "upper" in tri
        for side in ("lower", "upper"):
            assert "from" in tri[side]
            assert "to" in tri[side]
            assert "changed" in tri[side]
            assert isinstance(tri[side]["changed"], bool)
        # 1 -> 2 は全爻反転なので上卦・下卦とも変化する
        assert tri["lower"]["changed"] is True
        assert tri["upper"]["changed"] is True

    def test_invalid_hexagram_raises(self, engine):
        """範囲外の卦番号で ValueError が送出されること。"""
        with pytest.raises(ValueError):
            engine.analyze(0, 1)
        with pytest.raises(ValueError):
            engine.analyze(1, 65)
        with pytest.raises(ValueError):
            engine.analyze(-1, 10)


# ============================================================
# 2. DiaryExtractionEngine テスト（デモモード、APIキーなし）
# ============================================================

class TestDiaryExtractionEngine:
    """DiaryExtractionEngine のデモモード動作を検証する。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        """テスト中は ANTHROPIC_API_KEY を確実に除去する。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture
    def engine(self):
        return DiaryExtractionEngine()

    def test_extract_dual_demo_mode(self, engine):
        """APIキー未設定時、extract_dual はデモデータを返すこと。"""
        result = engine.extract_dual("今日もまた同じ一日だった。停滞している。")
        assert result is not None
        assert "current" in result
        assert "ideal" in result
        assert "diary_meta" in result

    def test_demo_keyword_matching_frustration(self, engine):
        """キーワード「停滞」で frustration シナリオが選択されること。"""
        result = engine.extract_dual("停滞している日々が続く")
        expected = DIARY_DEMO_SCENARIOS["frustration"]
        assert result["current"]["current_state"]["primary"] == \
            expected["current"]["current_state"]["primary"]

    def test_demo_keyword_matching_crossroads(self, engine):
        """キーワード「転職」で crossroads シナリオが選択されること。"""
        result = engine.extract_dual("転職するか迷っている")
        expected = DIARY_DEMO_SCENARIOS["crossroads"]
        assert result["current"]["current_state"]["primary"] == \
            expected["current"]["current_state"]["primary"]

    def test_demo_keyword_matching_emptiness(self, engine):
        """キーワード「空虚」で emptiness シナリオが選択されること。"""
        result = engine.extract_dual("毎日が空虚に感じる")
        expected = DIARY_DEMO_SCENARIOS["emptiness"]
        assert result["current"]["current_state"]["primary"] == \
            expected["current"]["current_state"]["primary"]

    def test_demo_default_fallback(self, engine):
        """キーワードにマッチしない場合、frustration がフォールバックとして選ばれること。"""
        result = engine.extract_dual("今日は天気が良かった")
        expected = DIARY_DEMO_SCENARIOS["frustration"]
        assert result["current"]["current_state"]["primary"] == \
            expected["current"]["current_state"]["primary"]

    def test_assess_dual_confidence(self, engine):
        """assess_dual_confidence の戻り値構造を検証する。"""
        dual = engine.extract_dual("停滞している日々")
        assessment = engine.assess_dual_confidence(dual)
        assert "current_assessment" in assessment
        assert "ideal_assessment" in assessment
        assert "gap_clarity" in assessment
        assert "needs_ideal_followup" in assessment
        assert "ideal_followup_question" in assessment
        # current_assessment の内部構造
        cur = assessment["current_assessment"]
        assert "action" in cur
        assert "low_axes" in cur
        assert "overall_confidence" in cur
        assert cur["action"] in ("proceed", "follow_up", "broad_follow_up")

    def test_summarize_dual(self, engine):
        """summarize_dual の戻り値に3つのサマリーが含まれること。"""
        dual = engine.extract_dual("停滞している日々")
        summaries = engine.summarize_dual(dual)
        assert "current_summary" in summaries
        assert "ideal_summary" in summaries
        assert "gap_summary" in summaries
        # 各サマリーは空でない文字列であること
        assert isinstance(summaries["current_summary"], str)
        assert len(summaries["current_summary"]) > 0
        assert isinstance(summaries["ideal_summary"], str)
        assert len(summaries["ideal_summary"]) > 0
        assert isinstance(summaries["gap_summary"], str)
        assert len(summaries["gap_summary"]) > 0

    def test_map_to_hexagrams(self, engine):
        """map_to_hexagrams が current_candidates(3件) と goal_hexagram(1件) を返すこと。"""
        dual = engine.extract_dual("停滞している日々")
        hexagrams = engine.map_to_hexagrams(dual)
        assert "current_candidates" in hexagrams
        assert "goal_hexagram" in hexagrams
        assert "current_db_labels" in hexagrams
        assert "ideal_db_labels" in hexagrams
        # current_candidates は 3 件
        assert len(hexagrams["current_candidates"]) == 3
        # goal_hexagram は 1 件（dict）
        assert hexagrams["goal_hexagram"] is not None
        assert isinstance(hexagrams["goal_hexagram"], dict)
        # 各候補に hexagram_number がある
        for c in hexagrams["current_candidates"]:
            assert "hexagram_number" in c
            assert 1 <= c["hexagram_number"] <= 64

    def test_extract_dual_empty_text(self, engine):
        """空テキストでは None が返ること。"""
        assert engine.extract_dual("") is None
        assert engine.extract_dual("   ") is None

    def test_is_available_returns_false(self, engine):
        """APIキー未設定時、is_available() は False を返すこと。"""
        assert engine.is_available() is False


# ============================================================
# 3. ChangeAdviceEngine テスト（デモモード）
# ============================================================

class TestChangeAdviceEngine:
    """ChangeAdviceEngine のデモモード動作と品質ゲートを検証する。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture
    def engine(self):
        return ChangeAdviceEngine()

    @pytest.fixture
    def gap_engine(self):
        return GapAnalysisEngine()

    def _make_gap(self, gap_engine, a, g):
        return gap_engine.analyze(a, g)

    def _make_diary_meta(self):
        return {
            "emotional_tone": "mixed",
            "gap_clarity": 0.65,
            "key_tension": "現状の安定を保ちたい気持ちと、変化への渇望が拮抗している",
            "domain": "個人",
        }

    def test_demo_advice_easy(self, engine):
        """easy 難易度のデモアドバイスに 7 つの必須キーが含まれること。"""
        advice = DEMO_ADVICE["easy"]
        for key in _REQUIRED_KEYS:
            assert key in advice, f"easy に '{key}' が欠落"
            assert isinstance(advice[key], str) and len(advice[key]) > 0

    def test_demo_advice_moderate(self, engine):
        """moderate 難易度のデモアドバイスに 7 つの必須キーが含まれること。"""
        advice = DEMO_ADVICE["moderate"]
        for key in _REQUIRED_KEYS:
            assert key in advice, f"moderate に '{key}' が欠落"
            assert isinstance(advice[key], str) and len(advice[key]) > 0

    def test_demo_advice_hard(self, engine):
        """hard 難易度のデモアドバイスに 7 つの必須キーが含まれること。"""
        advice = DEMO_ADVICE["hard"]
        for key in _REQUIRED_KEYS:
            assert key in advice, f"hard に '{key}' が欠落"
            assert isinstance(advice[key], str) and len(advice[key]) > 0

    def test_quality_gate_pass(self, engine):
        """全デモアドバイステンプレートが品質ゲートを通過すること。"""
        for difficulty, advice in DEMO_ADVICE.items():
            is_valid, issues = engine._validate_advice(advice)
            assert is_valid, (
                f"DEMO_ADVICE['{difficulty}'] が品質ゲート不通過: {issues}"
            )

    def test_quality_gate_prohibited_words(self, engine):
        """禁止ワードを含むアドバイスが品質ゲートで不合格になること。"""
        bad_advice = dict(DEMO_ADVICE["easy"])
        bad_advice["core_tension"] = "頑張ってください！必ず運気が上がります"
        is_valid, issues = engine._validate_advice(bad_advice)
        assert not is_valid
        # 禁止ワード関連の issue が含まれること
        prohibited_found = [i for i in issues if "禁止ワード" in i]
        assert len(prohibited_found) > 0

    def test_generate_advice_demo_mode(self, engine, gap_engine):
        """APIキー未設定時、generate_advice がデモアドバイスを返すこと。"""
        gap = self._make_gap(gap_engine, 12, 11)
        meta = self._make_diary_meta()
        advice = engine.generate_advice(
            hexagram_a=12, hexagram_g=11,
            gap_analysis=gap, diary_meta=meta,
        )
        assert advice is not None
        for key in _REQUIRED_KEYS:
            assert key in advice
        # key_tension が 5 文字以上のため core_tension に反映される
        assert advice["core_tension"] == meta["key_tension"]

    def test_generate_advice_invalid_hexagram(self, engine, gap_engine):
        """不正な卦番号で None が返ること。"""
        gap = self._make_gap(gap_engine, 1, 2)
        meta = self._make_diary_meta()
        assert engine.generate_advice(0, 2, gap, meta) is None
        assert engine.generate_advice(1, 65, gap, meta) is None


# ============================================================
# 4. DiarySessionOrchestrator テスト（統合、デモモード）
# ============================================================

class TestDiarySessionOrchestrator:
    """DiarySessionOrchestrator の全フローをデモモードで検証する。"""

    @pytest.fixture(autouse=True)
    def unset_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    @pytest.fixture
    def orchestrator(self):
        from diary_session_orchestrator import DiarySessionOrchestrator
        return DiarySessionOrchestrator()

    def test_create_session(self, orchestrator):
        """作成されたセッションが正しい初期状態を持つこと。"""
        session = orchestrator.create_session()
        assert session["mode"] == "diary"
        assert session["phase"] == "input"
        assert session["user_text"] == ""
        assert session["dual_extraction"] is None
        assert session["assessment"] is None
        assert session["summaries"] is None
        assert session["hexagram_mapping"] is None
        assert session["selected_candidate_index"] == 0
        assert session["gap_analysis"] is None
        assert session["feedback"] is None
        assert session["change_advice"] is None
        assert session["followup_count"] == 0

    def test_extract_diary_updates_session(self, orchestrator):
        """extract_diary がセッションの phase を diary-reviewing に更新すること。"""
        session = orchestrator.create_session()
        result = orchestrator.extract_diary(session, "停滞している日々が続く")
        assert "error" not in result
        assert result["phase"] == "diary-reviewing"
        assert session["phase"] == "diary-reviewing"
        assert session["dual_extraction"] is not None
        assert session["assessment"] is not None
        assert session["summaries"] is not None

    def test_confirm_dual_updates_session(self, orchestrator):
        """confirm_dual がセッションの phase を diary-confirmed に更新すること。"""
        session = orchestrator.create_session()
        orchestrator.extract_diary(session, "停滞している日々が続く")
        result = orchestrator.confirm_dual(session)
        assert "error" not in result
        assert result["phase"] == "diary-confirmed"
        assert session["phase"] == "diary-confirmed"
        assert "current_candidates" in result
        assert "goal_hexagram" in result
        assert "gap_analysis" in result

    def test_generate_change_advice_updates_session(self, orchestrator):
        """generate_change_advice がセッションの phase を diary-result に更新すること。"""
        session = orchestrator.create_session()
        orchestrator.extract_diary(session, "停滞している日々が続く")
        orchestrator.confirm_dual(session)
        result = orchestrator.generate_change_advice(session)
        assert "error" not in result
        assert result["phase"] == "diary-result"
        assert session["phase"] == "diary-result"
        assert "feedback" in result
        assert "change_advice" in result
        assert result["feedback"] is not None

    def test_full_flow_demo(self, orchestrator):
        """フルフロー: extract -> confirm -> advice がデモモードで完走すること。"""
        session = orchestrator.create_session()

        # Phase 1: extract
        r1 = orchestrator.extract_diary(session, "毎日が退屈で停滞している")
        assert "error" not in r1
        assert r1.get("demo_mode") is True

        # Phase 2: confirm
        r2 = orchestrator.confirm_dual(session)
        assert "error" not in r2
        candidates = r2["current_candidates"]
        assert len(candidates) >= 1
        goal = r2["goal_hexagram"]
        assert goal is not None

        # Phase 3: advice
        r3 = orchestrator.generate_change_advice(session)
        assert "error" not in r3
        advice = r3["change_advice"]
        assert advice is not None
        for key in _REQUIRED_KEYS:
            assert key in advice

    def test_followup_limit(self, orchestrator):
        """フォローアップは1回まで。2回目はエラーになること。"""
        session = orchestrator.create_session()
        orchestrator.extract_diary(session, "停滞している日々が続く")

        # 1回目のフォローアップ: 成功
        r1 = orchestrator.add_ideal_followup(session, "自由になりたい")
        assert "error" not in r1

        # 2回目のフォローアップ: エラー
        r2 = orchestrator.add_ideal_followup(session, "もっと成長したい")
        assert "error" in r2

    def test_extract_diary_empty_text(self, orchestrator):
        """空テキストでエラーが返ること。"""
        session = orchestrator.create_session()
        result = orchestrator.extract_diary(session, "")
        assert "error" in result

    def test_confirm_without_extract_fails(self, orchestrator):
        """extract 前に confirm を呼ぶとエラーが返ること。"""
        session = orchestrator.create_session()
        result = orchestrator.confirm_dual(session)
        assert "error" in result

    def test_advice_without_confirm_fails(self, orchestrator):
        """confirm 前に advice を呼ぶとエラーが返ること。"""
        session = orchestrator.create_session()
        orchestrator.extract_diary(session, "停滞している")
        result = orchestrator.generate_change_advice(session)
        assert "error" in result


# ============================================================
# 5. API E2E テスト（Flask テストクライアント、デモモード）
# ============================================================

class TestDiaryAPI:
    """Flask API の日記モードエンドポイントを E2E で検証する。"""

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

    def _create_diary_session(self, client):
        """日記モードのセッションを作成し session_id を返すヘルパー。"""
        resp = client.post(
            "/api/session",
            data=json.dumps({"mode": "diary"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "session_id" in data
        assert data.get("mode") == "diary"
        return data["session_id"]

    def test_diary_session_creation(self, client):
        """POST /api/session (mode=diary) で日記セッションが作成されること。"""
        resp = client.post(
            "/api/session",
            data=json.dumps({"mode": "diary"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mode"] == "diary"
        assert "session_id" in data

    def test_diary_extract(self, client):
        """POST /api/diary/extract が dual extraction を返すこと。"""
        sid = self._create_diary_session(client)
        resp = client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": sid, "text": "停滞している日々が続く"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "dual_extraction" in data
        assert "assessment" in data
        assert "summaries" in data
        assert data["phase"] == "diary-reviewing"

    def test_diary_confirm(self, client):
        """POST /api/diary/confirm が candidates と gap_analysis を返すこと。"""
        sid = self._create_diary_session(client)
        # extract first
        client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": sid, "text": "停滞している日々が続く"}),
            content_type="application/json",
        )
        # then confirm
        resp = client.post(
            "/api/diary/confirm",
            data=json.dumps({"session_id": sid, "candidate_index": 0}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_candidates" in data
        assert "goal_hexagram" in data
        assert "gap_analysis" in data
        assert data["phase"] == "diary-confirmed"

    def test_diary_advice(self, client):
        """POST /api/diary/advice が feedback と change_advice を返すこと。"""
        sid = self._create_diary_session(client)
        # extract
        client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": sid, "text": "停滞している日々が続く"}),
            content_type="application/json",
        )
        # confirm
        client.post(
            "/api/diary/confirm",
            data=json.dumps({"session_id": sid, "candidate_index": 0}),
            content_type="application/json",
        )
        # advice
        resp = client.post(
            "/api/diary/advice",
            data=json.dumps({"session_id": sid, "candidate_index": 0}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "feedback" in data
        assert "change_advice" in data
        assert data["phase"] == "diary-result"

    def test_diary_full_flow(self, client):
        """フル E2E: session -> extract -> confirm -> advice。"""
        # 1. session
        sid = self._create_diary_session(client)

        # 2. extract
        resp_extract = client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": sid, "text": "毎日が退屈で停滞している"}),
            content_type="application/json",
        )
        assert resp_extract.status_code == 200
        d_extract = resp_extract.get_json()
        assert d_extract["phase"] == "diary-reviewing"

        # 3. confirm
        resp_confirm = client.post(
            "/api/diary/confirm",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert resp_confirm.status_code == 200
        d_confirm = resp_confirm.get_json()
        assert d_confirm["phase"] == "diary-confirmed"
        assert len(d_confirm["current_candidates"]) >= 1

        # 4. advice
        resp_advice = client.post(
            "/api/diary/advice",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        assert resp_advice.status_code == 200
        d_advice = resp_advice.get_json()
        assert d_advice["phase"] == "diary-result"
        assert d_advice["feedback"] is not None
        assert d_advice["change_advice"] is not None
        for key in _REQUIRED_KEYS:
            assert key in d_advice["change_advice"]

    def test_standard_mode_unaffected(self, client):
        """標準モードのエンドポイント（POST /api/session, GET /api/health）が
        日記モード追加後も正常に動作すること（リグレッションチェック）。"""
        # health
        resp_health = client.get("/api/health")
        assert resp_health.status_code == 200
        assert resp_health.get_json()["status"] == "ok"

        # standard session (mode を指定しない)
        resp_session = client.post("/api/session")
        assert resp_session.status_code == 200
        data = resp_session.get_json()
        assert "session_id" in data
        # 標準モードには "mode" キーがない
        assert data.get("mode") is None or data.get("mode") != "diary"

    def test_diary_extract_wrong_session_mode(self, client):
        """標準モードのセッションで日記エンドポイントを呼ぶと 400 エラーになること。"""
        # standard session
        resp = client.post("/api/session")
        sid = resp.get_json()["session_id"]
        # diary extract on standard session
        resp_extract = client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": sid, "text": "テスト"}),
            content_type="application/json",
        )
        assert resp_extract.status_code == 400

    def test_diary_confirm_before_extract(self, client):
        """extract 前に confirm を呼ぶと 400 エラーになること。"""
        sid = self._create_diary_session(client)
        resp = client.post(
            "/api/diary/confirm",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        # phase が "input" なので "diary-reviewing" ではない -> 400
        assert resp.status_code == 400

    def test_diary_advice_before_confirm(self, client):
        """confirm 前に advice を呼ぶと 400 エラーになること。"""
        sid = self._create_diary_session(client)
        client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": sid, "text": "停滞している"}),
            content_type="application/json",
        )
        resp = client.post(
            "/api/diary/advice",
            data=json.dumps({"session_id": sid}),
            content_type="application/json",
        )
        # phase が "diary-reviewing" なので "diary-confirmed" ではない -> 400
        assert resp.status_code == 400

    def test_diary_invalid_session_id(self, client):
        """存在しないセッションIDで 404 エラーになること。"""
        resp = client.post(
            "/api/diary/extract",
            data=json.dumps({"session_id": "nonexistent", "text": "テスト"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
