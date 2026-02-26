#!/usr/bin/env python3
"""
Flask Web API テスト

app.py の全6エンドポイントをテストする。
- LLM呼び出し（extract_axes, generate_followup_question）はモック
- ProbabilityMapper, FeedbackEngine は実物を使用（データファイルから実計算）
"""

import json
import os
import sys
import uuid
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# パス設定（app.py と同じプロジェクトルートを使う）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from app import app, sessions, dialogue_engine

# ---------------------------------------------------------------------------
# モックデータ
# ---------------------------------------------------------------------------

MOCK_EXTRACTION = {
    "current_state": {"primary": "停滞・閉塞", "confidence": 0.85, "reasoning": "業績低迷"},
    "energy_direction": {"primary": "内向・収束", "confidence": 0.70, "reasoning": "内部に集中"},
    "intended_action": {"primary": "刷新・破壊", "confidence": 0.80, "reasoning": "変革を望む"},
    "trigger_nature": {"primary": "内発的衝動", "confidence": 0.75, "reasoning": "自発的な判断"},
    "phase_stage": {"primary": "展開中期", "confidence": 0.65, "reasoning": "進行中"},
    "domain": "企業",
}

# 全軸が高確信度のケース（proceed判定になる）
MOCK_EXTRACTION_HIGH_CONF = {
    "current_state": {"primary": "停滞・閉塞", "confidence": 0.90, "reasoning": ""},
    "energy_direction": {"primary": "内向・収束", "confidence": 0.85, "reasoning": ""},
    "intended_action": {"primary": "刷新・破壊", "confidence": 0.88, "reasoning": ""},
    "trigger_nature": {"primary": "内発的衝動", "confidence": 0.80, "reasoning": ""},
    "phase_stage": {"primary": "展開中期", "confidence": 0.75, "reasoning": ""},
    "domain": "企業",
}

# 低確信度のケース（follow_up判定になる）
MOCK_EXTRACTION_LOW_CONF = {
    "current_state": {"primary": "停滞・閉塞", "confidence": 0.85, "reasoning": ""},
    "energy_direction": {"primary": "内向・収束", "confidence": 0.40, "reasoning": "不明瞭"},
    "intended_action": {"primary": "刷新・破壊", "confidence": 0.80, "reasoning": ""},
    "trigger_nature": {"primary": "内発的衝動", "confidence": 0.50, "reasoning": "曖昧"},
    "phase_stage": {"primary": "展開中期", "confidence": 0.65, "reasoning": ""},
    "domain": "企業",
}

MOCK_FOLLOWUP = "変化のきっかけについて、もう少し詳しく教えていただけますか？"


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Flaskテストクライアントを返す。テスト前にセッションをクリアする。"""
    app.config["TESTING"] = True
    sessions.clear()
    with app.test_client() as c:
        yield c


def _create_session(client):
    """セッションを作成し、session_id を返すヘルパー。"""
    resp = client.post("/api/session")
    return resp.get_json()["session_id"]


# ===========================================================================
# TestHealth
# ===========================================================================

class TestHealth:
    """GET /api/health"""

    def test_health_returns_ok(self, client):
        """ヘルスチェックが status=ok と case_count > 0 を返すこと。"""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["case_count"] > 0
        assert "llm_available" in data


# ===========================================================================
# TestSession
# ===========================================================================

class TestSession:
    """POST /api/session"""

    def test_create_session(self, client):
        """セッション作成が成功し、session_id が返ること。"""
        resp = client.post("/api/session")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_session_id_is_uuid(self, client):
        """session_id が有効なUUID形式であること。"""
        resp = client.post("/api/session")
        data = resp.get_json()
        sid = data["session_id"]
        # UUID形式であることを確認（例外が出なければOK）
        parsed = uuid.UUID(sid)
        assert str(parsed) == sid


# ===========================================================================
# TestExtract
# ===========================================================================

class TestExtract:
    """POST /api/extract"""

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_HIGH_CONF)
    def test_extract_success(self, mock_extract, client):
        """正常な抽出が成功し、extraction, assessment, summary が返ること。"""
        sid = _create_session(client)
        resp = client.post("/api/extract", json={
            "session_id": sid,
            "text": "会社の業績が低迷していて、新しい方向に舵を切りたい。",
        })
        assert resp.status_code == 200
        data = resp.get_json()

        # 必須フィールドの存在確認
        assert "extraction" in data
        assert "assessment" in data
        assert "summary" in data
        assert "phase" in data
        assert data["phase"] == "reviewing"

        # extraction の5軸が存在すること
        ext = data["extraction"]
        for axis in ["current_state", "energy_direction", "intended_action",
                     "trigger_nature", "phase_stage"]:
            assert axis in ext

        # assessment 構造の確認
        assessment = data["assessment"]
        assert "action" in assessment
        assert "low_axes" in assessment
        assert "overall_confidence" in assessment

    @patch("app.dialogue_engine.generate_followup_question", return_value=MOCK_FOLLOWUP)
    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_LOW_CONF)
    def test_extract_with_followup(self, mock_extract, mock_followup, client):
        """低確信度の場合にfollowup_questionが返ること。"""
        sid = _create_session(client)
        resp = client.post("/api/extract", json={
            "session_id": sid,
            "text": "よくわからないが何かが起きている。",
        })
        assert resp.status_code == 200
        data = resp.get_json()

        # assessment.action が proceed でないこと
        assert data["assessment"]["action"] != "proceed"
        # フォローアップ質問が返ること
        assert data["followup_question"] is not None
        assert len(data["followup_question"]) > 0

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION)
    def test_extract_no_session(self, mock_extract, client):
        """存在しないsession_idで404が返ること。"""
        resp = client.post("/api/extract", json={
            "session_id": "nonexistent-session-id",
            "text": "テスト",
        })
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    @patch("app.dialogue_engine.extract_axes", return_value=None)
    def test_extract_llm_failure_falls_back_to_demo(self, mock_extract, client):
        """LLM抽出が失敗した場合にデモモードにフォールバックすること。"""
        sid = _create_session(client)
        resp = client.post("/api/extract", json={
            "session_id": sid,
            "text": "テスト",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["demo_mode"] is True
        assert "extraction" in data


# ===========================================================================
# TestFollowup
# ===========================================================================

class TestFollowup:
    """POST /api/followup"""

    @patch("app.dialogue_engine.generate_followup_question", return_value=MOCK_FOLLOWUP)
    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_LOW_CONF)
    @patch("app.dialogue_engine.merge_extractions")
    def test_followup_success(self, mock_merge, mock_extract, mock_followup, client):
        """フォローアップが成功し、再抽出結果が返ること。"""
        # merge_extractions は低確信度の結果をそのまま返す（簡易モック）
        mock_merge.return_value = MOCK_EXTRACTION_LOW_CONF

        sid = _create_session(client)

        # まず extract を実行して reviewing フェーズにする
        client.post("/api/extract", json={
            "session_id": sid,
            "text": "よくわからないが何かが起きている。",
        })

        # フォローアップ
        resp = client.post("/api/followup", json={
            "session_id": sid,
            "answer": "外部からの圧力でした。",
        })
        assert resp.status_code == 200
        data = resp.get_json()

        assert "extraction" in data
        assert "assessment" in data
        assert "summary" in data
        assert data["phase"] == "reviewing"

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_LOW_CONF)
    @patch("app.dialogue_engine.generate_followup_question", return_value=MOCK_FOLLOWUP)
    @patch("app.dialogue_engine.merge_extractions")
    def test_followup_max_reached(self, mock_merge, mock_followup, mock_extract, client):
        """followup_count >= 2 でフォローアップの上限エラーが返ること。"""
        mock_merge.return_value = MOCK_EXTRACTION_LOW_CONF

        sid = _create_session(client)

        # extract → reviewing フェーズ
        client.post("/api/extract", json={
            "session_id": sid,
            "text": "よくわからない。",
        })

        # followup 1回目
        client.post("/api/followup", json={
            "session_id": sid,
            "answer": "追加情報1",
        })

        # followup 2回目
        client.post("/api/followup", json={
            "session_id": sid,
            "answer": "追加情報2",
        })

        # 3回目 → 上限エラー
        resp = client.post("/api/followup", json={
            "session_id": sid,
            "answer": "追加情報3",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "上限" in data["error"]

    def test_followup_wrong_phase(self, client):
        """reviewingフェーズでない場合に400が返ること。"""
        sid = _create_session(client)
        # input フェーズのまま followup を呼ぶ
        resp = client.post("/api/followup", json={
            "session_id": sid,
            "answer": "テスト",
        })
        assert resp.status_code == 400


# ===========================================================================
# TestConfirm
# ===========================================================================

class TestConfirm:
    """POST /api/confirm"""

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_HIGH_CONF)
    def test_confirm_success(self, mock_extract, client):
        """確定→候補返却が成功すること。candidates配列の構造を検証。"""
        sid = _create_session(client)

        # extract → reviewing
        client.post("/api/extract", json={
            "session_id": sid,
            "text": "会社の業績が低迷していて、変革したい。",
        })

        # confirm
        resp = client.post("/api/confirm", json={"session_id": sid})
        assert resp.status_code == 200
        data = resp.get_json()

        assert "candidates" in data
        assert "db_labels" in data
        assert data["phase"] == "confirmed"

        # candidates の構造検証
        candidates = data["candidates"]
        assert len(candidates) > 0
        for c in candidates:
            assert "hexagram_number" in c
            assert "hexagram_name" in c
            assert "probability" in c
            assert "rank" in c
            assert "lower_trigram" in c
            assert "upper_trigram" in c
            # UX改善: 説明データの存在検証
            assert "meaning" in c, "候補にmeaning（意味）が含まれていること"
            assert "situation" in c, "候補にsituation（状況説明）が含まれていること"
            assert "keywords" in c, "候補にkeywords（キーワード）が含まれていること"
            assert isinstance(c["keywords"], list), "keywordsがリストであること"
            assert "archetype" in c, "候補にarchetype（元型）が含まれていること"
            assert "modern_interpretation" in c, "候補にmodern_interpretation（現代的解釈）が含まれていること"

        # db_labels の構造検証
        db = data["db_labels"]
        assert "db_state" in db
        assert "db_action" in db
        assert "db_trigger" in db
        assert "db_phase" in db
        assert "overall_confidence" in db

    def test_confirm_no_extraction(self, client):
        """extraction未実行で確定しようとすると400が返ること。"""
        sid = _create_session(client)
        # input フェーズのまま confirm
        resp = client.post("/api/confirm", json={"session_id": sid})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


# ===========================================================================
# TestFeedback
# ===========================================================================

class TestFeedback:
    """POST /api/feedback"""

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_HIGH_CONF)
    def test_feedback_success(self, mock_extract, client):
        """フィードバック生成が成功し、5レイヤー構造が返ること。"""
        sid = _create_session(client)

        # extract → reviewing
        client.post("/api/extract", json={
            "session_id": sid,
            "text": "会社の業績が低迷していて、変革したい。",
        })

        # confirm → confirmed
        client.post("/api/confirm", json={"session_id": sid})

        # feedback
        resp = client.post("/api/feedback", json={
            "session_id": sid,
            "candidate_index": 0,
        })
        assert resp.status_code == 200
        data = resp.get_json()

        assert "feedback" in data
        assert "selected_candidate" in data
        assert data["phase"] == "result"

        # 5レイヤー構造の存在確認
        fb = data["feedback"]
        assert "layer1_current" in fb
        assert "layer2_direction" in fb
        assert "layer3_hidden" in fb
        assert "layer4_reference" in fb
        assert "layer5_question" in fb

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_HIGH_CONF)
    def test_feedback_invalid_index(self, mock_extract, client):
        """不正なcandidate_indexで400が返ること。"""
        sid = _create_session(client)

        # extract → confirm
        client.post("/api/extract", json={
            "session_id": sid,
            "text": "テスト",
        })
        client.post("/api/confirm", json={"session_id": sid})

        # 存在しないインデックスを指定
        resp = client.post("/api/feedback", json={
            "session_id": sid,
            "candidate_index": 99,
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_HIGH_CONF)
    def test_feedback_negative_index(self, mock_extract, client):
        """負のcandidate_indexで400が返ること。"""
        sid = _create_session(client)

        client.post("/api/extract", json={
            "session_id": sid,
            "text": "テスト",
        })
        client.post("/api/confirm", json={"session_id": sid})

        resp = client.post("/api/feedback", json={
            "session_id": sid,
            "candidate_index": -1,
        })
        assert resp.status_code == 400

    def test_feedback_wrong_phase(self, client):
        """confirmedフェーズでない場合に400が返ること。"""
        sid = _create_session(client)
        resp = client.post("/api/feedback", json={
            "session_id": sid,
            "candidate_index": 0,
        })
        assert resp.status_code == 400


# ===========================================================================
# TestFullFlow
# ===========================================================================

class TestFullFlow:
    """セッション作成→抽出→確定→フィードバックの一連フロー"""

    @patch("app.dialogue_engine.extract_axes", return_value=MOCK_EXTRACTION_HIGH_CONF)
    def test_full_flow_without_followup(self, mock_extract, client):
        """全ステップが高確信度でproceedし、フィードバックまで到達すること。"""
        # 1. セッション作成
        resp = client.post("/api/session")
        assert resp.status_code == 200
        sid = resp.get_json()["session_id"]

        # 2. 抽出
        resp = client.post("/api/extract", json={
            "session_id": sid,
            "text": "会社の業績が低迷していて、大きな変革を起こしたい。内部からの衝動で決断した。",
        })
        assert resp.status_code == 200
        extract_data = resp.get_json()
        assert extract_data["assessment"]["action"] == "proceed"
        assert extract_data["followup_question"] is None

        # 3. 確定
        resp = client.post("/api/confirm", json={"session_id": sid})
        assert resp.status_code == 200
        confirm_data = resp.get_json()
        assert len(confirm_data["candidates"]) > 0

        # 候補の卦番号が1〜64の範囲であること
        for c in confirm_data["candidates"]:
            assert 1 <= c["hexagram_number"] <= 64

        # 4. フィードバック
        resp = client.post("/api/feedback", json={
            "session_id": sid,
            "candidate_index": 0,
        })
        assert resp.status_code == 200
        fb_data = resp.get_json()

        # 5レイヤー全てが存在すること
        fb = fb_data["feedback"]
        assert fb["layer1_current"]["hexagram"]["id"] >= 1
        assert fb["layer2_direction"]["resulting_hexagram"]["id"] >= 1
        assert fb["layer3_hidden"]["nuclear"]["hexagram_id"] >= 1
        assert len(fb["layer4_reference"]["similar_cases"]) >= 0
        assert len(fb["layer5_question"]["question"]) > 0

        # フェーズが result に遷移していること
        assert fb_data["phase"] == "result"

    @patch("app.dialogue_engine.generate_followup_question", return_value=MOCK_FOLLOWUP)
    @patch("app.dialogue_engine.extract_axes")
    @patch("app.dialogue_engine.merge_extractions")
    def test_full_flow_with_followup(self, mock_merge, mock_extract, mock_followup, client):
        """フォローアップを1回挟んで、フィードバックまで到達すること。"""
        # extract_axes: 1回目は低確信度、2回目も低確信度
        mock_extract.return_value = MOCK_EXTRACTION_LOW_CONF
        # merge 後は高確信度に改善
        mock_merge.return_value = MOCK_EXTRACTION_HIGH_CONF

        # 1. セッション作成
        sid = _create_session(client)

        # 2. 抽出（低確信度 → フォローアップ必要）
        resp = client.post("/api/extract", json={
            "session_id": sid,
            "text": "何かが変わりつつある。",
        })
        assert resp.status_code == 200
        assert resp.get_json()["assessment"]["action"] != "proceed"

        # 3. フォローアップ（マージ後は高確信度）
        resp = client.post("/api/followup", json={
            "session_id": sid,
            "answer": "外部からの圧力で会社の構造を変えることにした。",
        })
        assert resp.status_code == 200

        # 4. 確定
        resp = client.post("/api/confirm", json={"session_id": sid})
        assert resp.status_code == 200
        confirm_data = resp.get_json()
        assert len(confirm_data["candidates"]) > 0

        # 5. フィードバック
        resp = client.post("/api/feedback", json={
            "session_id": sid,
            "candidate_index": 0,
        })
        assert resp.status_code == 200
        assert resp.get_json()["phase"] == "result"
