#!/usr/bin/env python3
"""
易経変化ロジックDB — Flask APIサーバー

既存の3エンジン（LLMDialogueEngine, ProbabilityMapper, FeedbackEngine）を
Webブラウザから呼び出せるREST APIを提供する。

エンドポイント:
    GET  /api/health    — ヘルスチェック
    POST /api/session   — セッション作成
    POST /api/extract   — テキストから5軸抽出
    POST /api/followup  — フォローアップ回答で再抽出
    POST /api/confirm   — 抽出結果を確定し卦候補を取得
    POST /api/feedback  — 候補を選択しフィードバック生成
"""

import json
import os
import sys
import uuid

from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# ---------------------------------------------------------------------------
# エンジンインポート
# ---------------------------------------------------------------------------
from llm_dialogue import LLMDialogueEngine
from probability_tables import ProbabilityMapper
from feedback_engine import FeedbackEngine

# ---------------------------------------------------------------------------
# Flask アプリ
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.json.ensure_ascii = False

# ---------------------------------------------------------------------------
# エンジン初期化（グローバル・1回のみ）
# ---------------------------------------------------------------------------
dialogue_engine = LLMDialogueEngine()
prob_mapper = ProbabilityMapper()
feedback_engine = FeedbackEngine()

# ---------------------------------------------------------------------------
# セッション管理（インメモリ）
# ---------------------------------------------------------------------------
sessions = {}  # session_id -> dict


def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "session_id": sid,
        "accumulated_text": "",
        "current_extraction": None,
        "followup_count": 0,
        "db_labels": None,
        "candidates": None,
        "phase": "input",  # input -> reviewing -> confirmed -> result
    }
    return sessions[sid]


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _serialize_low_axes(low_axes):
    """low_axes はタプルのリスト。JSONシリアライズ用にリストのリストに変換する。"""
    return [list(item) for item in low_axes]


def _get_session_or_404(session_id):
    """セッションを取得する。見つからなければ (None, error_response) を返す。"""
    s = sessions.get(session_id)
    if s is None:
        return None, (jsonify({"error": "セッションが見つかりません"}), 404)
    return s, None


# ---------------------------------------------------------------------------
# 静的ファイル配信
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")


# ---------------------------------------------------------------------------
# API エンドポイント
# ---------------------------------------------------------------------------

# 1. GET /api/health
@app.route("/api/health")
def health():
    case_count = sum(
        1 for _ in open(
            os.path.join(PROJECT_ROOT, "data", "raw", "cases.jsonl"),
            encoding="utf-8",
        )
    )
    return jsonify({
        "status": "ok",
        "llm_available": dialogue_engine.is_available(),
        "case_count": case_count,
    })


# 2. POST /api/session
@app.route("/api/session", methods=["POST"])
def new_session():
    s = create_session()
    return jsonify({"session_id": s["session_id"]})


# 3. POST /api/extract
@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    text = data.get("text", "")

    s, err = _get_session_or_404(session_id)
    if err:
        return err

    if not dialogue_engine.is_available():
        return jsonify({"error": "LLMサービスが利用できません"}), 503

    # 抽出
    extraction = dialogue_engine.extract_axes(text)
    if extraction is None:
        return jsonify({"error": "分析に失敗しました"}), 400

    # 確信度評価
    assessment = dialogue_engine.assess_confidence(extraction)

    # low_axes をシリアライズ可能に変換
    assessment["low_axes"] = _serialize_low_axes(assessment["low_axes"])

    # フォローアップ質問
    followup_question = None
    if assessment["action"] != "proceed":
        # generate_followup_question には元のタプルリストが必要
        low_axes_tuples = [tuple(item) for item in assessment["low_axes"]]
        followup_question = dialogue_engine.generate_followup_question(
            extraction, low_axes_tuples, text
        )

    # セッション更新
    s["accumulated_text"] = text
    s["current_extraction"] = extraction
    s["phase"] = "reviewing"

    # ユーザー向け要約
    summary = dialogue_engine.summarize_for_user(extraction)

    return jsonify({
        "extraction": extraction,
        "assessment": assessment,
        "summary": summary,
        "followup_question": followup_question,
        "phase": "reviewing",
    })


# 4. POST /api/followup
@app.route("/api/followup", methods=["POST"])
def followup():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    answer = data.get("answer", "")

    s, err = _get_session_or_404(session_id)
    if err:
        return err

    if s["phase"] != "reviewing":
        return jsonify({"error": "この操作は現在のフェーズでは実行できません"}), 400

    if not dialogue_engine.is_available():
        return jsonify({"error": "LLMサービスが利用できません"}), 503

    if s["followup_count"] >= 2:
        return jsonify({"error": "フォローアップの上限に達しました"}), 400

    # テキスト連結
    s["accumulated_text"] += "\n\n追加情報:\n" + answer

    # 再抽出
    new_extraction = dialogue_engine.extract_axes(s["accumulated_text"])
    if new_extraction is None:
        return jsonify({"error": "分析に失敗しました"}), 400

    # マージ
    merged = dialogue_engine.merge_extractions(
        s["current_extraction"], new_extraction
    )

    # 確信度評価
    assessment = dialogue_engine.assess_confidence(merged)
    assessment["low_axes"] = _serialize_low_axes(assessment["low_axes"])

    # フォローアップ回数インクリメント
    s["followup_count"] += 1

    # まだフォローアップが必要か
    need_more = (
        assessment["action"] != "proceed" and s["followup_count"] < 2
    )

    followup_question = None
    if need_more:
        low_axes_tuples = [tuple(item) for item in assessment["low_axes"]]
        followup_question = dialogue_engine.generate_followup_question(
            merged, low_axes_tuples, s["accumulated_text"]
        )

    # セッション更新
    s["current_extraction"] = merged

    # ユーザー向け要約
    summary = dialogue_engine.summarize_for_user(merged)

    return jsonify({
        "extraction": merged,
        "assessment": assessment,
        "summary": summary,
        "followup_question": followup_question,
        "phase": "reviewing",
    })


# 5. POST /api/confirm
@app.route("/api/confirm", methods=["POST"])
def confirm():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")

    s, err = _get_session_or_404(session_id)
    if err:
        return err

    if s["phase"] != "reviewing":
        return jsonify({"error": "この操作は現在のフェーズでは実行できません"}), 400

    if s["current_extraction"] is None:
        return jsonify({"error": "抽出結果がありません"}), 400

    # DB用ラベルに変換
    db_labels = dialogue_engine.extraction_to_db_labels(s["current_extraction"])

    # 確率マッピングで候補取得
    result = prob_mapper.get_top_candidates(
        db_labels["db_state"],
        db_labels["db_action"],
        db_labels["db_trigger"],
        db_labels["db_phase"],
        db_labels["db_energy"],
        n=3,
    )

    # セッション更新
    s["db_labels"] = db_labels
    s["candidates"] = result
    s["phase"] = "confirmed"

    return jsonify({
        "db_labels": db_labels,
        "candidates": result["candidates"],
        "phase": "confirmed",
    })


# 6. POST /api/feedback
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    candidate_index = data.get("candidate_index", 0)

    s, err = _get_session_or_404(session_id)
    if err:
        return err

    if s["phase"] != "confirmed":
        return jsonify({"error": "この操作は現在のフェーズでは実行できません"}), 400

    if s["candidates"] is None:
        return jsonify({"error": "候補が存在しません"}), 400

    candidates_list = s["candidates"]["candidates"]
    if candidate_index < 0 or candidate_index >= len(candidates_list):
        return jsonify({"error": "候補インデックスが範囲外です"}), 400

    candidate = candidates_list[candidate_index]

    # yao がない場合はデフォルト3
    yao = candidate.get("yao", 3)

    # フィードバック生成
    fb = feedback_engine.generate(
        candidate["hexagram_number"],
        yao,
        s["db_labels"]["db_state"],
        s["db_labels"]["db_action"],
        s["db_labels"]["overall_confidence"],
    )

    # セッション更新
    s["phase"] = "result"

    return jsonify({
        "feedback": fb,
        "selected_candidate": candidate,
        "phase": "result",
    })


# ---------------------------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    case_count = sum(
        1 for _ in open(
            os.path.join(PROJECT_ROOT, "data", "raw", "cases.jsonl"),
            encoding="utf-8",
        )
    )
    print(f"事例数: {case_count}")
    print(f"LLM: {'利用可能' if dialogue_engine.is_available() else '利用不可'}")
    print(f"http://localhost:5000")
    app.run(debug=True, port=5000)
