#!/usr/bin/env python3
"""
Validation Dry Run — 10ペルソナを使った全APIフローの自動検証

validation/personas/V01_persona.md 〜 V10_persona.md を読み込み、
各ペルソナのナラティブテキストで以下のフローを実行する:
  1. セッション作成
  2. POST /api/extract（ナラティブテキスト送信）
  3. POST /api/confirm（危機検出でなければ）
  4. POST /api/feedback（format=5point, candidate_index=0）

結果を validation/runs/dryrun_{timestamp}.json に保存し、
サマリーテーブルを標準出力に表示する。
"""

import json
import os
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# パス設定（tests/test_web_app.py と同じパターン）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from app import app, sessions


# ---------------------------------------------------------------------------
# ペルソナ読み込み
# ---------------------------------------------------------------------------

def extract_narrative(filepath: str) -> str:
    """ペルソナMarkdownから「ナラティブテキスト」セクションのテキストを抽出する。"""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # 「## ナラティブテキスト」の後、次の「##」までを取得
    pattern = r"## ナラティブテキスト\s*\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()

    raise ValueError(f"ナラティブテキストが見つかりません: {filepath}")


def load_personas(personas_dir: str) -> list:
    """V01〜V10のペルソナファイルを読み込み、リストで返す。"""
    personas = []
    for i in range(1, 11):
        pid = f"V{i:02d}"
        filepath = os.path.join(personas_dir, f"{pid}_persona.md")
        if not os.path.exists(filepath):
            print(f"  [WARN] {filepath} が見つかりません — スキップ")
            continue
        narrative = extract_narrative(filepath)
        personas.append({
            "persona_id": pid,
            "filepath": filepath,
            "narrative": narrative,
        })
    return personas


# ---------------------------------------------------------------------------
# APIフロー実行
# ---------------------------------------------------------------------------

def run_persona_flow(client, persona: dict) -> dict:
    """1ペルソナ分のAPIフローを実行し、結果dictを返す。"""
    pid = persona["persona_id"]
    narrative = persona["narrative"]
    result = {
        "persona_id": pid,
        "narrative_length": len(narrative),
        "crisis_detected": False,
        "crisis_category": None,
        "crisis_severity": None,
        "extract_status": None,
        "confirm_status": None,
        "feedback_status": None,
        "point1_summary": None,
        "point2_summary": None,
        "point3_summary": None,
        "point4_summary": None,
        "point5_summary": None,
        "quality_warnings": [],
        "safety_flag": None,
        "evidence_label": None,
        "error": None,
        "demo_mode": None,
    }

    # セッションクリア
    sessions.clear()

    try:
        # 1. セッション作成
        resp = client.post("/api/session")
        if resp.status_code != 200:
            result["error"] = f"session creation failed: {resp.status_code}"
            return result
        session_id = resp.get_json()["session_id"]

        # 2. Extract
        resp = client.post("/api/extract", json={
            "session_id": session_id,
            "text": narrative,
        })
        result["extract_status"] = resp.status_code
        if resp.status_code != 200:
            result["error"] = f"extract failed: {resp.status_code}"
            return result

        extract_data = resp.get_json()
        result["demo_mode"] = extract_data.get("demo_mode", False)

        # 危機検出チェック
        if extract_data.get("crisis_detected"):
            result["crisis_detected"] = True
            result["crisis_category"] = extract_data.get("crisis_category")
            result["crisis_severity"] = extract_data.get("crisis_severity")
            # critical/high は遮断 — confirm/feedback をスキップ
            if extract_data.get("crisis_severity") in ("critical", "high"):
                return result
            # medium は継続

        # 3. Confirm
        resp = client.post("/api/confirm", json={"session_id": session_id})
        result["confirm_status"] = resp.status_code
        if resp.status_code != 200:
            result["error"] = f"confirm failed: {resp.status_code} - {resp.get_json()}"
            return result

        # 4. Feedback (5point format)
        resp = client.post("/api/feedback", json={
            "session_id": session_id,
            "candidate_index": 0,
            "format": "5point",
        })
        result["feedback_status"] = resp.status_code
        if resp.status_code != 200:
            result["error"] = f"feedback failed: {resp.status_code} - {resp.get_json()}"
            return result

        fb_data = resp.get_json()
        view = fb_data.get("feedback_5point", {})

        # 5point の各ポイントを要約
        p1 = view.get("point1_current_position", {})
        result["point1_summary"] = p1.get("title", "")

        p2 = view.get("point2_do_not", {})
        result["point2_summary"] = _truncate(p2.get("action", ""), 40)

        p3 = view.get("point3_do", {})
        result["point3_summary"] = _truncate(p3.get("action", ""), 40)

        p4 = view.get("point4_opposite_view", {})
        primary = p4.get("primary", {})
        result["point4_summary"] = primary.get("hexagram_name", "")

        p5 = view.get("point5_reference_cases", {})
        result["point5_summary"] = f"matched={p5.get('matched_n', '?')}/{p5.get('corpus_n', '?')}"

        result["quality_warnings"] = view.get("quality_warnings", [])
        result["evidence_label"] = p5.get("evidence_label")

        # safety_flag
        if fb_data.get("safety_flag"):
            result["safety_flag"] = fb_data["safety_flag"]

    except Exception as e:
        result["error"] = f"exception: {type(e).__name__}: {e}"

    return result


def _truncate(text: str, maxlen: int) -> str:
    """テキストを最大長に切り詰める。"""
    if not text:
        return ""
    if len(text) <= maxlen:
        return text
    return text[:maxlen - 3] + "..."


# ---------------------------------------------------------------------------
# サマリーテーブル表示
# ---------------------------------------------------------------------------

def print_summary_table(results: list):
    """結果のサマリーテーブルを標準出力に表示する。"""
    # ヘッダー
    print()
    print("=" * 120)
    print(f"{'ID':<5} {'Crisis':<10} {'Sev':<10} {'Extract':<8} {'Confirm':<8} {'FB':<8} "
          f"{'Demo':<5} {'Point1 (現在地)':<30} {'Warnings':<10}")
    print("-" * 120)

    for r in results:
        crisis = "YES" if r["crisis_detected"] else "-"
        sev = r["crisis_severity"] or "-"
        ext = str(r["extract_status"] or "-")
        conf = str(r["confirm_status"] or "-")
        fb = str(r["feedback_status"] or "-")
        demo = "Y" if r.get("demo_mode") else "N"
        p1 = _truncate(r["point1_summary"] or "", 28)
        warns = str(len(r["quality_warnings"]))

        print(f"{r['persona_id']:<5} {crisis:<10} {sev:<10} {ext:<8} {conf:<8} {fb:<8} "
              f"{demo:<5} {p1:<30} {warns:<10}")

        if r["error"]:
            print(f"      ERROR: {r['error']}")
        if r["safety_flag"]:
            sf = r["safety_flag"]
            print(f"      SAFETY: {sf.get('crisis_category', '?')} / {sf.get('crisis_severity', '?')}")
        if r["evidence_label"]:
            print(f"      EVIDENCE: {r['evidence_label']}")

    print("=" * 120)

    # サマリー統計
    total = len(results)
    crisis_count = sum(1 for r in results if r["crisis_detected"])
    success_count = sum(1 for r in results if r["feedback_status"] == 200)
    error_count = sum(1 for r in results if r["error"])
    safety_count = sum(1 for r in results if r["safety_flag"])
    demo_count = sum(1 for r in results if r.get("demo_mode"))

    print(f"\nTotal: {total} | Completed: {success_count} | Crisis blocked: {crisis_count} "
          f"| Safety flagged: {safety_count} | Errors: {error_count} | Demo mode: {demo_count}")
    print()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    personas_dir = os.path.join(PROJECT_ROOT, "validation", "personas")
    runs_dir = os.path.join(PROJECT_ROOT, "validation", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"[validation_dryrun] Loading personas from: {personas_dir}")
    personas = load_personas(personas_dir)
    print(f"[validation_dryrun] Loaded {len(personas)} personas")

    if not personas:
        print("[ERROR] No personas found. Exiting.")
        sys.exit(1)

    app.config["TESTING"] = True
    results = []

    with app.test_client() as client:
        for persona in personas:
            pid = persona["persona_id"]
            print(f"  Running {pid} ({len(persona['narrative'])} chars)...", end=" ", flush=True)
            result = run_persona_flow(client, persona)
            results.append(result)

            if result["crisis_detected"] and result["crisis_severity"] in ("critical", "high"):
                print(f"CRISIS ({result['crisis_category']})")
            elif result["error"]:
                print(f"ERROR")
            else:
                print(f"OK (fb={result['feedback_status']})")

    # サマリーテーブル表示
    print_summary_table(results)

    # JSON保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(runs_dir, f"dryrun_{timestamp}.json")
    output_data = {
        "run_timestamp": timestamp,
        "persona_count": len(personas),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[validation_dryrun] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
