#!/usr/bin/env python3
"""
パイロット100件のLLMアノテーション（2エージェント独立）

Annotator A: 改訂版規約に従い、内卦・外卦を選択 + reasoning
Annotator B: 同じ規約だが、事例の提示順序をシャッフルして独立性を確保
異なるシステムプロンプトで独立性を強化。
"""

import json
import random
import re
import sys
import time
from pathlib import Path

# Anthropic SDK
try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic が必要です")
    sys.exit(1)

SEED_A = 42
SEED_B = 7777
BATCH_SIZE = 10
MAX_RETRIES = 3

BASE = Path(__file__).resolve().parent.parent
PILOT_PATH = BASE / "analysis" / "gold_set" / "pilot_100.json"
OUTPUT_PATH = BASE / "analysis" / "gold_set" / "pilot_annotations.json"

VALID_TRIGRAMS = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}

# ── King Wen lookup ──
KING_WEN = {
    ("乾","乾"):1,("坤","坤"):2,("震","坎"):3,("艮","坎"):4,("乾","坎"):5,
    ("坎","乾"):6,("坎","坤"):7,("坤","坎"):8,("乾","巽"):9,("兌","乾"):10,
    ("乾","坤"):11,("坤","乾"):12,("離","乾"):13,("乾","離"):14,("艮","坤"):15,
    ("坤","震"):16,("震","兌"):17,("巽","艮"):18,("坤","兌"):19,("坤","巽"):20,
    ("震","離"):21,("離","艮"):22,("坤","艮"):23,("震","坤"):24,("震","乾"):25,
    ("乾","艮"):26,("震","艮"):27,("巽","兌"):28,("坎","坎"):29,("離","離"):30,
    ("艮","兌"):31,("巽","震"):32,("艮","乾"):33,("乾","震"):34,("坤","離"):35,
    ("離","坤"):36,("離","巽"):37,("兌","離"):38,("艮","坎"):39,("坎","震"):40,
    ("兌","艮"):41,("震","巽"):42,("乾","兌"):43,("巽","乾"):44,("坤","兌"):45,
    ("巽","坤"):46,("坎","兌"):47,("巽","坎"):48,("離","兌"):49,("巽","離"):50,
    ("震","震"):51,("艮","艮"):52,("艮","巽"):53,("兌","震"):54,("離","震"):55,
    ("艮","離"):56,("巽","巽"):57,("兌","兌"):58,("坎","巽"):59,("兌","坎"):60,
    ("兌","巽"):61,("艮","震"):62,("離","坎"):63,("坎","離"):64,
}

# ── System prompts for independence ──

SYSTEM_A = """あなたは易経アノテーション専門家Aです。以下のアノテーション規約v2.0に厳密に従ってください。

## 決定木（判定優先規則）
内卦（主体の内的状態）と外卦（外部環境）を以下の優先順で判定する:
1. 積極的拡大行動 → 乾
2. 受容・基盤形成 → 坤
3. 突発的行動開始 → 震
4. 漸進的浸透・適応 → 巽
5. 困難・リスク直面 → 坎
6. 明確なビジョン・注目 → 離
7. 停止・蓄積・内省 → 艮
8. 交流・成果享受 → 兌

## 重要ルール
- 内卦と外卦は独立に判断する（同じ卦になっても良いが、必ず別々に検討する）
- state_labelだけで機械的に決めない。story_summary全体から判断する
- action_typeは内的「状態」ではない。行動ではなく状態を見る
- 2つ以上該当する場合、テキスト中の最も顕著な特徴を優先
- テキストが不十分で判定困難な場合のみ uncertain: true

## 境界事例ルール
- 困難の中の積極性: 内卦は坎（困難が本質）、乾は例外的
- 安定 vs 停滞: 意図的維持→艮、環境に従っている→坤
- 漸進 vs 受容: 方向性明確→巽、不明確→坤
- 注目 vs 衝撃: 継続的注目→離、一時的衝撃→震
- 内省 vs 困難: 自発的縮小→艮、外部圧力→坎"""

SYSTEM_B = """あなたは易経アノテーション専門家Bです。独立した判定を行ってください。

## 判定基準
内卦（下卦）= 主体の内面状態、外卦（上卦）= 外部環境・対外的状況。

八卦の選択は以下の操作的定義に基づく:
- 乾: 強い意志・積極的拡大・リーダーシップ
- 坤: 受容・従順・基盤形成・静かな蓄積
- 震: 衝撃・突発的変化・新規行動開始
- 巽: 浸透・柔軟な適応・漸進的影響
- 坎: 困難・リスク・試練・流動的不安定
- 離: 明晰・可視化・注目・情熱
- 艮: 停止・蓄積・内省・現状維持
- 兌: 喜び・交流・開放・成果享受

## 判定手順
1. story_summaryを読み全体像を把握
2. 主体の内面状態から内卦を選択
3. 外部環境から外卦を選択（内卦とは独立に判断）
4. reasoningを記述

## 注意
- story_summary全体から判断する。state_labelに機械的にマッピングしない
- 内卦と外卦は別の観点。常に独立に検討する
- action_typeは「とった行動」であり「内的状態」ではない
- 2つ以上の卦が候補になる場合、テキストで最も記述量の多い側面を優先
- 判定困難な場合のみ uncertain: true（全体の10%以下に制限）"""


USER_TEMPLATE = """以下の事例について、Before（変化前）とAfter（変化後）それぞれの内卦（lower）と外卦（upper）を判定してください。

【事例情報】
- ID: {transition_id}
- 対象: {target_name}
- 時期: {period}
- 規模: {scale}
- 概要: {story_summary}
- 変化前の状態: {before_state}
- トリガー: {trigger_type}
- 取った行動: {action_type}
- 変化後の状態: {after_state}

以下のJSON形式のみで回答してください（説明文不要）:
```json
{{
  "before_lower": "<八卦名>",
  "before_upper": "<八卦名>",
  "before_uncertain": <true/false>,
  "before_reasoning": "<選定理由>",
  "after_lower": "<八卦名>",
  "after_upper": "<八卦名>",
  "after_uncertain": <true/false>,
  "after_reasoning": "<選定理由>"
}}
```"""


def parse_json_response(text: str) -> dict | None:
    """LLM応答からJSONを抽出"""
    # Try code block first
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    m = re.search(r'\{[^{}]*"before_lower"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Last resort: find outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


def validate_annotation(ann: dict) -> bool:
    """アノテーション結果のバリデーション"""
    required = ["before_lower", "before_upper", "after_lower", "after_upper"]
    for key in required:
        if key not in ann:
            return False
        if ann[key] not in VALID_TRIGRAMS:
            return False
    return True


def annotate_case(client, case: dict, system_prompt: str, model: str = "claude-sonnet-4-20250514") -> dict | None:
    """1事例をアノテーション"""
    user_msg = USER_TEMPLATE.format(
        transition_id=case.get("transition_id", "?"),
        target_name=case.get("target_name", "?"),
        period=case.get("period", "?"),
        scale=case.get("scale", "?"),
        story_summary=case.get("story_summary", "?"),
        before_state=case.get("before_state", "?"),
        trigger_type=case.get("trigger_type", "?"),
        action_type=case.get("action_type", "?"),
        after_state=case.get("after_state", "?"),
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0 if attempt == 0 else 0.2,
            )
            text = response.content[0].text
            parsed = parse_json_response(text)
            if parsed and validate_annotation(parsed):
                return parsed
            print(f"    [Retry {attempt+1}] Parse/validate failed for {case.get('transition_id')}")
        except Exception as e:
            print(f"    [Retry {attempt+1}] API error: {e}")
            time.sleep(2 ** attempt)

    return None


def main():
    # Load pilot data
    with open(PILOT_PATH, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    print(f"パイロットデータ: {len(pilot_data)}件")

    client = anthropic.Anthropic()

    # Prepare orderings
    order_a = list(range(len(pilot_data)))
    random.seed(SEED_A)
    random.shuffle(order_a)

    order_b = list(range(len(pilot_data)))
    random.seed(SEED_B)
    random.shuffle(order_b)

    results = {}  # transition_id -> {annotator_a, annotator_b}

    # Initialize results dict
    for case in pilot_data:
        tid = case["transition_id"]
        results[tid] = {"transition_id": tid, "annotator_a": None, "annotator_b": None}

    # Annotator A
    print("\n=== Annotator A ===")
    failed_a = 0
    for batch_start in range(0, len(order_a), BATCH_SIZE):
        batch_indices = order_a[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{(len(order_a) + BATCH_SIZE - 1) // BATCH_SIZE}")

        for idx in batch_indices:
            case = pilot_data[idx]
            tid = case["transition_id"]
            ann = annotate_case(client, case, SYSTEM_A)
            if ann:
                results[tid]["annotator_a"] = {
                    "before_lower": ann["before_lower"],
                    "before_upper": ann["before_upper"],
                    "before_uncertain": ann.get("before_uncertain", False),
                    "before_reasoning": ann.get("before_reasoning", ""),
                    "after_lower": ann["after_lower"],
                    "after_upper": ann["after_upper"],
                    "after_uncertain": ann.get("after_uncertain", False),
                    "after_reasoning": ann.get("after_reasoning", ""),
                }
            else:
                failed_a += 1
                print(f"    FAILED: {tid}")

    print(f"  Annotator A完了: {len(order_a) - failed_a}/{len(order_a)} 成功")

    # Annotator B
    print("\n=== Annotator B ===")
    failed_b = 0
    for batch_start in range(0, len(order_b), BATCH_SIZE):
        batch_indices = order_b[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{(len(order_b) + BATCH_SIZE - 1) // BATCH_SIZE}")

        for idx in batch_indices:
            case = pilot_data[idx]
            tid = case["transition_id"]
            ann = annotate_case(client, case, SYSTEM_B)
            if ann:
                results[tid]["annotator_b"] = {
                    "before_lower": ann["before_lower"],
                    "before_upper": ann["before_upper"],
                    "before_uncertain": ann.get("before_uncertain", False),
                    "before_reasoning": ann.get("before_reasoning", ""),
                    "after_lower": ann["after_lower"],
                    "after_upper": ann["after_upper"],
                    "after_uncertain": ann.get("after_uncertain", False),
                    "after_reasoning": ann.get("after_reasoning", ""),
                }
            else:
                failed_b += 1
                print(f"    FAILED: {tid}")

    print(f"  Annotator B完了: {len(order_b) - failed_b}/{len(order_b)} 成功")

    # Compile output
    output = list(results.values())
    output.sort(key=lambda x: x["transition_id"])

    # Add summary stats
    n_both = sum(1 for r in output if r["annotator_a"] and r["annotator_b"])
    n_uncertain_a = sum(1 for r in output if r["annotator_a"] and
                       (r["annotator_a"].get("before_uncertain") or r["annotator_a"].get("after_uncertain")))
    n_uncertain_b = sum(1 for r in output if r["annotator_b"] and
                       (r["annotator_b"].get("before_uncertain") or r["annotator_b"].get("after_uncertain")))

    summary = {
        "total": len(output),
        "both_annotated": n_both,
        "failed_a": failed_a,
        "failed_b": failed_b,
        "uncertain_a": n_uncertain_a,
        "uncertain_b": n_uncertain_b,
        "uncertain_rate_a": round(n_uncertain_a / max(1, len(order_a) - failed_a), 3),
        "uncertain_rate_b": round(n_uncertain_b / max(1, len(order_b) - failed_b), 3),
    }

    final = {"summary": summary, "annotations": output}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n出力: {OUTPUT_PATH}")
    print(f"サマリー: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    # Warn if uncertain rate > 10%
    if summary["uncertain_rate_a"] > 0.10:
        print(f"\nWARNING: Annotator Aのuncertain率が{summary['uncertain_rate_a']*100:.1f}% (>10%) — 規約の見直しが必要")
    if summary["uncertain_rate_b"] > 0.10:
        print(f"\nWARNING: Annotator Bのuncertain率が{summary['uncertain_rate_b']*100:.1f}% (>10%) — 規約の見直しが必要")


if __name__ == "__main__":
    main()
