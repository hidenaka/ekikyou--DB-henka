#!/usr/bin/env python3
"""
check_report_quality.py - Report quality gate checker

Checks generated reports against Gate 1-3 automatically and
produces a human review checklist for Gate 4-5.

Usage:
    python3 scripts/check_report_quality.py --file report.md --plan-type full
    cat report.md | python3 scripts/check_report_quality.py --plan-type light
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Gate 1: Forbidden word / pattern definitions
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS = [
    # --- CRITICAL: Assertive predictions ---
    {
        "category": "断定的予測",
        "severity": "CRITICAL",
        "pattern": r".になります",
        "label": "〜になります",
        "needs_polite_check": True,
    },
    {
        "category": "断定的予測",
        "severity": "CRITICAL",
        "pattern": r"でしょう",
        "label": "〜でしょう",
    },
    {
        "category": "断定的予測",
        "severity": "CRITICAL",
        "pattern": r"に違いない",
        "label": "〜に違いない",
    },
    {
        "category": "断定的予測",
        "severity": "CRITICAL",
        "pattern": r"するはずです",
        "label": "〜するはずです",
    },
    # --- CRITICAL: Probability expressions ---
    {
        "category": "確率表現",
        "severity": "CRITICAL",
        "pattern": r"\d+\s*%",
        "label": "〇%",
    },
    {
        "category": "確率表現",
        "severity": "CRITICAL",
        "pattern": r"確率",
        "label": "確率",
    },
    {
        "category": "確率表現",
        "severity": "CRITICAL",
        "pattern": r"可能性が高い",
        "label": "可能性が高い",
    },
    {
        "category": "確率表現",
        "severity": "CRITICAL",
        "pattern": r"可能性が低い",
        "label": "可能性が低い",
    },
    {
        "category": "確率表現",
        "severity": "CRITICAL",
        "pattern": r"的中率",
        "label": "的中率",
    },
    {
        "category": "確率表現",
        "severity": "CRITICAL",
        "pattern": r"成功率",
        "label": "成功率",
    },
    # --- CRITICAL: I Ching terminology ---
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"卦",
        "label": "卦",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"爻",
        "label": "爻",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"八卦",
        "label": "八卦",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])乾(?![燥杯麺])",
        "label": "乾",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"坤",
        "label": "坤",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])震(?![災度源])",
        "label": "震",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"巽",
        "label": "巽",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"坎",
        "label": "坎",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])離(?![れ陸婚職脱岸散])",
        "label": "離",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"艮",
        "label": "艮",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])兌(?![換])",
        "label": "兌",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])陰(?![性謀湿口影鬱])",
        "label": "陰",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])陽(?![性気光炎])",
        "label": "陽",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"易経",
        "label": "易経",
    },
    {
        "category": "易経用語",
        "severity": "CRITICAL",
        "pattern": r"変化の書",
        "label": "変化の書",
    },
    # --- CRITICAL: Fortune-telling terminology ---
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"運勢",
        "label": "運勢",
    },
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"運命",
        "label": "運命",
    },
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"宿命",
        "label": "宿命",
    },
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"相性",
        "label": "相性",
    },
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"開運",
        "label": "開運",
    },
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"(?<![a-zA-Z])厄(?![介])",
        "label": "厄",
    },
    {
        "category": "占い用語",
        "severity": "CRITICAL",
        "pattern": r"吉凶",
        "label": "吉凶",
    },
    # --- HIGH: Recommendation expressions ---
    {
        "category": "推奨表現",
        "severity": "HIGH",
        "pattern": r"おすすめ",
        "label": "おすすめ",
    },
    {
        "category": "推奨表現",
        "severity": "HIGH",
        "pattern": r"推奨",
        "label": "推奨",
    },
    {
        "category": "推奨表現",
        "severity": "HIGH",
        "pattern": r"最適",
        "label": "最適",
    },
    {
        "category": "推奨表現",
        "severity": "HIGH",
        "pattern": r"ベスト",
        "label": "ベスト",
    },
    {
        "category": "推奨表現",
        "severity": "HIGH",
        "pattern": r"正解",
        "label": "正解",
    },
    {
        "category": "推奨表現",
        "severity": "HIGH",
        "pattern": r"間違い",
        "label": "間違い",
    },
    # --- MEDIUM: Obligation expressions ---
    {
        "category": "義務表現",
        "severity": "MEDIUM",
        "pattern": r"すべき",
        "label": "〜すべき",
    },
    {
        "category": "義務表現",
        "severity": "MEDIUM",
        "pattern": r"しなければならない",
        "label": "〜しなければならない",
    },
    {
        "category": "義務表現",
        "severity": "MEDIUM",
        "pattern": r"する必要がある",
        "label": "〜する必要がある",
    },
    # --- MEDIUM: Type-casting ---
    {
        "category": "タイプ分け",
        "severity": "MEDIUM",
        "pattern": r"[\u3041-\u3093\u30A1-\u30F6\u30FC\w]+タイプ",
        "label": "〇〇タイプ",
    },
    {
        "category": "タイプ分け",
        "severity": "MEDIUM",
        "pattern": r"[\u3041-\u3093\u30A1-\u30F6\u30FC\w]+型",
        "label": "〇〇型",
    },
    {
        "category": "タイプ分け",
        "severity": "MEDIUM",
        "pattern": r"あなたは[\u3041-\u3093\u30A1-\u30F6\u30FC\w]+な人",
        "label": "あなたは〇〇な人",
    },
    # --- HIGH: Technical terms (internal field names) ---
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"current_state",
        "label": "current_state",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"energy_direction",
        "label": "energy_direction",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"intended_action",
        "label": "intended_action",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"trigger_nature",
        "label": "trigger_nature",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"phase_stage",
        "label": "phase_stage",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"(?<!\w)constraints(?!\w)",
        "label": "constraints",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"risk_tolerance",
        "label": "risk_tolerance",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"choice_set",
        "label": "choice_set",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"(?<!\w)confidence(?!\w)",
        "label": "confidence",
    },
    {
        "category": "技術用語",
        "severity": "HIGH",
        "pattern": r"(?<!\w)extraction(?!\w)",
        "label": "extraction",
    },
]


# ---------------------------------------------------------------------------
# Polite-form exclusion for "になります"
# ---------------------------------------------------------------------------

_POLITE_NARIMASU_PREFIXES = [
    r"以下のように",
    r"次のように",
    r"下記のように",
    r"ご案内",
    r"ご説明",
    r"ご確認",
    r"ご参考",
    r"お知らせ",
    r"表のように",
    r"こちら",
    r"構成",
    r"内容",
    r"結果",
]

_POLITE_PATTERN = re.compile(
    r"(?:" + "|".join(_POLITE_NARIMASU_PREFIXES) + r").{0,10}になります"
)


def _is_polite_narimasu(line: str, match_start: int) -> bool:
    """Return True if 'になります' in this context is polite, not predictive.

    The match_start points to the start of the '.になります' match (i.e. the
    character before 'に').  We locate 'になります' within that match and then
    look backwards up to 30 characters for a polite prefix.
    """
    # Find the position of 'になります' inside the match
    ni_pos = line.find("になります", match_start)
    if ni_pos < 0:
        return False
    window_start = max(0, ni_pos - 30)
    window = line[window_start : ni_pos + len("になります")]
    return bool(_POLITE_PATTERN.search(window))


# ---------------------------------------------------------------------------
# Gate 1 implementation
# ---------------------------------------------------------------------------


def check_gate_1(lines: list) -> dict:
    """Run forbidden-word checks. Returns gate_1 result dict."""
    violations = []

    for line_no, line in enumerate(lines, start=1):
        for rule in FORBIDDEN_PATTERNS:
            pattern = rule["pattern"]
            for m in re.finditer(pattern, line):
                # Special handling for "になります" polite exclusion
                if rule.get("needs_polite_check"):
                    if _is_polite_narimasu(line, m.start()):
                        continue

                # Build context snippet (up to 60 chars around match)
                start = max(0, m.start() - 20)
                end = min(len(line), m.end() + 20)
                context = (
                    ("..." if start > 0 else "")
                    + line[start:end]
                    + ("..." if end < len(line) else "")
                )

                violations.append(
                    {
                        "category": rule["category"],
                        "severity": rule["severity"],
                        "word": rule["label"],
                        "line": line_no,
                        "context": context,
                    }
                )

    has_critical = any(v["severity"] == "CRITICAL" for v in violations)
    return {
        "status": "FAIL" if has_critical else "PASS",
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Gate 2 implementation
# ---------------------------------------------------------------------------

FULL_SECTIONS = [
    {
        "id": 1,
        "name": "現在地の構造化",
        "patterns": [r"#+\s*(?:セクション\s*1|1\.\s|1[:：]\s*|現在地)"],
    },
    {
        "id": 2,
        "name": "このまま進んだ場合のシナリオ",
        "patterns": [r"#+\s*(?:セクション\s*2|2\.\s|2[:：]\s*|このまま|現行ルート|シナリオ)"],
    },
    {
        "id": 3,
        "name": "別ルート",
        "patterns": [r"#+\s*(?:セクション\s*3|3\.\s|3[:：]\s*|別ルート|代替|オプション|選択肢)"],
    },
    {
        "id": 4,
        "name": "各ルートの比較表",
        "patterns": [r"#+\s*(?:セクション\s*4|4\.\s|4[:：]\s*|比較|ルート.*比較)"],
    },
    {
        "id": 5,
        "name": "今日の1アクション",
        "patterns": [r"#+\s*(?:セクション\s*5|5\.\s|5[:：]\s*|今日.*アクション|1アクション|ファーストステップ|最初の一歩)"],
    },
    {
        "id": 6,
        "name": "30日/90日実験プラン",
        "patterns": [r"#+\s*(?:セクション\s*6|6\.\s|6[:：]\s*|30日|90日|実験プラン)"],
    },
]

LIGHT_SECTIONS = [
    FULL_SECTIONS[0],  # Section 1
    FULL_SECTIONS[3],  # Section 4
    FULL_SECTIONS[4],  # Section 5
]


def _detect_section(lines: list, section_def: dict) -> bool:
    """Return True if the section is found in the text."""
    for line in lines:
        for pat in section_def["patterns"]:
            if re.search(pat, line):
                return True
    return False


def _has_markdown_table(text: str) -> bool:
    """Check if text contains a markdown table (|---|)."""
    return bool(re.search(r"\|[\s\-]+\|", text))


def _has_evidence_labels(text: str) -> bool:
    """Check if evidence labels are present."""
    has_observed = "観測済み" in text
    has_hypothesis = "仮説" in text
    return has_observed or has_hypothesis


def _has_action_with_duration(text: str) -> bool:
    """Check if an action item with time/duration indication exists."""
    action_section = False
    for line in text.split("\n"):
        if re.search(
            r"#+\s*(?:セクション\s*5|5\.\s|5[:：]|今日.*アクション|1アクション|ファーストステップ|最初の一歩)",
            line,
        ):
            action_section = True
        if action_section:
            if re.search(r"\d+\s*分|\d+\s*時間|\d+\s*秒|今日|今すぐ|まず", line):
                return True
    # Fallback: check globally for action + time
    return bool(re.search(r"今日.*(?:分|時間|秒|やる|する|試す|始める)", text))


def _has_experiment_plan(text: str) -> bool:
    """Check if 30-day/90-day experiment plan exists."""
    return bool(re.search(r"(?:30\s*日|90\s*日|1\s*ヶ月|3\s*ヶ月)", text))


def check_gate_2(lines: list, text: str, plan_type: str) -> dict:
    """Run structural checks. Returns gate_2 result dict."""
    sections = FULL_SECTIONS if plan_type == "full" else LIGHT_SECTIONS
    char_count = len(text)

    if plan_type == "full":
        char_min, char_max = 3000, 5000
    else:
        char_min, char_max = 1000, 2000

    # Section presence
    missing = []
    for sec in sections:
        present = _detect_section(lines, sec)
        if not present:
            missing.append(f"セクション{sec['id']}: {sec['name']}")

    sections_all_present = len(missing) == 0
    char_in_range = char_min <= char_count <= char_max
    evidence_labels = _has_evidence_labels(text)
    comparison_table = _has_markdown_table(text)
    action_duration = _has_action_with_duration(text)

    checks = {
        "sections_present": sections_all_present,
        "char_count": char_count,
        "char_count_in_range": char_in_range,
        "evidence_labels_present": evidence_labels,
        "comparison_table_present": comparison_table,
        "action_with_duration": action_duration,
    }

    if plan_type == "full":
        experiment_plan = _has_experiment_plan(text)
        checks["experiment_plan_present"] = experiment_plan

    # MUST failures
    must_failures = list(missing)  # copy
    if not evidence_labels:
        must_failures.append("根拠ラベル（観測済み/仮説）が見つかりません")
    if not comparison_table:
        must_failures.append("比較表（マークダウンテーブル）が見つかりません")

    status = "FAIL" if must_failures else "PASS"

    return {
        "status": status,
        "checks": checks,
        "missing": must_failures,
    }


# ---------------------------------------------------------------------------
# Gate 3 implementation
# ---------------------------------------------------------------------------


def check_gate_3(text: str) -> dict:
    """Run evidence consistency checks. Returns gate_3 result dict."""
    lines = text.split("\n")

    observed_ok = True
    hypothesis_ok = True

    for i, line in enumerate(lines):
        if "観測済み" in line:
            # Check surrounding lines (within 5 lines) for outcome-like content
            window = "\n".join(lines[max(0, i - 2) : i + 6])
            if not re.search(r"(結果|アウトカム|実際|成果|経過|推移|その後|達成|失敗|成功)", window):
                observed_ok = False
        if "仮説" in line:
            # Check surrounding lines for reasoning
            window = "\n".join(lines[max(0, i - 2) : i + 6])
            if not re.search(r"(根拠|理由|推論|なぜなら|ため|背景|因|基づ|類似|傾向|パターン)", window):
                hypothesis_ok = False

    # Check for "事例数不足" label
    insufficient_data_label = "事例数不足" in text

    auto_checks = {
        "observed_has_outcome": observed_ok,
        "hypothesis_has_reasoning": hypothesis_ok,
        "insufficient_data_label_present": insufficient_data_label,
    }

    all_auto_pass = observed_ok and hypothesis_ok

    human_review_needed = [
        "匿名化の確認",
        "事例の妥当性チェック",
        "アウトカム正確性の確認",
    ]

    if all_auto_pass:
        status = "NEEDS_HUMAN_REVIEW"
    else:
        status = "FAIL"

    return {
        "status": status,
        "auto_checks": auto_checks,
        "human_review_needed": human_review_needed,
    }


# ---------------------------------------------------------------------------
# Gate 4 & 5: Human checklist
# ---------------------------------------------------------------------------


def build_human_checklist() -> dict:
    """Build checklists for human reviewers (Gate 4: Tone, Gate 5: Safety)."""
    return {
        "gate_4_tone": [
            "レポート全体のトーンが中立的・探索的であること（断定的でないこと）",
            "ユーザーの自律性を尊重する表現になっていること",
            "過度に楽観的/悲観的な表現がないこと",
            "コーチング的な語り口ではなく、情報提供の語り口であること",
            "各シナリオが公平に記述されていること（特定ルートへの誘導がないこと）",
            "読者が自分で判断できる材料が提供されていること",
        ],
        "gate_5_safety": [
            "個人を特定できる情報が含まれていないこと",
            "医療・法律・金融の専門的アドバイスに該当する記述がないこと",
            "自傷・他害を助長する可能性のある記述がないこと",
            "差別的・偏見的な表現がないこと",
            "事例に登場する人物・企業が適切に匿名化されていること",
            "免責事項・注意書きが適切に記載されていること",
        ],
    }


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------


def compute_overall(gate_1: dict, gate_2: dict, gate_3: dict) -> str:
    """Compute overall status from gate results."""
    if (
        gate_1["status"] == "FAIL"
        or gate_2["status"] == "FAIL"
        or gate_3["status"] == "FAIL"
    ):
        return "FAIL"
    if gate_3["status"] == "NEEDS_HUMAN_REVIEW":
        return "NEEDS_HUMAN_REVIEW"
    return "PASS"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Report quality gate checker for life-direction reports"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the report markdown file. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--plan-type",
        type=str,
        choices=["full", "light"],
        default="full",
        help="Report plan type: 'full' (6 sections) or 'light' (3 sections). Default: full",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON result. If omitted, prints to stdout.",
    )

    args = parser.parse_args()

    # Read input
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
        report_file = args.file
    else:
        if sys.stdin.isatty():
            print(
                "Error: No input provided. Use --file or pipe input via stdin.",
                file=sys.stderr,
            )
            sys.exit(1)
        text = sys.stdin.read()
        report_file = "<stdin>"

    # Validate non-empty
    if not text.strip():
        print("Error: Input is empty.", file=sys.stderr)
        sys.exit(1)

    lines = text.split("\n")

    # Run gates
    gate_1 = check_gate_1(lines)
    gate_2 = check_gate_2(lines, text, args.plan_type)
    gate_3 = check_gate_3(text)
    human_checklist = build_human_checklist()
    overall = compute_overall(gate_1, gate_2, gate_3)

    # Timestamp in JST
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(jst).isoformat()

    result = {
        "report_file": report_file,
        "plan_type": args.plan_type,
        "timestamp": timestamp,
        "gates": {
            "gate_1": gate_1,
            "gate_2": gate_2,
            "gate_3": gate_3,
        },
        "overall_status": overall,
        "human_checklist": human_checklist,
    }

    output_json = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Exit code: 2 for FAIL, 0 otherwise
    if overall == "FAIL":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
