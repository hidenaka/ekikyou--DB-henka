#!/usr/bin/env python3
"""
LLM-based dual annotation of gold set cases for trigram assignment.

Uses Claude API (anthropic SDK) to annotate each case with:
  - before_lower_trigram (internal driver of before-state)
  - before_upper_trigram (external manifestation of before-state)
  - after_lower_trigram (internal driver of after-state)
  - after_upper_trigram (external manifestation of after-state)

Two independent passes with different system prompts and temperatures
to enable inter-rater agreement analysis.

Usage:
  python3 scripts/annotate_gold_set.py                    # full run
  python3 scripts/annotate_gold_set.py --dry-run           # 5 cases only
  python3 scripts/annotate_gold_set.py --resume             # skip already-done
  python3 scripts/annotate_gold_set.py --batch-size 100     # process in batches of 100
  python3 scripts/annotate_gold_set.py --pass-number 1      # run only pass 1
  python3 scripts/annotate_gold_set.py --pass-number 2      # run only pass 2
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE / "analysis" / "gold_set" / "selected_800_cases.json"
OUTPUT_PASS1 = BASE / "analysis" / "gold_set" / "annotations_pass1.json"
OUTPUT_PASS2 = BASE / "analysis" / "gold_set" / "annotations_pass2.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_TRIGRAMS = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RATE_LIMIT_DELAY = 1.2  # seconds between API calls

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE / "analysis" / "gold_set" / "annotation_log.txt", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PASS1 = """You are an expert annotator for the I Ching (易経) trigram classification system.
Your task is to read case descriptions of organizational/personal/national transitions
and assign trigrams based SOLELY on the text content.

CRITICAL RULES:
1. Read the story text carefully. Do NOT simply map state labels to trigrams.
2. Each trigram is equally likely a priori. Consider all 8 options for every position.
3. Inner (lower) and outer (upper) trigrams are INDEPENDENT assessments.
4. Base your judgment on textual evidence, not on label-to-trigram shortcuts.

You are precise, analytical, and evidence-based. Cite specific phrases from the text."""

SYSTEM_PROMPT_PASS2 = """You are an expert annotator for the I Ching (易経) trigram classification system.
Your task is to read case descriptions of organizational/personal/national transitions
and assign trigrams based on the overall narrative arc and emotional tenor.

CRITICAL RULES:
1. Read the story text carefully. Do NOT simply map state labels to trigrams.
2. Each trigram is equally likely a priori. Consider all 8 options for every position.
3. Inner (lower) and outer (upper) trigrams are INDEPENDENT assessments.
4. Focus on the underlying dynamics and emotional quality of the transition.

You read narratives holistically, sensing the deeper currents beneath surface descriptions."""

# ---------------------------------------------------------------------------
# User Prompt Template
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """Below is a case describing a transition (change event). Read ALL fields carefully, then assign
trigrams for the before-state and after-state.

=== CASE DATA ===
Story Summary: {story_summary}
Before State Label: {before_state}
After State Label: {after_state}
Trigger Type: {trigger_type}
Action Type: {action_type}
Scale: {scale}
Target: {target_name}
Period: {period}
=================

## YOUR TASK

For each of the 4 positions below, select ONE trigram from the 8 options.

### The 8 Trigrams and Their Meanings

IMPORTANT: Each trigram has DIFFERENT meanings depending on whether it is assigned as
an inner (lower) trigram or an outer (upper) trigram. Read both columns carefully.

| Trigram | As INNER (Lower) Trigram — Subject's Internal State | As OUTER (Upper) Trigram — External Environment |
|---------|------------------------------------------------------|------------------------------------------------|
| 乾 (Qian) | The subject has strong will, confidence, and actively drives expansion/leadership. Energy is abundant and pushing forward. | The environment supports growth/expansion. Markets or society are in an upward trend. Strong external authority is at work. |
| 坤 (Kun) | The subject is receptive, passive, building foundations quietly. Following rather than leading. Steady accumulation without assertion. | The environment is stable with no major upheaval. Mature market. Calm, supportive conditions. |
| 震 (Zhen) | The subject made a sudden decision or action. Internal shock or disruption occurred. A new initiative was launched abruptly. | External shock hit (disaster, regulation change, market crash). Sudden environmental disruption. Disruptive technology appeared. |
| 巽 (Xun) | The subject adapts flexibly, changing gradually. Incremental strategy, penetrating like wind. DX, optimization, step-by-step reform. | Environmental change is slow but steady. Social trends gradually penetrating. Regulations evolving incrementally. |
| 坎 (Kan) | The subject faces serious difficulty, structural risk, or crisis. Financial distress, organizational collapse, loss of trust. Trapped in hardship. | The environment is harsh and dangerous. Recession, pandemic, intense competition threatening survival. Regulatory pressure. |
| 離 (Li) | The subject's core driver is clarity, vision, discernment, or passionate illumination. Ideas/technology shine brightly. Analytical insight or creative passion drives action. Transparency is central. | Media attention is focused on the subject. Information is being revealed/publicized. Society's gaze is concentrated. Under evaluation or scrutiny. |
| 艮 (Gen) | The subject intentionally stops, accumulates, or reflects. A deliberate pause. Restructuring preparation. "We will not proceed further" as a strategic choice. | The environment is stuck/stagnant. Market has plateaued. Barriers block progress. Society is in a "wait" mode. |
| 兌 (Dui) | The subject's core driver is joy, openness, communication, or collaborative exchange. Satisfaction with results. Dialogue and connection are central values. Open organizational culture. | Market/society is welcoming and favorable. High praise from customers/stakeholders. Partnership opportunities abound. Celebratory/optimistic mood. |

### Position Definitions

- **Inner (Lower) Trigram**: The subject's INTERNAL state — motivation, psychology, organizational health, foundational driver. What is happening INSIDE, often invisible from outside.
- **Outer (Upper) Trigram**: The EXTERNAL environment — market conditions, social context, visible situation, how things appear from outside.

### Critical Instructions

1. **Do NOT mechanically map labels to trigrams.** "どん底・危機" does NOT automatically mean 坎.
   A company in crisis might have a leader with burning vision (離 as inner) or might be
   deliberately stopping to restructure (艮 as inner). READ THE STORY.

2. **Inner and outer are INDEPENDENT.** Assess them separately. They CAN be the same trigram
   but usually differ. Never default to making them the same.

3. **Consider 離 and 兌 as inner trigrams equally.** They are just as valid as any other:
   - 離 as inner: When the subject's core driver is clarity, vision, discernment, or passionate
     illumination. Example: a company driven by R&D brilliance, a leader whose analytical
     clarity guides decisions, a startup whose core is innovative vision.
   - 兌 as inner: When the subject's core driver is joy, openness, communication, or
     collaborative exchange. Example: a company whose culture centers on dialogue and
     employee satisfaction, a leader driven by building connections, an organization that
     thrives on stakeholder relationships.

4. **State labels are hints, not answers.** They provide context but the story_summary
   is the primary evidence source.

5. **For each position, also provide:**
   - Your confidence level (high/medium/low)
   - One alternative trigram you seriously considered
   - Brief reasoning citing specific text evidence

### Output Format

Return ONLY valid JSON in this exact structure (no markdown fences, no extra text):

{{
  "before_lower": {{
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  }},
  "before_upper": {{
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  }},
  "after_lower": {{
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  }},
  "after_upper": {{
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  }}
}}

Trigram values must be one of: 乾, 坤, 震, 巽, 坎, 離, 艮, 兌"""


def build_user_prompt(case: dict) -> str:
    """Build the user prompt from a case dict."""
    return USER_PROMPT_TEMPLATE.format(
        story_summary=case.get("story_summary", "(not provided)"),
        before_state=case.get("before_state", "(not provided)"),
        after_state=case.get("after_state", "(not provided)"),
        trigger_type=case.get("trigger_type", "(not provided)"),
        action_type=case.get("action_type", "(not provided)"),
        scale=case.get("scale", "(not provided)"),
        target_name=case.get("target_name", "(not provided)"),
        period=case.get("period", "(not provided)"),
    )


def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON from the LLM response, handling markdown fences and extra text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_annotation(data: dict) -> tuple[bool, str]:
    """Validate that the annotation has the correct structure and values."""
    required_fields = ["before_lower", "before_upper", "after_lower", "after_upper"]

    for field in required_fields:
        if field not in data:
            return False, f"Missing field: {field}"

        entry = data[field]
        if not isinstance(entry, dict):
            return False, f"Field {field} is not a dict"

        trigram = entry.get("trigram")
        if trigram not in VALID_TRIGRAMS:
            return False, f"Invalid trigram '{trigram}' in {field}. Must be one of {VALID_TRIGRAMS}"

        confidence = entry.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            return False, f"Invalid confidence '{confidence}' in {field}. Must be one of {CONFIDENCE_LEVELS}"

        alternative = entry.get("alternative")
        if alternative not in VALID_TRIGRAMS:
            return False, f"Invalid alternative trigram '{alternative}' in {field}"

        if not entry.get("reasoning"):
            return False, f"Missing reasoning in {field}"

    return True, "OK"


def flatten_annotation(data: dict) -> dict:
    """Flatten the nested annotation into a flat dict for storage."""
    flat = {}
    for field in ["before_lower", "before_upper", "after_lower", "after_upper"]:
        entry = data[field]
        flat[field] = entry["trigram"]
        flat[f"{field}_confidence"] = entry["confidence"]
        flat[f"{field}_alternative"] = entry["alternative"]
        flat[f"{field}_reasoning"] = entry["reasoning"]
    return flat


def call_claude_api(client, case: dict, system_prompt: str, temperature: float) -> dict | None:
    """Call the Claude API and return parsed, validated annotation."""
    user_prompt = build_user_prompt(case)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            response_text = response.content[0].text
            parsed = extract_json_from_response(response_text)

            if parsed is None:
                logger.warning(
                    f"  Attempt {attempt}: Could not parse JSON from response. "
                    f"Response preview: {response_text[:200]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            valid, msg = validate_annotation(parsed)
            if not valid:
                logger.warning(f"  Attempt {attempt}: Validation failed: {msg}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            return flatten_annotation(parsed)

        except Exception as e:
            error_str = str(e)
            logger.warning(f"  Attempt {attempt}: API error: {error_str}")

            # Handle rate limiting specifically
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait_time = RETRY_DELAY * attempt * 2
                logger.info(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return None


def load_existing_annotations(path: Path) -> dict:
    """Load existing annotations file, return dict keyed by transition_id."""
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        return {a["transition_id"]: a for a in annotations if a.get("transition_id")}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_annotations(path: Path, annotations: list[dict], metadata: dict):
    """Save annotations to JSON file."""
    output = {
        "metadata": metadata,
        "annotations": annotations,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def run_annotation_pass(
    client,
    cases: list[dict],
    pass_number: int,
    system_prompt: str,
    temperature: float,
    output_path: Path,
    batch_size: int,
    resume: bool,
):
    """Run a single annotation pass over all cases."""
    logger.info(f"=== Pass {pass_number} (temperature={temperature}) ===")
    logger.info(f"  Total cases: {len(cases)}")
    logger.info(f"  Output: {output_path}")

    # Load existing annotations for resume
    existing = load_existing_annotations(output_path) if resume else {}
    if existing:
        logger.info(f"  Resuming: {len(existing)} already annotated")

    annotations = list(existing.values())
    annotated_ids = set(existing.keys())

    # Filter to unannotated cases
    remaining = [c for c in cases if get_case_id(c) not in annotated_ids]
    logger.info(f"  Remaining: {len(remaining)} cases")

    if not remaining:
        logger.info("  All cases already annotated. Skipping.")
        return

    # Process in batches
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    success_count = 0
    fail_count = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining))
        batch = remaining[start:end]

        logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} cases)")

        for i, case in enumerate(batch):
            case_id = get_case_id(case)
            case_name = case.get("target_name", case_id)
            global_idx = start + i + 1 + len(annotated_ids)

            logger.info(f"  [{global_idx}/{len(cases)}] {case_name}")

            result = call_claude_api(client, case, system_prompt, temperature)

            if result is not None:
                annotation = {
                    "transition_id": case_id,
                    "target_name": case_name,
                    **result,
                }
                annotations.append(annotation)
                success_count += 1
            else:
                logger.error(f"  FAILED after {MAX_RETRIES} retries: {case_name}")
                annotations.append({
                    "transition_id": case_id,
                    "target_name": case_name,
                    "error": f"Failed after {MAX_RETRIES} retries",
                })
                fail_count += 1

            # Rate limiting delay
            time.sleep(RATE_LIMIT_DELAY)

        # Save after each batch
        metadata = {
            "pass_number": pass_number,
            "model": MODEL,
            "temperature": temperature,
            "total_cases": len(cases),
            "annotated": success_count + len(annotated_ids),
            "failed": fail_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_annotations(output_path, annotations, metadata)
        logger.info(f"  Batch saved. Total: {success_count + len(annotated_ids)} done, {fail_count} failed")

    logger.info(
        f"=== Pass {pass_number} complete: "
        f"{success_count} new, {len(annotated_ids)} resumed, {fail_count} failed ==="
    )


def get_case_id(case: dict) -> str:
    """Get a unique identifier for a case."""
    tid = case.get("transition_id")
    if tid:
        return str(tid)
    # Fallback: use target_name + period
    return f"{case.get('target_name', 'unknown')}_{case.get('period', '')}"


def main():
    parser = argparse.ArgumentParser(description="LLM-based trigram annotation for gold set")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 cases for testing")
    parser.add_argument("--resume", action="store_true", help="Skip already-annotated cases")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of cases per batch (default: 50)")
    parser.add_argument(
        "--pass-number",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only pass 1 or pass 2 (default: both)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Input file path (default: {INPUT_PATH})",
    )
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is not set.")
        logger.error("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Import anthropic SDK
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed. Run: pip install anthropic")
        sys.exit(1)

    # Resolve input path
    input_path = Path(args.input) if args.input else INPUT_PATH
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run the gold set selection script first to create this file.")
        sys.exit(1)

    # Load cases
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    logger.info(f"Loaded {len(cases)} cases from {input_path}")

    # Dry run: limit to 5 cases
    if args.dry_run:
        cases = cases[:5]
        logger.info(f"DRY RUN: limited to {len(cases)} cases")

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Run passes
    if args.pass_number is None or args.pass_number == 1:
        run_annotation_pass(
            client=client,
            cases=cases,
            pass_number=1,
            system_prompt=SYSTEM_PROMPT_PASS1,
            temperature=0,
            output_path=OUTPUT_PASS1,
            batch_size=args.batch_size,
            resume=args.resume,
        )

    if args.pass_number is None or args.pass_number == 2:
        run_annotation_pass(
            client=client,
            cases=cases,
            pass_number=2,
            system_prompt=SYSTEM_PROMPT_PASS2,
            temperature=0.3,
            output_path=OUTPUT_PASS2,
            batch_size=args.batch_size,
            resume=args.resume,
        )

    logger.info("All annotation passes complete.")


if __name__ == "__main__":
    main()
