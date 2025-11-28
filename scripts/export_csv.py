import sys
import json
import csv
from pathlib import Path
from pydantic import ValidationError
from schema_v3 import Case

COLUMNS = [
    "transition_id",
    "target_name",
    "scale",
    "period",
    "story_summary",
    "before_state",
    "trigger_type",
    "action_type",
    "after_state",
    "before_hex",
    "trigger_hex",
    "action_hex",
    "after_hex",
    "pattern_type",
    "outcome",
    "free_tags",
    "source_type",
    "credibility_rank",
    "classical_before_hexagram",
    "classical_action_hexagram",
    "classical_after_hexagram",
    "logic_memo",
]

def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def cases_path() -> Path:
    return root_dir() / "data" / "raw" / "cases.jsonl"

def csv_path() -> Path:
    return root_dir() / "data" / "raw" / "cases.csv"

def main() -> int:
    src = cases_path()
    dst = csv_path()
    if not src.exists():
        print("no cases.jsonl", file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(src, "r", encoding="utf-8") as inp, open(dst, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=COLUMNS)
            writer.writeheader()
            for i, line in enumerate(inp, start=1):
                s = line.strip()
                if not s:
                    raise ValueError(f"Line {i}: empty row in cases.jsonl")
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {i}: invalid JSON ({e})") from e
                try:
                    case = Case(**obj)
                except ValidationError as e:
                    raise ValueError(f"Line {i}: invalid record ({e})") from e
                data = case.model_dump()
                tags = data.get("free_tags")
                if not isinstance(tags, list):
                    raise ValueError(f"Line {i}: free_tags must be an array")
                data["free_tags"] = ";".join(tags)
                writer.writerow({col: (data.get(col) if data.get(col) is not None else "") for col in COLUMNS})
        print(str(dst))
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
