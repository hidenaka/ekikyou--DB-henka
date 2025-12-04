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
    "country",
    "main_domain",
    "life_domain",
    "tech_layer",
    "sources",
    "confidence_percent",
    "evidence_notes",
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
                if isinstance(obj.get("country"), str):
                    data["country"] = obj.get("country")
                if isinstance(obj.get("main_domain"), str):
                    data["main_domain"] = obj.get("main_domain")
                ld = obj.get("life_domain")
                if isinstance(ld, list):
                    try:
                        data["life_domain"] = ",".join(str(x) for x in ld)
                    except Exception:
                        data["life_domain"] = ""
                elif isinstance(ld, str):
                    data["life_domain"] = ld
                tl = obj.get("tech_layer")
                if isinstance(tl, list):
                    try:
                        data["tech_layer"] = ",".join(str(x) for x in tl)
                    except Exception:
                        data["tech_layer"] = ""
                elif isinstance(tl, str):
                    data["tech_layer"] = tl
                srcs = obj.get("sources")
                if isinstance(srcs, list):
                    try:
                        data["sources"] = json.dumps(srcs, ensure_ascii=False)
                    except Exception:
                        data["sources"] = ""
                elif isinstance(srcs, str):
                    data["sources"] = srcs
                cp = obj.get("confidence_percent")
                if isinstance(cp, int) and 0 <= cp <= 100:
                    data["confidence_percent"] = cp
                elif isinstance(cp, str):
                    try:
                        n = int(cp)
                        if 0 <= n <= 100:
                            data["confidence_percent"] = n
                    except Exception:
                        pass
                if isinstance(obj.get("evidence_notes"), str):
                    data["evidence_notes"] = obj.get("evidence_notes")
                writer.writerow({col: (data.get(col) if data.get(col) is not None else "") for col in COLUMNS})
        print(str(dst))
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
