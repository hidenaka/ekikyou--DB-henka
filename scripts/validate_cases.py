import sys
import json
from pathlib import Path
from pydantic import ValidationError
from schema_v3 import Case

def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def cases_path() -> Path:
    return root_dir() / "data" / "raw" / "cases.jsonl"

def main() -> int:
    ok = 0
    ng = 0
    dup = 0
    p = cases_path()
    if not p.exists():
        print("OK: 0 records valid / NG: 0 records invalid")
        return 0
    seen = set()
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                ng += 1
                print(f"Line {i}: empty line (should be a JSON object)")
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                ng += 1
                print(f"Line {i}: invalid JSON ({str(e)})")
                continue
            try:
                model = Case(**obj)
                tid = obj.get("transition_id")
                if isinstance(tid, str) and tid.strip():
                    if tid in seen:
                        ng += 1
                        dup += 1
                        print(f"Line {i}: duplicate transition_id: {tid}")
                        continue
                    seen.add(tid)
                ok += 1
            except ValidationError as ve:
                ng += 1
                tid = obj.get("transition_id")
                print(f"Line {i}: NG transition_id={tid}")
                for err in ve.errors():
                    loc = ".".join(str(x) for x in err.get("loc", []))
                    msg = err.get("msg", "")
                    print(f"  - {loc}: {msg}")
    suffix = " (includes duplicate IDs)" if dup > 0 else ""
    print(f"OK: {ok} records valid / NG: {ng} records invalid{suffix}")
    return 1 if ng > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
