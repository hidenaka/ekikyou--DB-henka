import sys
import json
from pathlib import Path
from datetime import datetime
from schema_v3 import Case
from id_utils import cases_path, next_id

def backup_path() -> Path:
    d = datetime.now().strftime("%Y%m%d")
    p = cases_path()
    return p.parent / f"cases_backup_{d}.jsonl"

def main() -> int:
    try:
        src = cases_path()
        if not src.exists():
            print("no cases.jsonl", file=sys.stderr)
            return 1
        bkp = backup_path()
        # create backup
        bkp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        seen = set()
        fixed = 0
        total = 0

        with open(bkp, "r", encoding="utf-8") as inp, open(src, "w", encoding="utf-8") as out:
            for i, line in enumerate(inp, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    print(f"Line {i}: invalid JSON in backup ({str(e)})", file=sys.stderr)
                    return 1
                try:
                    case = Case(**obj)
                except Exception as e:
                    print(f"Line {i}: schema invalid ({str(e)})", file=sys.stderr)
                    return 1
                tid = case.transition_id
                if isinstance(tid, str) and tid.strip():
                    if tid in seen:
                        out.flush()
                        new_tid = next_id(case.scale)
                        case.transition_id = new_tid
                        fixed += 1
                        tid = new_tid
                    seen.add(tid)
                total += 1
                out.write(json.dumps(case.model_dump(), ensure_ascii=False))
                out.write("\n")
        print(f"Fixed {fixed} duplicated IDs. Total records: {total}")
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
