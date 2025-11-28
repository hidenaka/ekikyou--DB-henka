import sys
import json
from pathlib import Path
from datetime import datetime
from schema_v3 import Case
from id_utils import cases_path, load_existing_ids, generate_next_id

TARGETS = {"日立製作所", "日本航空（JAL）", "日本マクドナルド", "ソニーグループ", "カシオ計算機"}

def backup_path() -> Path:
    d = datetime.now().strftime("%Y%m%d")
    p = cases_path()
    return p.parent / f"cases_backup_shock_{d}.jsonl"

def main() -> int:
    try:
        src = cases_path()
        if not src.exists():
            print("no cases.jsonl", file=sys.stderr)
            return 1
        bkp = backup_path()
        bkp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        used = load_existing_ids()
        fixed = 0
        total = 0

        with open(bkp, "r", encoding="utf-8") as inp, open(src, "w", encoding="utf-8") as out:
            for i, line in enumerate(inp, start=1):
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                case = Case(**obj)
                tid = case.transition_id or ""
                name = case.target_name
                if name in TARGETS and tid == "CORP_JP_007":
                    new_tid = generate_next_id(case.scale, used)
                    case.transition_id = new_tid
                    used.add(new_tid)
                    fixed += 1
                else:
                    if tid:
                        used.add(tid)
                total += 1
                out.write(json.dumps(case.model_dump(), ensure_ascii=False))
                out.write("\n")
        print(f"Reassigned {fixed} IDs for Shock_Recovery batch. Total records: {total}")
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
