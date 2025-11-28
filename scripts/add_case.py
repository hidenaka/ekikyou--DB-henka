import sys
import json
from pathlib import Path
from typing import Optional
from schema_v3 import Case
from id_utils import load_existing_ids, generate_next_id, ensure_unique

def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def cases_path() -> Path:
    return root_dir() / "data" / "raw" / "cases.jsonl"

def read_input(path: Optional[str]) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = sys.stdin.read()
    return json.loads(data)


def main() -> int:
    try:
        arg = sys.argv[1] if len(sys.argv) > 1 else None
        raw = read_input(arg)
        model = Case(**raw)
        used = load_existing_ids()
        tid = model.transition_id
        if not tid or (isinstance(tid, str) and not tid.strip()) or tid in used:
            new_tid = generate_next_id(model.scale, used)
            model.transition_id = new_tid
            used.add(new_tid)
        else:
            ensure_unique(tid)
        p = cases_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(model.model_dump(), ensure_ascii=False))
            f.write("\n")
        print(f"追加完了: {model.transition_id}")
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
