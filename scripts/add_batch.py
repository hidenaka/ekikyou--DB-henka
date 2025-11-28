import sys
import json
from typing import List
from schema_v3 import Case
from id_utils import cases_path, load_existing_ids, generate_next_id

def main() -> int:
    try:
        if len(sys.argv) < 2:
            print("usage: python scripts/add_batch.py <path_to_json_array>", file=sys.stderr)
            return 1
        src = sys.argv[1]
        with open(src, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            print("input must be a JSON array", file=sys.stderr)
            return 1
        count = 0
        path = cases_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        used = load_existing_ids()
        with open(path, "a", encoding="utf-8") as out:
            for idx, item in enumerate(arr, start=1):
                try:
                    case = Case(**item)
                    tid = case.transition_id
                    if not tid or (isinstance(tid, str) and not tid.strip()):
                        new_tid = generate_next_id(case.scale, used)
                        case.transition_id = new_tid
                        used.add(new_tid)
                    else:
                        if tid in used:
                            # 重複IDはその場で採番し直して続行
                            new_tid = generate_next_id(case.scale, used)
                            case.transition_id = new_tid
                            used.add(new_tid)
                            print(f"Index {idx}: duplicate transition_id {tid} -> reassigned {new_tid}", file=sys.stderr)
                        else:
                            used.add(tid)
                    out.write(json.dumps(case.model_dump(), ensure_ascii=False))
                    out.write("\n")
                    count += 1
                except Exception as e:
                    print(f"Index {idx}: {str(e)}", file=sys.stderr)
                    return 1
        print(f"追加完了: {count}件")
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
