import json
import re
from pathlib import Path
from schema_v3 import Scale

def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def cases_path() -> Path:
    return root_dir() / "data" / "raw" / "cases.jsonl"

def next_id(scale: Scale) -> str:
    prefix = {
        Scale.company: "CORP_JP_",
        Scale.individual: "PERS_JP_",
        Scale.family: "FAM_JP_",
        Scale.country: "COUN_JP_",
        Scale.other: "OTHR_JP_",
    }[scale]
    path = cases_path()
    max_n = 0
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                tid = obj.get("transition_id")
                if isinstance(tid, str) and tid.startswith(prefix):
                    m = re.search(r"(\d+)$", tid)
                    if m:
                        n = int(m.group(1))
                        if n > max_n:
                            max_n = n
    return f"{prefix}{max_n+1:03d}"

def ensure_unique(tid: str) -> None:
    path = cases_path()
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if obj.get("transition_id") == tid:
                raise ValueError("transition_id already exists")

def load_existing_ids() -> set[str]:
    ids: set[str] = set()
    p = cases_path()
    if not p.exists():
        return ids
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            tid = obj.get("transition_id")
            if isinstance(tid, str) and tid.strip():
                ids.add(tid)
    return ids

def generate_next_id(scale: Scale, used_ids: set[str]) -> str:
    prefix = {
        Scale.company: "CORP_JP_",
        Scale.individual: "PERS_JP_",
        Scale.family: "FAM_JP_",
        Scale.country: "COUN_JP_",
        Scale.other: "OTHR_JP_",
    }[scale]
    max_n = 0
    for tid in used_ids:
        if isinstance(tid, str) and tid.startswith(prefix):
            m = re.search(r"(\d+)$", tid)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
    return f"{prefix}{max_n+1:03d}"
