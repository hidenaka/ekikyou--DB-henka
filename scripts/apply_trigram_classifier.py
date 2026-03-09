#!/usr/bin/env python3
"""
Step 4: Apply trained trigram classifiers to all cases in cases.jsonl.

1. Load models from models/trigram_classifier.pkl
2. Predict 4 trigram fields for all 11,336 cases
3. Update cases.jsonl with new predictions
4. Recalculate classical_before/after_hexagram
5. Create backup before updating
"""

import json
import os
import pickle
import shutil
from datetime import datetime
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/trigram_classifier.pkl")
CASES_PATH = os.path.join(BASE_DIR, "data/raw/cases.jsonl")

# King Wen sequence
KING_WEN = {
    ("乾","乾"):1, ("坤","乾"):11, ("震","乾"):34, ("巽","乾"):9,
    ("坎","乾"):5, ("離","乾"):14, ("艮","乾"):26, ("兌","乾"):43,
    ("乾","坤"):12, ("坤","坤"):2, ("震","坤"):16, ("巽","坤"):20,
    ("坎","坤"):8, ("離","坤"):35, ("艮","坤"):23, ("兌","坤"):45,
    ("乾","震"):25, ("坤","震"):24, ("震","震"):51, ("巽","震"):42,
    ("坎","震"):3, ("離","震"):21, ("艮","震"):27, ("兌","震"):17,
    ("乾","巽"):44, ("坤","巽"):46, ("震","巽"):32, ("巽","巽"):57,
    ("坎","巽"):48, ("離","巽"):50, ("艮","巽"):18, ("兌","巽"):28,
    ("乾","坎"):6, ("坤","坎"):7, ("震","坎"):40, ("巽","坎"):59,
    ("坎","坎"):29, ("離","坎"):64, ("艮","坎"):4, ("兌","坎"):47,
    ("乾","離"):13, ("坤","離"):36, ("震","離"):55, ("巽","離"):37,
    ("坎","離"):63, ("離","離"):30, ("艮","離"):22, ("兌","離"):49,
    ("乾","艮"):33, ("坤","艮"):15, ("震","艮"):62, ("巽","艮"):53,
    ("坎","艮"):39, ("離","艮"):56, ("艮","艮"):52, ("兌","艮"):31,
    ("乾","兌"):10, ("坤","兌"):19, ("震","兌"):54, ("巽","兌"):61,
    ("坎","兌"):60, ("離","兌"):38, ("艮","兌"):41, ("兌","兌"):58,
}

HEX_NAMES = {
    1: "乾為天", 2: "坤為地", 3: "水雷屯", 4: "山水蒙",
    5: "水天需", 6: "天水訟", 7: "地水師", 8: "水地比",
    9: "風天小畜", 10: "天沢履", 11: "地天泰", 12: "天地否",
    13: "天火同人", 14: "火天大有", 15: "地山謙", 16: "雷地豫",
    17: "沢雷随", 18: "山風蠱", 19: "地沢臨", 20: "風地観",
    21: "火雷噬嗑", 22: "山火賁", 23: "山地剥", 24: "地雷復",
    25: "天雷無妄", 26: "山天大畜", 27: "山雷頤", 28: "沢風大過",
    29: "坎為水", 30: "離為火", 31: "沢山咸", 32: "雷風恒",
    33: "天山遯", 34: "雷天大壮", 35: "火地晋", 36: "地火明夷",
    37: "風火家人", 38: "火沢睽", 39: "水山蹇", 40: "雷水解",
    41: "山沢損", 42: "風雷益", 43: "沢天夬", 44: "天風姤",
    45: "沢地萃", 46: "地風升", 47: "沢水困", 48: "水風井",
    49: "沢火革", 50: "火風鼎", 51: "震為雷", 52: "艮為山",
    53: "風山漸", 54: "雷沢帰妹", 55: "雷火豊", 56: "火山旅",
    57: "巽為風", 58: "兌為沢", 59: "風水渙", 60: "水沢節",
    61: "風沢中孚", 62: "雷山小過", 63: "水火既済", 64: "火水未済"
}


def get_hexagram_label(lower, upper):
    """Get hexagram label from lower and upper trigrams."""
    num = KING_WEN.get((lower, upper))
    if num:
        return f"{num}_{HEX_NAMES.get(num, '')}"
    return None


def main():
    print("=== Step 4: Apply Trigram Classifiers to All Cases ===")

    # Load models
    print(f"Loading models from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} classifiers: {list(models.keys())}")

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CASES_PATH.replace(".jsonl", f"_backup_{timestamp}.jsonl")
    shutil.copy2(CASES_PATH, backup_path)
    print(f"Backup created: {backup_path}")

    # Load all cases
    cases = []
    with open(CASES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"Loaded {len(cases)} cases")

    # Prepare text features
    before_texts = []
    after_texts = []
    for case in cases:
        summary = case.get("story_summary", "") or ""
        before_state = case.get("before_state", "") or ""
        after_state = case.get("after_state", "") or ""
        before_texts.append(f"{before_state} {summary}")
        after_texts.append(f"{after_state} {summary}")

    # Predict
    print("\nPredicting...")
    predictions = {}
    for field, model in models.items():
        if field.startswith("before"):
            texts = before_texts
        else:
            texts = after_texts
        preds = model.predict(texts)
        predictions[field] = preds
        print(f"  {field}: {Counter(preds)}")

    # Update cases
    print("\nUpdating cases...")
    updated = 0
    for i, case in enumerate(cases):
        bl = predictions["before_lower"][i]
        bu = predictions["before_upper"][i]
        al = predictions["after_lower"][i]
        au = predictions["after_upper"][i]

        case["before_lower_trigram"] = bl
        case["before_upper_trigram"] = bu
        case["after_lower_trigram"] = al
        case["after_upper_trigram"] = au

        # Recalculate hexagrams
        before_hex = get_hexagram_label(bl, bu)
        after_hex = get_hexagram_label(al, au)

        if before_hex:
            case["classical_before_hexagram"] = before_hex
        if after_hex:
            case["classical_after_hexagram"] = after_hex

        updated += 1

    # Write back
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Updated {updated} cases")
    print(f"Saved to: {CASES_PATH}")

    # Summary statistics
    print("\n=== Distribution Summary ===")
    for field in ["before_lower_trigram", "before_upper_trigram",
                   "after_lower_trigram", "after_upper_trigram"]:
        dist = Counter(case.get(field) for case in cases)
        print(f"\n{field}:")
        for k, v in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v} ({v/len(cases)*100:.1f}%)")

    # Hexagram stats
    before_hex_set = set(case.get("classical_before_hexagram") for case in cases)
    after_hex_set = set(case.get("classical_after_hexagram") for case in cases)
    before_hex_set.discard(None)
    after_hex_set.discard(None)

    pure_before = sum(1 for c in cases
                      if c.get("before_lower_trigram") == c.get("before_upper_trigram"))
    pure_after = sum(1 for c in cases
                     if c.get("after_lower_trigram") == c.get("after_upper_trigram"))

    print(f"\nUnique before hexagrams: {len(before_hex_set)}/64")
    print(f"Unique after hexagrams: {len(after_hex_set)}/64")
    print(f"Pure before rate: {pure_before/len(cases)*100:.1f}%")
    print(f"Pure after rate: {pure_after/len(cases)*100:.1f}%")


if __name__ == "__main__":
    main()
