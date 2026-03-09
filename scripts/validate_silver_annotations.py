#!/usr/bin/env python3
"""Validate silver set annotation outputs against schema."""

import json
import sys
import os

VALID_TRIGRAMS = {'乾', '坤', '震', '巽', '坎', '離', '艮', '兌'}
VALID_CONFIDENCE = {'high', 'medium', 'low'}
FIELDS = ['before_lower', 'before_upper', 'after_lower', 'after_upper']

def validate_annotation(ann, anchors_map):
    errors = []
    tid = ann.get('transition_id', '?')

    for field in FIELDS:
        if field not in ann:
            errors.append(f"{tid}: missing field '{field}'")
            continue
        entry = ann[field]
        if not isinstance(entry, dict):
            errors.append(f"{tid}.{field}: not a dict")
            continue
        trigram = entry.get('trigram')
        if trigram not in VALID_TRIGRAMS:
            errors.append(f"{tid}.{field}.trigram: invalid '{trigram}'")
        conf = entry.get('confidence')
        if conf not in VALID_CONFIDENCE:
            errors.append(f"{tid}.{field}.confidence: invalid '{conf}'")
        alt = entry.get('alternative')
        if alt and alt not in VALID_TRIGRAMS:
            errors.append(f"{tid}.{field}.alternative: invalid '{alt}'")

    # Check calibration anchor accuracy
    if tid in anchors_map:
        anchor = anchors_map[tid]
        for field in FIELDS:
            if field not in ann or not isinstance(ann[field], dict):
                continue
            assigned = ann[field].get('trigram')
            expected_key = f"expected_{field}"
            expected = anchor.get(expected_key)
            acceptable = anchor.get('acceptable_alternatives', {}).get(field, [])
            if expected and assigned != expected and assigned not in acceptable:
                errors.append(f"{tid}.{field}: calibration mismatch: got '{assigned}', expected '{expected}' or {acceptable}")

    return errors

def validate_file(path, anchors):
    anchors_map = {a['transition_id']: a for a in anchors}
    with open(path) as f:
        data = json.load(f)

    annotations = data if isinstance(data, list) else data.get('annotations', [])
    all_errors = []
    n_valid = 0
    for ann in annotations:
        errs = validate_annotation(ann, anchors_map)
        if errs:
            all_errors.extend(errs)
        else:
            n_valid += 1

    return n_valid, len(annotations), all_errors

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_silver_annotations.py <annotation_file.json> [anchors_file.json]")
        sys.exit(1)

    ann_path = sys.argv[1]
    anchors_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), '..', 'analysis', 'gold_set', 'calibration_anchors.json')

    with open(anchors_path) as f:
        anchors = json.load(f)

    n_valid, n_total, errors = validate_file(ann_path, anchors)
    print(f"Validated: {n_valid}/{n_total} annotations OK")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All annotations valid.")

if __name__ == '__main__':
    main()
