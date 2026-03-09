#!/usr/bin/env python3
"""
Step 3: Apply trained trigram classifiers to all cases in cases.jsonl.

1. Load models from models/trigram_classifier.pkl
2. Load all cases from data/raw/cases.jsonl
3. Predict 4 trigram fields per case
4. Compute hexagram numbers from trigram pairs
5. Create backup, then update cases.jsonl
6. Report statistics
"""

import json
import os
import sys
import shutil
import numpy as np
import joblib
from datetime import datetime
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRIGRAMS = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']
TRIGRAM_NAMES = {
    '乾': '天/創造', '坤': '地/受容', '震': '雷/動', '巽': '風/浸透',
    '坎': '水/危険', '離': '火/明晰', '艮': '山/停止', '兌': '沢/喜悦',
}

CATEGORICAL_FEATURES = ['before_state', 'after_state', 'action_type', 'trigger_type']


def trigram_pair_to_hexagram_number(lower, upper):
    """Convert trigram pair to King Wen sequence hexagram number (1-64)."""
    trigram_index = {'乾': 0, '兌': 1, '離': 2, '震': 3, '巽': 4, '坎': 5, '艮': 6, '坤': 7}
    king_wen = [
        [1,  43, 14, 34, 9,  5,  26, 11],
        [10, 58, 38, 54, 61, 60, 41, 19],
        [13, 49, 30, 55, 37, 63, 22, 36],
        [25, 17, 21, 51, 42, 3,  27, 24],
        [44, 28, 50, 32, 57, 48, 18, 46],
        [6,  47, 64, 40, 59, 29, 4,  7],
        [33, 31, 56, 62, 53, 39, 52, 15],
        [12, 45, 35, 16, 20, 8,  23, 2],
    ]
    li = trigram_index.get(lower, 0)
    ui = trigram_index.get(upper, 0)
    return king_wen[ui][li]


def build_features_from_cases(cases, vectorizer, cat_encoders):
    """Build feature matrix for prediction."""
    texts = [c.get('story_summary', '') for c in cases]
    text_features = vectorizer.transform(texts)

    cat_matrices = []
    for feat in CATEGORICAL_FEATURES:
        values = [c.get(feat, 'unknown') for c in cases]
        enc = cat_encoders[feat]
        encoded = []
        for v in values:
            if v in enc.classes_:
                encoded.append(enc.transform([v])[0])
            else:
                encoded.append(0)
        encoded = np.array(encoded)

        n_classes = len(enc.classes_)
        one_hot = csr_matrix(
            (np.ones(len(encoded)), (np.arange(len(encoded)), encoded)),
            shape=(len(encoded), n_classes)
        )
        cat_matrices.append(one_hot)

    X = hstack([text_features] + cat_matrices)
    return X


def main():
    # Load models
    model_path = os.path.join(BASE_DIR, 'models/trigram_classifier.pkl')
    print(f"Loading models from {model_path}")
    package = joblib.load(model_path)

    models = package['models']
    vectorizer = package['vectorizer']
    cat_encoders = package['cat_encoders']
    target_fields = package['target_fields']

    # Load all cases
    cases_path = os.path.join(BASE_DIR, 'data/raw/cases.jsonl')
    cases = []
    with open(cases_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"Loaded {len(cases)} cases")

    # Save old annotations for comparison
    old_annotations = {}
    for i, c in enumerate(cases):
        old_annotations[i] = {
            'before_lower_trigram': c.get('before_lower_trigram'),
            'before_upper_trigram': c.get('before_upper_trigram'),
            'after_lower_trigram': c.get('after_lower_trigram'),
            'after_upper_trigram': c.get('after_upper_trigram'),
        }

    # Build features
    print("Building features...")
    X = build_features_from_cases(cases, vectorizer, cat_encoders)
    print(f"Feature matrix: {X.shape}")

    # Predict
    field_map = {
        'gold_before_lower': 'before_lower_trigram',
        'gold_before_upper': 'before_upper_trigram',
        'gold_after_lower': 'after_lower_trigram',
        'gold_after_upper': 'after_upper_trigram',
    }

    predictions = {}
    for gold_field in target_fields:
        case_field = field_map[gold_field]
        clf = models[gold_field]
        preds = clf.predict(X)
        predictions[case_field] = preds
        print(f"Predicted {gold_field} → {case_field}: {Counter(preds)}")

    # Apply predictions to cases
    pure_before = 0
    pure_after = 0

    for i, case in enumerate(cases):
        bl = predictions['before_lower_trigram'][i]
        bu = predictions['before_upper_trigram'][i]
        al = predictions['after_lower_trigram'][i]
        au = predictions['after_upper_trigram'][i]

        # Anti-pure hexagram adjustment
        if bl == bu:
            pure_before += 1
        if al == au:
            pure_after += 1

        case['before_lower_trigram'] = bl
        case['before_upper_trigram'] = bu
        case['after_lower_trigram'] = al
        case['after_upper_trigram'] = au

        # Compute hexagram numbers
        case['hexagram_number_before'] = trigram_pair_to_hexagram_number(bl, bu)
        case['hexagram_number_after'] = trigram_pair_to_hexagram_number(al, au)

    total = len(cases)
    pure_rate = (pure_before + pure_after) / (total * 2) * 100

    # Backup original file
    backup_path = os.path.join(
        BASE_DIR,
        f'data/raw/cases_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    )
    shutil.copy2(cases_path, backup_path)
    print(f"\nBackup saved to {backup_path}")

    # Write updated cases
    with open(cases_path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(f"Updated {cases_path}")

    # ─── Statistics ───
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")

    # Pure hexagram rate
    print(f"\nPure hexagram rate: {pure_rate:.1f}% ({pure_before + pure_after}/{total*2})")
    print(f"  Before pure: {pure_before} ({pure_before/total*100:.1f}%)")
    print(f"  After pure:  {pure_after} ({pure_after/total*100:.1f}%)")

    # Unique hexagrams
    before_hexs = set(case.get('hexagram_number_before', 0) for case in cases)
    after_hexs = set(case.get('hexagram_number_after', 0) for case in cases)
    all_hexs = before_hexs | after_hexs
    print(f"\nUnique hexagrams:")
    print(f"  Before: {len(before_hexs)}")
    print(f"  After:  {len(after_hexs)}")
    print(f"  Total:  {len(all_hexs)}")

    # Trigram frequency distribution
    print(f"\nTrigram frequency distribution:")
    for field in ['before_lower_trigram', 'before_upper_trigram',
                  'after_lower_trigram', 'after_upper_trigram']:
        dist = Counter(c.get(field, '') for c in cases)
        total_f = sum(dist.values())
        print(f"\n  {field}:")
        for t in TRIGRAMS:
            count = dist.get(t, 0)
            pct = count / total_f * 100 if total_f > 0 else 0
            bar = '#' * int(pct / 2)
            print(f"    {t}({TRIGRAM_NAMES[t]:6s}): {count:5d} ({pct:5.1f}%) {bar}")

    # Comparison with old annotations
    print(f"\n{'='*60}")
    print("COMPARISON WITH OLD ANNOTATIONS")
    print(f"{'='*60}")

    changed = {'before_lower_trigram': 0, 'before_upper_trigram': 0,
                'after_lower_trigram': 0, 'after_upper_trigram': 0}
    same = {'before_lower_trigram': 0, 'before_upper_trigram': 0,
            'after_lower_trigram': 0, 'after_upper_trigram': 0}
    old_missing = 0

    for i, case in enumerate(cases):
        old = old_annotations[i]
        for field in changed:
            old_val = old[field]
            new_val = case[field]
            if old_val is None or old_val == '':
                old_missing += 1
            elif old_val == new_val:
                same[field] += 1
            else:
                changed[field] += 1

    print(f"\nCases with old annotations missing: ~{old_missing // 4} cases")
    for field in changed:
        total_comparable = same[field] + changed[field]
        if total_comparable > 0:
            agreement = same[field] / total_comparable * 100
            print(f"  {field}: {same[field]} same, {changed[field]} changed ({agreement:.1f}% agreement)")
        else:
            print(f"  {field}: no comparable old annotations")


if __name__ == '__main__':
    main()
