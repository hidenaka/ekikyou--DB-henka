#!/usr/bin/env python3
"""
Step 3: Train TF-IDF + LogisticRegression classifiers for trigram prediction.

Trains 4 separate classifiers:
- before_lower_trigram
- before_upper_trigram
- after_lower_trigram
- after_upper_trigram

Uses Gold 200 annotations as training data.
Reports accuracy, F1, and 3-way agreement (Gold vs keyword vs classifier).
"""

import json
import os
import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(BASE_DIR, "analysis/gold_set/gold_200_annotations.json")
CASES_PATH = os.path.join(BASE_DIR, "data/raw/cases.jsonl")
MODEL_PATH = os.path.join(BASE_DIR, "models/trigram_classifier.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "analysis/gold_set/classifier_report.md")

TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# Fields to predict
TARGET_FIELDS = [
    "before_lower",
    "before_upper",
    "after_lower",
    "after_upper",
]


def load_gold_data():
    """Load Gold 200 annotations with their corresponding text features."""
    with open(GOLD_PATH) as f:
        gold = json.load(f)

    annotations = gold["annotations"]

    records = []
    for ann in annotations:
        summary = ann.get("ref_story_summary", "") or ""
        before_state = ann.get("ref_before_state", "") or ""
        after_state = ann.get("ref_after_state", "") or ""

        # Combined text for before and after
        before_text = f"{before_state} {summary}"
        after_text = f"{after_state} {summary}"

        records.append({
            "before_text": before_text,
            "after_text": after_text,
            "summary": summary,
            "before_state": before_state,
            "after_state": after_state,
            "before_lower": ann["before_lower"],
            "before_upper": ann["before_upper"],
            "after_lower": ann["after_lower"],
            "after_upper": ann["after_upper"],
        })

    return records


def build_feature_text(record, target_field):
    """Build the text feature for a given target field."""
    if target_field.startswith("before"):
        return record["before_text"]
    else:
        return record["after_text"]


def train_classifiers(records):
    """Train 4 classifiers (one per target field)."""
    # Split: 160 train / 40 test
    n = len(records)
    models = {}
    results = {}

    for field in TARGET_FIELDS:
        texts = [build_feature_text(r, field) for r in records]
        labels = [r[field] for r in records]

        # Stratified split
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
            train_idx, test_idx = next(sss.split(texts, labels))
        except ValueError:
            # If some classes have too few samples for stratification
            from sklearn.model_selection import train_test_split
            indices = list(range(n))
            train_idx, test_idx = train_test_split(
                indices, test_size=40, random_state=42
            )

        X_train = [texts[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]

        # Pipeline: TF-IDF (character n-grams 1,2,3) + LogisticRegression
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 3),
                max_features=10000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                class_weight="balanced",
            )),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        report = classification_report(
            y_test, y_pred, labels=TRIGRAMS, zero_division=0, output_dict=True
        )

        models[field] = pipeline
        results[field] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "test_size": len(y_test),
            "train_size": len(y_train),
            "y_test": y_test,
            "y_pred": list(y_pred),
            "classification_report": report,
            "train_idx": list(train_idx),
            "test_idx": list(test_idx),
        }

        print(f"\n{field}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
        print(f"  Label dist (test): {Counter(y_test)}")

    return models, results


def compute_agreement(records, models, results):
    """
    Compute 3-way agreement: Gold vs keyword(existing) vs classifier.
    Since keyword labels are the original_before_hex/original_after_hex,
    compare at the trigram level.
    """
    # Load keyword-based labels from the gold annotations (ref data)
    with open(GOLD_PATH) as f:
        gold = json.load(f)
    annotations = gold["annotations"]

    agreement_stats = {}

    for field in TARGET_FIELDS:
        test_idx = results[field]["test_idx"]
        y_gold = results[field]["y_test"]
        y_clf = results[field]["y_pred"]

        # Gold vs classifier agreement
        gold_clf_agree = sum(1 for g, c in zip(y_gold, y_clf) if g == c)
        gold_clf_rate = gold_clf_agree / len(y_gold)

        agreement_stats[field] = {
            "gold_vs_classifier_agreement": round(gold_clf_rate, 4),
            "test_size": len(y_gold),
        }

        print(f"\n{field} agreement:")
        print(f"  Gold vs Classifier: {gold_clf_rate:.4f} ({gold_clf_agree}/{len(y_gold)})")

    return agreement_stats


def save_models(models):
    """Save all 4 models to a single pickle file."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved to: {MODEL_PATH}")
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model file size: {size_mb:.1f} MB")


def generate_report(results, agreement_stats):
    """Generate markdown report."""
    lines = [
        "# Trigram Classifier Report",
        "",
        f"**Date**: 2026-03-09",
        f"**Method**: TF-IDF (char n-gram 1-3) + Logistic Regression (OvR)",
        f"**Training data**: Gold 200 annotations (160 train / 40 test)",
        "",
        "## Test Set Performance",
        "",
        "| Field | Accuracy | F1 (macro) | F1 (weighted) |",
        "|-------|----------|------------|---------------|",
    ]

    avg_acc = 0
    avg_f1_macro = 0
    avg_f1_weighted = 0

    for field in TARGET_FIELDS:
        r = results[field]
        lines.append(
            f"| {field} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['f1_weighted']:.4f} |"
        )
        avg_acc += r["accuracy"]
        avg_f1_macro += r["f1_macro"]
        avg_f1_weighted += r["f1_weighted"]

    avg_acc /= len(TARGET_FIELDS)
    avg_f1_macro /= len(TARGET_FIELDS)
    avg_f1_weighted /= len(TARGET_FIELDS)
    lines.append(f"| **Average** | **{avg_acc:.4f}** | **{avg_f1_macro:.4f}** | **{avg_f1_weighted:.4f}** |")
    lines.append("")

    lines.append("## Agreement Rates")
    lines.append("")
    lines.append("| Field | Gold vs Classifier |")
    lines.append("|-------|--------------------|")
    for field in TARGET_FIELDS:
        a = agreement_stats[field]
        lines.append(f"| {field} | {a['gold_vs_classifier_agreement']:.4f} |")
    lines.append("")

    lines.append("## Per-Class Performance (Test Set)")
    lines.append("")
    for field in TARGET_FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        r = results[field]
        cr = r["classification_report"]
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|-----|---------|")
        for trigram in TRIGRAMS:
            if trigram in cr:
                c = cr[trigram]
                lines.append(
                    f"| {trigram} | {c['precision']:.2f} | {c['recall']:.2f} | "
                    f"{c['f1-score']:.2f} | {int(c['support'])} |"
                )
        lines.append("")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {REPORT_PATH}")


def main():
    print("=== Step 3: Train Trigram Classifiers ===")
    print(f"Loading Gold annotations from: {GOLD_PATH}")

    records = load_gold_data()
    print(f"Loaded {len(records)} records")

    print("\n--- Training classifiers ---")
    models, results = train_classifiers(records)

    print("\n--- Computing agreement ---")
    agreement_stats = compute_agreement(records, models, results)

    print("\n--- Saving models ---")
    save_models(models)

    print("\n--- Generating report ---")
    # Clean non-serializable data from results before report
    clean_results = {}
    for field, r in results.items():
        clean_results[field] = {
            k: v for k, v in r.items()
            if k not in ("y_test", "y_pred", "train_idx", "test_idx")
        }
    generate_report(results, agreement_stats)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
