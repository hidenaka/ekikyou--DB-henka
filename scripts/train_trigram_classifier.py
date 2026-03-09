#!/usr/bin/env python3
"""
Step 2: Train TF-IDF + LogisticRegression classifiers for trigram prediction.

Trains 4 separate classifiers:
- before_lower_trigram
- before_upper_trigram
- after_lower_trigram
- after_upper_trigram

Features: TF-IDF on story_summary + one-hot state/action/trigger labels
Split: 160 train / 40 test
"""

import json
import os
import sys
import numpy as np
import joblib
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRIGRAMS = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']

TARGET_FIELDS = [
    'gold_before_lower',
    'gold_before_upper',
    'gold_after_lower',
    'gold_after_upper',
]

CATEGORICAL_FEATURES = ['before_state', 'after_state', 'action_type', 'trigger_type']


def load_data():
    """Load gold annotations."""
    path = os.path.join(BASE_DIR, 'analysis/gold_set/gold_200_annotations.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotated cases")
    return data


def build_features(data, fit=True, vectorizer=None, cat_encoders=None):
    """Build feature matrix from text + categorical features."""
    # Text features
    texts = [c.get('story_summary', '') for c in data]

    if fit:
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=3000,
            sublinear_tf=True,
        )
        text_features = vectorizer.fit_transform(texts)
    else:
        text_features = vectorizer.transform(texts)

    # Categorical features - one-hot encode
    if fit:
        cat_encoders = {}

    cat_matrices = []
    for feat in CATEGORICAL_FEATURES:
        values = [c.get(feat, 'unknown') for c in data]
        if fit:
            enc = LabelEncoder()
            encoded = enc.fit_transform(values)
            cat_encoders[feat] = enc
        else:
            # Handle unseen labels
            enc = cat_encoders[feat]
            encoded = []
            for v in values:
                if v in enc.classes_:
                    encoded.append(enc.transform([v])[0])
                else:
                    encoded.append(0)  # fallback
            encoded = np.array(encoded)

        # Convert to one-hot sparse matrix
        n_classes = len(cat_encoders[feat].classes_) if not fit else len(enc.classes_)
        one_hot = csr_matrix(
            (np.ones(len(encoded)), (np.arange(len(encoded)), encoded)),
            shape=(len(encoded), n_classes)
        )
        cat_matrices.append(one_hot)

    # Combine all features
    X = hstack([text_features] + cat_matrices)

    return X, vectorizer, cat_encoders


def train_and_evaluate():
    """Train classifiers and report results."""
    data = load_data()

    # Split
    train_data, test_data = train_test_split(
        data, test_size=40, random_state=42, shuffle=True
    )
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Build features
    X_train, vectorizer, cat_encoders = build_features(train_data, fit=True)
    X_test, _, _ = build_features(test_data, fit=False,
                                   vectorizer=vectorizer,
                                   cat_encoders=cat_encoders)

    print(f"Feature matrix shape: {X_train.shape}")

    # Train 4 classifiers
    models = {}
    results = {}

    for field in TARGET_FIELDS:
        print(f"\n{'='*60}")
        print(f"Training classifier for: {field}")
        print(f"{'='*60}")

        y_train = [c[field] for c in train_data]
        y_test = [c[field] for c in test_data]

        # Label distribution
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        print(f"Train distribution: {dict(sorted(train_dist.items()))}")
        print(f"Test distribution:  {dict(sorted(test_dist.items()))}")

        # Train
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\nAccuracy: {acc:.3f}")
        print(f"Macro F1: {f1_macro:.3f}")
        print(f"Weighted F1: {f1_weighted:.3f}")

        # Per-class report
        labels_present = sorted(set(y_train) | set(y_test))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=labels_present, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels_present)
        print(f"Confusion Matrix (labels: {labels_present}):")
        print(cm)

        models[field] = clf
        results[field] = {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'labels': labels_present,
        }

    # Save everything
    model_package = {
        'models': models,
        'vectorizer': vectorizer,
        'cat_encoders': cat_encoders,
        'target_fields': TARGET_FIELDS,
        'categorical_features': CATEGORICAL_FEATURES,
        'results': results,
    }

    model_path = os.path.join(BASE_DIR, 'models/trigram_classifier.pkl')
    joblib.dump(model_package, model_path)
    print(f"\nModels saved to {model_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for field in TARGET_FIELDS:
        r = results[field]
        print(f"{field:25s} | Acc={r['accuracy']:.3f} | F1(macro)={r['f1_macro']:.3f} | F1(weighted)={r['f1_weighted']:.3f}")

    return results


if __name__ == '__main__':
    train_and_evaluate()
