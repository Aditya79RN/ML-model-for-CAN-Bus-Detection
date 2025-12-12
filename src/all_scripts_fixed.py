# all_scripts_fixed.py
# Generates Image for confusion matrix

"""
Integrated CAN ML Pipeline (fixed):
- discover labels
- collect balanced stratified sample for training
- preprocess (safe hex->numeric conversion)
- train LightGBM and save model + features
- create balanced stratified test sample (fixed bug)
- evaluate model on test sample (classification report + confusion matrix)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# CONFIG
# ===============================
DATA_PATH = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
MODEL_PATH = "D:/CAN_ML_Project/models/can_model_final_v2.pkl"
FEATURES_PATH = "D:/CAN_ML_Project/models/can_model_final_v2_features.json"
TEST_PATH = "D:/CAN_ML_Project/processed/test_sample.csv"

CHUNK_SIZE = 200_000
MAX_PER_CLASS = 200_000
TEST_SAMPLE_SIZE = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Label"

# ===============================
# HELPERS
# ===============================
def get_all_labels(csv_path):
    labels = set()
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        if TARGET_COL in chunk.columns:
            labels.update(chunk[TARGET_COL].dropna().unique())
        else:
            raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")
    labels = sorted([str(x) for x in labels])
    print(f"✅ Found labels in dataset: {labels}")
    return labels


def collect_stratified_sample(csv_path, all_labels, max_per_class=MAX_PER_CLASS):
    counts = {lbl: 0 for lbl in all_labels}
    samples = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        print(f"[collect] chunk {i+1}, rows={len(chunk)}")
        for lbl in all_labels:
            if counts[lbl] < max_per_class:
                subset = chunk[chunk[TARGET_COL] == lbl]
                if not subset.empty:
                    need = max_per_class - counts[lbl]
                    take_n = min(len(subset), need)
                    # take the top rows (fast) — you can randomize if required
                    chosen = subset.head(take_n)
                    samples.append(chosen)
                    counts[lbl] += len(chosen)
        # Stop only if every label has reached its quota
        if all(counts[lbl] >= max_per_class for lbl in all_labels):
            print("[collect] reached target for all labels, stopping collection.")
            break

    df = pd.concat(samples, ignore_index=True)
    print(f"[collect] final sample shape: {df.shape}, class dist: {df[TARGET_COL].value_counts().to_dict()}")
    return df


def try_num(val):
    """Safe numeric conversion:
    - If looks like hex (hex digits), parse base 16
    - Else try float
    - Else return 0
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s == "":
        return 0.0
    # if purely hex-like (only 0-9a-fA-F) -> parse hex
    s_n = s.lower().replace("0x", "")
    if all(ch in "0123456789abcdef" for ch in s_n) and len(s_n) > 0:
        try:
            return float(int(s_n, 16))
        except Exception:
            pass
    # try decimal
    try:
        return float(s)
    except Exception:
        # fallback: drop non-digits and try again
        cleaned = "".join(ch for ch in s if ch.isdigit() or ch == "." or ch == "-")
        try:
            return float(cleaned) if cleaned != "" else 0.0
        except Exception:
            return 0.0


def preprocess(df):
    """Return numeric X, and y series."""
    # Defensive copy
    df = df.copy()
    # Make sure target exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column not found in dataframe.")

    X = df.drop(columns=[TARGET_COL], errors="ignore")
    # map per-column with try_num (avoids deprecated DataFrame.applymap warnings)
    X = X.apply(lambda col: col.map(try_num))
    y = df[TARGET_COL].astype(str)
    return X, y


def train_and_save(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[]
    )

    preds = clf.predict(X_val)
    print("\n🔹 Validation Classification Report:\n", classification_report(y_val, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Feature list saved to {FEATURES_PATH}")


def create_test_sample(csv_path, out_path, sample_size=TEST_SAMPLE_SIZE, all_labels=None):
    """Create a balanced stratified test sample across all_labels without reading full CSV."""
    if all_labels is None:
        all_labels = get_all_labels(csv_path)
    n_labels = len(all_labels)
    if n_labels == 0:
        raise ValueError("No labels found to create test sample.")
    max_per_class = max(1, sample_size // n_labels)
    counts = {lbl: 0 for lbl in all_labels}
    samples = []

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        print(f"[test_sample] Processing chunk {i+1}, rows={len(chunk)}")
        for lbl in all_labels:
            if counts[lbl] < max_per_class:
                subset = chunk[chunk[TARGET_COL] == lbl]
                if not subset.empty:
                    need = max_per_class - counts[lbl]
                    take_n = min(len(subset), need)
                    chosen = subset.sample(n=take_n, random_state=RANDOM_STATE) if len(subset) > take_n else subset.head(take_n)
                    samples.append(chosen)
                    counts[lbl] += len(chosen)
        # stop when we've collected enough total or each label reached quota
        total_collected = sum(counts.values())
        if total_collected >= max_per_class * n_labels:
            print("[test_sample] Collected enough samples for all classes.")
            break

    if len(samples) == 0:
        raise RuntimeError("Could not collect any samples for the test set (maybe CSV empty or labels mismatch).")

    df_sample = pd.concat(samples, ignore_index=True)
    # If we accidentally collected more than requested because of rounding, downsample
    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    df_sample.to_csv(out_path, index=False)
    print(f"✅ Saved stratified test sample with {len(df_sample)} rows to {out_path}")
    print("Class distribution:", df_sample[TARGET_COL].value_counts().to_dict())
    return df_sample


def test_on_csv(model_path, features_path, csv_path):
    clf = joblib.load(model_path)
    with open(features_path, "r") as f:
        expected_features = json.load(f)

    print(f"✅ Loaded model with {len(expected_features)} features.")

    df = pd.read_csv(csv_path)
    X, y_true = preprocess(df)

    # align features (add any missing features as zeros)
    X = X.reindex(columns=expected_features, fill_value=0)

    preds = clf.predict(X)

    print("\n🔹 Classification Report on Test Data:\n")
    print(classification_report(y_true, preds))

    cm = confusion_matrix(y_true, preds, labels=clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Data")
    plt.tight_layout()
    plt.show()


# ===============================
# MAIN PIPELINE
# ===============================
def main():
    print("🔹 Discovering labels...")
    all_labels = get_all_labels(DATA_PATH)

    print("\n🔹 Collecting balanced stratified sample for training...")
    df = collect_stratified_sample(DATA_PATH, all_labels, MAX_PER_CLASS)

    print("\n🔹 Preprocessing training data...")
    X, y = preprocess(df)

    print("\n🔹 Training model...")
    train_and_save(X, y)

    print("\n🔹 Creating stratified test sample...")
    create_test_sample(DATA_PATH, TEST_PATH, TEST_SAMPLE_SIZE, all_labels)

    print("\n🔹 Evaluating model on test data...")
    test_on_csv(MODEL_PATH, FEATURES_PATH, TEST_PATH)


if __name__ == "__main__":
    main()
