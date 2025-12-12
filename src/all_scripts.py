# src/can_model_pipeline_full.py
"""
Integrated CAN ML Pipeline (All Features):
1. Discover labels
2. Collect balanced stratified sample for training
3. Preprocess data (convert hex -> numeric)
4. Train LightGBM model with ALL features and save
5. Create stratified test sample
6. Evaluate model with classification report + confusion matrix
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# CONFIG
# ===============================
DATA_PATH = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
MODEL_PATH = "D:/CAN_ML_Project/models/can_model_all_features.pkl"
FEATURES_PATH = "D:/CAN_ML_Project/models/can_model_all_features.json"
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
        labels.update(chunk[TARGET_COL].unique())
    print(f"✅ Found labels in dataset: {labels}")
    return labels


def collect_stratified_sample(csv_path, all_labels, max_per_class=MAX_PER_CLASS):
    counts = defaultdict(int)
    samples = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        print(f"[collect] chunk {i+1}, rows={len(chunk)}")
        for lbl in all_labels:
            if counts[lbl] < max_per_class:
                need = max_per_class - counts[lbl]
                subset = chunk[chunk[TARGET_COL] == lbl].head(need)
                if not subset.empty:
                    samples.append(subset)
                    counts[lbl] += len(subset)
        if all(counts[lbl] >= max_per_class for lbl in all_labels):
            print("[collect] reached target for all labels, stopping.")
            break
    df = pd.concat(samples, ignore_index=True)
    print(f"[collect] final sample shape: {df.shape}, class dist: {df[TARGET_COL].value_counts().to_dict()}")
    return df


def preprocess(df):
    """Convert all columns except label into numeric features."""
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    def try_num(x):
        try:
            if isinstance(x, str) and x.strip().isalnum():
                return int(x, 16)  # hex string → int
            return float(x)       # numeric string → float
        except Exception:
            return 0              # fallback for missing/invalid
    X = X.applymap(try_num)
    y = df[TARGET_COL].astype(str)
    return X, y


def train_and_save(X, y):
    """Train LightGBM model with all features and save."""
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
        eval_metric="logloss",
        callbacks=[]
    )

    preds = clf.predict(X_val)
    print("\n🔹 Validation Classification Report:\n", classification_report(y_val, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Full feature list saved to {FEATURES_PATH} ({len(X.columns)} features)")


def create_test_sample(csv_path, out_path, sample_size=TEST_SAMPLE_SIZE):
    """Create balanced stratified test sample from full dataset."""
    labels = get_all_labels(csv_path)
    counts = defaultdict(int)
    max_per_class = sample_size // len(labels)
    samples = []

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        print(f"[test_sample] Processing chunk {i+1}, rows={len(chunk)}")
        for lbl in chunk[TARGET_COL].unique():
            if counts[lbl] < max_per_class:
                need = max_per_class - counts[lbl]
                subset = chunk[chunk[TARGET_COL] == lbl].head(need)
                if not subset.empty:
                    samples.append(subset)
                    counts[lbl] += len(subset)
        if all(c >= max_per_class for c in counts.values()):
            break

    df_sample = pd.concat(samples, ignore_index=True)
    df_sample.to_csv(out_path, index=False)
    print(f"✅ Saved stratified test sample with {len(df_sample)} rows to {out_path}")
    print("Class distribution:", df_sample[TARGET_COL].value_counts().to_dict())


def test_on_csv(model_path, features_path, csv_path):
    """Evaluate saved model on test dataset."""
    clf = joblib.load(model_path)
    with open(features_path, "r") as f:
        expected_features = json.load(f)

    print(f"✅ Loaded model with {len(expected_features)} features.")

    df = pd.read_csv(csv_path)
    X, y_true = preprocess(df)

    # align to training features
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

    print("\n🔹 Training model with ALL features...")
    train_and_save(X, y)

    print("\n🔹 Creating stratified test sample...")
    create_test_sample(DATA_PATH, TEST_PATH, TEST_SAMPLE_SIZE)

    print("\n🔹 Evaluating model on test data...")
    test_on_csv(MODEL_PATH, FEATURES_PATH, TEST_PATH)


if __name__ == "__main__":
    main()
