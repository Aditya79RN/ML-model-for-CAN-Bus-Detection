# src/train_model_final_v2.py

# 89% Accuracy on test set (v2 dataset)

"""
Final training pipeline:
- Discovers all labels in dataset
- Collects balanced stratified samples across classes
- Preprocesses (hex/string → numeric)
- Trains LightGBM classifier
- Saves model + feature names together
"""

import os
import joblib
import pandas as pd
from collections import defaultdict
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# CONFIG
# ===============================
DATA_PATH = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
OUT_MODEL = "D:/CAN_ML_Project/models/can_model_final_v2.pkl"

CHUNK_SIZE = 200_000         # adjust downward if memory issues
MAX_PER_CLASS = 200_000      # maximum rows to collect per label
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Label"

# ===============================
# STEP 1: DISCOVER ALL LABELS
# ===============================
def get_all_labels(csv_path):
    labels = set()
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        labels.update(chunk[TARGET_COL].unique())
    print(f"✅ Found labels in dataset: {labels}")
    return labels

# ===============================
# STEP 2: STRATIFIED SAMPLER
# ===============================
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

        # stop when ALL labels have enough
        if all(counts[lbl] >= max_per_class for lbl in all_labels):
            print("[collect] reached target for all labels, stopping collection.")
            break

    df = pd.concat(samples, ignore_index=True)
    # shuffle to avoid order bias
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[collect] final sample shape: {df.shape}, class dist: {df[TARGET_COL].value_counts().to_dict()}")
    return df

# ===============================
# STEP 3: FEATURE ENGINEERING
# ===============================
def preprocess(df):
    # drop label
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    # convert hex/strings to numeric safely
    def try_num(x):
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return 0
        try:
            # hex-like (contains a-f or starts with "0x")
            if any(c in s.lower() for c in "abcdef") or s.lower().startswith("0x"):
                return int(s.replace("0x", ""), 16)
            return float(s)
        except Exception:
            return 0

    X = X.applymap(try_num)
    y = df[TARGET_COL].astype(str)
    return X, y

# ===============================
# STEP 4: TRAIN MODEL
# ===============================
def train_and_save(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,
        random_state=RANDOM_STATE
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="logloss",
        callbacks=[]
    )

    preds = clf.predict(X_test)

    print("\n🔹 Classification Report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds, labels=clf.classes_)
    print("\n🔹 Confusion Matrix:\n", pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_))

    # save model + feature names together
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    payload = {"model": clf, "features": list(X.columns)}
    joblib.dump(payload, OUT_MODEL)

    print(f"\n✅ Model + features saved to {OUT_MODEL}")

# ===============================
# MAIN
# ===============================
def main():
    print("🔹 Discovering labels...")
    all_labels = get_all_labels(DATA_PATH)

    print("🔹 Collecting balanced stratified sample...")
    df = collect_stratified_sample(DATA_PATH, all_labels, MAX_PER_CLASS)

    print("🔹 Preprocessing...")
    X, y = preprocess(df)

    print("🔹 Training model...")
    train_and_save(X, y)


if __name__ == "__main__":
    main()
