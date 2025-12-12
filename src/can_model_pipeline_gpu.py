# src/can_model_pipeline_gpu.py
"""
Integrated CAN ML Pipeline with optional LightGBM GPU (OpenCL) support.

It will:
- Try to detect OpenCL devices (pyopencl -> clinfo)
- Prefer an AMD GPU (Radeon) if present
- Try to run LightGBM with device='gpu' and gpu_platform_id/gpu_device_id
- If GPU not available / LightGBM wasn't built with GPU -> automatically fall back to CPU
"""

import os
import json
import joblib
import subprocess
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# CONFIG (edit paths if needed)
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
# OPENCL / GPU DETECTION HELPERS
# ===============================
def detect_opencl_devices():
    """
    Tries to detect OpenCL platforms & devices.
    1) Try pyopencl (best)
    2) Fallback to clinfo command (if installed on system)
    Returns: (platform_id, device_id, info_dict) or (None, None, None) if none found
    """
    # try pyopencl first
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        info = []
        for p_idx, p in enumerate(platforms):
            for d_idx, d in enumerate(p.get_devices()):
                info.append({
                    "platform_index": p_idx,
                    "device_index": d_idx,
                    "platform_name": p.name,
                    "device_name": d.name,
                    "vendor": d.vendor,
                    "device_type": cl.device_type.to_string(d.type) if hasattr(cl.device_type, 'to_string') else str(d.type)
                })
        if info:
            # prefer AMD/Radeon device if available
            for entry in info:
                if "amd" in entry["vendor"].lower() or "radeon" in entry["device_name"].lower():
                    return entry["platform_index"], entry["device_index"], entry
            # else return first device
            return info[0]["platform_index"], info[0]["device_index"], info[0]
    except Exception:
        pass

    # fallback: clinfo (if installed)
    try:
        proc = subprocess.run(["clinfo"], capture_output=True, text=True, check=True)
        out = proc.stdout
        # naive parsing: look for "Platform #" and "Device #", vendor/device lines
        # We'll select the first AMD platform/device if present
        lines = out.splitlines()
        platform_idx = None
        device_idx = None
        found_amd = False
        # crude search to find "Platform Profile" blocks
        for i, line in enumerate(lines):
            low = line.lower()
            if "platform" in low and "platform" in low[:20].lower():
                # attempt to detect vendor in the following few lines
                block = "\n".join(lines[i:i+10]).lower()
                if "amd" in block or "advanced micro devices" in block or "radeon" in block:
                    # record this platform and try to find device afterwards
                    platform_idx = 0  # clinfo numbering is messy; return 0 as default
                    # find a "Device Name" after this line
                    for j in range(i, min(len(lines), i+40)):
                        if "device" in lines[j].lower() and "name" in lines[j].lower():
                            device_idx = 0
                            found_amd = True
                            break
                    if found_amd:
                        return platform_idx, device_idx, {"platform_name":"AMD (via clinfo)", "device_name":"unknown", "vendor":"AMD"}
        # if nothing found, return None
    except Exception:
        pass

    return None, None, None


def try_get_gpu_params_prefer_amd():
    """
    Returns a tuple (use_gpu:bool, params:dict, info:dict)
    """
    platform_id, device_id, info = detect_opencl_devices()
    if platform_id is not None:
        print(f"🔹 OpenCL device detected: {info}")
        # LightGBM expects gpu_platform_id and gpu_device_id int values
        return True, {"device": "gpu", "gpu_platform_id": int(platform_id), "gpu_device_id": int(device_id)}, info
    else:
        print("⚠️ No OpenCL GPU device detected (pyopencl/clinfo not found or no compatible GPU). Will use CPU.")
        return False, {}, None

# ===============================
# DATA HELPERS (unchanged)
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
            print("[collect] reached target for all labels, stopping collection.")
            break
    df = pd.concat(samples, ignore_index=True)
    print(f"[collect] final sample shape: {df.shape}, class dist: {df[TARGET_COL].value_counts().to_dict()}")
    return df


def preprocess(df):
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    def try_num(x):
        try:
            # convert hex-looking strings to int
            s = str(x).strip()
            if s.startswith("0x") or all(c in "0123456789abcdefABCDEF" for c in s) and len(s) <= 16:
                return int(s, 16)
            return float(s)
        except Exception:
            return 0.0

    X = X.applymap(try_num)
    y = df[TARGET_COL].astype(str)
    return X, y


# ===============================
# TRAIN / SAVE (with GPU attempt + fallback)
# ===============================
def train_and_save(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # try to detect GPU
    use_gpu, gpu_params, gpu_info = try_get_gpu_params_prefer_amd()

    base_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        # verbosity: reduce LightGBM native logs if you want
        verbose=-1
    )

    # merge params if GPU available
    model_params = base_params.copy()
    if use_gpu:
        model_params.update(gpu_params)

    # create classifier
    #clf = LGBMClassifier(**model_params)
    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        device="gpu",             # 👈 enable GPU
        gpu_platform_id=0,        # usually 0
        gpu_device_id=0           # your RX 6500M
    )


    # Try to fit. If GPU isn't enabled in build or a runtime error occurs, catch and retry on CPU.
    try:
        print(f"🔹 Training LightGBM with params: {model_params}")
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            callbacks=[]
        )

    except Exception as e:
        # common failure: LightGBM wasn't built with GPU support or OpenCL runtime issues
        warnings.warn(f"GPU training failed with error: {e}\nFalling back to CPU (device='cpu').")
        # retry with CPU params
        cpu_params = base_params.copy()
        clf = LGBMClassifier(**cpu_params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss")

    preds = clf.predict(X_val)
    print("\n🔹 Validation Classification Report:\n", classification_report(y_val, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Feature list saved to {FEATURES_PATH}")


def create_test_sample(csv_path, out_path, sample_size=TEST_SAMPLE_SIZE):
    df = pd.read_csv(csv_path)
    df_sample = df.groupby(TARGET_COL, group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // df[TARGET_COL].nunique()),
                           random_state=RANDOM_STATE)
    )
    df_sample.to_csv(out_path, index=False)
    print(f"✅ Saved stratified test sample with {len(df_sample)} rows to {out_path}")
    print("Class distribution:", df_sample[TARGET_COL].value_counts().to_dict())


def test_on_csv(model_path, features_path, csv_path):
    clf = joblib.load(model_path)
    with open(features_path, "r") as f:
        expected_features = json.load(f)

    print(f"✅ Loaded model with {len(expected_features)} features.")

    df = pd.read_csv(csv_path)
    X, y_true = preprocess(df)

    # align features
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

    print("\n🔹 Training model (GPU preferred, auto-fallback to CPU)...")
    train_and_save(X, y)

    print("\n🔹 Creating stratified test sample...")
    create_test_sample(DATA_PATH, TEST_PATH, TEST_SAMPLE_SIZE)

    print("\n🔹 Evaluating model on test data...")
    test_on_csv(MODEL_PATH, FEATURES_PATH, TEST_PATH)


if __name__ == "__main__":
    main()
