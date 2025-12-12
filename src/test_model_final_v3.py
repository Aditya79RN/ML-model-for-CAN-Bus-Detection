# src/test_model_final_v3.py
"""
Robust memory-safe test script (fixed version):
- Handles missing Timestamp column expected by the model
- Ignores engineered features not used in training
- Maps features safely using heuristics
- Processes in chunks
- Prints metrics
"""

import re
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from difflib import get_close_matches

# ---------- CONFIG ----------
MODEL_PATH = "D:/CAN_ML_Project/models/can_model_v2.pkl"
DATA_PATH = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
CHUNK_SIZE = 100_000
LABEL_COL = "Label"
# ----------------------------

HEX_CLEAN_RE = re.compile(r'[^0-9a-fA-F]')

def safe_to_num(val):
    """Convert messy CAN values into numbers."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan
    s = s.rstrip("Rr ").replace("0x", "")
    s_clean = re.sub(r'[^0-9a-fA-F]', '', s)
    if s_clean == "":
        try:
            return float(s)
        except Exception:
            return np.nan
    try:
        return int(s_clean, 16)
    except Exception:
        try:
            return float(s_clean)
        except Exception:
            return np.nan

def find_best_match(expected_name, available_cols):
    """Heuristic feature name mapping."""
    if expected_name in available_cols:
        return expected_name

    low_map = {c.lower(): c for c in available_cols}
    en = expected_name.lower()

    if en in low_map:
        return low_map[en]

    def normalize(s):
        return re.sub(r'[^0-9a-z]', '', s.lower())

    norm_expected = normalize(expected_name)
    norm_map = {normalize(c): c for c in available_cols}
    if norm_expected in norm_map:
        return norm_map[norm_expected]

    candidates = get_close_matches(expected_name, available_cols, n=1, cutoff=0.7)
    if candidates:
        return candidates[0]

    return None

def main():
    print("🔹 Loading model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        print("ℹ️ Model expects features:", expected_features)
    elif hasattr(model, "n_features_in_"):
        expected_features = None
        nfeat = int(model.n_features_in_)
        print("ℹ️ Model reports n_features_in_ =", nfeat)
    else:
        raise RuntimeError("Model does not expose feature names or n_features_in_.")

    print(f"🔹 Reading test CSV in chunks: {DATA_PATH} (chunksize={CHUNK_SIZE})")
    reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, dtype=str, low_memory=False)

    y_true_all, y_pred_all = [], []
    processed_rows, chunk_idx = 0, 0
    feature_cols = None

    for chunk in reader:
        chunk_idx += 1
        print(f"  📦 Chunk {chunk_idx}: rows={len(chunk)}")

        if LABEL_COL not in chunk.columns:
            raise ValueError(f"Label column '{LABEL_COL}' missing in chunk {chunk_idx}.")

        if feature_cols is None:
            available_cols = list(chunk.columns)
            available_cols_nolabel = [c for c in available_cols if c != LABEL_COL]

            if expected_features is not None:
                mapped, missing = [], []
                for ef in expected_features:
                    m = find_best_match(ef, available_cols_nolabel)
                    if m is None:
                        missing.append(ef)
                        mapped.append(None)
                    else:
                        mapped.append(m)
                feature_cols = mapped
                print(f"  ℹ️ Features mapped: {sum(1 for m in mapped if m)} / {len(mapped)}")
                if missing:
                    print("  ⚠️ Missing (will be zero-padded):", missing)
            else:
                feature_cols = available_cols_nolabel[:nfeat]
                print(f"  ℹ️ Using first {nfeat} columns:", feature_cols)

        if expected_features is not None:
            X_chunk = pd.DataFrame(index=chunk.index)
            for idx, ef in enumerate(expected_features):
                mapped_col = feature_cols[idx]
                if mapped_col is None:
                    X_chunk[ef] = 0.0
                else:
                    X_chunk[ef] = chunk[mapped_col].astype(str).map(safe_to_num)
        else:
            X_chunk = chunk[feature_cols].applymap(safe_to_num)

        X_chunk = X_chunk.fillna(0.0).astype(np.float32)
        y_chunk = chunk.loc[X_chunk.index, LABEL_COL].astype(str).tolist()

        try:
            preds = model.predict(X_chunk.values)
        except Exception:
            preds = model.predict(X_chunk)

        y_true_all.extend(y_chunk)
        y_pred_all.extend(list(preds))
        processed_rows += X_chunk.shape[0]

        del chunk, X_chunk, y_chunk, preds
        gc.collect()

        if processed_rows % (CHUNK_SIZE * 5) == 0:
            print(f"  ℹ️ Processed rows: {processed_rows}")

    print("\n✅ Done. Total rows:", processed_rows)
    print("🔹 Accuracy:", accuracy_score(y_true_all, y_pred_all))
    print("\n🔹 Classification report:\n", classification_report(y_true_all, y_pred_all))
    print("\n🔹 Confusion matrix:\n", confusion_matrix(y_true_all, y_pred_all))

if __name__ == "__main__":
    main()
