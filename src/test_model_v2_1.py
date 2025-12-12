# src/test_model_final_v2.py
"""
Robust memory-safe test script that:
- reads the model and asks it which feature names it expects (if available)
- maps test CSV columns to those expected features (fuzzy mapping)
- pads missing features with zeros (so dimensions match)
- safely converts hex and messy string payloads to numeric
- processes data in chunks to avoid memory exhaustion
- prints final metrics
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
    """Convert value to number:
       - strip and drop trailing 'R' or spaces
       - remove non-hex chars then try int(base=16)
       - fallback to float conversion
       - return np.nan if cannot parse
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan
    # remove trailing 'R' or 'r' and whitespace
    s = s.rstrip("Rr ").replace("0x", "")
    # remove separators
    s_clean = re.sub(r'[^0-9a-fA-F]', '', s)
    if s_clean == "":
        # try decimal
        try:
            return float(s)
        except Exception:
            return np.nan
    # try hex
    try:
        return int(s_clean, 16)
    except Exception:
        try:
            return float(s_clean)
        except Exception:
            return np.nan

def find_best_match(expected_name, available_cols):
    """Try a set of heuristics to map expected_name -> a column in available_cols.
       Returns matched column name or None.
    """
    if expected_name in available_cols:
        return expected_name

    low_map = {c.lower(): c for c in available_cols}
    en = expected_name.lower()

    # exact lower-case match
    if en in low_map:
        return low_map[en]

    # ignore underscores/dots/hyphens and compare
    def normalize(s):
        return re.sub(r'[^0-9a-z]', '', s.lower())

    norm_expected = normalize(expected_name)
    norm_map = {normalize(c): c for c in available_cols}
    if norm_expected in norm_map:
        return norm_map[norm_expected]

    # try 'contains' match
    for c in available_cols:
        if norm_expected in normalize(c) or normalize(c) in norm_expected:
            return c

    # try close matches using difflib
    candidates = get_close_matches(expected_name, available_cols, n=1, cutoff=0.7)
    if candidates:
        return candidates[0]

    # some special heuristics:
    # expected D1_mean vs D1.mean etc.
    for suffix in ["mean","std","min","max","roll_mean","roll_std","freq","entropy"]:
        if suffix in en:
            for c in available_cols:
                if suffix in c.lower() and expected_name.split('_')[0].lower() in c.lower():
                    return c

    # try prefix variations (Data0 vs D1)
    # e.g., expected "Data0" -> search for D1 or D0 or D1_mean etc.
    # fallback None
    return None

def main():
    print("🔹 Loading model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Get expected feature names if available
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        print("ℹ️ Model.feature_names_in_ found; expected features count =", len(expected_features))
    else:
        # fallback: use n_features_in_, but we need names -> will attempt heuristic later
        if hasattr(model, "n_features_in_"):
            nfeat = int(model.n_features_in_)
            expected_features = None
            print("ℹ️ Model reports n_features_in_ =", nfeat, " (no names available).")
        else:
            raise RuntimeError("Model does not expose feature names or n_features_in_. Cannot align features.")

    # prepare chunked reader (read as strings to avoid parser dtype issues)
    print(f"🔹 Reading test CSV in chunks from: {DATA_PATH} (chunksize={CHUNK_SIZE})")
    reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, dtype=str, low_memory=False)

    y_true_all = []
    y_pred_all = []
    processed_rows = 0
    chunk_idx = 0
    # If expected_features is None, we will pick the first chunk's numeric-like columns later
    feature_cols = None
    inferred_expected_count = None
    if expected_features is None:
        inferred_expected_count = nfeat

    for chunk in reader:
        chunk_idx += 1
        print(f"  📦 Chunk {chunk_idx}: rows={len(chunk)}")
        # Ensure label exists
        if LABEL_COL not in chunk.columns:
            lab_candidates = [c for c in chunk.columns if c.lower() == "label"]
            if lab_candidates:
                chunk.rename(columns={lab_candidates[0]: LABEL_COL}, inplace=True)
            else:
                raise ValueError(f"Label column '{LABEL_COL}' not found in chunk {chunk_idx}. Columns: {list(chunk.columns)[:20]}")

        # On first chunk determine feature column names to use
        if feature_cols is None:
            available_cols = list(chunk.columns)
            # remove label from candidates
            available_cols_nolabel = [c for c in available_cols if c != LABEL_COL]

            if expected_features is not None:
                # try to map each expected feature to a column present (with heuristics)
                mapped = []
                missing = []
                for ef in expected_features:
                    m = find_best_match(ef, available_cols_nolabel)
                    if m is None:
                        missing.append(ef)
                        mapped.append(None)
                    else:
                        mapped.append(m)
                # If some are missing, we will pad them later; but proceed with mapped list
                feature_cols = mapped
                print(f"  ℹ️ Mapped features: {sum(1 for m in mapped if m)} found / {len(mapped)} expected.")
                if missing:
                    print("  ⚠️ Missing features (will be zero-padded):", missing)
            else:
                # no expected feature names: choose first numeric-like columns up to nfeat
                # heuristic: prefer D1..D8, D1_mean..., CAN_ID_freq, Payload_entropy etc.
                # We'll use earlier chooser but ensure we return exactly nfeat columns
                cols = choose_candidates = [c for c in available_cols_nolabel]
                # prefer patterns
                prefer_patterns = ["d1","d2","d3","d4","d5","d6","d7","d8",
                                   "data","dlc","can_id","payload","entropy","freq","mean","std"]
                preferred = []
                for p in prefer_patterns:
                    for c in cols:
                        if p in c.lower() and c not in preferred:
                            preferred.append(c)
                for c in cols:
                    if c not in preferred:
                        preferred.append(c)
                feature_cols = preferred[:inferred_expected_count]
                print(f"  ℹ️ No model feature names available — inferred feature columns (count={len(feature_cols)}):")
                print("     ", feature_cols)

        # Now build X_chunk with columns corresponding to expected_features order
        # feature_cols currently is a list of mapped column names or None entries (if expected_features given)
        if expected_features is not None:
            # for ordered expected_features, build DataFrame columns in same order
            X_chunk = pd.DataFrame(index=chunk.index)
            mapped_list = feature_cols  # same length as expected_features
            for idx, ef in enumerate(expected_features):
                mapped_col = mapped_list[idx]
                if mapped_col is None:
                    # create zero column
                    X_chunk[ef] = 0.0
                else:
                    # copy raw string column
                    X_chunk[ef] = chunk[mapped_col].astype(str).map(safe_to_num)
        else:
            # feature_cols contains chosen column names; use them directly
            X_chunk = chunk[feature_cols].copy()
            # convert to numeric safely
            for col in feature_cols:
                X_chunk[col] = X_chunk[col].map(safe_to_num)

        # drop rows where all features are NaN (cannot predict)
        X_chunk = X_chunk.dropna(how="all")
        if X_chunk.shape[0] == 0:
            print(f"  ⚠️ After cleaning, chunk {chunk_idx} has 0 usable rows — skipping.")
            del chunk
            gc.collect()
            continue

        # Align label values
        y_chunk = chunk.loc[X_chunk.index, LABEL_COL].astype(str).tolist()

        # Ensure correct shape: columns should equal expected feature count
        if expected_features is not None:
            final_expected = len(expected_features)
            if X_chunk.shape[1] != final_expected:
                # if extra columns, trim; if fewer, pad with zeros
                if X_chunk.shape[1] > final_expected:
                    X_chunk = X_chunk.iloc[:, :final_expected]
                else:
                    for i in range(X_chunk.shape[1], final_expected):
                        X_chunk[f"_pad_{i}"] = 0.0
        else:
            # if model only had n_features_in_, ensure X_chunk has that many columns
            if X_chunk.shape[1] != inferred_expected_count:
                if X_chunk.shape[1] > inferred_expected_count:
                    X_chunk = X_chunk.iloc[:, :inferred_expected_count]
                else:
                    for i in range(X_chunk.shape[1], inferred_expected_count):
                        X_chunk[f"_pad_{i}"] = 0.0

        # Final fill NaN and cast to float32
        X_chunk = X_chunk.fillna(0.0).astype(np.float32)

        # Predict (use values to avoid name mismatch warnings)
        try:
            preds = model.predict(X_chunk.values)
        except Exception:
            preds = model.predict(X_chunk)

        # collect
        y_true_all.extend(y_chunk)
        y_pred_all.extend(list(preds))
        processed_rows += X_chunk.shape[0]

        # cleanup
        del chunk, X_chunk, y_chunk, preds
        gc.collect()

        if processed_rows % (CHUNK_SIZE * 5) == 0:
            print(f"  ℹ️ Processed rows so far: {processed_rows}")

    # Final metrics
    print("\n✅ Done processing. Total predicted rows:", processed_rows)
    if len(y_true_all) == 0:
        print("⚠️ No rows predicted — check CSV and columns.")
        return

    print("🔹 Accuracy:", accuracy_score(y_true_all, y_pred_all))
    print("\n🔹 Classification report:\n")
    print(classification_report(y_true_all, y_pred_all))
    print("\n🔹 Confusion matrix:\n")
    print(confusion_matrix(y_true_all, y_pred_all))

if __name__ == "__main__":
    main()
