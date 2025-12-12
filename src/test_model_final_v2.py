# src/test_model_final_v2.py
"""
Load trained CAN intrusion model and run predictions on new data.
"""

import joblib
import pandas as pd

MODEL_PATH = "D:/CAN_ML_Project/models/can_model_final_v2.pkl"
TARGET_COL = "Label"   # only needed if test CSV includes labels

# ===============================
# LOAD MODEL + FEATURES
# ===============================
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
expected_features = bundle["features"]

print(f"✅ Loaded model with {len(expected_features)} features.")

# ===============================
# PREPROCESSING FUNCTION
# (must match training)
# ===============================
def preprocess(df):
    X = df.copy()

    def try_num(x):
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return 0
        try:
            if any(c in s.lower() for c in "abcdef") or s.lower().startswith("0x"):
                return int(s.replace("0x", ""), 16)
            return float(s)
        except Exception:
            return 0

    X = X.applymap(try_num)

    # keep only expected features, fill missing
    X = X.reindex(columns=expected_features, fill_value=0)
    return X

# ===============================
# TEST ON NEW DATA
# ===============================
def test_on_csv(csv_path):
    df = pd.read_csv(csv_path)

    # if labels exist (for evaluation), keep them
    y_true = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    X_proc = preprocess(X)
    preds = model.predict(X_proc)

    if y_true is not None:
        from sklearn.metrics import classification_report
        print("\n🔹 Classification Report on Test Data:\n",
              classification_report(y_true, preds))
    else:
        print("\n🔹 Predictions (first 20 rows):")
        print(preds[:20])

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    # example: test on a processed CSV
    TEST_PATH = "D:/CAN_ML_Project/processed/test_sample.csv"
    test_on_csv(TEST_PATH)
