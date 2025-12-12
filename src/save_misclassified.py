# save_misclassified.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os

MODEL_PATH = "D:/CAN_ML_Project/models/can_model_v2.pkl"
DATA_PATH = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
OUT_CSV = "D:/CAN_ML_Project/analysis/misclassified_sample.csv"
CHUNK = 200_000  # adjust

os.makedirs("D:/CAN_ML_Project/analysis", exist_ok=True)
model = joblib.load(MODEL_PATH)
print("Model loaded.")

cols_to_keep = None
col_label = "Label"

mis_rows = []
y_true = []
y_pred = []

for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CHUNK, dtype=str, low_memory=False), 1):
    # quick preprocessing: convert all columns to numeric best-effort (hex handling)
    chunk_proc = chunk.copy()
    for c in chunk_proc.columns:
        if c == col_label: continue
        chunk_proc[c] = chunk_proc[c].astype(str).str.rstrip("Rr ").str.replace("0x","", regex=False)
        chunk_proc[c] = chunk_proc[c].str.replace(r'[^0-9a-fA-F\-\.]', '', regex=True)
        # try hex first, else numeric
        def conv(x):
            if x == '' or x is None: return 0
            try:
                return int(x, 16)
            except:
                try:
                    return float(x)
                except:
                    return 0
        chunk_proc[c] = chunk_proc[c].apply(conv)
    # features = all except label
    features = [c for c in chunk_proc.columns if c != col_label]
    X = chunk_proc[features]
    y = chunk[col_label].astype(str)
    try:
        preds = model.predict(X.values)
    except Exception:
        preds = model.predict(X)
    # select misclassified rows sample (up to 200 per chunk)
    mis_mask = preds != y.values
    mis = chunk.loc[mis_mask]
    if len(mis) > 0:
        mis_rows.append(mis.sample(n=min(200, len(mis)), random_state=42))
    y_true.extend(list(y))
    y_pred.extend(list(preds))
    print(f"Chunk {i}: rows={len(chunk)}, mis={mis.shape[0]}")
    if i >= 10: break  # limit so this script runs quickly; remove to scan full file

if mis_rows:
    pd.concat(mis_rows).to_csv(OUT_CSV, index=False)
    print("Saved misclassified sample to:", OUT_CSV)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
