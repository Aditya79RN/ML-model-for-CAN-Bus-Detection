# src/preprocess_v2.py
import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.stats import entropy

DATA_DIR = "data"
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names (adjust if needed)
columns = ["Timestamp", "CAN_ID", "DLC", "D1","D2","D3","D4","D5","D6","D7","D8"]

def load_and_label(file_path, label):
    df = pd.read_csv(file_path, names=columns, dtype=str)
    df["Label"] = label
    return df

print("🔹 Loading datasets...")

# Load datasets with labels
dos = load_and_label(os.path.join(DATA_DIR, "DoS_dataset.csv"), "DoS")
fuzzy = load_and_label(os.path.join(DATA_DIR, "Fuzzy_dataset.csv"), "Fuzzy")
gear = load_and_label(os.path.join(DATA_DIR, "gear_dataset.csv"), "Gear")
rpm = load_and_label(os.path.join(DATA_DIR, "RPM_dataset.csv"), "RPM")
normal = load_and_label(os.path.join(DATA_DIR, "normal_run_data.txt"), "Normal")

df = pd.concat([dos, fuzzy, gear, rpm, normal], ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

# Convert numeric fields
for col in ["DLC"] + [f"D{i}" for i in range(1, 9)]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# --- Feature Engineering ---
print("🔹 Extracting features...")

# Frequency of each CAN_ID
freq = df["CAN_ID"].value_counts().to_dict()
df["CAN_ID_freq"] = df["CAN_ID"].map(freq)

# Payload entropy (randomness)
def payload_entropy(row):
    values = [row[f"D{i}"] for i in range(1, 9)]
    counts = Counter(values)
    return entropy(list(counts.values()), base=2)

df["Payload_entropy"] = df.apply(payload_entropy, axis=1)

# Rolling mean & std for first two bytes (example)
df["D1_roll_mean"] = df["D1"].rolling(window=10, min_periods=1).mean()
df["D1_roll_std"] = df["D1"].rolling(window=10, min_periods=1).std().fillna(0)

# Aggregate features per CAN_ID
agg = df.groupby("CAN_ID")[[f"D{i}" for i in range(1, 9)]].agg(["mean", "std", "min", "max"])
agg.columns = ["_".join(col) for col in agg.columns]
agg = agg.reset_index()
df = df.merge(agg, on="CAN_ID", how="left")

# Drop unused columns
df = df.drop(columns=["Timestamp"])

# Save engineered dataset
output_path = os.path.join(OUTPUT_DIR, "ml_ready_v2.csv")
df.to_csv(output_path, index=False)
print(f"✅ ML-ready v2 dataset saved at {output_path}")
