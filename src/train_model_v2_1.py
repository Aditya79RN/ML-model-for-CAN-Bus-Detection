import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
import joblib

# ==============================
# CONFIG
# ==============================
dataset_path = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"   # your dataset file
chunksize = 500000                                             # process 500k rows per chunk
model_path = "D:/CAN_ML_Project/models/can_model_v2.pkl"       # where to save model

# ==============================
# STEP 1: Initialize Model
# ==============================
print("🔹 Initializing model...")
model = SGDClassifier(loss="log_loss", max_iter=5)

# ==============================
# STEP 2: Scan dataset once to collect all labels
# ==============================
print("🔹 Scanning dataset for all labels (may take some time)...")
labels = pd.read_csv(dataset_path, usecols=["Label"])
all_classes = labels["Label"].unique()
print(f"✅ Labels found: {all_classes}")

# ==============================
# STEP 3: Train in Chunks
# ==============================
print("🔹 Training model in chunks...")

imputer = SimpleImputer(strategy="most_frequent")  # Replace NaNs with most frequent value

reader = pd.read_csv(dataset_path, chunksize=chunksize)

for i, chunk in enumerate(reader, 1):
    print(f"  📦 Processing chunk {i}...")
    
    # Split features and labels
    X = chunk.drop(columns=["Label"])
    y = chunk["Label"]

    # ✅ Fit imputer on first chunk, then only transform
    if i == 1:
        X = imputer.fit_transform(X)
        model.partial_fit(X, y, classes=all_classes)
    else:
        X = imputer.transform(X)
        model.partial_fit(X, y)    

print("✅ Training complete!")

# ==============================
# STEP 4: Save Model
# ==============================
joblib.dump(model, model_path)
print(f"💾 Model saved to {model_path}")
