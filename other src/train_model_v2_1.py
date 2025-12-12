#Needed ram of 64GB+


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"

print("🔹 Loading ML-ready v2 dataset in chunks...")

# Load dataset in chunks to avoid memory crash
chunks = []
chunk_size = 1000_000   # adjust if needed
for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
    chunks.append(chunk)
    print(f"✅ Loaded chunk with shape {chunk.shape}, total rows so far: {sum(len(c) for c in chunks)}")

df = pd.concat(chunks, ignore_index=True)
print(f"📊 Final dataset shape: {df.shape}, Labels: {df['Label'].value_counts().to_dict()}")

# -----------------------------
# Features and labels
# -----------------------------
X = df.drop("Label", axis=1)
y = df["Label"]

print("🔹 Handling missing values...")
X.fillna(0, inplace=True)

print("🔹 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# -----------------------------
# 🚀 Train RandomForest with class weights
# -----------------------------
print("🔹 Training RandomForest with class weights (no SMOTE)...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model training completed!")

# -----------------------------
# Evaluation
# -----------------------------
print("🔹 Evaluating model...")
y_pred = model.predict(X_test)

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "D:/CAN_ML_Project/processed/can_model_v2.pkl")
print("💾 Model saved at: processed/can_model_v2.pkl")
