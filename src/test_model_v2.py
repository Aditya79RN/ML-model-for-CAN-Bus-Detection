# test_model_v2.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse

# -------------------------------
# Load the trained model
# -------------------------------
print("🔹 Loading trained model...")
model = joblib.load("D:/CAN_ML_Project/processed/can_model_v2.pkl")

# -------------------------------
# Argument Parser (2 modes)
# -------------------------------
parser = argparse.ArgumentParser(description="Test CAN ML Model")
parser.add_argument("--mode", choices=["eval", "single"], default="eval",
                    help="Run full dataset evaluation or test a single frame")
args = parser.parse_args()

# -------------------------------
# Mode 1: Full dataset evaluation
# -------------------------------
if args.mode == "eval":
    print("🔹 Loading test dataset in chunks...")
    dtype_map = {
        "Timestamp": "str",
        "ID": "str",
        "DLC": "int8",
        "Data0": "int8", "Data1": "int8", "Data2": "int8", "Data3": "int8",
        "Data4": "int8", "Data5": "int8", "Data6": "int8", "Data7": "int8",
        "Label": "category",
    }

    chunk_size = 100000
    csv_file = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"

    y_true_all, y_pred_all = [], []

    start = time.time()
    for chunk in pd.read_csv(csv_file, dtype=dtype_map, chunksize=chunk_size):
        X_test = chunk.drop(columns=["Label"])
        y_test = chunk["Label"]

        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    end = time.time()

    print("✅ Model evaluation completed!\n")
    print("🔹 Accuracy:", accuracy_score(y_true_all, y_pred_all))
    print("\n🔹 Detailed Report:")
    print(classification_report(y_true_all, y_pred_all))

    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print(f"⏱️ Total time: {end-start:.2f} sec")

# -------------------------------
# Mode 2: Single manual frame test
# -------------------------------
elif args.mode == "single":
    print("\n🔹 Testing a manual CAN frame...")

    # Example frame (replace with real values)
    sample_frame = pd.DataFrame([{
        "Timestamp": "1478198376",
        "ID": "018f",
        "DLC": 8,
        "Data0": 254,
        "Data1": 91,
        "Data2": 0,
        "Data3": 0,
        "Data4": 0,
        "Data5": 60,
        "Data6": 0,
        "Data7": 0,
    }])

    print("Sample input:", sample_frame.to_dict(orient="records")[0])
    pred = model.predict(sample_frame)
    print("🔮 Predicted Label:", pred[0])
