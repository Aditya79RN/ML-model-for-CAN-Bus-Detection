#sample for testing test_model_final_v2.py
#takes few samples from the all lables available in the dataset

import pandas as pd

INPUT_FILE = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
OUTPUT_FILE = "D:/CAN_ML_Project/processed/test_sample.csv"

SAMPLE_SIZE = 5000   # total rows you want in test sample
TARGET_COL = "Label"

def create_stratified_test_sample():
    # Load a manageable chunk (or the whole file if memory allows)
    df = pd.read_csv(INPUT_FILE)

    # Calculate per-class quota
    labels = df[TARGET_COL].unique()
    per_class = SAMPLE_SIZE // len(labels)

    samples = []
    for lbl in labels:
        subset = df[df[TARGET_COL] == lbl]
        take_n = min(per_class, len(subset))  # handle small classes
        samples.append(subset.sample(n=take_n, random_state=42))

    df_sample = pd.concat(samples, ignore_index=True)
    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved stratified test sample with {len(df_sample)} rows to {OUTPUT_FILE}")
    print("Class distribution:", df_sample[TARGET_COL].value_counts().to_dict())

if __name__ == "__main__":
    create_stratified_test_sample()
