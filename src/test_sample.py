#sample for testing test_model_final_v2.py
#only creates first available data

import pandas as pd

INPUT_FILE = "D:/CAN_ML_Project/processed/ml_ready_v2.csv"
OUTPUT_FILE = "D:/CAN_ML_Project/processed/test_sample.csv"

CHUNK_SIZE = 100_000   # process 100k rows at a time
SAMPLE_SIZE = 5000     # total rows you want in test sample

def create_test_sample():
    sample_chunks = []
    total_collected = 0

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        # take a small random sample from this chunk
        frac = (SAMPLE_SIZE - total_collected) / len(chunk)
        if frac > 0:
            frac = min(0.1, frac)  # max 10% from each chunk
            sampled = chunk.sample(frac=frac, random_state=42)
            sample_chunks.append(sampled)
            total_collected += len(sampled)

        if total_collected >= SAMPLE_SIZE:
            break

    df_sample = pd.concat(sample_chunks, ignore_index=True)
    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved test sample with {len(df_sample)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_test_sample()
