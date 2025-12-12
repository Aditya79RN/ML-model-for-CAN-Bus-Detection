import pandas as pd

def load_csv_dataset(path, label):
    """
    Loads CAN dataset from CSV with tab/space-separated values.
    Columns: Timestamp, CAN_ID, DLC, D1...D8, Flag
    """
    col_names = ["Timestamp", "CAN_ID", "DLC",
                 "D1","D2","D3","D4","D5","D6","D7","D8",
                 "Flag"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=col_names, engine="python")
    df["Label"] = label
    return df


def load_txt_dataset(file_path, label):
    """
    Loads CAN dataset from TXT (normal_run_data.txt) with format like:
    Timestamp: 1479121434.850202 ID: 0350 DLC: 8 05 28 84 ...
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            # Example format: Timestamp: <time> ID: <id> DLC: <dlc> <data...>
            timestamp = parts[1]
            can_id = parts[3]
            dlc = parts[5]
            data_bytes = parts[6:]

            row = [timestamp, can_id, dlc] + data_bytes + [label]
            data.append(row)

    # Pad rows so all have same length
    max_len = max(len(row) for row in data)
    for row in data:
        while len(row) < max_len:
            row.insert(-1, "00")  # fill missing bytes with "00"

    # Build column names dynamically
    col_names = ["Timestamp", "CAN_ID", "DLC"] + [f"D{i}" for i in range(1, max_len-3)] + ["Label"]

    df = pd.DataFrame(data, columns=col_names)
    return df


if __name__ == "__main__":
    print("🔄 Loading datasets...")

    # Attack datasets
    dos = load_csv_dataset("DoS_dataset.csv", 1)
    fuzzy = load_csv_dataset("Fuzzy_dataset.csv", 2)
    gear = load_csv_dataset("gear_dataset.csv", 3)
    rpm = load_csv_dataset("RPM_dataset.csv", 4)

    # Normal dataset
    normal = load_txt_dataset("normal_run_data.txt", 0)

    # Merge all datasets
    full_dataset = pd.concat([normal, dos, fuzzy, gear, rpm], ignore_index=True)

    # Save final processed dataset
    full_dataset.to_csv("processed_CAN_dataset.csv", index=False)
    print("✅ Dataset preprocessed and saved as processed_CAN_dataset.csv")
