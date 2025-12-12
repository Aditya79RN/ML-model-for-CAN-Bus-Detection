import pandas as pd

def load_csv_dataset(path, label):
    col_names = ["Timestamp", "CAN_ID", "DLC", "D1","D2","D3","D4","D5","D6","D7","D8","Flag"]
    # read with tab separator
    df = pd.read_csv(path, sep=r"\s+", header=None, names=col_names, engine="python")
    df["Label"] = label
    return df



def load_txt_dataset(file_path, label):
    """
    Loads CAN dataset from TXT (normal_run_data.txt) with format:
    Timestamp: 1479121434.850202 ID: 0350 DLC: 8 05 28 84 ...
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            timestamp = parts[1]
            can_id = parts[3]
            dlc = parts[5]
            data_bytes = parts[6:]
            
            row = [timestamp, can_id, dlc] + data_bytes + [label]
            data.append(row)

    # Convert to DataFrame
    max_len = max(len(row) for row in data)
    col_names = ["Timestamp", "ID", "DLC"] + [f"Data{i}" for i in range(max_len-4)] + ["Label"]

    df = pd.DataFrame(data, columns=col_names)
    return df


if __name__ == "__main__":
    # Load datasets
    dos = load_csv_dataset("DoS_dataset.csv", 1)
    fuzzy = load_csv_dataset("Fuzzy_dataset.csv", 1)
    gear = load_csv_dataset("gear_dataset.csv", 1)
    rpm = load_csv_dataset("RPM_dataset.csv", 1)
    normal = load_txt_dataset("normal_run_data.txt", 0)

    # Merge all
    full_dataset = pd.concat([dos, fuzzy, gear, rpm, normal], ignore_index=True)

    # Save processed dataset
    full_dataset.to_csv("processed_CAN_dataset.csv", index=False)
    print("✅ Dataset preprocessed and saved as processed_CAN_dataset.csv")
