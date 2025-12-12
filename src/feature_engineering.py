import pandas as pd

def preprocess_for_ml(input_csv, output_csv):
    # Load processed dataset with safe options
    df = pd.read_csv(input_csv, low_memory=False)

    # Drop Timestamp if not needed
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Convert CAN_ID safely (hex → int, else 0)
    if "CAN_ID" in df.columns:
        def safe_hex_to_int(x):
            try:
                return int(str(x), 16)
            except:
                return 0
        df["CAN_ID"] = df["CAN_ID"].apply(safe_hex_to_int)

    # Convert Data bytes (D1...D8) safely (hex → int)
    for col in df.columns:
        if col.startswith("D"):
            def safe_byte_to_int(x):
                try:
                    return int(str(x), 16)
                except:
                    return 0
            df[col] = df[col].apply(safe_byte_to_int)

    # Handle Flag (R → 1, others → 0)
    if "Flag" in df.columns:
        df["Flag"] = df["Flag"].apply(lambda x: 1 if str(x).upper() == "R" else 0)

    # Save ML-ready dataset
    df.to_csv(output_csv, index=False)
    print(f"✅ ML-ready dataset saved as {output_csv}")


if __name__ == "__main__":
    preprocess_for_ml("processed_CAN_dataset.csv", "ml_ready_CAN_dataset.csv")
