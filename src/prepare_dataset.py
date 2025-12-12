import pandas as pd

# Define CAN column names
column_names = [
    "Timestamp", "CAN_ID", "DLC",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
    "Label"
]

def prepare_combined_dataset():
    print("🔄 Loading datasets...")

    # Load datasets
    dos_df = pd.read_csv("DoS_dataset.csv", names=column_names[:-1], dtype=str)
    fuzzy_df = pd.read_csv("Fuzzy_dataset.csv", names=column_names[:-1], dtype=str)
    gear_df = pd.read_csv("gear_dataset.csv", names=column_names[:-1], dtype=str)
    rpm_df = pd.read_csv("RPM_dataset.csv", names=column_names[:-1], dtype=str)
    normal_df = pd.read_csv("normal_run_data.txt", names=column_names[:-1], dtype=str, sep="\t")

    # Add labels
    dos_df["Label"] = "DoS"
    fuzzy_df["Label"] = "Fuzzy"
    gear_df["Label"] = "Gear"
    rpm_df["Label"] = "RPM"
    normal_df["Label"] = "Normal"

    # Combine everything
    combined = pd.concat([dos_df, fuzzy_df, gear_df, rpm_df, normal_df], ignore_index=True)

    # Save
    combined.to_csv("combined_CAN_dataset.csv", index=False)
    print("✅ Combined dataset saved as combined_CAN_dataset.csv")

if __name__ == "__main__":
    prepare_combined_dataset()
