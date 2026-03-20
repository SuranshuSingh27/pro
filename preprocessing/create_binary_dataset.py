import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_path = PROJECT_ROOT / "data/processed/master_dataset.csv"

df = pd.read_csv(data_path)

# convert to binary
df["binary_label"] = df["label"].apply(
    lambda x: "CANCER" if x == "OSCC" else "NON_CANCER"
)

print(df["binary_label"].value_counts())

# save
out_path = PROJECT_ROOT / "data/processed/binary_dataset.csv"
df.to_csv(out_path, index=False)

print("Binary dataset saved.")