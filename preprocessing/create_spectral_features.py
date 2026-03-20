import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_path = PROJECT_ROOT / "data/processed/master_dataset.csv"

df = pd.read_csv(data_path)

# original spectral bands
bands = [f"b{i}" for i in range(1, 17)]

# compute first derivative features
for i in range(len(bands) - 1):
    new_col = f"d{i + 1}"

    df[new_col] = df[bands[i + 1]] - df[bands[i]]

# save dataset
out_path = PROJECT_ROOT / "data/processed/msi_features_extended.csv"

df.to_csv(out_path, index=False)

print("Extended spectral features saved.")