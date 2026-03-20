import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_path = PROJECT_ROOT / "data/processed/binary_dataset.csv"

df = pd.read_csv(data_path)

train, temp = train_test_split(
    df,
    test_size=0.3,
    stratify=df["binary_label"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp["binary_label"],
    random_state=42
)

split_dir = PROJECT_ROOT / "data/processed/binary_splits"
split_dir.mkdir(exist_ok=True)

train.to_csv(split_dir / "train.csv", index=False)
val.to_csv(split_dir / "val.csv", index=False)
test.to_csv(split_dir / "test.csv", index=False)

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))