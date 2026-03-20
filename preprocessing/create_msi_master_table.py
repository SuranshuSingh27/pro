import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

msi_dir = PROJECT_ROOT / "data/raw/msi"
mask_dir = PROJECT_ROOT / "data/raw/masks"

labels_df = pd.read_csv(PROJECT_ROOT / "data/processed/master_dataset.csv")

records = []

print("Creating MSI master dataset...")

hdr_files = sorted(msi_dir.glob("*.hdr"))

for i, hdr_path in enumerate(hdr_files):

    image_id = hdr_path.stem

    mask_path = mask_dir / f"{image_id}.png"

    if not mask_path.exists():
        continue

    if i >= len(labels_df):
        break

    label = labels_df.iloc[i]["label"]

    records.append({
        "image_id": image_id,
        "msi_hdr_path": str(hdr_path),
        "mask_path": str(mask_path),
        "label": label,
        "binary_label": "CANCER" if label == "OSCC" else "NON_CANCER"
    })

df = pd.DataFrame(records)

output_path = PROJECT_ROOT / "data/processed/msi_master_dataset.csv"

df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Total samples:", len(df))