import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

msi_dir = PROJECT_ROOT / "data/raw/msi"
mask_dir = PROJECT_ROOT / "data/raw/masks"

master_df = pd.read_csv(PROJECT_ROOT / "data/processed/master_dataset.csv")

records = []

print("Creating MSI dataset from master_dataset...")

for _, row in master_df.iterrows():

    rgb_path = Path(row["rgb_path"])
    image_id = rgb_path.stem

    hdr_path = msi_dir / f"{image_id}.hdr"
    mask_path = mask_dir / f"{image_id}.png"

    if not hdr_path.exists():
        print("Missing MSI:", image_id)
        continue

    if not mask_path.exists():
        print("Missing Mask:", image_id)
        continue

    records.append({
        "image_id": image_id,
        "msi_hdr_path": str(hdr_path),
        "mask_path": str(mask_path),
        "label": row["label"],
        "binary_label": "CANCER" if row["label"] == "OSCC" else "NON_CANCER"
    })

df = pd.DataFrame(records)

save_path = PROJECT_ROOT / "data/processed/msi_master_dataset.csv"

df.to_csv(save_path, index=False)

print("Saved:", save_path)
print("Total samples:", len(df))