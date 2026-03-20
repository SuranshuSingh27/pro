import spectral
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(PROJECT_ROOT / "data/processed/msi_master_dataset.csv")

output_dir = PROJECT_ROOT / "data/processed/msi_roi"
output_dir.mkdir(exist_ok=True)

records = []

print("Applying masks to MSI cubes...")

for _, row in df.iterrows():

    img = spectral.open_image(row["msi_hdr_path"])
    cube = np.array(img.load())

    mask = np.array(Image.open(row["mask_path"]).convert("L"))
    mask = mask > 0

    for b in range(cube.shape[2]):
        band = cube[:,:,b]
        band[~mask] = 0
        cube[:,:,b] = band

    save_path = output_dir / f"{row['image_id']}.npy"

    np.save(save_path, cube)

    records.append({
        "image_id": row["image_id"],
        "msi_roi_path": str(save_path),
        "binary_label": row["binary_label"]
    })

out_df = pd.DataFrame(records)

out_df.to_csv(PROJECT_ROOT / "data/processed/msi_roi_dataset.csv", index=False)

print("Masked MSI dataset saved")