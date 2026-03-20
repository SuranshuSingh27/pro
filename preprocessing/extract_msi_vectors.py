import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from load_msi import load_msi_cube

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCEL = PROJECT_ROOT / "data/raw/MODID_DESCRIPTOR.xlsx"
MSI_DIR = PROJECT_ROOT / "data/raw/msi"
MASK_DIR = PROJECT_ROOT / "data/raw/masks"
OUT_DIR = PROJECT_ROOT / "data/processed/msi_vectors"

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(EXCEL, sheet_name="Image ID")

vectors = []

for _, r in df.iterrows():
    img_id = int(r["Image ID"])
    pid = r["Patient ID"]
    label = r.get("Diagnosis", None)

    hdr = MSI_DIR / f"{img_id}.hdr"
    mask = MASK_DIR / f"{img_id}.png"

    if not hdr.exists() or not mask.exists():
        continue

    msi = load_msi_cube(hdr)           # (H, W, 16)
    lesion_mask = cv2.imread(str(mask), 0)
    lesion_mask = lesion_mask > 0

    band_means = []
    for b in range(msi.shape[-1]):
        band = msi[:, :, b]
        band_means.append(band[lesion_mask].mean())

    vectors.append({
        "image_id": img_id,
        "patient_id": pid,
        "label": label,
        **{f"b{i+1}": band_means[i] for i in range(16)}
    })

df_out = pd.DataFrame(vectors)
df_out.to_csv(OUT_DIR / "msi_vectors.csv", index=False)

print("✅ MSI vectors saved:", df_out.shape)