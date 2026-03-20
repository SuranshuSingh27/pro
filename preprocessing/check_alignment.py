import os
from pathlib import Path
import pandas as pd

# -----------------------------
# Resolve project root safely
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXCEL = DATA_DIR / "MODID_DESCRIPTOR.xlsx"

RGB_DIR = DATA_DIR / "rgb"
MASK_DIR = DATA_DIR / "masks"
MSI_DIR = DATA_DIR / "msi"

print("📁 Project root :", PROJECT_ROOT)
print("📄 Excel file   :", EXCEL)

# -----------------------------
# Load Excel
# -----------------------------
df = pd.read_excel(EXCEL, sheet_name="Image ID")

missing = []

for _, r in df.iterrows():
    img_id = int(r["Image ID"])
    pid = str(r["Patient ID"])
    num = int(r["Image Number"])

    rgb_path = RGB_DIR / f"{pid}_{num}.png"
    mask_path = MASK_DIR / f"{pid}_{num}.png"
    msi_path = MSI_DIR / f"{img_id}.hdr"

    if not rgb_path.exists():
        missing.append(f"RGB  → {rgb_path.name}")
    if not mask_path.exists():
        missing.append(f"MASK → {mask_path.name}")
    if not msi_path.exists():
        missing.append(f"MSI  → {msi_path.name}")

# -----------------------------
# Report
# -----------------------------
if missing:
    print("❌ Missing files:")
    for m in missing:
        print("   ", m)
else:
    print("✅ All data aligned correctly")