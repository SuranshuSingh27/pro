import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Project root
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCEL = PROJECT_ROOT / "data/raw/MODID_DESCRIPTOR.xlsx"
MSI_CSV = PROJECT_ROOT / "data/processed/msi_vectors/msi_vectors.csv"
RGB_ROI_DIR = PROJECT_ROOT / "data/processed/rgb_roi"
OUT_PATH = PROJECT_ROOT / "data/processed/master_dataset.csv"

print("Loading files...")

# -------------------------------------------------
# Load MSI vectors
# -------------------------------------------------
msi_df = pd.read_csv(MSI_CSV)

# -------------------------------------------------
# Load Image ID sheet
# -------------------------------------------------
image_df = pd.read_excel(EXCEL, sheet_name="Image ID")

# -------------------------------------------------
# Load Patient sheet (header fix for your file)
# -------------------------------------------------
patient_df = pd.read_excel(
    EXCEL,
    sheet_name="Patient data",
    header=1   # <-- This worked in your case
)

# -------------------------------------------------
# Clean column names (remove hidden spaces)
# -------------------------------------------------
image_df.columns = image_df.columns.str.strip()
patient_df.columns = patient_df.columns.str.strip()
msi_df.columns = msi_df.columns.str.strip()

print("Image sheet columns:", image_df.columns.tolist())
print("Patient sheet columns:", patient_df.columns.tolist())

# -------------------------------------------------
# Merge Image ID -> Patient ID
# -------------------------------------------------
merged = msi_df.merge(
    image_df[["Image ID", "Patient ID"]],
    left_on="image_id",
    right_on="Image ID",
    how="inner"
)

# -------------------------------------------------
# Merge Patient ID -> Diagnosis
# -------------------------------------------------
merged = merged.merge(
    patient_df[["Patient ID", "Diagnosis"]],
    on="Patient ID",
    how="left"
)

# -------------------------------------------------
# Add RGB ROI path
# -------------------------------------------------
merged["rgb_path"] = merged["image_id"].apply(
    lambda x: str(RGB_ROI_DIR / f"{int(x)}.png")
)

# -------------------------------------------------
# Keep only rows where RGB file exists
# -------------------------------------------------
merged = merged[merged["rgb_path"].apply(lambda x: Path(x).exists())]

# -------------------------------------------------
# Select final columns
# -------------------------------------------------
final_df = merged[
    ["image_id", "rgb_path", "Diagnosis"] +
    [f"b{i}" for i in range(1, 17)]
].copy()

# Rename Diagnosis -> label
final_df = final_df.rename(columns={"Diagnosis": "label"})

# -------------------------------------------------
# CLEAN LABELS (VERY IMPORTANT)
# -------------------------------------------------
final_df["label"] = final_df["label"].astype(str)
final_df["label"] = final_df["label"].str.strip()     # remove spaces
final_df["label"] = final_df["label"].str.upper()     # consistent case

# Optional: unify synonyms
label_map = {
    "NORMAL": "HEALTHY",
    "HEALTHY": "HEALTHY",
    "LEUKOPLAKIA": "LEUKOPLAKIA",
    "KERATOSIS": "KERATOSIS",
    "OSMF": "OSMF",
    "OSCC": "OSCC"
}

final_df["label"] = final_df["label"].replace(label_map)

# -------------------------------------------------
# Save master dataset
# -------------------------------------------------
final_df.to_csv(OUT_PATH, index=False)

print("\n✅ Master dataset created successfully")
print("Shape:", final_df.shape)
print("\nClean Label distribution:")
print(final_df["label"].value_counts())