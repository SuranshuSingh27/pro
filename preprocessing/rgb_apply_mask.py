import cv2
from pathlib import Path

# ✅ project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RGB_DIR = PROJECT_ROOT / "data/raw/rgb"
MASK_DIR = PROJECT_ROOT / "data/raw/masks"
OUT_DIR = PROJECT_ROOT / "data/processed/rgb_roi"

OUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0

for rgb_path in RGB_DIR.glob("*.png"):
    mask_path = MASK_DIR / rgb_path.name
    if not mask_path.exists():
        continue

    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), 0)
    mask = mask > 0

    roi = rgb.copy()
    roi[~mask] = 0

    cv2.imwrite(str(OUT_DIR / rgb_path.name), roi)
    count += 1

print(f"✅ RGB ROI images saved: {count}")
print(f"📁 Saved to: {OUT_DIR}")