import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_path = PROJECT_ROOT / "data/processed/binary_splits/train.csv"
test_path = PROJECT_ROOT / "data/processed/binary_splits/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# -----------------------------
# Encode Labels
# -----------------------------

le = LabelEncoder()

train_df["label_encoded"] = le.fit_transform(train_df["binary_label"])
test_df["label_encoded"] = le.transform(test_df["binary_label"])

# -----------------------------
# ResNet Weights + Transforms
# -----------------------------

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# -----------------------------
# Dataset
# -----------------------------

class RGBDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = row["rgb_path"]

        image = Image.open(img_path).convert("RGB")

        image = transform(image)

        label = row["label_encoded"]

        return image, label


train_dataset = RGBDataset(train_df)
test_dataset = RGBDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# -----------------------------
# ResNet18 Model
# -----------------------------

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 20

# -----------------------------
# Training
# -----------------------------

print("Training ResNet18 (Pretrained)...")

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

print("Training finished")

# -----------------------------
# Evaluation
# -----------------------------

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs,1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_true, y_pred, target_names=le.classes_))