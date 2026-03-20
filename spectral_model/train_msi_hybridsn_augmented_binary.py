import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(PROJECT_ROOT / "data/processed/msi_roi_dataset.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["binary_label"])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# -----------------------
# Data Augmentation
# -----------------------

def augment(cube):

    # horizontal flip
    if random.random() < 0.5:
        cube = torch.flip(cube, dims=[2])

    # vertical flip
    if random.random() < 0.5:
        cube = torch.flip(cube, dims=[1])

    # 90 degree rotation
    if random.random() < 0.5:
        cube = torch.rot90(cube, 1, dims=[1,2])

    return cube


# -----------------------
# Dataset
# -----------------------

class MSIDataset(Dataset):

    def __init__(self, dataframe, train=True):
        self.df = dataframe
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        cube = np.load(row["msi_roi_path"])

        cube = cube / cube.max()

        # crop lesion region
        mask = cube.sum(axis=2) > 0
        coords = np.argwhere(mask)

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        cube = cube[y0:y1, x0:x1]

        cube = np.transpose(cube, (2,0,1))

        cube = torch.tensor(cube, dtype=torch.float32)

        cube = F.interpolate(
            cube.unsqueeze(0),
            size=(128,128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # augmentation only for training
        if self.train:
            cube = augment(cube)

        cube = cube.unsqueeze(0)

        label = torch.tensor(row["label"], dtype=torch.long)

        return cube, label


train_loader = DataLoader(MSIDataset(train_df, train=True), batch_size=8, shuffle=True)
test_loader = DataLoader(MSIDataset(test_df, train=False), batch_size=8)

# -----------------------
# HybridSN Model
# -----------------------

class HybridSN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(3,3,3), padding=1)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(3,3,3), padding=1)

        self.pool3d = nn.MaxPool3d((1,2,2))

        self.conv2d_1 = nn.Conv2d(16*16, 64, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool2d = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,2)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.relu(self.conv3d_1(x))
        x = self.pool3d(self.relu(self.conv3d_2(x)))

        b,c,d,h,w = x.shape
        x = x.view(b, c*d, h, w)

        x = self.pool2d(self.relu(self.conv2d_1(x)))
        x = self.pool2d(self.relu(self.conv2d_2(x)))

        x = self.gap(x)

        x = torch.flatten(x,1)

        x = self.relu(self.fc1(x))

        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridSN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 30

print("Training HybridSN with augmentation...")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X,y in train_loader:

        X,y = X.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs,y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# -----------------------
# Evaluation
# -----------------------

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for X,y in test_loader:

        X = X.to(device)

        outputs = model(X)

        preds = torch.argmax(outputs,1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())

acc = accuracy_score(y_true,y_pred)

print("\nBinary Accuracy:", acc)