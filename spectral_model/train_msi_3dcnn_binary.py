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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(PROJECT_ROOT / "data/processed/msi_roi_dataset.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["binary_label"])

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)


class MSIDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        cube = np.load(row["msi_roi_path"])

        cube = cube / cube.max()

        cube = np.transpose(cube, (2,0,1))

        cube = torch.tensor(cube, dtype=torch.float32)

        cube = F.interpolate(
            cube.unsqueeze(0),
            size=(128,128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        cube = cube.unsqueeze(0)

        label = torch.tensor(row["label"], dtype=torch.long)

        return cube, label


train_loader = DataLoader(MSIDataset(train_df), batch_size=8, shuffle=True)
test_loader = DataLoader(MSIDataset(test_df), batch_size=8)


class MSI3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv3d(1,16,(3,3,3),padding=1)
        self.conv2 = nn.Conv3d(16,32,(3,3,3),padding=1)

        self.pool = nn.MaxPool3d(2)

        self.relu = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Linear(32,2)

    def forward(self,x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.gap(x)

        x = torch.flatten(x,1)

        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MSI3DCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 20

print("Training MSI 3D CNN...")

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

    print(f"Epoch {epoch+1}/{epochs} Loss {total_loss:.4f}")


model.eval()

y_true=[]
y_pred=[]

with torch.no_grad():

    for X,y in test_loader:

        X = X.to(device)

        outputs = model(X)

        preds = torch.argmax(outputs,1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())


acc = accuracy_score(y_true,y_pred)

print("Binary Accuracy:", acc)