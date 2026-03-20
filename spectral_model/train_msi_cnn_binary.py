import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_ROOT / "data/processed/msi_roi_dataset.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["binary_label"])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Dataset class
class MSIDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cube = np.load(row["msi_roi_path"])

        # (H,W,16) → (16,H,W)
        cube = np.transpose(cube, (2, 0, 1))

        cube = torch.tensor(cube, dtype=torch.float32)

        # resize to fixed size
        cube = F.interpolate(
            cube.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        label = torch.tensor(row["label"], dtype=torch.long)

        return cube, label


train_loader = DataLoader(MSIDataset(train_df), batch_size=8, shuffle=True)
test_loader = DataLoader(MSIDataset(test_df), batch_size=8)

# CNN Model
class MSICNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # adaptive pooling so any image size works
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.gap(x)

        x = torch.flatten(x,1)

        x = self.relu(self.fc1(x))

        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MSICNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 20

print("Training MSI CNN...")

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


# Evaluation
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

print("\nClassification Report:")
print(classification_report(y_true,y_pred))


# Save experiment results
results_file = PROJECT_ROOT / "results_binary.csv"

new_row = pd.DataFrame([{
    "model":"msi_cnn_spatial",
    "accuracy":acc
}])

if results_file.exists():

    old = pd.read_csv(results_file)
    new_row = pd.concat([old,new_row])

new_row.to_csv(results_file,index=False)

print("\nResults saved to results_binary.csv")