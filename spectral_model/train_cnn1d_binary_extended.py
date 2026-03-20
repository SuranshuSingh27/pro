import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
# Create Spectral Derivatives
# -----------------------------

bands = [f"b{i}" for i in range(1,17)]

for i in range(len(bands)-1):

    d_col = f"d{i+1}"

    train_df[d_col] = train_df[bands[i+1]] - train_df[bands[i]]
    test_df[d_col] = test_df[bands[i+1]] - test_df[bands[i]]

features = bands + [f"d{i}" for i in range(1,16)]

X_train = train_df[features].values
X_test = test_df[features].values

y_train = train_df["binary_label"].values
y_test = test_df["binary_label"].values

# -----------------------------
# Label Encoding
# -----------------------------

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Dataset
# -----------------------------

class SpectralDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx].unsqueeze(0)  # (1,31)
        y = self.y[idx]

        return x, y


train_dataset = SpectralDataset(X_train, y_train)
test_dataset = SpectralDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -----------------------------
# CNN Model
# -----------------------------

class SpectralCNN(nn.Module):

    def __init__(self):

        super(SpectralCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # feature size after conv layers
        self.fc1 = nn.Linear(384, 64)

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x


# -----------------------------
# Model Setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpectralCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 40

# -----------------------------
# Training
# -----------------------------

print("Training Extended Feature CNN...")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

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

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        preds = torch.argmax(outputs, 1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_true, y_pred, target_names=le.classes_))