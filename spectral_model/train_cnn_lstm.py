import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# -----------------------------------
# Paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_path = PROJECT_ROOT / "data/processed/splits/train.csv"
test_path = PROJECT_ROOT / "data/processed/splits/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

features = [f"b{i}" for i in range(1,17)]

X_train = train_df[features].values
X_test = test_df[features].values

y_train = train_df["label"].values
y_test = test_df["label"].values

# -----------------------------------
# Label Encoding
# -----------------------------------

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# -----------------------------------
# Scaling
# -----------------------------------

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# Dataset
# -----------------------------------

class SpectralDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx].unsqueeze(1)  # shape (16,1)
        y = self.y[idx]

        return x, y


train_dataset = SpectralDataset(X_train, y_train)
test_dataset = SpectralDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -----------------------------------
# CNN-LSTM Model
# -----------------------------------

class CNN_LSTM(nn.Module):

    def __init__(self, num_classes):

        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(0,2,1)

        lstm_out, _ = self.lstm(x)

        x = lstm_out[:,-1,:]

        x = self.fc(x)

        return x

# -----------------------------------
# Model Setup
# -----------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM(num_classes=len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

# -----------------------------------
# Training
# -----------------------------------

print("Training CNN-LSTM...")

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

# -----------------------------------
# Evaluation
# -----------------------------------

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        preds = torch.argmax(outputs,1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))