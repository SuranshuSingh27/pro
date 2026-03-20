import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
# Image Transforms
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# Dataset
# -----------------------------

class RGBDataset(Dataset):

    def __init__(self, df, transform):

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = row["rgb_path"]

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        label = row["label_encoded"]

        return image, label


train_dataset = RGBDataset(train_df, train_transform)
test_dataset = RGBDataset(test_df, test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# -----------------------------
# CNN Model
# -----------------------------

class RGB_CNN(nn.Module):

    def __init__(self):

        super(RGB_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 28 * 28, 256)

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
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

model = RGB_CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 30

# -----------------------------
# Training
# -----------------------------

print("Training RGB CNN (Binary)...")

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