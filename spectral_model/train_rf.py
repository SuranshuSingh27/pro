import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_path = PROJECT_ROOT / "data/processed/splits/train.csv"
test_path = PROJECT_ROOT / "data/processed/splits/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# spectral feature columns
features = [f"b{i}" for i in range(1,17)]

X_train = train_df[features]
y_train = train_df["label"]

X_test = test_df[features]
y_test = test_df["label"]

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training complete")

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))