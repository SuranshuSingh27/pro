import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_path = PROJECT_ROOT / "data/processed/binary_splits/train.csv"
test_path = PROJECT_ROOT / "data/processed/binary_splits/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

features = [f"b{i}" for i in range(1,17)]

X_train = train_df[features]
y_train = train_df["binary_label"]

X_test = test_df[features]
y_test = test_df["binary_label"]

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

print("Training Binary Random Forest...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))