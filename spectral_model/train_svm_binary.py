import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
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

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(
    kernel="rbf",
    C=10,
    class_weight="balanced"
)

print("Training Binary SVM...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))