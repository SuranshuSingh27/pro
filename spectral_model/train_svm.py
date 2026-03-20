import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_path = PROJECT_ROOT / "data/processed/splits/train.csv"
test_path = PROJECT_ROOT / "data/processed/splits/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# spectral features
features = [f"b{i}" for i in range(1,17)]

X_train = train_df[features]
y_train = train_df["label"]

X_test = test_df[features]
y_test = test_df["label"]

# ----------------------------------
# Scaling (important for SVM)
# ----------------------------------

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------
# Hyperparameter search
# ----------------------------------

param_grid = {
    "C": [0.1, 1, 10, 50],
    "gamma": ["scale", 0.1, 0.01],
    "kernel": ["rbf", "linear"]
}

svm = SVC(class_weight="balanced")

grid = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

print("Training SVM with hyperparameter tuning...")

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

model = grid.best_estimator_

# ----------------------------------
# Prediction
# ----------------------------------

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred, zero_division=0))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))