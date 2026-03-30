# =====================================
# Project Title: Student Pass/Fail Predictor
# =====================================

# =====================================
# Step 1 - Import Libraries
# =====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style("whitegrid")


# =====================================
# Step 2 - Load Dataset
# =====================================

file_path = "Student_Performance.csv"   # IMPORTANT: keep file in same folder

df = pd.read_csv(file_path)

print("\n=== Initial Data ===")
print(df.head())

print("\n=== Data Info ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe())


# =====================================
# Step 3 - Handle Missing Values
# =====================================

numeric_cols = ["StudyHours", "Attendance", "PreviousScore", "Result"]

for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)


# =====================================
# Step 4 - Split Features and Target
# =====================================

X = df.drop(["StudentID", "Result"], axis=1)
y = df["Result"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =====================================
# Step 5 - Feature Scaling
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================
# Step 6 - Train Logistic Regression Model
# =====================================

model = LogisticRegression()

model.fit(X_train_scaled, y_train)


# =====================================
# Step 7 - Predictions & Evaluation
# =====================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.2f}")

print("\n=== Confusion Matrix ===")
print(cm)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))


# =====================================
# Step 8 - Visualize Confusion Matrix
# =====================================

plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Student Pass/Fail")

plt.tight_layout()
plt.show()