import pandas as pd

# Load dataset
df = pd.read_csv("data/sample_credit_risk_india.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Display dataset shape
print("Dataset Shape:", df.shape)

# EDA

print("\nDataset Description")
print(df.describe())

print("\nLoan Status Count")
print(df["loan_status"].value_counts())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# Loan status graph
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="loan_status", data=df)
plt.title("Loan Status Distribution")
plt.savefig("outputs/loan_status.png")
plt.show()

# Handle Missing Values

print("Before Handling:")
print(df.isnull().sum())

# Fill numerical missing values
for col in df.select_dtypes(include="number"):
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])

print("After Handling:")
print(df.isnull().sum())

# Handle Outliers using IQR

num_cols = df.select_dtypes(include="number").columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

print("Outliers Handled Successfully")

# Convert categorical columns

df = pd.get_dummies(df, drop_first=True)

print("Categorical Encoding Completed")
print(df.head())
print("New Dataset Shape:", df.shape)

# Load original dataset
df = pd.read_csv("data/sample_credit_risk_india.csv")

# Separate input and output
X = df.drop("loan_status", axis=1)

y = df["loan_status"].map({
    "Non-Default": 0,
    "Default": 1
})

# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
print("Data Preparation Completed")

# Logistic Regression Model

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# Identify columns
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

# Preprocessing
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

# Create model
lr_model = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
])

# Train
lr_model.fit(X_train, y_train)

# Predict
pred = lr_model.predict(X_test)
prob = lr_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nLOGISTIC REGRESSION RESULTS")
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))

# Random Forest Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf_model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train model
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nRANDOM FOREST RESULTS")
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
print("Precision:", precision_score(y_test, rf_pred, zero_division=0))
print("Recall:", recall_score(y_test, rf_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# Model Comparison

import pandas as pd

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "ROC-AUC": [
        roc_auc_score(y_test, prob),
        roc_auc_score(y_test, rf_prob)
    ],
    "Precision": [
        precision_score(y_test, pred, zero_division=0),
        precision_score(y_test, rf_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, pred, zero_division=0),
        recall_score(y_test, rf_pred, zero_division=0)
    ]
})

print("\nMODEL COMPARISON")
print(results)

results.to_csv("outputs/model_comparison.csv", index=False)

print("\nComparison Saved Successfully!")

from sklearn.metrics import confusion_matrix

print("\nLOGISTIC REGRESSION CONFUSION MATRIX")
print(confusion_matrix(y_test, pred))

print("\nRANDOM FOREST CONFUSION MATRIX")
print(confusion_matrix(y_test, rf_pred))

import joblib

lr_auc = roc_auc_score(y_test, prob)
rf_auc = roc_auc_score(y_test, rf_prob)

best_model = rf_model if rf_auc > lr_auc else lr_model

joblib.dump(best_model, "outputs/credit_risk_model.pkl")

print("Best Model Saved Successfully!")

from sklearn.inspection import permutation_importance

result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=5, random_state=42,
    scoring="roc_auc"
)

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": result.importances_mean
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)
importance.to_csv("outputs/feature_importance.csv", index=False)
plt.figure(figsize=(10, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Credit Risk - Feature Importance")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png")
plt.show()

print("Feature Importance Graph Saved!")
from sklearn.metrics import precision_score, recall_score

risk_prob = best_model.predict_proba(X_test)[:, 1]

for t in [0.3, 0.4, 0.5, 0.6]:
    risk_pred = (risk_prob >= t).astype(int)

    print("\nThreshold:", t)
    print("Precision:", precision_score(y_test, risk_pred, zero_division=0))
    print("Recall:", recall_score(y_test, risk_pred, zero_division=0))
    from sklearn.metrics import classification_report, roc_curve

print("\nFINAL MODEL EVALUATION")
print(classification_report(y_test, best_model.predict(X_test)))

fpr, tpr, _ = roc_curve(y_test, risk_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("outputs/roc_curve.png")
plt.show()

print("ROC Curve Saved!")