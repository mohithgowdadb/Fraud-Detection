import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


df = pd.read_csv("synthetic_fraud_dataset2.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Transaction_Hour"] = df["Timestamp"].dt.hour
df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek

# Correct the inconsistent weekend feature using Timestamp
df["Is_Weekend_Corrected"] = (df["Day_Of_Week"] >= 5).astype(int)



X = df.drop(["Fraud_Label", "Transaction_ID", "User_ID", "Timestamp"], axis=1)
y = df["Fraud_Label"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)  # 0.25 x 0.8 = 0.2

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)



categorical_columns = [
    "Transaction_Type",
    "Device_Type",
    "Card_Type",
    "Authentication_Method",
    "Merchant_Category",
    "Location",
]

# Risk_Score is not used as a feature.
numerical_columns = [
    "Transaction_Amount",
    "Account_Balance",
    "IP_Address_Flag",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Avg_Transaction_Amount_7d",
    "Failed_Transaction_Count_7d",
    "Card_Age",
    "Transaction_Distance",
    "Is_Weekend",
    "Transaction_Hour",
    "Day_Of_Week",
    "Is_Weekend_Corrected",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", StandardScaler(), numerical_columns),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)


xg_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    random_state=42,
)

xg_model.fit(X_train_processed, y_train)
xg_val_prob = xg_model.predict_proba(X_val_processed)[:, 1]

feature_names = preprocessor.get_feature_names_out()
importance = pd.Series(
    xg_model.feature_importances_, index=feature_names
).sort_values(ascending=False)

print("\nXGBoost Feature Importance:")
print(importance)

print("\nXGBoost training score:", xg_model.score(X_train_processed, y_train))
print("XGBoost validation score:", xg_model.score(X_val_processed, y_val))



precision, recall, thresholds = precision_recall_curve(y_val, xg_val_prob)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = np.argmax(f1_values)

# Move just below the selected probability boundary.
final_threshold = np.nextafter(thresholds[best_index], -np.inf)

print("\nFinal threshold selected on validation:", final_threshold)



test_prob = xg_model.predict_proba(X_test_processed)[:, 1]
test_pred = (test_prob >= final_threshold).astype(int)

print("\n========== FINAL TEST RESULTS ==========")
print("Threshold:", final_threshold)
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Fraud precision:", round(precision_score(y_test, test_pred, zero_division=0), 4))
print("Fraud recall:", round(recall_score(y_test, test_pred, zero_division=0), 4))
print("Fraud F1:", round(f1_score(y_test, test_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))
print("Average precision:", round(average_precision_score(y_test, test_prob), 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))



joblib.dump(xg_model, "fraud_model.joblib")
joblib.dump(preprocessor, "fraud_preprocessor.joblib")

with open("fraud_threshold.txt", "w") as f:
    f.write(str(float(final_threshold)))

print("\n========== SAVING COMPLETE ==========")
print("Saved: fraud_model.joblib")
print("Saved: fraud_preprocessor.joblib")
print("Saved: fraud_threshold.txt")