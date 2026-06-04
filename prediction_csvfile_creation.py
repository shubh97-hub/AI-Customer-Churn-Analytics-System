import pandas as pd
import joblib

# Load original dataset
df = pd.read_csv(
    r"D:\Data Engineer\Console data\AI Powered churn\customer_churn_data.csv"
)

# Rename to match training data
df.rename(columns={"TenureMonths": "Tenure"}, inplace=True)

prediction_output = df.copy()

# Keep a copy BEFORE encoding
prediction_output = df.copy()

# -----------------------------
# Feature Engineering / Encoding
# -----------------------------

from sklearn.preprocessing import LabelEncoder

le_gender = LabelEncoder()
le_internet = LabelEncoder()
le_contract = LabelEncoder()
le_payment = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["InternetService"] = le_internet.fit_transform(df["InternetService"])
df["ContractType"] = le_contract.fit_transform(df["ContractType"])
df["PaymentMethod"] = le_payment.fit_transform(df["PaymentMethod"])

# -----------------------------
# Load Trained Model
# -----------------------------


model = joblib.load(
    r"C:\Users\shubh\PycharmProjects\Churn customer\customer_churn_model.pkl"
)

# -----------------------------
# Prediction
# -----------------------------

feature_columns = [
    "Gender",
    "Age",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "ContractType",
    "PaymentMethod"
]
print("\nColumns in CSV:")
for col in df.columns:
    print(col)

X = df[feature_columns]

prediction_output["Predicted_Churn"] = model.predict(X)

prediction_output["Churn_Probability"] = model.predict_proba(X)[:, 1]

# -----------------------------
# Risk Segmentation
# -----------------------------

def risk_segment(prob):
    if prob >= 0.80:
        return "High Risk"
    elif prob >= 0.50:
        return "Medium Risk"
    else:
        return "Low Risk"

prediction_output["Risk Segment"] = prediction_output[
    "Churn_Probability"
].apply(risk_segment)

# -----------------------------
# Export Dashboard File
# -----------------------------

prediction_output.to_csv(
    "customer_churn_predictions_dashboard.csv",
    index=False
)

print("Dashboard-ready prediction file created successfully!")