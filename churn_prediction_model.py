import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ─────────────────────────────────────────────
# STEP 1A: Generate sample customer data
# (Replace this block with: df = pd.read_csv('customer_churn_data.csv')
#  if you already have a real dataset)
# ─────────────────────────────────────────────
np.random.seed(42)
n = 500

customer_ids = [f"C{str(i).zfill(3)}" for i in range(1, n + 1)]

df = pd.DataFrame({
    'CustomerID':     customer_ids,
    'Gender':         np.random.choice(['Male', 'Female'], n),
    'Age':            np.random.randint(18, 70, n),
    'Tenure':         np.random.randint(1, 72, n),
    'MonthlyCharges': np.round(np.random.uniform(20, 120, n), 2),
    'TotalCharges':   np.round(np.random.uniform(100, 8000, n), 2),
    'InternetService':np.random.choice(['DSL', 'Fiber optic', 'No'], n),
    'ContractType':   np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
    'PaymentMethod':  np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'Churn':          np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73])
})

print("✅ Sample dataset created — shape:", df.shape)
print(df.head(3))

# ─────────────────────────────────────────────
# STEP 1B: Encode categorical columns
# ─────────────────────────────────────────────
le = LabelEncoder()
categorical_cols = ['Gender', 'InternetService', 'ContractType', 'PaymentMethod', 'Churn']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ─────────────────────────────────────────────
# STEP 1C: Features & Target
# ─────────────────────────────────────────────
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# STEP 1D: Train the model
# ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# STEP 1E: Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# STEP 1F: Save model + raw data for Step 2
# ─────────────────────────────────────────────
joblib.dump(model, 'customer_churn_model.pkl')
df.to_csv('encoded_customer_data.csv', index=False)
# Save original CustomerIDs separately for Step 2
pd.DataFrame({'CustomerID': customer_ids}).to_csv('customer_ids.csv', index=False)

print("\n✅ Model saved → customer_churn_model.pkl")
print("✅ Encoded data saved → encoded_customer_data.csv")
