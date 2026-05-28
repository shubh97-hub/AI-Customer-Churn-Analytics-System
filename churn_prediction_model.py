
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load Dataset
df = pd.read_csv('customer_churn_data.csv')

# Encode categorical columns
le = LabelEncoder()

categorical_cols = ['Gender', 'InternetService', 'ContractType', 'PaymentMethod', 'Churn']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, 'customer_churn_model.pkl')

print("Model saved successfully!")
