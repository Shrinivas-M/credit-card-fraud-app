# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/creditcard.csv")

# Scale 'Time' and 'Amount'
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
df = df[['scaled_amount', 'scaled_time'] + [col for col in df.columns if col not in ['Class', 'scaled_amount', 'scaled_time']] + ['Class']]

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model/fraud_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and scaler saved successfully.")
