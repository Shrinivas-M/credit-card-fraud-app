import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score

# 1. Load data
print("📥 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

# 2. Scale 'Time' and 'Amount'
print("⚙️ Scaling 'Time' and 'Amount'...")
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
df = df[['scaled_amount', 'scaled_time'] + [col for col in df.columns if col not in ['Class', 'scaled_amount', 'scaled_time']] + ['Class']]

# 3. Feature and target separation
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Train-test split
print("🔀 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Handle class imbalance using SMOTE
print("⚖️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_score = 0
best_model_name = ""

print("🤖 Training models and evaluating performance...\n")
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"🔍 {name} F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name

# 7. Save the best model and scaler
print(f"\n🏆 Best model: {best_model_name} (F1: {best_score:.4f})")

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved successfully to 'model/' folder.")
