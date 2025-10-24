import joblib
import numpy as np

model = joblib.load("models/xgboost_churn/churn_xgb_model.pkl")
threshold = float(open("models/xgboost_churn/threshold.txt").read())

X = [[5, 300.50, 220]]  # Frequency, Monetary, LifetimeDays
proba = model.predict_proba(X)[0,1]
pred = int(proba >= threshold)

print(f"Churn probability: {proba:.2f}, Prediction: {pred}")
