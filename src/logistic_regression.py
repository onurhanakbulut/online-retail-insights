import sqlite3 
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns


#sqlite import
DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()


##lr 

features = [
    #'RecencyDays',#label leakage
    'Frequency',
    'Monetary',
    'CustomerLifetimeDays',
    
    ]



target = 'Churn'


x = df[features].copy()
y = df[target].astype(int)

# =============================================================================
# df[["RecencyDays", "Churn"]].corr()
# df[["AvgOrderValue", "Frequency"]].corr()
# =============================================================================

df[['Frequency', 'F_Score']].corr()

#------SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=11, stratify=y)

#SCALE
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

#LR
lr = LogisticRegression(
    max_iter = 100, #200
    solver = 'lbfgs',   #liblinear 
    class_weight='balanced'
    )

lr.fit(x_train_s, y_train)

#TEST
y_pred = lr.predict(x_test_s)
y_prob = lr.predict_proba(x_test_s)[:,1]

print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))


#matplotlip roc curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()


#coefficient
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="viridis")
plt.title("Feature Importance (Logistic Coefficients)")
plt.show()

print(coef_df)
















