import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()




#XGBOOST
features = [
    'Frequency',
    'Monetary',
    'CustomerLifetimeDays',
    ]




target = 'Churn'



x = df[features].copy()
y = df[target].astype(int)





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=11)
pos_w = (y_train==0).sum() / (y_train==1).sum() 


xgb_model = xgb.XGBClassifier(
    n_estimators =300,  ##ardı ardına agaç sayisi
    learning_rate=0.05, #dengeli ögrenme orani
    max_depth=4,    #dallanabilecegi agac derinligi
    scale_pos_weight=pos_w,
    subsample=0.8,  #agaclar verinin %80ini kullanır.
    colsample_bytree=0.8,   #colomnların (featureların) %80i
    reg_lambda=1.0, #L2 regularization Ridge cezası katsayısı (dengeli ogrenim, overfiti engeller,underfit olmaz)
    random_state=11,
    use_label_encoder=False,
    eval_metric='logloss'   #binary classificiation için
    )


xgb_model.fit(x_train, y_train)

#y_pred = xgb_model.predict(x_test)
y_proba = xgb_model.predict_proba(x_test)[:,1]

# =============================================================================
# #rapor
# print(classification_report(y_test, y_pred))
# print("ROC-AUC", round(roc_auc_score(y_test, y_proba), 3))
# =============================================================================

import numpy as np
from sklearn.metrics import precision_score, recall_score


best_recall = 0
best_threshold = 0


for t in np.linspace(0.1, 0.9, 17):
    y_pred = (y_proba >= t).astype(int)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    if prec >= 0.75 and rec > best_recall:
        best_recall = rec
        best_threshold = t
        

print(f"Best threshold: {best_threshold:.2f}")
print(f"Precision >= 0.7, Recall = {best_recall:.2f}")



import matplotlib.pyplot as plt

plt.hist(y_proba[y_test==0], bins=50, alpha=0.6, label='Churn değil (0)')
plt.hist(y_proba[y_test==1], bins=50, alpha=0.6, label='Churn (1)')
plt.axvline(0.25, color='red', linestyle='--', label='Eşik = 0.25')
plt.xlabel("Tahmin edilen churn olasılığı")
plt.ylabel("Müşteri sayısı")
plt.legend()
plt.show()




