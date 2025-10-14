import sqlite3 
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#sqlite import
DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()


#----------rf
features = [
    'Frequency',
    'Monetary',
    'CustomerLifetimeDays',
    ]


x = df[features].copy()
y = df['Churn'].astype(int)


#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=11, stratify=y)

#rf
rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 6,
    random_state = 11,
    class_weight='balanced'
    )

rf.fit(x_train, y_train)

#pred & probabilty
y_pred = rf.predict(x_test)
y_proba = rf.predict_proba(x_test)[:,1]


#class report
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))



#feature importance
importances = rf.feature_importances_#verdiği değerin + mı - mi katkı yaptıüını bilmiyoruz (shap)
feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

#ROC AUC CURVE GRAPH
plt.figure(figsize=(6,4))
sns.barplot(x=[x[1] for x in feat_imp], y=[x[0] for x in feat_imp], palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


#SHAP

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values[1], x_test, plot_type='bar')

shap.summary_plot(shap_values[1], x_test)





















