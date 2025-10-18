import sqlite3
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

import shap
import seaborn as sns

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



xgb_model = xgb.XGBClassifier(
    n_estimators =300,  ##ardı ardına agaç sayisi
    learning_rate=0.05, #dengeli ögrenme orani
    max_depth=4,    #dallanabilecegi agac derinligi
    subsample=0.8,  #agaclar verinin %80ini kullanır.
    colsample_bytree=0.8,   #colomnların (featureların) %80i
    reg_lambda=1.0, #L2 regularization Ridge cezası katsayısı (dengeli ogrenim, overfiti engeller,underfit olmaz)
    random_state=11,
    use_label_encoder=False,
    eval_metric='logloss'   #binary classificiation için
    )


xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
y_proba = xgb_model.predict_proba(x_test)[:,1]

#rapora
print(classification_report(y_test, y_pred))
print("ROC-AUC", round(roc_auc_score(y_test, y_proba), 3))


##conf-matrix

cm = confusion_matrix(y_test, y_pred)
#print(cm)
ConfusionMatrixDisplay.from_estimator(xgb_model, x_test, y_test)        ##319-63
                                                                        ##807-113# Recall = TP/FP+TP
                                                                        
##SHAP- FEATURE IMPORTANCE


importances = xgb_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature' : features,
    'Importance' : importances
    }).sort_values('Importance', ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title('Feature Importance (XGBoost - Gain)')
plt.xlabel('Average Gain')
plt.ylabel('Feature')
plt.show()







