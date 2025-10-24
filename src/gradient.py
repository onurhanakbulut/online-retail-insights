import sqlite3
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from xgboost.callback import EarlyStopping
from sklearn.metrics import precision_score, recall_score

import shap
import seaborn as sns
import numpy as np



def choose_threshold(y_true, y_score, p_min): #y_true =gerçekten churn olup olmadıkları
    thr_grid = np.linspace(0.01, 0.99, 99)          #y_score proba'larının kaç olduğu
    best = (0.50, 0.0, 0.0)      #thr, prec, recc   #p_min ise istediğimiz min precision
    
    for t in thr_grid:
        y_pred = (y_score >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if prec >= p_min and rec > best[2]:
            best = (t, prec, rec)
            
    return best





DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()


# =============================================================================
# print(xgb.__version__)
# =============================================================================


#XGBOOST
features = [
    'Frequency',
    'Monetary',
    'CustomerLifetimeDays',
    ]

target = 'Churn'


x = df[features].copy()
y = df[target].astype(int)




#1000 verimiz var. 700 train 300 test böldük.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=11)
#pos_w = (y_train==0).sum() / (y_train==1).sum() 

#ilk xgboost parametrelerimiz
# =============================================================================
# xgb_model = xgb.XGBClassifier(
#     n_estimators =300,  ##ardı ardına agaç sayisi
#     learning_rate=0.05, #dengeli ögrenme orani
#     max_depth=4,    #dallanabilecegi agac derinligi
#     scale_pos_weight = pos_w,
#     subsample=0.8,  #agaclar verinin %80ini kullanır.
#     colsample_bytree=0.8,   #colomnların (featureların) %80i
#     reg_lambda=1.0, #L2 regularization Ridge cezası katsayısı (dengeli ogrenim, overfiti engeller,underfit olmaz)
#     random_state=11,
#     use_label_encoder=False,
#     eval_metric='logloss'   #binary classificiation için
#     )
# =============================================================================

best_params = {
    'n_estimators': 295,
    'max_depth': 4,
    'learning_rate': 0.014465630909740271,
    'subsample': 0.921851046071778,
    'colsample_bytree': 0.9152188451514542,
    'reg_lambda': 1.661669285001975,
    'scale_pos_weight': 3.704465644603432,
    'random_state': 11,
    'eval_metric': 'logloss'
}


#700 verimiz kalmıştı. bunu 595 train 105 test olarak böldük. x_train2, y_train2 = 595,, x_valid ve y_valid = 105, x_test ve y_test test verimiz
x_train2, x_validation2, y_train2, y_validation2 = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train, random_state=11)

xgb_model = xgb.XGBClassifier(**best_params, early_stopping_rounds = 50)
xgb_model.fit(x_train2, y_train2, eval_set=[(x_validation2, y_validation2)], verbose=False)



val_proba = xgb_model.predict_proba(x_validation2)[:,1]     #validation data üzerinden thr ayarı buluyorz çünkü test verisini öğrenip data leakge yapmasın diye
thr_opt, prec_opt, rec_opt = choose_threshold(y_validation2, val_proba, p_min=0.75)
print(f"[VALID] Chosen threshold={thr_opt:.3f} | Precision={prec_opt:.3f} | Recall={rec_opt:.3f}")
####[VALID] Chosen threshold=0.670 | Precision=0.754 | Recall=0.849


y_proba = xgb_model.predict_proba(x_test)[:,1]
#y_pred = xgb_model.predict(x_test)
# =============================================================================
# thr_opt = 0.670 #choose thr methodu ile bulduk.
# =============================================================================
y_pred = (y_proba >= thr_opt).astype(int)



# =============================================================================
# threshold = 0.50 ###xgboost_threshold.py
# y_pred = (y_proba >= threshold).astype(int)
# =============================================================================

#rapor
print("\n--- TEST METRICS ---")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC", round(roc_auc_score(y_test, y_proba), 3))





##conf-matrix

cm = confusion_matrix(y_test, y_pred)
#print(cm)
ConfusionMatrixDisplay.from_estimator(xgb_model, x_test, y_test)        ##319-63
                                                                        ##807-113# Recall = TP/FP+TP
                                                                        
##FEATURE IMPORTANCE
#######------------feature bu ağacı nek adar iyi hala getirdi

importances = xgb_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature' : features,
     'Importance' : importances
    }).sort_values('Importance', ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title('Feature Importance (XGBoost - Gain)')        ##Gain: bir özelliğin, modelin bilgi kazancına yaptığı katkı
plt.xlabel('Average Gain')                              ##Cover: bir özelliğin karar ağaçlarında ne kadar sıklıkla kullanıldığı
plt.ylabel('Feature')                                   ##Weight: özelliğin kaç kez split (bölme) noktası olarak seçildiği
plt.show()

##SHAP (Shapley Additive Explanations ##test verisi üzerinden global ve davranışsal sonuç



explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values  
shap.summary_plot(shap_values, x_test, plot_type="bar")     #ortalama etkiyi gösteren özet
shap.summary_plot(sv, x_test)




import joblib

joblib.dump(xgb_model, 'models/xgboost_churn/churn_xgb_model.pkl')
with open("models/xgboost_churn/threshold.txt", "w") as f:
    f.write(str(thr_opt))
    
    






