import sqlite3 
import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#sqlite import
DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()


#----------rf
##korelasyon haritası
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
##RecencyDays Yüksek korelasyon,
##rfmtotalscore %70 korelasyon
##
#precision artırma denemeleri
# =============================================================================
# scaler = MinMaxScaler()
# df['RFMScore_Normalized'] = scaler.fit_transform(df[['RFM_TotalScore']])
# =============================================================================

#df["FM_Score"] = 0.6 * df["F_Score"] + 0.4 * df["M_Score"]
#-----------------------

#en sağlıklı featurelar.
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
    n_estimators = 100,
    max_depth = 6,
    random_state = 11,
    class_weight='balanced'
    )

#Optimizer ile ama recall düştü churn kaçtı.
# =============================================================================
# rf = RandomForestClassifier(
#     n_estimators = 100,
#     max_depth = 8,
#     min_samples_leaf=4,
#     min_samples_split=2,
#     random_state = 11,
#     class_weight='balanced'
#     )
# 
# =============================================================================
rf.fit(x_train, y_train)

#pred & probabilty------------------
y_pred = rf.predict(x_test)
y_proba = rf.predict_proba(x_test)[:,1]


#class report---------------
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))


#recall@Top10%--------------------------------
y_true = y_test.values

threshold_index = int(len(y_proba) * 0.10)
top_indices = np.argsort(y_proba)[-threshold_index:]

top_churns = y_true[top_indices]
recall_top10 = sum(top_churns) / sum(y_true)    # tüm churn'lerin yüzde kaçı top10'da
precision_top10 = sum(top_churns) / len(top_churns) # top10'da kaç kişi gerçekten churn


######################################################################################
print(f"Recall@Top10%: {recall_top10:.2%}")
print(f"Precision@Top10%: {precision_top10:.2%}")       ##Recall@Top10%: 30.09%
                                                        ##Precision@Top10%: 100.00%
# =============================================================================
# Model, gerçek churn müşterilerinin %30’unu en riskli %10’luk grupta tespit edebiliyor.
# ani kampanyayı sadece bu kitleye yöneltmek, tüm churn’lerin neredeyse yarısını kurtarma potansiyeli taşıyor.”
# =============================================================================





# =============================================================================
# #feature importance
# importances = rf.feature_importances_#verdiği değerin + mı - mi katkı yaptıüını bilmiyoruz (shap)
# feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
# 
# plt.figure(figsize=(6,4))
# sns.barplot(x=[x[1] for x in feat_imp], y=[x[0] for x in feat_imp], palette='viridis')
# plt.title('Feature Importance (Random Forest)')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()
# 
# =============================================================================

#SHAP(begenmedim )

# =============================================================================
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(x_test)
# 
# sv = shap_values[1] if isinstance(shap_values, list) else shap_values   
# shap.summary_plot(sv, x_test)
# =============================================================================

##PER FEATURE CHURN ETKİSİ[ÇOK İYİ]PARTIAL DEPENDENCE--------------------------------------

from sklearn.inspection import PartialDependenceDisplay



PartialDependenceDisplay.from_estimator(rf, x_test, features)
plt.suptitle("Partial Dependence Plots (PDP) for Churn Model")
plt.tight_layout()
plt.show()

##MODEL ÜZERİNDE FEATURELARIN ETKİSİ (ÇOK İYİ) PERMUTATION IMPORTANCE-----------------------------
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, x_test, y_test, n_repeats=10, random_state=11)
sorted_idx = result.importances_mean.argsort()

# =============================================================================
# plt.barh(x_test.columns[sorted_idx], result.importances_mean[sorted_idx])
# plt.title("Permutation Feature Importance")
# plt.show()
# =============================================================================

imp_df = pd.DataFrame({
    'Feature': x_test.columns,
    'MeanImportance': result.importances_mean,
    'StdImportance': result.importances_std
}).sort_values('MeanImportance', ascending=False)

print(imp_df)
plt.barh(imp_df['Feature'], imp_df['MeanImportance'], xerr=imp_df['StdImportance'])
plt.title('Permutation Importance')
plt.xlabel('Mean decrease in ROC-AUC')
plt.show()



#CONFUSION MATRIX & ROC CURVE-------------------------------------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

ConfusionMatrixDisplay.from_estimator(rf, x_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(rf, x_test, y_test)
plt.title("ROC Curve")
plt.show()

##################CROSS VALIDATION, 5 -FOLD--------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

auc_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=11),
    x,
    y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    error_score='raise'
)

print("ROC-AUC Scores:", np.round(auc_scores, 3))
print("Mean AUC:", np.round(np.mean(auc_scores), 3))
print("Std:", np.round(np.std(auc_scores), 3))


