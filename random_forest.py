import sqlite3 
import pandas as pd
from pathlib import Path

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

# =============================================================================
#optimizer ile ama recall düştü churn kaçtı.
# rf = RandomForestClassifier(
#     n_estimators = 100,
#     max_depth = 8,
#     min_samples_leaf=4,
#     min_samples_split=2,
#     random_state = 11,
#     class_weight='balanced'
#     )
# =============================================================================

rf.fit(x_train, y_train)

#pred & probabilty
y_pred = rf.predict(x_test)
y_proba = rf.predict_proba(x_test)[:,1]


#class report
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))



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

##PER FEATURE CHURN ETKİSİ[ÇOK İYİ]PARTIAL DEPENDENCE

from sklearn.inspection import PartialDependenceDisplay



PartialDependenceDisplay.from_estimator(rf, x_test, features)
plt.suptitle("Partial Dependence Plots (PDP) for Churn Model")
plt.tight_layout()
plt.show()

##MODEL ÜZERİNDE FEATURELARIN ETKİSİ (ÇOK İYİ) PERMUTATION IMPORTANCE
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



#CONFUSION MATRIX & ROC CURVE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

ConfusionMatrixDisplay.from_estimator(rf, x_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(rf, x_test, y_test)
plt.title("ROC Curve")
plt.show()







