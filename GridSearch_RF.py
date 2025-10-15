import sqlite3 
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


#sqlite import
DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()




features = [
    'Frequency',
    'Monetary',
    'CustomerLifetimeDays',
    ]


x = df[features].copy()
y = df['Churn'].astype(int)


#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=11, stratify=y)




#####gridsearch

param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1,2,4],
    'class_weight': ['balanced']
    }


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

grid = GridSearchCV(RandomForestClassifier(random_state=11), param_grid, scoring='roc_auc', n_jobs=-1, cv=cv, verbose=2)



grid.fit(x_train, y_train)



print("Best Params:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 3))


best_rf = grid.best_estimator_
y_pred = best_rf.predict(x_test)
y_proba = best_rf.predict_proba(x_test)[:,1]

print("\n=== Test Set Results ===")
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))




with open("tuning_results.txt", "w") as f:
    f.write(str(grid.best_params_))
    f.write(f"\nBest CV ROC-AUC: {grid.best_score_:.3f}")















