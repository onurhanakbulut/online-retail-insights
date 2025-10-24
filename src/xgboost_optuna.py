import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score, recall_score



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
       'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
       'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 4.0),
       'random_state': 11,
       'eval_metric': 'logloss'
        }
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    recalls = []
    
    
    for train_idx, val_idx in kfold.split(x, y):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params) ##dictionary unpacking
        model.fit(x_train, y_train)
        
        y_proba = model.predict_proba(x_val)[:,1]
        
        
        
        best_recall = 0
        for t in np.linspace(0.1, 0.9, 17):
            y_pred = (y_proba>=t).astype(int)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            if prec >= 0.75 and rec > best_recall:
                best_recall = rec
        recalls.append(best_recall)
        
    return np.mean(recalls) ##recalli maximize diyoruz
        




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


####OPTUNA

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best mean recall (CV):", study.best_value)
print("Best params:", study.best_params)
























