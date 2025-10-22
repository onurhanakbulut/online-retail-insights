import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


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


pos_w = (y_train==0).sum() / (y_train==1).sum()     #scale_pos_weight, 1/0 oranı

P_MIN = 0.7         #rpecision 0.7 hedefi
BETA = 2.0          #f beta recall için






