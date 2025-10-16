import sqlite3
import pandas as pd


DB_PATH = "db/online_retail.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM churn_features;", conn)
conn.close()




