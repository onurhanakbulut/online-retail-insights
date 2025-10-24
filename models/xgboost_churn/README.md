# 🧠 Churn Prediction Model – Delivery Notes for ML Engineer

## 1️⃣ Project Overview
This model predicts **customer churn probability** in an e-commerce environment based on customer activity and spending patterns.  
The objective is to identify customers at risk of leaving so that the marketing team can take preventive actions.

**Model type:** XGBoost Classifier (binary classification)  
**Optimization tool:** Optuna (hyperparameter tuning)  
**Goal metric:** Maximize Recall (with Precision ≥ 0.75)  
**Final ROC-AUC:** 0.947  
**Chosen threshold:** 0.67  
**Precision / Recall:** 0.76 / 0.85  

---

## 2️⃣ Data Summary
**Source:** `online_retail.db → churn_features`  
**Total records:** 4338  
**Target column:** `Churn` (0 = active, 1 = churned)

**Feature set used for model training:**
| Feature | Type | Description |
|----------|------|-------------|
| `Frequency` | int | Number of unique transactions per customer |
| `Monetary` | float | Total revenue (GBP) from the customer |
| `CustomerLifetimeDays` | int | Number of days between first and last purchase |

---

## 3️⃣ Data Splitting Strategy
| Split | Percentage | Description |
|--------|-------------|-------------|
| **Train** | 70% of total | Used for Optuna optimization & model training |
| **Validation** | 15% of train | Used for threshold and early stopping |
| **Test** | 30% of total | Final model evaluation |

Validation set was used to determine the **best threshold (0.67)** ensuring Precision ≥ 0.75 and Recall as high as possible.

---

## 4️⃣ Model Configuration

**Final optimized hyperparameters (via Optuna):**
| Parameter | Value |
|------------|--------|
| `n_estimators` | 295 |
| `max_depth` | 4 |
| `learning_rate` | 0.014 |
| `subsample` | 0.92 |
| `colsample_bytree` | 0.91 |
| `reg_lambda` | 1.66 |
| `scale_pos_weight` | 3.70 |

**Regularization:** L2 (Ridge)  
**Evaluation metric:** `logloss`  
**Random seed:** 11  
**Early stopping rounds:** 50  

---

## 5️⃣ Model Files
| File | Description |
|------|-------------|
| `models/xgboost_churn/churn_xgb_model.pkl` | Trained and serialized XGBoost model |
| `models/xgboost_churn/threshold.txt` | Contains best threshold value (0.67) |
| `src/gradient.py` | Full training + validation + Optuna optimization script |
| `src/example.py` | Example usage for loading and predicting |

---

## 6️⃣ Environment & Dependencies
**Python version:** 3.11  
**Libraries:**
xgboost==2.1.0
optuna==3.3.0
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.3
seaborn==0.13.2
shap==0.44.1


---


**Input JSON example:**
```json
{
  "Frequency": 5,
  "Monetary": 300.50,
  "CustomerLifetimeDays": 220
}
**Output JSON example:**

{
  "churn_probability": 0.83,
  "prediction": 1
}