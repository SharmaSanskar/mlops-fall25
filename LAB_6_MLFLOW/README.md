# MLflow Lab

## Additional Implementations

- Applied **SMOTE** to handle class imbalance in the training data.  
- Trained and compared **two models**: Random Forest and XGBoost.  
- Picked a **champion model** based on validation AUC and registered both champion and challenger models in MLflow.  
- Performed **data drift testing** to simulate changes in input features and measure impact on model performance.  
- Demonstrated **batch predictions** using the champion model.  
- Showed **real-time inference** using MLflow model serving.  
- Used **StandardScaler** to normalize `Amount` and `Time` features.

---

## Screenshots

<img width="856" height="775" alt="Screenshot 2025-12-01 at 7 49 06 PM" src="https://github.com/user-attachments/assets/5757c36b-d2a7-43cf-bd6a-6659c7bd9ed3" />
<img width="1436" height="641" alt="Screenshot 2025-12-01 at 7 49 35 PM" src="https://github.com/user-attachments/assets/e547cf37-7d58-44b9-9c17-37fc047b3598" />
<img width="1446" height="748" alt="Screenshot 2025-12-01 at 7 50 52 PM" src="https://github.com/user-attachments/assets/43f764c2-1561-4d2a-be34-a512022956bf" />
<img width="1443" height="347" alt="Screenshot 2025-12-01 at 7 52 02 PM" src="https://github.com/user-attachments/assets/bfdbec43-9762-4991-9028-2af2f9bb3865" />
 <img width="864" height="328" alt="Screenshot 2025-12-01 at 7 54 51 PM" src="https://github.com/user-attachments/assets/37611caa-2605-4730-acaf-4e37de137620" />


---

## Prerequisites

Before starting the lab, ensure that you have the following:

* Python 3.9+ environment with the required libraries installed.
* Dataset: `creditcard.csv` (truncated due to file size constraints).
* MLflow server running locally for logging, model registry, and model serving.

---

## Step 1: Load Data

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])
```

---

## Step 2: Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

---

## Step 3: Handle Class Imbalance (SMOTE)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
```

---

## Step 4: Train & Log Random Forest

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def log_metrics(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_val, proba),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
    }

with mlflow.start_run(run_name="rf"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_bal, y_train_bal)
    metrics = log_metrics(rf, X_val, y_val)
    mlflow.log_params({"model": "RandomForest", "n_estimators": 100})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        rf,
        "model",
        input_example=X_train_bal[:5],
        signature=mlflow.models.infer_signature(X_train_bal, rf.predict_proba(X_train_bal))
    )
    rf_run_id = mlflow.active_run().info.run_id
```

---

## Step 5: Train & Log XGBoost

```python
import xgboost as xgb
import mlflow.xgboost

with mlflow.start_run(run_name="xgb"):
    xgb_model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=120, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_bal, y_train_bal)
    metrics = log_metrics(xgb_model, X_val, y_val)
    mlflow.log_params({"model": "XGBoost", "max_depth": 6, "lr": 0.1})
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(
        xgb_model,
        "model",
        input_example=X_train_bal[:5],
        signature=mlflow.models.infer_signature(X_train_bal, xgb_model.predict_proba(X_train_bal))
    )
    xgb_run_id = mlflow.active_run().info.run_id
```

---

## Step 6: Pick Champion Model

```python
rf_auc = mlflow.get_run(rf_run_id).data.metrics["auc"]
xgb_auc = mlflow.get_run(xgb_run_id).data.metrics["auc"]

if xgb_auc >= rf_auc:
    champion_run = xgb_run_id
    champion_name = "XGBoost"
    challenger_run = rf_run_id
    challenger_name = "RandomForest"
else:
    champion_run = rf_run_id
    champion_name = "RandomForest"
    challenger_run = xgb_run_id
    challenger_name = "XGBoost"
```

---

## Step 7: Register Models & Transition Stages

```python
from mlflow.tracking import MlflowClient
import time

client = MlflowClient()
model_name = "fraud_detector"

# Champion
champ_version = mlflow.register_model(f"runs:/{champion_run}/model", model_name)
time.sleep(3)
client.transition_model_version_stage(name=model_name, version=champ_version.version, stage="Production")
client.set_registered_model_alias(model_name, "champion", champ_version.version)

# Challenger
chall_version = mlflow.register_model(f"runs:/{challenger_run}/model", model_name)
time.sleep(3)
client.transition_model_version_stage(name=model_name, version=chall_version.version, stage="Staging")
client.set_registered_model_alias(model_name, "challenger", chall_version.version)
```

---

## Step 8: Load Champion Model

```python
champion_loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
```

---

## Step 9: Data Drift Test

```python
import numpy as np
from sklearn.metrics import roc_auc_score

clean_proba = champion_loaded.predict(X_test)
clean_auc = roc_auc_score(y_test, clean_proba)

X_test_drifted = X_test.copy()
np.random.seed(42)
X_test_drifted["Amount"] += np.random.normal(0, 1.2, len(X_test_drifted))

drift_proba = champion_loaded.predict(X_test_drifted)
drift_auc = roc_auc_score(y_test, drift_proba)

print(f"Clean AUC:   {clean_auc:.4f}")
print(f"Drifted AUC: {drift_auc:.4f}")
print(f"Drift Drop:  {clean_auc - drift_auc:.4f}")
```

---

## Step 10: Batch Predictions

```python
batch = X_test.head(10)
batch_proba = champion_loaded.predict(batch)
batch_pred = (batch_proba >= 0.5).astype(int)

import pandas as pd
results = pd.DataFrame({
    'Fraud_Prob': batch_proba,
    'Predicted': batch_pred,
    'Actual': y_test.head(10).values
})
print(results.to_string(index=False))
```

---

## Step 11: Real-Time Inference

```python
import requests

API = "http://localhost:5001"
payload = {"dataframe_split": X_test.head(10).to_dict(orient="split")}
res = requests.post(f"{API}/invocations", json=payload)
predictions = res.json()
print(predictions)
```

---

## Step 12: Conclusion

In this lab, we:

* Preprocessed and scaled numeric data.
* Handled class imbalance using SMOTE.
* Trained, compared, and registered two models: Random Forest and XGBoost.
* Selected a champion model based on validation metrics.
* Tested for data drift to simulate real-world changes.
* Performed batch and real-time inference using MLflow.

This workflow demonstrates the end-to-end machine learning lifecycle for fraud detection, from data preparation to production-ready deployment.

---
