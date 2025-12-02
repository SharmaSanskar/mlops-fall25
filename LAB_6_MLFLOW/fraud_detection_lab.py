import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import time
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("MLFLOW LAB")
print("="*60)

# =========================================================
# 1. Load Data
# =========================================================
data = pd.read_csv("data/creditcard.csv")
print(f"Data: {data.shape}, Fraud Cases: {data['Class'].sum()}")

X = data.drop("Class", axis=1)
y = data["Class"]

# Scale numeric fields
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# Train/Val/Test = 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# =========================================================
# 2. Apply SMOTE
# =========================================================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# =========================================================
# 3. Train & Log Random Forest
# =========================================================
mlflow.set_experiment("fraud_detection")

def log_metrics(model, X_val, y_val):
    """Helper to compute metrics."""
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_val, proba),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
    }

print("\nTraining Random Forest...")

with mlflow.start_run(run_name="rf"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_bal, y_train_bal)

    metrics = log_metrics(rf, X_val, y_val)

    mlflow.log_params({"model": "RandomForest", "n_estimators": 100})
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        rf,
        "model",  # Changed to "model" for consistency
        input_example=X_train_bal[:5],
        signature=mlflow.models.infer_signature(X_train_bal, rf.predict_proba(X_train_bal)),
    )

    rf_run_id = mlflow.active_run().info.run_id

print(f"RF Metrics: {metrics}")

# =========================================================
# 4. Train & Log XGBoost
# =========================================================
print("\nTraining XGBoost...")

with mlflow.start_run(run_name="xgb"):
    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=120,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train_bal, y_train_bal)

    metrics = log_metrics(xgb_model, X_val, y_val)

    mlflow.log_params({"model": "XGBoost", "max_depth": 6, "lr": 0.1})
    mlflow.log_metrics(metrics)

    mlflow.xgboost.log_model(
        xgb_model,
        "model",  # Changed to "model" for consistency
        input_example=X_train_bal[:5],
        signature=mlflow.models.infer_signature(
            X_train_bal, xgb_model.predict_proba(X_train_bal)
        )
    )

    xgb_run_id = mlflow.active_run().info.run_id

print(f"XGB Metrics: {metrics}")

# =========================================================
# 5. Pick Champion Model
# =========================================================
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

print(f"\nBest Model: {champion_name}")

# =========================================================
# 6. Register Champion/Challenger
# =========================================================
client = MlflowClient()
model_name = "fraud_detector"

print("\nRegistering Champion...")
champ_version = mlflow.register_model(f"runs:/{champion_run}/model", model_name)
time.sleep(3)

client.transition_model_version_stage(
    name=model_name,
    version=champ_version.version,
    stage="Production"
)
client.set_registered_model_alias(model_name, "champion", champ_version.version)

print(f"Champion v{champ_version.version}: {champion_name}")

print("\nRegistering Challenger...")
chall_version = mlflow.register_model(f"runs:/{challenger_run}/model", model_name)
time.sleep(3)

client.transition_model_version_stage(
    name=model_name,
    version=chall_version.version,
    stage="Staging"
)
client.set_registered_model_alias(model_name, "challenger", chall_version.version)

print(f"Challenger v{chall_version.version}: {challenger_name}")

# =========================================================
# 7. Load Champion Model
# =========================================================
print("\nLoading Champion...")

# Use pyfunc for universal loading (works for both sklearn and xgboost)
champion_loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

# =========================================================
# 8. Drift Test
# =========================================================
print("\nTesting Data Drift...")

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

# =========================================================
# 9. Batch Predictions
# =========================================================
print("\nBatch Predictions:")

batch = X_test.head(10)
batch_proba = champion_loaded.predict(batch)
batch_pred = (batch_proba >= 0.5).astype(int)

results = pd.DataFrame({
    'Fraud_Prob': batch_proba,
    'Predicted': batch_pred,
    'Actual': y_test.head(10).values
})

print(results.to_string(index=False))

print("\n" + "="*60)
print("COMPLETE")
print("="*60)