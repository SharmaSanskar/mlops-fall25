import requests
import pandas as pd
import time

API = "http://localhost:5001"
INVOKE = f"{API}/invocations"

# Ping
requests.get(f"{API}/ping", timeout=2)

# Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"]).head(10)
y = df["Class"].head(10).values

# Single prediction
payload = {"dataframe_split": X.iloc[:1].to_dict(orient="split")}
res = requests.post(INVOKE, json=payload)
print("Single prediction:", res.json())

# Batch predictions
payload = {"dataframe_split": X.to_dict(orient="split")}
t0 = time.time()
res = requests.post(INVOKE, json=payload)
preds = res.json()
t1 = time.time() - t0

print("Batch predictions:", preds)
print("Time:", t1, "sec")
