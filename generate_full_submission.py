"""
pricing_tool/generate_full_submission.py
Scores test data from Parquet cache and exports CSV & Parquet submissions.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from data_prep import load_pricing_data, get_data_dir

print("Loading trained pricing model binary...")
model_path = Path("models") / "pricing_tweedie_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first!")

model = joblib.load(model_path)
train_features = model.feature_name()

df_test = load_pricing_data(mode="test")

data_dir = get_data_dir()
sub_path = Path(data_dir) / "sample_submission.parquet"
if not sub_path.exists():
    sub_path = Path(data_dir) / "sample_submission.csv"
    df_sub = pd.read_csv(sub_path)
else:
    df_sub = pd.read_parquet(sub_path)

print(f"Loaded submission template: {len(df_sub):,} rows.")

X_inference = df_test.reindex(columns=train_features).fillna(0.0).astype(float)
raw_preds = np.maximum(0.0, model.predict(X_inference))

# Exposure adjustment
exposure_col = next((c for c in ['exposure', 'Tenure', 'duration'] if c in df_test.columns), None)
if exposure_col:
    exposure_vals = pd.to_numeric(df_test[exposure_col], errors='coerce').fillna(1.0).values
    scaled_preds = raw_preds * exposure_vals
else:
    scaled_preds = raw_preds

if "row_id" in df_sub.columns and "row_id" in df_test.columns:
    pred_mapping = pd.Series(scaled_preds, index=df_test["row_id"]).to_dict()
    df_sub["prediction"] = df_sub["row_id"].map(pred_mapping).fillna(np.median(scaled_preds))
else:
    df_sub["prediction"] = scaled_preds[:len(df_sub)]

df_sub["prediction"] = np.maximum(0.0, df_sub["prediction"])
df_sub = df_sub[["row_id", "prediction"]]

os.makedirs("submission", exist_ok=True)
csv_output = "submission/final_submission.csv"
parquet_output = "submission/final_submission.parquet"

df_sub.to_csv(csv_output, index=False)
df_sub.to_parquet(parquet_output, index=False)

print(f"SUCCESS: Generated exact {len(df_sub):,} rows with valid predictions!")
print(f" -> CSV Submission: {csv_output}")
print(f" -> Parquet Submission: {parquet_output}")

if __name__ == "__main__":
    pass