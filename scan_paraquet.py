import os
import pandas as pd

print("Scanning for Parquet files in data/raw...")
for root, dirs, files in os.walk("data/raw"):
    for file in files:
        if file.endswith(".parquet"):
            path = os.path.join(root, file)
            try:
                df = pd.read_parquet(path)
                print(f"File: {file} | Rows: {len(df):,} | Path: {path}")
            except Exception as e:
                print(f"File: {file} | Error reading: {e}")