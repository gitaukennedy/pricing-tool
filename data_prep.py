"""
pricing_tool/data_prep.py
Robust relational data loader with strict numeric validation for claim amounts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def get_data_dir() -> str:
    for root, dirs, files in os.walk("data/raw"):
        if "sample_submission.parquet" in files or "sample_submission.csv" in files or "policies_motor.parquet" in files:
            return root
    return "data/raw"

def load_table(name: str, data_dir: str) -> pd.DataFrame:
    p_path = Path(data_dir) / f"{name}.parquet"
    c_path = Path(data_dir) / f"{name}.csv"
    if p_path.exists():
        return pd.read_parquet(p_path)
    elif c_path.exists():
        return pd.read_csv(c_path)
    return None

def load_pricing_data(mode: str = "train", filename: str = None) -> pd.DataFrame:
    data_dir = get_data_dir()
    print(f"Loading relational tables from: {data_dir} [Mode: {mode}]")

    # 1. Load Core Policy Tables
    if mode == "train":
        policy_files = ["policies_motor", "policies_buildings", "policies_contents"]
    else:
        policy_files = ["test_motor", "test_buildings", "test_contents"]

    dfs = []
    for p_name in policy_files:
        df_p = load_table(p_name, data_dir)
        if df_p is not None:
            df_p['portfolio_type'] = p_name.replace("policies_", "").replace("test_", "")
            dfs.append(df_p)

    if not dfs:
        if filename:
            f_path = Path(data_dir) / filename
            df = pd.read_parquet(f_path) if str(f_path).endswith('.parquet') else pd.read_csv(f_path)
        else:
            raise FileNotFoundError(f"No policy files found for mode '{mode}' in {data_dir}")
    else:
        df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded base policies shape: {df.shape}")

    # 2. Merge Customers table via customer_id
    customers = load_table("customers", data_dir)
    if customers is not None and "customer_id" in df.columns and "customer_id" in customers.columns:
        df = df.merge(customers, on="customer_id", how="left", suffixes=('', '_cust'))
        print("Merged customers table.")

    # 3. Merge Addresses table via address_id
    addresses = load_table("addresses", data_dir)
    if addresses is not None and "address_id" in df.columns and "address_id" in addresses.columns:
        df = df.merge(addresses, on="address_id", how="left", suffixes=('', '_addr'))
        print("Merged addresses table.")

    # 4. Merge Weather table via address_id and year
    weather = load_table("weather", data_dir)
    if weather is not None and "address_id" in df.columns and "address_id" in addresses.columns:
        date_col = next((c for c in ['start_date', 'effective_date', 'date', 'year'] if c in df.columns), None)
        if date_col:
            if 'year' in date_col.lower() and df[date_col].dtype in [np.int64, np.float64]:
                df['policy_year'] = df[date_col]
            else:
                df['policy_year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
            
            if 'year' in weather.columns:
                df = df.merge(weather, left_on=["address_id", "policy_year"], right_on=["address_id", "year"], how="left", suffixes=('', '_weather'))
                print("Merged weather table on address_id and year.")
            else:
                df = df.merge(weather, on="address_id", how="left", suffixes=('', '_weather'))
                print("Merged weather table on address_id.")
        else:
            df = df.merge(weather, on="address_id", how="left", suffixes=('', '_weather'))
            print("Merged weather table on address_id.")

    # 5. Merge Claims table temporally via policy_id (Train mode only)
    if mode == "train":
        claims = load_table("claims", data_dir)
        if claims is not None and "policy_id" in df.columns and "policy_id" in claims.columns:
            # Strictly select numeric columns that represent amounts/costs, excluding IDs and dates
            amt_cols = [
                c for c in claims.columns 
                if any(k in c.lower() for k in ['amount', 'cost', 'paid', 'value', 'claim', 'loss']) 
                and 'id' not in c.lower() 
                and 'date' not in c.lower()
                and pd.api.types.is_numeric_dtype(claims[c])
            ]
            claim_amt_col = amt_cols[0] if amt_cols else None
            
            if claim_amt_col:
                claims_agg = claims.groupby("policy_id").agg(
                    total_claim_amount=(claim_amt_col, 'sum'),
                    claim_count=(claim_amt_col, 'count')
                ).reset_index()
                print(f"Aggregated claims using numeric monetary column '{claim_amt_col}'.")
            else:
                claims_agg = claims.groupby("policy_id").size().reset_index(name='claim_count')
                claims_agg['total_claim_amount'] = 0.0
                print("Aggregated claims count (no numeric claim amount column found in claims table).")

            df = df.merge(claims_agg, on="policy_id", how="left")
            df['total_claim_amount'] = df['total_claim_amount'].fillna(0.0)
            df['claim_count'] = df['claim_count'].fillna(0)
            print("Merged claims table successfully.")

    print(f"Final relational dataset shape [{mode}]: {df.shape}")
    return df