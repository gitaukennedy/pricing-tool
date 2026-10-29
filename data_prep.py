"""
pricing_tool/data_prep.py
Robust relational data loader incorporating incurred_to_date and paid_to_date claim targets.
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

def build_and_cache_data(mode: str = "train") -> pd.DataFrame:
    cache_dir = Path("data/processed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"merged_{mode}.parquet"

    data_dir = get_data_dir()
    print(f"\nBuilding relational {mode} dataset from: {data_dir}")

    # 1. Base Policy Tables
    policy_files = ["policies_motor", "policies_buildings", "policies_contents"] if mode == "train" else ["test_motor", "test_buildings", "test_contents"]
    dfs = []
    for p_name in policy_files:
        df_p = load_table(p_name, data_dir)
        if df_p is not None:
            df_p['portfolio_type'] = p_name.replace("policies_", "").replace("test_", "")
            dfs.append(df_p)

    if not dfs:
        raise FileNotFoundError(f"No policy files found for mode '{mode}' in {data_dir}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined base policies shape: {df.shape}")

    # 2. Customers Table
    customers = load_table("customers", data_dir)
    if customers is not None and "customer_id" in df.columns and "customer_id" in customers.columns:
        df = df.merge(customers, on="customer_id", how="left", suffixes=('', '_cust'))
        print("Merged customers table.")

    # 3. Addresses Table
    addresses = load_table("addresses", data_dir)
    if addresses is not None and "address_id" in df.columns and "address_id" in addresses.columns:
        df = df.merge(addresses, on="address_id", how="left", suffixes=('', '_addr'))
        print("Merged addresses table.")

    # 4. Weather Table
    weather = load_table("weather", data_dir)
    if weather is not None and "address_id" in df.columns and "address_id" in weather.columns:
        date_col = next((c for c in ['start_date', 'effective_date', 'date', 'year'] if c in df.columns), None)
        if date_col:
            df['policy_year'] = df[date_col] if 'year' in date_col.lower() and df[date_col].dtype in [np.int64, np.float64] else pd.to_datetime(df[date_col], errors='coerce').dt.year
            if 'year' in weather.columns:
                df = df.merge(weather, left_on=["address_id", "policy_year"], right_on=["address_id", "year"], how="left", suffixes=('', '_weather'))
            else:
                df = df.merge(weather, on="address_id", how="left", suffixes=('', '_weather'))
        else:
            df = df.merge(weather, on="address_id", how="left", suffixes=('', '_weather'))
        print("Merged weather table.")

    # 5. Claims Table & Target Identification (Train Mode Only)
    if mode == "train":
        claims = load_table("claims", data_dir)
        if claims is not None:
            if "policy_id" in df.columns and "policy_id" in claims.columns:
                # Explicitly prioritize actuarial incurred or paid loss columns
                claim_amt_col = next((c for c in ['incurred_to_date', 'paid_to_date', 'claim_amount', 'amount', 'cost'] if c in claims.columns), None)
                
                if not claim_amt_col:
                    amt_cols = [
                        c for c in claims.columns 
                        if any(k in c.lower() for k in ['incurred', 'paid', 'cost', 'amount', 'loss', 'recoveries']) 
                        and 'id' not in c.lower() 
                        and pd.api.types.is_numeric_dtype(claims[c])
                    ]
                    claim_amt_col = amt_cols[0] if amt_cols else None
                
                if claim_amt_col:
                    claims_agg = claims.groupby("policy_id").agg(
                        total_claim_amount=(claim_amt_col, 'sum'),
                        claim_count=(claim_amt_col, 'count')
                    ).reset_index()
                    print(f"Aggregated claims using actuarial target column '{claim_amt_col}' (Sum: {claims_agg['total_claim_amount'].sum():,.2f}).")
                else:
                    claims_agg = claims.groupby("policy_id").size().reset_index(name='claim_count')
                    claims_agg['total_claim_amount'] = 0.0
                    print("Aggregated claims count only.")

                df = df.merge(claims_agg, on="policy_id", how="left")
                df['total_claim_amount'] = df['total_claim_amount'].fillna(0.0)
                df['claim_count'] = df['claim_count'].fillna(0)

    # Save to compressed Parquet cache
    df.to_parquet(cache_file, index=False, compression="snappy")
    print(f"Successfully cached {mode} dataset to {cache_file} with shape {df.shape}")
    return df

def load_pricing_data(mode: str = "train", force_rebuild: bool = False) -> pd.DataFrame:
    cache_file = Path("data/processed") / f"merged_{mode}.parquet"
    if cache_file.exists() and not force_rebuild:
        print(f"Loading cached {mode} dataset from Parquet: {cache_file}")
        return pd.read_parquet(cache_file)
    else:
        return build_and_cache_data(mode)

if __name__ == "__main__":
    print("Building cached Parquet datasets with actuarial incurred loss detection...")
    build_and_cache_data("train")
    build_and_cache_data("test")