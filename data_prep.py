"""
pricing_tool/data_prep.py
Handles automated dataset downloading via kagglehub, parquet table joins, 
and ultimate claim target aggregation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import kagglehub
import shutil

def download_and_setup_data(target_dir: str = "data/raw"):
    """Automatically downloads competition data using kagglehub if not already present."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Check if parquet files already exist
    existing_files = list(target_path.glob("*.parquet"))
    if len(existing_files) >= 11:
        print(f"Dataset already exists in {target_path} ({len(existing_files)} files found).")
        return target_path

    print("Downloading IAA Hackathon dataset via kagglehub...")
    download_path = kagglehub.competition_download('iaa-ai-assisted-hackathon-2026-part-1')
    print(f"Downloaded files to temporary path: {download_path}")
    
    # Move files to pricing_tool/data/raw/
    for file_path in Path(download_path).glob("*"):
        if file_path.is_file():
            shutil.copy(file_path, target_path / file_path.name)
            print(f"Copied {file_path.name} to {target_path}")
            
    return target_path

def load_base_tables(data_dir: str = "data/raw"):
    """Ensures data is downloaded, then loads core shared tables with correct ID types."""
    actual_data_dir = download_and_setup_data(data_dir)
    
    customers = pd.read_parquet(actual_data_dir / "customers.parquet")
    addresses = pd.read_parquet(actual_data_dir / "addresses.parquet")
    weather = pd.read_parquet(actual_data_dir / "weather.parquet")
    claims = pd.read_parquet(actual_data_dir / "claims.parquet")
    
    id_cols = ["customer_id", "address_id", "policy_id", "claim_id"]
    for df in [customers, addresses, weather, claims]:
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
    return customers, addresses, weather, claims

def aggregate_claims_to_policy_rows(policies_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
    """Links claims to specific policy rows based on policy_id and loss_date falling in period."""
    valid_claims = claims_df[claims_df['status'] != 'repudiated'].copy()
    valid_claims['loss_date'] = pd.to_datetime(valid_claims['loss_date'])
    policies_df['period_start'] = pd.to_datetime(policies_df['period_start'])
    policies_df['period_end'] = pd.to_datetime(policies_df['period_end'])
    
    merged = pd.merge(
        policies_df,
        valid_claims[['claim_id', 'policy_id', 'loss_date', 'incurred_to_date']],
        on='policy_id',
        how='left'
    )
    
    in_period = (
        (merged['loss_date'] >= merged['period_start']) & 
        (merged['loss_date'] < merged['period_end'])
    )
    
    merged['claim_cost'] = np.where(in_period, merged['incurred_to_date'].fillna(0.0), 0.0)
    policy_cols = list(policies_df.columns)
    
    agg_claims = merged.groupby(policy_cols, as_index=False)['claim_cost'].sum()
    agg_claims.rename(columns={'claim_cost': 'ultimate_claim_cost'}, inplace=True)
    return agg_claims

def build_feature_pipeline(line_of_business: str = "motor", is_train: bool = True, data_dir: str = "data/raw"):
    """Complete feature engineering and merging pipeline for a given line of business."""
    actual_data_dir = download_and_setup_data(data_dir)
    customers, addresses, weather, claims = load_base_tables(data_dir)
    
    prefix = "policies_" if is_train else "test_"
    policies = pd.read_parquet(actual_data_dir / f"{prefix}{line_of_business}.parquet")
    
    for col in ["row_id", "policy_id", "customer_id", "address_id"]:
        if col in policies.columns:
            policies[col] = policies[col].astype(str)
            
    policies['period_start'] = pd.to_datetime(policies['period_start'])
    policies['year'] = policies['period_start'].dt.year
    
    df = policies.merge(customers, on='customer_id', how='left')
    df = df.merge(addresses, on='address_id', how='left')
    df = df.merge(weather, on=['address_id', 'year'], how='left')
    
    if is_train:
        df = aggregate_claims_to_policy_rows(df, claims)
        
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
        df['policyholder_age_calc'] = (df['period_start'] - df['date_of_birth']).dt.days // 365
        
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna("Unknown")
        
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'ultimate_claim_cost':
            df[col] = df[col].fillna(df[col].median(skipna=True))
            
    return df