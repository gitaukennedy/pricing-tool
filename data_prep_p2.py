"""
data_prep_p2.py
Upgraded and robust data preparation script for Part 2 (Emerging Market).
"""

import pandas as pd
import numpy as np
from pathlib import Path

EMERGING_DIR = Path("data/raw/emerging")

def get_emerging_path(filename):
    path = EMERGING_DIR / filename
    if not path.exists():
        # Fallback check
        found = list(Path(".").rglob(f"emerging/{filename}"))
        if found:
            return found[0]
        raise FileNotFoundError(f"Could not locate {filename} in {EMERGING_DIR}!")
    return path

def load_shared_tables():
    print("Loading shared tables for emerging market...")
    customers = pd.read_parquet(get_emerging_path("customers.parquet"))
    addresses = pd.read_parquet(get_emerging_path("addresses.parquet"))
    weather = pd.read_parquet(get_emerging_path("weather.parquet"))
    claims = pd.read_parquet(get_emerging_path("claims.parquet"))
    
    # Cast IDs to string strictly
    for df in [customers, addresses, weather, claims]:
        for col in ['customer_id', 'address_id', 'policy_id', 'claim_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
    return customers, addresses, weather, claims

def aggregate_claims(claims_df):
    """Filter valid claims and aggregate net incurred losses per policy and line."""
    print("Aggregating claims (excluding repudiated/declined claims)...")
    valid_claims = claims_df[claims_df['status'] != 'repudiated'].copy()
    
    valid_claims['incurred_to_date'] = pd.to_numeric(valid_claims['incurred_to_date'], errors='coerce').fillna(0.0)
    valid_claims['recoveries'] = pd.to_numeric(valid_claims['recoveries'], errors='coerce').fillna(0.0)
    valid_claims['net_incurred'] = np.maximum(0.0, valid_claims['incurred_to_date'] - valid_claims['recoveries'])
    
    # Group by policy_id and line
    policy_claims = valid_claims.groupby(['policy_id', 'line']).agg(
        total_claim_amount=('net_incurred', 'sum'),
        claim_count=('claim_id', 'count')
    ).reset_index()
    
    print(f"Aggregated {len(valid_claims):,} claims into {len(policy_claims):,} policy records. Total Net Incurred: {policy_claims['total_claim_amount'].sum():,.2f} CU")
    return policy_claims

def build_line_dataset(line_name, customers, addresses, weather, claims_agg):
    print(f"\n--- Processing Line: {line_name.upper()} ---")
    
    train_file = get_emerging_path(f"policies_{line_name}.parquet")
    test_file = get_emerging_path(f"test_{line_name}.parquet")
    
    df_train = pd.read_parquet(train_file)
    df_test = pd.read_parquet(test_file)
    
    # Average weather over available years per address to create stable features
    weather_avg = weather.groupby('address_id').mean(numeric_only=True).reset_index()
    
    datasets = []
    for mode, df in [('train', df_train), ('test', df_test)]:
        for c in ['customer_id', 'address_id', 'policy_id', 'row_id']:
            if c in df.columns:
                df[c] = df[c].astype(str)
                
        # Merges
        df = df.merge(customers, on='customer_id', how='left')
        df = df.merge(addresses, on='address_id', how='left')
        df = df.merge(weather_avg, on='address_id', how='left')
        
        if mode == 'train':
            line_claims = claims_agg[claims_agg['line'] == line_name]
            df = df.merge(line_claims[['policy_id', 'total_claim_amount', 'claim_count']], on='policy_id', how='left')
            df['total_claim_amount'] = df['total_claim_amount'].fillna(0.0)
            df['claim_count'] = df['claim_count'].fillna(0)
            
        datasets.append(df)
        
    out_dir = Path("data/processed_p2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    datasets[0].to_parquet(out_dir / f"emerging_{line_name}_train.parquet", index=False)
    datasets[1].to_parquet(out_dir / f"emerging_{line_name}_test.parquet", index=False)
    print(f"Saved {line_name} processed datasets to {out_dir}")

def run_pipeline():
    customers, addresses, weather, claims = load_shared_tables()
    claims_agg = aggregate_claims(claims)
    
    for line in ['motor', 'buildings', 'contents']:
        build_line_dataset(line, customers, addresses, weather, claims_agg)

if __name__ == "__main__":
    run_pipeline()