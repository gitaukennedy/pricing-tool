"""
pricing_tool/data_prep.py
Loads hackathon CSV dataset (CS2S26Q3.csv) and computes actuarial survival curves 
from the retention tables (CS2S26Q1HA.csv / HB).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def find_file(filename: str) -> Path:
    """Recursively search for a file inside data/raw/ or project directories."""
    search_roots = [Path("data/raw"), Path("pricing_tool/data/raw"), Path(".")]
    for root in search_roots:
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Could not find '{filename}' in data/raw/ or its subfolders.")

def load_pricing_data(is_train: bool = True):
    """Loads the main portfolio pricing dataset (CS2S26Q3.csv)."""
    file_path = find_file("CS2S26Q3.csv")
    df = pd.read_csv(file_path)
    
    expected_cols = ["Premium", "Claims", "Tenure"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from CS2S26Q3.csv. Found: {df.columns.tolist()}")
            
    df = df.fillna(0.0)
    return df

def compute_survival_probabilities(table_name: str = "CS2S26Q1HA.csv") -> pd.DataFrame:
    """Computes Kaplan-Meier survival probabilities from actuarial tables."""
    file_path = find_file(table_name)
    df = pd.read_csv(file_path)
    
    required = ["t_j", "n_j", "d_j"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in survival table {table_name}")
            
    # Kaplan-Meier estimator: cumulative product of conditional survival rates
    df['conditional_survival'] = 1.0 - (df['d_j'] / df['n_j'].replace(0, np.nan))
    df['survival_probability'] = df['conditional_survival'].cumprod().fillna(1.0)
    
    return df