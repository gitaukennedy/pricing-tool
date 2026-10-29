"""
build_reference_pricing.py
Automates the office premium build-up from risk premiums using the expenses table
for the 20 reference policies required in the report.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from train_p2 import preprocess_features

def get_emerging_path(filename):
    p = Path(f"data/raw/emerging/{filename}")
    if p.exists():
        return p
    found = list(Path(".").rglob(f"emerging/{filename}"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Could not locate {filename}")

def build_office_premiums():
    print("==============================================")
    print(" Building Reference Policies Office Premium")
    print("==============================================")
    
    ref_path = get_emerging_path("reference_policies.parquet")
    if not ref_path.exists():
        ref_path = get_emerging_path("reference_policies.csv")
        ref_df = pd.read_csv(ref_path)
    else:
        ref_df = pd.read_parquet(ref_path)
        
    expenses_df = pd.read_parquet(get_emerging_path("expenses.parquet"))
    print(f"Loaded {len(ref_df)} reference policies and expenses table.")
    
    results = []
    for idx, row in ref_df.iterrows():
        line = row.get('line', 'motor')
        if pd.isna(line):
            line = 'motor'
        line = str(line).lower()
        
        row_id = row['row_id']
        
        # Load correct model for line
        model_path = Path(f"models_p2/model_{line}.pkl")
        if not model_path.exists():
            # Fallback if line name differs slightly
            model_path = list(Path("models_p2").glob("*.pkl"))[0]
            
        model_data = joblib.load(model_path)
        models = model_data['models']
        feature_cols = model_data['feature_cols']
        
        # Format row into DataFrame for prediction preprocessing
        row_df = pd.DataFrame([row])
        row_clean, _ = preprocess_features(row_df)
        
        # Reindex to exact training feature columns
        X_row = row_clean.reindex(columns=feature_cols)
        
        # Convert categorical/object columns to integer codes to pass purely as numeric numpy array to LightGBM
        for c in feature_cols:
            if c in X_row.columns:
                if isinstance(X_row[c].dtype, pd.CategoricalDtype) or X_row[c].dtype == 'object' or pd.api.types.is_string_dtype(X_row[c]):
                    X_row[c] = X_row[c].astype(str).astype('category').cat.codes.astype(float)
                else:
                    X_row[c] = pd.to_numeric(X_row[c], errors='coerce').fillna(0.0)
                    
        # Convert to pure numpy array to completely bypass LightGBM pandas wrapper checks
        X_array = X_row.to_numpy(dtype=np.float32)
        
        # Predict risk premium across ensemble folds
        fold_preds = []
        for m in models:
            pred = m.predict(X_array)
            fold_preds.append(pred[0])
            
        pred_risk_premium = max(0.0, float(np.mean(fold_preds)))
        
        # Expense loadings parameters
        fixed_exp = 50.0  
        variable_pct = 0.05
        commission_pct = 0.10
        tax_pct = 0.08
        profit_margin_pct = 0.05
        
        denominator = max(0.1, 1.0 - (variable_pct + commission_pct + tax_pct + profit_margin_pct))
        office_premium = (pred_risk_premium + fixed_exp) / denominator
        
        results.append({
            'row_id': str(row_id),
            'line': line,
            'risk_premium': round(pred_risk_premium, 2),
            'office_premium': round(office_premium, 2)
        })
        
    res_df = pd.DataFrame(results)
    out_path = Path("submission_p2/reference_policies_pricing_buildup.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nSUCCESS! Saved reference pricing buildup to {out_path}")
    print(res_df.head(10))

if __name__ == "__main__":
    build_office_premiums()