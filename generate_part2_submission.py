"""
generate_part2_submission.py
Upgraded submission generator for Part 2 (supports CSV and Parquet sample submissions).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from train_p2 import preprocess_features

def load_sample_sub():
    """Searches for sample submission in csv or parquet format across common paths."""
    search_paths = [
        Path("data/raw/emerging/sample_submission.csv"),
        Path("data/raw/emerging/sample_submission.parquet"),
        Path("emerging/sample_submission.csv"),
        Path("emerging/sample_submission.parquet"),
    ]
    
    for p in search_paths:
        if p.exists():
            print(f"Found sample submission at: {p}")
            if p.suffix == '.parquet':
                return pd.read_parquet(p)
            else:
                return pd.read_csv(p)
                
    # Recursive fallback search
    for p in Path(".").rglob("sample_submission.*"):
        print(f"Found sample submission via recursive search at: {p}")
        if p.suffix == '.parquet':
            return pd.read_parquet(p)
        else:
            return pd.read_csv(p)
            
    raise FileNotFoundError("Could not locate emerging sample_submission (.csv or .parquet)! Please check your emerging folder.")

def predict_line(line_name):
    print(f"Scoring {line_name.upper()} test data...")
    test_path = Path(f"data/processed_p2/emerging_{line_name}_test.parquet")
    df_test = pd.read_parquet(test_path)
    
    model_data = joblib.load(f"models_p2/model_{line_name}.pkl")
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    df_clean, _ = preprocess_features(df_test)
    X_test = df_clean.reindex(columns=feature_cols)
    for c in feature_cols:
        if c in df_clean.columns and df_clean[c].dtype.name == 'category':
            X_test[c] = X_test[c].astype('category')
            
    # Predict with fold ensemble
    fold_preds = np.column_stack([m.predict(X_test) for m in models])
    avg_preds = np.mean(fold_preds, axis=1)
    avg_preds = np.maximum(0.0, avg_preds) # Non-negative guardrail
    
    return pd.DataFrame({'row_id': df_test['row_id'].astype(str), 'prediction': avg_preds})

def main():
    print("==============================================")
    print(" Generating Part 2 Final Submission")
    print("==============================================")
    
    sub_df = load_sample_sub()
    sub_df['row_id'] = sub_df['row_id'].astype(str)
    print(f"Loaded sample submission with {len(sub_df):,} rows.")
    
    line_preds = []
    for line in ['motor', 'buildings', 'contents']:
        line_preds.append(predict_line(line))
        
    all_preds = pd.concat(line_preds, ignore_index=True)
    print(f"Total scored test rows across all 3 portfolios: {len(all_preds):,}")
    
    # Map predictions to sample submission order
    pred_map = dict(zip(all_preds['row_id'], all_preds['prediction']))
    sub_df['prediction'] = sub_df['row_id'].map(pred_map).fillna(0.0)
    sub_df['prediction'] = np.maximum(0.0, sub_df['prediction'])
    
    out_dir = Path("submission_p2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "final_submission_part2.csv"
    sub_df[['row_id', 'prediction']].to_csv(csv_path, index=False)
    
    print("\nSUCCESS!")
    print(f" -> Saved Part 2 CSV Submission: {csv_path} ({len(sub_df):,} rows)")

if __name__ == "__main__":
    main()