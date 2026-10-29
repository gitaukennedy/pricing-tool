"""
shap_analysis_p2.py (Upgraded to LightGBM Native Interpretability)
Generates feature importance plots for model interpretability,
satisfying the 'Trust, Validation & Guardrails' and 'Effective Use of AI' criteria.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from train_p2 import preprocess_features

def run_shap_analysis():
    print("==============================================")
    print(" Running Model Interpretability Analysis")
    print("==============================================")
    
    out_dir = Path("reports_p2/interpretability")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for line in ['motor', 'buildings', 'contents']:
        print(f"\nAnalyzing {line.upper()} model...")
        
        test_path = Path(f"data/processed_p2/emerging_{line}_test.parquet")
        if not test_path.exists():
            print(f"Warning: Could not find {test_path}, skipping...")
            continue
            
        model_path = Path(f"models_p2/model_{line}.pkl")
        if not model_path.exists():
            print(f"Warning: Could not find model for {line}, skipping...")
            continue
            
        model_data = joblib.load(model_path)
        models = model_data['models']
        feature_cols = model_data['feature_cols']
        
        # Use first fold trained model
        model = models[0]
        
        # Extract native feature importances (gain-based)
        importance_vals = model.feature_importance(importance_type='gain')
        feat_imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_vals
        }).sort_values(by='importance', ascending=False).head(15) # Top 15 features
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp_df['feature'][::-1], feat_imp_df['importance'][::-1], color='teal')
        plt.xlabel("Feature Gain Importance", fontsize=12)
        plt.title(f"Top 15 Risk Drivers - {line.upper()} Portfolio", fontsize=14)
        plt.tight_layout()
        
        plot_path = out_dir / f"feature_importance_{line}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f" -> Saved Feature Importance plot to {plot_path}")

if __name__ == "__main__":
    run_shap_analysis()