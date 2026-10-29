"""
pricing_tool/engine.py
Core Actuarial Pricing Engine supporting multi-LOB pricing, technical premium 
loadings, excess relativities, and SHAP model explainability.
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import shap

class PricingEngine:
    def __init__(self, config_path: str = "config.yaml", models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.explainers = {}
        
        self.config = {
            "lines_of_business": ["motor", "buildings", "contents"],
            "actuarial_assumptions": {
                "expense_loading": 0.20,
                "profit_margin": 0.05
            },
            "excess_relativities": {
                "motor": {100: 1.10, 250: 1.00, 500: 0.90},
                "buildings": {100: 1.05, 250: 1.00, 500: 0.95},
                "contents": {100: 1.05, 250: 1.00, 500: 0.95}
            }
        }
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self.config.update(loaded_config)
                    
        for lob in self.config.get("lines_of_business", ["motor"]):
            model_path = self.models_dir / f"lgb_tweedie_{lob}.pkl"
            if not model_path.exists() and lob == "motor":
                fallback_path = self.models_dir / "pricing_tweedie_model.pkl"
                if fallback_path.exists():
                    model_path = fallback_path
                    
            if model_path.exists():
                model = joblib.load(model_path)
                self.models[lob] = model
                try:
                    self.explainers[lob] = shap.TreeExplainer(model)
                except Exception:
                    self.explainers[lob] = None

    def calculate_technical_premium(self, pure_premium: float, lob: str, voluntary_excess: int) -> dict:
        assumptions = self.config["actuarial_assumptions"]
        target_denominator = 1.0 - assumptions["expense_loading"] - assumptions["profit_margin"]
        
        if target_denominator <= 0:
            raise ValueError("Expense loading + profit margin must be less than 1.0")
            
        excess_table = self.config.get("excess_relativities", {}).get(lob, {250: 1.0})
        excess_factor = excess_table.get(int(voluntary_excess), 1.0)
        
        adjusted_pure_premium = pure_premium * excess_factor
        technical_premium = adjusted_pure_premium / target_denominator
        
        return {
            "pure_risk_premium": round(float(pure_premium), 2),
            "excess_adjusted_premium": round(float(adjusted_pure_premium), 2),
            "technical_premium": round(float(technical_premium), 2),
            "loading_breakdown": {
                "expense_loading": round(float(assumptions["expense_loading"] * technical_premium), 2),
                "profit_margin": round(float(assumptions["profit_margin"] * technical_premium), 2),
                "excess_relativity_applied": float(excess_factor)
            }
        }

    def predict_policy(self, features_df: pd.DataFrame, lob: str = "motor") -> dict:
        if lob not in self.models and len(self.models) > 0:
            lob = list(self.models.keys())[0]
            
        if lob not in self.models:
            raise ValueError(f"Model for Line of Business '{lob}' not loaded.")
            
        model = self.models[lob]
        exposure = features_df['exposure'].values if 'exposure' in features_df.columns else np.ones(len(features_df))
        
        # FIX: Restrict model input strictly to the features it was trained on: ["Premium", "Tenure"]
        model_features = ["Premium", "Tenure"]
        X_model = features_df[model_features]
        
        predicted_rate = model.predict(X_model)
        pure_premiums = predicted_rate * exposure
        pure_premiums = np.maximum(0.0, pure_premiums)
        
        results = []
        for i, pp in enumerate(pure_premiums):
            excess = int(features_df.iloc[i].get('voluntary_excess', 250))
            pricing = self.calculate_technical_premium(pp, lob, excess)
            results.append(pricing)
            
        return results[0] if len(results) == 1 else results