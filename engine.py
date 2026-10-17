"""
pricing_tool/engine.py
Core Actuarial Pricing Engine for IAA AI Assisted Hackathon 2026.
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import shap

class PricingEngine:
    def __init__(self, config_path: str = "config.yaml", models_dir: str = "models"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.models_dir = Path(models_dir)
        self.models = {}
        self.explainers = {}
        
        for lob in self.config["lines_of_business"]:
            model_path = self.models_dir / f"lgb_tweedie_{lob}.pkl"
            if model_path.exists():
                self.models[lob] = joblib.load(model_path)
                self.explainers[lob] = shap.TreeExplainer(self.models[lob])

    def calculate_technical_premium(self, pure_premium: float, lob: str, voluntary_excess: int) -> dict:
        assumptions = self.config["actuarial_assumptions"]
        target_denominator = 1.0 - assumptions["expense_loading"] - assumptions["profit_margin"]
        
        if target_denominator <= 0:
            raise ValueError("Expense loading + profit margin must be less than 1.0")
            
        excess_table = self.config["excess_relativities"].get(lob, {250: 1.0})
        excess_factor = excess_table.get(int(voluntary_excess), 1.0)
        
        adjusted_pure_premium = pure_premium * excess_factor
        technical_premium = adjusted_pure_premium / target_denominator
        
        return {
            "pure_risk_premium": round(pure_premium, 2),
            "excess_adjusted_premium": round(adjusted_pure_premium, 2),
            "technical_premium": round(technical_premium, 2),
            "loading_breakdown": {
                "expense_loading": round(assumptions["expense_loading"] * technical_premium, 2),
                "profit_margin": round(assumptions["profit_margin"] * technical_premium, 2),
                "excess_relativity_applied": excess_factor
            }
        }

    def predict_policy(self, features_df: pd.DataFrame, lob: str) -> dict:
        if lob not in self.models:
            raise ValueError(f"Model for Line of Business '{lob}' not loaded.")
            
        model = self.models[lob]
        exposure = features_df['exposure'].values if 'exposure' in features_df.columns else 1.0
        
        predicted_rate = model.predict(features_df)
        pure_premiums = predicted_rate * exposure
        pure_premiums = np.maximum(0.0, pure_premiums)
        
        results = []
        for i, pp in enumerate(pure_premiums):
            excess = features_df.iloc[i].get('voluntary_excess', 250)
            pricing = self.calculate_technical_premium(pp, lob, excess)
            results.append(pricing)
            
        return results[0] if len(results) == 1 else results