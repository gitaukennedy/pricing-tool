"""
pricing_tool/app.py
FastAPI backend exposing the actuarial pricing engine for real-time quote generation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from engine import PricingEngine

app = FastAPI(
    title="IAA AI-Assisted Actuarial Pricing Engine",
    description="Actuarial pricing API incorporating Tweedie generalized linear models / gradient boosting, excess relativities, and expense loadings.",
    version="2.0.0"
)

pricing_engine = PricingEngine()

@app.get("/health", summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "loaded_models": list(pricing_engine.models.keys())
    }

@app.post("/pricing/quote", summary="Calculate Technical Premium Quote")
def calculate_quote(lob: str = "motor", premium_input: float = 500.0, tenure: float = 1.0, voluntary_excess: int = 250):
    """Generates an actuarial technical premium quote with expense and profit loadings."""
    try:
        # Construct feature dataframe for model prediction
        df = pd.DataFrame([{
            "Premium": premium_input,
            "Tenure": tenure,
            "exposure": tenure,
            "voluntary_excess": voluntary_excess
        }])
        
        quote_result = pricing_engine.predict_policy(df, lob=lob)
        return {
            "line_of_business": lob,
            "pricing_details": quote_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))