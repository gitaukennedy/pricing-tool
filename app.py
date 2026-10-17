"""
pricing_tool/app.py
FastAPI backend for single quotes, batch pricing, and actuary exports.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import io
from engine import PricingEngine

app = FastAPI(title="IAA AI-Assisted Actuarial Pricing Engine", version="1.0.0")
engine = PricingEngine()

class PolicyQuoteRequest(BaseModel):
    line_of_business: str
    voluntary_excess: int = 250
    exposure: float = 1.0
    features: dict

@app.post("/quote/single")
def get_single_quote(payload: PolicyQuoteRequest):
    try:
        df = pd.DataFrame([payload.features])
        df['exposure'] = payload.exposure
        df['voluntary_excess'] = payload.voluntary_excess
        
        pricing_result = engine.predict_policy(df, payload.line_of_business)
        return {
            "status": "success",
            "line_of_business": payload.line_of_business,
            "pricing": pricing_result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch/pricing")
async def batch_pricing(line_of_business: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
            
        predictions = engine.predict_policy(df, line_of_business)
        if isinstance(predictions, dict):
            predictions = [predictions]
            
        df['predicted_pure_premium'] = [p['pure_risk_premium'] for p in predictions]
        df['technical_premium'] = [p['technical_premium'] for p in predictions]
        
        output_path = f"submission/batch_scored_{line_of_business}.xlsx"
        df.to_excel(output_path, index=False)
        
        return FileResponse(output_path, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=f"scored_{line_of_business}.xlsx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))