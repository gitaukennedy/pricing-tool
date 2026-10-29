"""
pricing_tool/tests/test_pricing.py
Automated pytest suite for schema validation, survival curves, and model constraints.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from data_prep import load_pricing_data, compute_survival_probabilities

def test_load_pricing_data():
    """Verify that pricing dataset loads correctly with expected columns."""
    df = load_pricing_data()
    assert not df.empty, "Pricing dataset is empty."
    for col in ["Premium", "Claims", "Tenure"]:
        assert col in df.columns, f"Required column '{col}' missing from pricing dataset."

def test_survival_curves():
    """Verify Kaplan-Meier survival probability calculations."""
    surv_df = compute_survival_probabilities("CS2S26Q1HA.csv")
    assert not surv_df.empty, "Survival table is empty."
    assert "survival_probability" in surv_df.columns, "Survival probability calculation failed."
    # Survival probabilities must stay bounded between 0 and 1
    assert surv_df['survival_probability'].between(0.0, 1.0).all(), "Survival probabilities out of bounds."

def test_model_predictions_non_negative():
    """Verify that the trained Tweedie model satisfies insurance non-negativity constraints."""
    model_path = Path("models/pricing_tweedie_model.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        df = load_pricing_data()
        X = df[["Premium", "Tenure"]]
        preds = model.predict(X)
        
        assert len(preds) == len(df), "Prediction count mismatch."
        assert (preds >= 0.0).all(), "Model generated negative claim predictions, violating insurance principles."
    else:
        pytest.skip("Trained model binary not found. Run python train.py first.")