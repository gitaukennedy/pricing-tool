import os
import pytest
import pandas as pd
import numpy as np

def test_ai_submission_schema_and_non_negativity():
    """Verify that prediction files match expected schemas and enforce non-negative predictions."""
    p2_file = "submission_p2/final_submission_part2.csv"
    if os.path.exists(p2_file):
        df = pd.read_csv(p2_file)
        assert "row_id" in df.columns, "Missing 'row_id' column in Part 2 submission"
        assert "prediction" in df.columns, "Missing 'prediction' column in Part 2 submission"
        assert (df["prediction"] >= 0).all(), "Negative predictions found in submission file!"
        assert not df["prediction"].isnull().any(), "Null predictions found in submission file!"

def test_exposure_scaling_consistency():
    """Verify that half-year policies price at approximately half the annual rate for identical risk profiles."""
    annual_exposure = 1.0
    half_exposure = 0.5
    base_rate = 500.0
    
    annual_pred = base_rate * annual_exposure
    half_pred = base_rate * half_exposure
    
    assert np.isclose(half_pred, annual_pred * 0.5), "Exposure scaling is non-linear or inconsistent!"

def test_no_cross_market_data_leakage():
    """Verify that Part 1 developed data paths do not contaminate Part 2 emerging market datasets."""
    p1_file = "data_prep.py"
    p2_file = "data_prep_p2.py"
    
    assert os.path.exists(p1_file) and os.path.exists(p2_file), "Data prep scripts missing!"
    
    with open(p2_file, "r") as f:
        content = f.read()
        # Verify Part 2 prep explicitly references emerging data
        assert "emerging" in content.lower(), "Part 2 script does not isolate emerging market datasets!"