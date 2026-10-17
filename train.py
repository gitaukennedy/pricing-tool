"""
pricing_tool/train.py
Trains LightGBM Tweedie models (p=1.5) for Motor, Buildings, and Contents,
saves model binaries for the PricingEngine, and outputs the final submission file.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
from pathlib import Path
from data_prep import build_feature_pipeline

def train_and_save_models(data_dir: str = "data/raw", models_dir: str = "models", sub_dir: str = "submission"):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(sub_dir).mkdir(parents=True, exist_ok=True)
    
    lobs = ["motor", "buildings", "contents"]
    all_test_submissions = []
    
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'tweedie',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    for lob in lobs:
        print(f"\n==============================================")
        print(f"Training Model for Line of Business: {lob.upper()}")
        print(f"==============================================")
        
        # 1. Load train and test data via preprocessing pipeline
        train_df = build_feature_pipeline(line_of_business=lob, is_train=True, data_dir=data_dir)
        test_df = build_feature_pipeline(line_of_business=lob, is_train=False, data_dir=data_dir)
        
        # Exclude non-feature columns
        exclude_cols = {
            'row_id', 'policy_id', 'customer_id', 'address_id', 
            'ultimate_claim_cost', 'period_start', 'period_end', 'renewal_decision'
        }
        features = [col for col in train_df.columns if col not in exclude_cols]
        
        # Ensure categorical features are properly formatted for LightGBM
        cat_cols = train_df[features].select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            train_df[col] = train_df[col].astype('category')
            test_df[col] = test_df[col].astype('category')
            
        X = train_df[features]
        y = train_df['ultimate_claim_cost']
        exposure = train_df['exposure']
        
        X_test = test_df[features]
        test_exposure = test_df['exposure']
        
        # 2. Train with 5-Fold Cross Validation to ensure robust generalization
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        test_preds = np.zeros(len(test_df))
        
        best_model = None
        best_score = float('inf')
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr, exp_tr = X.iloc[train_idx], y.iloc[train_idx], exposure.iloc[train_idx]
            X_va, y_va, exp_va = X.iloc[val_idx], y.iloc[val_idx], exposure.iloc[val_idx]
            
            train_data = lgb.Dataset(X_tr, label=y_tr, weight=exp_tr)
            val_data = lgb.Dataset(X_va, label=y_va, weight=exp_va, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=400,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
            )
            
            # Accumulate test predictions across folds (scaled by exposure)
            fold_test_rate = model.predict(X_test, num_iteration=model.best_iteration)
            test_preds += (fold_test_rate * test_exposure) / kf.n_splits
            
            if model.best_score['valid_0']['tweedie'] < best_score:
                best_score = model.best_score['valid_0']['tweedie']
                best_model = model
                
        print(f"Finished training {lob}. Best Validation Tweedie Loss: {best_score:.4f}")
        
        # 3. Save the best trained model binary for the PricingEngine
        model_path = Path(models_dir) / f"lgb_tweedie_{lob}.pkl"
        joblib.dump(best_model, model_path)
        print(f"Saved model binary to {model_path}")
        
        # 4. Format test predictions for submission
        test_preds = np.maximum(0.0, test_preds) # Non-negativity constraint
        sub_part = pd.DataFrame({
            'row_id': test_df['row_id'],
            'prediction': test_preds
        })
        all_test_submissions.append(sub_part)
        
    # Combine all lines of business into the final Kaggle submission file
    final_submission = pd.concat(all_test_submissions, ignore_index=True)
    submission_file = Path(sub_dir) / "final_submission.csv"
    final_submission.to_csv(submission_file, index=False)
    
    print(f"\n==============================================")
    print(f"SUCCESS: Final Submission generated at {submission_file}")
    print(f"Total prediction rows: {len(final_submission)}")
    print(f"==============================================")

if __name__ == "__main__":
    train_and_save_models()