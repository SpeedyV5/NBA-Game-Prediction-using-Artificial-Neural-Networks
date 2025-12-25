#!/usr/bin/env python3
"""
Debug script to test regressor predictions on training data.
This helps verify if the sign flip issue is fixed.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.predict_today import (
    load_feature_cols, load_scaler, load_train_median, prepare_X
)

def test_on_training_data():
    """Test regressor on a few training examples where home wins."""
    
    print("="*60)
    print("REGRESSOR DEBUG TEST")
    print("="*60)
    
    # Load training data
    train_path = project_root / "data_processed" / "train_set.csv"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        return
    
    train_df = pd.read_csv(train_path)
    print(f"\nLoaded training data: {len(train_df)} rows")
    
    # Get a few examples where home wins (strong favorites)
    home_wins = train_df[train_df["home_team_win"] == 1].copy()
    if len(home_wins) == 0:
        print("ERROR: No home wins in training data")
        return
    
    # Sample a few rows
    sample = home_wins.head(5)
    print(f"\nTesting on {len(sample)} examples where home_team_win=1")
    print(f"True score_diff range: {sample['score_diff'].min():.1f} to {sample['score_diff'].max():.1f}")
    print(f"Mean true score_diff: {sample['score_diff'].mean():.1f}")
    
    # Load models and scalers
    models_dir = project_root / "models"
    feature_cols = load_feature_cols(models_dir / "feature_cols.json")
    scaler_input = load_scaler(models_dir / "scaler.joblib")
    train_median = load_train_median(models_dir / "train_median.joblib")
    
    try:
        scaler_target = load_scaler(models_dir / "scaler_target.joblib")
        print(f"\nTarget scaler loaded: mean={scaler_target.mean_[0]:.2f}, scale={scaler_target.scale_[0]:.2f}")
    except FileNotFoundError:
        print("\nWARNING: Target scaler not found!")
        scaler_target = None
    
    try:
        reg_model = load_model(str(models_dir / "mlp_regressor_best.h5"))
        print("Regressor model loaded")
    except Exception as e:
        print(f"ERROR loading regressor: {e}")
        return
    
    # Prepare features (need to match inference pipeline)
    # Note: This is simplified - real inference uses build_features_for_today
    # which creates features differently. This test just checks if the model
    # would predict correctly given the same features it was trained on.
    
    # Select only numeric features (excluding labels)
    feature_cols_clean = [c for c in feature_cols if c not in ["home_team_win", "score_diff"]]
    
    # Prepare X
    X_sample = sample[feature_cols_clean].copy()
    X_sample = X_sample.reindex(columns=feature_cols_clean)
    
    # Fill NaN with train median
    if train_median is not None:
        med = train_median.reindex(feature_cols_clean)
        X_sample = X_sample.fillna(med)
    else:
        X_sample = X_sample.fillna(0.0)
    
    # Scale
    X_scaled = scaler_input.transform(X_sample.values)
    
    # Predict
    pred_scaled = reg_model.predict(X_scaled, verbose=0).reshape(-1, 1)
    
    if scaler_target is not None:
        pred_real = scaler_target.inverse_transform(pred_scaled).reshape(-1)
    else:
        pred_real = pred_scaled.reshape(-1)
    
    # Compare
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"{'True Diff':<12} {'Pred Diff':<12} {'Error':<12} {'Sign Match':<12}")
    print("-"*60)
    
    sign_matches = 0
    for i in range(len(sample)):
        true_diff = sample.iloc[i]["score_diff"]
        pred_diff = pred_real[i]
        error = abs(true_diff - pred_diff)
        sign_match = "✓" if (true_diff > 0) == (pred_diff > 0) else "✗"
        if sign_match == "✓":
            sign_matches += 1
        
        print(f"{true_diff:>11.1f} {pred_diff:>11.1f} {error:>11.1f} {sign_match:>11}")
    
    print("-"*60)
    print(f"\nSign matches: {sign_matches}/{len(sample)} ({100*sign_matches/len(sample):.1f}%)")
    print(f"Mean true diff: {sample['score_diff'].mean():.2f}")
    print(f"Mean pred diff: {pred_real.mean():.2f}")
    
    if pred_real.mean() < -5 and sample['score_diff'].mean() > 0:
        print("\n⚠️  WARNING: Model predicting negative when target is positive!")
        print("   This suggests the sign flip issue is still present.")
    elif pred_real.mean() > 0 and sample['score_diff'].mean() > 0:
        print("\n✓ Model predicting positive for home wins (correct sign)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_on_training_data()



