# src/models/ensemble_voting.py
"""
Soft Voting Ensemble for NBA game prediction.
Combines predictions from multiple models using weighted averaging.
"""
import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss, mean_absolute_error,
    mean_squared_error
)

# Prediction file paths
BASELINE_PRED_PATH = "data_processed/predictions_baseline.csv"
MLP_PRED_PATH = "data_processed/predictions_mlp.csv"
XGB_TUNED_PRED_PATH = "data_processed/predictions_xgboost_tuned.csv"

# Output paths
ENSEMBLE_PRED_PATH = "data_processed/predictions_ensemble.csv"
ENSEMBLE_METRICS_PATH = "reports/metrics/ensemble_results.json"

# Ensemble weights (must sum to 1.0)
WEIGHTS = {
    "baseline": 0.5,  # Best model gets highest weight
    "mlp": 0.25,
    "xgboost_tuned": 0.25
}


def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs("data_processed", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)


def load_predictions(file_path: str, model_name: str) -> Optional[pd.DataFrame]:
    """
    Load predictions from a CSV file.
    
    Args:
        file_path: Path to predictions CSV
        model_name: Name of the model (for column naming)
    
    Returns:
        DataFrame with predictions or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} not found. Skipping {model_name}.")
        return None
    
    df = pd.read_csv(file_path)
    print(f"[OK] Loaded {model_name}: {len(df)} rows from {file_path}")
    
    # Rename columns to include model name
    rename_dict = {
        "y_prob": f"y_prob_{model_name}",
        "score_diff_pred": f"score_diff_pred_{model_name}"
    }
    df = df.rename(columns=rename_dict)
    
    return df


def merge_predictions(
    baseline_df: Optional[pd.DataFrame],
    mlp_df: Optional[pd.DataFrame],
    xgb_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge predictions from multiple models.
    
    Args:
        baseline_df: Baseline predictions
        mlp_df: MLP predictions
        xgb_df: XGBoost tuned predictions
    
    Returns:
        Merged DataFrame
    """
    # Start with the first available dataframe
    if baseline_df is not None:
        merged = baseline_df[["split", "y_true", "score_diff_true"]].copy()
        if "y_prob_baseline" in baseline_df.columns:
            merged["y_prob_baseline"] = baseline_df["y_prob_baseline"]
        if "score_diff_pred_baseline" in baseline_df.columns:
            merged["score_diff_pred_baseline"] = baseline_df["score_diff_pred_baseline"]
    elif mlp_df is not None:
        merged = mlp_df[["split", "y_true", "score_diff_true"]].copy()
    elif xgb_df is not None:
        merged = xgb_df[["split", "y_true", "score_diff_true"]].copy()
    else:
        raise ValueError("No prediction files found!")
    
    # Merge other models
    if mlp_df is not None:
        if "y_prob_mlp" in mlp_df.columns:
            merged["y_prob_mlp"] = mlp_df["y_prob_mlp"]
        if "score_diff_pred_mlp" in mlp_df.columns:
            merged["score_diff_pred_mlp"] = mlp_df["score_diff_pred_mlp"]
    
    if xgb_df is not None:
        if "y_prob_xgboost_tuned" in xgb_df.columns:
            merged["y_prob_xgboost_tuned"] = xgb_df["y_prob_xgboost_tuned"]
        if "score_diff_pred_xgboost_tuned" in xgb_df.columns:
            merged["score_diff_pred_xgboost_tuned"] = xgb_df["score_diff_pred_xgboost_tuned"]
    
    return merged


def create_ensemble_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ensemble predictions using weighted averaging.
    
    Args:
        df: Merged predictions DataFrame
    
    Returns:
        DataFrame with ensemble predictions added
    """
    df = df.copy()
    
    # Collect available probability columns
    prob_cols = []
    prob_weights = []
    
    if "y_prob_baseline" in df.columns:
        prob_cols.append("y_prob_baseline")
        prob_weights.append(WEIGHTS["baseline"])
    
    if "y_prob_mlp" in df.columns:
        prob_cols.append("y_prob_mlp")
        prob_weights.append(WEIGHTS["mlp"])
    
    if "y_prob_xgboost_tuned" in df.columns:
        prob_cols.append("y_prob_xgboost_tuned")
        prob_weights.append(WEIGHTS["xgboost_tuned"])
    
    if not prob_cols:
        raise ValueError("No probability columns found for ensemble!")
    
    # Normalize weights to sum to 1.0
    total_weight = sum(prob_weights)
    prob_weights = [w / total_weight for w in prob_weights]
    
    print(f"\nEnsemble weights (normalized):")
    for col, weight in zip(prob_cols, prob_weights):
        print(f"  {col}: {weight:.3f}")
    
    # Weighted average for classification probabilities
    df["y_prob_ensemble"] = 0.0
    for col, weight in zip(prob_cols, prob_weights):
        df["y_prob_ensemble"] += weight * df[col].fillna(0.5)  # Fill NaN with 0.5 (neutral)
    
    # Binary prediction
    df["y_pred_ensemble"] = (df["y_prob_ensemble"] >= 0.5).astype(int)
    
    # Collect available regression columns
    reg_cols = []
    
    if "score_diff_pred_baseline" in df.columns:
        reg_cols.append("score_diff_pred_baseline")
    
    if "score_diff_pred_mlp" in df.columns:
        reg_cols.append("score_diff_pred_mlp")
    
    if "score_diff_pred_xgboost_tuned" in df.columns:
        reg_cols.append("score_diff_pred_xgboost_tuned")
    
    # Simple average for regression (equal weights)
    if reg_cols:
        df["score_diff_pred_ensemble"] = df[reg_cols].mean(axis=1)
        print(f"\nRegression ensemble using: {reg_cols}")
    else:
        print("[WARN] No regression predictions found. Setting to 0.")
        df["score_diff_pred_ensemble"] = 0.0
    
    return df


def evaluate_ensemble(df: pd.DataFrame, split: str = "test") -> Dict:
    """
    Evaluate ensemble predictions on a specific split.
    
    Args:
        df: DataFrame with ensemble predictions
        split: Split to evaluate ("train", "val", "test", or "all")
    
    Returns:
        Dictionary with evaluation metrics
    """
    if split == "all":
        eval_df = df
    else:
        eval_df = df[df["split"] == split].copy()
    
    if len(eval_df) == 0:
        return {}
    
    y_true = eval_df["y_true"].values
    y_prob = eval_df["y_prob_ensemble"].values
    y_pred = eval_df["y_pred_ensemble"].values
    
    # Classification metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # Regression metrics (if available)
    if "score_diff_true" in eval_df.columns and "score_diff_pred_ensemble" in eval_df.columns:
        # Filter out NaN values
        mask = ~(eval_df["score_diff_true"].isna() | eval_df["score_diff_pred_ensemble"].isna())
        if mask.sum() > 0:
            y_true_reg = eval_df.loc[mask, "score_diff_true"].values
            y_pred_reg = eval_df.loc[mask, "score_diff_pred_ensemble"].values
            
            metrics["regression"] = {
                "mae": float(mean_absolute_error(y_true_reg, y_pred_reg)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))),
                "n_samples": int(mask.sum())
            }
    
    return metrics


def save_ensemble_results(df: pd.DataFrame, metrics: Dict):
    """
    Save ensemble predictions and metrics.
    
    Args:
        df: DataFrame with ensemble predictions
        metrics: Evaluation metrics dictionary
    """
    # Save predictions CSV
    output_cols = [
        "split", "y_true", "y_prob_ensemble", "y_pred_ensemble",
        "score_diff_true", "score_diff_pred_ensemble"
    ]
    
    # Add individual model predictions for reference
    for col in df.columns:
        if col.startswith("y_prob_") or col.startswith("score_diff_pred_"):
            if col not in output_cols:
                output_cols.append(col)
    
    output_df = df[output_cols].copy()
    output_df.to_csv(ENSEMBLE_PRED_PATH, index=False)
    print(f"\n[OK] Saved ensemble predictions: {ENSEMBLE_PRED_PATH}")
    
    # Save metrics JSON
    ensemble_results = {
        "ensemble_weights": WEIGHTS,
        "models_used": {
            "baseline": os.path.exists(BASELINE_PRED_PATH),
            "mlp": os.path.exists(MLP_PRED_PATH),
            "xgboost_tuned": os.path.exists(XGB_TUNED_PRED_PATH)
        },
        "metrics": metrics,
        "predictions_path": ENSEMBLE_PRED_PATH
    }
    
    with open(ENSEMBLE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(ensemble_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved ensemble metrics: {ENSEMBLE_METRICS_PATH}")


def main():
    ensure_dirs()
    
    print("=" * 60)
    print("Soft Voting Ensemble")
    print("=" * 60)
    
    # Load predictions from all models
    print("\nLoading predictions...")
    baseline_df = load_predictions(BASELINE_PRED_PATH, "baseline")
    mlp_df = load_predictions(MLP_PRED_PATH, "mlp")
    xgb_df = load_predictions(XGB_TUNED_PRED_PATH, "xgboost_tuned")
    
    # Merge predictions
    print("\nMerging predictions...")
    merged_df = merge_predictions(baseline_df, mlp_df, xgb_df)
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(f"Available columns: {merged_df.columns.tolist()}")
    
    # Create ensemble predictions
    print("\nCreating ensemble predictions...")
    ensemble_df = create_ensemble_predictions(merged_df)
    
    # Evaluate on each split
    print("\n" + "=" * 60)
    print("Ensemble Evaluation")
    print("=" * 60)
    
    metrics = {}
    for split in ["train", "val", "test"]:
        split_metrics = evaluate_ensemble(ensemble_df, split=split)
        if split_metrics:
            metrics[split] = split_metrics
            print(f"\n{split.upper()} Metrics:")
            print(f"  Accuracy: {split_metrics.get('accuracy', 0):.4f}")
            print(f"  F1: {split_metrics.get('f1', 0):.4f}")
            print(f"  ROC AUC: {split_metrics.get('roc_auc', 0):.4f}")
            if "regression" in split_metrics:
                print(f"  Regression MAE: {split_metrics['regression']['mae']:.4f}")
                print(f"  Regression RMSE: {split_metrics['regression']['rmse']:.4f}")
    
    # Overall metrics
    all_metrics = evaluate_ensemble(ensemble_df, split="all")
    if all_metrics:
        metrics["all"] = all_metrics
    
    # Save results
    save_ensemble_results(ensemble_df, metrics)
    
    print("\n" + "=" * 60)
    print("Ensemble Complete")
    print("=" * 60)
    if "test" in metrics:
        print(f"Test AUC: {metrics['test'].get('roc_auc', 0):.4f}")
        print(f"Test Accuracy: {metrics['test'].get('accuracy', 0):.4f}")
        if "regression" in metrics["test"]:
            print(f"Test MAE: {metrics['test']['regression']['mae']:.4f}")


if __name__ == "__main__":
    main()

