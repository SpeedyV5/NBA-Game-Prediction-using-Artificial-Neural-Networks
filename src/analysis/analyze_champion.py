# src/analysis/analyze_champion.py
"""
Champion Model Analysis - Baseline GBM
Analyzes the winning Baseline GBM model that achieved 0.717 AUC.
"""
import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss, mean_absolute_error,
    mean_squared_error
)

RANDOM_SEED = 30

TRAIN_PATH = "data_processed/train_set.csv"
VAL_PATH   = "data_processed/val_set.csv"
TEST_PATH  = "data_processed/test_set.csv"

METRICS_PATH = "reports/metrics/model_results.json"
ENSEMBLE_METRICS_PATH = "reports/metrics/ensemble_results.json"

# Output paths
CHAMPION_FEATURES_CSV = "reports/champion_features.csv"
CHAMPION_FEATURES_PLOT = "reports/figures/champion_feature_importance.png"
FINAL_REPORT_PATH = "reports/FINAL_PROJECT_SUMMARY.txt"


def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)


def load_splits():
    """Load train/val/test splits."""
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, val, test


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """Infer feature columns (same logic as baseline)."""
    drop_cols = [c for c in ["home_team_win", "score_diff"] if c in df.columns]

    # drop obvious ids (varsa)
    for c in ["game_id", "match_id", "id"]:
        if c in df.columns:
            drop_cols.append(c)

    # only numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in drop_cols]
    return feat_cols


def prepare_tabular(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """Prepare tabular data (same as baseline)."""
    feat_cols = infer_feature_cols(train)

    X_train = train[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_val   = val[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_test  = test[feat_cols].to_numpy(dtype=np.float32, copy=True)

    # 1) Drop columns that are all-NaN in TRAIN
    all_nan_mask = np.all(np.isnan(X_train), axis=0)
    if np.any(all_nan_mask):
        dropped = int(all_nan_mask.sum())
        print(f"[WARN] Dropping {dropped} all-NaN columns from train features")
        X_train = X_train[:, ~all_nan_mask]
        X_val   = X_val[:, ~all_nan_mask]
        X_test  = X_test[:, ~all_nan_mask]
        feat_cols = [c for c, keep in zip(feat_cols, ~all_nan_mask) if keep]

    # 2) Fill NaNs using TRAIN median (no leakage)
    train_median = np.nanmedian(X_train, axis=0)
    train_median = np.where(np.isnan(train_median), 0.0, train_median)

    X_train = np.where(np.isnan(X_train), train_median, X_train)
    X_val   = np.where(np.isnan(X_val),   train_median, X_val)
    X_test  = np.where(np.isnan(X_test),  train_median, X_test)

    # 3) Scale (fit only on train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, feat_cols, scaler


def extract_feature_importance(
    model: GradientBoostingClassifier,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importance from the trained model.
    
    Args:
        model: Trained GradientBoostingClassifier
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature names and importance scores
    """
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Get top N
    top_features = importance_df.head(top_n).copy()
    
    # Add rank
    top_features["rank"] = range(1, len(top_features) + 1)
    
    # Reorder columns
    top_features = top_features[["rank", "feature", "importance"]]
    
    return top_features


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    top_n: int = 20
):
    """
    Create horizontal bar plot of feature importance.
    
    Args:
        importance_df: DataFrame with feature names and importance scores
        output_path: Path to save the plot
        top_n: Number of features to plot
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Horizontal bar plot
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features["importance"].values, align="center")
    plt.yticks(y_pos, top_features["feature"].values)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Most Important Features (Baseline GBM)", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, v in enumerate(top_features["importance"].values):
        plt.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[OK] Feature importance plot saved: {output_path}")


def read_metrics_json(path: str) -> Dict:
    """Read metrics JSON file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_model_leaderboard(metrics_all: Dict, ensemble_metrics: Dict = None) -> List[Tuple[str, float, float]]:
    """
    Extract model leaderboard from metrics.
    
    Returns:
        List of tuples: (model_name, test_auc, test_mae)
    """
    leaderboard = []
    
    # Baseline GBM
    if "baseline_gbm" in metrics_all:
        test_metrics = metrics_all["baseline_gbm"].get("test", {})
        clf_metrics = test_metrics.get("classification", {})
        reg_metrics = test_metrics.get("regression", {})
        leaderboard.append((
            "Baseline GBM",
            clf_metrics.get("roc_auc", 0.0),
            reg_metrics.get("mae", 0.0)
        ))
    
    # MLP Classifier
    if "mlp_classifier" in metrics_all:
        test_metrics = metrics_all["mlp_classifier"].get("test", {})
        leaderboard.append((
            "MLP Classifier",
            test_metrics.get("roc_auc", 0.0),
            0.0  # MLP regressor separate
        ))
    
    if "mlp_regressor" in metrics_all:
        test_metrics = metrics_all["mlp_regressor"].get("test", {})
        # Update MLP entry if exists, otherwise add new
        mlp_idx = next((i for i, (name, _, _) in enumerate(leaderboard) if name == "MLP Classifier"), None)
        if mlp_idx is not None:
            leaderboard[mlp_idx] = (leaderboard[mlp_idx][0], leaderboard[mlp_idx][1], test_metrics.get("mae", 0.0))
        else:
            leaderboard.append(("MLP Regressor", 0.0, test_metrics.get("mae", 0.0)))
    
    # XGBoost
    if "xgb_classifier" in metrics_all:
        test_metrics = metrics_all["xgb_classifier"].get("test", {})
        leaderboard.append((
            "XGBoost (Baseline)",
            test_metrics.get("roc_auc", 0.0),
            0.0
        ))
    
    if "xgb_regressor" in metrics_all:
        test_metrics = metrics_all["xgb_regressor"].get("test", {})
        xgb_idx = next((i for i, (name, _, _) in enumerate(leaderboard) if name == "XGBoost (Baseline)"), None)
        if xgb_idx is not None:
            leaderboard[xgb_idx] = (leaderboard[xgb_idx][0], leaderboard[xgb_idx][1], test_metrics.get("mae", 0.0))
    
    # XGBoost Tuned
    if "xgb_classifier_tuned" in metrics_all:
        test_metrics = metrics_all["xgb_classifier_tuned"].get("test", {})
        leaderboard.append((
            "XGBoost (Tuned)",
            test_metrics.get("roc_auc", 0.0),
            0.0
        ))
    
    if "xgb_regressor_tuned" in metrics_all:
        test_metrics = metrics_all["xgb_regressor_tuned"].get("test", {})
        xgb_tuned_idx = next((i for i, (name, _, _) in enumerate(leaderboard) if name == "XGBoost (Tuned)"), None)
        if xgb_tuned_idx is not None:
            leaderboard[xgb_tuned_idx] = (leaderboard[xgb_tuned_idx][0], leaderboard[xgb_tuned_idx][1], test_metrics.get("mae", 0.0))
    
    # Ensemble
    if ensemble_metrics and "metrics" in ensemble_metrics:
        test_metrics = ensemble_metrics["metrics"].get("test", {})
        leaderboard.append((
            "Ensemble (Soft Voting)",
            test_metrics.get("roc_auc", 0.0),
            test_metrics.get("regression", {}).get("mae", 0.0) if "regression" in test_metrics else 0.0
        ))
    
    # Sort by AUC (descending)
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    
    return leaderboard


def generate_final_report(
    leaderboard: List[Tuple[str, float, float]],
    top_features: pd.DataFrame,
    best_auc: float,
    best_mae: float
):
    """
    Generate final project summary report.
    
    Args:
        leaderboard: List of (model_name, auc, mae) tuples
        top_features: DataFrame with top features
        best_auc: Best AUC achieved
        best_mae: Best MAE achieved
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("NBA GAME PREDICTION - FINAL PROJECT SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Project: NBA Game Prediction using Artificial Neural Networks")
    report_lines.append("Course: CENG 481")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("MODEL LEADERBOARD")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"{'Rank':<5} {'Model':<30} {'Test AUC':<12} {'Test MAE':<12}")
    report_lines.append("-" * 80)
    
    for rank, (model_name, auc, mae) in enumerate(leaderboard, 1):
        auc_str = f"{auc:.4f}" if auc > 0 else "N/A"
        mae_str = f"{mae:.2f}" if mae > 0 else "N/A"
        report_lines.append(f"{rank:<5} {model_name:<30} {auc_str:<12} {mae_str:<12}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("BEST METRICS ACHIEVED")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Best Classification AUC: {best_auc:.4f}")
    report_lines.append(f"Best Regression MAE:     {best_mae:.2f}")
    report_lines.append("")
    report_lines.append("WINNER: Baseline GBM (GradientBoostingClassifier with sklearn defaults)")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("TOP 5 MOST IMPORTANT FEATURES")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    top_5 = top_features.head(5)
    for idx, row in top_5.iterrows():
        report_lines.append(f"{int(row['rank'])}. {row['feature']:<40} Importance: {row['importance']:.4f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("ANALYSIS & CONCLUSIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Why Baseline GBM Outperformed Complex Models:")
    report_lines.append("")
    report_lines.append("1. ROBUSTNESS vs OVERFITTING:")
    report_lines.append("   - The Baseline GBM used sklearn default parameters, which are well-tuned")
    report_lines.append("     for general-purpose use. These defaults include conservative settings")
    report_lines.append("     (e.g., max_depth=3, learning_rate=0.1) that prevent overfitting.")
    report_lines.append("   - Complex models (MLP with 512-256-128-64 layers, tuned XGBoost) may have")
    report_lines.append("     overfit to training patterns that don't generalize to test data.")
    report_lines.append("")
    report_lines.append("2. FEATURE UTILIZATION:")
    report_lines.append("   - Baseline GBM used all 203 features without aggressive feature selection.")
    report_lines.append("   - Tree-based models (GBM) naturally perform feature selection through")
    report_lines.append("     splitting, making them robust to noisy features.")
    report_lines.append("   - MLP used only 50 features (after RandomForest selection), potentially")
    report_lines.append("     losing important signal.")
    report_lines.append("")
    report_lines.append("3. MODEL SIMPLICITY:")
    report_lines.append("   - Gradient Boosting is inherently well-suited for tabular data.")
    report_lines.append("   - The default parameters strike a good balance between bias and variance.")
    report_lines.append("   - Neural networks require more hyperparameter tuning and may struggle")
    report_lines.append("     with the limited dataset size (~17K samples) relative to model complexity.")
    report_lines.append("")
    report_lines.append("4. REGRESSION PERFORMANCE:")
    report_lines.append("   - All models converged to ~10.0 MAE, suggesting this may be a fundamental")
    report_lines.append("     limit of the current feature set for predicting score differences.")
    report_lines.append("   - Score differences in NBA games have high variance, making precise")
    report_lines.append("     prediction challenging even with good features.")
    report_lines.append("")
    report_lines.append("5. ENSEMBLE FAILURE:")
    report_lines.append("   - The ensemble (weighted average of Baseline, MLP, XGBoost) failed to")
    report_lines.append("     beat the Baseline alone (0.694 vs 0.717 AUC).")
    report_lines.append("   - This suggests the other models' predictions were not complementary")
    report_lines.append("     enough to improve upon the champion model.")
    report_lines.append("   - The Baseline's predictions may have already captured most of the")
    report_lines.append("     learnable signal, leaving little room for improvement via ensembling.")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS FOR FUTURE WORK")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("1. Feature Engineering:")
    report_lines.append("   - Focus on creating more predictive features based on the top 5 identified")
    report_lines.append("     important features.")
    report_lines.append("   - Consider domain-specific features (player matchups, recent head-to-head,")
    report_lines.append("     rest days, travel distance, etc.).")
    report_lines.append("")
    report_lines.append("2. Data Collection:")
    report_lines.append("   - Expand dataset with more historical seasons.")
    report_lines.append("   - Include real-time injury data for inference-time adjustments.")
    report_lines.append("")
    report_lines.append("3. Model Improvements:")
    report_lines.append("   - Try LightGBM or CatBoost as alternatives to XGBoost.")
    report_lines.append("   - Experiment with stacking instead of simple weighted averaging.")
    report_lines.append("   - Consider time-series aware models if temporal patterns are important.")
    report_lines.append("")
    report_lines.append("4. Hyperparameter Tuning:")
    report_lines.append("   - Use cross-validation instead of single validation set.")
    report_lines.append("   - Apply Bayesian optimization (Optuna) more extensively.")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"[OK] Final report saved: {FINAL_REPORT_PATH}")


def main():
    set_seed(RANDOM_SEED)
    ensure_dirs()
    
    print("=" * 80)
    print("CHAMPION MODEL ANALYSIS - Baseline GBM")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data splits...")
    train, val, test = load_splits()
    print(f"  Train: {train.shape}")
    print(f"  Val:   {val.shape}")
    print(f"  Test:  {test.shape}")
    
    # Prepare features
    print("\n[2/5] Preparing features...")
    X_train, X_val, X_test, feat_cols, scaler = prepare_tabular(train, val, test)
    print(f"  Features: {len(feat_cols)}")
    
    # Re-train Baseline GBM
    print("\n[3/5] Re-training Baseline GBM (Champion Model)...")
    y_train = train["home_team_win"].values.astype(int)
    y_test  = test["home_team_win"].values.astype(int)
    
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    
    # Evaluate
    test_prob = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_prob)
    print(f"  Test AUC: {test_auc:.4f}")
    
    # Extract feature importance
    print("\n[4/5] Extracting feature importance...")
    importance_df = extract_feature_importance(clf, feat_cols, top_n=20)
    
    # Save feature importance CSV
    importance_df.to_csv(CHAMPION_FEATURES_CSV, index=False)
    print(f"  [OK] Saved: {CHAMPION_FEATURES_CSV}")
    
    # Create feature importance plot
    plot_feature_importance(importance_df, CHAMPION_FEATURES_PLOT, top_n=20)
    
    # Load metrics for leaderboard
    print("\n[5/5] Generating final report...")
    metrics_all = read_metrics_json(METRICS_PATH)
    ensemble_metrics = read_metrics_json(ENSEMBLE_METRICS_PATH)
    
    # Get leaderboard
    leaderboard = get_model_leaderboard(metrics_all, ensemble_metrics)
    
    # Get best metrics
    best_auc = max([auc for _, auc, _ in leaderboard if auc > 0])
    best_mae = min([mae for _, _, mae in leaderboard if mae > 0])
    
    # Generate final report
    generate_final_report(leaderboard, importance_df, best_auc, best_mae)
    
    print("\n" + "=" * 80)
    print("CHAMPION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTop 5 Features:")
    top_5 = importance_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"  {int(row['rank'])}. {row['feature']:<40} ({row['importance']:.4f})")
    print(f"\nAll outputs saved to reports/ directory")


if __name__ == "__main__":
    main()

