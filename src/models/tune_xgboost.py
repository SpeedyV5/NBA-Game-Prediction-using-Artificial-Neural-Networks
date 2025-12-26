# src/models/tune_xgboost.py
"""
XGBoost hyperparameter optimization using Optuna.
Optimizes for ROC AUC on validation set.
"""
import os
import json
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss, mean_absolute_error,
    mean_squared_error
)

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Install with: pip install xgboost")
    raise

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    raise

RANDOM_SEED = 30
N_TRIALS = 50  # Number of Optuna trials

TRAIN_PATH = "data_processed/train_set.csv"
VAL_PATH   = "data_processed/val_set.csv"
TEST_PATH  = "data_processed/test_set.csv"

OUT_MODEL_CLF = "models/xgb_classifier_tuned.json"
OUT_MODEL_REG = "models/xgb_regressor_tuned.json"
BEST_PARAMS_PATH = "models/best_params_xgboost.json"

PRED_PATH = "data_processed/predictions_xgboost_tuned.csv"
METRICS_PATH = "reports/metrics/model_results.json"
FIG_BASE = "reports/figures/xgboost_tuned"
FIG_CLS  = os.path.join(FIG_BASE, "classifier")
FIG_REG  = os.path.join(FIG_BASE, "regressor")


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data_processed", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs(FIG_CLS, exist_ok=True)
    os.makedirs(FIG_REG, exist_ok=True)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, val, test


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """Infer feature columns (same logic as MLP)."""
    drop_cols = [c for c in ["home_team_win", "score_diff"] if c in df.columns]
    
    for c in ["game_id", "match_id", "id"]:
        if c in df.columns:
            drop_cols.append(c)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in drop_cols]
    return feat_cols


def prepare_tabular(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare tabular data for XGBoost.
    """
    feat_cols = infer_feature_cols(train)

    X_train = train[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_val   = val[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_test  = test[feat_cols].to_numpy(dtype=np.float32, copy=True)

    # Fill NaNs with median
    train_median = np.nanmedian(X_train, axis=0)
    train_median = np.where(np.isnan(train_median), 0.0, train_median)

    X_train = np.where(np.isnan(X_train), train_median, X_train)
    X_val   = np.where(np.isnan(X_val),   train_median, X_val)
    X_test  = np.where(np.isnan(X_test),  train_median, X_test)

    return X_train, X_val, X_test, feat_cols


def objective_classifier(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for classifier optimization."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    val_pred = model.predict(dval)
    val_auc = roc_auc_score(y_val, val_pred)
    
    return val_auc


def plot_classification_diagnostics(y_true, y_prob, fig_prefix: str):
    """Plot classification diagnostics."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    from sklearn.calibration import calibration_curve

    y_pred = (y_prob >= 0.5).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_roc.png", dpi=150)
    plt.close()

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("mean predicted probability")
    plt.ylabel("fraction of positives")
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_calibration.png", dpi=150)
    plt.close()


def plot_regression_scatter(y_true, y_pred, fig_path: str):
    """Plot regression scatter plot."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="red", label="Perfect prediction")
    plt.xlabel("true score_diff")
    plt.ylabel("pred score_diff")
    plt.legend()
    plt.title(f"Regression Predictions (MAE: {mean_absolute_error(y_true, y_pred):.2f})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def eval_classifier(y_true, y_prob) -> Dict:
    """Evaluate classifier."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def eval_regression(y_true, y_pred) -> Dict:
    """Evaluate regressor."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def read_metrics_json(path: str) -> Dict:
    """Read metrics JSON."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metrics_json(path: str, obj: Dict) -> None:
    """Write metrics JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    set_global_seed(RANDOM_SEED)
    ensure_dirs()

    print("=" * 60)
    print("XGBoost Hyperparameter Optimization with Optuna")
    print("=" * 60)

    train, val, test = load_splits()
    
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")

    X_train, X_val, X_test, feat_cols = prepare_tabular(train, val, test)
    print(f"\nFeatures: {len(feat_cols)}")

    # -------------------------
    # CLASSIFIER OPTIMIZATION
    # -------------------------
    print("\n" + "=" * 60)
    print("Optimizing XGBoost Classifier")
    print("=" * 60)

    y_train = train["home_team_win"].values.astype(int)
    y_val   = val["home_team_win"].values.astype(int)
    y_test  = test["home_team_win"].values.astype(int)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_classifier_optimization",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    print(f"\nStarting optimization with {N_TRIALS} trials...")
    study.optimize(
        lambda trial: objective_classifier(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    print(f"\nBest trial:")
    print(f"  Value (AUC): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Train final model with best parameters
    best_params_clf = study.best_params.copy()
    best_params_clf.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 1
    })

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("\nTraining final classifier with best parameters...")
    clf_model = xgb.train(
        best_params_clf,
        dtrain,
        num_boost_round=best_params_clf["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=50
    )

    clf_model.save_model(OUT_MODEL_CLF)
    print(f"[OK] Saved classifier: {OUT_MODEL_CLF}")

    # Predictions
    val_prob = clf_model.predict(dval)
    test_prob = clf_model.predict(dtest)

    clf_val_metrics = eval_classifier(y_val, val_prob)
    clf_test_metrics = eval_classifier(y_test, test_prob)

    print(f"\nClassifier Validation Metrics:")
    print(f"  AUC: {clf_val_metrics['roc_auc']:.4f}")
    print(f"  Accuracy: {clf_val_metrics['accuracy']:.4f}")
    print(f"  F1: {clf_val_metrics['f1']:.4f}")

    print(f"\nClassifier Test Metrics:")
    print(f"  AUC: {clf_test_metrics['roc_auc']:.4f}")
    print(f"  Accuracy: {clf_test_metrics['accuracy']:.4f}")
    print(f"  F1: {clf_test_metrics['f1']:.4f}")

    plot_classification_diagnostics(y_test, test_prob, os.path.join(FIG_CLS, "test"))

    # -------------------------
    # REGRESSOR (using same best params structure)
    # -------------------------
    print("\n" + "=" * 60)
    print("Training XGBoost Regressor")
    print("=" * 60)

    y_train_r = train["score_diff"].values.astype(float)
    y_val_r   = val["score_diff"].values.astype(float)
    y_test_r  = test["score_diff"].values.astype(float)

    # Remove NaN values
    train_mask = ~np.isnan(y_train_r)
    val_mask = ~np.isnan(y_val_r)
    test_mask = ~np.isnan(y_test_r)

    X_train_reg = X_train[train_mask] if not train_mask.all() else X_train
    y_train_r_reg = y_train_r[train_mask] if not train_mask.all() else y_train_r
    X_val_reg = X_val[val_mask] if not val_mask.all() else X_val
    y_val_r_reg = y_val_r[val_mask] if not val_mask.all() else y_val_r
    X_test_reg = X_test[test_mask] if not test_mask.all() else X_test
    y_test_r_reg = y_test_r[test_mask] if not test_mask.all() else y_test_r

    if not train_mask.all():
        print(f"[WARN] Dropping {np.sum(~train_mask)} rows with NaN score_diff from train")

    # Use similar params for regressor (remove scale_pos_weight)
    reg_params = best_params_clf.copy()
    reg_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    })
    reg_params.pop("scale_pos_weight", None)

    dtrain_reg = xgb.DMatrix(X_train_reg, label=y_train_r_reg)
    dval_reg = xgb.DMatrix(X_val_reg, label=y_val_r_reg)
    dtest_reg = xgb.DMatrix(X_test_reg, label=y_test_r_reg)

    print("Training regressor...")
    reg_model = xgb.train(
        reg_params,
        dtrain_reg,
        num_boost_round=reg_params["n_estimators"],
        evals=[(dtrain_reg, "train"), (dval_reg, "val")],
        early_stopping_rounds=20,
        verbose_eval=50
    )

    reg_model.save_model(OUT_MODEL_REG)
    print(f"[OK] Saved regressor: {OUT_MODEL_REG}")

    # Predictions
    val_pred_r = reg_model.predict(dval_reg)
    test_pred_r = reg_model.predict(dtest_reg)

    reg_val_metrics = eval_regression(y_val_r_reg, val_pred_r)
    reg_test_metrics = eval_regression(y_test_r_reg, test_pred_r)

    print(f"\nRegressor Validation Metrics:")
    print(f"  MAE: {reg_val_metrics['mae']:.4f}")
    print(f"  RMSE: {reg_val_metrics['rmse']:.4f}")

    print(f"\nRegressor Test Metrics:")
    print(f"  MAE: {reg_test_metrics['mae']:.4f}")
    print(f"  RMSE: {reg_test_metrics['rmse']:.4f}")

    plot_regression_scatter(y_test_r_reg, test_pred_r, os.path.join(FIG_REG, "test_scatter.png"))

    # -------------------------
    # Save best parameters
    # -------------------------
    best_params_to_save = {
        "classifier": best_params_clf,
        "regressor": reg_params,
        "best_auc": float(study.best_value),
        "n_trials": N_TRIALS,
        "seed": RANDOM_SEED
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_params_to_save, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved best parameters: {BEST_PARAMS_PATH}")

    # -------------------------
    # Predictions CSV
    # -------------------------
    def make_pred_df(split_name: str, y_true_c, y_prob_c, y_true_r, y_pred_r):
        return pd.DataFrame({
            "split": split_name,
            "y_true": y_true_c.astype(int),
            "y_prob": y_prob_c.astype(float),
            "y_pred": (y_prob_c >= 0.5).astype(int),
            "score_diff_true": y_true_r.astype(float),
            "score_diff_pred": y_pred_r.astype(float),
        })

    # Predict on all data for CSV
    dtrain_all = xgb.DMatrix(X_train)
    dval_all = xgb.DMatrix(X_val)
    dtest_all = xgb.DMatrix(X_test)

    train_prob = clf_model.predict(dtrain_all)
    train_pred_r = reg_model.predict(dtrain_all)
    
    val_pred_r_all = reg_model.predict(dval_all)
    test_pred_r_all = reg_model.predict(dtest_all)

    pred_train = make_pred_df("train", y_train, train_prob, y_train_r, train_pred_r)
    pred_val = make_pred_df("val", y_val, val_prob, y_val_r, val_pred_r_all)
    pred_test = make_pred_df("test", y_test, test_prob, y_test_r, test_pred_r_all)

    preds = pd.concat([pred_train, pred_val, pred_test], axis=0, ignore_index=True)
    preds.to_csv(PRED_PATH, index=False)
    print(f"[OK] predictions -> {PRED_PATH}")

    # -------------------------
    # Metrics JSON
    # -------------------------
    metrics_all = read_metrics_json(METRICS_PATH)
    metrics_all["xgb_classifier_tuned"] = {
        "seed": RANDOM_SEED,
        "features_used": len(feat_cols),
        "n_trials": N_TRIALS,
        "best_hyperparams": best_params_clf,
        "val": clf_val_metrics,
        "test": clf_test_metrics,
        "model_path": OUT_MODEL_CLF,
        "predictions_path": PRED_PATH,
        "figures": {
            "test_confusion_matrix": os.path.join(FIG_CLS, "test_confusion_matrix.png"),
            "test_roc": os.path.join(FIG_CLS, "test_roc.png"),
            "test_calibration": os.path.join(FIG_CLS, "test_calibration.png"),
        }
    }

    metrics_all["xgb_regressor_tuned"] = {
        "seed": RANDOM_SEED,
        "features_used": len(feat_cols),
        "hyperparams": reg_params,
        "val": reg_val_metrics,
        "test": reg_test_metrics,
        "model_path": OUT_MODEL_REG,
        "predictions_path": PRED_PATH,
        "figures": {
            "test_scatter": os.path.join(FIG_REG, "test_scatter.png"),
        }
    }

    write_metrics_json(METRICS_PATH, metrics_all)
    print(f"[OK] metrics -> {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("XGBoost Optimization Complete")
    print("=" * 60)
    print(f"Best Validation AUC: {study.best_value:.4f}")
    print(f"Classifier Test AUC: {clf_test_metrics['roc_auc']:.4f}")
    print(f"Regressor Test MAE: {reg_test_metrics['mae']:.4f}")
    print(f"Regressor Test RMSE: {reg_test_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()

