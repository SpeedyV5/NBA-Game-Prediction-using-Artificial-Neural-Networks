# src/models/baselines.py
import os, json, random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss, mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

RANDOM_SEED = 30

TRAIN_PATH = "data_processed/train_set.csv"
VAL_PATH   = "data_processed/val_set.csv"
TEST_PATH  = "data_processed/test_set.csv"

PRED_PATH = "data_processed/predictions_baseline.csv"
METRICS_PATH = "reports/metrics/model_results.json"

# Figures
FIG_DIR = "reports/figures/baseline"
FIG_CLF_DIR = os.path.join(FIG_DIR, "classifier")
FIG_REG_DIR = os.path.join(FIG_DIR, "regressor")


def set_seed(seed=RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    os.makedirs("data_processed", exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    os.makedirs(FIG_CLF_DIR, exist_ok=True)
    os.makedirs(FIG_REG_DIR, exist_ok=True)


def load_splits():
    return pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH), pd.read_csv(TEST_PATH)


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
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


def eval_classifier(y_true, y_prob) -> Dict:
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    # sklearn sürüm uyumu: eps yerine clip
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob_clipped)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def eval_regression(y_true, y_pred) -> Dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": rmse}


def plot_classifier_figs(y_true, y_prob, out_prefix: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    from sklearn.calibration import calibration_curve

    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    plt.savefig(out_prefix + "_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png", dpi=150)
    plt.close()

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("mean predicted probability")
    plt.ylabel("fraction of positives")
    plt.tight_layout()
    plt.savefig(out_prefix + "_calibration.png", dpi=150)
    plt.close()


def plot_reg_scatter(y_true, y_pred, out_path: str) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("true score_diff")
    plt.ylabel("pred score_diff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def read_metrics_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metrics_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    set_seed(RANDOM_SEED)
    ensure_dirs()

    train, val, test = load_splits()
    X_train, X_val, X_test, feat_cols, scaler = prepare_tabular(train, val, test)

    y_train = train["home_team_win"].values.astype(int)
    y_val   = val["home_team_win"].values.astype(int)
    y_test  = test["home_team_win"].values.astype(int)

    y_train_r = train["score_diff"].values.astype(float)
    y_val_r   = val["score_diff"].values.astype(float)
    y_test_r  = test["score_diff"].values.astype(float)

    # Classification baseline
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    train_prob = clf.predict_proba(X_train)[:, 1]
    val_prob   = clf.predict_proba(X_val)[:, 1]
    test_prob  = clf.predict_proba(X_test)[:, 1]

    clf_val  = eval_classifier(y_val, val_prob)
    clf_test = eval_classifier(y_test, test_prob)

    # Regression baseline
    reg = GradientBoostingRegressor(random_state=RANDOM_SEED)
    reg.fit(X_train, y_train_r)
    train_pred_r = reg.predict(X_train)
    val_pred_r   = reg.predict(X_val)
    test_pred_r  = reg.predict(X_test)

    reg_val  = eval_regression(y_val_r, val_pred_r)
    reg_test = eval_regression(y_test_r, test_pred_r)

    # ---- Figures (TEST)
    plot_classifier_figs(
        y_test, test_prob,
        out_prefix=os.path.join(FIG_CLF_DIR, "baseline_test")
    )
    plot_reg_scatter(
        y_test_r, test_pred_r,
        out_path=os.path.join(FIG_REG_DIR, "baseline_test_scatter.png")
    )

    # predictions csv
    def pack(split, y_true_c, y_prob_c, y_true_r, y_pred_r):
        return pd.DataFrame({
            "split": split,
            "y_true": y_true_c,
            "y_prob": y_prob_c,
            "y_pred": (np.asarray(y_prob_c) >= 0.5).astype(int),
            "score_diff_true": y_true_r,
            "score_diff_pred": y_pred_r,
        })

    preds = pd.concat([
        pack("train", y_train, train_prob, y_train_r, train_pred_r),
        pack("val",   y_val,   val_prob,   y_val_r,   val_pred_r),
        pack("test",  y_test,  test_prob,  y_test_r,  test_pred_r),
    ], ignore_index=True)

    preds.to_csv(PRED_PATH, index=False)
    print(f"[OK] predictions -> {PRED_PATH}")

    # metrics json update
    metrics_all = read_metrics_json(METRICS_PATH)
    metrics_all["baseline_gbm"] = {
        "seed": RANDOM_SEED,
        "scaler": "StandardScaler(train_fit_only)",
        "features_used": len(feat_cols),
        "hyperparams": {"model": "GradientBoostingClassifier/Regressor (sklearn default)"},
        "val": {"classification": clf_val, "regression": reg_val},
        "test": {"classification": clf_test, "regression": reg_test},
        "predictions_path": PRED_PATH,
        "figures": {
            "classifier": {
                "confusion_matrix": os.path.join(FIG_CLF_DIR, "baseline_test_confusion_matrix.png"),
                "roc": os.path.join(FIG_CLF_DIR, "baseline_test_roc.png"),
                "calibration": os.path.join(FIG_CLF_DIR, "baseline_test_calibration.png"),
            },
            "regressor": {
                "scatter": os.path.join(FIG_REG_DIR, "baseline_test_scatter.png")
            }
        }
    }

    write_metrics_json(METRICS_PATH, metrics_all)
    print(f"[OK] metrics updated -> {METRICS_PATH}")

    print("\nDONE: baseline\n")


if __name__ == "__main__":
    main()
