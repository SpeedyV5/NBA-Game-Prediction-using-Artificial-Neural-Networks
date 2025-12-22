# src/models/mlp.py
import os
import json
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss, mean_absolute_error,
    mean_squared_error
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


RANDOM_SEED = 30

TRAIN_PATH = "data_processed/train_set.csv"
VAL_PATH   = "data_processed/val_set.csv"
TEST_PATH  = "data_processed/test_set.csv"

OUT_MODEL_CLF = "models/mlp_classifier_best.h5"
OUT_MODEL_REG = "models/mlp_regressor_best.h5"

PRED_PATH = "data_processed/predictions_mlp.csv"
METRICS_PATH = "reports/metrics/model_results.json"
FIG_BASE = "reports/figures/mlp"
FIG_CLS  = os.path.join(FIG_BASE, "classifier")
FIG_REG  = os.path.join(FIG_BASE, "regressor")



LEAKAGE_HINTS = [
    "home_team_win", "score_diff",
    "home_score", "away_score", "team_score", "opp_score",
    "final", "result", "winner", "margin",
    "pts", "box_score", "post_game"
]


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Determinism (best effort)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


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


def sanity_check(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    print("=== SANITY CHECK ===")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"[{name}] shape = {df.shape}")
        nan_total = df.isna().sum().sum()
        print(f"[{name}] total NaN = {nan_total}")

        if "home_team_win" in df.columns:
            dist = df["home_team_win"].value_counts(dropna=False).to_dict()
            print(f"[{name}] home_team_win dist = {dist}")

    # leakage candidates
    cols = set(train.columns)
    flagged = []
    for c in cols:
        cl = c.lower()
        if any(h in cl for h in LEAKAGE_HINTS):
            flagged.append(c)
    if flagged:
        print("\n[LEAKAGE?] Flagged columns by heuristic:")
        for c in sorted(flagged):
            print(" -", c)
    print("====================\n")


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    # drop labels
    drop_cols = [c for c in ["home_team_win", "score_diff"] if c in df.columns]

    # drop obvious ids
    for c in ["game_id", "match_id", "id"]:
        if c in df.columns:
            drop_cols.append(c)

    # KEEP ONLY numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feat_cols = [c for c in numeric_cols if c not in drop_cols]
    return feat_cols



def prepare_tabular(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str], np.ndarray]:
    feat_cols = infer_feature_cols(train)

    X_train = train[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_val   = val[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_test  = test[feat_cols].to_numpy(dtype=np.float32, copy=True)

    # 1) Drop columns that are all-NaN in TRAIN (otherwise nanmedian -> NaN)
    all_nan_mask = np.all(np.isnan(X_train), axis=0)
    if np.any(all_nan_mask):
        dropped = int(all_nan_mask.sum())
        print(f"[WARN] Dropping {dropped} all-NaN columns from train features")
        X_train = X_train[:, ~all_nan_mask]
        X_val   = X_val[:, ~all_nan_mask]
        X_test  = X_test[:, ~all_nan_mask]
        feat_cols = [c for c, keep in zip(feat_cols, ~all_nan_mask) if keep]

    # 2) Fill NaNs using TRAIN median (NO leakage)
    train_median = np.nanmedian(X_train, axis=0)

    # Safety: if any median still NaN (e.g., weird columns), set to 0
    train_median = np.where(np.isnan(train_median), 0.0, train_median)

    X_train = np.where(np.isnan(X_train), train_median, X_train)
    X_val   = np.where(np.isnan(X_val),   train_median, X_val)
    X_test  = np.where(np.isnan(X_test),  train_median, X_test)

    # 3) Scale (fit only on train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, scaler, feat_cols, train_median



def build_mlp_classifier(input_dim: int, hp: Dict) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = inp
    for units in hp["hidden_units"]:
        x = layers.Dense(units, activation="relu")(x)
        if hp.get("batchnorm", False):
            x = layers.BatchNormalization()(x)
        if hp.get("dropout", 0.0) > 0:
            x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="mlp_classifier")

    opt = keras.optimizers.Adam(learning_rate=hp["lr"])
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")]
    )
    return model


def build_mlp_regressor(input_dim: int, hp: Dict) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = inp
    for units in hp["hidden_units"]:
        x = layers.Dense(units, activation="relu")(x)
        if hp.get("batchnorm", False):
            x = layers.BatchNormalization()(x)
        if hp.get("dropout", 0.0) > 0:
            x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out, name="mlp_regressor")

    opt = keras.optimizers.Adam(learning_rate=hp["lr"])
    loss = hp["loss"]
    model.compile(optimizer=opt, loss=loss, metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


def plot_and_save_history(history: keras.callbacks.History, fig_path_prefix: str, is_classifier: bool):
    import matplotlib.pyplot as plt

    # Loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_path_prefix}_loss.png", dpi=150)
    plt.close()

    if is_classifier:
        # Accuracy curve (if present)
        if "acc" in history.history:
            plt.figure()
            plt.plot(history.history["acc"], label="train_acc")
            plt.plot(history.history["val_acc"], label="val_acc")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{fig_path_prefix}_accuracy.png", dpi=150)
            plt.close()

        # AUC curve (optional)
        if "auc" in history.history:
            plt.figure()
            plt.plot(history.history["auc"], label="train_auc")
            plt.plot(history.history["val_auc"], label="val_auc")
            plt.xlabel("epoch")
            plt.ylabel("AUC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{fig_path_prefix}_auc.png", dpi=150)
            plt.close()


def plot_classification_diagnostics(y_true, y_prob, fig_prefix: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    from sklearn.calibration import calibration_curve

    # Confusion matrix
    y_pred = (y_prob >= 0.5).astype(int)
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
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("true score_diff")
    plt.ylabel("pred score_diff")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def eval_classifier(y_true, y_prob) -> Dict:
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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def read_metrics_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metrics_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    set_global_seed(RANDOM_SEED)
    ensure_dirs()

    train, val, test = load_splits()
    sanity_check(train, val, test)

    X_train, X_val, X_test, scaler, feat_cols, train_median = prepare_tabular(train, val, test)

    # === SAVE INFERENCE ARTIFACTS (TABULAR) ===
    import joblib

    os.makedirs("models", exist_ok=True)

    # 1) feature list
    with open("models/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: models/feature_cols.json (n={len(feat_cols)})")

    # 2) scaler
    joblib.dump(scaler, "models/scaler.joblib")
    print("[OK] Saved: models/scaler.joblib")

    # 3) train median (NaN fill için)
    joblib.dump(train_median, "models/train_median.joblib")
    print("[OK] Saved: models/train_median.joblib")

    # -------------------------
    # B) MLP CLASSIFIER
    # -------------------------
    y_train = train["home_team_win"].values.astype(int)
    y_val   = val["home_team_win"].values.astype(int)
    y_test  = test["home_team_win"].values.astype(int)

    clf_variants = [
        {"name": "MLP_C1", "hidden_units": [256, 128, 64], "dropout": 0.2, "batchnorm": False, "lr": 1e-3},
        {"name": "MLP_C2", "hidden_units": [128, 64],      "dropout": 0.0, "batchnorm": False, "lr": 1e-3},
        {"name": "MLP_C3", "hidden_units": [512, 256, 128, 64], "dropout": 0.3, "batchnorm": True, "lr": 5e-4},
    ]

    best_val_auc = -1
    best_hp = None
    best_model_path = OUT_MODEL_CLF

    for hp in clf_variants:
        tf.keras.backend.clear_session()
        model = build_mlp_classifier(X_train.shape[1], hp)

        ckpt_path = best_model_path  # required name
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=256,
            callbacks=callbacks,
            verbose=2
        )

        # evaluate on val
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        val_auc = roc_auc_score(y_val, val_prob)
        print(f"[{hp['name']}] val_auc={val_auc:.4f}")

        plot_and_save_history(history, os.path.join(FIG_CLS, f"train_{hp['name']}"), is_classifier=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_hp = hp

    # Load best checkpoint (safety)
    clf_best = keras.models.load_model(best_model_path)

    # metrics val/test
    val_prob = clf_best.predict(X_val, verbose=0).reshape(-1)
    test_prob = clf_best.predict(X_test, verbose=0).reshape(-1)

    clf_val_metrics = eval_classifier(y_val, val_prob)
    clf_test_metrics = eval_classifier(y_test, test_prob)

    plot_classification_diagnostics(y_test, test_prob, os.path.join(FIG_CLS, "test"))

    # -------------------------
    # C) MLP REGRESSOR
    # -------------------------
    y_train_r = train["score_diff"].values.astype(float)
    y_val_r   = val["score_diff"].values.astype(float)
    y_test_r  = test["score_diff"].values.astype(float)

    reg_variants = [
        {"name": "MLP_R1", "hidden_units": [256, 128, 64], "dropout": 0.1, "batchnorm": False, "lr": 1e-3, 
        "loss": "mse", "loss_name": "mse"},
        {"name": "MLP_R2", "hidden_units": [256, 128, 64], "dropout": 0.1, "batchnorm": False, "lr": 1e-3, 
        "loss": keras.losses.Huber(delta=5.0), "loss_name": "huber(delta=5.0)"},
    ]


    best_val_rmse = 1e18
    best_reg_hp = None

    for hp in reg_variants:
        tf.keras.backend.clear_session()
        model = build_mlp_regressor(X_train.shape[1], hp)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(OUT_MODEL_REG, monitor="val_loss", mode="min", save_best_only=True)
        ]

        history = model.fit(
            X_train, y_train_r,
            validation_data=(X_val, y_val_r),
            epochs=200,
            batch_size=256,
            callbacks=callbacks,
            verbose=2
        )

        val_pred = model.predict(X_val, verbose=0).reshape(-1)
        val_rmse = float(np.sqrt(mean_squared_error(y_val_r, val_pred)))
        print(f"[{hp['name']}] val_rmse={val_rmse:.4f}")

        plot_and_save_history(history, os.path.join(FIG_REG, f"train_{hp['name']}"), is_classifier=False)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_reg_hp = hp

    reg_best = keras.models.load_model(OUT_MODEL_REG)

    val_pred_r = reg_best.predict(X_val, verbose=0).reshape(-1)
    test_pred_r = reg_best.predict(X_test, verbose=0).reshape(-1)

    reg_val_metrics = eval_regression(y_val_r, val_pred_r)
    reg_test_metrics = eval_regression(y_test_r, test_pred_r)

    plot_regression_scatter(y_test_r, test_pred_r, os.path.join(FIG_REG, "test_scatter.png"))


    # -------------------------
    # Predictions CSV (required)
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

    pred_train = make_pred_df(
        "train",
        y_train,
        clf_best.predict(X_train, verbose=0).reshape(-1),
        y_train_r,
        reg_best.predict(X_train, verbose=0).reshape(-1),
    )
    pred_val = make_pred_df("val", y_val, val_prob, y_val_r, val_pred_r)
    pred_test = make_pred_df("test", y_test, test_prob, y_test_r, test_pred_r)

    preds = pd.concat([pred_train, pred_val, pred_test], axis=0, ignore_index=True)
    preds.to_csv(PRED_PATH, index=False)
    print(f"[OK] predictions -> {PRED_PATH}")

    # -------------------------
    # Metrics JSON (append/update)
    # -------------------------
    metrics_all = read_metrics_json(METRICS_PATH)
    metrics_all["mlp_classifier"] = {
        "seed": RANDOM_SEED,
        "scaler": "StandardScaler(train_fit_only)",
        "features_used": len(feat_cols),
        "best_hyperparams": best_hp,
        "val": clf_val_metrics,
        "test": clf_test_metrics,
        "model_path": OUT_MODEL_CLF,
        "predictions_path": PRED_PATH,
        "figures": {
            "train_loss": os.path.join(FIG_CLS, f"train_{best_hp['name']}_loss.png"),
            "train_accuracy": os.path.join(FIG_CLS, f"train_{best_hp['name']}_accuracy.png"),
            "train_auc": os.path.join(FIG_CLS, f"train_{best_hp['name']}_auc.png"),
            "test_confusion_matrix": os.path.join(FIG_CLS, "test_confusion_matrix.png"),
            "test_roc": os.path.join(FIG_CLS, "test_roc.png"),
            "test_calibration": os.path.join(FIG_CLS, "test_calibration.png"),
        }

    }
    # best_reg_hp JSON-serializable hale getir
    best_reg_hp_safe = dict(best_reg_hp)
    if "loss" in best_reg_hp_safe and not isinstance(best_reg_hp_safe["loss"], (str, int, float, bool, type(None))):
        # Huber gibi objeleri stringe çevir
        best_reg_hp_safe["loss"] = best_reg_hp_safe.get("loss_name", str(best_reg_hp_safe["loss"]))

    metrics_all["mlp_regressor"] = {
        "seed": RANDOM_SEED,
        "scaler": "StandardScaler(train_fit_only)",
        "features_used": len(feat_cols),
        "best_hyperparams": best_reg_hp_safe,   # <-- BURASI değişti
        "val": reg_val_metrics,
        "test": reg_test_metrics,
        "model_path": OUT_MODEL_REG,
        "predictions_path": PRED_PATH,
        "figures": {
            "test_scatter": os.path.join(FIG_REG, "test_scatter.png"),
            "train_loss": os.path.join(FIG_REG, f"train_{best_reg_hp_safe['name']}_loss.png"),
        }
    }

    write_metrics_json(METRICS_PATH, metrics_all)
    print(f"[OK] metrics -> {METRICS_PATH}")

    print("\nDONE: MLP classifier + regressor\n")


if __name__ == "__main__":
    main()
