# src/models/mlp.py
import os
import json
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss, mean_absolute_error,
    mean_squared_error
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


RANDOM_SEED = 30
TOP_N_FEATURES = 50  # Feature selection target

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
FEATURE_SELECTION_LOG = "models/feature_selection_log.json"


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


def select_features_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feat_cols: List[str],
    n_features: int = TOP_N_FEATURES,
    task_type: str = "classification"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Use RandomForest to select top N most important features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feat_cols: Original feature column names
        n_features: Number of features to select
        task_type: "classification" or "regression"
    
    Returns:
        X_train_selected, X_val_selected, X_test_selected, selected_feat_cols
    """
    print(f"\n=== FEATURE SELECTION (RandomForest) ===")
    print(f"Original features: {len(feat_cols)}")
    print(f"Target features: {n_features}")
    
    # Fit RandomForest to get feature importances
    if task_type == "classification":
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=0
        )
    else:  # regression
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=0
        )
    
    print("Fitting RandomForest for feature importance...")
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:n_features]
    selected_feat_cols = [feat_cols[i] for i in top_indices]
    
    # Log selected features
    feature_importance_log = {
        "n_original": len(feat_cols),
        "n_selected": len(selected_feat_cols),
        "selected_features": selected_feat_cols,
        "importances": {feat_cols[i]: float(importances[i]) for i in top_indices}
    }
    
    os.makedirs("models", exist_ok=True)
    with open(FEATURE_SELECTION_LOG, "w", encoding="utf-8") as f:
        json.dump(feature_importance_log, f, indent=2, ensure_ascii=False)
    
    print(f"Selected {len(selected_feat_cols)} features")
    print(f"Top 10 features by importance:")
    for i, idx in enumerate(top_indices[:10]):
        print(f"  {i+1}. {feat_cols[idx]}: {importances[idx]:.4f}")
    print(f"Feature selection log saved to: {FEATURE_SELECTION_LOG}")
    print("=" * 50)
    
    # Return indices for feature selection
    return top_indices, selected_feat_cols


def prepare_tabular(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    use_feature_selection: bool = True,
    n_features: int = TOP_N_FEATURES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler, List[str], np.ndarray]:
    """
    Prepare tabular data with optional feature selection.
    """
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

    # 3) Feature Selection (before scaling)
    if use_feature_selection and len(feat_cols) > n_features:
        # Use classification target for feature selection (more balanced)
        y_train_clf = train["home_team_win"].values.astype(int)
        selected_indices, selected_feat_cols = select_features_rf(
            X_train, y_train_clf, feat_cols, n_features=n_features, task_type="classification"
        )
        
        # Apply feature selection
        X_train = X_train[:, selected_indices]
        X_val   = X_val[:, selected_indices]
        X_test  = X_test[:, selected_indices]
        feat_cols = selected_feat_cols
        train_median = train_median[selected_indices]
        print(f"[INFO] After feature selection: {X_train.shape[1]} features")

    # 4) Scale using RobustScaler (fit only on train) - better for outliers
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, scaler, feat_cols, train_median


def build_mlp_classifier(input_dim: int, hp: Dict, class_weights: Optional[Dict] = None) -> keras.Model:
    """
    Build MLP classifier with improved architecture.
    
    Args:
        input_dim: Input feature dimension
        hp: Hyperparameters dict
        class_weights: Optional class weights dict for imbalanced data
    """
    inp = keras.Input(shape=(input_dim,))
    x = inp
    
    # Add initial dropout for regularization
    if hp.get("input_dropout", 0.0) > 0:
        x = layers.Dropout(hp["input_dropout"])(x)
    
    for i, units in enumerate(hp["hidden_units"]):
        x = layers.Dense(units, activation="relu")(x)
        if hp.get("batchnorm", False):
            x = layers.BatchNormalization()(x)
        if hp.get("dropout", 0.0) > 0:
            x = layers.Dropout(hp["dropout"])(x)
    
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="mlp_classifier")

    opt = keras.optimizers.Adam(learning_rate=hp["lr"])
    
    # Use weighted binary crossentropy if class weights provided
    if class_weights is not None:
        # TensorFlow expects class weights as a list [weight_for_class_0, weight_for_class_1]
        loss_fn = keras.losses.BinaryCrossentropy()
        # We'll handle class weights in the fit() call
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")]
        )
    else:
        model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")]
        )
    
    return model


def build_mlp_regressor(input_dim: int, hp: Dict) -> keras.Model:
    """
    Build MLP regressor with improved architecture.
    
    Args:
        input_dim: Input feature dimension
        hp: Hyperparameters dict
    """
    inp = keras.Input(shape=(input_dim,))
    x = inp
    
    # Add initial dropout for regularization
    if hp.get("input_dropout", 0.0) > 0:
        x = layers.Dropout(hp["input_dropout"])(x)
    
    for i, units in enumerate(hp["hidden_units"]):
        x = layers.Dense(units, activation="relu")(x)
        if hp.get("batchnorm", False):
            x = layers.BatchNormalization()(x)
        if hp.get("dropout", 0.0) > 0:
            x = layers.Dropout(hp["dropout"])(x)
    
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out, name="mlp_regressor")

    opt = keras.optimizers.Adam(learning_rate=hp["lr"])
    
    # Use MAE or Huber loss instead of MSE
    loss = hp.get("loss", "mae")
    if loss == "mae":
        loss_fn = keras.losses.MeanAbsoluteError()
    elif loss == "huber":
        delta = hp.get("huber_delta", 1.0)
        loss_fn = keras.losses.Huber(delta=delta)
    else:
        loss_fn = loss  # Allow custom loss functions
    
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def compute_class_weights_dict(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced binary classification.
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


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

    # Prepare data with feature selection
    X_train, X_val, X_test, scaler, feat_cols, train_median = prepare_tabular(
        train, val, test, use_feature_selection=True, n_features=TOP_N_FEATURES
    )

    # === SAVE INFERENCE ARTIFACTS (TABULAR) ===
    import joblib

    os.makedirs("models", exist_ok=True)

    # 1) feature list
    with open("models/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: models/feature_cols.json (n={len(feat_cols)})")

    # 2) scaler (RobustScaler for input features)
    joblib.dump(scaler, "models/scaler.joblib")
    print("[OK] Saved: models/scaler.joblib (RobustScaler)")

    # 3) train median (NaN fill için)
    joblib.dump(train_median, "models/train_median.joblib")
    print("[OK] Saved: models/train_median.joblib")

    # -------------------------
    # B) MLP CLASSIFIER
    # -------------------------
    y_train = train["home_team_win"].values.astype(int)
    y_val   = val["home_team_win"].values.astype(int)
    y_test  = test["home_team_win"].values.astype(int)

    # Compute class weights for home-court bias
    class_weights_dict = compute_class_weights_dict(y_train)
    print(f"\n[INFO] Class weights: {class_weights_dict}")
    print(f"  Class 0 (away win): weight = {class_weights_dict[0]:.3f}")
    print(f"  Class 1 (home win): weight = {class_weights_dict[1]:.3f}")

    clf_variants = [
        {
            "name": "MLP_C1",
            "hidden_units": [256, 128, 64],
            "dropout": 0.3,
            "input_dropout": 0.1,
            "batchnorm": True,
            "lr": 1e-3
        },
        {
            "name": "MLP_C2",
            "hidden_units": [128, 64, 32],
            "dropout": 0.2,
            "input_dropout": 0.05,
            "batchnorm": True,
            "lr": 5e-4
        },
        {
            "name": "MLP_C3",
            "hidden_units": [512, 256, 128, 64],
            "dropout": 0.4,
            "input_dropout": 0.15,
            "batchnorm": True,
            "lr": 1e-3
        },
    ]

    best_val_auc = -1
    best_hp = None
    best_model_path = OUT_MODEL_CLF

    for hp in clf_variants:
        tf.keras.backend.clear_session()
        model = build_mlp_classifier(X_train.shape[1], hp, class_weights=class_weights_dict)

        ckpt_path = best_model_path  # required name
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=256,
            callbacks=callbacks,
            class_weight=class_weights_dict,  # Apply class weights
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
    
    # Remove NaN values from score_diff (drop rows with NaN for training only)
    train_mask = ~np.isnan(y_train_r)
    val_mask = ~np.isnan(y_val_r)
    test_mask = ~np.isnan(y_test_r)
    
    # Keep original arrays for predictions CSV, use filtered for training
    X_train_reg = X_train[train_mask] if not train_mask.all() else X_train
    y_train_r_reg = y_train_r[train_mask] if not train_mask.all() else y_train_r
    X_val_reg = X_val[val_mask] if not val_mask.all() else X_val
    y_val_r_reg = y_val_r[val_mask] if not val_mask.all() else y_val_r
    X_test_reg = X_test[test_mask] if not test_mask.all() else X_test
    y_test_r_reg = y_test_r[test_mask] if not test_mask.all() else y_test_r
    
    if not train_mask.all():
        print(f"[WARN] Dropping {np.sum(~train_mask)} rows with NaN score_diff from train for regressor")
    if not val_mask.all():
        print(f"[WARN] Dropping {np.sum(~val_mask)} rows with NaN score_diff from val for regressor")
    if not test_mask.all():
        print(f"[WARN] Dropping {np.sum(~test_mask)} rows with NaN score_diff from test for regressor")
    
    # === TARGET SCALING FOR REGRESSION ===
    # Create and fit target scaler on training data (reshaped to 2D)
    target_scaler = StandardScaler()
    y_train_r_reg_scaled = target_scaler.fit_transform(y_train_r_reg.reshape(-1, 1)).reshape(-1)
    y_val_r_reg_scaled = target_scaler.transform(y_val_r_reg.reshape(-1, 1)).reshape(-1)
    y_test_r_reg_scaled = target_scaler.transform(y_test_r_reg.reshape(-1, 1)).reshape(-1)
    
    print(f"[INFO] Target scaler fitted: mean={target_scaler.mean_[0]:.2f}, scale={target_scaler.scale_[0]:.2f}")
    
    # Save target scaler
    joblib.dump(target_scaler, "models/scaler_target.joblib")
    print("[OK] Saved: models/scaler_target.joblib")

    reg_variants = [
        {
            "name": "MLP_R1",
            "hidden_units": [256, 128, 64],
            "dropout": 0.3,
            "input_dropout": 0.1,
            "batchnorm": True,
            "lr": 1e-3,
            "loss": "mae",
            "loss_name": "mae"
        },
        {
            "name": "MLP_R2",
            "hidden_units": [256, 128, 64],
            "dropout": 0.3,
            "input_dropout": 0.1,
            "batchnorm": True,
            "lr": 1e-3,
            "loss": "huber",
            "huber_delta": 2.0,
            "loss_name": "huber(delta=2.0)"
        },
    ]

    best_val_rmse = 1e18
    best_reg_hp = None

    for hp in reg_variants:
        tf.keras.backend.clear_session()
        model = build_mlp_regressor(X_train_reg.shape[1], hp)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                OUT_MODEL_REG,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = model.fit(
            X_train_reg, y_train_r_reg_scaled,
            validation_data=(X_val_reg, y_val_r_reg_scaled),
            epochs=200,
            batch_size=256,
            callbacks=callbacks,
            verbose=2
        )

        # Predict on scaled target, then inverse transform for evaluation
        val_pred_scaled = model.predict(X_val_reg, verbose=0).reshape(-1)
        val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).reshape(-1)
        
        val_rmse = float(np.sqrt(mean_squared_error(y_val_r_reg, val_pred)))
        val_mae = float(mean_absolute_error(y_val_r_reg, val_pred))
        print(f"[{hp['name']}] val_rmse={val_rmse:.4f}, val_mae={val_mae:.4f}")

        plot_and_save_history(history, os.path.join(FIG_REG, f"train_{hp['name']}"), is_classifier=False)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_reg_hp = hp

    reg_best = keras.models.load_model(OUT_MODEL_REG)

    # Use filtered data for metrics (predict on scaled, then inverse transform)
    val_pred_r_filtered_scaled = reg_best.predict(X_val_reg, verbose=0).reshape(-1)
    test_pred_r_filtered_scaled = reg_best.predict(X_test_reg, verbose=0).reshape(-1)
    
    # Inverse transform to original scale
    val_pred_r_filtered = target_scaler.inverse_transform(val_pred_r_filtered_scaled.reshape(-1, 1)).reshape(-1)
    test_pred_r_filtered = target_scaler.inverse_transform(test_pred_r_filtered_scaled.reshape(-1, 1)).reshape(-1)

    reg_val_metrics = eval_regression(y_val_r_reg, val_pred_r_filtered)
    reg_test_metrics = eval_regression(y_test_r_reg, test_pred_r_filtered)
    
    # For predictions CSV, predict on all data (including NaN rows)
    val_pred_r_scaled = reg_best.predict(X_val, verbose=0).reshape(-1)
    test_pred_r_scaled = reg_best.predict(X_test, verbose=0).reshape(-1)
    
    # Inverse transform all predictions
    val_pred_r = target_scaler.inverse_transform(val_pred_r_scaled.reshape(-1, 1)).reshape(-1)
    test_pred_r = target_scaler.inverse_transform(test_pred_r_scaled.reshape(-1, 1)).reshape(-1)

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

    # For train predictions, also need to inverse transform
    train_pred_r_scaled = reg_best.predict(X_train, verbose=0).reshape(-1)
    train_pred_r = target_scaler.inverse_transform(train_pred_r_scaled.reshape(-1, 1)).reshape(-1)
    
    pred_train = make_pred_df(
        "train",
        y_train,
        clf_best.predict(X_train, verbose=0).reshape(-1),
        y_train_r,
        train_pred_r,
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
        "scaler": "RobustScaler(train_fit_only)",
        "features_used": len(feat_cols),
        "feature_selection": "RandomForest (top 50)",
        "class_weights": class_weights_dict,
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
        best_reg_hp_safe["loss"] = best_reg_hp_safe.get("loss_name", str(best_reg_hp_safe["loss"]))

    metrics_all["mlp_regressor"] = {
        "seed": RANDOM_SEED,
        "scaler": "RobustScaler(train_fit_only)",
        "target_scaler": "StandardScaler(train_fit_only)",
        "features_used": len(feat_cols),
        "feature_selection": "RandomForest (top 50)",
        "best_hyperparams": best_reg_hp_safe,
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
    print(f"Classifier Test AUC: {clf_test_metrics['roc_auc']:.4f}")
    print(f"Regressor Test MAE: {reg_test_metrics['mae']:.4f}")
    print(f"Regressor Test RMSE: {reg_test_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
