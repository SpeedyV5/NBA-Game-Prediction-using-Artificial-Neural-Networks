# src/models/sequence_models.py
import os, json, random
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, brier_score_loss
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 30

TRAIN_PATH = "data_processed/train_set.csv"
VAL_PATH   = "data_processed/val_set.csv"
TEST_PATH  = "data_processed/test_set.csv"

# outputs
OUT_MODEL = "models/sequence_lstm_classifier_best.h5"
PRED_PATH = "data_processed/predictions_sequence_lstm.csv"
METRICS_PATH = "reports/metrics/model_results.json"

# figures folder structure like baseline/mlp
FIG_BASE = os.path.join("reports", "figures", "sequence")
FIG_CLF_DIR = os.path.join(FIG_BASE, "classifier")


def set_seed(seed=RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data_processed", exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    os.makedirs(FIG_CLF_DIR, exist_ok=True)


def load_splits():
    return pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH), pd.read_csv(TEST_PATH)


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    IMPORTANT: numeric-only to avoid 'could not convert string to float: 2025-11-08'
    """
    drop_cols = [c for c in ["home_team_win", "score_diff"] if c in df.columns]
    for c in ["game_id", "match_id", "id"]:
        if c in df.columns:
            drop_cols.append(c)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in drop_cols]
    return feat_cols


def pick_col(df, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def can_build_real_sequences(df: pd.DataFrame) -> bool:
    cols = set([c.lower() for c in df.columns])
    has_date = any(c in cols for c in ["date", "game_date", "match_date"])
    has_home = any(c in cols for c in ["home_team", "home_team_id", "home_abbr"])
    has_away = any(c in cols for c in ["away_team", "away_team_id", "away_abbr"])
    return has_date and has_home and has_away


def prepare_tabular_scaled(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """
    - numeric-only features
    - drop all-NaN columns based on TRAIN
    - fill NaN with TRAIN median (no leakage)
    - StandardScaler fit only on train
    """
    feat_cols = infer_feature_cols(train)

    X_train = train[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_val   = val[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_test  = test[feat_cols].to_numpy(dtype=np.float32, copy=True)

    # 1) Drop all-NaN columns in TRAIN
    all_nan_mask = np.all(np.isnan(X_train), axis=0)
    if np.any(all_nan_mask):
        dropped = int(all_nan_mask.sum())
        print(f"[WARN] Dropping {dropped} all-NaN columns from train features")
        keep_mask = ~all_nan_mask
        X_train = X_train[:, keep_mask]
        X_val   = X_val[:, keep_mask]
        X_test  = X_test[:, keep_mask]
        feat_cols = [c for c, keep in zip(feat_cols, keep_mask) if keep]

    # 2) Fill NaNs using TRAIN median
    train_median = np.nanmedian(X_train, axis=0)
    train_median = np.where(np.isnan(train_median), 0.0, train_median)

    X_train = np.where(np.isnan(X_train), train_median, X_train)
    X_val   = np.where(np.isnan(X_val),   train_median, X_val)
    X_test  = np.where(np.isnan(X_test),  train_median, X_test)

    # 3) Scale (fit only on train)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)

    # put back into DF copies
    train_s = train.copy()
    val_s   = val.copy()
    test_s  = test.copy()
    train_s[feat_cols] = Xtr
    val_s[feat_cols]   = Xva
    test_s[feat_cols]  = Xte

    return train_s, val_s, test_s, scaler, feat_cols


def build_sequences_real(
    df_all: pd.DataFrame,
    df_split: pd.DataFrame,
    feat_cols: List[str],
    N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Real many-to-one sequence:
    For each game, use last N games for home + last N games for away, concat feature axis.
    Output: (samples, N, 2F)
    """
    date_col = pick_col(df_all, ["date", "game_date", "match_date"])
    home_col = pick_col(df_all, ["home_team", "home_team_id", "home_abbr"])
    away_col = pick_col(df_all, ["away_team", "away_team_id", "away_abbr"])

    if date_col is None or home_col is None or away_col is None:
        # fallback safety
        return build_sequences_fallback(df_split, feat_cols, N)

    df_all2 = df_all.copy()
    df_all2[date_col] = pd.to_datetime(df_all2[date_col], errors="coerce")
    df_all2 = df_all2.sort_values(date_col).reset_index(drop=True)

    df_split2 = df_split.copy()
    df_split2[date_col] = pd.to_datetime(df_split2[date_col], errors="coerce")
    df_split2 = df_split2.sort_values(date_col).reset_index(drop=True)

    y = df_split2["home_team_win"].values.astype(int)

    # bucket split rows by date
    bucket: Dict[pd.Timestamp, List[Tuple[int, pd.Series]]] = {}
    for idx, r in df_split2.iterrows():
        bucket.setdefault(r[date_col], []).append((idx, r))

    team_hist: Dict[str, List[np.ndarray]] = {}
    X_seq: List[Optional[np.ndarray]] = [None] * len(df_split2)

    def get_last_n(hist: List[np.ndarray], n: int, fdim: int) -> np.ndarray:
        if len(hist) >= n:
            return np.array(hist[-n:], dtype=np.float32)
        pad = np.zeros((n - len(hist), fdim), dtype=np.float32)
        if len(hist) == 0:
            return pad
        return np.vstack([pad, np.array(hist, dtype=np.float32)])

    F = len(feat_cols)

    for _, row in df_all2.iterrows():
        d = row[date_col]

        # compute sequences for split rows at this date BEFORE adding current match -> no same-game leakage
        if d in bucket:
            for idx, r in bucket[d]:
                h = str(r[home_col])
                a = str(r[away_col])
                h_hist = team_hist.get(h, [])
                a_hist = team_hist.get(a, [])
                h_seq = get_last_n(h_hist, N, F)
                a_seq = get_last_n(a_hist, N, F)
                X_seq[idx] = np.concatenate([h_seq, a_seq], axis=1)  # (N, 2F)

        # update history (use this row's scaled numeric features)
        feat_vec = row[feat_cols].to_numpy(dtype=np.float32, copy=True)
        hteam = str(row[home_col])
        ateam = str(row[away_col])
        team_hist.setdefault(hteam, []).append(feat_vec)
        team_hist.setdefault(ateam, []).append(feat_vec)

    for i in range(len(X_seq)):
        if X_seq[i] is None:
            X_seq[i] = np.zeros((N, 2 * F), dtype=np.float32)

    X_seq_arr = np.stack(X_seq, axis=0)
    return X_seq_arr, y


def build_sequences_fallback(
    df_split: pd.DataFrame,
    feat_cols: List[str],
    N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback: repeat tabular row N times => (samples, N, F)
    """
    X = df_split[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X_seq = np.repeat(X[:, None, :], repeats=N, axis=1)
    y = df_split["home_team_win"].values.astype(int)
    return X_seq, y


def build_model(
    seq_len: int,
    feat_dim: int,
    model_type: str = "lstm",
    lr: float = 1e-3,
    clipnorm: float = 1.0
) -> keras.Model:
    inp = keras.Input(shape=(seq_len, feat_dim))
    if model_type == "gru":
        x = layers.GRU(64, return_sequences=False)(inp)
    else:
        x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out, name=f"{model_type}_classifier")
    opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")]
    )
    return model


def eval_classifier(y_true, y_prob) -> Dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    # sklearn compatibility (no eps param)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob_clipped)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_history(history, fig_prefix: str):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_loss.png", dpi=150)
    plt.close()

    if "acc" in history.history:
        plt.figure()
        plt.plot(history.history["acc"], label="train_acc")
        plt.plot(history.history["val_acc"], label="val_acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{fig_prefix}_accuracy.png", dpi=150)
        plt.close()

    if "auc" in history.history:
        plt.figure()
        plt.plot(history.history["auc"], label="train_auc")
        plt.plot(history.history["val_auc"], label="val_auc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{fig_prefix}_auc.png", dpi=150)
        plt.close()


def plot_classification_diagnostics(y_true, y_prob, out_prefix: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # roc
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=150)
    plt.close()

    # calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("mean predicted probability")
    plt.ylabel("fraction of positives")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_calibration.png", dpi=150)
    plt.close()


def read_metrics_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metrics_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    set_seed(RANDOM_SEED)
    ensure_dirs()

    train, val, test = load_splits()

    # scale numeric-only features + NaN handling
    train_s, val_s, test_s, scaler, feat_cols = prepare_tabular_scaled(train, val, test)

    use_real = (
        can_build_real_sequences(train_s) and
        can_build_real_sequences(val_s) and
        can_build_real_sequences(test_s)
    )

    # build candidates
    candidates = []
    for N in [5, 10]:
        if use_real:
            df_all = pd.concat([train_s, val_s, test_s], ignore_index=True)
            X_train_seq, y_train = build_sequences_real(df_all, train_s, feat_cols, N)
            X_val_seq, y_val = build_sequences_real(df_all, val_s, feat_cols, N)
            X_test_seq, y_test = build_sequences_real(df_all, test_s, feat_cols, N)
            feat_dim = X_train_seq.shape[-1]
        else:
            X_train_seq, y_train = build_sequences_fallback(train_s, feat_cols, N)
            X_val_seq, y_val = build_sequences_fallback(val_s, feat_cols, N)
            X_test_seq, y_test = build_sequences_fallback(test_s, feat_cols, N)
            feat_dim = len(feat_cols)

        for model_type in ["lstm"]:  # add "gru" optionally
            candidates.append((N, model_type, X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, feat_dim))

    best_auc = -1.0
    best_spec = None  # (N, model_type, use_real)

    for N, model_type, X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, feat_dim in candidates:
        tf.keras.backend.clear_session()
        model = build_model(seq_len=N, feat_dim=feat_dim, model_type=model_type, lr=1e-3, clipnorm=1.0)

        best_auc = -1.0
        best_spec = None          # (N, model_type, use_real)
        best_model_path = None    # <-- NEW

        for N, model_type, X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, feat_dim in candidates:
            tf.keras.backend.clear_session()
            model = build_model(seq_len=N, feat_dim=feat_dim, model_type=model_type, lr=1e-3, clipnorm=1.0)

            # candidate-specific checkpoint (NO overwrite)
            ckpt_path = os.path.join("models", f"sequence_{model_type}_N{N}_best.h5")

            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True),
            ]

            history = model.fit(
                X_train_seq, y_train,
                validation_data=(X_val_seq, y_val),
                epochs=100,
                batch_size=128,
                callbacks=callbacks,
                verbose=2
            )

            val_prob = model.predict(X_val_seq, verbose=0).reshape(-1)
            val_auc = float(roc_auc_score(y_val, val_prob))
            print(f"[SEQ N={N} {model_type}] val_auc={val_auc:.4f} (real_seq={use_real})")

            curve_prefix = os.path.join(FIG_CLF_DIR, f"train_{model_type}_N{N}")
            plot_history(history, curve_prefix)

            if val_auc > best_auc:
                best_auc = val_auc
                best_spec = (N, model_type, use_real)
                best_model_path = ckpt_path  # <-- NEW

        if best_spec is None or best_model_path is None:
            raise RuntimeError("No candidate model was trained or best checkpoint missing.")

        # Load the correct best checkpoint
        best_model = keras.models.load_model(best_model_path)

        # Optional: also save/copy it to required OUT_MODEL name for consistency
        try:
            best_model.save(OUT_MODEL)
        except Exception:
            pass
        
        val_prob = model.predict(X_val_seq, verbose=0).reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob))
        print(f"[SEQ N={N} {model_type}] val_auc={val_auc:.4f} (real_seq={use_real})")

        # save curves under sequence/classifier with a clear name
        curve_prefix = os.path.join(FIG_CLF_DIR, f"train_{model_type}_N{N}")
        plot_history(history, curve_prefix)

        if val_auc > best_auc:
            best_auc = val_auc
            best_spec = (N, model_type, use_real)

    if best_spec is None:
        raise RuntimeError("No candidate model was trained.")

    best_N, best_type, best_real = best_spec
    best_model = keras.models.load_model(OUT_MODEL)

    # rebuild best sequences for final predictions
    if best_real:
        df_all = pd.concat([train_s, val_s, test_s], ignore_index=True)
        X_train_seq, y_train = build_sequences_real(df_all, train_s, feat_cols, best_N)
        X_val_seq, y_val = build_sequences_real(df_all, val_s, feat_cols, best_N)
        X_test_seq, y_test = build_sequences_real(df_all, test_s, feat_cols, best_N)
    else:
        X_train_seq, y_train = build_sequences_fallback(train_s, feat_cols, best_N)
        X_val_seq, y_val = build_sequences_fallback(val_s, feat_cols, best_N)
        X_test_seq, y_test = build_sequences_fallback(test_s, feat_cols, best_N)

    train_prob = best_model.predict(X_train_seq, verbose=0).reshape(-1)
    val_prob = best_model.predict(X_val_seq, verbose=0).reshape(-1)
    test_prob = best_model.predict(X_test_seq, verbose=0).reshape(-1)

    val_metrics = eval_classifier(y_val, val_prob)
    test_metrics = eval_classifier(y_test, test_prob)

    # test diagnostics figs
    test_prefix = os.path.join(FIG_CLF_DIR, f"test_{best_type}_N{best_N}")
    plot_classification_diagnostics(y_test, test_prob, test_prefix)

    # predictions csv
    preds = pd.concat([
        pd.DataFrame({"split": "train", "y_true": y_train, "y_prob": train_prob, "y_pred": (train_prob >= 0.5).astype(int)}),
        pd.DataFrame({"split": "val", "y_true": y_val, "y_prob": val_prob, "y_pred": (val_prob >= 0.5).astype(int)}),
        pd.DataFrame({"split": "test", "y_true": y_test, "y_prob": test_prob, "y_pred": (test_prob >= 0.5).astype(int)}),
    ], ignore_index=True)

    preds.to_csv(PRED_PATH, index=False)
    print(f"[OK] predictions -> {PRED_PATH}")

    # metrics json update
    metrics_all = read_metrics_json(METRICS_PATH)
    metrics_all["sequence_lstm_classifier"] = {
        "seed": RANDOM_SEED,
        "scaler": "StandardScaler(train_fit_only)",
        "features_used": int(len(feat_cols)),
        "sequence": {
            "N": int(best_N),
            "real_sequence_used": bool(best_real),
            "model_type": best_type,
            "note": "If real_sequence_used=False, fallback repeats tabular rows as sequences."
        },
        "val": val_metrics,
        "test": test_metrics,
        "model_path": OUT_MODEL,
        "predictions_path": PRED_PATH,
        "figures": {
            "curves": {
                "loss": os.path.join(FIG_CLF_DIR, f"train_{best_type}_N{best_N}_loss.png"),
                "accuracy": os.path.join(FIG_CLF_DIR, f"train_{best_type}_N{best_N}_accuracy.png"),
                "auc": os.path.join(FIG_CLF_DIR, f"train_{best_type}_N{best_N}_auc.png"),
            },
            "test": {
                "confusion_matrix": f"{test_prefix}_confusion_matrix.png",
                "roc": f"{test_prefix}_roc.png",
                "calibration": f"{test_prefix}_calibration.png",
            }
        }
    }

    write_metrics_json(METRICS_PATH, metrics_all)
    print(f"[OK] metrics updated -> {METRICS_PATH}")
    print("\nDONE: sequence model\n")


if __name__ == "__main__":
    main()
