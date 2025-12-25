# src/inference/predict_today.py
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Optional deps
try:
    import joblib
except Exception:
    joblib = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


# -----------------------------
# Paths / Config
# -----------------------------
@dataclass
class ProjectPaths:
    repo_root: Path

    notebooks_dir: Path
    models_dir: Path
    src_dir: Path

    data_processed_dir: Path
    data_interim_dir: Path
    data_raw_dir: Path

    # Current-data notebook outputs
    rest_days_stats_csv: Path
    schedule_rest_days_csv: Path
    player_stats_raw_csv: Path
    team_stats_raw_csv: Path

    # Injury outputs
    latest_injury_team_status_csv: Path

    # Outputs for today predictions
    predictions_today_csv: Path

    # Training artifacts
    feature_cols_json: Path
    scaler_joblib: Path
    scaler_target_joblib: Path  # <--- EKLENDI
    train_median_joblib: Path
    mlp_model_clf: Path         # <--- ISIM DEGISTI
    mlp_model_reg: Path         # <--- EKLENDI

    # Injury script + notebook
    download_injury_script: Path
    nb_current_data: Path


def resolve_paths() -> ProjectPaths:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]

    notebooks_dir = repo_root / "notebooks"
    models_dir = repo_root / "models"
    src_dir = repo_root / "src"

    data_processed_dir = repo_root / "data_processed"
    data_interim_dir = repo_root / "data_interim"
    data_raw_dir = repo_root / "data_raw"

    rest_days_stats_csv = data_raw_dir / "nbastuffer_2025_2026_rest_days_stats.csv"
    schedule_rest_days_csv = data_raw_dir / "nbastuffer_2025_2026_schedule_rest_days.csv"
    player_stats_raw_csv = data_raw_dir / "nbastuffer_2025_2026_player_stats_raw.csv"
    team_stats_raw_csv = data_raw_dir / "nbastuffer_2025_2026_team_stats_raw.csv"

    # produced by injury pipeline
    latest_injury_team_status_csv = data_raw_dir / "injury_reports_raw" / "latest_injury_team_status.csv"

    predictions_today_csv = data_processed_dir / "predictions_today.csv"

    feature_cols_json = models_dir / "feature_cols.json"
    scaler_joblib = models_dir / "scaler.joblib"
    scaler_target_joblib = models_dir / "scaler_target.joblib" # <--- Target Scaler
    train_median_joblib = models_dir / "train_median.joblib"
    
    mlp_model_clf = models_dir / "mlp_classifier_best.h5"
    mlp_model_reg = models_dir / "mlp_regressor_best.h5"       # <--- Regressor Model

    download_injury_script = src_dir / "data" / "download_injury_report.py"
    nb_current_data = notebooks_dir / "01_current_data_download.ipynb"

    return ProjectPaths(
        repo_root=repo_root,
        notebooks_dir=notebooks_dir,
        models_dir=models_dir,
        src_dir=src_dir,
        data_processed_dir=data_processed_dir,
        data_interim_dir=data_interim_dir,
        data_raw_dir=data_raw_dir,
        rest_days_stats_csv=rest_days_stats_csv,
        schedule_rest_days_csv=schedule_rest_days_csv,
        player_stats_raw_csv=player_stats_raw_csv,
        team_stats_raw_csv=team_stats_raw_csv,
        latest_injury_team_status_csv=latest_injury_team_status_csv,
        predictions_today_csv=predictions_today_csv,
        feature_cols_json=feature_cols_json,
        scaler_joblib=scaler_joblib,
        scaler_target_joblib=scaler_target_joblib,
        train_median_joblib=train_median_joblib,
        mlp_model_clf=mlp_model_clf,
        mlp_model_reg=mlp_model_reg,
        download_injury_script=download_injury_script,
        nb_current_data=nb_current_data,
    )


# -----------------------------
# Utils
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def has_module(mod_name: str) -> bool:
    return importlib.util.find_spec(mod_name) is not None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def today_str(date_override: Optional[str] = None) -> str:
    return date_override or datetime.now().strftime("%Y-%m-%d")


def run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> None:
    log(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout.strip():
            log("[STDOUT]\n" + proc.stdout)
        if proc.stderr.strip():
            log("[STDERR]\n" + proc.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    if proc.stdout.strip():
        log("[STDOUT]\n" + proc.stdout)
    if proc.stderr.strip():
        log("[STDERR]\n" + proc.stderr)


def run_notebook_nbconvert(nb_path: Path, *, cwd: Path) -> None:
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]
    run_subprocess(cmd, cwd=cwd)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    repo_root = resolve_paths().repo_root
    alt1 = repo_root / "data_raw" / path.name
    if alt1.exists():
        return pd.read_csv(alt1)
    alt2 = repo_root / path.name
    if alt2.exists():
        return pd.read_csv(alt2)
    raise FileNotFoundError(f"Missing CSV: {path} (also tried {alt1} and {alt2})")


def _sanitize_stat_col(col: str) -> str:
    s = str(col).strip()
    s = s.replace("%", "pct")
    s = s.replace(".", "")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# -----------------------------
# Today games (from rest_days_stats OPPONENT TODAY)
# -----------------------------
def load_today_games_from_rest_days(paths: ProjectPaths) -> pd.DataFrame:
    df = read_csv_safe(paths.rest_days_stats_csv)

    team_col = _first_existing_col(df, ["TEAM NAME", "TEAM", "Team"])
    opp_col = _first_existing_col(df, ["OPPONENT TODAY", "OPPONENT_TODAY", "Opponent Today"])
    if team_col is None or opp_col is None:
        raise ValueError(
            f"rest_days_stats CSV must include TEAM NAME and OPPONENT TODAY. Found: {list(df.columns)}"
        )

    sub = df[[team_col, opp_col]].copy()
    sub[opp_col] = sub[opp_col].astype(str).str.strip()
    sub = sub[sub[opp_col].notna() & (sub[opp_col].str.lower() != "nan")]

    if sub.empty:
        log("[WARN] rest_days_stats has no OPPONENT TODAY rows. No games detected.")
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    # Use only "vs." as canonical home rows (dedup)
    home_rows = sub[sub[opp_col].str.lower().str.startswith("vs")].copy()
    if home_rows.empty:
        log("[WARN] No 'vs.' rows found; cannot reliably infer home/away. Returning empty.")
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    def parse_opp(s: str) -> str:
        s = s.strip()
        s = s.replace("vs.", "", 1).replace("vs", "", 1)
        return s.strip()

    home_rows["home_team"] = home_rows[team_col].astype(str).str.strip()
    home_rows["away_team"] = home_rows[opp_col].map(parse_opp)

    out = home_rows[["home_team", "away_team"]].dropna().drop_duplicates().reset_index(drop=True)
    out["date"] = today_str()
    return out[["date", "home_team", "away_team"]]


# -----------------------------
# Injury aggregation
# -----------------------------
def load_injury_counts(paths: ProjectPaths) -> pd.DataFrame:
    try:
        df = read_csv_safe(paths.latest_injury_team_status_csv)
    except Exception as e:
        log(f"[INFO] No latest_injury_team_status.csv available; injury features = 0. Reason: {e}")
        return pd.DataFrame(columns=["TEAM", "injury_count"])

    team_col = _first_existing_col(df, ["TEAM", "TEAM NAME", "Team"])
    if team_col is None:
        return pd.DataFrame(columns=["TEAM", "injury_count"])

    out_col = _first_existing_col(df, ["inj_out", "OUT"])
    q_col = _first_existing_col(df, ["inj_questionable", "QUESTIONABLE"])
    d_col = _first_existing_col(df, ["inj_doubtful", "DOUBTFUL"])

    if out_col is None and q_col is None and d_col is None:
        return pd.DataFrame(columns=["TEAM", "injury_count"])

    tmp = df.copy()
    for c in [out_col, q_col, d_col]:
        if c is not None:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0).astype(int)

    any_inj = np.zeros(len(tmp), dtype=int)
    for c in [out_col, q_col, d_col]:
        if c is not None:
            any_inj |= (tmp[c].values.astype(int) > 0).astype(int)

    tmp["any_injury"] = any_inj
    agg = tmp.groupby(team_col, as_index=False)["any_injury"].sum()
    agg = agg.rename(columns={team_col: "TEAM", "any_injury": "injury_count"})
    agg["TEAM"] = agg["TEAM"].astype(str).str.strip()
    return agg


# -----------------------------
# Feature builder
# -----------------------------
def build_features_for_today(paths: ProjectPaths, games_today: pd.DataFrame) -> pd.DataFrame:
    if games_today.empty:
        return games_today.copy()

    team_stats = read_csv_safe(paths.team_stats_raw_csv)
    rest_stats = read_csv_safe(paths.rest_days_stats_csv)
    sched_stats = read_csv_safe(paths.schedule_rest_days_csv)
    inj_counts = load_injury_counts(paths)

    team_teamcol = _first_existing_col(team_stats, ["TEAM", "TEAM NAME", "Team"])
    rest_teamcol = _first_existing_col(rest_stats, ["TEAM NAME", "TEAM", "Team"])
    sched_teamcol = _first_existing_col(sched_stats, ["TEAMS", "TEAM", "TEAM NAME", "Team"])

    if team_teamcol is None or rest_teamcol is None or sched_teamcol is None:
        raise ValueError("Team name columns missing in stats files.")

    def prep_numeric(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
        out = df.copy()
        out[team_col] = out[team_col].astype(str).str.strip()
        num = out.select_dtypes(include=[np.number]).copy()
        num[team_col] = out[team_col]
        rename_map = {c: _sanitize_stat_col(c) for c in num.columns if c != team_col}
        num = num.rename(columns=rename_map)
        return num

    team_num = prep_numeric(team_stats, team_teamcol)
    rest_num = prep_numeric(rest_stats, rest_teamcol)
    sched_num = prep_numeric(sched_stats, sched_teamcol)

    if not inj_counts.empty:
        inj_counts = inj_counts.copy()
        inj_counts["TEAM"] = inj_counts["TEAM"].astype(str).str.strip()

    base = games_today.copy()
    base["home_team"] = base["home_team"].astype(str).str.strip()
    base["away_team"] = base["away_team"].astype(str).str.strip()

    home_team = team_num.add_prefix("home_team_").rename(columns={f"home_team_{team_teamcol}": "home_team"})
    away_team = team_num.add_prefix("away_team_").rename(columns={f"away_team_{team_teamcol}": "away_team"})
    base = base.merge(home_team, on="home_team", how="left")
    base = base.merge(away_team, on="away_team", how="left")

    home_rest = rest_num.add_prefix("home_rest_").rename(columns={f"home_rest_{rest_teamcol}": "home_team"})
    away_rest = rest_num.add_prefix("away_rest_").rename(columns={f"away_rest_{rest_teamcol}": "away_team"})
    base = base.merge(home_rest, on="home_team", how="left")
    base = base.merge(away_rest, on="away_team", how="left")

    home_sched = sched_num.add_prefix("home_schedule_").rename(columns={f"home_schedule_{sched_teamcol}": "home_team"})
    away_sched = sched_num.add_prefix("away_schedule_").rename(columns={f"away_schedule_{sched_teamcol}": "away_team"})
    base = base.merge(home_sched, on="home_team", how="left")
    base = base.merge(away_sched, on="away_team", how="left")

    if not inj_counts.empty:
        inj_h = inj_counts.rename(columns={"TEAM": "home_team", "injury_count": "injury_count_home"}).copy()
        inj_a = inj_counts.rename(columns={"TEAM": "away_team", "injury_count": "injury_count_away"}).copy()
        
        for c in ["injury_count_home", "injury_count_away"]:
            if c in base.columns: base = base.drop(columns=[c])

        base = base.merge(inj_h, on="home_team", how="left")
        base = base.merge(inj_a, on="away_team", how="left")

    if "injury_count_home" not in base.columns: base["injury_count_home"] = 0.0
    if "injury_count_away" not in base.columns: base["injury_count_away"] = 0.0

    base["injury_count_home"] = pd.to_numeric(base["injury_count_home"], errors="coerce").fillna(0.0)
    base["injury_count_away"] = pd.to_numeric(base["injury_count_away"], errors="coerce").fillna(0.0)

    # === ADD MISSING FEATURES TO MATCH TRAINING ===
    # Training expects 203 features, we only created 121
    # Missing: gameId, date features, ELO, diff features, rolling features
    
    # 1) gameId (required by training)
    if "gameId" not in base.columns:
        base["gameId"] = base.index.astype(str)  # Temporary ID
    
    # 2) Date features (from game_date or date column)
    if "game_date" not in base.columns and "date" in base.columns:
        base["game_date"] = pd.to_datetime(base["date"], errors="coerce")
    
    if "game_date" in base.columns:
        base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")
        base["month"] = base["game_date"].dt.month
        base["day_of_week"] = base["game_date"].dt.dayofweek
        base["is_weekend"] = (base["day_of_week"] >= 5).astype(int)
        base["is_playoff"] = 0  # Assume regular season for today's games
    
    # 3) Diff features (home - away)
    def add_diff_features(df: pd.DataFrame, home_prefix: str, away_prefix: str, diff_prefix: str):
        home_cols = [c for c in df.columns if c.startswith(home_prefix)]
        for hc in home_cols:
            suffix = hc[len(home_prefix):]
            ac = away_prefix + suffix
            if ac in df.columns:
                h = pd.to_numeric(df[hc], errors="coerce")
                a = pd.to_numeric(df[ac], errors="coerce")
                df[diff_prefix + suffix] = h - a
    
    add_diff_features(base, "home_team_", "away_team_", "diff_team_")
    add_diff_features(base, "home_rest_", "away_rest_", "diff_rest_")
    add_diff_features(base, "home_schedule_", "away_schedule_", "diff_schedule_")
    
    # 4) ELO features (set to neutral/default values since we don't have historical data)
    # Training uses base_rating=1500.0, so set both teams to 1500 for neutral prediction
    base["home_elo_before"] = 1500.0
    base["away_elo_before"] = 1500.0
    base["diff_elo"] = 0.0  # Neutral ELO difference
    
    # 5) Rolling features (set to NaN - will be filled with train_median)
    # Training has: home_roll_w5_*, home_roll_w10_*, away_roll_w5_*, away_roll_w10_*
    # and diff_roll_w5_*, diff_roll_w10_*
    # Since we don't have historical games, set to NaN (will be filled with train_median)
    rolling_windows = [5, 10]
    rolling_metrics = ["win_rate", "avg_score_diff", "avg_points_for", "avg_points_against"]
    
    for w in rolling_windows:
        for metric in rolling_metrics:
            # Home and away rolling features
            base[f"home_roll_w{w}_{metric}"] = np.nan
            base[f"away_roll_w{w}_{metric}"] = np.nan
            # Diff rolling features (will be computed from home - away, but set to NaN for now)
            base[f"diff_roll_w{w}_{metric}"] = np.nan
    
    # Compute diff_roll features from home - away (if both exist)
    for w in rolling_windows:
        for metric in rolling_metrics:
            home_col = f"home_roll_w{w}_{metric}"
            away_col = f"away_roll_w{w}_{metric}"
            diff_col = f"diff_roll_w{w}_{metric}"
            if home_col in base.columns and away_col in base.columns:
                # Will be NaN since both are NaN, but structure is correct
                base[diff_col] = base[home_col] - base[away_col]
    
    log(f"[INFO] Added missing features: gameId, date features, ELO (neutral), diff features, rolling (NaN)")
    log(f"[INFO] Total features after additions: {len(base.columns)}")

    return base


# -----------------------------
# Model + transforms
# -----------------------------
def load_feature_cols(path: Path) -> List[str]:
    if not path.exists(): raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_scaler(path: Path):
    if joblib is None: raise ImportError("joblib missing")
    if not path.exists(): raise FileNotFoundError(f"Missing {path}")
    return joblib.load(path)

def load_train_median(path: Path) -> Optional[pd.Series]:
    if joblib is None or not path.exists(): return None
    med = joblib.load(path)
    try:
        if isinstance(med, pd.Series): return med
        if isinstance(med, dict): return pd.Series(med)
    except: pass
    return None

def load_keras_model(path: Path):
    if load_model is None: raise ImportError("tensorflow missing")
    if not path.exists(): raise FileNotFoundError(f"Missing model: {path}")
    return load_model(str(path))

def prepare_X(df_features: pd.DataFrame, feature_cols: List[str], scaler_obj, train_median: Optional[pd.Series]) -> np.ndarray:
    Xdf = df_features.reindex(columns=feature_cols)
    
    # Feature alignment check
    missing_cols = set(feature_cols) - set(Xdf.columns)
    extra_cols = set(Xdf.columns) - set(feature_cols)
    if missing_cols:
        log(f"[ERROR] Missing features: {list(missing_cols)[:10]}... (showing first 10)")
    if extra_cols:
        log(f"[WARN] Extra features (will be ignored): {list(extra_cols)[:10]}... (showing first 10)")
    
    # Check for potential home/away swaps in first few columns
    if len(feature_cols) > 0:
        first_cols = feature_cols[:5]
        log(f"[DEBUG] First 5 feature columns: {first_cols}")
        home_cols = [c for c in first_cols if 'home_' in c.lower()]
        away_cols = [c for c in first_cols if 'away_' in c.lower()]
        if home_cols:
            log(f"[DEBUG] Sample home columns: {home_cols[:3]}")
        if away_cols:
            log(f"[DEBUG] Sample away columns: {away_cols[:3]}")
    
    for c in Xdf.columns:
        if not pd.api.types.is_numeric_dtype(Xdf[c]):
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    if train_median is not None:
        med = train_median.reindex(feature_cols)
        Xdf = Xdf.fillna(med)
    else:
        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med).fillna(0.0)

    try:
        X = scaler_obj.transform(Xdf.values)
    except:
        X = scaler_obj.transform(Xdf)
    return np.asarray(X, dtype=np.float32)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Daily NBA predictions (Classifier + Regressor)")
    parser.add_argument("--date", type=str, default=None, help="Override date YYYY-MM-DD")
    parser.add_argument("--run_current_data_nb", type=int, default=1, help="Run 01 notebook (1/0)")
    parser.add_argument("--run_injury_download", type=int, default=1, help="Run injury script (1/0)")

    args = parser.parse_args()
    paths = resolve_paths()
    ensure_dir(paths.data_processed_dir)
    run_date = today_str(args.date)
    log(f"[INFO] Running for date: {run_date}")

    # 1) Refresh Data
    if args.run_current_data_nb == 1 and paths.nb_current_data.exists():
        try:
            run_notebook_nbconvert(paths.nb_current_data, cwd=paths.repo_root)
        except Exception as e:
            log(f"[WARN] Data refresh failed: {e}")

    if args.run_injury_download == 1 and paths.download_injury_script.exists():
        try:
            run_subprocess([sys.executable, str(paths.download_injury_script)], cwd=paths.repo_root)
        except Exception as e:
            log(f"[WARN] Injury download failed: {e}")

    # 2) Load Today's Games
    games_today = load_today_games_from_rest_days(paths)
    if games_today.empty:
        log("[WARN] No games found today. Exiting.")
        return
    games_today["date"] = run_date

    # 3) Build Features
    features_df = build_features_for_today(paths, games_today)
    
    # Log feature creation summary
    log(f"[INFO] Created {len(features_df.columns)} features from build_features_for_today")
    home_cols = [c for c in features_df.columns if 'home_' in c.lower()]
    away_cols = [c for c in features_df.columns if 'away_' in c.lower()]
    log(f"[INFO] Home columns: {len(home_cols)}, Away columns: {len(away_cols)}")

    # 4) Load Artifacts (Models + Scalers)
    feature_cols = load_feature_cols(paths.feature_cols_json)
    log(f"[INFO] Training expects {len(feature_cols)} features")
    
    scaler_input = load_scaler(paths.scaler_joblib)
    
    # Target Scaler: Eğer yoksa (eski modelse) None döner, hata vermez ama skorlar ham kalır
    try:
        scaler_target = load_scaler(paths.scaler_target_joblib)
    except FileNotFoundError:
        log("[WARN] Target scaler not found. Predicted scores might be raw (scaled)!")
        scaler_target = None

    train_median = load_train_median(paths.train_median_joblib)
    
    clf_model = load_keras_model(paths.mlp_model_clf)
    reg_model = load_keras_model(paths.mlp_model_reg)

    # 5) Prepare Input
    X = prepare_X(features_df, feature_cols, scaler_input, train_median)

    # 6) PREDICT CLASSIFIER (Win Prob)
    win_probs = clf_model.predict(X, verbose=0)
    if win_probs.ndim == 2 and win_probs.shape[1] == 2:
        win_probs = win_probs[:, 1]
    win_probs = win_probs.reshape(-1)
    win_probs = np.clip(win_probs, 0.0, 1.0)

    # 7) PREDICT REGRESSOR (Score Diff)
    score_diffs_scaled = reg_model.predict(X, verbose=0).reshape(-1, 1)
    
    # DEBUG: Log raw scaled predictions
    log(f"[DEBUG] Raw scaled predictions (first 3): {score_diffs_scaled.flatten()[:3]}")
    if scaler_target is not None:
        log(f"[DEBUG] Scaler mean={scaler_target.mean_[0]:.2f}, scale={scaler_target.scale_[0]:.2f}")
        log(f"[DEBUG] Mean scaled prediction: {np.mean(score_diffs_scaled):.4f}")
    
    # Inverse Transform (CRITICAL STEP)
    if scaler_target is not None:
        score_diffs_real = scaler_target.inverse_transform(score_diffs_scaled).reshape(-1)
    else:
        score_diffs_real = score_diffs_scaled.reshape(-1)
    
    # DEBUG: Log inverse transformed predictions
    log(f"[DEBUG] Inverse transformed predictions (first 3): {score_diffs_real[:3]}")
    log(f"[DEBUG] Mean inverse transformed: {np.mean(score_diffs_real):.2f}")
    
    # Sanity check for inverted predictions
    if np.mean(score_diffs_real) < -10:
        log(f"[WARN] Mean prediction is very negative: {np.mean(score_diffs_real):.2f}")
        log(f"[WARN] This suggests model inversion or feature mismatch")
        log(f"[WARN] Classifier mean win prob: {np.mean(win_probs):.2f}")
        if np.mean(win_probs) > 0.5 and np.mean(score_diffs_real) < 0:
            log(f"[ERROR] CONFLICT: Classifier predicts home favorite but regressor predicts negative diff!")

    # 8) Output Table
    out = pd.DataFrame({
        "Date": games_today["date"],
        "Home": features_df["home_team"],
        "Away": features_df["away_team"],
        "Home Win %": (win_probs * 100).round(1),
        "Pred Diff": np.round(score_diffs_real, 1)
    })

    # Logic for "Pick"
    def get_pick(row):
        prob = row["Home Win %"]
        diff = row["Pred Diff"]
        
        # Classifier Prediction
        winner_clf = row["Home"] if prob >= 50 else row["Away"]
        conf_clf = "High" if (prob > 60 or prob < 40) else "Low"
        
        # Regressor Prediction
        winner_reg = row["Home"] if diff > 0 else row["Away"]
        
        # Consensus
        if winner_clf == winner_reg:
            return f"{winner_clf} (Strong)" if conf_clf == "High" else f"{winner_clf} (Lean)"
        else:
            return "CONFLICT (Risky)"

    out["Pick"] = out.apply(get_pick, axis=1)

    # Print nicely to console
    print("\n" + "="*60)
    print(f"PREDICTIONS FOR {run_date}")
    print("="*60)
    print(out.to_string(index=False))
    print("="*60 + "\n")

    # Save CSV
    out.to_csv(paths.predictions_today_csv, index=False)
    log(f"[OK] Saved to: {paths.predictions_today_csv}")


if __name__ == "__main__":
    main()