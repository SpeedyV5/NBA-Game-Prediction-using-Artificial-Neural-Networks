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
    train_median_joblib: Path
    mlp_model_h5: Path

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

    # produced by injury pipeline (03 notebook may fail; script still produces latest_injury.csv; and your other pipeline produces team_status)
    latest_injury_team_status_csv = data_raw_dir / "injury_reports_raw" / "latest_injury_team_status.csv"

    predictions_today_csv = data_processed_dir / "predictions_today.csv"

    feature_cols_json = models_dir / "feature_cols.json"
    scaler_joblib = models_dir / "scaler.joblib"
    train_median_joblib = models_dir / "train_median.joblib"
    mlp_model_h5 = models_dir / "mlp_classifier_best.h5"

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
        train_median_joblib=train_median_joblib,
        mlp_model_h5=mlp_model_h5,
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
    """
    Try:
      1) given path
      2) repo_root/data_raw/<name>
      3) repo_root/<name>
    """
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
    """
    Make raw column names match training naming style:
      - spaces -> _
      - % -> pct
      - remove dots
      - remove parentheses
      - collapse __
    Examples:
      WIN% -> WINpct
      1 DAY W% -> 1_DAY_Wpct
      5 IN 7 -> 5_IN_7
    """
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
    """
    rest_days_stats has per-team "OPPONENT TODAY" like:
      - "vs. Miami Heat"  (HOME)
      - "@ Boston Celtics" (AWAY)

    We build games by taking only the "vs." rows:
      home = TEAM NAME
      away = opponent parsed from "vs."
    """
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
        # fallback: if only @ exists, infer games by pairing
        log("[WARN] No 'vs.' rows found; cannot reliably infer home/away. Returning empty.")
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    def parse_opp(s: str) -> str:
        s = s.strip()
        # "vs. X" or "vs X"
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
    """
    latest_injury_team_status.csv is player-level with:
      TEAM, PLAYER_NAME, inj_out, inj_questionable, inj_doubtful (0/1)
    We aggregate per TEAM:
      injury_count = count of players where (out|questionable|doubtful)=1
    """
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
# Feature builder (matching training naming)
# -----------------------------
def build_features_for_today(paths: ProjectPaths, games_today: pd.DataFrame) -> pd.DataFrame:
    """
    Build a row per game with columns matching training-style names:
      - home_team_* from team_stats_raw
      - away_team_* from team_stats_raw
      - home_rest_* from rest_days_stats
      - away_rest_* from rest_days_stats
      - home_schedule_* from schedule_rest_days
      - away_schedule_* from schedule_rest_days
      - injury_count_home / injury_count_away
    """
    if games_today.empty:
        return games_today.copy()

    # --- Load sources
    team_stats = read_csv_safe(paths.team_stats_raw_csv)
    rest_stats = read_csv_safe(paths.rest_days_stats_csv)
    sched_stats = read_csv_safe(paths.schedule_rest_days_csv)
    inj_counts = load_injury_counts(paths)

    # --- Identify team name column consistently (full name)
    team_teamcol = _first_existing_col(team_stats, ["TEAM", "TEAM NAME", "Team"])
    rest_teamcol = _first_existing_col(rest_stats, ["TEAM NAME", "TEAM", "Team"])
    sched_teamcol = _first_existing_col(sched_stats, ["TEAMS", "TEAM", "TEAM NAME", "Team"])

    if team_teamcol is None:
        raise ValueError("team_stats_raw must have TEAM column (full team name).")
    if rest_teamcol is None:
        raise ValueError("rest_days_stats must have TEAM NAME column (full team name).")
    if sched_teamcol is None:
        raise ValueError("schedule_rest_days must have TEAMS column (full team name).")

    # --- Prepare numeric-only tables with sanitized column names
    def prep_numeric(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
        out = df.copy()
        out[team_col] = out[team_col].astype(str).str.strip()
        # keep numeric cols only
        num = out.select_dtypes(include=[np.number]).copy()
        num[team_col] = out[team_col]
        # sanitize numeric col names (except key)
        rename_map = {c: _sanitize_stat_col(c) for c in num.columns if c != team_col}
        num = num.rename(columns=rename_map)
        return num

    team_num = prep_numeric(team_stats, team_teamcol)
    rest_num = prep_numeric(rest_stats, rest_teamcol)
    sched_num = prep_numeric(sched_stats, sched_teamcol)

    # injury already aggregated
    if not inj_counts.empty:
        inj_counts = inj_counts.copy()
        inj_counts["TEAM"] = inj_counts["TEAM"].astype(str).str.strip()

    # --- Merge per game
    base = games_today.copy()
    base["home_team"] = base["home_team"].astype(str).str.strip()
    base["away_team"] = base["away_team"].astype(str).str.strip()

    # Team stats
    home_team = team_num.add_prefix("home_team_").rename(columns={f"home_team_{team_teamcol}": "home_team"})
    away_team = team_num.add_prefix("away_team_").rename(columns={f"away_team_{team_teamcol}": "away_team"})
    base = base.merge(home_team, on="home_team", how="left")
    base = base.merge(away_team, on="away_team", how="left")

    # Rest stats
    home_rest = rest_num.add_prefix("home_rest_").rename(columns={f"home_rest_{rest_teamcol}": "home_team"})
    away_rest = rest_num.add_prefix("away_rest_").rename(columns={f"away_rest_{rest_teamcol}": "away_team"})
    base = base.merge(home_rest, on="home_team", how="left")
    base = base.merge(away_rest, on="away_team", how="left")

    # Schedule rest-days distribution stats
    home_sched = sched_num.add_prefix("home_schedule_").rename(columns={f"home_schedule_{sched_teamcol}": "home_team"})
    away_sched = sched_num.add_prefix("away_schedule_").rename(columns={f"away_schedule_{sched_teamcol}": "away_team"})
    base = base.merge(home_sched, on="home_team", how="left")
    base = base.merge(away_sched, on="away_team", how="left")

        # Injury counts (match training feature names)
    if not inj_counts.empty:
        inj_h = inj_counts.rename(columns={"TEAM": "home_team", "injury_count": "injury_count_home"}).copy()
        inj_a = inj_counts.rename(columns={"TEAM": "away_team", "injury_count": "injury_count_away"}).copy()

        # avoid column-name collisions
        for c in ["injury_count_home", "injury_count_away"]:
            if c in base.columns:
                base = base.drop(columns=[c])

        base = base.merge(inj_h, on="home_team", how="left")
        base = base.merge(inj_a, on="away_team", how="left")

    # guarantee columns exist (no KeyError)
    if "injury_count_home" not in base.columns:
        base["injury_count_home"] = 0.0
    if "injury_count_away" not in base.columns:
        base["injury_count_away"] = 0.0

    base["injury_count_home"] = pd.to_numeric(base["injury_count_home"], errors="coerce").fillna(0.0)
    base["injury_count_away"] = pd.to_numeric(base["injury_count_away"], errors="coerce").fillna(0.0)


    return base


# -----------------------------
# Model + transforms
# -----------------------------
def load_feature_cols(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_cols.json at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_cols.json must be a non-empty list.")
    return cols


def load_scaler(path: Path):
    if joblib is None:
        raise ImportError("joblib not installed. Install: pip install joblib")
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler.joblib at: {path}")
    return joblib.load(path)


def load_train_median(path: Path) -> Optional[pd.Series]:
    if joblib is None:
        return None
    if not path.exists():
        return None
    med = joblib.load(path)
    # could be Series or dict-like
    try:
        if isinstance(med, pd.Series):
            return med
        if isinstance(med, dict):
            return pd.Series(med)
    except Exception:
        pass
    return None


def load_mlp_model(path: Path):
    if load_model is None:
        raise ImportError("tensorflow is not available. Install: pip install tensorflow")
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return load_model(str(path))


def prepare_X(df_features: pd.DataFrame, feature_cols: List[str], scaler_obj, train_median: Optional[pd.Series]) -> np.ndarray:
    # Align columns in one shot (prevents fragmentation + guarantees order)
    Xdf = df_features.reindex(columns=feature_cols)

    # numeric coercion
    for c in Xdf.columns:
        if not pd.api.types.is_numeric_dtype(Xdf[c]):
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # fill NaNs with training median if available, else inference median, else 0
    if train_median is not None:
        # align index
        med = train_median.reindex(feature_cols)
        Xdf = Xdf.fillna(med)
    else:
        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med).fillna(0.0)

    # transform
    try:
        X = scaler_obj.transform(Xdf.values)
    except Exception:
        X = scaler_obj.transform(Xdf)
    return np.asarray(X, dtype=np.float32)


def predict_win_prob_home(model, X: np.ndarray) -> np.ndarray:
    preds = model.predict(X, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 2:
        preds = preds[:, 1]
    preds = preds.reshape(-1)
    return np.clip(preds, 0.0, 1.0)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Daily NBA predictions (today games) - NO referee features.")
    parser.add_argument("--date", type=str, default=None, help="Override date YYYY-MM-DD (output only).")
    parser.add_argument("--run_current_data_nb", type=int, default=1, help="Run 01_current_data_download.ipynb (1/0).")
    parser.add_argument("--run_injury_download", type=int, default=1, help="Run src/data/download_injury_report.py (1/0).")

    args = parser.parse_args()

    paths = resolve_paths()
    ensure_dir(paths.data_processed_dir)

    run_date = today_str(args.date)
    log(f"[INFO] Date = {run_date}")
    log(f"[INFO] Repo root = {paths.repo_root}")

    # 1) Refresh current data CSVs
    if args.run_current_data_nb == 1:
        if not paths.nb_current_data.exists():
            raise FileNotFoundError(f"Notebook not found: {paths.nb_current_data}")
        try:
            run_notebook_nbconvert(paths.nb_current_data, cwd=paths.repo_root)
        except Exception as e:
            # If outputs exist, continue
            required = [paths.rest_days_stats_csv, paths.schedule_rest_days_csv, paths.player_stats_raw_csv, paths.team_stats_raw_csv]
            missing = [p for p in required if not p.exists()]
            if missing:
                raise RuntimeError(f"01_current_data_download failed and outputs missing: {missing}. Reason: {e}")
            log(f"[WARN] 01_current_data_download failed but outputs exist; continuing. Reason: {e}")

    # 2) Refresh injury report (player-level CSVs)
    if args.run_injury_download == 1:
        if paths.download_injury_script.exists():
            try:
                run_subprocess([sys.executable, str(paths.download_injury_script)], cwd=paths.repo_root)
            except Exception as e:
                log(f"[WARN] Injury download failed; continuing without refreshed injury. Reason: {e}")
        else:
            log("[WARN] Injury download script not found; skipping.")

    # 3) Today games (from rest_days_stats OPPONENT TODAY)
    games_today = load_today_games_from_rest_days(paths)
    if games_today.empty:
        log("[WARN] No games detected for today. Writing empty predictions.")
        out = pd.DataFrame(columns=["date", "home_team", "away_team", "win_prob_home", "predicted_winner"])
        out.to_csv(paths.predictions_today_csv, index=False)
        log(f"[OK] Wrote: {paths.predictions_today_csv}")
        return

    # set date for output consistency
    games_today["date"] = run_date

    # 4) Build features matching training naming
    features_df = build_features_for_today(paths, games_today)

    # 5) Load training artifacts
    feature_cols = load_feature_cols(paths.feature_cols_json)
    scaler_obj = load_scaler(paths.scaler_joblib)
    train_median = load_train_median(paths.train_median_joblib)
    model = load_mlp_model(paths.mlp_model_h5)

    # 6) Prepare X
    X = prepare_X(features_df, feature_cols, scaler_obj, train_median)

    # 7) Predict
    win_prob_home = predict_win_prob_home(model, X)

    # 8) Output
    out = pd.DataFrame({
        "date": pd.to_datetime(features_df["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna(run_date),
        "home_team": features_df["home_team"].astype(str),
        "away_team": features_df["away_team"].astype(str),
        "win_prob_home": win_prob_home.astype(float),
    })
    out["predicted_winner"] = np.where(out["win_prob_home"] >= 0.5, out["home_team"], out["away_team"])
    out["model_used"] = paths.mlp_model_h5.name

    out.to_csv(paths.predictions_today_csv, index=False)
    log(f"[OK] Wrote: {paths.predictions_today_csv} (rows={len(out)})")


if __name__ == "__main__":
    main()
