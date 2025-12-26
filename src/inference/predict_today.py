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

from src.features.build_features import (
    compute_elo_ratings,
    _build_team_game_long,
    _compute_rolling_features,
    _build_diff_features,
    _add_date_features,
)

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
    league_gamelog_csv: Path

    # Injury outputs
    latest_injury_team_status_csv: Path
    latest_injury_csv: Path

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
    league_gamelog_csv = data_raw_dir / "league_gamelog_2025_26.csv"

    # produced by injury pipeline (03 notebook may fail; script still produces latest_injury.csv; and your other pipeline produces team_status)
    latest_injury_team_status_csv = data_raw_dir / "injury_reports_raw" / "latest_injury_team_status.csv"
    latest_injury_csv = data_raw_dir / "injury_reports_raw" / "latest_injury.csv"

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
        league_gamelog_csv=league_gamelog_csv,
        latest_injury_team_status_csv=latest_injury_team_status_csv,
        latest_injury_csv=latest_injury_csv,
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
    s = str(col).strip()
    s = s.replace("%", "pct")
    s = s.replace("+", "")          # <-- EKLE
    s = s.replace(".", "")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def normalize_team_name(name: str) -> str:
    """
    Takım adlarını (kısaltma/mascot/full) tekil şehir tabanlı isme indirger.
    Eğitimdeki veri seti şehir isimleriyle çalıştığı için burada da aynı
    kanonikleri kullanıyoruz.
    """
    if not isinstance(name, str):
        return ""
    n = name.strip().lower()
    n_no_space = n.replace(" ", "").replace("\t", "").replace("\n", "")
    mapping = {
        "atlanta hawks": "Atlanta",
        "atlanta": "Atlanta",
        "atlantahawks": "Atlanta",
        "bos": "Boston",
        "boston celtics": "Boston",
        "boston": "Boston",
        "bostonceltics": "Boston",
        "brooklyn nets": "Brooklyn",
        "brooklyn": "Brooklyn",
        "bkn": "Brooklyn",
        "brooklynnets": "Brooklyn",
        "cha": "Charlotte",
        "charlotte hornets": "Charlotte",
        "charlotte": "Charlotte",
        "charlottehornets": "Charlotte",
        "chi": "Chicago",
        "chicago bulls": "Chicago",
        "chicago": "Chicago",
        "chicagobulls": "Chicago",
        "cle": "Cleveland",
        "cleveland cavaliers": "Cleveland",
        "cleveland": "Cleveland",
        "clevelandcavaliers": "Cleveland",
        "dal": "Dallas",
        "dallas mavericks": "Dallas",
        "dallas": "Dallas",
        "dallasmavericks": "Dallas",
        "den": "Denver",
        "denver nuggets": "Denver",
        "denver": "Denver",
        "denvernuggets": "Denver",
        "det": "Detroit",
        "detroit pistons": "Detroit",
        "detroit": "Detroit",
        "detroitpistons": "Detroit",
        "gsw": "Golden State",
        "gs": "Golden State",
        "golden state warriors": "Golden State",
        "golden state": "Golden State",
        "goldenstatewarriors": "Golden State",
        "hou": "Houston",
        "houston rockets": "Houston",
        "houston": "Houston",
        "houstonrockets": "Houston",
        "ind": "Indiana",
        "indiana pacers": "Indiana",
        "indiana": "Indiana",
        "indianapacers": "Indiana",
        "lac": "LA Clippers",
        "la clippers": "LA Clippers",
        "los angeles clippers": "LA Clippers",
        "laclippers": "LA Clippers",
        "lal": "LA Lakers",
        "la lakers": "LA Lakers",
        "los angeles lakers": "LA Lakers",
        "lalakers": "LA Lakers",
        "mem": "Memphis",
        "memphis grizzlies": "Memphis",
        "memphis": "Memphis",
        "memphisgrizzlies": "Memphis",
        "mia": "Miami",
        "miami heat": "Miami",
        "miami": "Miami",
        "miamiheat": "Miami",
        "mil": "Milwaukee",
        "milwaukee bucks": "Milwaukee",
        "milwaukee": "Milwaukee",
        "milwaukeebucks": "Milwaukee",
        "min": "Minnesota",
        "minnesota timberwolves": "Minnesota",
        "minnesota": "Minnesota",
        "minnesotatimberwolves": "Minnesota",
        "nop": "New Orleans",
        "no": "New Orleans",
        "new orleans pelicans": "New Orleans",
        "new orleans": "New Orleans",
        "neworleanspelicans": "New Orleans",
        "nyk": "New York",
        "new york knicks": "New York",
        "new york": "New York",
        "newyorkknicks": "New York",
        "okc": "Oklahoma City",
        "oklahoma city thunder": "Oklahoma City",
        "oklahoma city": "Oklahoma City",
        "oklahomacitythunder": "Oklahoma City",
        "orl": "Orlando",
        "orlando magic": "Orlando",
        "orlando": "Orlando",
        "orlandomagic": "Orlando",
        "phi": "Philadelphia",
        "philadelphia 76ers": "Philadelphia",
        "philadelphia": "Philadelphia",
        "philadelphia76ers": "Philadelphia",
        "phx": "Phoenix",
        "phoenix suns": "Phoenix",
        "phoenix": "Phoenix",
        "phoenixsuns": "Phoenix",
        "por": "Portland",
        "portland trail blazers": "Portland",
        "portland": "Portland",
        "portlandtrailblazers": "Portland",
        "sac": "Sacramento",
        "sacramento kings": "Sacramento",
        "sacramento": "Sacramento",
        "sacramentokings": "Sacramento",
        "sas": "San Antonio",
        "san antonio spurs": "San Antonio",
        "san antonio": "San Antonio",
        "sanantoniospurs": "San Antonio",
        "tor": "Toronto",
        "toronto raptors": "Toronto",
        "toronto": "Toronto",
        "torontoraptors": "Toronto",
        "uta": "Utah",
        "utah jazz": "Utah",
        "utah": "Utah",
        "utahjazz": "Utah",
        "was": "Washington",
        "washington wizards": "Washington",
        "washington": "Washington",
        "washingtonwizards": "Washington",
    }
    if n in mapping:
        return mapping[n]
    if n_no_space in mapping:
        return mapping[n_no_space]
    return name.strip()


def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower()
    for ch in [",", ".", "'", "-", " ", "\t", "\n"]:
        n = n.replace(ch, "")
    return n



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
    Enriched injury aggregation:
      - injury_count: out/doubtful/questionable oyuncu sayısı
      - expected_minutes_lost: player_stats_raw MpG * (out/doubtful=1, questionable=0.5)
      - any_key_player_out: top dakika oyuncularından biri out/doubtful ise 1
    Fallback: latest_injury_team_status.csv yoksa latest_injury.csv kullanılır.
    """
    player_minutes = None
    mpg_col_std = "mpg"
    try:
        ps = read_csv_safe(paths.player_stats_raw_csv)
        name_col = _first_existing_col(ps, ["NAME", "Player", "PLAYER_NAME"])
        team_col = _first_existing_col(ps, ["TEAM", "TEAM NAME", "Team"])
        mpg_col = _first_existing_col(ps, ["MpG", "MPG", "MIN", "MIN/G"])
        if name_col and team_col and mpg_col:
            tmp = ps[[name_col, team_col, mpg_col]].copy()
            tmp = tmp.rename(columns={name_col: "player_name", team_col: "TEAM", mpg_col: mpg_col_std})
            tmp["player_name"] = tmp["player_name"].astype(str).map(normalize_player_name)
            tmp["TEAM"] = tmp["TEAM"].astype(str).map(normalize_team_name)
            tmp[mpg_col_std] = pd.to_numeric(tmp[mpg_col_std], errors="coerce")
            player_minutes = tmp
    except Exception as e:
        log(f"[WARN] player_stats_raw okunamadı, minutes yok: {e}")

    def aggregate_from_df(df_in: pd.DataFrame, team_col: str, name_col: str = None, status_col: str = None) -> pd.DataFrame:
        df = df_in.copy()
        df[team_col] = df[team_col].astype(str).map(normalize_team_name)
        if status_col:
            df[status_col] = df[status_col].astype(str).str.lower().str.strip()
        # default status mapping
        def status_weight(s: str) -> float:
            if s in ["out", "doubtful"]:
                return 1.0
            if s == "questionable":
                return 0.5
            return 0.0

        df["status_weight"] = df[status_col].map(status_weight) if status_col else 0.0
        # injury_count as count of status_weight>0
        df["inj_flag"] = (df["status_weight"] > 0).astype(int)

        # expected minutes
        df["minutes_est"] = 0.0
        if player_minutes is not None and name_col:
            df[name_col] = df[name_col].astype(str).map(normalize_player_name)
            pm = player_minutes.set_index("player_name")
            df["minutes_est"] = df.index.map(lambda idx: pm.at[df.loc[idx, name_col], mpg_col_std] if df.loc[idx, name_col] in pm.index else np.nan)
            df["minutes_est"] = pd.to_numeric(df["minutes_est"], errors="coerce")
            df["minutes_est"] = df["minutes_est"].fillna(df["minutes_est"].median())
            df["minutes_est"] = df["minutes_est"] * df["status_weight"]

        # key player: üst dakikalı oyuncular (ilk 5) out/doubtful
        df["key_out"] = 0
        if player_minutes is not None and name_col:
            # per team top 5
            top_players = (
                player_minutes.sort_values(mpg_col_std, ascending=False)
                .groupby("TEAM")
                .head(5)[["TEAM", "player_name"]]
            )
            top_set = { (r["TEAM"], r["player_name"]) for _, r in top_players.iterrows() }
            def is_key(row):
                return 1 if (row[team_col], row[name_col]) in top_set and row["status_weight"] >= 1.0 else 0
            df["key_out"] = df.apply(is_key, axis=1)

        agg = df.groupby(team_col, as_index=False).agg(
            injury_count=("inj_flag", "sum"),
            expected_minutes_lost=("minutes_est", "sum"),
            any_key_player_out=("key_out", "max"),
        )
        agg = agg.rename(columns={team_col: "TEAM"})
        return agg

    # primary: latest_injury_team_status
    df_lits = None
    try:
        df_lits = read_csv_safe(paths.latest_injury_team_status_csv)
        df_lits.columns = df_lits.columns.str.strip().str.lower()
    except Exception as e:
        log(f"[INFO] latest_injury_team_status.csv okunamadı, fallback latest_injury.csv. Reason: {e}")

    if df_lits is not None and not df_lits.empty:
        # Preferred direct path: known columns
        if {"team", "player_name", "inj_out", "inj_questionable", "inj_doubtful"}.issubset(set(df_lits.columns)):
            tmp = df_lits.copy()
            tmp["status"] = "available"
            tmp.loc[pd.to_numeric(tmp["inj_out"], errors="coerce").fillna(0) > 0, "status"] = "out"
            tmp.loc[pd.to_numeric(tmp["inj_doubtful"], errors="coerce").fillna(0) > 0, "status"] = "doubtful"
            tmp.loc[pd.to_numeric(tmp["inj_questionable"], errors="coerce").fillna(0) > 0, "status"] = "questionable"
            return aggregate_from_df(tmp[["team", "player_name", "status"]], team_col="team", name_col="player_name", status_col="status")

        # Direct path if inj_out/inj_questionable/inj_doubtful mevcut
        if {"team", "player_name", "inj_out", "inj_questionable", "inj_doubtful"}.issubset(set(df_lits.columns)):
            tmp = df_lits.copy()
            tmp["status"] = "available"
            tmp.loc[pd.to_numeric(tmp["inj_out"], errors="coerce").fillna(0) > 0, "status"] = "out"
            tmp.loc[pd.to_numeric(tmp["inj_doubtful"], errors="coerce").fillna(0) > 0, "status"] = "doubtful"
            tmp.loc[pd.to_numeric(tmp["inj_questionable"], errors="coerce").fillna(0) > 0, "status"] = "questionable"
            return aggregate_from_df(tmp[["team", "player_name", "status"]], team_col="team", name_col="player_name", status_col="status")
        else:
            # fallback heuristic with column guessing
            team_col = _first_existing_col(df_lits, ["TEAM", "TEAM NAME", "Team", "team"])
            name_col = _first_existing_col(df_lits, ["PLAYER_NAME", "PLAYER", "Player", "player_name", "player name"])
            status_cols = {
                "inj_out": ["inj_out", "OUT"],
                "inj_questionable": ["inj_questionable", "QUESTIONABLE"],
                "inj_doubtful": ["inj_doubtful", "DOUBTFUL"],
            }
            status_col = _first_existing_col(df_lits, ["status", "STATUS"])
            if status_col is None:
                out_c = _first_existing_col(df_lits, status_cols["inj_out"])
                q_c = _first_existing_col(df_lits, status_cols["inj_questionable"])
                d_c = _first_existing_col(df_lits, status_cols["inj_doubtful"])
                if out_c or q_c or d_c:
                    df_lits["__status_tmp"] = "available"
                    if out_c:
                        df_lits.loc[pd.to_numeric(df_lits[out_c], errors="coerce").fillna(0) > 0, "__status_tmp"] = "out"
                    if d_c:
                        df_lits.loc[pd.to_numeric(df_lits[d_c], errors="coerce").fillna(0) > 0, "__status_tmp"] = "doubtful"
                    if q_c:
                        df_lits.loc[pd.to_numeric(df_lits[q_c], errors="coerce").fillna(0) > 0, "__status_tmp"] = "questionable"
                    status_col = "__status_tmp"

            if team_col and name_col and status_col and all(c in df_lits.columns for c in [team_col, name_col, status_col]):
                try:
                    return aggregate_from_df(df_lits[[team_col, name_col, status_col]], team_col=team_col, name_col=name_col, status_col=status_col)
                except Exception as e:
                    log(f"[WARN] latest_injury_team_status aggregation failed: {e}")

    # fallback: latest_injury.csv
    try:
        raw = read_csv_safe(paths.latest_injury_csv)
        raw.columns = raw.columns.str.strip().str.lower()
        # Some parsers might name player column differently; try best-effort
        name_col = _first_existing_col(raw, ["player_name", "player", "name"])
        if name_col is None:
            raise ValueError("latest_injury.csv missing player_name column.")
        if "team" not in raw.columns or "status" not in raw.columns:
            raise ValueError("latest_injury.csv must have 'team' and 'status' columns.")
        subset_cols = ["team", name_col, "status"]
        missing_cols = [c for c in subset_cols if c not in raw.columns]
        if missing_cols:
            raise ValueError(f"latest_injury.csv missing columns: {missing_cols}")
        return aggregate_from_df(raw[subset_cols], team_col="team", name_col=name_col, status_col="status")
    except Exception as e2:
        log(f"[INFO] No injury data available; injury features = 0. Reason: {e2}")
        return pd.DataFrame(columns=["TEAM", "injury_count", "expected_minutes_lost", "any_key_player_out"])

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
        out[team_col] = out[team_col].astype(str).str.strip().map(normalize_team_name)
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

    # --- Merge per game
    base = games_today.copy()
    base["home_team"] = base["home_team"].astype(str).str.strip().map(normalize_team_name)
    base["away_team"] = base["away_team"].astype(str).str.strip().map(normalize_team_name)

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
        inj_counts = inj_counts.copy()
        inj_counts["TEAM"] = inj_counts["TEAM"].astype(str).str.strip().map(normalize_team_name)
        try:
            uniq_inj = set(inj_counts["TEAM"].unique())
            uniq_games = set(base["home_team"].unique()).union(set(base["away_team"].unique()))
            overlap = uniq_inj.intersection(uniq_games)
            log(f"[DEBUG] injury teams: {len(uniq_inj)}, games teams: {len(uniq_games)}, overlap: {len(overlap)}")
        except Exception:
            pass

        inj_h = inj_counts.rename(columns={
            "TEAM": "home_team",
            "injury_count": "injury_count_home",
            "expected_minutes_lost": "expected_minutes_lost_home",
            "any_key_player_out": "any_key_player_out_home",
        }).copy()
        inj_a = inj_counts.rename(columns={
            "TEAM": "away_team",
            "injury_count": "injury_count_away",
            "expected_minutes_lost": "expected_minutes_lost_away",
            "any_key_player_out": "any_key_player_out_away",
        }).copy()

        # avoid column-name collisions
        for c in ["injury_count_home", "injury_count_away", "expected_minutes_lost_home", "expected_minutes_lost_away", "any_key_player_out_home", "any_key_player_out_away"]:
            if c in base.columns:
                base = base.drop(columns=[c])

        base = base.merge(inj_h, on="home_team", how="left")
        base = base.merge(inj_a, on="away_team", how="left")
        try:
            sample = base[["home_team", "away_team", "injury_count_home", "injury_count_away"]].head()
            log(f"[DEBUG] injury merged sample:\n{sample}")
        except Exception:
            pass

    # guarantee columns exist (no KeyError)
    if "injury_count_home" not in base.columns:
        base["injury_count_home"] = 0.0
    if "injury_count_away" not in base.columns:
        base["injury_count_away"] = 0.0
    for c in ["expected_minutes_lost_home", "expected_minutes_lost_away", "any_key_player_out_home", "any_key_player_out_away"]:
        if c not in base.columns:
            base[c] = 0.0

    base["injury_count_home"] = pd.to_numeric(base["injury_count_home"], errors="coerce").fillna(0.0)
    base["injury_count_away"] = pd.to_numeric(base["injury_count_away"], errors="coerce").fillna(0.0)
    base["expected_minutes_lost_home"] = pd.to_numeric(base["expected_minutes_lost_home"], errors="coerce").fillna(0.0)
    base["expected_minutes_lost_away"] = pd.to_numeric(base["expected_minutes_lost_away"], errors="coerce").fillna(0.0)
    base["any_key_player_out_home"] = pd.to_numeric(base["any_key_player_out_home"], errors="coerce").fillna(0.0)
    base["any_key_player_out_away"] = pd.to_numeric(base["any_key_player_out_away"], errors="coerce").fillna(0.0)

    return base


# -----------------------------
# Historical stats for ELO / rolling
# -----------------------------
def load_historical_games(paths: ProjectPaths) -> pd.DataFrame:
    """
    Build single-row-per-game table from league_gamelog (two rows per game: home/away).
    Returns columns: gameId, game_date, home_team, away_team, home_score, away_score
    """
    try:
        df = read_csv_safe(paths.league_gamelog_csv)
    except Exception as e:
        log(f"[WARN] league_gamelog yüklenemedi, ELO/rolling olmadan devam: {e}")
        return pd.DataFrame(columns=["gameId", "game_date", "home_team", "away_team", "home_score", "away_score"])

    required_cols = ["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_NAME", "PTS"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log(f"[WARN] league_gamelog eksik kolonlar: {missing}. ELO/rolling atlanacak.")
        return pd.DataFrame(columns=["gameId", "game_date", "home_team", "away_team", "home_score", "away_score"])

    def canon(row):
        # Önce tam isim, yoksa 3 harfli kısaltma
        name = str(row.get("TEAM_NAME", "")).strip()
        abbr = str(row.get("TEAM_ABBREVIATION", "")).strip()
        if name:
            c = normalize_team_name(name)
            if c:
                return c
        if abbr:
            c = normalize_team_name(abbr)
            if c:
                return c
        return name or abbr

    df["MATCHUP"] = df["MATCHUP"].astype(str)
    games = []
    for game_id, g in df.groupby("GAME_ID"):
        home_row = g[g["MATCHUP"].str.contains("vs.", case=False, regex=False)]
        away_row = g[g["MATCHUP"].str.contains("@", case=False, regex=False)]

        if home_row.empty or away_row.empty:
            # try fallback: first row home, second away
            if len(g) >= 2:
                home_row = g.iloc[[0]]
                away_row = g.iloc[[1]]
            else:
                continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        games.append(
            {
                "gameId": game_id,
                "game_date": home_row["GAME_DATE"],
                "home_team": canon(home_row),
                "away_team": canon(away_row),
                "home_score": home_row["PTS"],
                "away_score": away_row["PTS"],
            }
        )

    out = pd.DataFrame(games)
    if not out.empty:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        out["home_team"] = out["home_team"].map(normalize_team_name)
        out["away_team"] = out["away_team"].map(normalize_team_name)
    return out


def _latest_per_team_from_elo(elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take compute_elo_ratings output and return per-team last elo_before.
    """
    if elo_df.empty:
        return pd.DataFrame(columns=["team", "elo_before"])

    rows = []
    for _, r in elo_df.iterrows():
        rows.append({"team": str(r["home_team"]).strip(), "game_date": r["game_date"], "elo_before": r["home_elo_before"]})
        rows.append({"team": str(r["away_team"]).strip(), "game_date": r["game_date"], "elo_before": r["away_elo_before"]})

    df = pd.DataFrame(rows)
    df = df.sort_values(["team", "game_date"]).dropna(subset=["elo_before"])
    latest = df.groupby("team", as_index=False).tail(1)
    return latest[["team", "elo_before"]].rename(columns={"team": "TEAM"})


def _latest_per_team_from_rolling(roll_long: pd.DataFrame) -> pd.DataFrame:
    """
    Use _compute_rolling_features output (long format) and pick last stats per team.
    """
    if roll_long.empty:
        return pd.DataFrame()
    roll_cols = [c for c in roll_long.columns if c.startswith("roll_w")]
    keep = ["team"] + roll_cols
    df = roll_long.sort_values(["team", "game_date", "gameId"]).dropna(subset=roll_cols, how="all")
    latest = df.groupby("team", as_index=False).tail(1)
    latest = latest[keep].copy()
    latest["team"] = latest["team"].astype(str).str.strip()
    return latest


def _ensure_diff_for_prefixes(df: pd.DataFrame, prefixes: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Create diff_<...> = home - away for given prefixes if missing.
    """
    out = df.copy()
    for home_prefix, away_prefix, diff_prefix in prefixes:
        home_cols = [c for c in out.columns if c.startswith(home_prefix)]
        for hc in home_cols:
            suffix = hc[len(home_prefix):]
            ac = away_prefix + suffix
            if ac not in out.columns:
                continue
            dc = diff_prefix + suffix
            if dc in out.columns:
                continue
            h = pd.to_numeric(out[hc], errors="coerce")
            a = pd.to_numeric(out[ac], errors="coerce")
            if h.notna().any() or a.notna().any():
                out[dc] = h - a
    return out


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
        # if we don't have training median, use scaler mean as neutral fill
        try:
            fill_values = pd.Series(scaler_obj.mean_, index=feature_cols)
            Xdf = Xdf.fillna(fill_values)
        except Exception:
            med = Xdf.median(numeric_only=True)
            Xdf = Xdf.fillna(med)
        Xdf = Xdf.fillna(0.0)


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

    # 4b) Tarih feature'ları
    features_df["game_date"] = pd.to_datetime(run_date)
    features_df = _add_date_features(features_df, "game_date")
    if "is_playoff" not in features_df.columns:
        features_df["is_playoff"] = 0

    # 4c) gameId placeholder (tahmin satırları için)
    if "gameId" not in features_df.columns:
        features_df["gameId"] = np.arange(1, len(features_df) + 1)

    # 4d) ELO ve rolling geçmişi
    hist_games = load_historical_games(paths)
    if not hist_games.empty:
        hist_games = hist_games.dropna(subset=["game_date"])
        hist_games = hist_games[hist_games["game_date"] < pd.to_datetime(run_date)]

    if not hist_games.empty:
        try:
            elo_df = compute_elo_ratings(hist_games.copy())
            elo_latest = _latest_per_team_from_elo(elo_df)

            if not elo_latest.empty:
                elo_home = elo_latest.rename(columns={"TEAM": "home_team", "elo_before": "home_elo_before"})
                elo_away = elo_latest.rename(columns={"TEAM": "away_team", "elo_before": "away_elo_before"})
                features_df = features_df.merge(elo_home, on="home_team", how="left")
                features_df = features_df.merge(elo_away, on="away_team", how="left")
        except Exception as e:
            log(f"[WARN] ELO hesaplanamadı: {e}")

        try:
            roll_long = _build_team_game_long(hist_games)
            roll_long = _compute_rolling_features(roll_long)
            roll_latest = _latest_per_team_from_rolling(roll_long)
            if not roll_latest.empty:
                roll_home = roll_latest.rename(columns={"team": "home_team"}).add_prefix("home_")
                roll_home = roll_home.rename(columns={"home_home_team": "home_team"})
                roll_away = roll_latest.rename(columns={"team": "away_team"}).add_prefix("away_")
                roll_away = roll_away.rename(columns={"away_away_team": "away_team"})
                features_df = features_df.merge(roll_home, on="home_team", how="left")
                features_df = features_df.merge(roll_away, on="away_team", how="left")

                # diff_roll_* kolonları
                roll_cols = [c for c in features_df.columns if c.startswith("home_roll_w")]
                for hc in roll_cols:
                    suffix = hc[len("home_"):]
                    ac = "away_" + suffix
                    if ac in features_df.columns:
                        features_df[f"diff_{suffix}"] = pd.to_numeric(features_df[hc], errors="coerce") - pd.to_numeric(features_df[ac], errors="coerce")
        except Exception as e:
            log(f"[WARN] Rolling feature hesaplanamadı: {e}")

    # 4e) diff_* kolonları (takım / rest / schedule)
    try:
        features_df = _build_diff_features(features_df)
    except Exception as e:
        log(f"[WARN] diff feature üretilemedi: {e}")

    features_df = _ensure_diff_for_prefixes(
        features_df,
        [
            ("home_team_", "away_team_", "diff_team_"),
            ("home_rest_", "away_rest_", "diff_rest_"),
            ("home_schedule_", "away_schedule_", "diff_schedule_"),
            ("home_roll_w", "away_roll_w", "diff_roll_w"),
        ],
    )

    schedule_metrics = [
        "TOTAL_GAMES",
        "5_IN_7",
        "3IN4_B2B",
        "SOFT_B2B",
        "ALL_B2B",
        "TOTAL_B2B_ON_THE_ROAD",
        "TOTAL_B2B_AT_HOME",
        "3IN4",
        "1_DAY_REST",
        "2_DAYS_REST",
        "3DAYS_REST",
        "REST_ADVANTAGE",
        "REST_DISADVANTAGE",
        "BOTH_TEAMS_RESTED_or_NO_REST",
    ]
    for metric in schedule_metrics:
        hc = f"home_schedule_{metric}"
        ac = f"away_schedule_{metric}"
        dc = f"diff_schedule_{metric}"
        if dc not in features_df.columns and hc in features_df.columns and ac in features_df.columns:
            h = pd.to_numeric(features_df[hc], errors="coerce")
            a = pd.to_numeric(features_df[ac], errors="coerce")
            features_df[dc] = h - a

    # 4f) Sakatlık genişletmeleri (en azından kolon var olsun)
    for c in ["expected_minutes_lost_home", "expected_minutes_lost_away", "any_key_player_out_home", "any_key_player_out_away"]:
        if c not in features_df.columns:
            features_df[c] = 0.0
    for c in ["home_elo_before", "away_elo_before", "diff_elo"]:
        if c not in features_df.columns:
            features_df[c] = 1500.0 if c != "diff_elo" else 0.0

    # diff_elo ekle (varsa)
    if "home_elo_before" in features_df.columns and "away_elo_before" in features_df.columns:
        features_df["diff_elo"] = pd.to_numeric(features_df["home_elo_before"], errors="coerce") - pd.to_numeric(features_df["away_elo_before"], errors="coerce")

    # 5) Load training artifacts
    feature_cols = load_feature_cols(paths.feature_cols_json)
    scaler_obj = load_scaler(paths.scaler_joblib)
    train_median = load_train_median(paths.train_median_joblib)
    model = load_mlp_model(paths.mlp_model_h5)


    feature_cols = load_feature_cols(paths.feature_cols_json)
    made_cols = set(features_df.columns)
    need_cols = set(feature_cols)

    missing = [c for c in feature_cols if c not in made_cols]
    extra = [c for c in features_df.columns if c not in need_cols]

    log(f"[DEBUG] features_df cols = {features_df.shape[1]}")
    log(f"[DEBUG] need(feature_cols) = {len(feature_cols)}")
    log(f"[DEBUG] missing needed cols = {len(missing)}")
    log(f"[DEBUG] first 30 missing: {missing[:30]}")

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
