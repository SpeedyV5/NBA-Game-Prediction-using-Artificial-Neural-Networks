from __future__ import annotations

from pathlib import Path
from typing import List, Union, Dict, Tuple

import pandas as pd

ID_KEEP_COLS = [
    "gameId",
    "game_date",
    "season_year",
    "season_type",
    "home_team",
    "away_team",
]

ROLLING_WINDOWS = (5, 10)


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _add_date_features(df: pd.DataFrame, date_col: str = "game_date") -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["month"] = df[date_col].dt.month
        df["day_of_week"] = df[date_col].dt.dayofweek  # Mon=0
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _one_hot_season_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "season_type" in df.columns:
        df["season_type"] = df["season_type"].astype(str).str.lower().str.strip()
        df["is_playoff"] = (df["season_type"] == "playoff").astype(int)
    else:
        df["season_type"] = "regular"
        df["is_playoff"] = 0
    return df


def _make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "home_score" in df.columns:
        df["home_score"] = _to_numeric_safe(df["home_score"])
    if "away_score" in df.columns:
        df["away_score"] = _to_numeric_safe(df["away_score"])

    df["home_team_win"] = (df["home_score"] > df["away_score"]).astype("Int64")
    df["score_diff"] = (df["home_score"] - df["away_score"]).astype(float)
    return df


def _build_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only numeric diffs:
      home_team_<num> - away_team_<num> -> diff_team_<num>
      home_rest_<num> - away_rest_<num> -> diff_rest_<num>
      home_schedule_<num> - away_schedule_<num> -> diff_schedule_<num>
    Avoids categorical/string columns producing full-NaN diffs.
    """
    df = df.copy()

    def build_for_prefix(home_prefix: str, away_prefix: str, diff_prefix: str):
        home_cols = [c for c in df.columns if c.startswith(home_prefix)]
        for hc in home_cols:
            suffix = hc[len(home_prefix):]
            ac = away_prefix + suffix
            if ac not in df.columns:
                continue

            h = _to_numeric_safe(df[hc])
            a = _to_numeric_safe(df[ac])

            # only if at least some numeric signal exists
            if h.notna().any() or a.notna().any():
                df[diff_prefix + suffix] = h - a

    build_for_prefix("home_team_", "away_team_", "diff_team_")
    build_for_prefix("home_rest_", "away_rest_", "diff_rest_")
    build_for_prefix("home_schedule_", "away_schedule_", "diff_schedule_")

    return df


def _build_team_game_long(df: pd.DataFrame) -> pd.DataFrame:
    req = ["gameId", "game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Rolling features için eksik kolonlar: {missing}")

    base = df[req].copy()
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")

    home = base.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    )[["gameId", "game_date", "team", "opponent", "points_for", "points_against"]].copy()
    home["is_home"] = 1

    away = base.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    )[["gameId", "game_date", "team", "opponent", "points_for", "points_against"]].copy()
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)

    long_df["points_for"] = _to_numeric_safe(long_df["points_for"])
    long_df["points_against"] = _to_numeric_safe(long_df["points_against"])

    long_df["score_diff"] = long_df["points_for"] - long_df["points_against"]
    long_df["win"] = (long_df["score_diff"] > 0).astype(int)

    long_df = long_df.sort_values(["team", "game_date", "gameId"]).reset_index(drop=True)
    return long_df


def _compute_rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df = df.sort_values(["team", "game_date", "gameId"]).reset_index(drop=True)
    g = df.groupby("team", sort=False)

    for w in ROLLING_WINDOWS:
        df[f"roll_w{w}_win_rate"] = g["win"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"roll_w{w}_avg_score_diff"] = g["score_diff"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"roll_w{w}_avg_points_for"] = g["points_for"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"roll_w{w}_avg_points_against"] = g["points_against"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    return df


def _merge_rolling_back(game_df: pd.DataFrame, rolling_long: pd.DataFrame) -> pd.DataFrame:
    df = game_df.copy()

    roll_cols = [c for c in rolling_long.columns if c.startswith("roll_w")]
    base_cols = ["gameId", "team"] + roll_cols

    home_roll = rolling_long[base_cols].rename(columns={"team": "home_team"}).copy()
    home_roll = home_roll.rename(columns={c: f"home_{c}" for c in roll_cols})

    away_roll = rolling_long[base_cols].rename(columns={"team": "away_team"}).copy()
    away_roll = away_roll.rename(columns={c: f"away_{c}" for c in roll_cols})

    df = df.merge(home_roll, on=["gameId", "home_team"], how="left")
    df = df.merge(away_roll, on=["gameId", "away_team"], how="left")

    for c in roll_cols:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in df.columns and ac in df.columns:
            df[f"diff_{c}"] = _to_numeric_safe(df[hc]) - _to_numeric_safe(df[ac])

    return df


# ---------------------------
# ✅ ELO FEATURE ENGINEERING
# ---------------------------
def compute_elo_ratings(
    games_df: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 0.0,
) -> pd.DataFrame:
    """
    Adds:
      home_elo_before, away_elo_before
    Then updates ratings after each game.

    Expected score:
      E_home = 1 / (1 + 10^((R_away - (R_home + H))/400))
    Update:
      R_home += K * (S_home - E_home)
      R_away += K * (S_away - E_away)
    """
    req = ["game_date", "gameId", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in req if c not in games_df.columns]
    if missing:
        raise ValueError(f"ELO için eksik kolonlar: {missing}")

    df = games_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Sort chronologically (stable)
    df = df.sort_values(["game_date", "gameId"]).reset_index(drop=True)

    ratings: Dict[str, float] = {}

    home_before_list: List[float] = []
    away_before_list: List[float] = []

    def get_rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = float(base_rating)
        return ratings[team]

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        r_home = get_rating(home)
        r_away = get_rating(away)

        home_before_list.append(r_home)
        away_before_list.append(r_away)

        hs = row["home_score"]
        as_ = row["away_score"]

        # If scores missing, skip update but keep "before" ratings
        if pd.isna(hs) or pd.isna(as_):
            continue

        hs = float(hs)
        as_ = float(as_)

        # Actual outcome
        if hs > as_:
            s_home, s_away = 1.0, 0.0
        elif hs < as_:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Expected outcome (home advantage as Elo points)
        r_home_adj = r_home + float(home_advantage)
        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home_adj) / 400.0))
        e_away = 1.0 - e_home

        # Update
        ratings[home] = r_home + float(k_factor) * (s_home - e_home)
        ratings[away] = r_away + float(k_factor) * (s_away - e_away)

    df["home_elo_before"] = home_before_list
    df["away_elo_before"] = away_before_list
    df["diff_elo"] = df["home_elo_before"] - df["away_elo_before"]

    return df


def _select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ID_KEEP_COLS if c in df.columns]
    keep += ["home_team_win", "score_diff"]

    for c in ["month", "day_of_week", "is_weekend", "is_playoff"]:
        if c in df.columns:
            keep.append(c)

    # ELO
    for c in ["home_elo_before", "away_elo_before", "diff_elo"]:
        if c in df.columns:
            keep.append(c)

    diff_cols = [
        c for c in df.columns
        if c.startswith(("diff_team_", "diff_rest_", "diff_schedule_", "diff_roll_w"))
    ]
    keep += diff_cols

    roll_home_away = [c for c in df.columns if c.startswith(("home_roll_w", "away_roll_w"))]
    keep += roll_home_away

    # NOTE: Injury features removed (currently all-zero / sparse).
    # We will use injuries later at inference-time as a post-prediction adjustment.

    seen = set()
    keep_unique = []
    for c in keep:
        if c in df.columns and c not in seen:
            keep_unique.append(c)
            seen.add(c)

    return df[keep_unique].copy()


def _median_impute_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())
    return out


def build_model_dataset(
    master_csv: Union[str, Path] = "data_processed/master_merged.csv",
    output_csv: Union[str, Path] = "data_processed/model_dataset.csv",
    write_interim: bool = True,
) -> pd.DataFrame:
    master_csv = Path(master_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_csv, low_memory=False)

    df = _add_date_features(df, "game_date")
    df = _one_hot_season_type(df)

    df = _make_labels(df)

    # ✅ ELO BEFORE rolling (ELO uses chronological games)
    df = compute_elo_ratings(
        df,
        base_rating=1500.0,
        k_factor=20.0,
        home_advantage=0.0,  # istersen 50-100 arası deneyebilirsin
    )

    # rolling features
    long_df = _build_team_game_long(df)
    long_df = _compute_rolling_features(long_df)
    df = _merge_rolling_back(df, long_df)

    df = _build_diff_features(df)

    # drop raw scores after labels/rolling (ELO already computed)
    for c in ["home_score", "away_score"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # ✅ Optional interim output for "core features"
    if write_interim:
        interim_path = Path("data_interim") / "games_with_core_features.csv"
        interim_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(interim_path, index=False)
        print(f"[OK] games_with_core_features.csv written -> {interim_path} ({len(df)} rows, {len(df.columns)} cols)")

    model_df = _select_model_columns(df)
    model_df = model_df.dropna(subset=["home_team_win", "score_diff"]).reset_index(drop=True)
    model_df = _median_impute_numeric(model_df, exclude=["home_team_win", "score_diff"])

    model_df.to_csv(output_csv, index=False)
    print(f"[OK] model_dataset.csv written -> {output_csv} ({len(model_df)} rows, {len(model_df.columns)} cols)")
    return model_df


if __name__ == "__main__":
    build_model_dataset()
