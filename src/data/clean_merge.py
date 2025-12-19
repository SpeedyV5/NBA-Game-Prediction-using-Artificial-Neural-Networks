"""
NBA Veri Temizleme ve Birleştirme Modülü

Bu modül, ham NBA verilerini temizleyip birleştirmek için fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Union
import warnings

warnings.filterwarnings("ignore")


def _resolve_repo_path(path_like: Union[str, Path]) -> Path:
    """
    Path resolution that works both from repo root and inside notebooks.
    Order:
    1) Absolute path as-is.
    2) Repo root (src/../../) + relative.
    3) Current working directory + relative.
    """
    p = Path(path_like)
    if p.is_absolute():
        return p

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / p
    if candidate.exists():
        return candidate

    return Path.cwd() / p


# NBA Takım İsmi Mapping Dictionary
TEAM_NAME_MAPPING = {
    # Kısaltmalar -> Tam İsimler
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "BRK": "Brooklyn Nets",
    "NJN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHO": "Charlotte Hornets",
    "CHH": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "GS": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NOH": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "SA": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
    "WIZ": "Washington Wizards",
    # Varyasyonlar -> Tam İsimler
    "Los Angeles Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "LA Lakers": "Los Angeles Lakers",
    "Lakers": "Los Angeles Lakers",
    "Boston Celtics": "Boston Celtics",
    "Celtics": "Boston Celtics",
    "Brooklyn Nets": "Brooklyn Nets",
    "Nets": "Brooklyn Nets",
    "New Jersey Nets": "Brooklyn Nets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Hornets": "Charlotte Hornets",
    "Charlotte Bobcats": "Charlotte Hornets",
    "Chicago Bulls": "Chicago Bulls",
    "Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Cavaliers": "Cleveland Cavaliers",
    "Cavs": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Mavericks": "Dallas Mavericks",
    "Mavs": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Pistons": "Detroit Pistons",
    "Golden State Warriors": "Golden State Warriors",
    "Warriors": "Golden State Warriors",
    "Houston Rockets": "Houston Rockets",
    "Rockets": "Houston Rockets",
    "Indiana Pacers": "Indiana Pacers",
    "Pacers": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers",
    "L.A. Clippers": "LA Clippers",
    "Clippers": "LA Clippers",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Grizzlies": "Memphis Grizzlies",
    "Miami Heat": "Miami Heat",
    "Heat": "Miami Heat",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Bucks": "Milwaukee Bucks",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "Timberwolves": "Minnesota Timberwolves",
    "Wolves": "Minnesota Timberwolves",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "Pelicans": "New Orleans Pelicans",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New York Knicks": "New York Knicks",
    "Knicks": "New York Knicks",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Thunder": "Oklahoma City Thunder",
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "Orlando Magic": "Orlando Magic",
    "Magic": "Orlando Magic",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
    "Sixers": "Philadelphia 76ers",
    "Phoenix Suns": "Phoenix Suns",
    "Suns": "Phoenix Suns",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Trail Blazers": "Portland Trail Blazers",
    "Blazers": "Portland Trail Blazers",
    "Sacramento Kings": "Sacramento Kings",
    "Kings": "Sacramento Kings",
    "San Antonio Spurs": "San Antonio Spurs",
    "Spurs": "San Antonio Spurs",
    "Toronto Raptors": "Toronto Raptors",
    "Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Jazz": "Utah Jazz",
    "Washington Wizards": "Washington Wizards",
    "Wizards": "Washington Wizards",
    "Atlanta Hawks": "Atlanta Hawks",
    "Hawks": "Atlanta Hawks",
    # NBAstuffer formatı (sadece şehir/ekip ismi)
    "Atlanta": "Atlanta Hawks",
    "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons",
    "Golden State": "Golden State Warriors",
    "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves",
    "New Orleans": "New Orleans Pelicans",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings",
    "San Antonio": "San Antonio Spurs",
    "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz",
    "Washington": "Washington Wizards",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("%", "pct", regex=False)
        .str.replace(r"[\s\-\/]+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
    )
    return df


def stable_game_id(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:12], 16)


def normalize_date(date_col: pd.Series, format_hints: Optional[List[str]] = None) -> pd.Series:
    if format_hints is None:
        format_hints = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%Y.%m.%d",
            "%m.%d.%Y",
            "%d.%m.%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        ]

    if pd.api.types.is_datetime64_any_dtype(date_col):
        return date_col

    date_str = date_col.astype(str)
    date_str = date_str.replace(["nan", "None", "NaT", ""], np.nan)

    result = pd.Series(index=date_col.index, dtype="datetime64[ns]")

    for fmt in format_hints:
        mask = result.isna()
        if not mask.any():
            break
        try:
            parsed = pd.to_datetime(date_str[mask], format=fmt, errors="coerce")
            result.loc[mask] = parsed
        except Exception:
            continue

    if result.isna().any():
        mask = result.isna()
        try:
            parsed = pd.to_datetime(date_str[mask], errors="coerce", infer_datetime_format=True)
            result.loc[mask] = parsed
        except Exception:
            pass

    return result


def standardize_team_names(df: pd.DataFrame, team_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()

    if team_cols is None:
        possible_cols = [
            "team",
            "team_name",
            "team_abbr",
            "team_id",
            "TEAM",
            "home_team",
            "away_team",
            "team_home",
            "team_away",
            "home",
            "away",
            "home_team_name",
            "away_team_name",
            "team_name_home",
            "team_name_away",
        ]
        team_cols = [col for col in possible_cols if col in df.columns]

    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].map(TEAM_NAME_MAPPING).fillna(df[col])

            remaining = df[col].unique()
            for val in remaining:
                if val not in TEAM_NAME_MAPPING.values():
                    val_lower = str(val).lower()
                    for key, mapped_val in TEAM_NAME_MAPPING.items():
                        if str(key).lower() == val_lower:
                            df.loc[df[col] == val, col] = mapped_val
                            break

    return df


def clean_game_data(
    df: pd.DataFrame,
    score_cols: Optional[List[str]] = None,
    date_col: str = "date",
    min_score: int = 0,
    max_score: int = 200,
) -> pd.DataFrame:
    df = df.copy()
    original_len = len(df)

    if score_cols is None:
        possible_score_cols = [
            "home_score",
            "away_score",
            "pts_home",
            "pts_away",
            "score_home",
            "score_away",
            "home_pts",
            "away_pts",
            "team_score",
            "opponent_score",
            "points",
            "opp_points",
            "PTS",
        ]
        score_cols = [col for col in possible_score_cols if col in df.columns]

    if score_cols:
        df = df.dropna(subset=score_cols)

    if date_col in df.columns:
        df[date_col] = normalize_date(df[date_col])
        df = df.dropna(subset=[date_col])

    if score_cols:
        for col in score_cols:
            df = df[df[col] >= min_score]
            df = df[df[col] <= max_score]

    if len(score_cols) >= 2:
        mask = (df[score_cols[0]] == 0) & (df[score_cols[1]] == 0)
        df = df[~mask]

    if "gameId" in df.columns:
        df = df.drop_duplicates(subset=["gameId"], keep="first")
    elif date_col in df.columns and "home_team" in df.columns and "away_team" in df.columns:
        df = df.drop_duplicates(subset=[date_col, "home_team", "away_team"], keep="first")

    cleaned_len = len(df)
    removed = original_len - cleaned_len
    if removed > 0:
        print(f"Temizleme: {removed} satır kaldırıldı ({original_len} -> {cleaned_len})")

    return df


def convert_boxscore_to_game_level(boxscore_df: pd.DataFrame) -> pd.DataFrame:
    df = boxscore_df.copy()

    required_cols = ["gameId", "game_date", "matchup", "teamTricode", "points", "season_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik sütunlar: {missing_cols}")

    def colsum(team_df: pd.DataFrame, col: str) -> float:
        return float(team_df[col].sum()) if col in team_df.columns else 0.0

    games_list = []

    for game_id, game_data in df.groupby("gameId", sort=False):
        game_data = game_data.copy()

        first_row = game_data.iloc[0]
        matchup = str(first_row["matchup"]).strip()
        game_date = first_row["game_date"]
        season_year = first_row["season_year"] if "season_year" in game_data.columns else ""
        season_type = first_row["season_type"]

        home_team_code = None
        away_team_code = None

        if "@" in matchup:
            parts = matchup.split("@")
            away_team_code = parts[0].strip()
            home_team_code = parts[1].strip()
        elif "vs." in matchup or "vs" in matchup:
            parts = matchup.replace("vs.", "vs").split("vs")
            if len(parts) == 2:
                home_team_code = parts[0].strip()
                away_team_code = parts[1].strip()

        if not home_team_code or not away_team_code:
            teams = game_data["teamTricode"].dropna().unique()
            if len(teams) == 2:
                home_team_code = teams[0]
                away_team_code = teams[1]
            else:
                continue

        home_team_data = game_data[game_data["teamTricode"] == home_team_code]
        away_team_data = game_data[game_data["teamTricode"] == away_team_code]

        home_score = float(home_team_data["points"].sum()) if len(home_team_data) > 0 else 0.0
        away_score = float(away_team_data["points"].sum()) if len(away_team_data) > 0 else 0.0

        if home_score == 0.0 and away_score == 0.0 and "fieldGoalsMade" in game_data.columns:
            home_score = colsum(home_team_data, "fieldGoalsMade") * 2 + colsum(home_team_data, "threePointersMade") + colsum(home_team_data, "freeThrowsMade")
            away_score = colsum(away_team_data, "fieldGoalsMade") * 2 + colsum(away_team_data, "threePointersMade") + colsum(away_team_data, "freeThrowsMade")

        home_team_name = TEAM_NAME_MAPPING.get(home_team_code, home_team_code)
        away_team_name = TEAM_NAME_MAPPING.get(away_team_code, away_team_code)

        games_list.append(
            {
                "gameId": game_id,
                "game_date": game_date,
                "season_year": season_year,
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_team_code": home_team_code,
                "away_team_code": away_team_code,
                "home_score": home_score,
                "away_score": away_score,
                "season_type": season_type,
                "matchup": matchup,
            }
        )

    games_df = pd.DataFrame(games_list)
    print(f"Box score'dan {len(games_df)} maç oluşturuldu (orijinal {len(df)} satırdan)")
    return games_df


# -----------------------------
# ✅ NEW: LeagueGameLog -> game-level extractor (tüm lig)
# -----------------------------
def extract_games_from_league_gamelog(league_gamelog_path: Union[str, Path]) -> pd.DataFrame:
    """
    LeagueGameLog (team-level, game başına 2 satır) CSV'sini okuyup game-level formata çevirir.

    Beklenen kolonlar (nba_api LeagueGameLog standart):
      GAME_ID, GAME_DATE, TEAM_ABBREVIATION, MATCHUP, PTS
    """
    league_gamelog_path = _resolve_repo_path(league_gamelog_path)
    if not league_gamelog_path.exists():
        print(f"⚠ LeagueGameLog dosyası bulunamadı: {league_gamelog_path}")
        return pd.DataFrame()

    lg = pd.read_csv(league_gamelog_path, low_memory=False)
    lg = normalize_columns(lg)

    needed = {"GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS"}
    if not needed.issubset(set(lg.columns)):
        print(f"⚠ LeagueGameLog beklenen kolonları içermiyor. Var olan: {list(lg.columns)}")
        return pd.DataFrame()

    lg["GAME_DATE"] = lg["GAME_DATE"].astype(str).str.strip('"')
    lg["GAME_DATE"] = normalize_date(lg["GAME_DATE"])
    lg["PTS"] = pd.to_numeric(lg["PTS"], errors="coerce")

    games = []
    for gid, g in lg.groupby("GAME_ID", sort=False):
        if len(g) < 2:
            continue

        # '@' içeren satır away takım satırıdır (genelde)
        g_at = g[g["MATCHUP"].astype(str).str.contains("@", na=False)]
        if len(g_at) >= 1:
            away_code = str(g_at.iloc[0]["TEAM_ABBREVIATION"]).strip()
            other = g[g["TEAM_ABBREVIATION"] != away_code]
            if len(other) == 0:
                continue
            home_code = str(other.iloc[0]["TEAM_ABBREVIATION"]).strip()
        else:
            # fallback: vs olan satır home olabilir, ama iki satırdan birini home kabul edip diğerini away alalım
            home_code = str(g.iloc[0]["TEAM_ABBREVIATION"]).strip()
            away_code = str(g.iloc[1]["TEAM_ABBREVIATION"]).strip()

        home_row = g[g["TEAM_ABBREVIATION"] == home_code]
        away_row = g[g["TEAM_ABBREVIATION"] == away_code]
        if len(home_row) == 0 or len(away_row) == 0:
            continue

        game_date = home_row.iloc[0]["GAME_DATE"]
        home_score = float(home_row.iloc[0]["PTS"]) if pd.notna(home_row.iloc[0]["PTS"]) else np.nan
        away_score = float(away_row.iloc[0]["PTS"]) if pd.notna(away_row.iloc[0]["PTS"]) else np.nan

        home_team = TEAM_NAME_MAPPING.get(home_code, home_code)
        away_team = TEAM_NAME_MAPPING.get(away_code, away_code)

        games.append(
            {
                "gameId": stable_game_id(str(gid)),
                "game_date": game_date,
                "season_year": "2025-26",
                "season_type": "regular",
                "home_team": home_team,
                "away_team": away_team,
                "home_team_code": home_code,
                "away_team_code": away_code,
                "home_score": home_score,
                "away_score": away_score,
                "matchup": f"{away_code} @ {home_code}",
            }
        )

    out = pd.DataFrame(games)
    out = out.drop_duplicates(subset=["game_date", "home_team", "away_team"]).reset_index(drop=True)
    print(f"LeagueGameLog'dan {len(out)} maç çıkarıldı")
    return out


def merge_nocturnebear_data(
    base_path: Union[str, Path] = "data_raw/nocturnebear_2010_2024/",
    convert_to_game_level: bool = True,
) -> pd.DataFrame:
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"NocturneBear veri klasörü bulunamadı: {base_path}")

    all_dataframes = []

    for part_num in [1, 2, 3]:
        file_path = base_path / f"regular_season_box_scores_2010_2024_part_{part_num}.csv"
        if file_path.exists():
            print(f"Yükleniyor: {file_path.name}")
            dfx = pd.read_csv(file_path, low_memory=False)
            dfx["season_type"] = "regular"
            all_dataframes.append(dfx)
        else:
            print(f"Uyarı: {file_path.name} bulunamadı")

    playoff_file = base_path / "play_off_box_scores_2010_2024.csv"
    if playoff_file.exists():
        print(f"Yükleniyor: {playoff_file.name}")
        dfx = pd.read_csv(playoff_file, low_memory=False)
        dfx["season_type"] = "playoff"
        all_dataframes.append(dfx)
    else:
        print("Uyarı: play_off_box_scores_2010_2024.csv bulunamadı")

    if not all_dataframes:
        raise ValueError("Hiçbir veri dosyası bulunamadı!")

    print("DataFrame'ler birleştiriliyor...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"Toplam {len(merged_df)} satır birleştirildi (box score formatında)")
    print(f"Sütunlar: {list(merged_df.columns)}")

    if convert_to_game_level:
        print("\nBox score verisi maç bazlı formata dönüştürülüyor...")
        merged_df = convert_boxscore_to_game_level(merged_df)
        print(f"Maç bazlı format: {len(merged_df)} satır")
        print(f"Sütunlar: {list(merged_df.columns)}")

    return merged_df


def merge_team_stats(
    game_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    team_col_in_stats: str = "TEAM",
) -> pd.DataFrame:
    game_df = game_df.copy()
    team_stats_df = team_stats_df.copy()

    if home_team_col not in game_df.columns or away_team_col not in game_df.columns:
        print("⚠ Uyarı: home_team/away_team sütunları bulunamadı. Team stats merge atlanıyor.")
        return game_df

    game_df = standardize_team_names(game_df, [home_team_col, away_team_col])
    team_stats_df = standardize_team_names(team_stats_df, [team_col_in_stats])

    excluded_cols = [team_col_in_stats, "RANK"]
    stat_cols = [
        c
        for c in team_stats_df.columns
        if c not in excluded_cols and not team_stats_df[c].isna().all()
    ]

    home_stats = team_stats_df.copy().rename(columns={team_col_in_stats: home_team_col})
    home_stats.columns = [f"home_team_{c}" if c in stat_cols else c for c in home_stats.columns]
    game_df = game_df.merge(home_stats, on=home_team_col, how="left")

    away_stats = team_stats_df.copy().rename(columns={team_col_in_stats: away_team_col})
    away_stats.columns = [f"away_team_{c}" if c in stat_cols else c for c in away_stats.columns]
    game_df = game_df.merge(away_stats, on=away_team_col, how="left")

    print(f"Takım istatistikleri birleştirildi. Toplam sütun sayısı: {len(game_df.columns)}")
    if stat_cols:
        sample_col = f"home_team_{stat_cols[0]}"
        if sample_col in game_df.columns:
            merged_count = game_df[sample_col].notna().sum()
            total_count = len(game_df)
            print(f"Merge başarı oranı: {merged_count}/{total_count} ({merged_count/total_count*100:.1f}%)")

    return game_df


def merge_rest_days_stats(game_df: pd.DataFrame, rest_days_path: Union[str, Path]) -> pd.DataFrame:
    rest_days_path = _resolve_repo_path(rest_days_path)
    if not rest_days_path.exists():
        print(f"⚠ Rest days stats dosyası bulunamadı: {rest_days_path}")
        return game_df

    game_df = game_df.copy()
    rest_days_df = pd.read_csv(rest_days_path, low_memory=False)
    rest_days_df = normalize_columns(rest_days_df)

    if "TEAM_NAME" in rest_days_df.columns:
        team_col = "TEAM_NAME"
    elif "TEAM" in rest_days_df.columns:
        team_col = "TEAM"
    else:
        print("⚠ Rest days stats'te takım sütunu bulunamadı (TEAM_NAME/TEAM).")
        return game_df

    rest_days_df = standardize_team_names(rest_days_df, [team_col])

    drop_cols = [c for c in ["RANK", "OPPONENT_TODAY"] if c in rest_days_df.columns]
    if drop_cols:
        rest_days_df = rest_days_df.drop(columns=drop_cols)

    stat_cols = [c for c in rest_days_df.columns if c != team_col and not rest_days_df[c].isna().all()]

    home_rest = rest_days_df.copy().rename(columns={team_col: "home_team"})
    home_rest.columns = [f"home_rest_{c}" if c in stat_cols else c for c in home_rest.columns]
    game_df = game_df.merge(home_rest, on="home_team", how="left")

    away_rest = rest_days_df.copy().rename(columns={team_col: "away_team"})
    away_rest.columns = [f"away_rest_{c}" if c in stat_cols else c for c in away_rest.columns]
    game_df = game_df.merge(away_rest, on="away_team", how="left")

    print(f"Rest days istatistikleri birleştirildi. Toplam sütun sayısı: {len(game_df.columns)}")
    return game_df


def merge_schedule_rest_days_stats(game_df: pd.DataFrame, schedule_rest_days_path: Union[str, Path]) -> pd.DataFrame:
    schedule_rest_days_path = _resolve_repo_path(schedule_rest_days_path)
    if not schedule_rest_days_path.exists():
        print(f"⚠ Schedule rest days dosyası bulunamadı: {schedule_rest_days_path}")
        return game_df

    game_df = game_df.copy()
    schedule_df = pd.read_csv(schedule_rest_days_path, low_memory=False)
    schedule_df = normalize_columns(schedule_df)

    if "TEAMS" in schedule_df.columns:
        team_col = "TEAMS"
    elif "TEAM" in schedule_df.columns:
        team_col = "TEAM"
    elif "TEAM_NAME" in schedule_df.columns:
        team_col = "TEAM_NAME"
    else:
        print("⚠ Schedule rest days'te takım sütunu bulunamadı (TEAMS/TEAM/TEAM_NAME).")
        return game_df

    schedule_df = standardize_team_names(schedule_df, [team_col])

    drop_cols = [c for c in ["RANK"] if c in schedule_df.columns]
    if drop_cols:
        schedule_df = schedule_df.drop(columns=drop_cols)

    stat_cols = [c for c in schedule_df.columns if c != team_col and not schedule_df[c].isna().all()]

    home_schedule = schedule_df.copy().rename(columns={team_col: "home_team"})
    home_schedule.columns = [f"home_schedule_{c}" if c in stat_cols else c for c in home_schedule.columns]
    game_df = game_df.merge(home_schedule, on="home_team", how="left")

    away_schedule = schedule_df.copy().rename(columns={team_col: "away_team"})
    away_schedule.columns = [f"away_schedule_{c}" if c in stat_cols else c for c in away_schedule.columns]
    game_df = game_df.merge(away_schedule, on="away_team", how="left")

    print(f"Schedule rest days birleştirildi. Toplam sütun sayısı: {len(game_df.columns)}")
    return game_df


def create_master_merged(
    output_dir: Union[str, Path] = "data_processed/",
    nocturnebear_path: Union[str, Path] = "data_raw/nocturnebear_2010_2024/",
    team_stats_path: Union[str, Path] = "data_raw/nbastuffer_2025_2026_team_stats_raw.csv",
    bdl_games_path: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    output_dir = _resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}

    print("=" * 60)
    print("1. NocturneBear Verilerini Yükleme ve Birleştirme")
    print("=" * 60)
    nocturnebear_df = merge_nocturnebear_data(_resolve_repo_path(nocturnebear_path), convert_to_game_level=True)

    date_col = "game_date" if "game_date" in nocturnebear_df.columns else None
    if date_col:
        nocturnebear_df[date_col] = normalize_date(nocturnebear_df[date_col])
    else:
        print("Uyarı: Tarih sütunu bulunamadı!")

    print("\nTakım isimleri standartlaştırılıyor...")
    nocturnebear_df = standardize_team_names(nocturnebear_df, ["home_team", "away_team"])

    print("\nVeri temizleniyor...")
    nocturnebear_df = clean_game_data(nocturnebear_df, date_col=date_col if date_col else "date")

    # 2. BDL (opsiyonel)
    if bdl_games_path:
        bdl_games_path = _resolve_repo_path(bdl_games_path)
    if bdl_games_path and bdl_games_path.exists():
        print("\n" + "=" * 60)
        print("2. BDL Games 2025 Verilerini Yükleme")
        print("=" * 60)
        bdl_df = pd.read_csv(bdl_games_path, low_memory=False)
        bdl_df = normalize_columns(bdl_df)
        bdl_df["season_type"] = "regular"

        if date_col and date_col in bdl_df.columns:
            bdl_df[date_col] = normalize_date(bdl_df[date_col])

        bdl_df = standardize_team_names(bdl_df)
        bdl_df = clean_game_data(bdl_df, date_col=date_col if date_col else "date")
        nocturnebear_df = pd.concat([nocturnebear_df, bdl_df], ignore_index=True)
        print(f"BDL verileri eklendi. Toplam satır: {len(nocturnebear_df)}")

    # ✅ 2b. LeagueGameLog 2025-26 (tüm lig)
    league_gamelog_path = _resolve_repo_path("data_raw/league_gamelog_2025_26.csv")
    if league_gamelog_path.exists():
        print("\n" + "=" * 60)
        print("2b. LeagueGameLog'dan 2025-26 Verilerini Çıkarma (TÜM LİG)")
        print("=" * 60)
        try:
            league_games = extract_games_from_league_gamelog(league_gamelog_path)
            if len(league_games) > 0:
                league_games = standardize_team_names(league_games, ["home_team", "away_team"])
                league_games = clean_game_data(league_games, date_col="game_date")

                nocturnebear_df = pd.concat([nocturnebear_df, league_games], ignore_index=True)
                print(f"LeagueGameLog verileri eklendi. Toplam satır: {len(nocturnebear_df)}")
            else:
                print("⚠ LeagueGameLog'dan maç çıkarılamadı (0 satır).")
        except Exception as e:
            print(f"⚠ LeagueGameLog işlenirken hata: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ LeagueGameLog dosyası bulunamadı: {league_gamelog_path}")

    # 3. Team stats
    team_stats_path = _resolve_repo_path(team_stats_path)
    if team_stats_path.exists():
        print("\n" + "=" * 60)
        print("3. Takım İstatistiklerini Yükleme ve Birleştirme")
        print("=" * 60)

        team_stats_df = pd.read_csv(team_stats_path, low_memory=False)
        team_stats_df = normalize_columns(team_stats_df)

        if "TEAM" in team_stats_df.columns:
            team_col = "TEAM"
        elif "TEAM_NAME" in team_stats_df.columns:
            team_col = "TEAM_NAME"
        else:
            print("⚠ Uyarı: Team stats dosyasında TEAM/TEAM_NAME kolonu yok. Merge atlanıyor.")
            team_col = None

        if team_col:
            team_stats_df = standardize_team_names(team_stats_df, [team_col])
            print(f"  Örnek takım isimleri: {list(team_stats_df[team_col].unique()[:5])}")

            nocturnebear_df = merge_team_stats(
                nocturnebear_df,
                team_stats_df,
                home_team_col="home_team",
                away_team_col="away_team",
                team_col_in_stats=team_col,
            )
    else:
        print(f"\nUyarı: Takım istatistikleri dosyası bulunamadı: {team_stats_path}")

    # 4. Rest days
    rest_days_path = _resolve_repo_path("data_raw/nbastuffer_2025_2026_rest_days_stats.csv")
    if rest_days_path.exists():
        print("\n" + "=" * 60)
        print("4. Rest Days İstatistiklerini Birleştirme")
        print("=" * 60)
        nocturnebear_df = merge_rest_days_stats(nocturnebear_df, rest_days_path)
    else:
        print(f"\nUyarı: Rest days stats dosyası bulunamadı: {rest_days_path}")

    # 4b. Schedule rest days
    schedule_rest_days_path = _resolve_repo_path("data_raw/nbastuffer_2025_2026_schedule_rest_days.csv")
    if schedule_rest_days_path.exists():
        print("\n" + "=" * 60)
        print("4b. Schedule Rest Days İstatistiklerini Birleştirme")
        print("=" * 60)
        nocturnebear_df = merge_schedule_rest_days_stats(nocturnebear_df, schedule_rest_days_path)
    else:
        print(f"\nUyarı: Schedule rest days dosyası bulunamadı: {schedule_rest_days_path}")

    # Tarihe göre sırala
    if date_col and date_col in nocturnebear_df.columns:
        nocturnebear_df = nocturnebear_df.sort_values(by=date_col).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("5. Çıktı Dosyalarını Oluşturma")
    print("=" * 60)

    long_term_df = nocturnebear_df.copy()
    long_term_df.to_csv(output_dir / "long_term_2010_2025.csv", index=False)
    print(f"[OK] long_term_2010_2025.csv kaydedildi: {len(long_term_df)} satır")
    results["long_term"] = long_term_df

    if date_col and date_col in nocturnebear_df.columns:
        season_2025_df = nocturnebear_df[nocturnebear_df[date_col] >= pd.Timestamp("2025-01-01")].copy()
        if len(season_2025_df) > 0:
            season_2025_df.to_csv(output_dir / "season_2025_current.csv", index=False)
            print(f"[OK] season_2025_current.csv kaydedildi: {len(season_2025_df)} satır")
            results["season_2025"] = season_2025_df
        else:
            print("⚠ 2025 sezonu verisi bulunamadı")
    else:
        print("⚠ Tarih sütunu olmadığı için season_2025_current.csv oluşturulamadı")

    master_df = nocturnebear_df.copy()
    master_df.to_csv(output_dir / "master_merged.csv", index=False)
    print(f"[OK] master_merged.csv kaydedildi: {len(master_df)} satır, {len(master_df.columns)} sütun")
    results["master"] = master_df

    print("\n" + "=" * 60)
    print("Tamamlandı!")
    print("=" * 60)

    return results
