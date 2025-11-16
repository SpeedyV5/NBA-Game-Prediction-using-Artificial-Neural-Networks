import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster, playergamelog
from pathlib import Path

def fetch_team_player_gamelogs(
    team_ids: list[int],
    season: str,
    output_dir: Path,
    save_combined: bool = True
) -> None:
    """
    Belirtilen NBA takım ID'lerine göre o takımın oyuncularının game log verilerini çeker
    ve CSV dosyasına kaydeder.

    Args:
        team_ids (list[int]): NBA takım ID'leri (örnek: [1610612747, 1610612755])
        season (str): Sezon (örnek: "2025-26")
        output_dir (Path): Verilerin kaydedileceği klasör
        save_combined (bool): Tüm oyuncuları tek bir dosyada birleştirip kaydetmek ister misin?
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_logs = []

    for team_id in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            players_df = roster.get_data_frames()[0]
            player_ids = players_df['PLAYER_ID'].tolist()
            team_abbr = teams.find_team_name_by_id(team_id)["abbreviation"]
        except Exception as e:
            print(f"[{team_id}] Roster alınamadı: {e}")
            continue

        for pid in player_ids:
            try:
                log = playergamelog.PlayerGameLog(player_id=pid, season=season)
                df = log.get_data_frames()[0]
                df["PLAYER_ID"] = pid
                df["TEAM_ID"] = team_id
                all_logs.append(df)
                time.sleep(0.6)
            except Exception as e:
                print(f"⛔ Oyuncu {pid} için hata: {e}")

    if all_logs:
        df_all = pd.concat(all_logs, ignore_index=True)
        if save_combined:
            path = output_dir / f"gamelogs_combined_{season.replace('-', '')}.csv"
            df_all.to_csv(path, index=False)
            print(f"✓ Tüm takımlar için kaydedildi → {path}")
        else:
            for tid in team_ids:
                team_df = df_all[df_all["TEAM_ID"] == tid]
                abbr = teams.find_team_name_by_id(tid)["abbreviation"]
                path = output_dir / f"gamelogs_{abbr}_{season.replace('-', '')}.csv"
                team_df.to_csv(path, index=False)
                print(f"✓ {abbr} için kaydedildi → {path}")
    else:
        print("Hiçbir veri çekilemedi.")


# Örnek kullanım:
# if __name__ == "__main__":
#     from pathlib import Path
#     team_ids = [1610612747, 1610612755]  # Lakers & Rockets
#     season = "2025-26"
#     output_dir = Path("data_raw/player_game_logs")
#     fetch_team_player_gamelogs(team_ids, season, output_dir)
