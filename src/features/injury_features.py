"""
NBA Injury Features Module

Bu modül, injury verilerini işleyip feature'lara dönüştürür.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np


# NBA Takım İsmi Mapping (kısa kodlar -> tam isimler)
TEAM_ABBR_MAPPING = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'BRK': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'GS': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'PHO': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'SA': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}

# Takım ismi varyasyonları -> standart isimler
TEAM_NAME_VARIANTS = {
    'LAClippers': 'LA Clippers',
    'LosAngelesClippers': 'LA Clippers',
    'LosAngelesLakers': 'Los Angeles Lakers',
    'BostonCeltics': 'Boston Celtics',
    'BrooklynNets': 'Brooklyn Nets',
    'SacramentoKings': 'Sacramento Kings',
    'SanAntonioSpurs': 'San Antonio Spurs',
    'GoldenStateWarriors': 'Golden State Warriors',
    'NewOrleansPelicans': 'New Orleans Pelicans',
    'OrlandoMagic': 'Orlando Magic',
    'HoustonRockets': 'Houston Rockets',
    'PortlandTrailBlazers': 'Portland Trail Blazers',
    'DallasMavericks': 'Dallas Mavericks',
    'AtlantaHawks': 'Atlanta Hawks',
    'PhoenixSuns': 'Phoenix Suns',
    'ChicagoBulls': 'Chicago Bulls',
    'UtahJazz': 'Utah Jazz',
    'WashingtonWizards': 'Washington Wizards',
    'MilwaukeeBucks': 'Milwaukee Bucks',
    'MiamiHeat': 'Miami Heat',
    'IndiaPacers': 'Indiana Pacers',
    'DetroitPistons': 'Detroit Pistons',
    'ClevelandCavaliers': 'Cleveland Cavaliers',
    'NewYorkKnicks': 'New York Knicks',
    'DenverNuggets': 'Denver Nuggets',
    'MemphisGrizzlies': 'Memphis Grizzlies',
    'MinnesotaTimberwolves': 'Minnesota Timberwolves',
    'OklahomaCityThunder': 'Oklahoma City Thunder',
    'TorontoRaptors': 'Toronto Raptors',
    'CharlotteHornets': 'Charlotte Hornets',
    'Philadelphia76ers': 'Philadelphia 76ers',
}


def normalize_team_name(name: str) -> str:
    """Takım ismini standart formata dönüştür."""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # Önce tam eşleşme dene
    if name in TEAM_ABBR_MAPPING:
        return TEAM_ABBR_MAPPING[name]
    
    if name in TEAM_NAME_VARIANTS:
        return TEAM_NAME_VARIANTS[name]
    
    # Case-insensitive kontrol
    name_lower = name.lower()
    for key, val in TEAM_NAME_VARIANTS.items():
        if key.lower() == name_lower:
            return val
    
    return name


def parse_injury_report_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Bozuk formattaki injury CSV dosyasını düzgün parse et.
    
    Orijinal PDF parsing'den gelen bozuk CSV formatını düzeltir.
    
    Parameters
    ----------
    csv_path : str or Path
        Injury report CSV dosyasının yolu
        
    Returns
    -------
    pd.DataFrame
        Temizlenmiş injury dataframe
        Kolonlar: date, player_name, team, status, reason
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"⚠️ Injury CSV bulunamadı: {csv_path}")
        return pd.DataFrame(columns=['date', 'player_name', 'team', 'status', 'reason'])
    
    # Raw text olarak oku (farklı encoding'leri dene)
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        # Hiçbiri çalışmazsa errors='ignore' ile utf-8 kullan
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    
    records = []
    current_date = None
    current_team = None
    
    for line in lines[1:]:  # Header'ı atla
        line = line.strip()
        if not line:
            continue
        
        # Tarih pattern'i: MM/DD/YYYY HH:MM(ET)
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
        if date_match:
            current_date = date_match.group(1)
        
        # Matchup pattern'i: AWAY@HOME
        matchup_match = re.search(r'([A-Z]{2,3})@([A-Z]{2,3})', line)
        if matchup_match:
            away_team = matchup_match.group(1)
            home_team = matchup_match.group(2)
        
        # Takım ismi pattern'i (bitişik yazılmış)
        team_patterns = [
            (r'LAClippers', 'LA Clippers'),
            (r'BostonCeltics', 'Boston Celtics'),
            (r'BrooklynNets', 'Brooklyn Nets'),
            (r'SacramentoKings', 'Sacramento Kings'),
            (r'SanAntonioSpurs', 'San Antonio Spurs'),
            (r'GoldenStateWarriors', 'Golden State Warriors'),
            (r'NewOrleansPelicans', 'New Orleans Pelicans'),
            (r'OrlandoMagic', 'Orlando Magic'),
            (r'HoustonRockets', 'Houston Rockets'),
            (r'PortlandTrailBlazers', 'Portland Trail Blazers'),
            (r'DallasMavericks', 'Dallas Mavericks'),
            (r'AtlantaHawks', 'Atlanta Hawks'),
            (r'PhoenixSuns', 'Phoenix Suns'),
            (r'ChicagoBulls', 'Chicago Bulls'),
            (r'UtahJazz', 'Utah Jazz'),
            (r'WashingtonWizards', 'Washington Wizards'),
            (r'MilwaukeeBucks', 'Milwaukee Bucks'),
            (r'MiamiHeat', 'Miami Heat'),
        ]
        
        for pattern, team_name in team_patterns:
            if re.search(pattern, line):
                current_team = team_name
                break
        
        # Oyuncu ismi ve status pattern'i: "Lastname,Firstname Status" veya "Lastname,Firstname Status"
        # Örnek: "Leonard,Kawhi Out" veya "Wembanyama,Victor Questionable"
        player_patterns = [
            r"([A-Za-z'\-]+),([A-Za-z'\-\.]+)\s+(Out|Questionable|Doubtful|Probable|Available)",
        ]
        
        for pattern in player_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                last_name, first_name, status = match
                player_name = f"{first_name} {last_name}"
                
                # Reason'ı bul
                reason = ""
                reason_match = re.search(rf"{status}\s*[,\-]?\s*(.+?)(?:,|$)", line)
                if reason_match:
                    reason = reason_match.group(1).strip()
                
                if current_date and current_team:
                    records.append({
                        'date': current_date,
                        'player_name': player_name.strip(),
                        'team': current_team,
                        'status': status,
                        'reason': reason
                    })
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        # Tarih formatını düzelt
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        
        # Duplicate'leri kaldır
        df = df.drop_duplicates(subset=['date', 'player_name', 'team']).reset_index(drop=True)
    
    return df


def load_and_clean_injury_reports(
    raw_dir: Union[str, Path] = "data_raw/injury_reports_raw/",
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Tüm injury report dosyalarını yükle ve birleştir.
    
    Parameters
    ----------
    raw_dir : str or Path
        Raw injury report dosyalarının bulunduğu klasör
    output_path : str or Path, optional
        Temizlenmiş CSV'nin kaydedileceği yol
        
    Returns
    -------
    pd.DataFrame
        Birleştirilmiş ve temizlenmiş injury dataframe
    """
    raw_dir = Path(raw_dir)
    
    if not raw_dir.exists():
        print(f"⚠️ Injury klasörü bulunamadı: {raw_dir}")
        return pd.DataFrame(columns=['date', 'player_name', 'team', 'status', 'reason'])
    
    all_dfs = []
    
    # Tüm parsed CSV dosyalarını bul
    csv_files = list(raw_dir.glob("*.parsed.csv"))
    
    if not csv_files:
        print(f"⚠️ Hiç parsed CSV dosyası bulunamadı: {raw_dir}")
        return pd.DataFrame(columns=['date', 'player_name', 'team', 'status', 'reason'])
    
    for csv_file in csv_files:
        print(f"  Yükleniyor: {csv_file.name}")
        df = parse_injury_report_csv(csv_file)
        if len(df) > 0:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame(columns=['date', 'player_name', 'team', 'status', 'reason'])
    
    # Birleştir
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Duplicate'leri kaldır
    combined_df = combined_df.drop_duplicates(
        subset=['date', 'player_name', 'team']
    ).reset_index(drop=True)
    
    # Takım isimlerini standartlaştır
    combined_df['team'] = combined_df['team'].apply(normalize_team_name)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Injury reports temizlendi ve kaydedildi: {output_path}")
        print(f"   Toplam {len(combined_df)} injury kaydı")
    
    return combined_df


def load_player_minutes(
    player_stats_path: Union[str, Path] = "data_raw/nbastuffer_2025_2026_player_stats_raw.csv"
) -> pd.DataFrame:
    """
    Oyuncu dakika istatistiklerini yükle.
    
    Parameters
    ----------
    player_stats_path : str or Path
        Player stats CSV dosyasının yolu
        
    Returns
    -------
    pd.DataFrame
        player_name, team, avg_minutes_per_game kolonları içeren dataframe
    """
    player_stats_path = Path(player_stats_path)
    
    if not player_stats_path.exists():
        print(f"⚠️ Player stats dosyası bulunamadı: {player_stats_path}")
        return pd.DataFrame(columns=['player_name', 'team', 'avg_minutes_per_game'])
    
    df = pd.read_csv(player_stats_path, low_memory=False)
    
    # Kolon isimlerini kontrol et
    if 'NAME' not in df.columns or 'MpG' not in df.columns:
        print(f"⚠️ Beklenen kolonlar bulunamadı. Mevcut kolonlar: {list(df.columns)}")
        return pd.DataFrame(columns=['player_name', 'team', 'avg_minutes_per_game'])
    
    # Gerekli kolonları seç ve yeniden adlandır
    result = df[['NAME', 'TEAM', 'MpG']].copy()
    result.columns = ['player_name', 'team', 'avg_minutes_per_game']
    
    # MpG'yi numeric yap
    result['avg_minutes_per_game'] = pd.to_numeric(result['avg_minutes_per_game'], errors='coerce')
    
    # Takım isimlerini standartlaştır
    # NBAstuffer formatı: "Hou", "Okc", etc. -> tam isimler
    team_mapping_short = {
        'Hou': 'Houston Rockets',
        'Okc': 'Oklahoma City Thunder',
        'Gol': 'Golden State Warriors',
        'Lal': 'Los Angeles Lakers',
        'Cle': 'Cleveland Cavaliers',
        'Nyk': 'New York Knicks',
        'Bro': 'Brooklyn Nets',
        'Cha': 'Charlotte Hornets',
        'Mia': 'Miami Heat',
        'Orl': 'Orlando Magic',
        'Tor': 'Toronto Raptors',
        'Atl': 'Atlanta Hawks',
        'Phi': 'Philadelphia 76ers',
        'Bos': 'Boston Celtics',
        'Det': 'Detroit Pistons',
        'Chi': 'Chicago Bulls',
        'Nor': 'New Orleans Pelicans',
        'Mem': 'Memphis Grizzlies',
        'Was': 'Washington Wizards',
        'Mil': 'Milwaukee Bucks',
        'Lac': 'LA Clippers',
        'Uta': 'Utah Jazz',
        'San': 'San Antonio Spurs',
        'Dal': 'Dallas Mavericks',
        'Sac': 'Sacramento Kings',
        'Pho': 'Phoenix Suns',
        'Min': 'Minnesota Timberwolves',
        'Por': 'Portland Trail Blazers',
        'Ind': 'Indiana Pacers',
        'Den': 'Denver Nuggets',
    }
    
    result['team'] = result['team'].map(team_mapping_short).fillna(result['team'])
    
    # NaN değerleri temizle
    result = result.dropna(subset=['player_name', 'avg_minutes_per_game'])
    
    print(f"✅ Player minutes yüklendi: {len(result)} oyuncu")
    
    return result


def add_injury_features(
    games_df: pd.DataFrame,
    injury_df: pd.DataFrame,
    player_minutes_df: pd.DataFrame,
    key_player_minutes_threshold: float = 25.0
) -> pd.DataFrame:
    """
    Maç verisine injury feature'larını ekle.
    
    Parameters
    ----------
    games_df : pd.DataFrame
        Maç verisi (game_date, home_team, away_team kolonları gerekli)
    injury_df : pd.DataFrame
        Injury verisi (date, player_name, team, status kolonları gerekli)
    player_minutes_df : pd.DataFrame
        Oyuncu dakika verisi (player_name, team, avg_minutes_per_game)
    key_player_minutes_threshold : float
        "Key player" olarak kabul edilecek minimum dakika eşiği
        
    Returns
    -------
    pd.DataFrame
        Injury feature'ları eklenmiş maç verisi
        Yeni kolonlar:
        - expected_minutes_lost_home/away: Out oyuncuların toplam kaybedilen dakikası
        - injury_count_home/away: Out oyuncu sayısı
        - any_key_player_out_home/away: Key player out mu? (1/0)
    """
    df = games_df.copy()
    
    # game_date'i datetime yap
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    
    # Injury verisi boşsa, sıfır değerlerle doldur
    if len(injury_df) == 0 or len(player_minutes_df) == 0:
        print("⚠️ Injury veya player minutes verisi boş. Sıfır değerlerle dolduruluyor.")
        for prefix in ['home', 'away']:
            df[f'expected_minutes_lost_{prefix}'] = 0.0
            df[f'injury_count_{prefix}'] = 0
            df[f'any_key_player_out_{prefix}'] = 0
        return df
    
    # Player minutes'ı merge et (injury_df'e)
    injury_with_minutes = injury_df.merge(
        player_minutes_df[['player_name', 'team', 'avg_minutes_per_game']],
        on=['player_name', 'team'],
        how='left'
    )
    
    # Eksik dakikaları 0 ile doldur
    injury_with_minutes['avg_minutes_per_game'] = injury_with_minutes['avg_minutes_per_game'].fillna(0)
    
    # Sadece "Out" status'lü oyuncuları filtrele
    out_players = injury_with_minutes[
        injury_with_minutes['status'].str.lower() == 'out'
    ].copy()
    
    # Her maç için injury stats hesapla
    expected_minutes_lost_home = []
    expected_minutes_lost_away = []
    injury_count_home = []
    injury_count_away = []
    any_key_player_out_home = []
    any_key_player_out_away = []
    
    for _, row in df.iterrows():
        game_date = row['game_date']
        home_team = row['home_team']
        away_team = row['away_team']
        
        # O gün ve o takım için out oyuncuları bul
        home_out = out_players[
            (out_players['date'] == game_date) & 
            (out_players['team'] == home_team)
        ]
        away_out = out_players[
            (out_players['date'] == game_date) & 
            (out_players['team'] == away_team)
        ]
        
        # Home team stats
        home_minutes = home_out['avg_minutes_per_game'].sum()
        home_count = len(home_out)
        home_key = int((home_out['avg_minutes_per_game'] >= key_player_minutes_threshold).any())
        
        # Away team stats
        away_minutes = away_out['avg_minutes_per_game'].sum()
        away_count = len(away_out)
        away_key = int((away_out['avg_minutes_per_game'] >= key_player_minutes_threshold).any())
        
        expected_minutes_lost_home.append(home_minutes)
        expected_minutes_lost_away.append(away_minutes)
        injury_count_home.append(home_count)
        injury_count_away.append(away_count)
        any_key_player_out_home.append(home_key)
        any_key_player_out_away.append(away_key)
    
    # Kolonları ekle
    df['expected_minutes_lost_home'] = expected_minutes_lost_home
    df['expected_minutes_lost_away'] = expected_minutes_lost_away
    df['injury_count_home'] = injury_count_home
    df['injury_count_away'] = injury_count_away
    df['any_key_player_out_home'] = any_key_player_out_home
    df['any_key_player_out_away'] = any_key_player_out_away
    
    # Diff feature'ları ekle
    df['diff_expected_minutes_lost'] = df['expected_minutes_lost_home'] - df['expected_minutes_lost_away']
    df['diff_injury_count'] = df['injury_count_home'] - df['injury_count_away']
    
    print(f"✅ Injury features eklendi")
    print(f"   - Home out players: {sum(injury_count_home)} total")
    print(f"   - Away out players: {sum(injury_count_away)} total")
    
    return df


def build_injury_features(
    games_csv: Union[str, Path] = "data_interim/games_with_core_features.csv",
    injury_raw_dir: Union[str, Path] = "data_raw/injury_reports_raw/",
    player_stats_path: Union[str, Path] = "data_raw/nbastuffer_2025_2026_player_stats_raw.csv",
    output_csv: Optional[Union[str, Path]] = None,
    injury_clean_csv: Optional[Union[str, Path]] = "data_interim/injury_reports_clean.csv"
) -> pd.DataFrame:
    """
    Injury feature pipeline'ını çalıştır.
    
    Parameters
    ----------
    games_csv : str or Path
        Core features içeren games CSV dosyası
    injury_raw_dir : str or Path
        Raw injury report dosyalarının klasörü
    player_stats_path : str or Path
        Player stats CSV dosyası
    output_csv : str or Path, optional
        Çıktı CSV dosyası
    injury_clean_csv : str or Path, optional
        Temizlenmiş injury verisi için çıktı yolu
        
    Returns
    -------
    pd.DataFrame
        Injury feature'ları eklenmiş games dataframe
    """
    print("=" * 60)
    print("Injury Features Pipeline")
    print("=" * 60)
    
    # 1. Games verisini yükle
    print("\n1. Games verisi yükleniyor...")
    games_csv = Path(games_csv)
    if not games_csv.exists():
        raise FileNotFoundError(f"Games CSV bulunamadı: {games_csv}")
    
    games_df = pd.read_csv(games_csv, low_memory=False)
    print(f"   {len(games_df)} maç yüklendi")
    
    # 2. Injury verilerini yükle ve temizle
    print("\n2. Injury verileri yükleniyor ve temizleniyor...")
    injury_df = load_and_clean_injury_reports(injury_raw_dir, injury_clean_csv)
    
    # 3. Player minutes verilerini yükle
    print("\n3. Player minutes verisi yükleniyor...")
    player_minutes_df = load_player_minutes(player_stats_path)
    
    # 4. Injury feature'larını ekle
    print("\n4. Injury features ekleniyor...")
    result_df = add_injury_features(games_df, injury_df, player_minutes_df)
    
    # 5. Kaydet
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"\n✅ Sonuç kaydedildi: {output_csv}")
        print(f"   {len(result_df)} satır, {len(result_df.columns)} kolon")
    
    return result_df


if __name__ == "__main__":
    build_injury_features(
        games_csv="data_interim/games_with_core_features.csv",
        output_csv="data_processed/games_with_all_features.csv"
    )

