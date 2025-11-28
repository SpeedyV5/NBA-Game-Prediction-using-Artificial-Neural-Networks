"""
NBA Veri Temizleme ve Birleştirme Modülü

Bu modül, ham NBA verilerini temizleyip birleştirmek için fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import warnings
warnings.filterwarnings('ignore')


# NBA Takım İsmi Mapping Dictionary
TEAM_NAME_MAPPING = {
    # Kısaltmalar -> Tam İsimler
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'BRK': 'Brooklyn Nets',
    'NJN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHO': 'Charlotte Hornets',
    'CHH': 'Charlotte Hornets',
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
    'NOH': 'New Orleans Pelicans',
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
    'WIZ': 'Washington Wizards',
    
    # Varyasyonlar -> Tam İsimler
    'Los Angeles Lakers': 'Los Angeles Lakers',
    'L.A. Lakers': 'Los Angeles Lakers',
    'LA Lakers': 'Los Angeles Lakers',
    'Lakers': 'Los Angeles Lakers',
    'Boston Celtics': 'Boston Celtics',
    'Celtics': 'Boston Celtics',
    'Brooklyn Nets': 'Brooklyn Nets',
    'Nets': 'Brooklyn Nets',
    'New Jersey Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets',
    'Hornets': 'Charlotte Hornets',
    'Charlotte Bobcats': 'Charlotte Hornets',
    'Chicago Bulls': 'Chicago Bulls',
    'Bulls': 'Chicago Bulls',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Cavaliers': 'Cleveland Cavaliers',
    'Cavs': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks',
    'Mavericks': 'Dallas Mavericks',
    'Mavs': 'Dallas Mavericks',
    'Denver Nuggets': 'Denver Nuggets',
    'Nuggets': 'Denver Nuggets',
    'Detroit Pistons': 'Detroit Pistons',
    'Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors',
    'Warriors': 'Golden State Warriors',
    'Houston Rockets': 'Houston Rockets',
    'Rockets': 'Houston Rockets',
    'Indiana Pacers': 'Indiana Pacers',
    'Pacers': 'Indiana Pacers',
    'LA Clippers': 'LA Clippers',
    'Los Angeles Clippers': 'LA Clippers',
    'L.A. Clippers': 'LA Clippers',
    'Clippers': 'LA Clippers',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat',
    'Heat': 'Miami Heat',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Bucks': 'Milwaukee Bucks',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'Timberwolves': 'Minnesota Timberwolves',
    'Wolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'Pelicans': 'New Orleans Pelicans',
    'New Orleans Hornets': 'New Orleans Pelicans',
    'New York Knicks': 'New York Knicks',
    'Knicks': 'New York Knicks',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Thunder': 'Oklahoma City Thunder',
    'Seattle SuperSonics': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic',
    'Magic': 'Orlando Magic',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    '76ers': 'Philadelphia 76ers',
    'Sixers': 'Philadelphia 76ers',
    'Phoenix Suns': 'Phoenix Suns',
    'Suns': 'Phoenix Suns',
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Trail Blazers': 'Portland Trail Blazers',
    'Blazers': 'Portland Trail Blazers',
    'Sacramento Kings': 'Sacramento Kings',
    'Kings': 'Sacramento Kings',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors',
    'Raptors': 'Toronto Raptors',
    'Utah Jazz': 'Utah Jazz',
    'Jazz': 'Utah Jazz',
    'Washington Wizards': 'Washington Wizards',
    'Wizards': 'Washington Wizards',
    'Atlanta Hawks': 'Atlanta Hawks',
    'Hawks': 'Atlanta Hawks',
    # NBAstuffer formatı (sadece şehir/ekip ismi)
    'Atlanta': 'Atlanta Hawks',
    'Boston': 'Boston Celtics',
    'Brooklyn': 'Brooklyn Nets',
    'Charlotte': 'Charlotte Hornets',
    'Chicago': 'Chicago Bulls',
    'Cleveland': 'Cleveland Cavaliers',
    'Dallas': 'Dallas Mavericks',
    'Denver': 'Denver Nuggets',
    'Detroit': 'Detroit Pistons',
    'Golden State': 'Golden State Warriors',
    'Houston': 'Houston Rockets',
    'Indiana': 'Indiana Pacers',
    'LA Clippers': 'LA Clippers',
    'LA Lakers': 'Los Angeles Lakers',
    'Memphis': 'Memphis Grizzlies',
    'Miami': 'Miami Heat',
    'Milwaukee': 'Milwaukee Bucks',
    'Minnesota': 'Minnesota Timberwolves',
    'New Orleans': 'New Orleans Pelicans',
    'New York': 'New York Knicks',
    'Oklahoma City': 'Oklahoma City Thunder',
    'Orlando': 'Orlando Magic',
    'Philadelphia': 'Philadelphia 76ers',
    'Phoenix': 'Phoenix Suns',
    'Portland': 'Portland Trail Blazers',
    'Sacramento': 'Sacramento Kings',
    'San Antonio': 'San Antonio Spurs',
    'Toronto': 'Toronto Raptors',
    'Utah': 'Utah Jazz',
    'Washington': 'Washington Wizards',
}


def normalize_date(date_col: pd.Series, format_hints: Optional[List[str]] = None) -> pd.Series:
    """
    Farklı tarih formatlarını standart 'YYYY-MM-DD' formatına çevir.
    
    Parameters:
    -----------
    date_col : pd.Series
        Tarih içeren pandas Series
    format_hints : list, optional
        Denenecek tarih formatları listesi
        
    Returns:
    --------
    pd.Series
        Normalize edilmiş tarih serisi (datetime64[ns])
    """
    if format_hints is None:
        format_hints = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%Y.%m.%d',
            '%m.%d.%Y',
            '%d.%m.%Y',
        ]
    
    # Eğer zaten datetime ise, sadece formatı kontrol et
    if pd.api.types.is_datetime64_any_dtype(date_col):
        return date_col
    
    # String'e çevir
    date_str = date_col.astype(str)
    
    # Boş değerleri handle et
    date_str = date_str.replace(['nan', 'None', 'NaT', ''], np.nan)
    
    result = pd.Series(index=date_col.index, dtype='datetime64[ns]')
    
    # Her formatı dene
    for fmt in format_hints:
        mask = result.isna()
        if not mask.any():
            break
            
        try:
            parsed = pd.to_datetime(date_str[mask], format=fmt, errors='coerce')
            result.loc[mask] = parsed
        except:
            continue
    
    # Eğer hala parse edilemeyen değerler varsa, pandas'ın otomatik parser'ını kullan
    if result.isna().any():
        mask = result.isna()
        try:
            parsed = pd.to_datetime(date_str[mask], errors='coerce', infer_datetime_format=True)
            result.loc[mask] = parsed
        except:
            pass
    
    return result


def standardize_team_names(df: pd.DataFrame, team_cols: List[str] = None) -> pd.DataFrame:
    """
    Takım isimlerini standart formata çevir.
    
    Parameters:
    -----------
    df : pd.DataFrame
        İşlenecek DataFrame
    team_cols : list, optional
        Takım ismi içeren sütunlar. None ise otomatik tespit edilir.
        
    Returns:
    --------
    pd.DataFrame
        Takım isimleri standartlaştırılmış DataFrame
    """
    df = df.copy()
    
    # Eğer team_cols belirtilmemişse, olası sütun isimlerini ara
    if team_cols is None:
        possible_cols = [
            'team', 'team_name', 'team_abbr', 'team_id', 'TEAM',
            'home_team', 'away_team', 'team_home', 'team_away',
            'home', 'away', 'home_team_name', 'away_team_name',
            'team_name_home', 'team_name_away'
        ]
        team_cols = [col for col in possible_cols if col in df.columns]
    
    # Her takım sütunu için mapping uygula
    for col in team_cols:
        if col in df.columns:
            # Önce mapping dictionary'den kontrol et
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].map(TEAM_NAME_MAPPING).fillna(df[col])
            
            # Eğer hala eşleşmeyen değerler varsa, case-insensitive kontrol yap
            remaining = df[col].unique()
            for val in remaining:
                if val not in TEAM_NAME_MAPPING.values():
                    # Case-insensitive eşleşme ara
                    val_lower = str(val).lower()
                    for key, mapped_val in TEAM_NAME_MAPPING.items():
                        if str(key).lower() == val_lower:
                            df.loc[df[col] == val, col] = mapped_val
                            break
    
    return df


def clean_game_data(df: pd.DataFrame, 
                   score_cols: List[str] = None,
                   date_col: str = 'date',
                   min_score: int = 0,
                   max_score: int = 200) -> pd.DataFrame:
    """
    Maç verilerini temizle: eksik skor, invalid satırları kaldır.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Temizlenecek DataFrame
    score_cols : list, optional
        Skor içeren sütunlar. None ise otomatik tespit edilir.
    date_col : str
        Tarih sütunu adı
    min_score : int
        Minimum geçerli skor
    max_score : int
        Maximum geçerli skor
        
    Returns:
    --------
    pd.DataFrame
        Temizlenmiş DataFrame
    """
    df = df.copy()
    original_len = len(df)
    
    # Skor sütunlarını tespit et
    if score_cols is None:
        possible_score_cols = [
            'home_score', 'away_score', 'pts_home', 'pts_away',
            'score_home', 'score_away', 'home_pts', 'away_pts',
            'team_score', 'opponent_score', 'points', 'opp_points'
        ]
        score_cols = [col for col in possible_score_cols if col in df.columns]
    
    # 1. Eksik skor satırlarını kaldır
    if score_cols:
        df = df.dropna(subset=score_cols)
    
    # 2. Tarih sütununu normalize et
    if date_col in df.columns:
        df[date_col] = normalize_date(df[date_col])
        # Tarihi olmayan satırları kaldır
        df = df.dropna(subset=[date_col])
    
    # 3. Invalid skorları temizle
    if score_cols:
        for col in score_cols:
            # Negatif skorları kaldır
            df = df[df[col] >= min_score]
            # Çok yüksek skorları kaldır
            df = df[df[col] <= max_score]
    
    # 4. 0-0 skorlu maçları kaldır (genellikle hata)
    if len(score_cols) >= 2:
        mask = (df[score_cols[0]] == 0) & (df[score_cols[1]] == 0)
        df = df[~mask]
    
    # 5. Duplicate satırları kontrol et ve kaldır
    # Tarih ve takım kombinasyonuna göre duplicate kontrolü
    if date_col in df.columns:
        team_cols = [col for col in df.columns if 'team' in col.lower() or 'home' in col.lower() or 'away' in col.lower()]
        if team_cols:
            duplicate_cols = [date_col] + team_cols[:2]  # İlk 2 takım sütunu
            df = df.drop_duplicates(subset=duplicate_cols, keep='first')
    
    cleaned_len = len(df)
    removed = original_len - cleaned_len
    
    if removed > 0:
        print(f"Temizleme: {removed} satır kaldırıldı ({original_len} -> {cleaned_len})")
    
    return df


def convert_boxscore_to_game_level(boxscore_df: pd.DataFrame) -> pd.DataFrame:
    """
    Box score formatındaki veriyi maç bazlı formata dönüştür.
    
    Her gameId için:
    - matchup'tan home/away takımları çıkar
    - Her takımın toplam skorunu hesapla
    - Tek satır oluştur
    
    Parameters:
    -----------
    boxscore_df : pd.DataFrame
        Box score formatında veri (her satır bir oyuncu)
        
    Returns:
    --------
    pd.DataFrame
        Maç bazlı format (her satır bir maç)
    """
    df = boxscore_df.copy()
    
    # Gerekli sütunları kontrol et
    required_cols = ['gameId', 'game_date', 'matchup', 'teamTricode', 'points', 'season_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik sütunlar: {missing_cols}")
    
    # Her gameId için maç bilgilerini topla
    games_list = []
    
    for game_id in df['gameId'].unique():
        game_data = df[df['gameId'] == game_id].copy()
        
        # İlk satırdan maç bilgilerini al
        first_row = game_data.iloc[0]
        matchup = str(first_row['matchup']).strip()
        game_date = first_row['game_date']
        season_year = first_row.get('season_year', '')
        season_type = first_row['season_type']
        
        # Matchup'tan home/away takımları çıkar
        # Format: "TEAM1 @ TEAM2" → TEAM2 home, TEAM1 away
        # Format: "TEAM1 vs. TEAM2" → TEAM1 home, TEAM2 away (genellikle)
        if '@' in matchup:
            parts = matchup.split('@')
            away_team_code = parts[0].strip()
            home_team_code = parts[1].strip()
        elif 'vs.' in matchup or 'vs' in matchup:
            parts = matchup.replace('vs.', 'vs').split('vs')
            if len(parts) == 2:
                home_team_code = parts[0].strip()
                away_team_code = parts[1].strip()
            else:
                # Eğer parse edilemezse, verideki takımları kullan
                teams = game_data['teamTricode'].unique()
                if len(teams) == 2:
                    home_team_code = teams[0]
                    away_team_code = teams[1]
                else:
                    continue
        else:
            # Matchup formatı tanınmıyor, verideki takımları kullan
            teams = game_data['teamTricode'].unique()
            if len(teams) == 2:
                home_team_code = teams[0]
                away_team_code = teams[1]
            else:
                continue
        
        # Her takımın toplam skorunu hesapla
        home_team_data = game_data[game_data['teamTricode'] == home_team_code]
        away_team_data = game_data[game_data['teamTricode'] == away_team_code]
        
        home_score = home_team_data['points'].sum() if len(home_team_data) > 0 else 0
        away_score = away_team_data['points'].sum() if len(away_team_data) > 0 else 0
        
        # Eğer skor 0 ise, başka bir yöntem dene (toplam field goals vb.)
        if home_score == 0 and away_score == 0:
            # Alternatif: fieldGoalsMade * 2 + threePointersMade + freeThrowsMade
            if 'fieldGoalsMade' in game_data.columns:
                home_score = (home_team_data['fieldGoalsMade'].sum() * 2 + 
                             home_team_data.get('threePointersMade', pd.Series([0])).sum() +
                             home_team_data.get('freeThrowsMade', pd.Series([0])).sum())
                away_score = (away_team_data['fieldGoalsMade'].sum() * 2 + 
                             away_team_data.get('threePointersMade', pd.Series([0])).sum() +
                             away_team_data.get('freeThrowsMade', pd.Series([0])).sum())
        
        # Takım isimlerini standartlaştır (teamTricode'dan tam isme)
        home_team_name = TEAM_NAME_MAPPING.get(home_team_code, home_team_code)
        away_team_name = TEAM_NAME_MAPPING.get(away_team_code, away_team_code)
        
        # Maç satırı oluştur
        game_row = {
            'gameId': game_id,
            'game_date': game_date,
            'season_year': season_year,
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_team_code': home_team_code,
            'away_team_code': away_team_code,
            'home_score': home_score,
            'away_score': away_score,
            'season_type': season_type,
            'matchup': matchup
        }
        
        games_list.append(game_row)
    
    # DataFrame oluştur
    games_df = pd.DataFrame(games_list)
    
    print(f"Box score'dan {len(games_df)} maç oluşturuldu (orijinal {len(df)} satırdan)")
    
    return games_df


def extract_games_from_gamelog(gamelog_path: Union[str, Path]) -> pd.DataFrame:
    """
    Gamelog dosyasından maç bazlı veri çıkar.
    Her unique (GAME_DATE, MATCHUP) kombinasyonu için bir maç satırı oluştur.
    
    Parameters:
    -----------
    gamelog_path : str or Path
        Gamelog CSV dosyasının yolu
        
    Returns:
    --------
    pd.DataFrame
        Maç bazlı format (her satır bir maç)
    """
    df = pd.read_csv(gamelog_path, low_memory=False)
    
    # GAME_DATE ve MATCHUP sütunlarını normalize et
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = df['GAME_DATE'].astype(str).str.strip('"')  # Tırnak işaretlerini kaldır
        df['GAME_DATE'] = normalize_date(df['GAME_DATE'])
    
    # Her unique (GAME_DATE, MATCHUP) kombinasyonu için bir maç oluştur
    games_list = []
    
    for (game_date, matchup), group in df.groupby(['GAME_DATE', 'MATCHUP']):
        # Matchup'tan home/away takımları çıkar
        matchup_str = str(matchup).strip()
        
        if '@' in matchup_str:
            parts = matchup_str.split('@')
            away_team_code = parts[0].strip()
            home_team_code = parts[1].strip()
        elif 'vs.' in matchup_str or 'vs' in matchup_str:
            parts = matchup_str.replace('vs.', 'vs').split('vs')
            if len(parts) == 2:
                home_team_code = parts[0].strip()
                away_team_code = parts[1].strip()
            else:
                continue
        else:
            continue
        
        # Takım isimlerini standartlaştır
        home_team_name = TEAM_NAME_MAPPING.get(home_team_code, home_team_code)
        away_team_name = TEAM_NAME_MAPPING.get(away_team_code, away_team_code)
        
        # Skor bilgisi yok, oyuncu istatistiklerinden toplam PTS alabiliriz
        # Ama bu sadece bir oyuncunun skorunu verir, takım skorunu değil
        # Şimdilik skorları None/NaN olarak bırakabiliriz
        
        game_row = {
            'game_date': game_date,
            'season_year': '2025-26',
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_team_code': home_team_code,
            'away_team_code': away_team_code,
            'home_score': None,  # Skor bilgisi yok
            'away_score': None,  # Skor bilgisi yok
            'season_type': 'regular',
            'matchup': matchup_str
        }
        
        games_list.append(game_row)
    
    games_df = pd.DataFrame(games_list)
    
    # Duplicate'leri kaldır (aynı maç birden fazla oyuncu için olabilir)
    games_df = games_df.drop_duplicates(subset=['game_date', 'matchup']).reset_index(drop=True)
    
    print(f"Gamelog'dan {len(games_df)} unique maç çıkarıldı")
    
    return games_df


def merge_nocturnebear_data(base_path: Union[str, Path] = 'data_raw/nocturnebear_2010_2024/',
                           convert_to_game_level: bool = True) -> pd.DataFrame:
    """
    NocturneBear verilerini birleştir: 3 part regular_season_box_scores dosyalarını birleştir,
    play_off verilerini ekle.
    
    Parameters:
    -----------
    base_path : str or Path
        NocturneBear verilerinin bulunduğu klasör yolu
        
    Returns:
    --------
    pd.DataFrame
        Birleştirilmiş DataFrame
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"NocturneBear veri klasörü bulunamadı: {base_path}")
    
    all_dataframes = []
    
    # Regular season box scores (3 part)
    for part_num in [1, 2, 3]:
        file_path = base_path / f'regular_season_box_scores_2010_2024_part_{part_num}.csv'
        if file_path.exists():
            print(f"Yükleniyor: {file_path.name}")
            df = pd.read_csv(file_path, low_memory=False)
            df['season_type'] = 'regular'
            all_dataframes.append(df)
        else:
            print(f"Uyarı: {file_path.name} bulunamadı")
    
    # Playoff box scores
    playoff_file = base_path / 'play_off_box_scores_2010_2024.csv'
    if playoff_file.exists():
        print(f"Yükleniyor: {playoff_file.name}")
        df = pd.read_csv(playoff_file, low_memory=False)
        df['season_type'] = 'playoff'
        all_dataframes.append(df)
    else:
        print(f"Uyarı: {playoff_file.name} bulunamadı")
    
    if not all_dataframes:
        raise ValueError("Hiçbir veri dosyası bulunamadı!")
    
    # Tüm DataFrame'leri birleştir
    print("DataFrame'ler birleştiriliyor...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Toplam {len(merged_df)} satır birleştirildi (box score formatında)")
    print(f"Sütunlar: {list(merged_df.columns)}")
    
    # Eğer maç bazlı formata dönüştürülmesi isteniyorsa
    if convert_to_game_level:
        print("\nBox score verisi maç bazlı formata dönüştürülüyor...")
        merged_df = convert_boxscore_to_game_level(merged_df)
        print(f"Maç bazlı format: {len(merged_df)} satır")
        print(f"Sütunlar: {list(merged_df.columns)}")
    
    return merged_df


def merge_team_stats(game_df: pd.DataFrame, 
                    team_stats_df: pd.DataFrame,
                    date_col: str = 'date',
                    home_team_col: str = 'home_team',
                    away_team_col: str = 'away_team',
                    team_col_in_stats: str = 'team') -> pd.DataFrame:
    """
    Takım istatistiklerini maç verisi ile birleştir.
    Her maç için home_team ve away_team istatistiklerini ekler.
    
    Parameters:
    -----------
    game_df : pd.DataFrame
        Maç verisi DataFrame'i
    team_stats_df : pd.DataFrame
        Takım istatistikleri DataFrame'i
    date_col : str
        Tarih sütunu adı (game_df'de)
    home_team_col : str
        Ev sahibi takım sütunu adı
    away_team_col : str
        Deplasman takımı sütunu adı
    team_col_in_stats : str
        Takım sütunu adı (team_stats_df'de)
        
    Returns:
    --------
    pd.DataFrame
        Takım istatistikleri ile birleştirilmiş maç verisi
    """
    game_df = game_df.copy()
    team_stats_df = team_stats_df.copy()
    
    # Eğer home_team veya away_team sütunları yoksa, otomatik tespit et
    if home_team_col not in game_df.columns:
        # Olası home team sütunlarını ara
        possible_home_cols = ['home_team', 'team_home', 'home', 'home_team_name']
        home_team_col = None
        for col in possible_home_cols:
            if col in game_df.columns:
                home_team_col = col
                break
        
        # Eğer hala bulunamadıysa, box score formatında olabilir (teamName, teamTricode vb.)
        if home_team_col is None:
            print("⚠ Uyarı: home_team/away_team sütunları bulunamadı.")
            print("   NocturneBear verisi box score formatında olabilir (her satır bir oyuncu).")
            print("   Takım istatistikleri merge edilemiyor - bu veri formatı için maç bazlı dönüşüm gerekli.")
            print("   Orijinal DataFrame döndürülüyor (takım istatistikleri olmadan).")
            return game_df
    
    if away_team_col not in game_df.columns:
        # Olası away team sütunlarını ara
        possible_away_cols = ['away_team', 'team_away', 'away', 'away_team_name']
        away_team_col = None
        for col in possible_away_cols:
            if col in game_df.columns:
                away_team_col = col
                break
        
        if away_team_col is None:
            print("⚠ Uyarı: away_team sütunu bulunamadı.")
            print("   Takım istatistikleri merge edilemiyor.")
            return game_df
    
    # Takım isimlerini standartlaştır
    game_df = standardize_team_names(game_df, [home_team_col, away_team_col])
    team_stats_df = standardize_team_names(team_stats_df, [team_col_in_stats])
    
    # İstatistik sütunlarını belirle (takım adı, tarih ve boş sütunlar hariç)
    # RANK sütunu genellikle boş olabilir, bu yüzden hariç tutuyoruz
    excluded_cols = [team_col_in_stats, date_col, 'date', 'season', 'RANK']
    stat_cols = [col for col in team_stats_df.columns 
                 if col not in excluded_cols]
    
    # Boş sütunları da hariç tut
    stat_cols = [col for col in stat_cols 
                 if not team_stats_df[col].isnull().all()]
    
    # Home team istatistiklerini merge et
    home_stats = team_stats_df.copy()
    # Önce takım sütununu rename et, sonra diğer sütunları prefix'le
    if team_col_in_stats in home_stats.columns:
        home_stats = home_stats.rename(columns={team_col_in_stats: home_team_col})
    
    home_stats.columns = [f'home_team_{col}' if col in stat_cols and col != home_team_col else col 
                         for col in home_stats.columns]
    
    # Merge home team stats
    game_df = game_df.merge(
        home_stats,
        on=home_team_col,
        how='left',
        suffixes=('', '_home_stats')
    )
    
    # Away team istatistiklerini merge et
    away_stats = team_stats_df.copy()
    # Önce takım sütununu rename et, sonra diğer sütunları prefix'le
    if team_col_in_stats in away_stats.columns:
        away_stats = away_stats.rename(columns={team_col_in_stats: away_team_col})
    
    away_stats.columns = [f'away_team_{col}' if col in stat_cols and col != away_team_col else col 
                         for col in away_stats.columns]
    
    # Merge away team stats
    game_df = game_df.merge(
        away_stats,
        on=away_team_col,
        how='left',
        suffixes=('', '_away_stats')
    )
    
    print(f"Takım istatistikleri birleştirildi. Toplam sütun sayısı: {len(game_df.columns)}")
    
    # Merge başarı oranını kontrol et
    if stat_cols:
        sample_col = f'home_team_{stat_cols[0]}'
        if sample_col in game_df.columns:
            merged_count = game_df[sample_col].notna().sum()
            total_count = len(game_df)
            print(f"Merge başarı oranı: {merged_count}/{total_count} ({merged_count/total_count*100:.1f}%)")
    
    return game_df


def extract_games_from_gamelog(gamelog_path: Union[str, Path]) -> pd.DataFrame:
    """
    Gamelog dosyasından maç bazlı veri çıkar.
    Her unique (GAME_DATE, MATCHUP) kombinasyonu için bir maç satırı oluştur.
    
    Parameters:
    -----------
    gamelog_path : str or Path
        Gamelog CSV dosyasının yolu
        
    Returns:
    --------
    pd.DataFrame
        Maç bazlı format (her satır bir maç)
    """
    if not Path(gamelog_path).exists():
        print(f"⚠ Gamelog dosyası bulunamadı: {gamelog_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(gamelog_path, low_memory=False)
    
    # GAME_DATE ve MATCHUP sütunlarını normalize et
    df['GAME_DATE'] = df['GAME_DATE'].astype(str).str.strip('"')  # Tırnak işaretlerini kaldır
    df['GAME_DATE'] = normalize_date(df['GAME_DATE'])
    
    # Her unique (GAME_DATE, MATCHUP) kombinasyonu için bir maç oluştur
    games_list = []
    
    for (game_date, matchup), group in df.groupby(['GAME_DATE', 'MATCHUP']):
        # Matchup'tan home/away takımları çıkar
        matchup_str = str(matchup).strip()
        
        if '@' in matchup_str:
            parts = matchup_str.split('@')
            away_team_code = parts[0].strip()
            home_team_code = parts[1].strip()
        elif 'vs.' in matchup_str or 'vs' in matchup_str:
            parts = matchup_str.replace('vs.', 'vs').split('vs')
            if len(parts) == 2:
                home_team_code = parts[0].strip()
                away_team_code = parts[1].strip()
            else:
                continue
        else:
            continue
        
        # Takım isimlerini standartlaştır
        home_team_name = TEAM_NAME_MAPPING.get(home_team_code, home_team_code)
        away_team_name = TEAM_NAME_MAPPING.get(away_team_code, away_team_code)
        
        # Skor bilgisi yok, oyuncu istatistiklerinden toplam PTS alabiliriz
        # Ama bu sadece bir oyuncunun skorunu verir, takım skorunu değil
        # Şimdilik skorları None/NaN olarak bırakabiliriz
        
        game_row = {
            'game_date': game_date,
            'season_year': '2025-26',
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_team_code': home_team_code,
            'away_team_code': away_team_code,
            'home_score': None,  # Skor bilgisi yok
            'away_score': None,  # Skor bilgisi yok
            'season_type': 'regular',
            'matchup': matchup_str
        }
        
        games_list.append(game_row)
    
    games_df = pd.DataFrame(games_list)
    
    # Duplicate'leri kaldır (aynı maç birden fazla oyuncu için olabilir)
    games_df = games_df.drop_duplicates(subset=['game_date', 'matchup']).reset_index(drop=True)
    
    print(f"Gamelog'dan {len(games_df)} unique maç çıkarıldı")
    
    return games_df


def merge_rest_days_stats(game_df: pd.DataFrame,
                         rest_days_path: Union[str, Path],
                         date_col: str = 'game_date') -> pd.DataFrame:
    """
    Rest days istatistiklerini maç verisi ile birleştir.
    
    Parameters:
    -----------
    game_df : pd.DataFrame
        Maç verisi DataFrame'i
    rest_days_path : str or Path
        Rest days stats CSV dosyasının yolu
    date_col : str
        Tarih sütunu adı
        
    Returns:
    --------
    pd.DataFrame
        Rest days istatistikleri ile birleştirilmiş maç verisi
    """
    if not Path(rest_days_path).exists():
        print(f"⚠ Rest days stats dosyası bulunamadı: {rest_days_path}")
        return game_df
    
    game_df = game_df.copy()
    rest_days_df = pd.read_csv(rest_days_path, low_memory=False)
    
    # Takım isimlerini standartlaştır
    if 'TEAM NAME' in rest_days_df.columns:
        rest_days_df = standardize_team_names(rest_days_df, ['TEAM NAME'])
        team_col = 'TEAM NAME'
    elif 'TEAM' in rest_days_df.columns:
        rest_days_df = standardize_team_names(rest_days_df, ['TEAM'])
        team_col = 'TEAM'
    else:
        print("⚠ Rest days stats'te takım sütunu bulunamadı")
        return game_df
    
    # Rest days istatistik sütunlarını belirle
    # RANK ve OPPONENT TODAY hariç, boş sütunları da hariç tut
    excluded_cols = [team_col, 'RANK', 'OPPONENT TODAY']
    stat_cols = [col for col in rest_days_df.columns 
                 if col not in excluded_cols]
    
    # Boş sütunları da hariç tut
    stat_cols = [col for col in stat_cols 
                 if not rest_days_df[col].isnull().all()]
    
    # Home team rest days stats
    home_rest = rest_days_df.copy()
    home_rest.columns = [f'home_rest_{col}' if col in stat_cols else col 
                        for col in home_rest.columns]
    if team_col in home_rest.columns:
        home_rest = home_rest.rename(columns={team_col: 'home_team'})
    
    game_df = game_df.merge(
        home_rest,
        on='home_team',
        how='left',
        suffixes=('', '_home_rest')
    )
    
    # Away team rest days stats
    away_rest = rest_days_df.copy()
    away_rest.columns = [f'away_rest_{col}' if col in stat_cols else col 
                        for col in away_rest.columns]
    if team_col in away_rest.columns:
        away_rest = away_rest.rename(columns={team_col: 'away_team'})
    
    game_df = game_df.merge(
        away_rest,
        on='away_team',
        how='left',
        suffixes=('', '_away_rest')
    )
    
    print(f"Rest days istatistikleri birleştirildi. Toplam sütun sayısı: {len(game_df.columns)}")
    
    return game_df


def create_master_merged(output_dir: Union[str, Path] = 'data_processed/',
                        nocturnebear_path: Union[str, Path] = 'data_raw/nocturnebear_2010_2024/',
                        team_stats_path: Union[str, Path] = 'data_raw/nbastuffer_2025_2026_team_stats_raw.csv',
                        bdl_games_path: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
    """
    Tüm verileri temizleyip birleştirerek master_merged.csv ve diğer çıktı dosyalarını oluştur.
    
    Parameters:
    -----------
    output_dir : str or Path
        Çıktı dosyalarının kaydedileceği klasör
    nocturnebear_path : str or Path
        NocturneBear verilerinin yolu
    team_stats_path : str or Path
        Takım istatistikleri dosyasının yolu
    bdl_games_path : str or Path, optional
        BDL games 2025 dosyasının yolu (varsa)
        
    Returns:
    --------
    dict
        Çıktı DataFrame'lerini içeren dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. NocturneBear verilerini yükle ve birleştir (maç bazlı formata dönüştür)
    print("=" * 60)
    print("1. NocturneBear Verilerini Yükleme ve Birleştirme")
    print("=" * 60)
    nocturnebear_df = merge_nocturnebear_data(nocturnebear_path, convert_to_game_level=True)
    
    # 2. Tarih sütununu normalize et
    date_cols = [col for col in nocturnebear_df.columns if 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        nocturnebear_df[date_col] = normalize_date(nocturnebear_df[date_col])
    else:
        print("Uyarı: Tarih sütunu bulunamadı!")
        date_col = None
    
    # Maç bazlı formata dönüştürüldüyse, date_col'u 'game_date' olarak ayarla
    if 'game_date' in nocturnebear_df.columns:
        date_col = 'game_date'
    
    # 3. Takım isimlerini standartlaştır
    print("\nTakım isimleri standartlaştırılıyor...")
    nocturnebear_df = standardize_team_names(nocturnebear_df)
    
    # 4. Veriyi temizle
    print("\nVeri temizleniyor...")
    nocturnebear_df = clean_game_data(nocturnebear_df, date_col=date_col if date_col else 'date')
    
    # 5. BDL games 2025 varsa yükle ve ekle
    if bdl_games_path and Path(bdl_games_path).exists():
        print("\n" + "=" * 60)
        print("2. BDL Games 2025 Verilerini Yükleme")
        print("=" * 60)
        bdl_df = pd.read_csv(bdl_games_path, low_memory=False)
        bdl_df['season_type'] = 'regular'
        
        if date_cols:
            bdl_df[date_col] = normalize_date(bdl_df[date_col])
        bdl_df = standardize_team_names(bdl_df)
        bdl_df = clean_game_data(bdl_df, date_col=date_col if date_col else 'date')
        
        # NocturneBear verilerine ekle
        nocturnebear_df = pd.concat([nocturnebear_df, bdl_df], ignore_index=True)
        print(f"BDL verileri eklendi. Toplam satır: {len(nocturnebear_df)}")
    
    # 5b. Gamelog'dan 2025 verilerini çıkar ve ekle
    gamelog_path = Path('data_raw/player_203999_gamelog_2025_26.csv')
    if gamelog_path.exists():
        print("\n" + "=" * 60)
        print("2b. Gamelog'dan 2025 Verilerini Çıkarma")
        print("=" * 60)
        try:
            gamelog_games = extract_games_from_gamelog(gamelog_path)
            if len(gamelog_games) > 0:
                # Tarih normalizasyonu
                if 'game_date' in gamelog_games.columns:
                    gamelog_games['game_date'] = normalize_date(gamelog_games['game_date'])
                
                # Takım isimlerini standartlaştır
                gamelog_games = standardize_team_names(gamelog_games, ['home_team', 'away_team'])
                
                # gameId oluştur (tarih ve matchup'tan)
                gamelog_games['gameId'] = (
                    gamelog_games['game_date'].astype(str) + '_' + 
                    gamelog_games['matchup'].astype(str)
                ).apply(lambda x: hash(x) % 1000000000)  # Basit hash
                
                # NocturneBear verilerine ekle
                nocturnebear_df = pd.concat([nocturnebear_df, gamelog_games], ignore_index=True)
                print(f"Gamelog verileri eklendi. Toplam satir: {len(nocturnebear_df)}")
        except Exception as e:
            print(f"Uyari: Gamelog'dan veri cikarilirken hata: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Takım istatistiklerini yükle ve merge et
    if Path(team_stats_path).exists():
        print("\n" + "=" * 60)
        print("3. Takım İstatistiklerini Yükleme ve Birleştirme")
        print("=" * 60)
        team_stats_df = pd.read_csv(team_stats_path, low_memory=False)
        
        # Takım isimlerini standartlaştır (TEAM sütunu için)
        team_stats_df = standardize_team_names(team_stats_df, ['TEAM'])
        
        print(f"\nTakım istatistikleri standartlaştırıldı:")
        print(f"  Örnek takım isimleri: {list(team_stats_df['TEAM'].unique()[:5])}")
        
        nocturnebear_df = merge_team_stats(
            nocturnebear_df,
            team_stats_df,
            date_col=date_col if date_col else 'game_date',
            home_team_col='home_team',
            away_team_col='away_team',
            team_col_in_stats='TEAM'
        )
    else:
        print(f"\nUyarı: Takım istatistikleri dosyası bulunamadı: {team_stats_path}")
    
    # 7. Rest days stats merge et
    rest_days_path = Path('data_raw/nbastuffer_2025_2026_rest_days_stats.csv')
    if rest_days_path.exists():
        print("\n" + "=" * 60)
        print("4. Rest Days İstatistiklerini Birleştirme")
        print("=" * 60)
        nocturnebear_df = merge_rest_days_stats(
            nocturnebear_df,
            rest_days_path,
            date_col=date_col if date_col else 'game_date'
        )
    else:
        print(f"\nUyarı: Rest days stats dosyası bulunamadı: {rest_days_path}")
    
    # 7b. Schedule rest days merge et (takım bazlı genel istatistikler)
    schedule_rest_days_path = Path('data_raw/nbastuffer_2025_2026_schedule_rest_days.csv')
    if schedule_rest_days_path.exists():
        print("\n" + "=" * 60)
        print("4b. Schedule Rest Days İstatistiklerini Birleştirme")
        print("=" * 60)
        try:
            schedule_df = pd.read_csv(schedule_rest_days_path, low_memory=False)
            
            # Takım isimlerini standartlaştır
            if 'TEAMS' in schedule_df.columns:
                schedule_df = standardize_team_names(schedule_df, ['TEAMS'])
                team_col = 'TEAMS'
            else:
                print("⚠ Schedule rest days'te takım sütunu bulunamadı")
                team_col = None
            
            if team_col:
                # İstatistik sütunlarını belirle (RANK hariç, boş sütunları hariç tut)
                stat_cols = [col for col in schedule_df.columns 
                            if col not in [team_col, 'RANK']]
                stat_cols = [col for col in stat_cols 
                            if not schedule_df[col].isnull().all()]
                
                # Home team schedule stats
                home_schedule = schedule_df.copy()
                home_schedule.columns = [f'home_schedule_{col}' if col in stat_cols else col 
                                        for col in home_schedule.columns]
                home_schedule = home_schedule.rename(columns={team_col: 'home_team'})
                
                nocturnebear_df = nocturnebear_df.merge(
                    home_schedule,
                    on='home_team',
                    how='left',
                    suffixes=('', '_home_schedule')
                )
                
                # Away team schedule stats
                away_schedule = schedule_df.copy()
                away_schedule.columns = [f'away_schedule_{col}' if col in stat_cols else col 
                                         for col in away_schedule.columns]
                away_schedule = away_schedule.rename(columns={team_col: 'away_team'})
                
                nocturnebear_df = nocturnebear_df.merge(
                    away_schedule,
                    on='away_team',
                    how='left',
                    suffixes=('', '_away_schedule')
                )
                
                print(f"Schedule rest days birleştirildi. Toplam sütun sayısı: {len(nocturnebear_df.columns)}")
        except Exception as e:
            print(f"⚠ Schedule rest days birleştirilirken hata: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nUyarı: Schedule rest days dosyası bulunamadı: {schedule_rest_days_path}")
    
    # 8. Tarihe göre sırala
    if date_col and date_col in nocturnebear_df.columns:
        nocturnebear_df = nocturnebear_df.sort_values(by=date_col).reset_index(drop=True)
    
    # 8. Çıktı dosyalarını oluştur
    print("\n" + "=" * 60)
    print("4. Çıktı Dosyalarını Oluşturma")
    print("=" * 60)
    
    # 8.1 long_term_2010_2025.csv
    long_term_df = nocturnebear_df.copy()
    long_term_path = output_dir / 'long_term_2010_2025.csv'
    long_term_df.to_csv(long_term_path, index=False)
    print(f"[OK] long_term_2010_2025.csv kaydedildi: {len(long_term_df)} satir")
    results['long_term'] = long_term_df
    
    # 8.2 season_2025_current.csv (sadece 2025 sezonu)
    if date_col and date_col in nocturnebear_df.columns:
        season_2025_df = nocturnebear_df[
            nocturnebear_df[date_col] >= pd.Timestamp('2025-01-01')
        ].copy()
        
        if len(season_2025_df) > 0:
            season_2025_path = output_dir / 'season_2025_current.csv'
            season_2025_df.to_csv(season_2025_path, index=False)
            print(f"[OK] season_2025_current.csv kaydedildi: {len(season_2025_df)} satir")
            results['season_2025'] = season_2025_df
        else:
            print("⚠ 2025 sezonu verisi bulunamadı")
    else:
        print("⚠ Tarih sütunu olmadığı için season_2025_current.csv oluşturulamadı")
    
    # 8.3 master_merged.csv
    master_df = nocturnebear_df.copy()
    master_path = output_dir / 'master_merged.csv'
    master_df.to_csv(master_path, index=False)
    print(f"[OK] master_merged.csv kaydedildi: {len(master_df)} satir, {len(master_df.columns)} sutun")
    results['master'] = master_df
    
    print("\n" + "=" * 60)
    print("Tamamlandı!")
    print("=" * 60)
    
    return results

