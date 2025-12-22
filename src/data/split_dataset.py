"""
NBA Dataset Splitting Module

Bu modül, zaman bazlı ve random shuffle train/val/test split işlemlerini gerçekleştirir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def analyze_date_range(df: pd.DataFrame, date_col: str = "game_date") -> Dict:
    """
    Verideki tarih aralığını analiz et.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek dataframe
    date_col : str
        Tarih kolonu adı
        
    Returns
    -------
    dict
        Tarih istatistikleri
    """
    if date_col not in df.columns:
        raise ValueError(f"Tarih kolonu bulunamadı: {date_col}")
    
    dates = pd.to_datetime(df[date_col], errors='coerce')
    
    stats = {
        'min_date': dates.min(),
        'max_date': dates.max(),
        'total_games': len(df),
        'date_range_days': (dates.max() - dates.min()).days,
        'games_per_year': df.groupby(dates.dt.year).size().to_dict(),
    }
    
    return stats


def split_dataset_by_time(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: Optional[str] = None,
    date_col: str = "game_date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Veriyi zaman bazlı olarak train/val/test setlerine ayır.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bölünecek dataframe
    train_end : str
        Train seti bitiş tarihi (bu tarihten önceki veriler train)
        Format: "YYYY-MM-DD"
    val_end : str
        Validation seti bitiş tarihi (train_end ile val_end arası val)
        Format: "YYYY-MM-DD"
    test_end : str, optional
        Test seti bitiş tarihi. None ise tüm kalan veri test olur.
        Format: "YYYY-MM-DD"
    date_col : str
        Tarih kolonu adı
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    if date_col not in df.columns:
        raise ValueError(f"Tarih kolonu bulunamadı: {date_col}")
    
    # Tarihleri datetime'a çevir
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    test_end_dt = pd.to_datetime(test_end) if test_end else None
    
    # Tarihe göre sırala
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Split
    train_mask = df[date_col] < train_end_dt
    val_mask = (df[date_col] >= train_end_dt) & (df[date_col] < val_end_dt)
    
    if test_end_dt:
        test_mask = (df[date_col] >= val_end_dt) & (df[date_col] < test_end_dt)
    else:
        test_mask = df[date_col] >= val_end_dt
    
    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path] = "data_processed/",
    prefix: str = ""
) -> Dict[str, Path]:
    """
    Split dosyalarını kaydet.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Train seti
    val_df : pd.DataFrame
        Validation seti
    test_df : pd.DataFrame
        Test seti
    output_dir : str or Path
        Çıktı klasörü
    prefix : str
        Dosya adı prefix'i (ör: "injury_" -> "injury_train_set.csv")
        
    Returns
    -------
    dict
        Kaydedilen dosya yolları
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    train_path = output_dir / f"{prefix}train_set.csv"
    val_path = output_dir / f"{prefix}val_set.csv"
    test_path = output_dir / f"{prefix}test_set.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    paths['train'] = train_path
    paths['val'] = val_path
    paths['test'] = test_path
    
    return paths


def print_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "game_date",
    label_col: str = "home_team_win"
) -> None:
    """
    Split istatistiklerini yazdır.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Train seti
    val_df : pd.DataFrame
        Validation seti
    test_df : pd.DataFrame
        Test seti
    date_col : str
        Tarih kolonu adı
    label_col : str
        Label kolonu adı
    """
    total = len(train_df) + len(val_df) + len(test_df)
    
    print("\n" + "=" * 60)
    print("SPLIT İSTATİSTİKLERİ")
    print("=" * 60)
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df) == 0:
            print(f"\n{name}: 0 maç (0%)")
            continue
            
        pct = len(df) / total * 100
        
        dates = pd.to_datetime(df[date_col], errors='coerce')
        min_date = dates.min()
        max_date = dates.max()
        
        print(f"\n{name}:")
        print(f"  Maç sayısı: {len(df):,} ({pct:.1f}%)")
        print(f"  Tarih aralığı: {min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}")
        
        if label_col in df.columns:
            label_counts = df[label_col].value_counts()
            if len(label_counts) > 0:
                win_rate = label_counts.get(1, 0) / len(df) * 100
                print(f"  Home win rate: {win_rate:.1f}%")
    
    print("\n" + "=" * 60)


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "game_date",
    required_cols: Optional[list] = None,
    check_date_overlap: bool = True
) -> bool:
    """
    Split'lerin geçerliliğini kontrol et.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Train seti
    val_df : pd.DataFrame
        Validation seti
    test_df : pd.DataFrame
        Test seti
    date_col : str
        Tarih kolonu adı
    required_cols : list, optional
        Her split'te olması gereken kolonlar
    check_date_overlap : bool
        Tarih overlap kontrolü yapılsın mı? (Random split için False)
        
    Returns
    -------
    bool
        Geçerli mi?
    """
    is_valid = True
    errors = []
    
    # 1. Boşluk kontrolü
    if len(train_df) == 0:
        errors.append("Train seti boş!")
        is_valid = False
    
    if len(val_df) == 0:
        errors.append("Val seti boş!")
        is_valid = False
    
    if len(test_df) == 0:
        errors.append("Test seti boş!")
        is_valid = False
    
    # 2. Overlap kontrolü (sadece time-based split için)
    if check_date_overlap:
        if date_col in train_df.columns and date_col in val_df.columns:
            train_max = pd.to_datetime(train_df[date_col]).max()
            val_min = pd.to_datetime(val_df[date_col]).min()
            
            if train_max >= val_min:
                errors.append(f"Train-Val overlap! Train max: {train_max}, Val min: {val_min}")
                is_valid = False
        
        if date_col in val_df.columns and date_col in test_df.columns:
            val_max = pd.to_datetime(val_df[date_col]).max()
            test_min = pd.to_datetime(test_df[date_col]).min()
            
            if val_max >= test_min:
                errors.append(f"Val-Test overlap! Val max: {val_max}, Test min: {test_min}")
                is_valid = False
    
    # 3. Gerekli kolon kontrolü
    if required_cols:
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                errors.append(f"{name}'de eksik kolonlar: {missing}")
                is_valid = False
    
    # Sonuçları yazdır
    if errors:
        print("\n⚠️ VALIDATION HATALARI:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✅ Tüm validation kontrolleri başarılı!")
    
    return is_valid


def create_time_based_split(
    input_csv: Union[str, Path],
    output_dir: Union[str, Path] = "data_processed/",
    train_end: str = "2021-07-01",
    val_end: str = "2023-07-01",
    test_end: Optional[str] = None,
    date_col: str = "game_date",
    required_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tam split pipeline'ı çalıştır.
    
    Parameters
    ----------
    input_csv : str or Path
        Girdi CSV dosyası
    output_dir : str or Path
        Çıktı klasörü
    train_end : str
        Train seti bitiş tarihi
    val_end : str
        Validation seti bitiş tarihi
    test_end : str, optional
        Test seti bitiş tarihi
    date_col : str
        Tarih kolonu adı
    required_cols : list, optional
        Her split'te olması gereken kolonlar
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    print("=" * 60)
    print("DATASET SPLITTING PIPELINE")
    print("=" * 60)
    
    # 1. Veriyi yükle
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV bulunamadı: {input_csv}")
    
    print(f"\n1. Veri yükleniyor: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"   {len(df):,} maç, {len(df.columns)} kolon")
    
    # 2. Tarih aralığını analiz et
    print("\n2. Tarih aralığı analizi:")
    stats = analyze_date_range(df, date_col)
    print(f"   Min tarih: {stats['min_date']}")
    print(f"   Max tarih: {stats['max_date']}")
    print(f"   Toplam gün: {stats['date_range_days']}")
    
    # 3. Split yap
    print(f"\n3. Split yapılıyor:")
    print(f"   Train: < {train_end}")
    print(f"   Val:   {train_end} - {val_end}")
    print(f"   Test:  >= {val_end}" + (f" and < {test_end}" if test_end else ""))
    
    train_df, val_df, test_df = split_dataset_by_time(
        df, train_end, val_end, test_end, date_col
    )
    
    # 4. İstatistikleri göster
    print_split_stats(train_df, val_df, test_df, date_col)
    
    # 5. Validation
    print("\n4. Validation:")
    default_required = ["home_team_win", "score_diff", date_col]
    if required_cols:
        default_required.extend(required_cols)
    
    is_valid = validate_splits(train_df, val_df, test_df, date_col, default_required)
    
    if not is_valid:
        print("\n⚠️ Uyarı: Validation hataları var, ancak split dosyaları yine de kaydedilecek.")
    
    # 6. Kaydet
    print("\n5. Dosyalar kaydediliyor...")
    paths = save_splits(train_df, val_df, test_df, output_dir)
    
    for name, path in paths.items():
        print(f"   ✅ {name}: {path}")
    
    print("\n" + "=" * 60)
    print("Split işlemi tamamlandı!")
    print("=" * 60)
    
    return train_df, val_df, test_df


def suggest_split_dates(
    df: pd.DataFrame,
    date_col: str = "game_date",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[str, str]:
    """
    Veri boyutuna göre split tarihlerini öner.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    date_col : str
        Tarih kolonu adı
    train_ratio : float
        Train seti oranı (varsayılan 0.7)
    val_ratio : float
        Validation seti oranı (varsayılan 0.15)
        
    Returns
    -------
    tuple
        (train_end, val_end) tarih stringleri
    """
    dates = pd.to_datetime(df[date_col], errors='coerce').sort_values()
    
    n = len(dates)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_end = dates.iloc[train_idx].strftime('%Y-%m-%d')
    val_end = dates.iloc[val_idx].strftime('%Y-%m-%d')
    
    print(f"Önerilen split tarihleri (train={train_ratio:.0%}, val={val_ratio:.0%}):")
    print(f"  Train end: {train_end}")
    print(f"  Val end: {val_end}")
    
    return train_end, val_end


def split_dataset_random(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Veriyi rastgele karıştırarak train/val/test setlerine ayır.
    
    Bu yöntem, tüm yıllardan veri içeren dengeli setler oluşturur
    ve distribution drift sorununu azaltır.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bölünecek dataframe
    train_ratio : float
        Train seti oranı (varsayılan 0.70)
    val_ratio : float
        Validation seti oranı (varsayılan 0.15)
    test_ratio : float
        Test seti oranı (varsayılan 0.15)
    random_state : int
        Reproducibility için random seed (varsayılan 42)
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    # Oranları kontrol et
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Oranların toplamı 1.0 olmalı, şu an: {total_ratio}")
    
    df = df.copy().reset_index(drop=True)
    
    # İlk bölme: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    # İkinci bölme: val vs test
    # val_ratio / (val_ratio + test_ratio) = val'ın temp içindeki oranı
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    # Index'leri resetle
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df


def print_split_stats_random(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "game_date",
    label_col: str = "home_team_win"
) -> None:
    """
    Random split istatistiklerini yazdır (yıl dağılımı ile).
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Train seti
    val_df : pd.DataFrame
        Validation seti
    test_df : pd.DataFrame
        Test seti
    date_col : str
        Tarih kolonu adı
    label_col : str
        Label kolonu adı
    """
    total = len(train_df) + len(val_df) + len(test_df)
    
    print("\n" + "=" * 60)
    print("RANDOM SPLIT İSTATİSTİKLERİ")
    print("=" * 60)
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df) == 0:
            print(f"\n{name}: 0 maç (0%)")
            continue
            
        pct = len(df) / total * 100
        
        print(f"\n{name}:")
        print(f"  Maç sayısı: {len(df):,} ({pct:.1f}%)")
        
        # Tarih bilgisi varsa yıl dağılımını göster
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            min_date = dates.min()
            max_date = dates.max()
            print(f"  Tarih aralığı: {min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}")
            
            # Yıl dağılımı
            year_counts = dates.dt.year.value_counts().sort_index()
            years_str = ", ".join([f"{y}: {c}" for y, c in year_counts.items()])
            print(f"  Yıl dağılımı: {years_str}")
        
        if label_col in df.columns:
            label_counts = df[label_col].value_counts()
            if len(label_counts) > 0:
                win_rate = label_counts.get(1, 0) / len(df) * 100
                print(f"  Home win rate: {win_rate:.1f}%")
    
    print("\n" + "=" * 60)


def create_random_split(
    input_csv: Union[str, Path],
    output_dir: Union[str, Path] = "data_processed/",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 30,
    date_col: str = "game_date",
    required_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Random shuffle split pipeline'ı çalıştır.
    
    Bu fonksiyon, veriyi rastgele karıştırarak train/val/test setlerine
    böler. Distribution drift sorununu azaltmak için tüm yıllardan
    veri içeren dengeli setler oluşturur.
    
    Parameters
    ----------
    input_csv : str or Path
        Girdi CSV dosyası
    output_dir : str or Path
        Çıktı klasörü
    train_ratio : float
        Train seti oranı (varsayılan 0.70)
    val_ratio : float
        Validation seti oranı (varsayılan 0.15)
    test_ratio : float
        Test seti oranı (varsayılan 0.15)
    random_state : int
        Reproducibility için random seed (varsayılan 42)
    date_col : str
        Tarih kolonu adı
    required_cols : list, optional
        Her split'te olması gereken kolonlar
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    print("=" * 60)
    print("RANDOM SHUFFLE SPLIT PIPELINE")
    print("=" * 60)
    
    # 1. Veriyi yükle
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV bulunamadı: {input_csv}")
    
    print(f"\n1. Veri yükleniyor: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"   {len(df):,} maç, {len(df.columns)} kolon")
    
    # 2. Tarih aralığını analiz et
    print("\n2. Tarih aralığı analizi:")
    stats = analyze_date_range(df, date_col)
    print(f"   Min tarih: {stats['min_date']}")
    print(f"   Max tarih: {stats['max_date']}")
    print(f"   Toplam gün: {stats['date_range_days']}")
    print(f"   Yıllara göre maç sayıları: {stats['games_per_year']}")
    
    # 3. Random Split yap
    print(f"\n3. Random split yapılıyor:")
    print(f"   Train: {train_ratio:.0%}")
    print(f"   Val:   {val_ratio:.0%}")
    print(f"   Test:  {test_ratio:.0%}")
    print(f"   Random state: {random_state}")
    
    train_df, val_df, test_df = split_dataset_random(
        df, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # 4. İstatistikleri göster
    print_split_stats_random(train_df, val_df, test_df, date_col)
    
    # 5. Validation (tarih overlap kontrolü kapalı)
    print("\n4. Validation:")
    default_required = ["home_team_win", "score_diff", date_col]
    if required_cols:
        default_required.extend(required_cols)
    
    is_valid = validate_splits(
        train_df, val_df, test_df, 
        date_col, 
        default_required,
        check_date_overlap=False  # Random split için overlap kontrolü kapalı
    )
    
    if not is_valid:
        print("\n⚠️ Uyarı: Validation hataları var, ancak split dosyaları yine de kaydedilecek.")
    
    # 6. Kaydet
    print("\n5. Dosyalar kaydediliyor...")
    paths = save_splits(train_df, val_df, test_df, output_dir)
    
    for name, path in paths.items():
        print(f"   ✅ {name}: {path}")
    
    print("\n" + "=" * 60)
    print("Random split işlemi tamamlandı!")
    print("=" * 60)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Örnek kullanım - Random Split (yeni varsayılan)
    create_random_split(
        input_csv="data_processed/games_with_all_features.csv",
        output_dir="data_processed/",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=30
    )
    
    # Eski zaman bazlı split için:
    # create_time_based_split(
    #     input_csv="data_processed/games_with_all_features.csv",
    #     output_dir="data_processed/",
    #     train_end="2021-07-01",
    #     val_end="2023-07-01"
    # )

