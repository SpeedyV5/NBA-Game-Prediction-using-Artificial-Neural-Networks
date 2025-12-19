# src/models/sanity_check.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main(csv_path: str = "data_processed/model_dataset.csv", season_focus: str = "2025-26") -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, low_memory=False)

    print("=" * 70)
    print(f"[OK] Loaded: {csv_path} | shape = {df.shape[0]} rows x {df.shape[1]} cols")
    print("=" * 70)

    # 1) Basic columns existence
    must_cols = ["gameId", "game_date", "season_year", "home_team", "away_team", "home_team_win"]
    missing = [c for c in must_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing expected columns: {missing}")
    else:
        print("[OK] Expected core columns present.")

    # 2) Label distribution
    if "home_team_win" in df.columns:
        y = df["home_team_win"]
        y_valid = y.dropna()
        if len(y_valid) == 0:
            print("[WARN] home_team_win has no valid values.")
        else:
            vc = y_valid.astype(int).value_counts().sort_index()
            total = vc.sum()
            print("\nLabel distribution (home_team_win):")
            for k, v in vc.items():
                print(f"  {k}: {v} ({v/total*100:.2f}%)")
            print(f"  total labeled: {total} / {len(df)}")
    else:
        print("[WARN] home_team_win column not found.")

    # 3) NaN check (overall + top columns with most NaN)
    na_counts = df.isna().sum()
    na_total = int(na_counts.sum())
    if na_total == 0:
        print("\n[OK] No NaN values in the dataset.")
    else:
        print(f"\n[WARN] NaN present. Total NaN cells: {na_total}")
        top = na_counts[na_counts > 0].sort_values(ascending=False).head(15)
        print("Top columns by NaN count:")
        for col, cnt in top.items():
            print(f"  {col}: {int(cnt)} ({cnt/len(df)*100:.2f}% rows)")

    # 4) Injury features check (optional)
    injury_cols = [
        c for c in df.columns
        if ("top3_" in c and c.endswith("_out_count")) or c.startswith("diff_top3_")
    ]

    if not injury_cols:
        print("\n[INFO] No injury feature columns found. (This is expected if injury features were disabled.)")
        print("\nDone.")
        return

    print("\nInjury feature columns found:")
    print("  " + ", ".join(injury_cols))

    # Ensure season_year filter
    if "season_year" not in df.columns:
        print("[WARN] season_year missing; cannot focus-check 2025-26.")
        print("\nDone.")
        return

    mask = df["season_year"].astype(str) == str(season_focus)
    n_focus = int(mask.sum())
    print(f"\nSeason focus = {season_focus}: {n_focus} rows")

    if n_focus == 0:
        print("[WARN] No rows for that season in dataset. Injury features will naturally look empty.")
        print("\nDone.")
        return

    # Numeric coercion for injury cols
    inj_focus = df.loc[mask, injury_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    nonzero_per_col = (inj_focus != 0).sum().sort_values(ascending=False)
    any_nonzero_rows = int((inj_focus.sum(axis=1) != 0).sum())

    print("\nInjury non-zero counts (within season focus):")
    for col, cnt in nonzero_per_col.items():
        print(f"  {col}: {int(cnt)} non-zero rows ({cnt/n_focus*100:.2f}%)")

    print(
        f"\n[CHECK] Rows with ANY non-zero injury signal in {season_focus}: "
        f"{any_nonzero_rows} / {n_focus} ({any_nonzero_rows/n_focus*100:.2f}%)"
    )

    # Quick peek of some non-zero examples
    if any_nonzero_rows > 0:
        example = df.loc[mask].copy()
        example[injury_cols] = inj_focus
        ex = example[example[injury_cols].sum(axis=1) != 0].head(10)
        show_cols = [c for c in ["game_date", "home_team", "away_team", "season_year"] if c in ex.columns] + injury_cols
        print("\nSample rows with non-zero injury features (first 10):")
        print(ex[show_cols].to_string(index=False))
    else:
        print(
            "\n[WARN] All injury features are zero in that season. "
            "Possible reasons: (1) injuries_latest.csv dates don't match game_date, "
            "(2) team names don't match after standardization, "
            "(3) player_name format mismatch with top3_map."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
