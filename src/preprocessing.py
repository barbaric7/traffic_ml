"""
=============================================================
  Metro Interstate Traffic Volume — Preprocessing
  BDA Course Project
=============================================================
  Cleans raw data, engineers features, saves cleaned CSV.
  Run: python preprocessing.py
=============================================================
"""

import pandas as pd
import numpy as np
import os

def preprocess_data(
    input_path="../data/raw/Metro_Interstate_Traffic_Volume.csv",
    output_path="../data/processed/cleaned.csv"
):
    print("=" * 55)
    print("  PREPROCESSING — Metro Interstate Traffic Volume")
    print("=" * 55)

    df = pd.read_csv(input_path)
    print(f"✅ Loaded  : {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── 1. Parse datetime ─────────────────────────────────────
    df["date_time"] = pd.to_datetime(df["date_time"])

    # ── 2. Remove physically impossible temperature (0 K rows) ─
    #       Only needed on raw data — safe to skip if already cleaned
    before = len(df)
    df = df[df["temp"] > 200]          # 200 K ≈ -73 °C lower bound
    print(f"🌡️  Temp filter  : removed {before - len(df)} rows (temp ≤ 200 K)")

    # ── 3. Remove rain extreme outlier ────────────────────────
    before = len(df)
    df = df[df["rain_1h"] < 9000]
    print(f"🌧️  Rain filter  : removed {before - len(df)} extreme rows")

    # ── 4. Remove duplicate timestamps ────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset="date_time", keep="last").reset_index(drop=True)
    print(f"🔁 Dedup        : removed {before - len(df)} duplicate timestamps")

    # ── 5. Holiday handling ────────────────────────────────────
    df["holiday"]    = df["holiday"].fillna("None")
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    df = df.drop(columns=["holiday"])

    # ── 6. Time-based features ────────────────────────────────
    df["hour"]       = df["date_time"].dt.hour
    df["day_of_week"]= df["date_time"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]      = df["date_time"].dt.month

    # ── 7. Derived binary features ────────────────────────────
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"]    = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # ── 8. Temperature: Kelvin → Celsius ──────────────────────
    df["temp_c"] = df["temp"] - 273.15
    df = df.drop(columns=["temp"])

    # ── 9. Drop low-value text columns ────────────────────────
    drop_cols = ["weather_description"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"🗑️  Dropped      : {col}")

    # ── 10. One-hot encode weather_main ───────────────────────
    #        drop_first=False → keep ALL categories for transparency
    df = pd.get_dummies(df, columns=["weather_main"], drop_first=False)

    # ── 11. Drop date_time before saving ──────────────────────
    #        datetime64 causes issues when reloaded for sklearn
    if "date_time" in df.columns:
        df = df.drop(columns=["date_time"])

    # ── 12. Safety: drop any remaining non-numeric columns ────
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        print(f"⚠️  Dropping non-numeric: {obj_cols}")
        df = df.drop(columns=obj_cols)

    # ── 13. Final check ───────────────────────────────────────
    print(f"\n✅ Final shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Target range : {df['traffic_volume'].min()} – {df['traffic_volume'].max()}")
    print(f"   Null values  : {df.isnull().sum().sum()}")
    print(f"   Columns      : {list(df.columns)}")

    # ── 14. Save ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to     : {output_path}")

    return df


if __name__ == "__main__":
    preprocess_data()