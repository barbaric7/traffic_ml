"""
=============================================================
  Metro Interstate Traffic Volume — Prediction Utility
  BDA Course Project
=============================================================
  Usage:
    from predict import predict_traffic
    result = predict_traffic(temp_c=25, rain_1h=0, snow_1h=0,
                             clouds_all=40, hour=8, day_of_week=0,
                             month=6, is_holiday=0, weather="Clear")
=============================================================
"""

import joblib
import numpy as np
import pandas as pd

# ── Load model + column list once at import time ──────────
MODEL_PATH = "../models/traffic_model.pkl"
COL_PATH   = "../models/columns.pkl"

_model   = joblib.load(MODEL_PATH)
_columns = joblib.load(COL_PATH)   # exact column order used during training


def predict_traffic(
    temp_c: float,
    rain_1h: float,
    snow_1h: float,
    clouds_all: int,
    hour: int,
    day_of_week: int,
    month: int,
    is_holiday: int = 0,
    weather: str = "Clear"
) -> dict:
    """
    Predict hourly traffic volume on I-94.

    Parameters
    ----------
    temp_c      : Temperature in Celsius
    rain_1h     : Rain in last 1 hour (mm)
    snow_1h     : Snow in last 1 hour (mm)
    clouds_all  : Cloud cover (%)
    hour        : Hour of day (0–23)
    day_of_week : 0=Monday … 6=Sunday
    month       : 1–12
    is_holiday  : 1 if public holiday, else 0
    weather     : One of Clear / Clouds / Rain / Snow / Fog /
                  Drizzle / Thunderstorm / Mist / Haze / Smoke /
                  Squall / Dust / Sand / Ash / Tornado

    Returns
    -------
    dict with keys: predicted_volume, traffic_level, confidence_band
    """

    is_weekend = int(day_of_week >= 5)
    is_peak    = int(hour in [7, 8, 9, 16, 17, 18, 19])

    # Build base dict with numeric features
    base = {
        "temp_c"      : temp_c,
        "rain_1h"     : rain_1h,
        "snow_1h"     : snow_1h,
        "clouds_all"  : clouds_all,
        "hour"        : hour,
        "day_of_week" : day_of_week,
        "month"       : month,
        "is_weekend"  : is_weekend,
        "is_holiday"  : is_holiday,
        "is_peak"     : is_peak,
    }

    # ── One-hot encode weather correctly ──────────────────
    #    Instead of pd.get_dummies (which only sees 1 row),
    #    we manually set the matching column to 1.
    #    All other weather_main_* columns default to 0 via reindex.
    weather_col = f"weather_main_{weather}"
    if weather_col in _columns:
        base[weather_col] = 1
    # If the weather label isn't in training columns, all flags stay 0
    # (model falls back to the dropped-first baseline)

    # Align to exact training column order; fill unseen cols with 0
    input_df = pd.DataFrame([base]).reindex(columns=_columns, fill_value=0)

    prediction = float(_model.predict(input_df)[0])
    prediction = max(0.0, prediction)   # traffic can't be negative

    # Human-readable level
    if prediction < 1000:
        level = "Very Low"
    elif prediction < 2500:
        level = "Low"
    elif prediction < 4000:
        level = "Medium"
    elif prediction < 5500:
        level = "High"
    else:
        level = "Very High"

    return {
        "predicted_volume": int(round(prediction)),
        "traffic_level"   : level,
    }


# ── Quick smoke-test ───────────────────────────────────────
if __name__ == "__main__":
    result = predict_traffic(
        temp_c=20, rain_1h=0, snow_1h=0, clouds_all=30,
        hour=8, day_of_week=0, month=6, weather="Clear"
    )
    print("Test prediction:", result)