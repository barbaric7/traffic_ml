"""
=============================================================
  Metro Interstate Traffic Volume — Model Training
  BDA Course Project
=============================================================
  Trains Random Forest (primary) + compares DT & GB,
  performs cross-validation, saves .pkl + metrics.
  Run: python train.py
=============================================================
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Run: pip install shap")

from preprocessing import preprocess_data

# ── Paths ──────────────────────────────────────────────────
RAW_PATH      = "../data/raw/Metro_Interstate_Traffic_Volume.csv"
MODEL_PATH    = "../models/traffic_model.pkl"
COL_PATH      = "../models/columns.pkl"
METRICS_PATH  = "../models/metrics.json"
SHAP_PATH     = "../models/shap_values.npy"
SHAP_COL_PATH = "../models/shap_columns.pkl"

# Fixed seed — academic reproducibility. Change here to re-run with different split.
SEED = 42

print("=" * 60)
print("  TRAINING — Metro Interstate Traffic Volume")
print("=" * 60)

# ── 1. Preprocess ──────────────────────────────────────────
df = preprocess_data(RAW_PATH)
X  = df.drop(["traffic_volume"], axis=1)
y  = df["traffic_volume"]
feature_cols = X.columns.tolist()
print(f"\n📐 Features used : {len(feature_cols)}")
print(f"   {feature_cols}")

# ── 2. Train / Test split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"\n✂️  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# ── 3. Define all candidate models ────────────────────────
candidate_models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10, random_state=SEED
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, n_jobs=-1, random_state=SEED
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED
    ),
}

# ── 4. 5-Fold Cross-Validation on ALL models ──────────────
print("\n" + "─" * 60)
print("  STEP 1 — 5-Fold Cross-Validation (all models)")
print("─" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

for name, m in candidate_models.items():
    cv_r2   = cross_val_score(m, X_train, y_train, cv=kf,
                               scoring="r2", n_jobs=-1)
    cv_mae  = cross_val_score(m, X_train, y_train, cv=kf,
                               scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = cross_val_score(m, X_train, y_train, cv=kf,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_results[name] = {
        "CV_R2_mean"   : round(float(cv_r2.mean()),   4),
        "CV_R2_std"    : round(float(cv_r2.std()),    4),
        "CV_MAE_mean"  : round(float(-cv_mae.mean()), 1),
        "CV_MAE_std"   : round(float(cv_mae.std()),   1),
        "CV_RMSE_mean" : round(float(-cv_rmse.mean()),1),
        "CV_RMSE_std"  : round(float(cv_rmse.std()),  1),
        "CV_R2_per_fold"  : [round(float(v), 4) for v in cv_r2],
        "CV_MAE_per_fold" : [round(float(v), 1) for v in -cv_mae],
    }
    print(f"\n  [{name}]")
    print(f"    CV R²   : {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")
    print(f"    CV MAE  : {-cv_mae.mean():,.1f}  ±  {cv_mae.std():.1f}")
    print(f"    CV RMSE : {-cv_rmse.mean():,.1f}  ±  {cv_rmse.std():.1f}")

# ── 5. Train ALL models on full train set & evaluate on test ─
print("\n" + "─" * 60)
print("  STEP 2 — Full Train + Test Evaluation (all models)")
print("─" * 60)

all_model_metrics = {}
all_predictions   = {}
trained_models    = {}

for name, m in candidate_models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)

    mae   = mean_absolute_error(y_test, pred)
    mse   = mean_squared_error(y_test, pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_test, pred)
    mape  = mean_absolute_percentage_error(y_test, pred) * 100  # as %
    # Adjusted R²
    n, p  = len(y_test), X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    all_model_metrics[name] = {
        "MAE"     : round(float(mae),    1),
        "MSE"     : round(float(mse),    1),
        "RMSE"    : round(float(rmse),   1),
        "R2"      : round(float(r2),     4),
        "Adj_R2"  : round(float(adj_r2), 4),
        "MAPE"    : round(float(mape),   2),
    }
    all_predictions[name] = pred.tolist()
    trained_models[name]  = m

    print(f"\n  [{name}]")
    print(f"    MAE      : {mae:,.1f}")
    print(f"    RMSE     : {rmse:,.1f}")
    print(f"    R²       : {r2:.4f}")
    print(f"    Adj R²   : {adj_r2:.4f}")
    print(f"    MAPE     : {mape:.2f}%")

# ── 6. Save the best model (Random Forest) ────────────────
best_name  = "Random Forest"
best_model = trained_models[best_name]
fi         = pd.Series(best_model.feature_importances_,
                       index=feature_cols).sort_values(ascending=False)

print(f"\n  🏆 Best model : {best_name}")
print(f"  Top 5 features:")
for feat, imp in fi.head(5).items():
    print(f"    {feat:<30} {imp:.4f}")

# ── 7. SHAP values ────────────────────────────────────────
if SHAP_AVAILABLE:
    print("\n  Computing SHAP values (sample = 500)...")
    sample_X  = X_test.sample(min(500, len(X_test)), random_state=SEED)
    explainer = shap.TreeExplainer(best_model)
    shap_vals = explainer.shap_values(sample_X)
    np.save(SHAP_PATH, shap_vals)
    joblib.dump(sample_X.columns.tolist(), SHAP_COL_PATH)
    print(f"  ✅ SHAP saved → {SHAP_PATH}")
else:
    print("  ⚠️  SHAP skipped (not installed)")

# ── 8. Save hyperparameters of each model ─────────────────
hyperparams = {
    "Decision Tree"     : {"max_depth": 10},
    "Random Forest"     : {"n_estimators": 200, "max_depth": 15},
    "Gradient Boosting" : {"n_estimators": 200, "max_depth": 5,
                           "learning_rate": 0.1},
}

# ── 9. Assemble full metrics dict ─────────────────────────
all_metrics = {
    # Best model test metrics (backward compat with dashboard)
    "test_metrics": all_model_metrics[best_name],

    # Per-model full test evaluation
    "all_model_metrics": all_model_metrics,

    # Per-model cross-validation results
    "cv_comparison": cv_results,

    # Feature importance (best model)
    "feature_importance": {k: round(float(v), 5)
                           for k, v in fi.items()},

    # Hyperparameters reference
    "hyperparameters": hyperparams,

    # Dataset info
    "train_size" : len(X_train),
    "test_size"  : len(X_test),
    "n_features" : len(feature_cols),
    "feature_cols": feature_cols,
    "seed"       : SEED,
    "best_model" : best_name,
}

# ── 10. Save all artifacts ────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(best_model,  MODEL_PATH)
joblib.dump(feature_cols, COL_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "=" * 60)
print("  SAVED")
print("=" * 60)
print(f"  Best model  → {MODEL_PATH}")
print(f"  Columns     → {COL_PATH}")
print(f"  Metrics     → {METRICS_PATH}")
print("\n✅ Training complete!")