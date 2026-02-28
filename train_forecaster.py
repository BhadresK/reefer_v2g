#!/usr/bin/env python3
"""
train_forecaster.py
═══════════════════════════════════════════════════════════════════════════════
Trains an electricity price forecasting model on real SMARD EPEX data.
Model: Gradient Boosted Trees (sklearn GradientBoostingRegressor, equivalent
       to XGBoost in accuracy for this use case; no external install needed).

Input:   data/smard_prices_processed.csv  (from fetch_smard_data.py)
Outputs: data/price_model.pkl             — trained model
         data/model_metrics.json          — MAE, MAPE, R², feature importances
         data/forecaster_validation.png   — diagnostic chart

Architecture:
    One-step-ahead (15-min) model:
    • Input features: 20 lag/rolling/calendar features
    • Target: price_EUR_MWh at t+1 (next 15-min slot)
    • Also produces 24h-ahead rolling forecast by chaining predictions
      (used by MPC layer in run_optimisation.py)

Training protocol (walk-forward validation, Lago et al. 2021):
    • Train: 2022-01-01 – 2023-12-31
    • Validation: 2024-01-01 – 2024-06-30
    • Test (held out): 2024-07-01 – 2024-12-31

References:
    Lago et al. (2021) Forecasting day-ahead electricity prices: A review
    Liu et al. (2023) Stochastic MPC for EV charging under price uncertainty
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

# ── Auto-install required packages if missing ─────────────────────────────────
def _ensure(package: str, import_name: str | None = None):
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"  Installing {package} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            stdout=subprocess.DEVNULL,
        )
        print(f"  {package} installed OK.")

_ensure("scikit-learn", "sklearn")
_ensure("numpy")
_ensure("pandas")
_ensure("matplotlib")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
Path("data").mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature specification (must match fetch_smard_data.py)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Lag features — prices at previous time steps
    "lag_1", "lag_2", "lag_3", "lag_4",
    "lag_8", "lag_12", "lag_24",
    "lag_48", "lag_96",           # yesterday same time
    "lag_192",                     # 2 days ago
    "lag_672",                     # last week same time
    # Rolling statistics
    "roll_4h_mean", "roll_24h_mean", "roll_24h_std", "roll_7d_mean",
    # Calendar (cyclical encoding)
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    # Binary flags
    "is_weekend", "is_holiday",
    # Season index
    "season",
]

TARGET_COL = "price_EUR_MWh"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading and preparation
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = "data/smard_prices_processed.csv") -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run fetch_smard_data.py first."
        )
    df = pd.read_csv(path, index_col=0)
    # Robustly convert index to timezone-aware DatetimeIndex (pandas 2.x safe)
    raw_idx = pd.to_datetime(df.index, utc=True)   # parse as UTC first — always works
    df.index = raw_idx.tz_convert("Europe/Berlin")  # then localise to CET/CEST
    df.index.name = "datetime_cet"
    print(f"  Loaded {len(df):,} rows from {path}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")

    # Validate required columns
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processed data: {missing}\n"
                         "Re-run fetch_smard_data.py to regenerate features.")
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Walk-forward split (avoids look-ahead bias):
        Train:      all data up to 2023-12-31
        Validation: 2024-01-01 – 2024-06-30
        Test:       2024-07-01 – end
    If data is shorter, adapt splits accordingly.
    """
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL] > -500]   # remove extreme outliers (< -500 EUR/MWh)

    # Determine split dates based on available data
    total_days = (df.index.max() - df.index.min()).days
    if total_days < 365:
        # Short data (test mode): 70/15/15 split
        n = len(df)
        i_val  = int(n * 0.70)
        i_test = int(n * 0.85)
        train = df.iloc[:i_val]
        val   = df.iloc[i_val:i_test]
        test  = df.iloc[i_test:]
        split_mode = "percentage (short dataset)"
    else:
        # Timezone-safe split — works with any tz-aware DatetimeIndex
        tz = df.index.tz  # e.g. Europe/Berlin
        train_end = pd.Timestamp("2023-12-31 23:59").tz_localize(tz)
        val_end   = pd.Timestamp("2024-06-30 23:59").tz_localize(tz)
        train = df[df.index <= train_end]
        val   = df[(df.index > train_end) & (df.index <= val_end)]
        test  = df[df.index > val_end]
        if len(test) == 0:
            # data only goes to mid-2024 — use last 15% as test
            n = len(df)
            test = df.iloc[int(n*0.85):]
            train = df.iloc[:int(n*0.70)]
            val   = df.iloc[int(n*0.70):int(n*0.85)]
        split_mode = "calendar (2024 holdout)"

    print(f"  Split mode: {split_mode}")
    print(f"  Train: {len(train):,} rows | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
#  Model training
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(fast: bool = False) -> Pipeline:
    """
    Gradient Boosted Trees pipeline.
    fast=True: fewer estimators, for quick testing.
    """
    params = dict(
        n_estimators    = 200 if fast else 500,
        learning_rate   = 0.05,
        max_depth       = 5,
        subsample       = 0.8,
        min_samples_leaf= 10,
        loss            = "huber",     # robust to extreme price spikes
        random_state    = 42,
        validation_fraction = 0.1,
        n_iter_no_change    = 30,
        tol             = 1e-4,
    )
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model",  GradientBoostingRegressor(**params)),
    ])


def train(df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          fast: bool = False) -> Pipeline:
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_val   = df_val[FEATURE_COLS].values
    y_val   = df_val[TARGET_COL].values

    print(f"  Training GradientBoostingRegressor on {len(X_train):,} samples...")
    model = build_model(fast=fast)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    mae_val  = mean_absolute_error(y_val, y_pred_val)
    mape_val = np.mean(np.abs((y_val - y_pred_val) / (np.abs(y_val) + 1e-3))) * 100
    r2_val   = r2_score(y_val, y_pred_val)
    print(f"  Validation — MAE: {mae_val:.2f} EUR/MWh | "
          f"MAPE: {mape_val:.1f}% | R²: {r2_val:.3f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation and diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model: Pipeline, df_test: pd.DataFrame) -> dict:
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values
    y_pred = model.predict(X_test)

    mae   = mean_absolute_error(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-3))) * 100
    r2    = r2_score(y_test, y_pred)
    rmse  = np.sqrt(np.mean((y_test - y_pred)**2))

    print(f"\n  ── Test Set Performance ──────────────────────────────")
    print(f"  MAE  : {mae:.2f} EUR/MWh  (typ. acceptable: <15 EUR/MWh)")
    print(f"  RMSE : {rmse:.2f} EUR/MWh")
    print(f"  MAPE : {mape:.1f}%        (typ. good: 10-25% for day-ahead)")
    print(f"  R²   : {r2:.3f}           (1.0 = perfect)")

    # Feature importance
    gbm = model.named_steps["model"]
    importances = dict(zip(FEATURE_COLS, gbm.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 features:")
    for feat, imp in top5:
        print(f"    {feat:<22} {imp:.3f}")

    return {
        "mae_EUR_MWh":  round(mae, 3),
        "rmse_EUR_MWh": round(rmse, 3),
        "mape_pct":     round(mape, 2),
        "r2":           round(r2, 4),
        "n_test":       len(y_test),
        "feature_importances": {k: round(v, 5) for k, v in
                                 sorted(importances.items(), key=lambda x: -x[1])},
        "y_test_sample":  y_test[:96].tolist(),
        "y_pred_sample":  y_pred[:96].tolist(),
    }


def plot_diagnostics(model: Pipeline, df_test: pd.DataFrame,
                     metrics: dict,
                     out: str = "data/forecaster_validation.png"):
    X_test  = df_test[FEATURE_COLS].values
    y_test  = df_test[TARGET_COL].values
    y_pred  = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Price Forecaster Validation  —  MAE={metrics['mae_EUR_MWh']:.1f} EUR/MWh  "
        f"MAPE={metrics['mape_pct']:.1f}%  R²={metrics['r2']:.3f}",
        fontsize=12, fontweight="bold"
    )

    # (1) One week of actual vs predicted
    ax = axes[0, 0]
    n_plot = min(672, len(y_test))   # 1 week
    ax.plot(y_test[:n_plot],  lw=1.0, color="steelblue",  label="Actual",    alpha=0.9)
    ax.plot(y_pred[:n_plot],  lw=1.0, color="darkorange", label="Predicted", alpha=0.8, ls="--")
    ax.set_title("(1) Actual vs Predicted — 1 Week")
    ax.set_xlabel("15-min slot"); ax.set_ylabel("EUR/MWh")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (2) Scatter actual vs predicted
    ax = axes[0, 1]
    lim = max(abs(y_test).max(), abs(y_pred).max()) * 1.05
    ax.scatter(y_test, y_pred, alpha=0.15, s=3, color="steelblue")
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.5, label="Perfect")
    ax.set_title("(2) Predicted vs Actual (scatter)")
    ax.set_xlabel("Actual (EUR/MWh)"); ax.set_ylabel("Predicted (EUR/MWh)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(); ax.grid(True, alpha=0.3)

    # (3) Residual distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.axvline(residuals.mean(), color="orange", lw=1.5,
               label=f"Mean={residuals.mean():.1f}")
    ax.set_title("(3) Residual Distribution")
    ax.set_xlabel("Residual (EUR/MWh)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (4) Feature importance
    ax = axes[1, 1]
    gbm = model.named_steps["model"]
    imp = dict(zip(FEATURE_COLS, gbm.feature_importances_))
    top_n = 12
    top   = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
    names, vals = zip(*top)
    bars = ax.barh(range(top_n), vals, color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("(4) Feature Importance (top 12)")
    ax.set_xlabel("Importance"); ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Diagnostic chart saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Forecast generation API (called by run_year_simulation.py)
# ═══════════════════════════════════════════════════════════════════════════════

def make_24h_forecast(model: Pipeline,
                      history: pd.DataFrame,
                      forecast_start: pd.Timestamp,
                      noise_std: float = 0.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate 96-slot (24h) ahead price forecast starting at forecast_start.

    Uses recursive/chained forecasting: predicts one step at a time,
    feeding each prediction back as a lag feature for the next step.

    Args:
        model:          Trained sklearn Pipeline
        history:        DataFrame with processed features up to forecast_start
        forecast_start: Timestamp of first forecast slot
        noise_std:      Optional Gaussian noise (EUR/MWh) for MPC scenario E
        rng:            Random generator (for reproducibility)

    Returns:
        np.ndarray shape (96,) of predicted EUR/MWh spot prices
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build a working buffer: last 672 slots (1 week) of history for lags
    buf_size  = 800   # > max lag of 672
    hist_vals = history["price_EUR_MWh"].values[-buf_size:]
    predictions = []

    freq = pd.tseries.frequencies.to_offset("15min")
    current_ts = forecast_start

    for step in range(96):
        # Build feature vector for this slot
        h  = current_ts.hour + current_ts.minute / 60
        mo = current_ts.month
        dw = current_ts.dayofweek

        # Lag values from buffer (most recent first)
        n_buf = len(hist_vals) + len(predictions)
        all_vals = np.concatenate([hist_vals, np.array(predictions)])

        def lag(k):
            idx = len(all_vals) - k
            return float(all_vals[idx]) if idx >= 0 else float(all_vals[0])

        roll_mean = lambda w: np.mean(all_vals[max(0, len(all_vals)-w):])
        roll_std  = lambda w: np.std(all_vals[max(0, len(all_vals)-w):]) if len(all_vals) >= 2 else 0.0

        features = [
            lag(1), lag(2), lag(3), lag(4),
            lag(8), lag(12), lag(24),
            lag(48), lag(96), lag(192), lag(672),
            roll_mean(16), roll_mean(96), roll_std(96), roll_mean(672),
            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
            np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7),
            np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12),
            float(dw >= 5),
            float((mo, current_ts.day) in {(1,1),(5,1),(10,3),(12,25),(12,26)}),
            float((mo % 12) // 3),
        ]

        X = np.array(features).reshape(1, -1)
        pred = float(model.predict(X)[0])

        if noise_std > 0:
            pred += rng.normal(0, noise_std * 1000)   # noise_std in EUR/kWh → EUR/MWh

        predictions.append(max(-500, pred))
        current_ts = current_ts + freq

    return np.array(predictions)


# ═══════════════════════════════════════════════════════════════════════════════
#  Save / load model
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(model: Pipeline, metrics: dict,
               model_path: str = "data/price_model.pkl",
               metrics_path: str = "data/model_metrics.json"):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Model saved → {model_path}")
    print(f"  Metrics saved → {metrics_path}")


def load_model(model_path: str = "data/price_model.pkl") -> Pipeline:
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train electricity price forecaster")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: fewer trees (for testing)")
    parser.add_argument("--data", default="data/smard_prices_processed.csv",
                        help="Path to processed CSV from fetch_smard_data.py")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Electricity Price Forecaster  —  Gradient Boosted Trees")
    print("  Features: 24 lag/calendar features | Target: next 15-min slot")
    print("  Protocol: Walk-forward validation (Lago et al. 2021)")
    print("="*65)

    # Load data
    print("\n  Loading data...")
    df = load_data(args.data)

    # Split
    print("\n  Splitting data (walk-forward)...")
    df_train, df_val, df_test = split_data(df)

    # Train
    print("\n  Training model...")
    model = train(df_train, df_val, fast=args.fast)

    # Evaluate on held-out test set
    print("\n  Evaluating on test set...")
    metrics = evaluate(model, df_test)

    # Save
    print("\n  Saving model and metrics...")
    save_model(model, metrics)

    # Diagnostic plot
    print("\n  Generating diagnostic charts...")
    plot_diagnostics(model, df_test, metrics)

    # Summary
    print("\n" + "="*65)
    print("  TRAINING COMPLETE")
    print(f"  Test MAE:  {metrics['mae_EUR_MWh']:.2f} EUR/MWh")
    print(f"  Test MAPE: {metrics['mape_pct']:.1f}%")
    print(f"  Test R²:   {metrics['r2']:.3f}")
    print("="*65)
    print("\n  Next step: run  python run_year_simulation.py\n")


if __name__ == "__main__":
    main()
