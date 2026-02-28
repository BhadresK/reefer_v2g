#!/usr/bin/env python3
"""
run_year_simulation.py
═══════════════════════════════════════════════════════════════════════════════
Full-year (52-week) V2G optimisation simulation for the S.KOe COOL reefer.

Modes:
  --year 2024           Simulate all of 2024 using real SMARD prices
  --year 2026           Simulate 2026 using ML-forecasted prices
  --forecast-only       Use ML model for all price inputs (no real data needed)

For each of the 365 days:
  1. Load that day's real/forecast price (96 × 15-min slots)
  2. Run Scenarios A (dumb) and C (MILP V2G) — the most important comparison
     (plus D-MPC with forecast noise for the ML scenario)
  3. Record daily cost, revenue, SoC profile, V2G export

Output:
  data/year_simulation_results.csv  — daily results table
  results_year_summary.png          — 4-panel publication chart
  results_weekly_heatmap.png        — 52-week P&L heatmap

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import argparse
import subprocess, sys
import subprocess, sys

def _ensure(package: str, import_name: str | None = None):
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"  Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"],
                              stdout=subprocess.DEVNULL)
        print(f"  {package} installed OK.")

_ensure("requests")
_ensure("numpy")
_ensure("pandas")
_ensure("openpyxl")
_ensure("scikit-learn", "sklearn")
_ensure("matplotlib")

import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# ── Import the core optimisation engine from existing script ──────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

# We import only the computational functions, not the plotting
from run_optimisation import (
    V2GParams, V2GResult,
    build_load_and_availability,
    run_dumb, run_milp_day_ahead, run_mpc_day_ahead,
    add_realtime_noise,
)

# BNetzA regulated fixed costs (EUR/kWh) — must match fetch_smard_data.py
FIXED_NET = (0.0663 + 0.01992 + 0.00816 + 0.00277 + 0.0205 + 0.01558)
VAT       = 0.19

# Try to import forecaster
try:
    from train_forecaster import load_model, make_24h_forecast, FEATURE_COLS
    HAS_FORECASTER = True
except ImportError:
    HAS_FORECASTER = False

Path("data").mkdir(exist_ok=True)

# BNetzA fixed costs (must match fetch_smard_data.py)
FCR_PEAK_EUR_KWH    = 0.132
FCR_MORNING_EUR_KWH = 0.040


# ═══════════════════════════════════════════════════════════════════════════════
#  Price loading strategies
# ═══════════════════════════════════════════════════════════════════════════════

class PriceLoader:
    """
    Unified price loader supporting three modes:
      1. Real SMARD data (processed CSV)
      2. ML-forecast prices (model predicts day-ahead)
      3. Synthetic fallback (seasonal pattern from make_data.py)
    """

    def __init__(self, mode: str = "auto",
                 smard_csv: str = "data/smard_prices_processed.csv",
                 model_path: str = "data/price_model.pkl"):
        self.mode = mode
        self.df_real = None
        self.model   = None

        # Always load real data if available — needed as lag history even in forecast mode
        if Path(smard_csv).exists():
            print(f"  Loading real SMARD prices from {smard_csv}...")
            self.df_real = pd.read_csv(smard_csv, index_col=0)
            self.df_real.index = pd.to_datetime(self.df_real.index, utc=True).tz_convert("Europe/Berlin")
            self.df_real.index.name = "datetime_cet"
            print(f"    {len(self.df_real):,} rows, "
                  f"{self.df_real.index.min().date()} → {self.df_real.index.max().date()}")
            if mode in ("auto", "real"):
                self.mode = "real"

        if (mode in ("auto", "forecast") and
                Path(model_path).exists() and HAS_FORECASTER):
            print(f"  Loading price forecasting model from {model_path}...")
            self.model = load_model(model_path)
            if self.df_real is None:
                self.mode = "forecast_only"
            else:
                self.mode = "forecast"   # use forecast for future dates, real for past
            print(f"    Model loaded. Mode: {self.mode}")

        if self.df_real is None and self.model is None:
            print("  WARNING: No real data or model found — using synthetic prices")
            self.mode = "synthetic"

    def get_day_prices(self, date: pd.Timestamp,
                       rng: np.random.Generator,
                       noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Returns (buy_EUR_kWh, v2g_EUR_kWh) arrays of shape (96,) for the given date.

        Routing logic:
          - Date exists in real SMARD data → use real prices (most accurate)
          - Date is in future / missing + model available → ML forecast using
            real historical lags as context (realistic future simulation)
          - No data, no model → synthetic seasonal fallback
        """
        # Try real data first regardless of mode
        if self.df_real is not None:
            result = self._get_real_day(date, rng, noise_std)
            if result is not None and self.mode != "forecast":
                return result
            # Date not in real data (future) or forecast mode → use ML model
            if self.model is not None:
                return self._get_forecast_day(date, rng, noise_std)
            if result is not None:
                return result   # real data exists, no model

        if self.model is not None:
            return self._get_forecast_day(date, rng, noise_std)

        return self._get_synthetic_day(date)

    def _get_real_day(self, date: pd.Timestamp,
                      rng, noise_std) -> tuple | None:
        """Slice 96 slots for a given date from real SMARD data."""
        date_str = date.strftime("%Y-%m-%d")
        mask = self.df_real.index.date == date.date()
        subset = self.df_real[mask]

        if len(subset) < 90:   # allow a few missing slots
            return None

        # Resample to exactly 96 slots
        if len(subset) != 96:
            # Some days have 92 or 100 slots (DST transitions)
            idx_96 = pd.date_range(date, periods=96, freq="15min",
                                   tz=subset.index.tz)
            subset = subset.reindex(idx_96, method="nearest")

        spot = subset["price_EUR_MWh"].values / 1000.0   # EUR/MWh → EUR/kWh

        # All-in buy price (what depot actually pays to grid)
        buy_allin = (spot + FIXED_NET) * (1 + VAT)

        # V2G optimizer signal uses SPOT-based prices so spread is preserved:
        # Fixed costs (network fee, taxes) are paid regardless of V2G activity
        # and thus cancel out of the arbitrage decision (Biedenbach & Strunz 2024).
        # The optimizer sees spot prices + FCR premium to correctly decide when
        # to discharge.  Cost accounting still uses all-in prices.
        h    = np.arange(96) * 0.25
        fcr  = np.where((h >= 16) & (h < 20), FCR_PEAK_EUR_KWH, 0.0)
        afrr = np.where((h >= 7)  & (h < 9),  FCR_MORNING_EUR_KWH, 0.0)

        # buy  = spot price (optimizer signal for charging decision)
        # v2g  = spot + FCR/aFRR (optimizer signal for discharge/V2G decision)
        # cost accounting: buy_allin used in KPI aggregation via _make_result
        buy = np.maximum(0.001, spot)
        v2g = np.maximum(buy, spot + fcr + afrr)

        if noise_std > 0:
            buy = np.maximum(0.001, buy + rng.normal(0, noise_std, 96))
            v2g = np.maximum(buy,   v2g + rng.normal(0, noise_std, 96))

        return buy, v2g, f"REAL SMARD {date_str}"

    def _get_forecast_day(self, date: pd.Timestamp,
                          rng, noise_std) -> tuple:
        """
        Use ML model to forecast 96-slot day-ahead prices.

        For future dates: seeds the lag buffer with the last 700 slots of real
        SMARD data, then chains predictions forward day-by-day so the model
        always has realistic lag features — not zeros.
        """
        # Build history buffer: real data up to the simulation start date
        # plus any previously predicted days stored in self._forecast_cache
        if self.df_real is not None:
            real_history = self.df_real[self.df_real.index < date]
            if len(real_history) < 700:
                return self._get_synthetic_day(date)
        else:
            return self._get_synthetic_day(date)

        # Merge real history with cached forecasts for continuity
        if not hasattr(self, "_forecast_cache"):
            self._forecast_cache = real_history.copy()

        # Use the most recent 700 rows (real + previously forecast)
        combined = pd.concat([real_history.tail(700),
                              self._forecast_cache[self._forecast_cache.index >= real_history.index.max()]
                              ]).drop_duplicates().sort_index().tail(700)

        spot_mwh = make_24h_forecast(
            self.model, combined, date,
            noise_std=noise_std, rng=rng
        )

        # Cache this day's predictions so tomorrow's forecast uses them as lags
        new_idx = pd.date_range(date, periods=96, freq="15min", tz=date.tz)
        new_rows = pd.DataFrame({"price_EUR_MWh": spot_mwh}, index=new_idx)
        # Add minimal feature columns needed (will be recomputed next call)
        self._forecast_cache = pd.concat([self._forecast_cache, new_rows]).tail(800)

        spot = spot_mwh / 1000.0

        # Same spot-based optimizer signal as real data path
        h    = np.arange(96) * 0.25
        fcr  = np.where((h >= 16) & (h < 20), FCR_PEAK_EUR_KWH, 0.0)
        afrr = np.where((h >= 7)  & (h < 9),  FCR_MORNING_EUR_KWH, 0.0)
        buy  = np.maximum(0.001, spot)
        v2g  = np.maximum(buy, spot + fcr + afrr)

        return buy, v2g, f"ML FORECAST {date.strftime('%Y-%m-%d')}"

    def _get_synthetic_day(self, date: pd.Timestamp) -> tuple:
        """Fallback: seasonal synthetic price pattern (from make_data.py)."""
        is_summer = date.month in [5, 6, 7, 8, 9]
        h = np.arange(96) * 0.25
        if is_summer:
            spot = np.select(
                [(h>=0)&(h<5),(h>=5)&(h<7),(h>=7)&(h<9),(h>=9)&(h<11),
                 (h>=11)&(h<15),(h>=15)&(h<17),(h>=17)&(h<20),(h>=20)&(h<22)],
                [0.038,0.055,0.095,0.088,0.018,0.072,0.121,0.085],
                default=0.038)
        else:
            spot = np.select(
                [(h>=0)&(h<5),(h>=5)&(h<7),(h>=7)&(h<9),(h>=9)&(h<12),
                 (h>=12)&(h<14),(h>=14)&(h<16),(h>=16)&(h<19),(h>=19)&(h<21)],
                [0.052,0.071,0.148,0.131,0.108,0.092,0.154,0.118],
                default=0.052)
        # Spot-only optimizer signal (fixed costs cancel in arbitrage)
        buy = np.maximum(0.001, spot)
        fcr = np.where((h>=16)&(h<20), FCR_PEAK_EUR_KWH, 0.0)
        v2g = np.maximum(buy, spot + fcr)
        season = "summer" if is_summer else "winter"
        return buy, v2g, f"SYNTHETIC {season}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Year simulation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_year(year: int,
             v2g: V2GParams,
             loader: PriceLoader,
             fleet_size: int = 1,
             soc_init_pct: float = 45.0,
             soc_final_pct: float = 80.0,
             dwell: str = "Extended",
             verbose: bool = True) -> pd.DataFrame:
    """
    Simulate every day of `year` with three scenarios:
      A: Dumb (uncontrolled)
      C: MILP Day-Ahead (full optimal)
      D: MPC with forecast-based prices (rolling horizon)

    Returns a DataFrame with one row per day.
    """
    tru, plugged = build_load_and_availability(v2g, dwell=dwell)
    rng = np.random.default_rng(2024)

    # Generate date range
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31",
                          freq="D", tz="Europe/Berlin")

    records = []
    n = len(dates)

    print(f"\n  Simulating {n} days of {year} ({fleet_size} trailer(s)...)  ")

    for i, date in enumerate(dates):
        if verbose and i % 30 == 0:
            pct = 100 * i / n
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:.0f}%  {date.strftime('%b %d')}   ", end="\r")

        # Load prices for this day
        buy, v2g_p, source = loader.get_day_prices(date, rng, noise_std=0.0)

        # Scenario A: dumb
        A = run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)

        # Scenario C: full MILP day-ahead
        C = run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)

        # Scenario D: MPC with mild noise (realistic operational scenario)
        D = run_mpc_day_ahead(
            v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct,
            forecast_noise_std=v2g.mpc_price_noise_std,
            label="D-MPC", seed=int(date.timestamp()) % 10000,
        )

        records.append({
            "date":               date.strftime("%Y-%m-%d"),
            "week":               date.isocalendar()[1],
            "month":              date.month,
            "month_name":         date.strftime("%b"),
            "weekday":            date.dayofweek,
            "is_weekend":         int(date.dayofweek >= 5),
            "price_source":       source,
            # Prices
            "buy_mean_EUR_kWh":   float(buy.mean()),
            "buy_max_EUR_kWh":    float(buy.max()),
            "v2g_max_EUR_kWh":    float(v2g_p.max()),
            # Scenario A (baseline)
            "A_cost_EUR":         A.cost_eur_day * fleet_size,
            "A_charge_EUR":       A.charge_cost_eur_day * fleet_size,
            # Scenario C (MILP V2G)
            "C_cost_EUR":         C.cost_eur_day * fleet_size,
            "C_v2g_rev_EUR":      C.v2g_revenue_eur_day * fleet_size,
            "C_charge_EUR":       C.charge_cost_eur_day * fleet_size,
            "C_deg_EUR":          C.deg_cost_eur_day * fleet_size,
            "C_v2g_kwh":          C.v2g_export_kwh_day * fleet_size,
            "C_saving_EUR":       (A.cost_eur_day - C.cost_eur_day) * fleet_size,
            # Scenario D (MPC)
            "D_cost_EUR":         D.cost_eur_day * fleet_size,
            "D_v2g_rev_EUR":      D.v2g_revenue_eur_day * fleet_size,
            "D_v2g_kwh":          D.v2g_export_kwh_day * fleet_size,
            "D_saving_EUR":       (A.cost_eur_day - D.cost_eur_day) * fleet_size,
        })

    print(f"\n  Done. {n} days simulated.                                    ")
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Annual summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def print_annual_summary(df: pd.DataFrame, year: int, fleet_size: int):
    print("\n" + "="*72)
    print(f"  ANNUAL SUMMARY — {year}  |  Fleet: {fleet_size} trailer(s)")
    print("="*72)

    total_A = df["A_cost_EUR"].sum()
    total_C = df["C_cost_EUR"].sum()
    total_D = df["D_cost_EUR"].sum()
    save_C  = df["C_saving_EUR"].sum()
    save_D  = df["D_saving_EUR"].sum()
    v2g_C   = df["C_v2g_kwh"].sum()
    rev_C   = df["C_v2g_rev_EUR"].sum()

    print(f"  {'Scenario':<35} {'Annual Cost':>12} {'Annual Saving':>14} {'vs Dumb %':>10}")
    print("-"*72)
    print(f"  {'A — Dumb (baseline)':<35} EUR {total_A:>9,.0f}")
    print(f"  {'C — MILP Day-Ahead (V2G)':<35} EUR {total_C:>9,.0f} "
          f"  EUR {save_C:>+9,.0f}   {save_C/abs(total_A)*100:>+7.1f}%")
    print(f"  {'D — MPC (noisy forecast)':<35} EUR {total_D:>9,.0f} "
          f"  EUR {save_D:>+9,.0f}   {save_D/abs(total_A)*100:>+7.1f}%")
    print("="*72)
    print(f"  V2G export (Scenario C):    {v2g_C:>8,.0f} kWh/yr  "
          f"({v2g_C/1000:.1f} MWh/yr)")
    print(f"  V2G revenue (Scenario C):   EUR {rev_C:>8,.0f}/yr")
    print(f"  Best day saving:            EUR {df['C_saving_EUR'].max():>8.2f}")
    print(f"  Worst day (cost increase):  EUR {df['C_saving_EUR'].min():>8.2f}")
    print(f"  Days V2G is profitable:     {(df['C_v2g_kwh']>0).sum():>3d} / {len(df)}")

    print(f"\n  Monthly breakdown (Scenario C savings):")
    monthly = df.groupby("month_name").agg(
        saving=("C_saving_EUR","sum"),
        v2g_kwh=("C_v2g_kwh","sum"),
        days=("date","count")
    )
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in month_order:
        if m in monthly.index:
            r = monthly.loc[m]
            print(f"    {m}:  saving EUR {r['saving']:>7,.0f}  "
                  f"V2G {r['v2g_kwh']:>6,.0f} kWh  ({int(r['days'])} days)")
    print("="*72)


# ═══════════════════════════════════════════════════════════════════════════════
#  Publication-quality charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_year_summary(df: pd.DataFrame, year: int, fleet_size: int,
                      out: str = "results_year_summary.png"):
    """4-panel year overview chart."""
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

    # Colour palette
    CA = "#d62728"   # dumb = red
    CC = "#1f77b4"   # MILP = blue
    CD = "#2ca02c"   # MPC  = green
    CG = "#ff7f0e"   # price = orange

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── (1) Weekly cumulative savings vs baseline ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    weekly = df.groupby("week").agg(
        C_saving=("C_saving_EUR","sum"),
        D_saving=("D_saving_EUR","sum"),
        v2g_kwh=("C_v2g_kwh","sum"),
    ).reset_index()
    x = weekly["week"].values
    ax1.fill_between(x, weekly["C_saving"], alpha=0.35, color=CC, label="MILP C savings")
    ax1.fill_between(x, weekly["D_saving"], alpha=0.25, color=CD, label="MPC D savings")
    ax1.plot(x, weekly["C_saving"], lw=1.5, color=CC)
    ax1.plot(x, weekly["D_saving"], lw=1.5, color=CD, ls="--")
    ax1.axhline(0, color="black", lw=0.8, ls=":")
    ax1.set_title(f"(1) Weekly Cost Saving vs Dumb ({fleet_size} trailer)", fontsize=11)
    ax1.set_xlabel("ISO Week"); ax1.set_ylabel("EUR / week")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 52)

    # ── (2) Monthly V2G revenue breakdown ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    monthly = df.groupby("month").agg(
        v2g_rev=("C_v2g_rev_EUR","sum"),
        charge=("C_charge_EUR","sum"),
        deg=("C_deg_EUR","sum"),
        saving=("C_saving_EUR","sum"),
    ).reindex(range(1,13), fill_value=0)

    x_mo = np.arange(12)
    w    = 0.28
    ax2.bar(x_mo - w, monthly["v2g_rev"],   width=w, color=CC, alpha=0.85, label="V2G Revenue")
    ax2.bar(x_mo,     monthly["saving"],    width=w, color=CD, alpha=0.85, label="Net Saving")
    ax2.bar(x_mo + w, monthly["deg"],       width=w, color="grey", alpha=0.6, label="Degrad. Cost")
    ax2.set_xticks(x_mo); ax2.set_xticklabels(month_labels, fontsize=9)
    ax2.set_title("(2) Monthly Revenue Components — MILP (Scenario C)", fontsize=11)
    ax2.set_ylabel("EUR / month"); ax2.legend(fontsize=9)
    ax2.axhline(0, color="black", lw=0.8); ax2.grid(True, alpha=0.3, axis="y")

    # ── (3) Cumulative P&L over the year ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    df_sorted  = df.sort_values("date")
    cumA  = df_sorted["A_cost_EUR"].cumsum()
    cumC  = df_sorted["C_cost_EUR"].cumsum()
    cumD  = df_sorted["D_cost_EUR"].cumsum()
    dates = df_sorted["date"]

    ax3.plot(dates, cumA, lw=2, color=CA, label="A — Dumb")
    ax3.plot(dates, cumC, lw=2, color=CC, label="C — MILP V2G")
    ax3.plot(dates, cumD, lw=2, color=CD, label="D — MPC", ls="--")
    # Shade saving area
    ax3.fill_between(dates, cumA, cumC, alpha=0.15, color=CC,
                     label=f"Saving (C): EUR {df['C_saving_EUR'].sum():,.0f}/yr")
    ax3.set_title("(3) Cumulative Annual Cost — All Scenarios", fontsize=11)
    ax3.set_ylabel("Cumulative EUR"); ax3.set_xlabel("")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
    # Add final value annotations
    for val, col, label in [
        (cumA.iloc[-1], CA, "A"),
        (cumC.iloc[-1], CC, "C"),
        (cumD.iloc[-1], CD, "D"),
    ]:
        ax3.annotate(f"EUR {val:,.0f}",
                     xy=(dates.iloc[-1], val),
                     xytext=(10, 0), textcoords="offset points",
                     color=col, fontsize=8, fontweight="bold",
                     va="center")
    ax3.tick_params(axis="x", rotation=30)

    # ── (4) Price volatility vs V2G revenue scatter ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(
        df["buy_max_EUR_kWh"],
        df["C_saving_EUR"],
        c=df["month"],
        cmap="RdYlGn_r",
        alpha=0.6, s=18,
        vmin=1, vmax=12
    )
    ax4.axhline(0, color="black", lw=0.8, ls=":")
    cb = plt.colorbar(scatter, ax=ax4, pad=0.01)
    cb.set_ticks([1,3,6,9,12])
    cb.set_ticklabels(["Jan","Mar","Jun","Sep","Dec"])
    ax4.set_title("(4) Daily Peak Price vs MILP Cost Saving", fontsize=11)
    ax4.set_xlabel("Daily peak buy price (EUR/kWh)")
    ax4.set_ylabel("Daily MILP saving vs Dumb (EUR)")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"S.KOe COOL  Reefer V2G — Full Year {year} Simulation  "
        f"({fleet_size} trailer{'s' if fleet_size>1 else ''})",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_weekly_heatmap(df: pd.DataFrame, year: int,
                        out: str = "results_weekly_heatmap.png"):
    """52-week × 7-day heatmap of daily cost savings."""
    pivot = df.pivot_table(
        index="weekday", columns="week",
        values="C_saving_EUR", aggfunc="sum"
    )

    fig, axes = plt.subplots(2, 1, figsize=(18, 8),
                              gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35})

    # Heatmap
    ax = axes[0]
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax*0.3, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(pivot.columns, pivot.index,
                       pivot.values, cmap="RdYlGn", norm=norm)
    plt.colorbar(im, ax=ax, label="Daily saving vs Dumb (EUR)", pad=0.01)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_title(f"(A) Daily V2G Cost Saving vs Dumb — {year}  [Scenario C: MILP Day-Ahead]",
                 fontsize=11)
    ax.set_xlabel("ISO Week"); ax.set_ylabel("")

    # Weekly bar
    ax2 = axes[1]
    weekly_save = df.groupby("week")["C_saving_EUR"].sum()
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in weekly_save.values]
    ax2.bar(weekly_save.index, weekly_save.values, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("(B) Weekly Total Saving (EUR)", fontsize=11)
    ax2.set_xlabel("ISO Week"); ax2.set_ylabel("EUR")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlim(weekly_save.index.min()-0.5, weekly_save.index.max()+0.5)

    fig.suptitle(f"S.KOe COOL V2G — 52-Week Performance Heatmap {year}",
                 fontsize=13, fontweight="bold")

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full-year V2G simulation with real or ML-forecast prices"
    )
    parser.add_argument("--year",   type=int, default=2024,
                        help="Year to simulate (default: 2024)")
    parser.add_argument("--fleet",  type=int, default=1,
                        help="Fleet size (trailers, default: 1)")
    parser.add_argument("--mode",   default="auto",
                        choices=["auto","real","forecast","synthetic"],
                        help="Price source mode (default: auto)")
    parser.add_argument("--save-csv", default="data/year_simulation_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print("\n" + "="*65)
    print(f"  S.KOe COOL  —  Full Year {args.year} Simulation")
    print(f"  Fleet: {args.fleet} trailer(s)")
    print("  Scenarios: A (Dumb), C (MILP V2G), D (MPC + noise)")
    print("="*65)

    # Load battery parameters
    v2g = V2GParams(mpc_price_noise_std=0.012)

    # Try to load from Excel if available
    from run_optimisation import load_battery_params
    v2g, batt_source = load_battery_params(v2g)
    print(f"\n  Battery: {batt_source}")

    # Initialise price loader
    print("\n  Initialising price loader...")
    loader = PriceLoader(mode=args.mode)

    # Run simulation
    df = run_year(
        year       = args.year,
        v2g        = v2g,
        loader     = loader,
        fleet_size = args.fleet,
        soc_init_pct  = 45.0,
        soc_final_pct = 80.0,
        dwell      = "Extended",
    )

    # Save raw results
    df.to_csv(args.save_csv, index=False)
    print(f"  Results saved → {args.save_csv}")

    # Print summary
    print_annual_summary(df, args.year, args.fleet)

    # Charts
    print("\n  Generating charts...")
    plot_year_summary(df, args.year, args.fleet)
    plot_weekly_heatmap(df, args.year)

    print(f"\n  All done.\n"
          f"  → data/year_simulation_results.csv\n"
          f"  → results_year_summary.png\n"
          f"  → results_weekly_heatmap.png\n")


if __name__ == "__main__":
    main()
