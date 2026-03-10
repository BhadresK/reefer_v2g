#!/usr/bin/env python3
"""
S.KOe COOL — Day-Ahead MILP + Receding-Horizon MPC V2G Optimisation
Schmitz Cargobull AG | 2025
Based on: Biedenbach & Strunz (2024), Agora Verkehrswende (2025)

Electricity prices: SMARD.de 2025 real DE/LU 15-min day-ahead spot data
"""

from __future__ import annotations
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 – ABBREVIATION LEGEND PNG
#  Generates a standalone reference card so readers can decode every symbol.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_abbreviation_legend(out="abbreviation_legend.png"):
    """
    Produce a clean reference PNG listing every abbreviation, variable,
    and scenario code used in the simulation outputs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle(
        "V2G Optimisation — Abbreviation & Symbol Reference Card",
        fontsize=15, fontweight="bold", color="#1A237E", y=0.97
    )

    # ── Column 1: Variables & Physical Symbols ─────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#EEF2FF")
    ax.axis("off")
    ax.text(0.5, 0.97, "VARIABLES & SYMBOLS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1A237E",
            transform=ax.transAxes)

    variables = [
        ("P_c",          "Charging power (kW) — power drawn from grid into battery"),
        ("P_d",          "Discharging power (kW) — power fed from battery to grid (V2G)"),
        ("E / SoC",      "State of Charge (kWh) — energy currently stored in battery"),
        ("E_min",        "Minimum usable energy (kWh) — cold-chain floor (SoC 20%)"),
        ("E_max",        "Maximum usable energy (kWh) — cycle ceiling (SoC 95%)"),
        ("η_c",          "Charge efficiency (dimensionless, e.g. 0.92)"),
        ("η_d",          "Discharge efficiency (dimensionless, e.g. 0.92)"),
        ("Δt / dt",      "Time step duration (hours, 0.25 = 15 min)"),
        ("T / N",        "Total number of time slots (96 slots = 24 h)"),
        ("t",            "Time slot index (0 … 95)"),
        ("h",            "Hour of day (0.0 … 23.75)"),
        ("buy[t]",       "Buy/import price at slot t (€/kWh) — what you pay to charge"),
        ("v2g[t]",       "V2G sell price at slot t (€/kWh) — revenue when discharging"),
        ("deg",          "Degradation cost (€/kWh cycled) — battery wear per kWh"),
        ("TRU",          "Transport Refrigeration Unit auxiliary load (kW)"),
        ("plugged[t]",   "Availability flag: 1 = truck at depot & plugged in, 0 = on road"),
        ("p_c_max",      "Maximum allowed charging power (kW) — hardware limit"),
        ("p_d_max",      "Maximum allowed V2G discharge power (kW) — hardware limit"),
        ("W",            "MILP / MPC optimisation horizon window length (slots)"),
        ("E_init",       "Battery energy at start of optimisation window (kWh)"),
        ("E_fin",        "Minimum required energy at end of window — departure SoC (kWh)"),
    ]

    y = 0.91
    for abbr, desc in variables:
        ax.text(0.03, y, f"  {abbr}", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#C62828", transform=ax.transAxes)
        words = desc.split()
        line, lines = "", []
        for w in words:
            if len(line + w) > 48:
                lines.append(line.strip()); line = w + " "
            else:
                line += w + " "
        lines.append(line.strip())
        ax.text(0.28, y, lines[0], ha="left", va="top", fontsize=7.8,
                color="#212121", transform=ax.transAxes)
        for i, l in enumerate(lines[1:], 1):
            ax.text(0.28, y - i*0.018, l, ha="left", va="top", fontsize=7.8,
                    color="#212121", transform=ax.transAxes)
        y -= 0.044 + max(0, (len(lines)-1)*0.018)

    # ── Column 2: Scenarios & Cost Terms ──────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#E8F5E9")
    ax.axis("off")
    ax.text(0.5, 0.97, "SCENARIOS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1B5E20",
            transform=ax.transAxes)

    scenarios = [
        ("A – Dumb",         "#AAAAAA",
         "Uncontrolled charging. Charges at full power (P_c_max)\n"
         "the moment truck arrives. No price awareness. No V2G.\n"
         "Baseline / worst case."),
        ("B – Smart\n(no V2G)", "#2196F3",
         "Price-optimal charging only. Day-ahead MILP shifts\n"
         "charging to cheapest slots but never discharges.\n"
         "Minimal battery wear."),
        ("C – MILP\nDay-Ahead", "#00BCD4",
         "Full 24h MILP solved once at 00:00 using complete\n"
         "day-ahead price forecast. Charges cheap, discharges\n"
         "at peak. Theoretical optimum (perfect information)."),
        ("D – MPC\nPerfect", "#FF7700",
         "Receding-Horizon Model Predictive Control using full\n"
         "remaining day as horizon. Re-solves MILP at every\n"
         "15-min slot. No forecast noise. Near-optimal."),
    ]

    y = 0.90
    for sc_label, col, desc in scenarios:
        patch = mpatches.FancyBboxPatch((0.02, y-0.015), 0.06, 0.035,
                                        boxstyle="round,pad=0.005",
                                        facecolor=col, edgecolor="white",
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(patch)
        ax.text(0.11, y+0.008, sc_label, ha="left", va="top", fontsize=9,
                fontweight="bold", color="#1B5E20", transform=ax.transAxes)
        for i, line in enumerate(desc.split("\n")):
            ax.text(0.11, y - 0.014 - i*0.020, line, ha="left", va="top",
                    fontsize=7.5, color="#333333", transform=ax.transAxes)
        y -= 0.16

    ax.text(0.5, 0.37, "COST / REVENUE TERMS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1B5E20",
            transform=ax.transAxes)

    cost_terms = [
        ("Net Cost (€/day)",        "= Charge cost − V2G revenue + Degradation cost"),
        ("Charge Cost (€/day)",     "= Σ_t  buy[t] · P_c[t] · dt"),
        ("V2G Revenue (€/day)",     "= Σ_t  v2g[t] · P_d[t] · dt"),
        ("Degradation Cost (€/day)","= Σ_t  deg · (P_c[t]+P_d[t]) · dt"),
        ("Savings vs A (€/day)",    "= Net Cost(A) − Net Cost(scenario)"),
        ("Annual Savings (€/yr)",   "= Savings/day × 365"),
    ]
    y = 0.31
    for term, formula in cost_terms:
        ax.text(0.03, y, f"• {term}", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#C62828", transform=ax.transAxes)
        ax.text(0.03, y-0.022, f"    {formula}", ha="left", va="top",
                fontsize=7.8, color="#333333", transform=ax.transAxes)
        y -= 0.055

    # ── Column 3: Seasons, Dwell Modes, Solver & Hardware ─────────────────
    ax = axes[2]
    ax.set_facecolor("#FFF8E1")
    ax.axis("off")
    ax.text(0.5, 0.97, "SEASONS & DWELL MODES", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#E65100",
            transform=ax.transAxes)

    seasons = [
        ("Winter WD",  "Oct–Mar, Mon–Fri. ~130 days/year"),
        ("Summer WD",  "Apr–Sep, Mon–Fri. ~131 days/year"),
        ("Winter WE",  "Oct–Mar, Sat–Sun. ~52 days/year"),
        ("Summer WE",  "Apr–Sep, Sat–Sun. ~52 days/year"),
        ("Extended",   "Plugged: 21:00–07:00 + 12:00–18:00 = 16h/day"),
        ("NightOnly",  "Plugged: 21:00–07:00 only = 10h/day"),
        ("Weekend",    "Plugged: 00:00–24:00 = fully available 24h/day"),
    ]
    y = 0.90
    for term, desc in seasons:
        ax.text(0.03, y, f"  {term}:", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#BF360C", transform=ax.transAxes)
        ax.text(0.30, y, desc, ha="left", va="top", fontsize=8,
                color="#333333", transform=ax.transAxes)
        y -= 0.048

    ax.text(0.5, y-0.01, "HARDWARE (S.KOe COOL)", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#E65100",
            transform=ax.transAxes)
    y -= 0.065

    hardware = [
        ("Battery capacity",  "70 kWh total / 60 kWh usable (SoC 20–95%)"),
        ("Max charge",        "22 kW (ISO 15118 AC)"),
        ("Max V2G discharge", "22 kW (ISO 15118-2 V2G)"),
        ("Charge η",          "0.92 (92% round-trip efficiency one way)"),
        ("Discharge η",       "0.92"),
        ("TRU load",          "~2.8–4.0 kW (sinusoidal model)"),
        ("Arrival SoC",       "45% (winter average)"),
        ("Departure SoC",     "80% (cold-chain requirement)"),
        ("deg cost default",  "0.07 €/kWh cycled (LFP cell ageing estimate)"),
    ]
    for term, desc in hardware:
        ax.text(0.03, y, f"  {term}:", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#BF360C", transform=ax.transAxes)
        ax.text(0.40, y, desc, ha="left", va="top", fontsize=8,
                color="#333333", transform=ax.transAxes)
        y -= 0.048

    ax.text(0.5, y-0.01, "SOLVER & DATA", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#E65100",
            transform=ax.transAxes)
    y -= 0.065

    solver_info = [
        ("MILP solver",    "scipy HiGHS (via scipy.optimize.milp)"),
        ("Objective",      "Minimise: Σ(buy·P_c − v2g·P_d + deg·(P_c+P_d))·dt"),
        ("Constraints",    "SoC dynamics, power bounds, mutex P_c/P_d, E_fin"),
        ("MPC principle",  "Receding horizon: solve full remaining day, apply first action"),
        ("Price data",     "SMARD.de 2025 DE/LU 15-min day-ahead spot (€/MWh)"),
        ("Buy price",      "Spot price (direct, no grid surcharge in model)"),
        ("V2G sell price", "Spot price (wholesale sell, no levy deducted)"),
        ("Seasons source", "Calendar-averaged representative weekday/weekend profiles"),
    ]
    for term, desc in solver_info:
        ax.text(0.03, y, f"  {term}:", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#BF360C", transform=ax.transAxes)
        words = desc.split()
        line, lines = "", []
        for w in words:
            if len(line + w) > 42:
                lines.append(line.strip()); line = w + " "
            else:
                line += w + " "
        lines.append(line.strip())
        ax.text(0.40, y, lines[0], ha="left", va="top", fontsize=8,
                color="#333333", transform=ax.transAxes)
        for i, l in enumerate(lines[1:], 1):
            ax.text(0.40, y - i*0.018, l, ha="left", va="top", fontsize=8,
                    color="#333333", transform=ax.transAxes)
        y -= 0.048 + max(0, (len(lines)-1)*0.018)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Abbreviation legend saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class V2GParams:
    battery_capacity_kWh:  float = 70.0
    usable_capacity_kWh:   float = 60.0     # usable window (SoC 20-95 %)
    soc_min_pct:           float = 20.0     # cold-chain floor
    soc_max_pct:           float = 95.0     # cycle ceiling
    charge_power_kW:       float = 22.0     # max AC charge (ISO 15118)
    discharge_power_kW:    float = 22.0     # max V2G discharge
    eta_charge:            float = 0.92
    eta_discharge:         float = 0.92
    deg_cost_eur_kwh:      float = 0.07     # battery wear €/kWh cycled
    dt_h:                  float = 0.25     # 15-min slots
    n_slots:               int   = 96       # 24 h x 4
    depot_connection_kVA:  float = 0.0      # Field 30 – 0 = no limit
    transformer_limit_kVA: float = 0.0      # Field 31 – 0 = no limit
    # mpc_price_noise_std:   float = 0.012  # [REMOVED] Gaussian noise disabled

    @property
    def E_min(self) -> float:
        return self.usable_capacity_kWh * self.soc_min_pct / 100.0

    @property
    def E_max(self) -> float:
        return self.usable_capacity_kWh * self.soc_max_pct / 100.0

    @property
    def _grid_kw_cap(self) -> float:
        PF = 0.95
        limits = [v for v in [self.depot_connection_kVA,
                               self.transformer_limit_kVA] if v > 0]
        return min(limits) * PF if limits else float("inf")

    @property
    def p_c_max(self) -> float:
        return min(self.charge_power_kW, self._grid_kw_cap)

    @property
    def p_d_max(self) -> float:
        return min(self.discharge_power_kW, self._grid_kw_cap)


@dataclass
class V2GResult:
    """Output from one optimisation run."""
    scenario:            str
    p_charge:            np.ndarray
    p_discharge:         np.ndarray
    soc:                 np.ndarray
    cost_eur_day:        float
    v2g_revenue_eur_day: float
    v2g_export_kwh_day:  float
    charge_cost_eur_day: float
    deg_cost_eur_day:    float
    price_buy:           np.ndarray
    price_v2g:           np.ndarray
    plugged:             np.ndarray
    tru_load:            np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – PRICE DATA  (real 2025 SMARD CSV)
# ═══════════════════════════════════════════════════════════════════════════════

_PRICE_CACHE: dict = {}


def _load_smard_csv(csv_path: str) -> pd.DataFrame:
    """Parse SMARD CSV → tidy DataFrame with datetime index and price_eur_kwh."""
    global _PRICE_CACHE
    if "df" in _PRICE_CACHE:
        return _PRICE_CACHE["df"]

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    col = "Germany/Luxembourg [€/MWh] Original resolutions"
    df = df[["Start date", col]].copy()
    df.columns = ["datetime_str", "price_eur_mwh"]

    df["datetime"] = pd.to_datetime(df["datetime_str"],
                                    format="%b %d, %Y %I:%M %p",
                                    errors="coerce")
    df = df.dropna(subset=["datetime", "price_eur_mwh"])
    df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0
    df = df.set_index("datetime").sort_index()
    df["slot"] = (df.index.hour * 4 + df.index.minute // 15)
    df["is_weekend"] = df.index.dayofweek >= 5
    df["month"] = df.index.month
    df["is_winter"] = df["month"].isin([1, 2, 3, 10, 11, 12])

    _PRICE_CACHE["df"] = df
    print(f"  Loaded {len(df):,} price slots from SMARD CSV "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def load_prices_from_csv(csv_path: str, v2g: "V2GParams",
                         season: str = "winter") -> tuple:
    """
    Build a representative 96-slot price profile for the requested season.

    season codes:
        'winter'         → Oct–Mar, weekdays (Mon–Fri)
        'summer'         → Apr–Sep, weekdays
        'winter_weekend' → Oct–Mar, weekends (Sat–Sun)
        'summer_weekend' → Apr–Sep, weekends

    Returns (buy_96, v2g_96, source_label) where each array has shape (96,).
    """
    df = _load_smard_csv(csv_path)

    is_wknd   = season.endswith("_weekend")
    is_winter = season.startswith("winter")

    mask = (df["is_winter"] == is_winter) & (df["is_weekend"] == is_wknd)
    sub  = df[mask]

    if len(sub) == 0:
        raise ValueError(f"No price data found for season='{season}'")

    profile = sub.groupby("slot")["price_eur_kwh"].mean().values
    assert len(profile) == 96, f"Expected 96 slots, got {len(profile)}"

    buy   = profile.copy()
    v2g_p = profile.copy()

    n_days    = int(len(sub) / 96)
    season_lbl = {"winter": "Oct–Mar WD", "summer": "Apr–Sep WD",
                  "winter_weekend": "Oct–Mar WE", "summer_weekend": "Apr–Sep WE"}
    source = (f"REAL — SMARD.de 2025 DE/LU spot  |  {season_lbl.get(season, season)}"
              f"  |  avg of {n_days} days"
              f"  |  range {profile.min()*1000:.1f}–{profile.max()*1000:.1f} €/MWh")
    return buy, v2g_p, source


def load_deg_sensitivity(v2g: "V2GParams") -> np.ndarray:
    return np.linspace(0.02, 0.15, 10)


def build_load_and_availability(v2g: "V2GParams", dwell: str = "Extended") -> tuple:
    """
    TRU auxiliary load and plug-in availability window.
    Extended  = night (21-07) + midday depot stop (12-18).  16h plugged/day.
    NightOnly = night (21-07) only.  10h plugged/day.
    Weekend   = truck at depot all day. 24h plugged.
    """
    N = v2g.n_slots
    h = np.arange(N) * v2g.dt_h
    tru = 2.8 + 1.2 * np.sin(2 * np.pi * np.arange(N) / N + np.pi)
    if dwell == "Weekend":
        plugged = np.ones(N)
    elif dwell == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    else:
        plugged = ((h >= 21) | (h < 7) | ((h >= 12) & (h < 18))).astype(float)
    return tru, plugged


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – MILP INNER SOLVER  (scipy HiGHS)
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_milp_window(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg):
    """
    Solve MILP over one window of length W = len(buy).

    OBJECTIVE (Biedenbach & Strunz 2024, eq. 2-6):
        min  Σ_t [ buy[t]·P_c[t] − v2g[t]·P_d[t] + deg·(P_c[t]+P_d[t]) ] · dt

    CONSTRAINTS:
        (i)   SoC dynamics:
                e[t] = e[t-1] + η_c·P_c[t]·dt − (1/η_d)·P_d[t]·dt − TRU[t]·dt
        (ii)  Power bounds: 0 ≤ P_c ≤ p_c_max·plugged,  0 ≤ P_d ≤ p_d_max·plugged
        (iii) SoC bounds:   E_min ≤ e[t] ≤ E_max
        (iv)  Linearised mutex: P_c[t] + P_d[t] ≤ max_power
        (v)   Departure SoC: e[W-1] ≥ E_fin
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix, csc_matrix
    except ImportError:
        return np.zeros(len(buy)), np.zeros(len(buy)), np.full(len(buy), E_init), False

    W  = len(buy)
    dt = v2g.dt_h
    idx_c = np.arange(W)
    idx_d = np.arange(W, 2*W)
    idx_e = np.arange(2*W, 3*W)
    nv    = 3*W

    c_vec        = np.zeros(nv)
    c_vec[idx_c] =  buy   * dt + deg * dt
    c_vec[idx_d] = -v2g_p * dt + deg * dt

    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    ub[idx_c] = v2g.p_c_max * plugged
    ub[idx_d] = v2g.p_d_max * plugged
    lb[idx_e] = v2g.E_min;  ub[idx_e] = v2g.E_max

    n_rows = W + W + 1
    A  = lil_matrix((n_rows, nv))
    lo = np.zeros(n_rows);  hi = np.zeros(n_rows)

    for t in range(W):
        A[t, idx_e[t]] =  1.0
        A[t, idx_c[t]] = -v2g.eta_charge * dt
        A[t, idx_d[t]] =  (1.0 / v2g.eta_discharge) * dt
        rhs = -tru[t] * dt
        if t == 0:
            rhs += E_init
        else:
            A[t, idx_e[t-1]] = -1.0
        lo[t] = hi[t] = rhs

    max_p = max(v2g.p_c_max, v2g.p_d_max)
    for t in range(W):
        row = W + t
        A[row, idx_c[t]] = 1.0; A[row, idx_d[t]] = 1.0
        lo[row] = -np.inf;      hi[row] =  max_p

    A[2*W, idx_e[W-1]] = 1.0
    lo[2*W] = E_fin;  hi[2*W] = v2g.E_max

    res = milp(c_vec,
               constraints=LinearConstraint(csc_matrix(A), lo, hi),
               bounds=Bounds(lb, ub),
               options={"disp": False, "time_limit": 60})

    if res.success:
        return (np.clip(res.x[idx_c], 0, None),
                np.clip(res.x[idx_d], 0, None),
                res.x[idx_e], True)
    return np.zeros(W), np.zeros(W), np.full(W, E_init), False


def _greedy_fallback(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg):
    """Rule-based fallback when MILP solver fails."""
    N, dt = len(buy), v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N)
    soc  = E_init
    for t in range(N):
        soc -= tru[t] * dt; soc = max(v2g.E_min, soc)
        if plugged[t]:
            margin = (v2g_p[t] - deg) - (buy[t] + deg)
            if margin > 0.02 and soc > v2g.E_min + v2g.p_d_max * dt / v2g.eta_discharge:
                p = min(v2g.p_d_max, (soc - v2g.E_min) * v2g.eta_discharge / dt)
                P_d[t] = p; soc = max(v2g.E_min, soc - p / v2g.eta_discharge * dt)
            elif buy[t] < 0.22 and soc < v2g.E_max - v2g.p_c_max * dt * v2g.eta_charge:
                p = min(v2g.p_c_max, (v2g.E_max - soc) / (v2g.eta_charge * dt))
                P_c[t] = p; soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    if soc < E_fin:
        for t in range(N-1, -1, -1):
            if plugged[t] and P_d[t] == 0:
                deficit = E_fin - soc
                extra   = min(v2g.p_c_max * dt * v2g.eta_charge, deficit)
                P_c[t] += extra / (v2g.eta_charge * dt); soc += extra
                if soc >= E_fin: break
    return P_c, P_d, e


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – SCENARIO A: DUMB BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init_pct=45.0, soc_final_pct=80.0):
    """Charge at max power whenever plugged in. No price awareness. No V2G."""
    N, dt  = v2g.n_slots, v2g.dt_h
    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N)
    soc  = E_init
    for t in range(N):
        soc -= tru[t] * dt; soc = max(v2g.E_min, soc)
        if plugged[t] and soc < v2g.E_max:
            p = min(v2g.p_c_max, (v2g.E_max - soc) / (v2g.eta_charge * dt))
            P_c[t] = p; soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return _make_result("A - Dumb (uncontrolled)", v2g, P_c, P_d, e,
                        buy, v2g_p, plugged, tru, deg=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – SCENARIO B: SMART CHARGING ONLY  (no V2G discharge)
# ═══════════════════════════════════════════════════════════════════════════════

def run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged,
                     soc_init_pct=45.0, soc_final_pct=80.0):
    """Day-ahead MILP: price-optimal charging, V2G discharge blocked. deg=0."""
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct / 100.0
    P_c, P_d, e, ok = _solve_milp_window(
        v2g, buy, np.zeros_like(v2g_p), tru, plugged, E_init, E_fin, deg=0.0)
    if not ok:
        P_c, P_d, e = _greedy_fallback(
            v2g, buy, np.zeros_like(v2g_p), tru, plugged, E_init, E_fin, deg=0.0)
    return _make_result("B - Smart (no V2G)", v2g, P_c, P_d, e,
                        buy, v2g_p, plugged, tru, deg=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – SCENARIO C: FULL DAY-AHEAD MILP
# ═══════════════════════════════════════════════════════════════════════════════

def run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged,
                       soc_init_pct=45.0, soc_final_pct=80.0):
    """Full 24h MILP solved once at 00:00. Perfect information, global optimum."""
    deg    = v2g.deg_cost_eur_kwh
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct / 100.0
    P_c, P_d, e, ok = _solve_milp_window(
        v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg)
    if not ok:
        warnings.warn("[MILP] Solver failed - greedy fallback")
        P_c, P_d, e = _greedy_fallback(
            v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg)
    return _make_result("C - MILP Day-Ahead (perfect)", v2g, P_c, P_d, e,
                        buy, v2g_p, plugged, tru, deg)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – SCENARIO D: RECEDING-HORIZON MPC  (no forecast noise)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mpc_day_ahead(v2g, buy_day_ahead, v2g_day_ahead, tru, plugged,
                      soc_init_pct=45.0, soc_final_pct=80.0,
                      label="D - MPC perfect",
                      # forecast_noise_std=0.0,   # [REMOVED] noise disabled
                      seed=42):
    """
    Receding-Horizon MPC — full remaining day as MILP horizon.
    At each slot t: solve MILP over [t…95], apply only P_c[0]/P_d[0], advance SoC.
    Forecast noise intentionally removed (clean signal only).
    """
    deg    = v2g.deg_cost_eur_kwh
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct / 100.0
    N, dt  = v2g.n_slots, v2g.dt_h

    P_c_all = np.zeros(N); P_d_all = np.zeros(N); e_all = np.zeros(N)
    soc = E_init

    for t in range(N):
        buy_fc = buy_day_ahead[t:].copy()
        v2g_fc = v2g_day_ahead[t:].copy()
        tru_w  = tru[t:];    plug_w = plugged[t:]

        # [REMOVED] Gaussian forecast noise disabled
        # if forecast_noise_std > 0:
        #     rng = np.random.default_rng(seed + t)
        #     buy_fc = add_realtime_noise(buy_fc, forecast_noise_std, rng)
        #     v2g_fc = add_realtime_noise(v2g_fc, forecast_noise_std, rng)

        P_c_w, P_d_w, _, ok = _solve_milp_window(
            v2g, buy_fc, v2g_fc, tru_w, plug_w,
            E_init=soc, E_fin=E_fin, deg=deg)
        if not ok:
            P_c_w, P_d_w, _ = _greedy_fallback(
                v2g, buy_fc, v2g_fc, tru_w, plug_w, soc, E_fin, deg)

        pc_t = float(np.clip(P_c_w[0], 0, v2g.p_c_max * plugged[t]))
        pd_t = float(np.clip(P_d_w[0], 0, v2g.p_d_max * plugged[t]))

        if pc_t > 1e-6 and pd_t > 1e-6:
            if (v2g_day_ahead[t] - deg) > (buy_day_ahead[t] + deg):
                pc_t = 0.0
            else:
                pd_t = 0.0

        soc -= tru[t] * dt
        soc += pc_t * v2g.eta_charge * dt
        soc -= pd_t / v2g.eta_discharge * dt
        soc  = float(np.clip(soc, v2g.E_min, v2g.E_max))

        P_c_all[t] = pc_t; P_d_all[t] = pd_t; e_all[t] = soc

    return _make_result(label, v2g, P_c_all, P_d_all, e_all,
                        buy_day_ahead, v2g_day_ahead, plugged, tru, deg)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – KPI BUILDER & SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def _make_result(label, v2g, P_c, P_d, e, buy, v2g_p, plugged, tru, deg):
    dt  = v2g.dt_h
    chg = float(np.sum(P_c * buy)   * dt)
    rev = float(np.sum(P_d * v2g_p) * dt)
    dgc = float(np.sum((P_c + P_d) * deg) * dt)
    return V2GResult(
        scenario=label, p_charge=P_c, p_discharge=P_d, soc=e,
        cost_eur_day=chg - rev + dgc,
        v2g_revenue_eur_day=rev,
        v2g_export_kwh_day=float(np.sum(P_d) * dt),
        charge_cost_eur_day=chg,
        deg_cost_eur_day=dgc,
        price_buy=buy, price_v2g=v2g_p, plugged=plugged, tru_load=tru)


def deg_sensitivity(v2g, buy, v2g_p, tru, plugged,
                    deg_values=None, soc_init=45.0, soc_final=80.0):
    """Sweep deg cost to find V2G breakeven."""
    if deg_values is None:
        deg_values = np.linspace(0.02, 0.15, 10)
    rows = []
    for dv in deg_values:
        E_i = v2g.usable_capacity_kWh * soc_init  / 100.0
        E_f = v2g.usable_capacity_kWh * soc_final / 100.0
        P_c, P_d, e, ok = _solve_milp_window(v2g, buy, v2g_p, tru, plugged, E_i, E_f, dv)
        if not ok:
            P_c, P_d, e = _greedy_fallback(v2g, buy, v2g_p, tru, plugged, E_i, E_f, dv)
        r = _make_result(f"deg={dv:.3f}", v2g, P_c, P_d, e, buy, v2g_p, plugged, tru, dv)
        rows.append({"DegCost_EUR_kWh": dv, "NetCost_EUR_day": r.cost_eur_day,
                     "V2G_Rev_EUR_day": r.v2g_revenue_eur_day,
                     "V2G_kWh_day": r.v2g_export_kwh_day,
                     "V2G_active": r.v2g_export_kwh_day > 0.1})
    return pd.DataFrame(rows)


# [FLEET SCALING DISABLED — uncomment to re-enable]
# def fleet_scaling(milp_r, mpc_r, fleet_sizes=[1, 5, 10, 25, 50]):
#     rows = []
#     for n in fleet_sizes:
#         rows.append({
#             "Fleet_n":              n,
#             "Peak_Charge_kW":       np.max(milp_r.p_charge)    * n,
#             "Peak_V2G_kW":          np.max(milp_r.p_discharge) * n,
#             "Annual_V2G_MWh":       milp_r.v2g_export_kwh_day * 365 * n / 1e3,
#             "MILP_Annual_Rev_EUR":  milp_r.v2g_revenue_eur_day * 365 * n,
#             "MPC_Annual_Rev_EUR":   mpc_r.v2g_revenue_eur_day  * 365 * n,
#         })
#     return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – MAIN RESULTS PLOT  (Scenarios A, B, C, D)
# ═══════════════════════════════════════════════════════════════════════════════

COL = {"dumb": "#AAAAAA", "smart": "#2196F3", "milp": "#00BCD4",
       "mpc":  "#FF7700", "price": "#007700", "tru": "#AA0000"}


def plot_all(hours, A, B, C, D, deg_df, season="winter", out="results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(
        f"S.KOe COOL  –  MILP + MPC V2G Optimisation  ({season})",
        fontsize=13, fontweight="bold")

    # (1) Charging schedules
    ax = axes[0, 0]
    ax.fill_between(hours, A.p_charge, step="pre", color=COL["dumb"],  alpha=0.55, label="A – Dumb")
    ax.fill_between(hours, B.p_charge, step="pre", color=COL["smart"], alpha=0.55, label="B – Smart (no V2G)")
    ax.fill_between(hours, C.p_charge, step="pre", color=COL["milp"],  alpha=0.45, label="C – MILP Day-Ahead")
    ax.fill_between(hours, D.p_charge, step="pre", color=COL["mpc"],   alpha=0.35, label="D – MPC Perfect")
    ax.step(hours, A.tru_load, where="post", color=COL["tru"], lw=1.2, ls="--", label="TRU load")
    ax.set_title("(1) P_c  –  Charging Power Schedule")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("P_c  (kW)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # (2) V2G discharge vs price
    ax = axes[0, 1]; ax2 = ax.twinx()
    w = 0.22
    ax.bar(hours - w/2, C.p_discharge, width=w, color=COL["milp"], alpha=0.8, label="C – MILP V2G (P_d)")
    ax.bar(hours + w/2, D.p_discharge, width=w, color=COL["mpc"],  alpha=0.7, label="D – MPC V2G (P_d)")
    ax2.step(hours, C.price_v2g, where="post", color=COL["price"], lw=1.8, label="V2G price (€/kWh)")
    ax.set_title("(2) P_d  –  V2G Discharge vs Price")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("P_d  (kW)")
    ax2.set_ylabel("Price  (€/kWh)", color=COL["price"])
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # (3) SoC traces
    ax = axes[0, 2]
    ax.plot(hours, A.soc, color=COL["dumb"],  lw=2.0, ls="-",  label="A – Dumb")
    ax.plot(hours, B.soc, color=COL["smart"], lw=2.0, ls="-",  label="B – Smart")
    ax.plot(hours, C.soc, color=COL["milp"],  lw=2.0, ls="-",  label="C – MILP")
    ax.plot(hours, D.soc, color=COL["mpc"],   lw=1.8, ls="--", label="D – MPC")
    ax.fill_between(hours, C.plugged * 4 + 57, 57, alpha=0.08, color="green", label="Plugged window")
    ax.axhline(v2g_global.E_min, color="red",   ls=":", lw=1, alpha=0.6, label=f"E_min ({v2g_global.E_min:.0f} kWh)")
    ax.axhline(v2g_global.E_max, color="navy",  ls=":", lw=1, alpha=0.6, label=f"E_max ({v2g_global.E_max:.0f} kWh)")
    ax.set_title("(3) SoC  –  Battery State of Charge"); ax.set_xlabel("Hour"); ax.set_ylabel("E  (kWh)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # (4) Daily cost comparison bar
    ax = axes[1, 0]
    labels = ["A\nDumb", "B\nSmart\n(no V2G)", "C\nMILP\nDay-Ahead", "D\nMPC\nPerfect"]
    costs  = [A.cost_eur_day, B.cost_eur_day, C.cost_eur_day, D.cost_eur_day]
    colors = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"]]
    ref    = A.cost_eur_day
    bars   = ax.bar(labels, costs, color=colors, alpha=0.85, edgecolor="black")
    for bar, v in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, max(v, 0) + 0.008,
                f"€{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(bars[1:], costs[1:]):
        saving = ref - v
        col = "darkgreen" if saving > 0 else "red"
        ax.text(bar.get_x() + bar.get_width()/2, min(v, 0) - 0.02,
                f"{saving:+.3f}/day", ha="center", fontsize=7.5, color=col)
    ax.axhline(ref, color=COL["dumb"], ls="--", lw=1, alpha=0.5, label="Dumb baseline")
    ax.set_title("(4) Net Daily Cost  [+ = cost,  − = revenue]")
    ax.set_ylabel("Net cost  (€/day)"); ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    # (5) Degradation sensitivity
    ax = axes[1, 1]; ax2 = ax.twinx()
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["NetCost_EUR_day"],
            "o-", color=COL["milp"], lw=2, label="Net Cost (€/day)")
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["V2G_Rev_EUR_day"],
            "s--", color=COL["mpc"], lw=2, label="V2G Revenue (€/day)")
    ax2.bar(deg_df["DegCost_EUR_kWh"], deg_df["V2G_kWh_day"],
            width=0.008, color=COL["mpc"], alpha=0.3, label="V2G export (kWh/day)")
    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    if not np.isnan(tipping):
        ax.axvline(tipping, color="red", ls=":", lw=1.5,
                   label=f"V2G cutoff ≈ {tipping:.3f} €/kWh")
    ax.axvline(v2g_global.deg_cost_eur_kwh, color="black", ls="--", lw=1,
               label=f"Active deg = {v2g_global.deg_cost_eur_kwh:.2f}")
    ax.set_title("(5) Degradation (deg) Sensitivity")
    ax.set_xlabel("deg  (€/kWh cycled)"); ax.set_ylabel("€/day")
    ax2.set_ylabel("V2G export  (kWh/day)", color=COL["mpc"])
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (6) Summary results table
    ax = axes[1, 2]; ax.axis("off")
    table_data = [
        ["Metric", "A – Dumb", "B – Smart", "C – MILP", "D – MPC"],
        ["Net cost (€/day)",
         f"{A.cost_eur_day:.3f}", f"{B.cost_eur_day:.3f}",
         f"{C.cost_eur_day:.3f}", f"{D.cost_eur_day:.3f}"],
        ["Charge cost (€/day)",
         f"{A.charge_cost_eur_day:.3f}", f"{B.charge_cost_eur_day:.3f}",
         f"{C.charge_cost_eur_day:.3f}", f"{D.charge_cost_eur_day:.3f}"],
        ["V2G revenue (€/day)",
         f"{A.v2g_revenue_eur_day:.3f}", f"{B.v2g_revenue_eur_day:.3f}",
         f"{C.v2g_revenue_eur_day:.3f}", f"{D.v2g_revenue_eur_day:.3f}"],
        ["deg cost (€/day)",
         f"{A.deg_cost_eur_day:.3f}", f"{B.deg_cost_eur_day:.3f}",
         f"{C.deg_cost_eur_day:.3f}", f"{D.deg_cost_eur_day:.3f}"],
        ["V2G export (kWh/day)",
         f"{A.v2g_export_kwh_day:.2f}", f"{B.v2g_export_kwh_day:.2f}",
         f"{C.v2g_export_kwh_day:.2f}", f"{D.v2g_export_kwh_day:.2f}"],
        ["Savings vs A (€/day)",
         "—",
         f"{ref - B.cost_eur_day:+.3f}",
         f"{ref - C.cost_eur_day:+.3f}",
         f"{ref - D.cost_eur_day:+.3f}"],
        ["Annual savings vs A (€/yr)",
         "—",
         f"{(ref-B.cost_eur_day)*365:+,.0f}",
         f"{(ref-C.cost_eur_day)*365:+,.0f}",
         f"{(ref-D.cost_eur_day)*365:+,.0f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#263238"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECEFF1")
        cell.set_edgecolor("#90A4AE")
    col_colors = ["#EEEEEE", "#BBDEFB", "#B2EBF2", "#FFE0B2"]
    for c_idx, bg in enumerate(col_colors):
        for r_idx in range(1, len(table_data)):
            tbl[(r_idx, c_idx+1)].set_facecolor(bg)
    ax.set_title("(6) KPI Summary Table", fontsize=10, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – 8 ADDITIONAL ANALYTICAL GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_additional_analysis(hours, A, B, C, D, buy, v2g_p,
                             all_season_results: dict, csv_path: str,
                             out="additional_analysis.png"):
    fig = plt.figure(figsize=(22, 20))
    gs  = GridSpec(4, 2, figure=fig, hspace=0.44, wspace=0.32)
    fig.suptitle("V2G Additional Analysis — Financial · Grid · Operational",
                 fontsize=14, fontweight="bold", y=0.99)

    results_list = [A, B, C, D]
    labels_short = ["A–Dumb", "B–Smart", "C–MILP", "D–MPC"]
    colors_list  = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"]]

    # ── Graph 1: Price Duration Curve ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_buy = np.sort(buy)[::-1]
    ax1.fill_between(range(len(sorted_buy)), sorted_buy * 1000,
                     color="#1976D2", alpha=0.6)
    ax1.plot(sorted_buy * 1000, color="#0D47A1", lw=1.5)
    ax1.axhline(np.mean(buy) * 1000, color="red", ls="--", lw=1.5,
                label=f"Mean: {np.mean(buy)*1000:.1f} €/MWh")
    pct_neg = np.sum(buy < 0) / len(buy) * 100
    pct_hi  = np.sum(buy > 0.10) / len(buy) * 100
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_title("① Price Duration Curve  (day-ahead DE/LU spot)",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("Sorted 15-min slots (0 = most expensive)")
    ax1.set_ylabel("Price  (€/MWh)")
    ax1.legend(fontsize=8)
    ax1.text(0.65, 0.85,
             f"Negative: {pct_neg:.1f}% of day\nHigh (>100 €/MWh): {pct_hi:.1f}%",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="#E3F2FD", alpha=0.8))
    ax1.grid(True, alpha=0.3)

    # ── Graph 2: Arbitrage Spread per Hour ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    spread = v2g_p - buy
    ax2.bar(hours, spread * 1000, width=0.22,
            color=np.where(spread > v2g_global.deg_cost_eur_kwh, "#43A047", "#E53935"),
            alpha=0.8, label="V2G spread (€/MWh)")
    ax2.axhline(v2g_global.deg_cost_eur_kwh * 1000, color="black", ls="--",
                lw=1.5, label=f"deg cost = {v2g_global.deg_cost_eur_kwh*1000:.0f} €/MWh")
    ax2.axhline(0, color="grey", lw=0.8)
    plug_mask = A.plugged > 0.5
    for i in range(len(hours)-1):
        if plug_mask[i]:
            ax2.axvspan(hours[i], hours[i]+0.25, alpha=0.06, color="blue")
    ax2.set_title("② Arbitrage Spread  (V2G price − buy price)",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Hour of day"); ax2.set_ylabel("Spread  (€/MWh)")
    ax2.legend(fontsize=8)
    ax2.text(0.02, 0.97, "■ Blue shading = plugged window",
             transform=ax2.transAxes, fontsize=7.5, va="top", color="navy")
    ax2.set_xlim(0, 24); ax2.grid(True, alpha=0.3)

    # ── Graph 3: Cumulative Cost Over 24h ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    dt   = v2g_global.dt_h
    for r, lbl, col in zip(results_list, labels_short, colors_list):
        cum = np.cumsum((r.p_charge * r.price_buy - r.p_discharge * r.price_v2g
                         + (r.p_charge + r.p_discharge) * r.deg_cost_eur_day / max(1, np.sum(r.p_charge + r.p_discharge))) * dt)
        ax3.plot(hours, cum, color=col, lw=2, label=lbl)
    ax3.axhline(0, color="grey", lw=0.8)
    ax3.fill_between(hours, 0, np.cumsum(
        (A.p_charge * A.price_buy) * dt), alpha=0.08, color=COL["dumb"])
    ax3.set_title("③ Cumulative Net Cost Over Day  (running €)",
                  fontsize=10, fontweight="bold")
    ax3.set_xlabel("Hour of day"); ax3.set_ylabel("Cumulative cost  (€)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_xlim(0, 24)

    # ── Graph 4: Power Heatmap ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    power_matrix = np.vstack([
        A.p_charge,
        B.p_charge,
        C.p_charge - C.p_discharge,
        D.p_charge - D.p_discharge,
    ])
    im = ax4.imshow(power_matrix, aspect="auto", cmap="RdYlGn",
                    vmin=-v2g_global.p_d_max, vmax=v2g_global.p_c_max,
                    extent=[0, 24, -0.5, 3.5])
    plt.colorbar(im, ax=ax4, label="Net power  (kW)  [green=charge, red=discharge]",
                 shrink=0.85)
    ax4.set_yticks([0, 1, 2, 3]); ax4.set_yticklabels(labels_short[::-1])
    ax4.set_title("④ Net Power Heatmap  (P_c − P_d  per scenario/hour)",
                  fontsize=10, fontweight="bold")
    ax4.set_xlabel("Hour of day")

    # ── Graph 5: Revenue Breakdown per Season ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    season_labels = list(all_season_results.keys())
    x = np.arange(len(season_labels))
    w = 0.18
    for i, (sc_key, col) in enumerate(zip(["A","B","C","D"], colors_list)):
        chg_costs = [all_season_results[s][sc_key].charge_cost_eur_day  for s in season_labels]
        v2g_revs  = [-all_season_results[s][sc_key].v2g_revenue_eur_day for s in season_labels]
        deg_costs = [all_season_results[s][sc_key].deg_cost_eur_day     for s in season_labels]
        offset    = (i - 1.5) * w
        ax5.bar(x + offset, chg_costs, width=w, color=col,   alpha=0.85)
        ax5.bar(x + offset, v2g_revs,  width=w, color=col,   alpha=0.85,
                bottom=chg_costs, hatch="///", edgecolor="white")
        ax5.bar(x + offset, deg_costs, width=w, color="black", alpha=0.25,
                bottom=[c+v for c,v in zip(chg_costs, v2g_revs)])
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.replace("_", "\n") for s in season_labels], fontsize=8)
    ax5.axhline(0, color="black", lw=0.8)
    legend_patches = [
        mpatches.Patch(color=c, label=l) for c,l in zip(colors_list, labels_short)]
    legend_patches += [mpatches.Patch(facecolor="grey", hatch="///",
                                       label="V2G revenue (hatched, reduces bar)")]
    ax5.legend(handles=legend_patches, fontsize=7, loc="upper right")
    ax5.set_title("⑤ Cost/Revenue Breakdown by Season & Scenario",
                  fontsize=10, fontweight="bold")
    ax5.set_ylabel("€/day  (charge cost + deg − V2G revenue)")
    ax5.grid(True, alpha=0.3, axis="y")

    # ── Graph 6: Grid Power Flow ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    for r, lbl, col in zip([C, D], ["C–MILP", "D–MPC"],
                            [COL["milp"], COL["mpc"]]):
        net_grid = r.p_charge - r.p_discharge
        ax6.step(hours, net_grid, where="post", color=col, lw=2, label=lbl)
        ax6.fill_between(hours, net_grid, 0,
                         where=net_grid >= 0, step="post",
                         alpha=0.15, color=col)
        ax6.fill_between(hours, net_grid, 0,
                         where=net_grid < 0, step="post",
                         alpha=0.25, color="red")
    ax6.step(hours, A.p_charge, where="post", color=COL["dumb"], lw=1.5,
             ls="--", alpha=0.7, label="A–Dumb (reference)")
    ax6.axhline(0, color="black", lw=1)
    ax6.set_title("⑥ Grid Power Flow  (P_c − P_d)",
                  fontsize=10, fontweight="bold")
    ax6.set_xlabel("Hour of day")
    ax6.set_ylabel("Net grid exchange  (kW)\n(+) import  /  (−) export to grid")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3); ax6.set_xlim(0, 24)
    ax6.text(0.02, 0.05, "Red fill = V2G grid export",
             transform=ax6.transAxes, fontsize=8, color="red")

    # ── Graph 7: SoC Risk Corridor ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    for r, lbl, col in zip(results_list, labels_short, colors_list):
        headroom = r.soc - v2g_global.E_min
        ax7.plot(hours, headroom, color=col, lw=2, label=lbl)
    ax7.fill_between(hours, 0, 2, alpha=0.15, color="red",
                     label="Risk zone (< 2 kWh above E_min)")
    ax7.axhline(0, color="red", lw=1.5, ls="--")
    ax7.set_title("⑦ SoC Risk Corridor  (SoC − E_min headroom)",
                  fontsize=10, fontweight="bold")
    ax7.set_xlabel("Hour of day")
    ax7.set_ylabel("E headroom  (kWh above E_min)")
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3); ax7.set_xlim(0, 24)

    # ── Graph 8: Annual Cost per Trailer ──────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    def annual(r_key):
        return (all_season_results["winter"][r_key].cost_eur_day * 130 +
                all_season_results["summer"][r_key].cost_eur_day * 131 +
                all_season_results.get("winter_weekend", {}).get(r_key,
                    all_season_results["winter"][r_key]).cost_eur_day * 52 +
                all_season_results.get("summer_weekend", {}).get(r_key,
                    all_season_results["summer"][r_key]).cost_eur_day * 52)

    ann_costs = [annual(k) for k in ["A","B","C","D"]]
    base = ann_costs[0]
    bars = ax8.bar(["A–Dumb","B–Smart","C–MILP","D–MPC"], ann_costs,
                   color=colors_list, alpha=0.85, edgecolor="black")
    for bar, v, base_v in zip(bars, ann_costs, [base]*4):
        saving = base - v
        ax8.text(bar.get_x() + bar.get_width()/2, v + 15,
                 f"€{v:,.0f}\n({saving:+,.0f} vs A)",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax8.set_title("⑧ Annual Net Cost per Trailer  (all seasons combined)",
                  fontsize=10, fontweight="bold")
    ax8.set_ylabel("Annual net cost  (€/year)")
    ax8.grid(True, alpha=0.3, axis="y")
    ax8.set_ylim(0, base * 1.15)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Additional analysis saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 – CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results, deg_df, season="winter", price_source=""):
    ref = results["A"].cost_eur_day
    print("\n" + "="*84)
    print(f"  RESULTS — {season.upper()}  |  {price_source[:65]}")
    print("="*84)
    print(f"  {'Scenario':<32} {'Net €/day':>9} {'Charge €':>9} "
          f"{'V2G Rev €':>10} {'deg €':>7} {'V2G kWh':>8} {'vs Dumb':>10}")
    print("-"*84)
    for r in results.values():
        print(f"  {r.scenario:<32} {r.cost_eur_day:>9.4f} "
              f"{r.charge_cost_eur_day:>9.4f} {r.v2g_revenue_eur_day:>10.4f} "
              f"{r.deg_cost_eur_day:>7.4f} {r.v2g_export_kwh_day:>8.2f} "
              f"  {ref - r.cost_eur_day:>+.4f}")
    print("="*84)
    print("  Annualised (365 days):")
    for r in results.values():
        s = (ref - r.cost_eur_day) * 365
        print(f"    {r.scenario:<32}  annual cost: €{r.cost_eur_day*365:>8,.0f}"
              f"   savings vs Dumb: €{s:>+,.0f}/yr")
    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    print(f"\n  Degradation tipping point: V2G profitable up to ≈€{tipping:.3f}/kWh  "
          f"({'OK' if v2g_global.deg_cost_eur_kwh <= tipping else 'WARN'})")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

v2g_global: V2GParams = None   # type: ignore


def main():
    global v2g_global

    print("\n" + "="*65)
    print("  S.KOe COOL  –  Day-Ahead MILP + Receding-Horizon MPC V2G")
    print("  Schmitz Cargobull AG  |  2025")
    print("  Biedenbach & Strunz (2024) · Agora Verkehrswende (2025)")
    print("="*65)

    import os
    csv_candidates = [
        Path(__file__).parent / "2025_Electricity_Price.csv",
        Path("2025_Electricity_Price.csv"),
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = str(p); break
    if csv_path is None:
        print("\n  ERROR: 2025_Electricity_Price.csv not found.")
        print("  Place the SMARD CSV in the same folder as this script.\n")
        sys.exit(1)
    print(f"\n  Price data: {csv_path}")

    v2g = V2GParams()
    v2g_global = v2g
    print(f"\n  Battery: {v2g.battery_capacity_kWh} kWh total, "
          f"{v2g.usable_capacity_kWh} kWh usable  |  "
          f"P_c_max={v2g.p_c_max:.0f} kW  |  P_d_max={v2g.p_d_max:.0f} kW")

    print("\n" + "─"*55)
    print("  BATTERY DEGRADATION COST")
    print(f"  Current value: deg = €{v2g.deg_cost_eur_kwh:.3f} / kWh cycled")
    answer = input("  Include battery degradation cost in optimisation? [Y/n]: ").strip().lower()
    if answer in ("n", "no"):
        v2g.deg_cost_eur_kwh = 0.0
        print("  → Degradation cost set to 0.  V2G arbitrage uses raw price spread only.")
    else:
        answer2 = input(f"  Keep default €{v2g.deg_cost_eur_kwh:.3f}/kWh? [Y / enter value]: ").strip()
        if answer2 and answer2.lower() not in ("y", "yes"):
            try:
                v2g.deg_cost_eur_kwh = float(answer2)
                print(f"  → Degradation cost set to €{v2g.deg_cost_eur_kwh:.3f}/kWh")
            except ValueError:
                print(f"  → Invalid input. Keeping default €{v2g.deg_cost_eur_kwh:.3f}/kWh")
        else:
            print(f"  → Degradation cost: €{v2g.deg_cost_eur_kwh:.3f}/kWh")
    print("─"*55)

    soc_init_pct  = 45.0
    soc_final_pct = 80.0
    deg_values    = load_deg_sensitivity(v2g)
    hours         = np.arange(v2g.n_slots) * v2g.dt_h

    print("\n  Generating abbreviation legend …")
    generate_abbreviation_legend("abbreviation_legend.png")

    all_season_results: dict = {}

    DAY_TYPES = [
        ("winter",         "Extended", 130, "Winter weekday  (Mon–Fri, Oct–Mar)"),
        ("summer",         "Extended", 131, "Summer weekday  (Mon–Fri, Apr–Sep)"),
        ("winter_weekend", "Weekend",   52, "Winter weekend  (Sat–Sun, Oct–Mar)"),
        ("summer_weekend", "Weekend",   52, "Summer weekend  (Sat–Sun, Apr–Sep)"),
    ]

    annual_cost_milp    = 0.0
    annual_v2g_milp     = 0.0
    annual_savings_dumb = 0.0

    for season, dwell_type, days_per_year, label in DAY_TYPES:
        print(f"\n{'='*65}")
        print(f"  {label}  ({days_per_year} days/year)")
        print(f"{'='*65}")

        tru, plugged = build_load_and_availability(v2g, dwell=dwell_type)
        buy, v2g_p, price_source = load_prices_from_csv(csv_path, v2g, season=season)

        print(f"  Prices: {price_source}")
        print(f"  Buy: €{buy.min()*1000:.1f}–{buy.max()*1000:.1f} €/MWh  |  "
              f"V2G peak: {v2g_p.max()*1000:.1f} €/MWh  |  "
              f"Plugged: {int(plugged.sum()*v2g.dt_h)}h/day")

        A = run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        B = run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        C = run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        D = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct,
                              label="D - MPC perfect")

        # [SCENARIO E DISABLED — Gaussian noise removed]
        # E = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct,
        #                       label="E - MPC noisy",
        #                       forecast_noise_std=v2g.mpc_price_noise_std, seed=42)

        results  = {"A": A, "B": B, "C": C, "D": D}
        deg_df   = deg_sensitivity(v2g, buy, v2g_p, tru, plugged,
                                   deg_values, soc_init_pct, soc_final_pct)

        # [FLEET SCALING DISABLED]
        # fleet_df = fleet_scaling(C, D)

        all_season_results[season] = results
        print_report(results, deg_df, season=label, price_source=price_source)

        out_png = f"results_{season}.png"
        plot_all(hours, A, B, C, D, deg_df, season=label, out=out_png)

        annual_cost_milp    += C.cost_eur_day         * days_per_year
        annual_v2g_milp     += C.v2g_revenue_eur_day  * days_per_year
        annual_savings_dumb += (A.cost_eur_day - C.cost_eur_day) * days_per_year

    print("\n  Generating 8 additional analysis graphs …")
    tru_w, plugged_w = build_load_and_availability(v2g, dwell="Extended")
    buy_w, v2g_p_w, _ = load_prices_from_csv(csv_path, v2g, season="winter")
    plot_additional_analysis(
        hours,
        all_season_results["winter"]["A"],
        all_season_results["winter"]["B"],
        all_season_results["winter"]["C"],
        all_season_results["winter"]["D"],
        buy_w, v2g_p_w,
        all_season_results,
        csv_path,
        out="additional_analysis.png"
    )

    print(f"\n{'='*65}")
    print(f"  ANNUAL SUMMARY — Single Trailer (Scenario C MILP)")
    print(f"{'='*65}")
    print(f"  Annual energy cost (MILP):        €{annual_cost_milp:>8,.0f}/year")
    print(f"  Annual V2G revenue (MILP):        €{annual_v2g_milp:>8,.0f}/year")
    print(f"  Annual savings vs Dumb charging:  €{annual_savings_dumb:>8,.0f}/year")
    print(f"  [Agora 2025 benchmark for car:    €~500/year for arbitrage only]")
    print(f"\n  Output files:")
    print(f"    abbreviation_legend.png")
    print(f"    results_winter.png / results_summer.png")
    print(f"    results_winter_weekend.png / results_summer_weekend.png")
    print(f"    additional_analysis.png")
    print()


if __name__ == "__main__":
    main()