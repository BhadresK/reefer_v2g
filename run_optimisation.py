#!/usr/bin/env python3

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path


#  SECTION 1 – PARAMETERS

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
    mpc_price_noise_std:   float = 0.012    # sigma in EUR/kWh (Liu 2023)

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
#  SECTION 2 – PRICE & LOAD DATA  (real Excel or synthetic fallback)
#
#  Priority order:
#    1. data/v2g_params.xlsx  → real EPEX-based prices (run make_data.py once)
#    2. Synthetic fallback     → hardcoded German tariff structure
#
#  To use real data: run  python make_data.py  once in the same folder.
#  To use your own:  fill data/v2g_params.xlsx → sheet "Prices15min" with
#                    columns: BuyPrice_EUR_kWh, V2G_Price_EUR_kWh  (96 rows)
# ═══════════════════════════════════════════════════════════════════════════════

import os as _os
DATA_FILE = Path(_os.path.dirname(_os.path.abspath(__file__))) / "data" / "v2g_params.xlsx"


def load_prices(v2g: V2GParams, season: str = "winter") -> tuple:
    """
    Load 96-slot day-ahead buy and V2G prices.

    If data/v2g_params.xlsx exists:
        Reads from sheet 'SeasonalPrices' (winter or summer column pair).
        These are real EPEX spot + BNetzA regulated tariffs + VAT.
        Source: EPEX SMARD.de 2024 representative weekday + BNetzA 2024.

    Otherwise:
        Synthetic fallback — German EPEX-style tariff structure.
        Still correctly shaped (FCR peak, midday dip) but not from real data.
        Results will be clearly labelled as synthetic.

    Returns: (buy_price, v2g_price) each shape (96,), plus data_source label.
    """
    if DATA_FILE.exists():
        try:
            df = pd.read_excel(DATA_FILE, sheet_name="SeasonalPrices", engine="openpyxl")
            if season == "summer_weekend":
                buy   = df["WkndSummer_Buy_EUR_kWh"].values.astype(float)
                v2g_p = df["WkndSummer_V2G_EUR_kWh"].values.astype(float)
            elif season == "winter_weekend":
                buy   = df["WkndWinter_Buy_EUR_kWh"].values.astype(float)
                v2g_p = df["WkndWinter_V2G_EUR_kWh"].values.astype(float)
            elif season == "summer":
                buy   = df["Summer_Buy_EUR_kWh"].values.astype(float)
                v2g_p = df["Summer_V2G_EUR_kWh"].values.astype(float)
            else:
                buy   = df["Winter_Buy_EUR_kWh"].values.astype(float)
                v2g_p = df["Winter_V2G_EUR_kWh"].values.astype(float)
            assert len(buy) == 96, "SeasonalPrices sheet must have 96 rows"
            source = f"REAL — data/v2g_params.xlsx (EPEX SMARD.de 2024, {season} WD)"
            return buy, v2g_p, source
        except Exception as e:
            warnings.warn(f"Could not read {DATA_FILE}: {e} — using synthetic prices")

    # Synthetic fallback
    N = v2g.n_slots
    h = np.arange(N) * v2g.dt_h
    buy = np.select(
        condlist=[
            (h >= 6)  & (h < 8),   (h >= 8)  & (h < 11),
            (h >= 11) & (h < 14),  (h >= 14) & (h < 16),
            (h >= 16) & (h < 20),  (h >= 20) & (h < 22),
        ],
        choicelist=[0.28, 0.30, 0.24, 0.22, 0.33, 0.26],
        default=0.16
    )
    v2g_p  = buy + np.where((h >= 16) & (h < 20), 0.132, 0.0)
    source = "SYNTHETIC — run make_data.py to use real EPEX prices"
    return buy, v2g_p, source


def load_battery_params(v2g_default: V2GParams) -> tuple:
    """
    Load battery parameters from Excel if available, else use defaults.
    Returns (V2GParams, source_label).
    """
    if DATA_FILE.exists():
        try:
            df = pd.read_excel(DATA_FILE, sheet_name="BatteryParams",
                               engine="openpyxl").set_index("Parameter")["Value"]
            v2g = V2GParams(
                battery_capacity_kWh   = float(df["BatteryCapacity_kWh"]),
                usable_capacity_kWh    = float(df["UsableBatteryCap_kWh"]),
                soc_min_pct            = float(df["SOC_min_pct"]),
                soc_max_pct            = float(df["SOC_max_pct"]),
                charge_power_kW        = float(df["ChargePower_kW"]),
                discharge_power_kW     = float(df["DischargePower_V2G_kW"]),
                eta_charge             = float(df["ChargeEfficiency"]),
                eta_discharge          = float(df["DischargeEfficiency"]),
                deg_cost_eur_kwh       = float(df["DegradationCost_EUR_kWh"]),
                mpc_price_noise_std    = v2g_default.mpc_price_noise_std,
            )
            return v2g, f"REAL — data/v2g_params.xlsx (S.KOe COOL spec)"
        except Exception as e:
            warnings.warn(f"BatteryParams read failed: {e} — using defaults")
    return v2g_default, "DEFAULT — hardcoded S.KOe COOL parameters"


def load_deg_sensitivity(v2g: V2GParams) -> np.ndarray:
    """Load degradation sweep values from Excel or use default range."""
    if DATA_FILE.exists():
        try:
            df = pd.read_excel(DATA_FILE, sheet_name="DegSensitivity", engine="openpyxl")
            return df["DegCost_EUR_kWh"].values.astype(float)
        except Exception:
            pass
    return np.linspace(0.02, 0.15, 10)


def build_load_and_availability(v2g: V2GParams, dwell: str = "Extended") -> tuple:
    """
    TRU auxiliary load and plug-in availability window.
    Extended = night (21-07) + midday depot stop (12-18).  16h plugged/day.
    NightOnly = night (21-07) only.  10h plugged/day.
    """
    N = v2g.n_slots
    h = np.arange(N) * v2g.dt_h
    tru = 2.8 + 1.2 * np.sin(2 * np.pi * np.arange(N) / N + np.pi)
    if dwell == "Weekend":
        # Trailer at depot all day Saturday/Sunday — fully available 24h
        plugged = np.ones(N)
    elif dwell == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    else:  # Extended (default)
        plugged = ((h >= 21) | (h < 7) | ((h >= 12) & (h < 18))).astype(float)
    return tru, plugged


def add_realtime_noise(prices: np.ndarray, std: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Gaussian intraday deviation from day-ahead forecast (Liu 2023)."""
    return np.maximum(0.01, prices + rng.normal(0, std, size=len(prices)))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – MILP INNER SOLVER  (scipy HiGHS)
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_milp_window(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg):
    """
    Solve MILP over one window of length W = len(buy).

    OBJECTIVE (Biedenbach & Strunz 2024, eq. 2-6):
        min  sum_t [ buy[t]*P_c[t] - v2g[t]*P_d[t] + deg*(P_c[t]+P_d[t]) ] * dt

    CONSTRAINTS:
        (i)   SoC dynamics:
                e[t] = e[t-1] + eta_c*P_c[t]*dt - (1/eta_d)*P_d[t]*dt - tru[t]*dt
        (ii)  Power bounds: 0 <= P_c <= p_c_max*plug, 0 <= P_d <= p_d_max*plug
        (iii) SoC bounds:   E_min <= e[t] <= E_max
        (iv)  Linearised mutex: P_c[t] + P_d[t] <= max_power
        (v)   Departure SoC: e[W-1] >= E_fin

    Variable layout: x = [P_c(0..W-1) | P_d(0..W-1) | e(0..W-1)]
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

    # Cost
    c_vec        = np.zeros(nv)
    c_vec[idx_c] =  buy   * dt + deg * dt
    c_vec[idx_d] = -v2g_p * dt + deg * dt

    # Bounds
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[idx_c] = v2g.p_c_max * plugged
    ub[idx_d] = v2g.p_d_max * plugged
    lb[idx_e] = v2g.E_min
    ub[idx_e] = v2g.E_max

    # Constraints
    n_rows = W + W + 1
    A  = lil_matrix((n_rows, nv))
    lo = np.zeros(n_rows)
    hi = np.zeros(n_rows)

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
        A[row, idx_c[t]] = 1.0
        A[row, idx_d[t]] = 1.0
        lo[row] = -np.inf
        hi[row] =  max_p

    A[2*W, idx_e[W-1]] = 1.0
    lo[2*W] = E_fin
    hi[2*W] = v2g.E_max

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
        soc -= tru[t] * dt
        soc  = max(v2g.E_min, soc)
        if plugged[t]:
            margin = (v2g_p[t] - deg) - (buy[t] + deg)
            if margin > 0.02 and soc > v2g.E_min + v2g.p_d_max * dt / v2g.eta_discharge:
                p = min(v2g.p_d_max, (soc - v2g.E_min) * v2g.eta_discharge / dt)
                P_d[t] = p
                soc = max(v2g.E_min, soc - p / v2g.eta_discharge * dt)
            elif buy[t] < 0.22 and soc < v2g.E_max - v2g.p_c_max * dt * v2g.eta_charge:
                p = min(v2g.p_c_max, (v2g.E_max - soc) / (v2g.eta_charge * dt))
                P_c[t] = p
                soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    if soc < E_fin:
        for t in range(N-1, -1, -1):
            if plugged[t] and P_d[t] == 0:
                deficit = E_fin - soc
                extra   = min(v2g.p_c_max * dt * v2g.eta_charge, deficit)
                P_c[t] += extra / (v2g.eta_charge * dt)
                soc    += extra
                if soc >= E_fin:
                    break
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
        soc -= tru[t] * dt
        soc  = max(v2g.E_min, soc)
        if plugged[t] and soc < v2g.E_max:
            p = min(v2g.p_c_max, (v2g.E_max - soc) / (v2g.eta_charge * dt))
            P_c[t] = p
            soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return _make_result("A - Dumb (uncontrolled)", v2g, P_c, P_d, e,
                        buy, v2g_p, plugged, tru, deg=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – SCENARIO B: SMART CHARGING ONLY (no V2G discharge)
# ═══════════════════════════════════════════════════════════════════════════════

def run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged,
                     soc_init_pct=45.0, soc_final_pct=80.0):
    """
    Day-ahead MILP: price-optimal charging, V2G discharge blocked.

    Degradation cost is set to ZERO here. Why:
      Degradation cost (EUR/kWh cycled) is an opportunity cost that exists
      because every kWh cycled through the battery ages it. In Scenario B
      the trailer charges once per day from arrival SoC to departure target —
      this is unavoidable charging that would happen anyway.
      The degradation cost only becomes relevant as an economic trade-off
      in Scenario C/D/E where V2G arbitrage adds *extra* cycling beyond
      the minimum needed to reach departure SoC.
      Including it in B would penalise B unfairly and make the comparison
      between B and C meaningless (B would look worse than dumb by design).

    This matches the Biedenbach & Strunz (2024) formulation where deg cost
    is applied only to the V2G bidirectional cycling delta.
    """
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct / 100.0

    # Block V2G: zero-out V2G price so discharge is never in objective
    # Degradation cost = 0 for unavoidable one-way daily charge
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
    """
    Full 24h MILP solved once at 00:00 using complete day-ahead price signal.
    Theoretical optimum — perfect information, global solve.
    """
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
#  SECTION 7 – SCENARIOS D & E: RECEDING-HORIZON MPC
#
#  WHY FULL REMAINING HORIZON FIXES THE PROBLEM:
#  ──────────────────────────────────────────────
#  Old approach (fixed 4h window):
#    At 08:00, horizon reaches 12:00 only.
#    Solver cannot see the 16:00 price peak → decides not to charge.
#    Battery is half empty at 16:00 → misses V2G revenue.
#
#  New approach (full remaining day):
#    At 08:00, horizon reaches 23:45.
#    Solver sees the 16:00 peak clearly from the very first slot.
#    Charges during cheap morning slots, discharges into the peak.
#    → MPC matches (or very nearly matches) the MILP day-ahead result.
#
#  Departure SoC is always enforced at slot 95 (end of day),
#  regardless of where in the day the MPC currently is.
# ═══════════════════════════════════════════════════════════════════════════════

def run_mpc_day_ahead(v2g, buy_day_ahead, v2g_day_ahead, tru, plugged,
                      soc_init_pct=45.0, soc_final_pct=80.0,
                      forecast_noise_std=0.0, label="MPC", seed=42):
    """
    Receding-Horizon MPC with full remaining day as the MILP horizon.

    At each slot t (00:00 to 23:45):

      Step 1 - FORECAST
        Take day-ahead prices for slots [t ... 95].
        If noise > 0, add Gaussian perturbation simulating intraday updates.

      Step 2 - SOLVE
        Run MILP over W = (96 - t) remaining slots.
        E_init = current real SoC.
        E_fin  = departure target enforced at slot 95 always.

        Key insight: at t=0 this is identical to full day-ahead MILP.
        At t=32 (08:00) there are still 64 slots left — the solver
        always sees the 16:00 evening peak no matter what time it is.

      Step 3 - EXECUTE
        Apply only P_c[0] and P_d[0] (receding horizon principle).
        Discard the rest of the optimised schedule.

      Step 4 - ADVANCE
        Update real SoC: soc += P_c*eta_c*dt - P_d/eta_d*dt - tru*dt

      Step 5 - REPEAT at t+1 with updated SoC and W-1 remaining slots.
    """
    deg    = v2g.deg_cost_eur_kwh
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct / 100.0
    N, dt  = v2g.n_slots, v2g.dt_h

    rng     = np.random.default_rng(seed)
    P_c_all = np.zeros(N)
    P_d_all = np.zeros(N)
    e_all   = np.zeros(N)
    soc     = E_init

    for t in range(N):
        # Step 1: Forecast = full remaining day
        W       = N - t
        buy_fc  = buy_day_ahead[t:].copy()
        v2g_fc  = v2g_day_ahead[t:].copy()
        tru_w   = tru[t:]
        plug_w  = plugged[t:]

        if forecast_noise_std > 0:
            buy_fc = add_realtime_noise(buy_fc, forecast_noise_std, rng)
            v2g_fc = add_realtime_noise(v2g_fc, forecast_noise_std, rng)

        # Step 2: Solve MILP over W remaining slots
        P_c_w, P_d_w, _, ok = _solve_milp_window(
            v2g, buy_fc, v2g_fc, tru_w, plug_w,
            E_init=soc, E_fin=E_fin, deg=deg)

        if not ok:
            P_c_w, P_d_w, _ = _greedy_fallback(
                v2g, buy_fc, v2g_fc, tru_w, plug_w, soc, E_fin, deg)

        # Step 3: Apply first action only
        pc_t = float(np.clip(P_c_w[0], 0, v2g.p_c_max * plugged[t]))
        pd_t = float(np.clip(P_d_w[0], 0, v2g.p_d_max * plugged[t]))

        # Hard mutual exclusion (pick economically better action)
        if pc_t > 1e-6 and pd_t > 1e-6:
            if (v2g_day_ahead[t] - deg) > (buy_day_ahead[t] + deg):
                pc_t = 0.0   # discharging more profitable
            else:
                pd_t = 0.0   # charging cheaper

        # Step 4: Advance real SoC
        soc -= tru[t] * dt
        soc += pc_t * v2g.eta_charge * dt
        soc -= pd_t / v2g.eta_discharge * dt
        soc  = float(np.clip(soc, v2g.E_min, v2g.E_max))

        P_c_all[t] = pc_t
        P_d_all[t] = pd_t
        e_all[t]   = soc

    return _make_result(label, v2g, P_c_all, P_d_all, e_all,
                        buy_day_ahead, v2g_day_ahead, plugged, tru, deg)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – KPI BUILDER, SENSITIVITY, FLEET SCALING
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
    """Sweep deg cost — finds V2G breakeven point. Uses Excel values if loaded."""
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


def fleet_scaling(milp_r, mpc_r, fleet_sizes=[1, 5, 10, 25, 50]):
    rows = []
    for n in fleet_sizes:
        rows.append({
            "Fleet_n":              n,
            "Peak_Charge_kW":       np.max(milp_r.p_charge)    * n,
            "Peak_V2G_kW":          np.max(milp_r.p_discharge) * n,
            "Annual_V2G_MWh":       milp_r.v2g_export_kwh_day * 365 * n / 1e3,
            "MILP_Annual_Rev_EUR":  milp_r.v2g_revenue_eur_day * 365 * n,
            "MPC_Annual_Rev_EUR":   mpc_r.v2g_revenue_eur_day  * 365 * n,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

COL = {"dumb": "#AAAAAA", "smart": "#2196F3", "milp": "#00BCD4",
       "mpc":  "#FF7700", "mpc_n": "#CC3300", "price": "#007700", "tru": "#AA0000"}

def plot_all(hours, A, B, C, D, E, deg_df, fleet_df, season="winter", out="results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(
        f"S.KOe COOL  -  MILP + MPC V2G Optimisation  ({season.capitalize()} Weekday)",
        fontsize=13, fontweight="bold")

    # (1) Charging schedules
    ax = axes[0, 0]
    ax.fill_between(hours, A.p_charge, step="pre", color=COL["dumb"],  alpha=0.55, label="A - Dumb")
    ax.fill_between(hours, B.p_charge, step="pre", color=COL["smart"], alpha=0.55, label="B - Smart (no V2G)")
    ax.fill_between(hours, C.p_charge, step="pre", color=COL["milp"],  alpha=0.45, label="C - MILP Day-Ahead")
    ax.fill_between(hours, D.p_charge, step="pre", color=COL["mpc"],   alpha=0.35, label="D - MPC (perfect)")
    ax.step(hours, A.tru_load, where="post", color=COL["tru"], lw=1.2, ls="--", label="TRU load")
    ax.set_title("(1) Charging Power Schedule"); ax.set_xlabel("Hour"); ax.set_ylabel("kW")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # (2) V2G discharge vs price
    ax = axes[0, 1]; ax2 = ax.twinx()
    w = 0.22
    ax.bar(hours - w, C.p_discharge, width=w, color=COL["milp"],  alpha=0.8, label="C - MILP V2G")
    ax.bar(hours,     D.p_discharge, width=w, color=COL["mpc"],   alpha=0.7, label="D - MPC perfect")
    ax.bar(hours + w, E.p_discharge, width=w, color=COL["mpc_n"], alpha=0.6, label="E - MPC noisy")
    ax2.step(hours, C.price_v2g, where="post", color=COL["price"], lw=1.8, label="V2G price")
    ax.set_title("(2) V2G Discharge vs Price"); ax.set_xlabel("Hour"); ax.set_ylabel("kW discharge")
    ax2.set_ylabel("EUR/kWh", color=COL["price"])
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # (3) SoC traces
    ax = axes[0, 2]
    ax.plot(hours, A.soc, color=COL["dumb"],  lw=2.0, ls="-",  label="A - Dumb")
    ax.plot(hours, B.soc, color=COL["smart"], lw=2.0, ls="-",  label="B - Smart")
    ax.plot(hours, C.soc, color=COL["milp"],  lw=2.0, ls="-",  label="C - MILP")
    ax.plot(hours, D.soc, color=COL["mpc"],   lw=1.8, ls="--", label="D - MPC perfect")
    ax.plot(hours, E.soc, color=COL["mpc_n"], lw=1.5, ls=":",  label="E - MPC noisy")
    ax.fill_between(hours, C.plugged * 4 + 57, 57, alpha=0.07, color="green", label="Plugged window")
    ax.set_title("(3) Battery State of Charge"); ax.set_xlabel("Hour"); ax.set_ylabel("kWh")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)

    # (4) Daily cost comparison
    ax = axes[1, 0]
    labels = ["A\nDumb", "B\nSmart\n(no V2G)", "C\nMILP\nDay-Ahead",
              "D\nMPC\nPerfect", "E\nMPC\nNoisy"]
    costs  = [A.cost_eur_day, B.cost_eur_day, C.cost_eur_day,
               D.cost_eur_day, E.cost_eur_day]
    colors = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"], COL["mpc_n"]]
    bars   = ax.bar(labels, costs, color=colors, alpha=0.85, edgecolor="black")
    ref    = A.cost_eur_day
    for bar, v in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, max(v, 0) + 0.05,
                f"E{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for i, (bar, v) in enumerate(zip(bars[1:], costs[1:]), 1):
        saving = ref - v
        col = "darkgreen" if saving > 0 else "red"
        ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                f"{saving:+.2f}/day", ha="center", fontsize=7, color=col)
    ax.axhline(ref, color=COL["dumb"], ls="--", lw=1, alpha=0.5)
    ax.set_title("(4) Net Daily Cost (EUR)  [+ = cost, - = revenue]")
    ax.set_ylabel("EUR / day"); ax.grid(True, alpha=0.3, axis="y")

    # (5) Degradation sensitivity
    ax = axes[1, 1]; ax2 = ax.twinx()
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["NetCost_EUR_day"],
            "o-", color=COL["milp"], lw=2, label="Net Cost (EUR/day)")
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["V2G_Rev_EUR_day"],
            "s--", color=COL["mpc"], lw=2, label="V2G Revenue (EUR/day)")
    ax2.bar(deg_df["DegCost_EUR_kWh"], deg_df["V2G_kWh_day"],
            width=0.008, color=COL["mpc_n"], alpha=0.3, label="V2G kWh/day")
    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    if not np.isnan(tipping):
        ax.axvline(tipping, color="red", ls=":", lw=1.5,
                   label=f"V2G cutoff ~{tipping:.3f} EUR/kWh")
    ax.axvline(0.07, color="black", ls="--", lw=1, label="Default 0.07")
    ax.set_title("(5) Degradation Sensitivity (Agora 2025)")
    ax.set_xlabel("Degradation cost (EUR/kWh cycled)"); ax.set_ylabel("EUR / day")
    ax2.set_ylabel("V2G export (kWh/day)", color=COL["mpc_n"])
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (6) Fleet scaling
    ax = axes[1, 2]; ax2 = ax.twinx()
    x  = np.arange(len(fleet_df)); w = 0.35
    ax.bar(x - w/2, fleet_df["MILP_Annual_Rev_EUR"] / 1e3, width=w,
           color=COL["milp"], alpha=0.8, label="MILP Annual Revenue (kEUR)")
    ax.bar(x + w/2, fleet_df["MPC_Annual_Rev_EUR"]  / 1e3, width=w,
           color=COL["mpc"],  alpha=0.8, label="MPC Annual Revenue (kEUR)")
    ax2.plot(x, fleet_df["Annual_V2G_MWh"], "D-",
             color=COL["mpc_n"], lw=2, label="Annual V2G MWh")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in fleet_df["Fleet_n"]])
    ax.set_title("(6) Fleet Scaling"); ax.set_xlabel("Fleet size (trailers)")
    ax.set_ylabel("Annual Revenue (kEUR)")
    ax2.set_ylabel("MWh / year", color=COL["mpc_n"])
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="center right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results, fleet_df, deg_df, season="winter", price_source=""):
    ref = list(results.values())[0].cost_eur_day
    print("\n" + "="*84)
    print(f"  RESULTS — {season.upper()} WEEKDAY  |  {price_source[:60]}")
    print("="*84)
    print(f"  {'Scenario':<32} {'Net EUR/day':>10} {'Charge EUR':>10} "
          f"{'V2G Rev EUR':>11} {'Deg EUR':>8} {'V2G kWh':>8} {'vs Dumb':>10}")
    print("-"*84)
    for r in results.values():
        print(f"  {r.scenario:<32} {r.cost_eur_day:>10.4f} "
              f"{r.charge_cost_eur_day:>10.4f} {r.v2g_revenue_eur_day:>11.4f} "
              f"{r.deg_cost_eur_day:>8.4f} {r.v2g_export_kwh_day:>8.2f} "
              f"  {ref - r.cost_eur_day:>+.4f}")
    print("="*84)

    print("\n  Annualised (365 days per trailer):")
    for r in results.values():
        s = (ref - r.cost_eur_day) * 365
        print(f"    {r.scenario:<32}  annual cost: EUR {r.cost_eur_day*365:>8,.0f}"
              f"   savings vs Dumb: EUR {s:>+,.0f}/yr")

    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    print(f"\n  Degradation tipping point: V2G profitable up to "
          f"~EUR {tipping:.3f}/kWh  "
          f"({'OK - default 0.07 is below cutoff' if 0.07 <= tipping else 'WARN - default above cutoff'})")

    print("\n  Fleet Scaling:")
    print(fleet_df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("  S.KOe COOL  -  Day-Ahead MILP + Receding-Horizon MPC")
    print("  Schmitz Cargobull AG  |  2025")
    print("  Based on: Biedenbach & Strunz (2024), Agora V'wende (2025)")
    print("="*65)

    # ── Step 1: Load battery parameters ───────────────────────────────────────
    v2g_default = V2GParams(mpc_price_noise_std=0.012)
    v2g, batt_source = load_battery_params(v2g_default)
    print(f"\n  Battery params:  {batt_source}")
    print(f"  Pack: {v2g.battery_capacity_kWh} kWh total, "
          f"{v2g.usable_capacity_kWh} kWh usable")
    print(f"  Charge: {v2g.charge_power_kW} kW  |  "
          f"V2G discharge: {v2g.discharge_power_kW} kW  |  "
          f"Deg cost: EUR {v2g.deg_cost_eur_kwh}/kWh")

    soc_init_pct  = 45.0   # trailer arrives at 45% SoC (winter average)
    soc_final_pct = 80.0   # must depart at 80% SoC
    dwell         = "Extended"
    tru, plugged  = build_load_and_availability(v2g, dwell=dwell)
    hours         = np.arange(v2g.n_slots) * v2g.dt_h
    deg_values    = load_deg_sensitivity(v2g)

    # ══════════════════════════════════════════════════════════════════════════
    #  Run both WINTER and SUMMER seasons
    # ══════════════════════════════════════════════════════════════════════════
    all_season_results = {}

    # ── Day types: weekday + weekend, both seasons ─────────────────────────
    # Weekly structure: 5 weekdays + 2 weekend days
    # Annual: 261 weekdays (130 winter + 131 summer approx) + 104 weekend days
    DAY_TYPES = [
        ("winter",         "Extended", 130, "Winter weekday  (Mon-Fri, Oct-Mar)"),
        ("summer",         "Extended", 131, "Summer weekday  (Mon-Fri, Apr-Sep)"),
        ("winter_weekend", "Weekend",   52, "Winter weekend  (Sat-Sun, Oct-Mar)"),
        ("summer_weekend", "Weekend",   52, "Summer weekend  (Sat-Sun, Apr-Sep)"),
    ]

    annual_cost_total = 0.0
    annual_v2g_total  = 0.0
    annual_savings_vs_dumb = 0.0

    for season, dwell_type, days_per_year, label in DAY_TYPES:
        print(f"\n{'='*65}")
        print(f"  {label}  ({days_per_year} days/year)")
        print(f"{'='*65}")

        tru, plugged = build_load_and_availability(v2g, dwell=dwell_type)
        buy, v2g_p, price_source = load_prices(v2g, season=season)

        print(f"  Prices: {price_source}")
        print(f"  Buy: EUR {buy.min():.3f}-{buy.max():.3f}/kWh  |  "
              f"V2G peak: EUR {v2g_p.max():.3f}/kWh  |  "
              f"Plugged: {int(plugged.sum()*v2g.dt_h)}h/day")

        A = run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        B = run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        C = run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        D = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct,
                              forecast_noise_std=0.0, label="D - MPC perfect")
        E = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct,
                              forecast_noise_std=v2g.mpc_price_noise_std,
                              label="E - MPC noisy", seed=42)

        results  = {"A": A, "B": B, "C": C, "D": D, "E": E}
        deg_df   = deg_sensitivity(v2g, buy, v2g_p, tru, plugged, deg_values, soc_init_pct, soc_final_pct)
        fleet_df = fleet_scaling(C, D, fleet_sizes=[1, 5, 10, 25, 50])

        print_report(results, fleet_df, deg_df, season=label, price_source=price_source)

        out_png = f"results_{season}.png"
        hours   = np.arange(v2g.n_slots) * v2g.dt_h
        plot_all(hours, A, B, C, D, E, deg_df, fleet_df, season=label, out=out_png)

        annual_cost_total        += C.cost_eur_day        * days_per_year
        annual_v2g_total         += C.v2g_revenue_eur_day * days_per_year
        annual_savings_vs_dumb   += (A.cost_eur_day - C.cost_eur_day) * days_per_year
        all_season_results[season] = results

    # ── Annual summary across all day types ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ANNUAL SUMMARY — Single Trailer (Scenario C MILP)")
    print(f"{'='*65}")
    print(f"  Annual energy cost (MILP):        EUR {annual_cost_total:>8,.0f}/year")
    print(f"  Annual V2G revenue (MILP):        EUR {annual_v2g_total:>8,.0f}/year")
    print(f"  Annual savings vs Dumb charging:  EUR {annual_savings_vs_dumb:>8,.0f}/year")
    print(f"  [Agora 2025 benchmark for car:    EUR ~500/year for arbitrage only]")
    print()

if __name__ == "__main__":
    main()
