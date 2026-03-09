#!/usr/bin/env python3
"""
debug_tool.py
═══════════════════════════════════════════════════════════════════════════════
S.KOe COOL  —  V2G Debug & Validation Tool
Schmitz Cargobull AG × TU Dortmund  |  Kuldip Bhadreshvara, 2026

HOW TO RUN:
    python debug_tool.py
    → opens browser automatically at http://localhost:PORT

DEPENDENCIES:
    pip install numpy scipy matplotlib pandas openpyxl
    (same as run_optimisation.py — no extras needed)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import json
import sys
import os
import io
import base64
import traceback
import threading
import webbrowser
import warnings
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# ── Try to import the optimisation engine ─────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_optimisation import (
        V2GParams, V2GResult,
        load_battery_params, load_prices, build_load_and_availability,
        run_dumb, run_smart_no_v2g, run_milp_day_ahead, run_mpc_day_ahead,
        deg_sensitivity, fleet_scaling, plot_all,
    )
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMPORT_OK = True
    IMPORT_ERROR = ""
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = traceback.format_exc()
    import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pass(msg):  return {"status": "PASS", "msg": msg}
def _warn(msg):  return {"status": "WARN", "msg": msg}
def _fail(msg):  return {"status": "FAIL", "msg": msg}
def _info(msg):  return {"status": "INFO", "msg": msg}


def validate_battery(v2g):
    checks = []
    checks.append(_pass(f"Battery capacity: {v2g.battery_capacity_kWh} kWh")
                  if 50 <= v2g.battery_capacity_kWh <= 150
                  else _fail(f"Battery capacity {v2g.battery_capacity_kWh} kWh outside expected 50–150 kWh range"))

    usable_pct = v2g.usable_capacity_kWh / v2g.battery_capacity_kWh * 100
    checks.append(_pass(f"Usable capacity: {v2g.usable_capacity_kWh} kWh ({usable_pct:.1f}% of total)")
                  if 50 <= usable_pct <= 90
                  else _warn(f"Usable fraction {usable_pct:.1f}% — expected 50–90%"))

    checks.append(_pass(f"E_min = {v2g.E_min:.1f} kWh  (SOC_min {v2g.soc_min_pct}% × usable)")
                  if v2g.E_min > 0
                  else _fail("E_min is zero or negative"))

    checks.append(_pass(f"E_max = {v2g.E_max:.1f} kWh  (SOC_max {v2g.soc_max_pct}% × usable)")
                  if v2g.E_max > v2g.E_min
                  else _fail("E_max must be greater than E_min"))

    rte = v2g.eta_charge * v2g.eta_discharge
    checks.append(_pass(f"Round-trip efficiency: {rte*100:.1f}%  (η_c={v2g.eta_charge}, η_d={v2g.eta_discharge})")
                  if 0.70 <= rte <= 0.95
                  else _warn(f"Round-trip efficiency {rte*100:.1f}% — expected 70–95%"))

    checks.append(_pass(f"Charge power: {v2g.charge_power_kW} kW (ISO 15118 AC Mode 3: 22 kW)")
                  if 10 <= v2g.charge_power_kW <= 50
                  else _warn(f"Charge power {v2g.charge_power_kW} kW — expected 10–50 kW"))

    checks.append(_pass(f"Degradation cost: €{v2g.deg_cost_eur_kwh}/kWh (Agora 2025: ~€0.07/kWh)")
                  if 0.01 <= v2g.deg_cost_eur_kwh <= 0.20
                  else _warn(f"Degradation cost €{v2g.deg_cost_eur_kwh}/kWh outside typical 0.01–0.20 range"))
    return checks


def validate_prices(buy, v2g_p, season):
    checks = []
    checks.append(_pass(f"Price array length: {len(buy)} slots (expected 96 = 24h × 4)")
                  if len(buy) == 96
                  else _fail(f"Price array has {len(buy)} slots, expected 96"))

    checks.append(_pass(f"Buy price range: €{buy.min():.3f}–€{buy.max():.3f}/kWh (typical German: €0.05–0.45)")
                  if 0.01 <= buy.min() and buy.max() <= 0.60
                  else _warn(f"Buy prices €{buy.min():.3f}–€{buy.max():.3f}/kWh outside expected range"))

    spread = buy.max() - buy.min()
    checks.append(_pass(f"Daily price spread: €{spread:.3f}/kWh — arbitrage opportunity exists")
                  if spread >= 0.05
                  else _warn(f"Daily spread only €{spread:.3f}/kWh — very little V2G arbitrage opportunity"))

    premium = (v2g_p - buy).max()
    checks.append(_pass(f"Peak V2G premium: +€{premium:.3f}/kWh (FCR/aFRR on top of buy price)")
                  if premium >= 0.05
                  else _warn(f"V2G premium only €{premium:.3f}/kWh — check FCR/aFRR values in make_data.py"))

    neg_prices = (buy < 0).sum()
    checks.append(_info(f"Negative price slots: {neg_prices}/96 — charging is free/paid during these")
                  if neg_prices > 0
                  else _pass("No negative prices in this profile"))

    if season == "summer":
        midday_buy = buy[44:60].mean()
        checks.append(_pass(f"Summer midday avg: €{midday_buy:.3f}/kWh (solar suppression expected < €0.15)")
                      if midday_buy < 0.18
                      else _warn(f"Summer midday price €{midday_buy:.3f}/kWh — solar suppression not visible"))
    return checks


def validate_tru(tru, plugged, v2g):
    checks = []
    tru_total = float(np.sum(tru) * v2g.dt_h)
    checks.append(_pass(f"TRU daily energy: {tru_total:.1f} kWh (expected 40–80 kWh/day for reefer)")
                  if 30 <= tru_total <= 100
                  else _warn(f"TRU energy {tru_total:.1f} kWh/day outside expected 30–100 kWh range"))

    plugged_h = float(np.sum(plugged) * v2g.dt_h)
    checks.append(_pass(f"Plugged-in hours: {plugged_h:.0f}h/day (Extended dwell = 16h)")
                  if 8 <= plugged_h <= 24
                  else _warn(f"Only {plugged_h:.0f}h plugged in — may limit V2G revenue"))

    tru_min = float(tru.min())
    tru_max = float(tru.max())
    checks.append(_pass(f"TRU load range: {tru_min:.2f}–{tru_max:.2f} kW (sinusoidal around 2.8 kW)")
                  if 1.0 <= tru_min and tru_max <= 6.0
                  else _warn(f"TRU load {tru_min:.2f}–{tru_max:.2f} kW outside expected 1–6 kW"))
    return checks


def validate_energy_balance(v2g, tru, soc_init_pct=45.0, soc_final_pct=80.0):
    checks = []
    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0
    E_fin  = v2g.usable_capacity_kWh * soc_final_pct / 100.0
    tru_total = float(np.sum(tru) * v2g.dt_h)
    soc_delta = E_fin - E_init
    losses_estimate = (tru_total + soc_delta) * 0.10
    expected_purchase = tru_total + soc_delta + losses_estimate

    checks.append(_info(f"E_init (arrival SOC {soc_init_pct}%): {E_init:.1f} kWh"))
    checks.append(_info(f"E_final (departure SOC {soc_final_pct}%): {E_fin:.1f} kWh"))
    checks.append(_info(f"TRU energy needed today: {tru_total:.1f} kWh"))
    checks.append(_info(f"SOC top-up needed: {soc_delta:.1f} kWh"))
    checks.append(_info(f"Efficiency losses (~10%): {losses_estimate:.1f} kWh"))
    checks.append(_pass(f"Expected grid purchase (Scenario A baseline): ~{expected_purchase:.0f} kWh/day")
                  if 50 <= expected_purchase <= 150
                  else _warn(f"Expected purchase {expected_purchase:.0f} kWh/day — check battery size and TRU load"))

    if E_init < v2g.E_min:
        checks.append(_fail(f"Arrival SOC ({E_init:.1f} kWh) is BELOW E_min ({v2g.E_min:.1f} kWh) — cold-chain violation at arrival!"))
    else:
        checks.append(_pass(f"Arrival SOC ({E_init:.1f} kWh) >= E_min ({v2g.E_min:.1f} kWh) — cold-chain safe at arrival"))

    if E_fin > v2g.E_max:
        checks.append(_fail(f"Target departure SOC ({E_fin:.1f} kWh) EXCEEDS E_max ({v2g.E_max:.1f} kWh) — impossible target!"))
    else:
        checks.append(_pass(f"Target departure SOC ({E_fin:.1f} kWh) <= E_max ({v2g.E_max:.1f} kWh) — feasible"))
    return checks, expected_purchase


def validate_scenario_result(r, v2g, label=""):
    checks = []
    soc_min_actual = float(np.min(r.soc))
    soc_max_actual = float(np.max(r.soc))
    checks.append(_pass(f"SOC never below E_min: min={soc_min_actual:.2f} kWh >= {v2g.E_min:.2f} kWh")
                  if soc_min_actual >= v2g.E_min - 0.05
                  else _fail(f"SOC VIOLATED minimum! min={soc_min_actual:.2f} kWh < E_min={v2g.E_min:.2f} kWh"))
    checks.append(_pass(f"SOC never above E_max: max={soc_max_actual:.2f} kWh <= {v2g.E_max:.2f} kWh")
                  if soc_max_actual <= v2g.E_max + 0.05
                  else _fail(f"SOC VIOLATED maximum! max={soc_max_actual:.2f} kWh > E_max={v2g.E_max:.2f} kWh"))

    soc_final = float(r.soc[-1])
    checks.append(_pass(f"Departure SOC met: {soc_final:.2f} kWh (target >= {v2g.usable_capacity_kWh*0.8:.2f} kWh)")
                  if soc_final >= v2g.usable_capacity_kWh * 0.78
                  else _warn(f"Departure SOC {soc_final:.2f} kWh may be below 80% target"))

    simultaneous = int(np.sum((r.p_charge > 0.1) & (r.p_discharge > 0.1)))
    checks.append(_pass(f"No simultaneous charge+discharge (mutex OK)")
                  if simultaneous == 0
                  else _fail(f"Mutex VIOLATED: {simultaneous} slots have both charge AND discharge > 0.1 kW"))

    if r.v2g_revenue_eur_day > 0:
        annual = r.v2g_revenue_eur_day * 365
        checks.append(_pass(f"V2G annual extrapolation: EUR {annual:.0f}/yr (Agora benchmark: EUR 500/yr for car)")
                      if 200 <= annual <= 3000
                      else _warn(f"Annual V2G revenue EUR {annual:.0f}/yr — outside expected EUR 200-3000 range"))
    return checks


def validate_scenario_ordering(results):
    checks = []
    costs = {k: results[k].cost_eur_day for k in results}

    if "A" in costs and "B" in costs:
        checks.append(_pass(f"A(EUR {costs['A']:.3f}) >= B(EUR {costs['B']:.3f}) — Smart beats Dumb")
                      if costs["A"] >= costs["B"] - 0.001
                      else _fail(f"ORDERING BROKEN: B(EUR {costs['B']:.3f}) > A(EUR {costs['A']:.3f}) — Smart charging is WORSE than Dumb!"))
    if "B" in costs and "C" in costs:
        checks.append(_pass(f"B(EUR {costs['B']:.3f}) >= C(EUR {costs['C']:.3f}) — V2G beats Smart-only")
                      if costs["B"] >= costs["C"] - 0.001
                      else _warn(f"B(EUR {costs['B']:.3f}) < C(EUR {costs['C']:.3f}) — V2G not beating Smart-only (high deg cost?)"))
    if "C" in costs and "D" in costs:
        gap_pct = abs(costs["D"] - costs["C"]) / (abs(costs["C"]) + 0.001) * 100
        checks.append(_pass(f"MPC gap vs MILP: {gap_pct:.1f}% — receding horizon works well (< 10% expected)")
                      if gap_pct <= 10
                      else _warn(f"MPC gap {gap_pct:.1f}% vs MILP — larger than expected 10%; check MPC horizon logic"))
    if "D" in costs and "E" in costs:
        checks.append(_pass(f"D(EUR {costs['D']:.3f}) <= E(EUR {costs['E']:.3f}) — Noise degrades MPC as expected")
                      if costs["D"] <= costs["E"] + 0.001
                      else _warn(f"E(EUR {costs['E']:.3f}) <= D(EUR {costs['D']:.3f}) — Noisy MPC outperforming perfect? Check noise seed"))
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_charging_schedule(hours, results):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"A": "#AAAAAA", "B": "#2196F3", "C": "#00BCD4", "D": "#FF7700", "E": "#CC3300"}
    labels = {"A": "A - Dumb", "B": "B - Smart (no V2G)", "C": "C - MILP Day-Ahead",
              "D": "D - MPC Perfect", "E": "E - MPC Noisy"}
    for k, r in results.items():
        ax.fill_between(hours, r.p_charge, step="pre", color=colors.get(k, "#999"),
                        alpha=0.5, label=labels.get(k, k))
    if results:
        first = list(results.values())[0]
        ax.step(hours, first.tru_load, where="post", color="#AA0000",
                lw=1.5, ls="--", label="TRU refrigeration load")
    ax.set_title("Charging Power Schedule — all selected scenarios", fontsize=12)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)
    plt.tight_layout()
    b64 = _fig_to_b64(fig); plt.close(); return b64


def plot_v2g_vs_price(hours, results):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twinx()
    colors = {"C": "#00BCD4", "D": "#FF7700", "E": "#CC3300"}
    labels = {"C": "C - MILP", "D": "D - MPC Perfect", "E": "E - MPC Noisy"}
    w = 0.22; offsets = {"C": -w, "D": 0, "E": w}
    for k, r in results.items():
        if k in colors and r.v2g_export_kwh_day > 0.01:
            ax.bar(hours + offsets.get(k, 0), r.p_discharge, width=w,
                   color=colors[k], alpha=0.75, label=labels[k])
    if results:
        first = list(results.values())[0]
        ax2.step(hours, first.price_v2g, where="post", color="#007700", lw=2, label="V2G price")
        ax2.set_ylabel("Price (EUR/kWh)", color="#007700")
    ax.set_title("V2G Discharge vs Price — discharge should align with price peaks", fontsize=12)
    ax.set_xlabel("Hour"); ax.set_ylabel("Discharge power (kW)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)
    plt.tight_layout()
    b64 = _fig_to_b64(fig); plt.close(); return b64


def plot_soc_traces(hours, results, v2g):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"A": "#AAAAAA", "B": "#2196F3", "C": "#00BCD4", "D": "#FF7700", "E": "#CC3300"}
    labels = {"A": "A - Dumb", "B": "B - Smart", "C": "C - MILP",
              "D": "D - MPC Perfect", "E": "E - MPC Noisy"}
    for k, r in results.items():
        ax.plot(hours, r.soc, color=colors.get(k, "#999"), lw=2, label=labels.get(k, k))
    ax.axhline(v2g.E_min, color="red", ls="--", lw=1.5,
               label=f"E_min = {v2g.E_min:.1f} kWh (cold-chain floor)")
    ax.axhline(v2g.E_max, color="orange", ls="--", lw=1.5,
               label=f"E_max = {v2g.E_max:.1f} kWh (charge ceiling)")
    ax.set_title("Battery State of Charge — must stay between red dashed lines at ALL times", fontsize=12)
    ax.set_xlabel("Hour"); ax.set_ylabel("Energy in battery (kWh)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)
    plt.tight_layout()
    b64 = _fig_to_b64(fig); plt.close(); return b64


def plot_cost_bars(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"A": "#AAAAAA", "B": "#2196F3", "C": "#00BCD4", "D": "#FF7700", "E": "#CC3300"}
    labels = {"A": "A\nDumb", "B": "B\nSmart\n(no V2G)", "C": "C\nMILP\nDay-Ahead",
              "D": "D\nMPC\nPerfect", "E": "E\nMPC\nNoisy"}
    ks = list(results.keys())
    vals = [results[k].cost_eur_day for k in ks]
    cols = [colors.get(k, "#999") for k in ks]
    xlabs = [labels.get(k, k) for k in ks]
    bars = ax.bar(xlabs, vals, color=cols, alpha=0.85, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, max(v, 0) + 0.02,
                f"EUR {v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Net Daily Cost per Scenario — lower is better; C/D/E can be negative (profit)", fontsize=11)
    ax.set_ylabel("Net cost (EUR/day)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    b64 = _fig_to_b64(fig); plt.close(); return b64


def plot_price_profile(hours, buy, v2g_p, season):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(hours, buy, step="pre", alpha=0.4, color="#2196F3", label="Buy price (all-in)")
    ax.step(hours, v2g_p, where="pre", color="#FF7700", lw=2,
            label="V2G sell price (incl. FCR premium)")
    ax.fill_between(hours, buy, v2g_p, step="pre", alpha=0.25, color="#FF7700",
                    label="FCR/aFRR premium (V2G extra revenue)")
    ax.set_title(f"Price Profile — {season.capitalize()} Weekday", fontsize=12)
    ax.set_xlabel("Hour"); ax.set_ylabel("EUR/kWh")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 24)
    plt.tight_layout()
    b64 = _fig_to_b64(fig); plt.close(); return b64


def plot_deg_sensitivity(deg_df):
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(deg_df["DegCost_EUR_kWh"], deg_df["NetCost_EUR_day"],
             "b-o", lw=2, ms=6, label="Net cost/day")
    ax2.plot(deg_df["DegCost_EUR_kWh"], deg_df["V2G_Rev_EUR_day"],
             "r--s", lw=2, ms=6, label="V2G revenue/day")
    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max() if deg_df["V2G_active"].any() else None
    if tipping:
        ax1.axvline(tipping, color="green", ls=":", lw=2, label=f"V2G cutoff ~EUR {tipping:.3f}/kWh")
    ax1.set_title("Degradation Sensitivity — at what battery wear cost does V2G stop being worth it?", fontsize=11)
    ax1.set_xlabel("Degradation cost (EUR/kWh cycled)")
    ax1.set_ylabel("Net cost (EUR/day)", color="blue")
    ax2.set_ylabel("V2G revenue (EUR/day)", color="red")
    ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    b64 = _fig_to_b64(fig); plt.close(); return b64


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE DEBUG RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_debug(config: dict) -> dict:
    out = {"steps": [], "error": None}

    if not IMPORT_OK:
        out["error"] = f"Cannot import run_optimisation.py:\n{IMPORT_ERROR}"
        return out

    season        = config.get("season", "winter")
    soc_init      = float(config.get("soc_init", 45.0))
    soc_fin       = float(config.get("soc_fin",  80.0))
    dwell         = config.get("dwell", "Extended")
    run_scenarios = config.get("scenarios", ["A"])
    run_plots     = config.get("plots", [])
    run_steps     = config.get("steps", [])

    v2g = V2GParams()

    try:
        # ── STEP 1: Battery Parameters ─────────────────────────────────────
        if "battery" in run_steps:
            v2g, v2g_src = load_battery_params(V2GParams())
            checks = validate_battery(v2g)
            params_table = {
                "Battery capacity (kWh)":   v2g.battery_capacity_kWh,
                "Usable capacity (kWh)":    v2g.usable_capacity_kWh,
                "SOC min (%)":              v2g.soc_min_pct,
                "SOC max (%)":              v2g.soc_max_pct,
                "E_min (kWh)":              round(v2g.E_min, 2),
                "E_max (kWh)":              round(v2g.E_max, 2),
                "Charge power (kW)":        v2g.charge_power_kW,
                "Discharge power (kW)":     v2g.discharge_power_kW,
                "eta_charge":               v2g.eta_charge,
                "eta_discharge":            v2g.eta_discharge,
                "Round-trip efficiency":    round(v2g.eta_charge * v2g.eta_discharge, 4),
                "Degradation cost (EUR/kWh)": v2g.deg_cost_eur_kwh,
            }
            out["steps"].append({
                "id": "battery", "title": "Step 1 — Battery Parameters",
                "source": v2g_src, "params": params_table, "checks": checks,
            })

        # ── STEP 2: Price Data ─────────────────────────────────────────────
        if "prices" in run_steps:
            buy, v2g_p, price_src = load_prices(v2g, season=season)
            hours_arr = np.arange(96) * 0.25
            checks = validate_prices(buy, v2g_p, season)
            price_table = []
            for i in range(0, 96, 4):
                h = i * 0.25
                price_table.append({
                    "Time":             f"{int(h):02d}:00",
                    "Buy (EUR/kWh)":    round(float(buy[i]), 4),
                    "V2G (EUR/kWh)":    round(float(v2g_p[i]), 4),
                    "Premium (EUR/kWh)":round(float(v2g_p[i] - buy[i]), 4),
                })
            price_summary = {
                "Buy min (EUR/kWh)":        round(float(buy.min()), 4),
                "Buy max (EUR/kWh)":        round(float(buy.max()), 4),
                "Buy avg (EUR/kWh)":        round(float(buy.mean()), 4),
                "Daily spread (EUR/kWh)":   round(float(buy.max() - buy.min()), 4),
                "V2G max (EUR/kWh)":        round(float(v2g_p.max()), 4),
                "Max FCR premium (EUR/kWh)":round(float((v2g_p - buy).max()), 4),
            }
            step = {
                "id": "prices",
                "title": f"Step 2 — Price Data ({season.capitalize()} Weekday)",
                "source": price_src, "summary": price_summary,
                "table": price_table, "checks": checks,
            }
            if "price_profile" in run_plots:
                step["plot"] = plot_price_profile(hours_arr, buy, v2g_p, season)
            out["steps"].append(step)

        # ── STEP 3: TRU Load & Availability ───────────────────────────────
        if "tru" in run_steps:
            tru, plugged = build_load_and_availability(v2g, dwell=dwell)
            checks = validate_tru(tru, plugged, v2g)
            tru_table = []
            for i in range(0, 96, 4):
                h = i * 0.25
                tru_table.append({
                    "Time":     f"{int(h):02d}:00",
                    "TRU (kW)": round(float(tru[i]), 3),
                    "Plugged":  "Yes" if plugged[i] > 0.5 else "No",
                })
            out["steps"].append({
                "id": "tru", "title": "Step 3 — TRU Load & Plug-in Availability",
                "dwell": dwell,
                "plugged_hours": round(float(np.sum(plugged) * v2g.dt_h), 1),
                "tru_total_kwh": round(float(np.sum(tru) * v2g.dt_h), 2),
                "tru_min_kw":    round(float(tru.min()), 3),
                "tru_max_kw":    round(float(tru.max()), 3),
                "table": tru_table, "checks": checks,
            })

        # ── STEP 4: Energy Balance Pre-flight ─────────────────────────────
        if "energy_balance" in run_steps:
            if "tru" not in run_steps:
                tru, plugged = build_load_and_availability(v2g, dwell=dwell)
            checks, expected_purchase = validate_energy_balance(v2g, tru, soc_init, soc_fin)
            out["steps"].append({
                "id": "energy_balance",
                "title": "Step 4 — Energy Balance Pre-flight Check",
                "expected_purchase_kwh": round(expected_purchase, 1),
                "checks": checks,
            })

        # ── STEP 5: Run Selected Scenarios ────────────────────────────────
        if run_scenarios:
            if "prices" not in run_steps:
                buy, v2g_p, price_src = load_prices(v2g, season=season)
            if "tru" not in run_steps:
                tru, plugged = build_load_and_availability(v2g, dwell=dwell)

            hours_arr = np.arange(96) * 0.25
            scenario_results = {}
            scenario_step = {
                "id": "scenarios",
                "title": f"Step 5 — Scenario Results ({season.capitalize()})",
                "scenarios": [],
            }

            scenario_fns = {
                "A": lambda: run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init, soc_fin),
                "B": lambda: run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged, soc_init, soc_fin),
                "C": lambda: run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init, soc_fin),
                "D": lambda: run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init, soc_fin,
                                               forecast_noise_std=0.0, label="D - MPC perfect"),
                "E": lambda: run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init, soc_fin,
                                               forecast_noise_std=0.012, label="E - MPC noisy"),
            }
            scenario_names = {
                "A": "A — Dumb (baseline)", "B": "B — Smart, no V2G",
                "C": "C — MILP Day-Ahead (optimal)", "D": "D — MPC Perfect forecast",
                "E": "E — MPC Noisy forecast",
            }

            for k in ["A", "B", "C", "D", "E"]:
                if k not in run_scenarios:
                    continue
                r = scenario_fns[k]()
                scenario_results[k] = r
                per_checks = validate_scenario_result(r, v2g, label=k)
                scenario_step["scenarios"].append({
                    "key": k, "name": scenario_names[k],
                    "net_cost_eur_day":    round(r.cost_eur_day, 4),
                    "charge_cost_eur_day": round(r.charge_cost_eur_day, 4),
                    "v2g_revenue_eur_day": round(r.v2g_revenue_eur_day, 4),
                    "deg_cost_eur_day":    round(r.deg_cost_eur_day, 4),
                    "v2g_export_kwh_day":  round(r.v2g_export_kwh_day, 2),
                    "annual_cost_eur":     round(r.cost_eur_day * 365, 0),
                    "annual_v2g_rev_eur":  round(r.v2g_revenue_eur_day * 365, 0),
                    "soc_min_kwh":         round(float(np.min(r.soc)), 2),
                    "soc_max_kwh":         round(float(np.max(r.soc)), 2),
                    "soc_final_kwh":       round(float(r.soc[-1]), 2),
                    "total_charge_kwh":    round(float(np.sum(r.p_charge) * v2g.dt_h), 2),
                    "checks": per_checks,
                })

            if len(scenario_results) >= 2:
                scenario_step["ordering_checks"] = validate_scenario_ordering(scenario_results)

            out["steps"].append(scenario_step)

            # ── Plots ──────────────────────────────────────────────────────
            if scenario_results:
                plots_step = {"id": "plots", "title": "Step 6 — Plots", "plots": []}
                if "charging" in run_plots:
                    plots_step["plots"].append({
                        "name": "Charging Schedule",
                        "img":  plot_charging_schedule(hours_arr, scenario_results)})
                if "v2g_price" in run_plots:
                    plots_step["plots"].append({
                        "name": "V2G Discharge vs Price",
                        "img":  plot_v2g_vs_price(hours_arr, scenario_results)})
                if "soc" in run_plots:
                    plots_step["plots"].append({
                        "name": "State of Charge Traces",
                        "img":  plot_soc_traces(hours_arr, scenario_results, v2g)})
                if "cost_bars" in run_plots:
                    plots_step["plots"].append({
                        "name": "Daily Cost Comparison",
                        "img":  plot_cost_bars(scenario_results)})
                if "deg_sensitivity" in run_plots and "C" in scenario_results:
                    deg_df = deg_sensitivity(v2g, buy, v2g_p, tru, plugged, soc_init, soc_fin)
                    plots_step["plots"].append({
                        "name": "Degradation Sensitivity",
                        "img":  plot_deg_sensitivity(deg_df)})
                if plots_step["plots"]:
                    out["steps"].append(plots_step)

    except Exception as e:
        out["error"] = traceback.format_exc()

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML PAGE
# ═══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V2G Debug Tool — S.KOe COOL</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#1a202c;min-height:100vh}
  header{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;padding:18px 32px;
    display:flex;align-items:center;gap:16px}
  header h1{font-size:1.4rem;font-weight:700}
  header p{font-size:0.82rem;opacity:0.85;margin-top:3px}
  .badge{background:rgba(255,255,255,0.18);border-radius:6px;padding:3px 10px;
    font-size:0.75rem;font-weight:600;letter-spacing:.5px}
  .layout{display:grid;grid-template-columns:320px 1fr;gap:0;min-height:calc(100vh - 72px)}
  .sidebar{background:#fff;border-right:1px solid #e2e8f0;padding:20px;overflow-y:auto}
  .main{padding:24px;overflow-y:auto}
  .section-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
    padding:16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  .section-title{font-size:0.75rem;font-weight:700;text-transform:uppercase;
    letter-spacing:.7px;color:#64748b;margin-bottom:10px;padding-bottom:6px;
    border-bottom:1px solid #f1f5f9}
  .cb-row{display:flex;align-items:center;gap:8px;padding:4px 0;cursor:pointer}
  .cb-row label{cursor:pointer;font-size:0.875rem;color:#374151;user-select:none}
  input[type=checkbox]{width:15px;height:15px;accent-color:#2563eb;cursor:pointer;flex-shrink:0}
  .field{margin-bottom:10px}
  .field label{display:block;font-size:0.78rem;font-weight:600;color:#6b7280;margin-bottom:3px}
  .field select,.field input{width:100%;border:1px solid #d1d5db;border-radius:6px;
    padding:6px 10px;font-size:0.875rem;color:#1a202c;background:#fafafa}
  .run-btn{width:100%;padding:13px;background:linear-gradient(135deg,#2563eb,#1d4ed8);
    color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:700;
    cursor:pointer;margin-top:6px;transition:all .2s}
  .run-btn:hover{background:linear-gradient(135deg,#1d4ed8,#1e40af);transform:translateY(-1px)}
  .run-btn:disabled{background:#9ca3af;cursor:not-allowed;transform:none}
  .spinner{display:none;text-align:center;padding:40px;color:#6b7280;font-size:1rem}
  .step-block{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
    margin-bottom:18px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.05)}
  .step-header{padding:14px 18px;background:linear-gradient(90deg,#f8fafc,#f1f5f9);
    border-bottom:1px solid #e2e8f0;display:flex;align-items:center;gap:10px}
  .step-header h3{font-size:1rem;font-weight:700;color:#1e3a5f}
  .step-body{padding:18px}
  .source-tag{background:#dbeafe;color:#1e40af;border-radius:4px;padding:2px 8px;
    font-size:0.72rem;font-weight:600}
  table{width:100%;border-collapse:collapse;font-size:0.83rem;margin-top:8px}
  th{background:#f8fafc;text-align:left;padding:7px 10px;font-weight:600;
    color:#374151;border-bottom:2px solid #e2e8f0;white-space:nowrap}
  td{padding:6px 10px;border-bottom:1px solid #f1f5f9;color:#4b5563}
  tr:hover td{background:#f9fafb}
  td.num{text-align:right;font-family:'Courier New',monospace;color:#1e3a5f;font-weight:600}
  .checks{display:flex;flex-direction:column;gap:5px;margin-top:10px}
  .check{display:flex;align-items:flex-start;gap:8px;padding:6px 10px;
    border-radius:6px;font-size:0.825rem;line-height:1.4}
  .check.PASS{background:#f0fdf4;border-left:3px solid #22c55e;color:#166534}
  .check.FAIL{background:#fef2f2;border-left:3px solid #ef4444;color:#991b1b}
  .check.WARN{background:#fffbeb;border-left:3px solid #f59e0b;color:#92400e}
  .check.INFO{background:#eff6ff;border-left:3px solid #3b82f6;color:#1e40af}
  .check-icon{font-size:1rem;flex-shrink:0;margin-top:1px}
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
    gap:10px;margin-bottom:14px}
  .kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px}
  .kpi-val{font-size:1.3rem;font-weight:800;color:#1e3a5f}
  .kpi-label{font-size:0.72rem;color:#64748b;margin-top:2px}
  .kpi.green .kpi-val{color:#16a34a}
  .kpi.red .kpi-val{color:#dc2626}
  .scenario-block{margin-bottom:16px;padding:14px;border:1px solid #e2e8f0;
    border-radius:8px;background:#fafafa}
  .scenario-title{font-size:0.9rem;font-weight:700;color:#1e3a5f;margin-bottom:8px;
    padding-bottom:6px;border-bottom:1px solid #e2e8f0}
  .plot-wrap{margin-top:12px;text-align:center}
  .plot-wrap img{max-width:100%;border-radius:8px;border:1px solid #e2e8f0}
  .plot-title{font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:8px}
  .error-box{background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
    padding:16px;color:#991b1b;font-family:monospace;font-size:0.8rem;
    white-space:pre-wrap;margin-bottom:16px}
  .select-all-btn{font-size:0.72rem;color:#2563eb;cursor:pointer;
    text-decoration:underline;background:none;border:none;padding:0;margin-left:4px}
  #results-placeholder{color:#94a3b8;font-size:0.95rem;text-align:center;padding:60px 20px}
  #results-placeholder .big{font-size:3rem;margin-bottom:12px}
  .summary-table-wrap{overflow-x:auto;margin-top:8px}
  .ordering-section{margin-top:14px;padding:12px;background:#f0f9ff;
    border-radius:8px;border:1px solid #bae6fd}
  .ordering-title{font-size:0.78rem;font-weight:700;color:#0c4a6e;margin-bottom:6px}
</style>
</head>
<body>
<header>
  <div>
    <h1>&#x1F50D; V2G Debug Tool &#x2014; S.KOe COOL Reefer Trailer</h1>
    <p>Schmitz Cargobull AG x TU Dortmund &nbsp;|&nbsp; Kuldip Bhadreshvara, 2026</p>
  </div>
  <div class="badge">DEBUG MODE</div>
</header>

<div class="layout">
  <div class="sidebar">

    <div class="section-card">
      <div class="section-title">&#x1F527; Simulation Settings</div>
      <div class="field">
        <label>Season</label>
        <select id="season">
          <option value="winter" selected>Winter weekday</option>
          <option value="summer">Summer weekday</option>
        </select>
      </div>
      <div class="field">
        <label>Dwell profile</label>
        <select id="dwell">
          <option value="Extended" selected>Extended (16h: night + midday)</option>
          <option value="NightOnly">Night only (10h: 21:00-07:00)</option>
        </select>
      </div>
      <div class="field">
        <label>Arrival SOC (%)</label>
        <input type="number" id="soc_init" value="45" min="10" max="95" step="5">
      </div>
      <div class="field">
        <label>Departure SOC target (%)</label>
        <input type="number" id="soc_fin" value="80" min="20" max="100" step="5">
      </div>
    </div>

    <div class="section-card">
      <div class="section-title">
        &#x1F4CB; Debug Steps
        <button class="select-all-btn" onclick="selectAll('step')">select all</button>
      </div>
      <div class="cb-row"><input type="checkbox" id="step_battery" class="step" checked>
        <label for="step_battery">1 &middot; Battery parameters</label></div>
      <div class="cb-row"><input type="checkbox" id="step_prices" class="step" checked>
        <label for="step_prices">2 &middot; Price data (EPEX)</label></div>
      <div class="cb-row"><input type="checkbox" id="step_tru" class="step" checked>
        <label for="step_tru">3 &middot; TRU load &amp; availability</label></div>
      <div class="cb-row"><input type="checkbox" id="step_energy_balance" class="step" checked>
        <label for="step_energy_balance">4 &middot; Energy balance pre-flight</label></div>
    </div>

    <div class="section-card">
      <div class="section-title">
        &#x26A1; Scenarios to Run
        <button class="select-all-btn" onclick="selectAll('scen')">select all</button>
      </div>
      <div class="cb-row"><input type="checkbox" id="scen_A" class="scen" checked>
        <label for="scen_A">A &#x2014; Dumb (baseline)</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_B" class="scen" checked>
        <label for="scen_B">B &#x2014; Smart, no V2G</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_C" class="scen" checked>
        <label for="scen_C">C &#x2014; MILP Day-Ahead</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_D" class="scen">
        <label for="scen_D">D &#x2014; MPC Perfect forecast</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_E" class="scen">
        <label for="scen_E">E &#x2014; MPC Noisy forecast</label></div>
    </div>

    <div class="section-card">
      <div class="section-title">
        &#x1F4CA; Plots to Show
        <button class="select-all-btn" onclick="selectAll('plot')">select all</button>
      </div>
      <div class="cb-row"><input type="checkbox" id="plot_price_profile" class="plot" checked>
        <label for="plot_price_profile">Price profile (buy + V2G premium)</label></div>
      <div class="cb-row"><input type="checkbox" id="plot_charging" class="plot" checked>
        <label for="plot_charging">Charging schedule</label></div>
      <div class="cb-row"><input type="checkbox" id="plot_soc" class="plot" checked>
        <label for="plot_soc">SOC traces</label></div>
      <div class="cb-row"><input type="checkbox" id="plot_v2g_price" class="plot" checked>
        <label for="plot_v2g_price">V2G discharge vs price</label></div>
      <div class="cb-row"><input type="checkbox" id="plot_cost_bars" class="plot" checked>
        <label for="plot_cost_bars">Daily cost comparison</label></div>
      <div class="cb-row"><input type="checkbox" id="plot_deg_sensitivity" class="plot">
        <label for="plot_deg_sensitivity">Degradation sensitivity</label></div>
    </div>

    <button class="run-btn" id="runBtn" onclick="runDebug()">
      &#x25B6; Run Selected Debug Steps
    </button>
  </div>

  <div class="main" id="resultsArea">
    <div id="results-placeholder">
      <div class="big">&#x1F52C;</div>
      <strong>Select steps and scenarios on the left, then click Run.</strong><br><br>
      Each step shows exactly what data goes in and what comes out,<br>
      with automatic PASS / WARN / FAIL checks at every stage.
    </div>
    <div class="spinner" id="spinner">&#x23F3; Running optimisation &mdash; please wait&hellip;</div>
  </div>
</div>

<script>
function selectAll(cls) {
  document.querySelectorAll('.' + cls).forEach(cb => cb.checked = true);
}

function getChecked(cls) {
  return [...document.querySelectorAll('.' + cls)]
    .filter(cb => cb.checked)
    .map(cb => cb.id.replace('step_','').replace('scen_','').replace('plot_',''));
}

async function runDebug() {
  const btn = document.getElementById('runBtn');
  const spinner = document.getElementById('spinner');
  const area = document.getElementById('resultsArea');
  const cfg = {
    season:    document.getElementById('season').value,
    dwell:     document.getElementById('dwell').value,
    soc_init:  document.getElementById('soc_init').value,
    soc_fin:   document.getElementById('soc_fin').value,
    steps:     getChecked('step'),
    scenarios: getChecked('scen'),
    plots:     getChecked('plot'),
  };
  btn.disabled = true;
  area.innerHTML = '';
  spinner.style.display = 'block';
  area.appendChild(spinner);
  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(cfg)
    });
    const data = await resp.json();
    spinner.style.display = 'none';
    renderResults(data, area);
  } catch(e) {
    spinner.style.display = 'none';
    area.innerHTML = `<div class="error-box">Network error: ${e.message}</div>`;
  }
  btn.disabled = false;
}

const ICONS = {PASS:'[PASS]', FAIL:'[FAIL]', WARN:'[WARN]', INFO:'[INFO]'};
const EMOJIS = {PASS:'✅', FAIL:'❌', WARN:'⚠️', INFO:'ℹ️'};

function checkHtml(checks) {
  if (!checks || !checks.length) return '';
  return '<div class="checks">' +
    checks.map(c => `<div class="check ${c.status}">
      <span class="check-icon">${EMOJIS[c.status]||'•'}</span>
      <span>${c.msg}</span></div>`).join('') + '</div>';
}

function paramsTable(obj) {
  return '<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>' +
    Object.entries(obj).map(([k,v]) =>
      `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join('') +
    '</tbody></table>';
}

function genericTable(rows) {
  if (!rows || !rows.length) return '';
  const keys = Object.keys(rows[0]);
  return '<div class="summary-table-wrap"><table><thead><tr>' +
    keys.map(k => `<th>${k}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + keys.map(k => {
      const v = r[k];
      return `<td${typeof v==='number'?' class="num"':''}>${v}</td>`;
    }).join('') + '</tr>').join('') +
    '</tbody></table></div>';
}

function renderResults(data, area) {
  area.innerHTML = '';
  if (data.error) {
    area.innerHTML = `<div class="error-box">Error:\n\n${data.error}</div>`;
    return;
  }
  if (!data.steps || !data.steps.length) {
    area.innerHTML = '<div class="error-box">No steps ran. Select at least one step.</div>';
    return;
  }

  for (const step of data.steps) {
    const block = document.createElement('div');
    block.className = 'step-block';

    if (step.id === 'battery') {
      block.innerHTML = `
        <div class="step-header">
          <h3>&#x1F50B; ${step.title}</h3>
          <span class="source-tag">${step.source}</span>
        </div>
        <div class="step-body">
          <p style="font-size:.82rem;color:#64748b;margin-bottom:10px">
            These are the values the code actually loaded and uses. Compare against the S.KOe COOL spec sheet.
          </p>
          ${paramsTable(step.params)}
          ${checkHtml(step.checks)}
        </div>`;
    }
    else if (step.id === 'prices') {
      block.innerHTML = `
        <div class="step-header">
          <h3>&#x1F4B6; ${step.title}</h3>
          <span class="source-tag">${step.source.slice(0,70)}</span>
        </div>
        <div class="step-body">
          <p style="font-size:.82rem;color:#64748b;margin-bottom:10px">
            Buy price = EPEX spot + network fees + levies + VAT. V2G price adds FCR/aFRR premium on top.
          </p>
          ${paramsTable(step.summary)}
          <p style="font-size:.78rem;font-weight:600;color:#6b7280;margin:12px 0 4px">
            Hourly price snapshot (every 4th slot = top of hour):
          </p>
          ${genericTable(step.table)}
          ${checkHtml(step.checks)}
          ${step.plot ? `<div class="plot-wrap">
            <div class="plot-title">Price Profile Chart</div>
            <img src="data:image/png;base64,${step.plot}"></div>` : ''}
        </div>`;
    }
    else if (step.id === 'tru') {
      block.innerHTML = `
        <div class="step-header"><h3>&#x1F9CA; ${step.title}</h3></div>
        <div class="step-body">
          <p style="font-size:.82rem;color:#64748b;margin-bottom:10px">
            The refrigeration unit runs continuously regardless of price. This is the non-negotiable energy load.
          </p>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-val">${step.tru_total_kwh} kWh</div>
              <div class="kpi-label">TRU energy today</div></div>
            <div class="kpi"><div class="kpi-val">${step.plugged_hours}h</div>
              <div class="kpi-label">Plugged-in (${step.dwell})</div></div>
            <div class="kpi"><div class="kpi-val">${step.tru_min_kw}&#x2013;${step.tru_max_kw} kW</div>
              <div class="kpi-label">TRU power range</div></div>
          </div>
          ${genericTable(step.table)}
          ${checkHtml(step.checks)}
        </div>`;
    }
    else if (step.id === 'energy_balance') {
      block.innerHTML = `
        <div class="step-header"><h3>&#x2696;&#xFE0F; ${step.title}</h3></div>
        <div class="step-body">
          <p style="font-size:.82rem;color:#64748b;margin-bottom:10px">
            Before running any solver: does the problem make physical sense?
            After running, check that Scenario A total charge matches this expected value.
          </p>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-val">~${step.expected_purchase_kwh} kWh</div>
              <div class="kpi-label">Expected Scenario A grid purchase/day</div></div>
          </div>
          ${checkHtml(step.checks)}
        </div>`;
    }
    else if (step.id === 'scenarios') {
      let html = `<div class="step-header"><h3>&#x26A1; ${step.title}</h3></div><div class="step-body">`;
      if (step.scenarios.length > 1) {
        html += `<p style="font-size:.82rem;color:#64748b;margin-bottom:8px">
          <strong>Cost ordering rule that must always hold:</strong> A &ge; B &ge; C &asymp; D &ge; E</p>
          <div class="summary-table-wrap"><table>
          <thead><tr><th>Scenario</th><th>Net EUR/day</th><th>Charge EUR/day</th>
            <th>V2G Rev EUR/day</th><th>V2G kWh/day</th>
            <th>Annual cost EUR</th><th>Annual V2G EUR</th></tr></thead><tbody>`;
        for (const s of step.scenarios) {
          const green = parseFloat(s.net_cost_eur_day) < 0 ? 'style="color:#16a34a;font-weight:800"' : '';
          html += `<tr>
            <td><strong>${s.key}</strong></td>
            <td class="num" ${green}>${s.net_cost_eur_day}</td>
            <td class="num">${s.charge_cost_eur_day}</td>
            <td class="num" style="color:#16a34a">${s.v2g_revenue_eur_day}</td>
            <td class="num">${s.v2g_export_kwh_day}</td>
            <td class="num">${s.annual_cost_eur}</td>
            <td class="num" style="color:#16a34a">${s.annual_v2g_rev_eur}</td></tr>`;
        }
        html += '</tbody></table></div>';
      }
      for (const s of step.scenarios) {
        html += `<div class="scenario-block">
          <div class="scenario-title">${s.name}</div>
          <div class="kpi-grid">
            <div class="kpi ${parseFloat(s.net_cost_eur_day)<0?'green':''}">
              <div class="kpi-val">EUR ${s.net_cost_eur_day}/day</div>
              <div class="kpi-label">Net cost</div></div>
            <div class="kpi green">
              <div class="kpi-val">EUR ${s.v2g_revenue_eur_day}/day</div>
              <div class="kpi-label">V2G revenue</div></div>
            <div class="kpi">
              <div class="kpi-val">${s.total_charge_kwh} kWh</div>
              <div class="kpi-label">Total charged from grid</div></div>
            <div class="kpi ${parseFloat(s.soc_min_kwh)<12?'red':''}">
              <div class="kpi-val">${s.soc_min_kwh} kWh</div>
              <div class="kpi-label">SOC minimum reached</div></div>
          </div>
          ${checkHtml(s.checks)}
        </div>`;
      }
      if (step.ordering_checks && step.ordering_checks.length) {
        html += `<div class="ordering-section">
          <div class="ordering-title">&#x1F4D0; Cross-Scenario Ordering Validation</div>
          ${checkHtml(step.ordering_checks)}
        </div>`;
      }
      html += '</div>';
      block.innerHTML = html;
    }
    else if (step.id === 'plots') {
      let html = `<div class="step-header"><h3>&#x1F4CA; ${step.title}</h3></div><div class="step-body">`;
      for (const p of step.plots) {
        html += `<div class="plot-wrap">
          <div class="plot-title">${p.name}</div>
          <img src="data:image/png;base64,${p.img}">
        </div>`;
      }
      html += '</div>';
      block.innerHTML = html;
    }

    area.appendChild(block);
  }
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class DebugHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress request logs in terminal

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        if urlparse(self.path).path != "/run":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            config = json.loads(body)
        except Exception:
            config = {}
        result  = run_debug(config)
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def find_free_port():
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    port   = find_free_port()
    server = HTTPServer(("localhost", port), DebugHandler)
    url    = f"http://localhost:{port}"
    print("\n" + "="*60)
    print("  V2G Debug Tool — S.KOe COOL")
    print("  Schmitz Cargobull AG x TU Dortmund | 2026")
    print("="*60)
    if not IMPORT_OK:
        print(f"\n  WARNING: Could not import run_optimisation.py")
        print(f"  Make sure debug_tool.py is in the same folder.")
    print(f"\n  Opening browser at: {url}")
    print(f"  Press Ctrl+C to stop\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()