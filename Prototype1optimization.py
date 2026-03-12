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
        ("A – Dumb",            "#AAAAAA",
         "Charges at full power on arrival.\nNo price awareness. No V2G. Baseline."),
        ("B – Smart (no V2G)",  "#2196F3",
         "MILP shifts charging to cheapest slots.\nNever discharges. Minimal battery wear."),
        ("C – MILP Day-Ahead",  "#00BCD4",
         "Full 24h MILP at 00:00, perfect forecast.\nCharges cheap, discharges at peak."),
        ("D – MPC Perfect",     "#FF7700",
         "Receding-horizon MPC, re-solves every\n15-min slot. No noise. Near-optimal."),
    ]

    y = 0.91
    for sc_label, col, desc in scenarios:
        patch = mpatches.FancyBboxPatch((0.02, y-0.012), 0.06, 0.030,
                                        boxstyle="round,pad=0.005",
                                        facecolor=col, edgecolor="white",
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(patch)
        ax.text(0.11, y+0.005, sc_label, ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#1B5E20", transform=ax.transAxes)
        for i, line in enumerate(desc.split("\n")):
            ax.text(0.11, y - 0.018 - i*0.018, line, ha="left", va="top",
                    fontsize=7.5, color="#333333", transform=ax.transAxes)
        y -= 0.13   # reduced from 0.16 → tighter but no overlap

    # y is now dynamic — cost section starts right below last scenario
    y -= 0.04
    ax.text(0.5, y, "COST / REVENUE TERMS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1B5E20",
            transform=ax.transAxes)
    y -= 0.06

    cost_terms = [
        ("Net Cost (€/day)",         "= Charge cost − V2G revenue + Deg cost"),
        ("Charge Cost (€/day)",      "= Σ_t  buy[t] · P_c[t] · dt"),
        ("V2G Revenue (€/day)",      "= Σ_t  v2g[t] · P_d[t] · dt"),
        ("Degradation Cost (€/day)", "= Σ_t  deg · (P_c[t]+P_d[t]) · dt"),
        ("Savings vs A (€/day)",     "= Net Cost(A) − Net Cost(scenario)"),
        ("Annual Savings (€/yr)",    "= Savings/day × 365"),
    ]
    for term, formula in cost_terms:
        ax.text(0.03, y, f"• {term}", ha="left", va="top", fontsize=8.0,
                fontweight="bold", color="#C62828", transform=ax.transAxes)
        ax.text(0.03, y-0.020, f"    {formula}", ha="left", va="top",
                fontsize=7.5, color="#333333", transform=ax.transAxes)
        y -= 0.050

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
        ("deg cost default",  "0.02 €/kWh cycled (LFP cell ageing estimate)"),
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


def generate_equations_card(out="equations_reference.png"):
    """
    Generates a detailed reference PNG explaining the MILP and MPC
    core equations, constraints, and algorithm flow.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "V2G Optimisation — Core Equations: MILP & Receding-Horizon MPC",
        fontsize=16, fontweight="bold", color="#0D1B2A", y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.04, left=0.03, right=0.97)

    # ── Box style helper ───────────────────────────────────────────────────
    BOX = dict(boxstyle="round,pad=0.5", facecolor="white",
               edgecolor="#90A4AE", linewidth=1.2)
    HEAD_BOX = dict(boxstyle="round,pad=0.4", facecolor="#1A237E",
                    edgecolor="#1A237E")

    # ══════════════════════════════════════════════════════════════════════
    #  Panel 1 (top-left): MILP Objective Function
    # ══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#E8EAF6"); ax.axis("off")
    ax.text(0.5, 0.97, "① MILP — Objective Function", ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A237E"))

    lines = [
        (0.5, 0.86, "min  J  =  Σ_t  [ C_buy(t) + C_deg(t) − R_v2g(t) ] · Δt",
         10.5, "bold", "#B71C1C"),
        (0.5, 0.76, "where:", 9, "bold", "#1A237E"),
        (0.07, 0.68, "C_buy(t)  =  buy[t] · P_c[t]",
         9, "normal", "#212121"),
        (0.60, 0.68, "← grid import cost at slot t",
         8.5, "italic", "#555555"),
        (0.07, 0.59, "R_v2g(t)  =  v2g[t] · P_d[t]",
         9, "normal", "#212121"),
        (0.60, 0.59, "← V2G revenue at slot t",
         8.5, "italic", "#555555"),
        (0.07, 0.50, "C_deg(t)  =  deg · (P_c[t] + P_d[t])",
         9, "normal", "#212121"),
        (0.60, 0.50, "← battery wear cost",
         8.5, "italic", "#555555"),
        (0.07, 0.38, "buy[t]    day-ahead spot price  (€/kWh)",
         8.5, "normal", "#333333"),
        (0.07, 0.30, "v2g[t]    V2G sell price  (€/kWh)",
         8.5, "normal", "#333333"),
        (0.07, 0.22, "deg       degradation cost  (€/kWh cycled)",
         8.5, "normal", "#333333"),
        (0.07, 0.14, "Δt        time step = 0.25 h  (15 min)",
         8.5, "normal", "#333333"),
        (0.07, 0.06, "t         slot index  0 … 95  (96 slots = 24 h)",
         8.5, "normal", "#333333"),
    ]
    for x, y, txt, fs, fw, col in lines:
        ax.text(x, y, txt, ha="left" if x < 0.5 else "left",
                va="top", fontsize=fs, fontstyle="italic" if fw == "italic" else "normal",
                fontweight="bold" if fw == "bold" else "normal",
                color=col, transform=ax.transAxes,
                bbox=BOX if (y in [0.86]) else None)

    # ══════════════════════════════════════════════════════════════════════
    #  Panel 2 (top-middle): MILP Constraints
    # ══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#E8F5E9"); ax.axis("off")
    ax.text(0.5, 0.97, "② MILP — Constraints", ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1B5E20"))

    constraints = [
        ("(i)  SoC Dynamics  [energy balance]", "#1B5E20", 0.88),
        ("e[t] = e[t−1]  +  η_c · P_c[t] · Δt", "#B71C1C", 0.80),
        ("            −  (1/η_d) · P_d[t] · Δt  −  TRU[t] · Δt", "#B71C1C", 0.73),
        ("      with  e[0]  initialised to  E_init", "#555555", 0.66),
        ("(ii)  Power Bounds  [hardware limits]", "#1B5E20", 0.57),
        ("0  ≤  P_c[t]  ≤  p_c_max · plugged[t]", "#333333", 0.50),
        ("0  ≤  P_d[t]  ≤  p_d_max · plugged[t]", "#333333", 0.43),
        ("      plugged[t] ∈ {0,1}  (availability flag)", "#555555", 0.37),
        ("(iii)  SoC Bounds  [battery safety]", "#1B5E20", 0.28),
        ("E_min  ≤  e[t]  ≤  E_max   ∀ t", "#333333", 0.21),
        ("(iv)  Mutex  [no simultaneous charge+discharge]", "#1B5E20", 0.13),
        ("P_c[t] + P_d[t]  ≤  max(p_c_max, p_d_max)", "#333333", 0.06),
    ]
    for txt, col, y in constraints:
        fw = "bold" if txt.startswith("(") else "normal"
        ax.text(0.05, y, txt, ha="left", va="top",
                fontsize=8.8, fontweight=fw, color=col,
                transform=ax.transAxes)

    # departure SoC constraint box
    ax.text(0.5, -0.02,
            "(v)  Departure SoC:  e[N−1]  ≥  E_fin",
            ha="center", va="top", fontsize=8.8,
            fontweight="bold", color="#B71C1C",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFEBEE", edgecolor="#B71C1C"))

    # ══════════════════════════════════════════════════════════════════════
    #  Panel 3 (top-right): Variable Layout (solver internals)
    # ══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor("#FFF3E0"); ax.axis("off")
    ax.text(0.5, 0.97, "③ MILP — Variable Layout (HiGHS solver)",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E65100"))

    ax.text(0.5, 0.87,
            "Decision vector  x  of length  3W:",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#E65100", transform=ax.transAxes)

    segments = [
        ("#BBDEFB", "x[0 … W−1]\n= P_c(0) … P_c(W−1)\nCharging power\n(kW per slot)"),
        ("#C8E6C9", "x[W … 2W−1]\n= P_d(0) … P_d(W−1)\nDischarge power\n(kW per slot)"),
        ("#FFCCBC", "x[2W … 3W−1]\n= e(0) … e(W−1)\nSoC trajectory\n(kWh per slot)"),
    ]
    for i, (fc, txt) in enumerate(segments):
        x0 = 0.04 + i * 0.33
        rect = mpatches.FancyBboxPatch((x0, 0.52), 0.28, 0.26,
                                        boxstyle="round,pad=0.01",
                                        facecolor=fc, edgecolor="#78909C",
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x0 + 0.14, 0.65, txt, ha="center", va="center",
                fontsize=7.8, color="#212121", transform=ax.transAxes)

    ax.text(0.5, 0.47, "W = remaining slots in horizon  (W = 96 at t=0)",
            ha="center", va="top", fontsize=8.5, color="#555555",
            transform=ax.transAxes)

    solver_notes = [
        "• Solver: scipy HiGHS  (scipy.optimize.milp)",
        "• Constraint matrix A: sparse (lil_matrix → csc_matrix)",
        "• Bounds object: lb = 0, ub = hardware limits",
        "• Time limit: 60 s per window",
        "• Fallback: greedy rule-based if solver fails",
        "• Cost vector c: [buy+deg, −v2g+deg, 0] × Δt",
    ]
    y = 0.38
    for note in solver_notes:
        ax.text(0.04, y, note, ha="left", va="top", fontsize=8.2,
                color="#333333", transform=ax.transAxes)
        y -= 0.055

    # ══════════════════════════════════════════════════════════════════════
    #  Panel 4 (bottom-left): MPC Algorithm Flow
    # ══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#EDE7F6"); ax.axis("off")
    ax.text(0.5, 0.97, "④ MPC — Receding-Horizon Algorithm",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#4A148C"))

    steps = [
        ("STEP 1 — FORECAST", "#4A148C",
         "Horizon = full remaining day  [t … 95]\n"
         "buy_fc = buy[t:],   v2g_fc = v2g[t:]\n"
         "Window size W = 96 − t  (shrinks each slot)"),
        ("STEP 2 — SOLVE MILP", "#4A148C",
         "Run MILP over W remaining slots\n"
         "E_init = current real SoC\n"
         "E_fin  = departure target (always at slot 95)"),
        ("STEP 3 — EXECUTE FIRST ACTION", "#4A148C",
         "Apply only P_c[0] and P_d[0]\n"
         "Discard the rest of the optimal schedule\n"
         "Mutex: if both > 0 → pick more profitable"),
        ("STEP 4 — ADVANCE SoC", "#4A148C",
         "soc += P_c·η_c·Δt − P_d/η_d·Δt − TRU·Δt\n"
         "soc  = clip(soc, E_min, E_max)\n"
         "t → t+1,  repeat until t = 95"),
    ]

    y = 0.87
    for title, tc, body in steps:
        ax.text(0.05, y, title, ha="left", va="top", fontsize=8.8,
                fontweight="bold", color=tc, transform=ax.transAxes)
        for i, line in enumerate(body.split("\n")):
            ax.text(0.07, y - 0.055 - i*0.045, line, ha="left", va="top",
                    fontsize=8.0, color="#212121", transform=ax.transAxes)
        y -= 0.21

    # ══════════════════════════════════════════════════════════════════════
    #  Panel 5 (bottom-middle): MILP vs MPC comparison
    # ══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#E0F2F1"); ax.axis("off")
    ax.text(0.5, 0.97, "⑤ MILP Day-Ahead vs MPC — Key Differences",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#004D40"))

    headers = ["Property", "C — MILP", "D — MPC"]
    rows = [
        ["Solved at",        "Once at t=0",         "Every slot t"],
        ["Horizon W",        "96 slots (fixed)",    "96−t (shrinking)"],
        ["Price info",       "Full day known",      "Full day known"],
        ["SoC update",       "Open-loop",           "Closed-loop"],
        ["Re-optimises",     "Never",               "Every 15 min"],
        ["Disturbance corr.","None",                "Yes (real SoC)"],
        ["Compute cost",     "1× per day",          "96× per day"],
        ["Result quality",   "Global optimum",      "Near-optimal"],
    ]

    col_x  = [0.03, 0.38, 0.70]
    col_bg = ["#B2DFDB", "#B2EBF2", "#FFE0B2"]
    y = 0.84
    # Header row
    for cx, hdr, bg in zip(col_x, headers, col_bg):
        ax.text(cx, y, hdr, ha="left", va="top", fontsize=8.8,
                fontweight="bold", color="#004D40",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.2",
                          facecolor=bg, edgecolor="#90A4AE"))
    y -= 0.09
    for row in rows:
        for cx, val, bg in zip(col_x, row, col_bg):
            fc = "#F5F5F5" if row.index(val) == 0 else bg
            ax.text(cx, y, val, ha="left", va="top", fontsize=8.0,
                    color="#212121", transform=ax.transAxes)
        y -= 0.075

    ax.text(0.5, 0.02,
            "In practice: MPC ≈ MILP because full-day horizon\n"
            "eliminates the myopic planning problem.",
            ha="center", va="bottom", fontsize=8.0,
            fontstyle="italic", color="#004D40",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#E0F7FA", edgecolor="#004D40"))

    # ══════════════════════════════════════════════════════════════════════
    #  Panel 6 (bottom-right): SoC Dynamics & Efficiency Chain
    # ══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor("#FCE4EC"); ax.axis("off")
    ax.text(0.5, 0.97, "⑥ SoC Dynamics & Efficiency Chain",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#880E4F"))

    ax.text(0.5, 0.88,
            "e[t]  =  e[t−1]  +  η_c·P_c[t]·Δt  −  P_d[t]/η_d·Δt  −  TRU[t]·Δt",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#B71C1C", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFEBEE", edgecolor="#B71C1C"))

    flow_items = [
        ("GRID", "#1565C0", 0.10, 0.73),
        ("P_c · η_c  →", "#1565C0", 0.30, 0.73),
        ("BATTERY\ne[t] kWh", "#2E7D32", 0.50, 0.73),
        ("→  P_d / η_d", "#C62828", 0.70, 0.73),
        ("GRID\n(V2G)", "#C62828", 0.87, 0.73),
    ]
    for txt, col, x, y_pos in flow_items:
        is_box = txt in ["GRID", "BATTERY\ne[t] kWh", "GRID\n(V2G)"]
        ax.text(x, y_pos, txt, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=col,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#E3F2FD" if "GRID" in txt else "#E8F5E9",
                          edgecolor=col) if is_box else None)

    # TRU drain arrow
    ax.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.68),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#AA0000", lw=1.5))
    ax.text(0.55, 0.63, "TRU drain\n−TRU[t]·Δt", ha="left", va="center",
            fontsize=7.8, color="#AA0000", transform=ax.transAxes)

    params_title = "Parameters"
    ax.text(0.5, 0.53, params_title, ha="center", va="top",
            fontsize=9, fontweight="bold", color="#880E4F",
            transform=ax.transAxes)

    params = [
        ("η_c = 0.92",   "Charge efficiency (8% lost to heat on AC→DC)"),
        ("η_d = 0.92",   "Discharge efficiency (8% lost on DC→AC invert)"),
        ("E_min = 12 kWh","SoC floor = 20% × 60 kWh usable"),
        ("E_max = 57 kWh","SoC ceiling = 95% × 60 kWh usable"),
        ("TRU ≈ 2.8–4 kW","Refrigeration load (sinusoidal over 24h)"),
        ("Δt = 0.25 h",  "15-min slot = 1/4 hour"),
    ]
    y = 0.46
    for param, desc in params:
        ax.text(0.03, y, param, ha="left", va="top", fontsize=8.2,
                fontweight="bold", color="#880E4F", transform=ax.transAxes)
        ax.text(0.28, y, desc, ha="left", va="top", fontsize=8.0,
                color="#333333", transform=ax.transAxes)
        y -= 0.063

    ax.text(0.5, 0.03,
            "Round-trip efficiency = η_c × η_d = 0.92² ≈ 84.6%\n"
            "Every kWh discharged costs 1/0.92 ≈ 1.087 kWh from battery",
            ha="center", va="bottom", fontsize=7.8,
            fontstyle="italic", color="#555555",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="#FCE4EC", edgecolor="#880E4F"))

    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Equations reference card saved → {out}")

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
    deg_cost_eur_kwh:      float = 0.02     # battery wear €/kWh cycled
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

def _load_smard_csv(csv_path: str) -> pd.DataFrame:
    global _PRICE_CACHE
    if "df" in _PRICE_CACHE:
        return _PRICE_CACHE["df"]

    # Auto-detect encoding and separator
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc)
            if len(df.columns) > 1:
                break
            df = pd.read_csv(csv_path, sep=",", encoding=enc)
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    if df is None or df.empty or len(df.columns) < 2:
        raise ValueError(f"Could not read CSV at {csv_path} — check the file exists and is a valid SMARD export.")

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
    tru = np.zeros(N)   # TRU off when stationary at depot
    if dwell == "Weekend":
        plugged = np.ones(N)
    elif dwell == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    elif dwell == "DayTrip":
        # Trailer departs 07:00, returns 17:00 → plugged 17:00-07:00
        plugged = ((h >= 17) | (h < 7)).astype(float)
    else:  # Extended (default)
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
    E_fin  = min(v2g.usable_capacity_kWh * soc_final_pct / 100.0, v2g.E_max)
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
    E_fin  = min(v2g.usable_capacity_kWh * soc_final_pct / 100.0, v2g.E_max)
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
    E_fin  = min(v2g.usable_capacity_kWh * soc_final_pct / 100.0, v2g.E_max)
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
    """
    Layout:
      Left column (4 rows) : one subplot per scenario, P_c + P_d + price overlay
      Right column (2×2)   : SoC | V2G discharge | deg sensitivity | KPI table
    All time axes rolled to start at 17:00 (trailer arrival).
    """
    # ── Roll display to 17:00 ─────────────────────────────────────────────
    ROLL = 68   # slot 68 = 17:00
    N    = len(hours)

    def roll_arr(a): return np.concatenate([a[ROLL:], a[:ROLL]])

    def roll_r(r):
        return V2GResult(
            scenario=r.scenario,
            p_charge=roll_arr(r.p_charge), p_discharge=roll_arr(r.p_discharge),
            soc=roll_arr(r.soc),
            cost_eur_day=r.cost_eur_day, v2g_revenue_eur_day=r.v2g_revenue_eur_day,
            v2g_export_kwh_day=r.v2g_export_kwh_day,
            charge_cost_eur_day=r.charge_cost_eur_day,
            deg_cost_eur_day=r.deg_cost_eur_day,
            price_buy=roll_arr(r.price_buy), price_v2g=roll_arr(r.price_v2g),
            plugged=roll_arr(r.plugged),   tru_load=roll_arr(r.tru_load))

    rA, rB, rC, rD = roll_r(A), roll_r(B), roll_r(C), roll_r(D)
    results_rolled  = [rA, rB, rC, rD]

    # x-axis: 17.00 … 40.75  (24+ = next-day hours)
    hours_disp = np.concatenate([hours[ROLL:], hours[:ROLL] + 24.0])
    tick_pos   = np.arange(17, 42, 2, dtype=float)
    tick_lbls  = [f"{int(h % 24):02d}:00" for h in tick_pos]

    labels_full = ["A – Dumb (uncontrolled)", "B – Smart (no V2G)",
                   "C – MILP Day-Ahead",      "D – MPC Perfect"]
    colors_list = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"]]

    # ── Figure & GridSpec ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    gs  = GridSpec(4, 3, figure=fig,
                   width_ratios=[1.8, 1.5, 1.5],
                   hspace=0.08, wspace=0.38,
                   top=0.93, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle(
        f"S.KOe COOL  –  MILP + MPC V2G Optimisation  ({season})\n"
        f"Time axis: 17:00 arrival  →  next day 17:00  (DayTrip dwell)",
        fontsize=12, fontweight="bold")

    y_max = max(r.p_charge.max() for r in results_rolled) * 1.18 or 25

    # ── Left column: 4 individual scenario plots ──────────────────────────
    for i, (r, lbl, col) in enumerate(zip(results_rolled, labels_full, colors_list)):
        ax  = fig.add_subplot(gs[i, 0])
        ax2 = ax.twinx()

        # Price dashed line (right axis)
        ax2.step(hours_disp, r.price_v2g * 1000, where="post",
                 color=COL["price"], lw=1.0, alpha=0.55, ls="--")
        ax2.set_ylabel("€/MWh", fontsize=6, color=COL["price"])
        ax2.tick_params(axis="y", labelsize=6, colors=COL["price"])

        # Charging power fill
        ax.fill_between(hours_disp, r.p_charge, step="pre",
                        color=col, alpha=0.80)
        # V2G discharge fill (negative)
        if r.p_discharge.max() > 0.01:
            ax.fill_between(hours_disp, -r.p_discharge, step="pre",
                            color="#E53935", alpha=0.70)
            ax.text(0.98, 0.05, "▼ P_d V2G", transform=ax.transAxes,
                    fontsize=6.5, color="#E53935", ha="right", va="bottom")

        # Plug-in / plug-out vertical markers
        plug = r.plugged
        for t in range(1, N):
            if plug[t] > 0.5 > plug[t-1]:
                ax.axvline(hours_disp[t], color="green", lw=1.2, ls="--", alpha=0.6)
                ax.text(hours_disp[t]+0.1, y_max*0.88, "IN",
                        fontsize=6, color="green", va="top")
            elif plug[t] < 0.5 < plug[t-1]:
                ax.axvline(hours_disp[t], color="red", lw=1.2, ls="--", alpha=0.6)
                ax.text(hours_disp[t]+0.1, y_max*0.88, "OUT",
                        fontsize=6, color="red", va="top")

        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlim(17, 41)
        ax.set_ylim(-v2g_global.p_d_max * 1.15, y_max)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.25)

        # Label inside plot
        ax.text(0.01, 0.95, f"({i+1}) {lbl}",
                transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", color=col, va="top")
        ax.set_ylabel("kW", fontsize=7)

        # X-axis only on bottom plot
        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbls, fontsize=7, rotation=35, ha="right")
            ax.set_xlabel("Hour  (17:00 = trailer arrival at depot)", fontsize=8)

    # ── Right top-left (rows 0-1): SoC ────────────────────────────────────
    ax = fig.add_subplot(gs[0:2, 1])
    for r, lbl, col, ls in zip(results_rolled, labels_full, colors_list,
                               ["-", "-", "-", "--"]):
        ax.plot(hours_disp, r.soc, color=col, lw=2, ls=ls,
                label=lbl.split("–")[0].strip())
    ax.axhline(v2g_global.E_min, color="red",  ls=":", lw=1.2,
               label=f"E_min = {v2g_global.E_min:.0f} kWh")
    ax.axhline(v2g_global.E_max, color="navy", ls=":", lw=1.2,
               label=f"E_max = {v2g_global.E_max:.0f} kWh")
    ax.axvline(31, color="grey", ls=":", lw=0.8, alpha=0.5)   # midnight
    ax.text(31.1, v2g_global.E_min + 1, "midnight", fontsize=7,
            color="grey", va="bottom")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbls, fontsize=7, rotation=35, ha="right")
    ax.set_xlim(17, 41)
    ax.set_title("(5) SoC — Battery State of Charge\n(17:00 → next day 17:00)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("E  (kWh)"); ax.set_xlabel("Hour")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(True, alpha=0.3)

    # ── Right top-right (rows 0-1): V2G discharge vs price ────────────────
    ax  = fig.add_subplot(gs[0:2, 2])
    ax2 = ax.twinx()
    w   = 0.18
    ax.bar(hours_disp - w/2, rC.p_discharge, width=w,
           color=COL["milp"], alpha=0.85, label="C – MILP P_d")
    ax.bar(hours_disp + w/2, rD.p_discharge, width=w,
           color=COL["mpc"],  alpha=0.75, label="D – MPC P_d")
    ax2.step(hours_disp, rC.price_v2g * 1000, where="post",
             color=COL["price"], lw=1.8, label="V2G price (€/MWh)")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbls, fontsize=7, rotation=35, ha="right")
    ax.set_xlim(17, 41)
    ax.set_title("(6) P_d — V2G Discharge vs Price\n(C–MILP vs D–MPC)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("P_d  (kW)"); ax.set_xlabel("Hour")
    ax2.set_ylabel("Price  (€/MWh)", color=COL["price"], fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── Right bottom-left (rows 2-3): deg sensitivity ─────────────────────
    ax  = fig.add_subplot(gs[2:4, 1])
    ax2 = ax.twinx()
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["NetCost_EUR_day"],
            "o-", color=COL["milp"], lw=2, label="Net Cost (€/day)")
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["V2G_Rev_EUR_day"],
            "s--", color=COL["mpc"], lw=2, label="V2G Revenue (€/day)")
    ax2.bar(deg_df["DegCost_EUR_kWh"], deg_df["V2G_kWh_day"],
            width=0.008, color=COL["mpc"], alpha=0.22, label="V2G kWh/day")
    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    if not np.isnan(tipping):
        ax.axvline(tipping, color="red", ls=":", lw=1.5,
                   label=f"V2G cutoff ≈ {tipping:.3f}")
    ax.axvline(v2g_global.deg_cost_eur_kwh, color="black", ls="--", lw=1.2,
               label=f"Active deg = {v2g_global.deg_cost_eur_kwh:.3f}")
    ax.set_title("(7) Degradation (deg) Sensitivity",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("deg  (€/kWh cycled)"); ax.set_ylabel("€/day")
    ax2.set_ylabel("V2G export  (kWh/day)", color=COL["mpc"], fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── Right bottom-right (rows 2-3): KPI table ──────────────────────────
    ax = fig.add_subplot(gs[2:4, 2])
    ax.axis("off")
    ref = A.cost_eur_day
    table_data = [
        ["Metric",                 "A\nDumb",   "B\nSmart",  "C\nMILP",   "D\nMPC"],
        ["Net cost\n(€/day)",
         f"{A.cost_eur_day:.3f}",  f"{B.cost_eur_day:.3f}",
         f"{C.cost_eur_day:.3f}",  f"{D.cost_eur_day:.3f}"],
        ["Charge cost\n(€/day)",
         f"{A.charge_cost_eur_day:.3f}", f"{B.charge_cost_eur_day:.3f}",
         f"{C.charge_cost_eur_day:.3f}", f"{D.charge_cost_eur_day:.3f}"],
        ["V2G revenue\n(€/day)",
         f"{A.v2g_revenue_eur_day:.3f}", f"{B.v2g_revenue_eur_day:.3f}",
         f"{C.v2g_revenue_eur_day:.3f}", f"{D.v2g_revenue_eur_day:.3f}"],
        ["deg cost\n(€/day)",
         f"{A.deg_cost_eur_day:.3f}", f"{B.deg_cost_eur_day:.3f}",
         f"{C.deg_cost_eur_day:.3f}", f"{D.deg_cost_eur_day:.3f}"],
        ["V2G export\n(kWh/day)",
         f"{A.v2g_export_kwh_day:.2f}", f"{B.v2g_export_kwh_day:.2f}",
         f"{C.v2g_export_kwh_day:.2f}", f"{D.v2g_export_kwh_day:.2f}"],
        ["Savings vs A\n(€/day)",
         "—", f"{ref-B.cost_eur_day:+.3f}",
         f"{ref-C.cost_eur_day:+.3f}", f"{ref-D.cost_eur_day:+.3f}"],
        ["Annual savings\n(€/yr)",
         "—", f"{(ref-B.cost_eur_day)*365:+,.0f}",
         f"{(ref-C.cost_eur_day)*365:+,.0f}",
         f"{(ref-D.cost_eur_day)*365:+,.0f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 2.15)

    col_colors = ["#EEEEEE", "#BBDEFB", "#B2EBF2", "#FFE0B2"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#90A4AE")
        if r == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif c == 0:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(fontweight="bold", fontsize=8)
            cell.set_width(0.38)   # wider Metric column
        else:
            cell.set_facecolor(col_colors[c - 1])

    ax.set_title("(8) KPI Summary Table", fontsize=9, fontweight="bold", pad=14)

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

    print("\n" + "─"*55)
    print("  ARRIVAL STATE OF CHARGE")
    soc_input = input("  Enter trailer arrival SoC % (e.g. 45): ").strip()
    try:
        soc_init_pct = float(soc_input)
        assert 20.0 <= soc_init_pct <= 95.0
    except:
        soc_init_pct = 45.0
        print(f"  → Invalid input. Using default 45%")
    print(f"  → Arrival SoC: {soc_init_pct:.0f}%   Departure SoC: 100% (fixed)")
    print("─"*55)
    soc_final_pct = 100.0   # Always depart fully charged

    deg_values    = load_deg_sensitivity(v2g)
    hours         = np.arange(v2g.n_slots) * v2g.dt_h

    print("\n  Generating abbreviation legend …")
    generate_abbreviation_legend("abbreviation_legend.png")

    print("\n  Generating equations reference card …")
    generate_equations_card("equations_reference.png")

    all_season_results: dict = {}

    DAY_TYPES = [
        ("winter",         "DayTrip", 130, "Winter weekday  (Mon–Fri, Oct–Mar)"),
        ("summer",         "DayTrip", 131, "Summer weekday  (Mon–Fri, Apr–Sep)"),
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

        # Roll arrays so simulation starts at 17:00 (trailer arrival)
        if dwell_type == "DayTrip":
            ROLL = 68   # 17:00 × 4 slots/hour
            buy     = np.roll(buy,     -ROLL)
            v2g_p   = np.roll(v2g_p,   -ROLL)
            tru     = np.roll(tru,     -ROLL)
            plugged = np.roll(plugged, -ROLL)

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