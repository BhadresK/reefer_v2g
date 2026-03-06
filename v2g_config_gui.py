#!/usr/bin/env python3
"""
v2g_config_gui.py
═══════════════════════════════════════════════════════════════════════════════
Interactive browser-based parameter editor for the S.KOe COOL V2G project.

Usage:
    python v2g_config_gui.py                # opens GUI, then runs optimisation
    python v2g_config_gui.py --year-sim     # opens GUI, then runs year simulation
    python v2g_config_gui.py --no-run       # just open GUI to inspect/edit values

How it works:
    1. Starts a local HTTP server on a free port
    2. Opens the config GUI in your default browser
    3. You review and edit every parameter
    4. Click "Confirm & Run" — values are sent back to Python
    5. Python saves them to a temp config JSON and runs the chosen script

Zero extra installs — uses Python built-ins only (http.server, webbrowser, json).
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ── Default parameter values ──────────────────────────────────────────────────
DEFAULTS = {
    # ── Battery (S.KOe COOL 70 kWh pack) ─────────────────────────────────────
    "battery_capacity_kWh":   70.0,
    "usable_capacity_kWh":    60.0,
    "soc_min_pct":            20.0,
    "soc_max_pct":            100.0,
    "charge_power_kW":        22.0,
    "discharge_power_kW":     22.0,
    "eta_charge":             0.92,
    "eta_discharge":          0.92,
    "deg_cost_eur_kwh":       0.03,
    # ── Simulation settings ───────────────────────────────────────────────────
    "soc_init_pct":           45.0,
    "soc_final_pct":          100.0,
    "dwell_profile":          "Extended",
    "mpc_price_noise_std":    0.012,
    # ── Electricity price (BNetzA 2024) ───────────────────────────────────────
    "network_fee":            0.0663,
    "concession":             0.01992,
    "offshore_levy":          0.00816,
    "chp_levy":               0.00277,
    "electricity_tax":        0.0205,
    "nev19":                  0.01558,
    "vat":                    0.19,
    # ── V2G premiums (Agora Verkehrswende 2025) ───────────────────────────────
    "fcr_premium_peak":       0.132,
    "fcr_window_start":       16,
    "fcr_window_end":         20,
    "afrr_premium":           0.040,
    "afrr_window_start":      7,
    "afrr_window_end":        9,
    # ── Fleet & year ──────────────────────────────────────────────────────────
    "fleet_size":             1,
    "year":                   2024,
    "season":                 "winter",
    "price_mode":             "auto",
}

# ── HTML page (self-contained, no CDN needed) ─────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S.KOe COOL — V2G Parameter Configuration</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --blue:   #1a56db;
    --blue-l: #e8f0fe;
    --green:  #057a55;
    --green-l:#def7ec;
    --amber:  #b45309;
    --amber-l:#fef3c7;
    --red:    #c81e1e;
    --red-l:  #fde8e8;
    --gray:   #374151;
    --gray-l: #f9fafb;
    --border: #d1d5db;
    --radius: 8px;
  }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f3f4f6;
    color: var(--gray);
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a56db 100%);
    color: white;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  .logo {
    width: 48px; height: 48px;
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
  }
  header h1 { font-size: 1.4rem; font-weight: 700; }
  header p  { font-size: 0.85rem; opacity: 0.8; margin-top: 2px; }
  .layout {
    display: grid;
    grid-template-columns: 220px 1fr;
    max-width: 1280px;
    margin: 24px auto;
    gap: 24px;
    padding: 0 24px;
  }
  .sidebar {
    position: sticky;
    top: 24px;
    height: fit-content;
    background: white;
    border-radius: var(--radius);
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    overflow: hidden;
  }
  .sidebar-title {
    padding: 12px 16px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #6b7280;
    border-bottom: 1px solid var(--border);
    background: var(--gray-l);
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    cursor: pointer;
    font-size: 0.875rem;
    color: #374151;
    border-left: 3px solid transparent;
    transition: all .15s;
    text-decoration: none;
  }
  .nav-item:hover { background: var(--blue-l); color: var(--blue); }
  .nav-item.active {
    background: var(--blue-l);
    color: var(--blue);
    border-left-color: var(--blue);
    font-weight: 600;
  }
  .nav-icon { font-size: 1rem; width: 20px; text-align: center; }
  .main { display: flex; flex-direction: column; gap: 20px; }
  .derived-bar {
    background: white;
    border-radius: var(--radius);
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 14px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px 32px;
    align-items: center;
  }
  .derived-bar-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #6b7280;
    width: 100%;
  }
  .kpi { display: flex; flex-direction: column; align-items: center; }
  .kpi-value { font-size: 1.25rem; font-weight: 700; color: var(--blue); }
  .kpi-label { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; text-align: center; }
  .kpi-sep { width: 1px; height: 36px; background: var(--border); }
  .section {
    background: white;
    border-radius: var(--radius);
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    overflow: hidden;
  }
  .section-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
  }
  .section-header h2 { font-size: 1rem; font-weight: 700; color: #111827; }
  .section-header p  { font-size: 0.78rem; color: #9ca3af; margin-top: 1px; }
  .section-body {
    padding: 20px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
  }
  .section-body.wide { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #374151;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .field label .unit {
    font-weight: 400;
    color: #9ca3af;
    font-size: 0.72rem;
    background: #f3f4f6;
    padding: 1px 5px;
    border-radius: 4px;
  }
  .field label .src {
    font-weight: 400;
    color: #6b7280;
    font-size: 0.7rem;
    margin-left: auto;
  }
  input[type="number"], input[type="text"], select {
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.875rem;
    color: #111827;
    background: white;
    transition: border-color .15s, box-shadow .15s;
    width: 100%;
  }
  input:focus, select:focus {
    outline: none;
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(26,86,219,.12);
  }
  input.changed { border-color: #d97706; background: #fffbeb; }
  .note {
    grid-column: 1 / -1;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.8rem;
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }
  .note.warn { background: var(--amber-l); color: var(--amber); }
  .note.info { background: var(--blue-l);  color: var(--blue); }
  .note.ok   { background: var(--green-l); color: var(--green); }
  .computed-grid {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 8px;
  }
  .computed-item {
    background: var(--gray-l);
    border-radius: 6px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .computed-item .cv { font-size: 1rem; font-weight: 700; color: var(--blue); }
  .computed-item .cl { font-size: 0.72rem; color: #9ca3af; }
  .cost-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 8px 16px;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.85rem;
  }
  .cost-row:last-child { border-bottom: none; }
  .cost-total {
    background: #eff6ff;
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    color: var(--blue);
    grid-column: 1 / -1;
    margin-top: 4px;
  }
  .action-bar {
    background: white;
    border-radius: var(--radius);
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  .action-bar .info-text {
    font-size: 0.8rem;
    color: #6b7280;
    flex: 1;
    min-width: 200px;
  }
  .btn {
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all .15s;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .btn-primary { background: var(--blue); color: white; }
  .btn-primary:hover { background: #1447c0; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(26,86,219,0.3); }
  .btn-secondary { background: white; color: var(--gray); border: 1px solid var(--border); }
  .btn-secondary:hover { background: var(--gray-l); }
  .btn-danger { background: var(--red-l); color: var(--red); border: 1px solid #fca5a5; }
  .btn-danger:hover { background: #fee2e2; }
  .overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: 1000;
    align-items: center;
    justify-content: center;
  }
  .overlay.show { display: flex; }
  .spinner-card {
    background: white;
    border-radius: 12px;
    padding: 32px 48px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
  }
  .spinner {
    width: 48px; height: 48px;
    border: 4px solid #e5e7eb;
    border-top-color: var(--blue);
    border-radius: 50%;
    animation: spin .8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .success-card {
    display: none;
    background: var(--green-l);
    border: 1px solid #6ee7b7;
    border-radius: var(--radius);
    padding: 16px 20px;
    color: var(--green);
    font-size: 0.9rem;
    align-items: center;
    gap: 10px;
  }
  .success-card.show { display: flex; }
  .changes-badge {
    background: #d97706;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
  }
</style>
</head>
<body>

<header>
  <div class="logo">&#x1F69B;</div>
  <div>
    <h1>S.KOe COOL &mdash; V2G Configuration</h1>
    <p>Schmitz Cargobull &middot; Review and edit all parameters before running the optimisation</p>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:10px;">
    <span id="changesBadge" class="changes-badge" style="display:none">0 changes</span>
    <span style="font-size:0.8rem;opacity:.7;">v2g_config_gui.py</span>
  </div>
</header>

<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-title">Sections</div>
    <a class="nav-item active" onclick="scrollTo('battery')"><span class="nav-icon">&#x1F50B;</span> Battery Pack</a>
    <a class="nav-item" onclick="scrollTo('simulation')"><span class="nav-icon">&#x2699;&#xFE0F;</span> Simulation</a>
    <a class="nav-item" onclick="scrollTo('prices')"><span class="nav-icon">&#x1F4B6;</span> Electricity Prices</a>
    <a class="nav-item" onclick="scrollTo('v2g')"><span class="nav-icon">&#x26A1;</span> V2G Premiums</a>
    <a class="nav-item" onclick="scrollTo('fleet')"><span class="nav-icon">&#x1F69B;</span> Fleet &amp; Year</a>
    <a class="nav-item" onclick="scrollTo('run')"><span class="nav-icon">&#x25B6;&#xFE0F;</span> Confirm &amp; Run</a>
  </nav>

  <div class="main">

    <div class="derived-bar" id="kpiBar">
      <div class="derived-bar-title">&#x1F4CA; Live Derived Values (auto-updates as you type)</div>
      <div class="kpi"><span class="kpi-value" id="kpi_emin">--</span><span class="kpi-label">E_min (kWh)</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_emax">--</span><span class="kpi-label">E_max (kWh)</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_range">--</span><span class="kpi-label">Usable Range (kWh)</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_rte">--</span><span class="kpi-label">Round-trip &eta;</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_einit">--</span><span class="kpi-label">E_init (kWh)</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_efin">--</span><span class="kpi-label">E_fin (kWh)</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_fixednet">--</span><span class="kpi-label">Fixed Net (EUR/kWh)</span></div>
      <div class="kpi-sep"></div>
      <div class="kpi"><span class="kpi-value" id="kpi_buyprice">--</span><span class="kpi-label">Buy @ spot=0.10</span></div>
    </div>

    <!-- BATTERY -->
    <div class="section" id="battery">
      <div class="section-header">
        <div class="section-icon" style="background:#dbeafe">&#x1F50B;</div>
        <div><h2>Battery Pack &mdash; S.KOe COOL 82 kWh</h2><p>Schmitz Cargobull 2025 &middot; ISO 15118-2 &middot; IEC 62196</p></div>
      </div>
      <div class="section-body wide">
        <div class="note warn">&#x26A0;&#xFE0F; <span><b>Note:</b> E_min = UsableCap &times; SoC_min%, E_max = UsableCap &times; SoC_max%. Live values shown in top bar.</span></div>
        <div class="field">
          <label>Battery Capacity (total pack) <span class="unit">kWh</span> <span class="src">Schmitz 2025</span></label>
          <input type="number" id="battery_capacity_kWh" step="0.1" min="1">
        </div>
        <div class="field">
          <label>Usable Capacity <span class="unit">kWh</span> <span class="src">Schmitz: usable window</span></label>
          <input type="number" id="usable_capacity_kWh" step="0.1" min="1">
        </div>
        <div class="field">
          <label>SoC Minimum <span class="unit">%</span> <span class="src">Agora 2025: cold-chain floor</span></label>
          <input type="number" id="soc_min_pct" step="0.5" min="0" max="50">
        </div>
        <div class="field">
          <label>SoC Maximum <span class="unit">%</span> <span class="src">Agora 2025: cycle ceiling</span></label>
          <input type="number" id="soc_max_pct" step="0.5" min="50" max="100">
        </div>
        <div class="field">
          <label>Max Charge Power <span class="unit">kW</span> <span class="src">ISO 15118-2 AC Mode 3</span></label>
          <input type="number" id="charge_power_kW" step="0.5" min="1">
        </div>
        <div class="field">
          <label>Max V2G Discharge Power <span class="unit">kW</span> <span class="src">Bidirectional OBC</span></label>
          <input type="number" id="discharge_power_kW" step="0.5" min="0.5">
        </div>
        <div class="field">
          <label>Charge Efficiency &eta;_c <span class="unit">&mdash;</span> <span class="src">IEC 62196</span></label>
          <input type="number" id="eta_charge" step="0.005" min="0.5" max="1.0">
        </div>
        <div class="field">
          <label>Discharge Efficiency &eta;_d <span class="unit">&mdash;</span> <span class="src">IEC 62196</span></label>
          <input type="number" id="eta_discharge" step="0.005" min="0.5" max="1.0">
        </div>
        <div class="field">
          <label>Degradation Cost <span class="unit">EUR/kWh cycled</span> <span class="src">Agora 2025, Table 3</span></label>
          <input type="number" id="deg_cost_eur_kwh" step="0.005" min="0.01" max="0.25">
        </div>
        <div class="computed-grid" id="battDerived"></div>
      </div>
    </div>

    <!-- SIMULATION -->
    <div class="section" id="simulation">
      <div class="section-header">
        <div class="section-icon" style="background:#d1fae5">&#x2699;&#xFE0F;</div>
        <div><h2>Simulation Settings</h2><p>Initial/final SoC, dwell profile, MPC noise</p></div>
      </div>
      <div class="section-body">
        <div class="field">
          <label>Initial SoC (arrival) <span class="unit">%</span></label>
          <input type="number" id="soc_init_pct" step="1" min="0" max="100">
        </div>
        <div class="field">
          <label>Final SoC (departure target) <span class="unit">%</span></label>
          <input type="number" id="soc_final_pct" step="1" min="0" max="100">
        </div>
        <div class="field">
          <label>Dwell Profile</label>
          <select id="dwell_profile">
            <option value="Extended">Extended (21:00-07:00 + 12:00-18:00 = 16h)</option>
            <option value="NightOnly">Night Only (21:00-07:00 = 10h)</option>
          </select>
        </div>
        <div class="field">
          <label>MPC Price Noise &sigma; <span class="unit">EUR/kWh</span> <span class="src">Liu 2023</span></label>
          <input type="number" id="mpc_price_noise_std" step="0.001" min="0" max="0.1">
        </div>
        <div class="note info">&#x1F4A1; <span><b>Dwell:</b> Extended = night depot + midday hub stop (16h/day). NightOnly = overnight only (10h/day).</span></div>
      </div>
    </div>

    <!-- PRICES -->
    <div class="section" id="prices">
      <div class="section-header">
        <div class="section-icon" style="background:#fef3c7">&#x1F4B6;</div>
        <div><h2>Electricity Price Components</h2><p>BNetzA regulated tariffs 2024 &middot; EUR/kWh excl. VAT</p></div>
      </div>
      <div class="section-body">
        <div class="note info">&#x1F4D0; <span><b>Formula:</b> Buy = (EPEX_spot + Fixed_Net) &times; (1 + VAT) &nbsp;|&nbsp; Fixed_Net = sum of all components below</span></div>
        <div style="grid-column:1/-1">
          <div class="cost-row" style="font-weight:600;font-size:0.78rem;padding-bottom:8px"><span>Component</span><span>EUR/kWh</span><span>Source</span></div>
          <div class="cost-row"><span>Network fee (Netzentgelt)</span><input type="number" id="network_fee" step="0.001" min="0" max="0.5" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">Avg. depot</span></div>
          <div class="cost-row"><span>Concession levy (Konzessionsabgabe)</span><input type="number" id="concession" step="0.0001" min="0" max="0.1" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">BNetzA 2024</span></div>
          <div class="cost-row"><span>Offshore surcharge (Offshore-Netzumlage)</span><input type="number" id="offshore_levy" step="0.0001" min="0" max="0.1" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">BNetzA 2024</span></div>
          <div class="cost-row"><span>CHP levy (KWKG-Umlage)</span><input type="number" id="chp_levy" step="0.0001" min="0" max="0.05" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">BNetzA 2024</span></div>
          <div class="cost-row"><span>Electricity tax (Stromsteuer)</span><input type="number" id="electricity_tax" step="0.001" min="0" max="0.1" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">§3 StromStG</span></div>
          <div class="cost-row"><span>NEV-19 levy</span><input type="number" id="nev19" step="0.0001" min="0" max="0.1" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">BNetzA 2024</span></div>
          <div class="cost-row"><span>VAT rate</span><input type="number" id="vat" step="0.01" min="0" max="0.3" style="width:110px"><span style="font-size:0.75rem;color:#9ca3af">Umsatzsteuer</span></div>
          <div class="cost-total"><span>Fixed Net Total (FIXED_NET)</span><span id="fixed_net_total">-- EUR/kWh</span></div>
        </div>
      </div>
    </div>

    <!-- V2G PREMIUMS -->
    <div class="section" id="v2g">
      <div class="section-header">
        <div class="section-icon" style="background:#fde8e8">&#x26A1;</div>
        <div><h2>V2G / Grid Service Premiums</h2><p>FCR and aFRR balancing premiums &middot; Agora Verkehrswende 2025, Fig. 4</p></div>
      </div>
      <div class="section-body">
        <div class="note info">&#x1F4A1; <span>V2G price = Buy + FCR_premium (evening peak window) + aFRR_premium (morning ramp window)</span></div>
        <div class="field"><label>FCR Peak Premium <span class="unit">EUR/kWh</span> <span class="src">Agora 2025</span></label><input type="number" id="fcr_premium_peak" step="0.005" min="0" max="0.5"></div>
        <div class="field"><label>FCR Window Start <span class="unit">hour 0-23</span></label><input type="number" id="fcr_window_start" step="1" min="0" max="23"></div>
        <div class="field"><label>FCR Window End <span class="unit">hour (exclusive)</span></label><input type="number" id="fcr_window_end" step="1" min="1" max="24"></div>
        <div class="field"><label>aFRR Morning Premium <span class="unit">EUR/kWh</span> <span class="src">Agora 2025</span></label><input type="number" id="afrr_premium" step="0.005" min="0" max="0.2"></div>
        <div class="field"><label>aFRR Window Start <span class="unit">hour 0-23</span></label><input type="number" id="afrr_window_start" step="1" min="0" max="23"></div>
        <div class="field"><label>aFRR Window End <span class="unit">hour (exclusive)</span></label><input type="number" id="afrr_window_end" step="1" min="1" max="24"></div>
        <div class="computed-grid" id="v2g_preview"></div>
      </div>
    </div>

    <!-- FLEET & YEAR -->
    <div class="section" id="fleet">
      <div class="section-header">
        <div class="section-icon" style="background:#e0e7ff">&#x1F69B;</div>
        <div><h2>Fleet &amp; Year Settings</h2><p>Fleet scaling &middot; Year to simulate &middot; Season for single-day run</p></div>
      </div>
      <div class="section-body">
        <div class="field"><label>Fleet Size <span class="unit">trailers</span></label><input type="number" id="fleet_size" step="1" min="1" max="500"></div>
        <div class="field"><label>Year to Simulate <span class="unit">YYYY</span></label><input type="number" id="year" step="1" min="2018" max="2030"></div>
        <div class="field">
          <label>Season (single-day run)</label>
          <select id="season"><option value="winter">Winter weekday</option><option value="summer">Summer weekday</option></select>
        </div>
        <div class="field">
          <label>Price Data Mode</label>
          <select id="price_mode">
            <option value="auto">Auto (real if available, else forecast, else synthetic)</option>
            <option value="real">Real SMARD only</option>
            <option value="forecast">ML forecast</option>
            <option value="synthetic">Synthetic fallback</option>
          </select>
        </div>
        <div class="note info">&#x1F4A1; <span>Fleet scaling is linear. For &gt;25 trailers check grid transformer capacity separately.</span></div>
      </div>
    </div>

    <!-- ACTION BAR -->
    <div class="section action-bar" id="run">
      <div class="info-text">All parameters validated. Changed fields highlighted amber. Click <b>Confirm &amp; Run</b> to send to Python.</div>
      <button class="btn btn-danger" onclick="resetAll()">&#x21BA; Reset to Defaults</button>
      <button class="btn btn-secondary" onclick="exportJSON()">&#x2B07; Export JSON</button>
      <button class="btn btn-primary" onclick="confirmAndRun()">&#x25B6; Confirm &amp; Run</button>
    </div>

    <div class="success-card" id="successCard">
      &#x2705; <div><strong>Configuration sent to Python.</strong><br><span id="successMsg">Optimisation running in terminal. You can close this tab.</span></div>
    </div>

  </div>
</div>

<div class="overlay" id="overlay">
  <div class="spinner-card">
    <div class="spinner"></div>
    <p style="font-weight:600;color:#374151">Sending config to Python...</p>
    <p style="font-size:0.8rem;color:#9ca3af;margin-top:6px">Optimisation will start in your terminal</p>
  </div>
</div>

<script>
const DEFAULTS = __DEFAULTS_JSON__;

let changedCount = 0;

window.onload = function() {
  for (const [key, val] of Object.entries(DEFAULTS)) {
    const el = document.getElementById(key);
    if (!el) continue;
    el.value = val;
    el.dataset.default = val;
    el.addEventListener('input',  () => { onChange(el); updateDerived(); });
    el.addEventListener('change', () => { onChange(el); updateDerived(); });
  }
  updateDerived();
};

function onChange(el) {
  el.classList.toggle('changed', String(el.value) !== String(el.dataset.default));
  changedCount = document.querySelectorAll('input.changed, select.changed').length;
  const badge = document.getElementById('changesBadge');
  badge.style.display = changedCount > 0 ? 'inline' : 'none';
  badge.textContent = changedCount + ' change' + (changedCount !== 1 ? 's' : '');
}

function g(id) { return parseFloat(document.getElementById(id)?.value) || 0; }

function updateDerived() {
  const usable = g('usable_capacity_kWh');
  const emin   = usable * g('soc_min_pct') / 100;
  const emax   = usable * g('soc_max_pct') / 100;
  const rte    = g('eta_charge') * g('eta_discharge');
  const einit  = usable * g('soc_init_pct') / 100;
  const efin   = usable * g('soc_final_pct') / 100;
  const fixed  = g('network_fee')+g('concession')+g('offshore_levy')+g('chp_levy')+g('electricity_tax')+g('nev19');
  const vat    = g('vat');
  const buy10  = (0.10 + fixed) * (1 + vat);

  setText('kpi_emin',     emin.toFixed(2));
  setText('kpi_emax',     emax.toFixed(2));
  setText('kpi_range',    (emax - emin).toFixed(2));
  setText('kpi_rte',      (rte * 100).toFixed(1) + '%');
  setText('kpi_einit',    einit.toFixed(2));
  setText('kpi_efin',     efin.toFixed(2));
  setText('kpi_fixednet', fixed.toFixed(5));
  setText('kpi_buyprice', buy10.toFixed(4));
  setText('fixed_net_total', fixed.toFixed(5) + ' EUR/kWh');

  document.getElementById('battDerived').innerHTML = `
    <div class="computed-item"><span class="cv">${emin.toFixed(2)} kWh</span><span class="cl">E_min (SoC floor)</span></div>
    <div class="computed-item"><span class="cv">${emax.toFixed(2)} kWh</span><span class="cl">E_max (SoC ceiling)</span></div>
    <div class="computed-item"><span class="cv">${(emax-emin).toFixed(2)} kWh</span><span class="cl">Operating range</span></div>
    <div class="computed-item"><span class="cv">${(rte*100).toFixed(1)}%</span><span class="cl">Round-trip &eta;</span></div>
    <div class="computed-item"><span class="cv">${einit.toFixed(2)} kWh</span><span class="cl">E_init (arrival)</span></div>
    <div class="computed-item"><span class="cv">${efin.toFixed(2)} kWh</span><span class="cl">E_fin (departure)</span></div>`;

  const fs = g('fcr_window_start'), fe = g('fcr_window_end');
  const as = g('afrr_window_start'), ae = g('afrr_window_end');
  document.getElementById('v2g_preview').innerHTML = `
    <div class="computed-item"><span class="cv">${fs}:00&ndash;${fe}:00</span><span class="cl">FCR active window</span></div>
    <div class="computed-item"><span class="cv">+${g('fcr_premium_peak').toFixed(3)}</span><span class="cl">FCR premium EUR/kWh</span></div>
    <div class="computed-item"><span class="cv">${as}:00&ndash;${ae}:00</span><span class="cl">aFRR active window</span></div>
    <div class="computed-item"><span class="cv">+${g('afrr_premium').toFixed(3)}</span><span class="cl">aFRR premium EUR/kWh</span></div>`;
}

function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

function scrollTo(id) {
  document.getElementById(id).scrollIntoView({ behavior: 'smooth', block: 'start' });
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  event.currentTarget.classList.add('active');
}

function collectValues() {
  const out = {};
  for (const key of Object.keys(DEFAULTS)) {
    const el = document.getElementById(key);
    if (!el) continue;
    out[key] = typeof DEFAULTS[key] === 'number' ? parseFloat(el.value) : el.value;
  }
  return out;
}

function resetAll() {
  for (const [key, val] of Object.entries(DEFAULTS)) {
    const el = document.getElementById(key);
    if (el) { el.value = val; el.classList.remove('changed'); }
  }
  changedCount = 0;
  document.getElementById('changesBadge').style.display = 'none';
  updateDerived();
}

function exportJSON() {
  const blob = new Blob([JSON.stringify(collectValues(), null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'v2g_config.json';
  a.click();
}

function confirmAndRun() {
  document.getElementById('overlay').classList.add('show');
  fetch('/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(collectValues()),
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById('overlay').classList.remove('show');
    const card = document.getElementById('successCard');
    card.classList.add('show');
    document.getElementById('successMsg').textContent = data.message || 'Configuration accepted.';
    card.scrollIntoView({ behavior: 'smooth' });
  })
  .catch(err => {
    document.getElementById('overlay').classList.remove('show');
    alert('Error: ' + err);
  });
}
</script>
</body>
</html>
"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────
_received_config: dict | None = None
_server_done = threading.Event()

class GUIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # suppress request logs

    def do_GET(self):
        html = HTML.replace("__DEFAULTS_JSON__", json.dumps(DEFAULTS, indent=2))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        global _received_config
        length = int(self.headers.get("Content-Length", 0))
        _received_config = json.loads(self.rfile.read(length))
        body = json.dumps({
            "ok": True,
            "message": "Configuration received — optimisation starting in terminal..."
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        # Wait 2s BEFORE signalling main thread — ensures browser fully reads response
        threading.Timer(2.0, _server_done.set).start()


def find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_gui(timeout: int = 600) -> dict:
    port   = find_free_port()
    server = HTTPServer(("127.0.0.1", port), GUIHandler)
    url    = f"http://127.0.0.1:{port}"

    # Daemon=True means this thread dies automatically when main thread exits
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"\n{'='*65}")
    print("  S.KOe COOL  --  V2G Parameter Configuration GUI")
    print(f"{'='*65}")
    print(f"  Opening browser at:  {url}")
    print(f"  (If browser does not open, paste the URL above manually)")
    print(f"  Waiting for your confirmation...  (timeout: {timeout}s)")
    print(f"{'='*65}\n")

    time.sleep(0.3)
    webbrowser.open(url)

    if not _server_done.wait(timeout=timeout):
        raise TimeoutError("GUI timed out — no config received.")

    # DO NOT call server.shutdown() here — it would kill the socket
    # before the browser finishes reading. The daemon thread dies on its own.
    print("\n  Config received from browser.\n")
    return _received_config.copy()


def apply_config(cfg: dict):
    sys.path.insert(0, str(Path(__file__).parent))
    from run_optimisation import V2GParams
    v2g = V2GParams(
        battery_capacity_kWh = cfg["battery_capacity_kWh"],
        usable_capacity_kWh  = cfg["usable_capacity_kWh"],
        soc_min_pct          = cfg["soc_min_pct"],
        soc_max_pct          = cfg["soc_max_pct"],
        charge_power_kW      = cfg["charge_power_kW"],
        discharge_power_kW   = cfg["discharge_power_kW"],
        eta_charge           = cfg["eta_charge"],
        eta_discharge        = cfg["eta_discharge"],
        deg_cost_eur_kwh     = cfg["deg_cost_eur_kwh"],
        mpc_price_noise_std  = cfg["mpc_price_noise_std"],
    )
    sim = {
        "soc_init_pct":  cfg["soc_init_pct"],
        "soc_final_pct": cfg["soc_final_pct"],
        "dwell":         cfg["dwell_profile"],
        "season":        cfg["season"],
        "fleet_size":    int(cfg["fleet_size"]),
        "year":          int(cfg["year"]),
        "price_mode":    cfg["price_mode"],
    }
    return v2g, sim


def print_config_summary(cfg: dict):
    print("  -- Parameter Summary " + "-"*44)
    print(f"  {'Parameter':<35} {'Value':>14}  {'Default':>14}")
    print("  " + "-"*65)
    for key, val in cfg.items():
        default = DEFAULTS.get(key, "?")
        marker  = "  << CHANGED" if str(val) != str(default) else ""
        print(f"  {key:<35} {str(val):>14}  {str(default):>14}{marker}")
    print("  " + "-"*65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="V2G GUI — opens browser config editor, then runs optimisation"
    )
    parser.add_argument("--year-sim", action="store_true",
                        help="Run full year simulation (default: single-day)")
    parser.add_argument("--no-run",   action="store_true",
                        help="GUI only — do not run any script after")
    parser.add_argument("--timeout",  type=int, default=600,
                        help="Browser wait timeout in seconds (default: 600)")
    args = parser.parse_args()

    cfg = launch_gui(timeout=args.timeout)
    print_config_summary(cfg)

    Path("data").mkdir(exist_ok=True)
    cfg_path = Path("data") / "last_gui_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  Config saved -> {cfg_path}\n")

    if args.no_run:
        print("  --no-run flag set. Done.\n")
        return

    v2g, sim = apply_config(cfg)

    if args.year_sim:
        print("  Starting full year simulation...\n")
        from run_year_simulation import (run_year, print_annual_summary,
                                         plot_year_summary, plot_weekly_heatmap,
                                         PriceLoader)
        loader = PriceLoader(mode=sim["price_mode"])
        df = run_year(year=sim["year"], v2g=v2g, loader=loader,
                      fleet_size=sim["fleet_size"],
                      soc_init_pct=sim["soc_init_pct"],
                      soc_final_pct=sim["soc_final_pct"],
                      dwell=sim["dwell"])
        df.to_csv("data/year_simulation_results.csv", index=False)
        print_annual_summary(df, sim["year"], sim["fleet_size"])
        plot_year_summary(df, sim["year"], sim["fleet_size"])
        plot_weekly_heatmap(df, sim["year"])

    else:
        print("  Starting single-day optimisation...\n")
        import numpy as np
        from run_optimisation import (
            load_prices, build_load_and_availability, load_deg_sensitivity,
            run_dumb, run_smart_no_v2g, run_milp_day_ahead, run_mpc_day_ahead,
            deg_sensitivity, fleet_scaling, print_report, plot_all,
        )
        tru, plugged = build_load_and_availability(v2g, dwell=sim["dwell"])
        hours        = np.arange(v2g.n_slots) * v2g.dt_h
        deg_values   = load_deg_sensitivity(v2g)
        buy, v2g_p, price_source = load_prices(v2g, season=sim["season"])

        print(f"  Season: {sim['season'].upper()}  |  Prices: {price_source}")
        A = run_dumb(v2g, buy, v2g_p, tru, plugged, sim["soc_init_pct"], sim["soc_final_pct"])
        B = run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged, sim["soc_init_pct"], sim["soc_final_pct"])
        C = run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, sim["soc_init_pct"], sim["soc_final_pct"])
        D = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, sim["soc_init_pct"], sim["soc_final_pct"],
                              forecast_noise_std=0.0, label="D - MPC (perfect)")
        E = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, sim["soc_init_pct"], sim["soc_final_pct"],
                              forecast_noise_std=v2g.mpc_price_noise_std,
                              label="E - MPC (noisy)", seed=42)

        results  = {"A": A, "B": B, "C": C, "D": D, "E": E}
        deg_df   = deg_sensitivity(v2g, buy, v2g_p, tru, plugged, deg_values,
                                   sim["soc_init_pct"], sim["soc_final_pct"])
        fleet_df = fleet_scaling(C, D, fleet_sizes=[1, 5, 10, 25, 50])

        print_report(results, fleet_df, deg_df,
                     season=sim["season"], price_source=price_source)
        out = f"results_{sim['season']}_gui.png"
        plot_all(hours, A, B, C, D, E, deg_df, fleet_df, season=sim["season"], out=out)
        print(f"\n  Chart saved -> {out}\n  Done.\n")


if __name__ == "__main__":
    main()