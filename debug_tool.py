#!/usr/bin/env python3

from __future__ import annotations
import json, sys, traceback, threading, webbrowser, socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_optimisation import (
        V2GParams, load_battery_params, load_prices,
        build_load_and_availability,
        run_dumb, run_smart_no_v2g, run_milp_day_ahead, run_mpc_day_ahead,
    )
    import numpy as np
    IMPORT_OK    = True
    IMPORT_ERROR = ""
except Exception:
    IMPORT_OK    = False
    IMPORT_ERROR = traceback.format_exc()
    import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — BATTERY  (each row: param, value, status, formula with numbers)
# ═══════════════════════════════════════════════════════════════════════════
def battery_rows(v2g):
    rte = v2g.eta_charge * v2g.eta_discharge
    up  = v2g.usable_capacity_kWh / v2g.battery_capacity_kWh * 100
    return [
        {"param": "Battery capacity",
         "value": f"{v2g.battery_capacity_kWh} kWh",
         "st":    "PASS" if 50 <= v2g.battery_capacity_kWh <= 150 else "FAIL",
         "formula": f"V2GParams.battery_capacity_kWh = {v2g.battery_capacity_kWh} kWh   [valid: 50-150 kWh]"},

        {"param": "Usable capacity",
         "value": f"{v2g.usable_capacity_kWh} kWh ({up:.1f}%)",
         "st":    "PASS" if 50 <= up <= 90 else "WARN",
         "formula": f"usable = {v2g.usable_capacity_kWh} kWh = {up:.1f}% of {v2g.battery_capacity_kWh} kWh   [valid: 50-90%]"},

        {"param": "SOC min",
         "value": f"{v2g.soc_min_pct} %",
         "st":    "PASS",
         "formula": f"V2GParams.soc_min_pct = {v2g.soc_min_pct}%   [cold-chain hard floor]"},

        {"param": "SOC max",
         "value": f"{v2g.soc_max_pct} %",
         "st":    "PASS",
         "formula": f"V2GParams.soc_max_pct = {v2g.soc_max_pct}%   [charge ceiling - limits degradation]"},

        {"param": "E_min",
         "value": f"{v2g.E_min:.2f} kWh",
         "st":    "PASS" if v2g.E_min > 0 else "FAIL",
         "formula": f"E_min = soc_min/100 x usable = {v2g.soc_min_pct}/100 x {v2g.usable_capacity_kWh} = {v2g.E_min:.2f} kWh"},

        {"param": "E_max",
         "value": f"{v2g.E_max:.2f} kWh",
         "st":    "PASS" if v2g.E_max > v2g.E_min else "FAIL",
         "formula": f"E_max = soc_max/100 x usable = {v2g.soc_max_pct}/100 x {v2g.usable_capacity_kWh} = {v2g.E_max:.2f} kWh"},

        {"param": "Charge power",
         "value": f"{v2g.charge_power_kW} kW",
         "st":    "PASS" if 10 <= v2g.charge_power_kW <= 50 else "WARN",
         "formula": f"V2GParams.charge_power_kW = {v2g.charge_power_kW} kW   [ISO 15118 AC max 22 kW; valid: 10-50 kW]"},

        {"param": "Discharge power",
         "value": f"{v2g.discharge_power_kW} kW",
         "st":    "PASS" if 10 <= v2g.discharge_power_kW <= 50 else "WARN",
         "formula": f"V2GParams.discharge_power_kW = {v2g.discharge_power_kW} kW   [V2G export limit; valid: 10-50 kW]"},

        {"param": "eta_charge",
         "value": str(v2g.eta_charge),
         "st":    "PASS" if 0.85 <= v2g.eta_charge <= 0.99 else "WARN",
         "formula": f"V2GParams.eta_charge = {v2g.eta_charge}   [AC to DC; valid: 0.85-0.99]"},

        {"param": "eta_discharge",
         "value": str(v2g.eta_discharge),
         "st":    "PASS" if 0.85 <= v2g.eta_discharge <= 0.99 else "WARN",
         "formula": f"V2GParams.eta_discharge = {v2g.eta_discharge}   [DC to AC; valid: 0.85-0.99]"},

        {"param": "Round-trip efficiency",
         "value": f"{rte * 100:.1f} %",
         "st":    "PASS" if 0.70 <= rte <= 0.95 else "WARN",
         "formula": f"RTE = eta_c x eta_d = {v2g.eta_charge} x {v2g.eta_discharge} = {rte:.4f} = {rte*100:.1f}%   [valid: 70-95%]"},

        {"param": "Degradation cost",
         "value": f"EUR {v2g.deg_cost_eur_kwh}/kWh",
         "st":    "PASS" if 0.01 <= v2g.deg_cost_eur_kwh <= 0.20 else "WARN",
         "formula": f"V2GParams.deg_cost_eur_kwh = EUR {v2g.deg_cost_eur_kwh}/kWh   [Agora 2025: ~EUR 0.07; valid: 0.01-0.20]"},

        {"param": "dt",
         "value": f"{v2g.dt_h} h",
         "st":    "INFO",
         "formula": f"V2GParams.dt_h = {v2g.dt_h} h   [15 min per slot x 96 slots = 24 h]"},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — PRICES  (weekday + weekend side-by-side)
# ═══════════════════════════════════════════════════════════════════════════
def _load_price_pair(v2g, season):
    def _load(wd_flag):
        try:
            b, p, _ = load_prices(v2g, season=season, weekday=wd_flag)
            return b, p
        except TypeError:
            b, p, _ = load_prices(v2g, season=season)
            if not wd_flag:
                b = np.clip(b * 0.92, 0.005, None)
                p = np.clip(p * 0.92, 0.005, None)
            return b, p
    bwd, vwd = _load(True)
    bwe, vwe = _load(False)
    return bwd, vwd, bwe, vwe


def price_step_data(v2g, season):
    bwd, vwd, bwe, vwe = _load_price_pair(v2g, season)

    def _rng(a): return "PASS" if 0.01 <= a.min() and a.max() <= 0.60 else "WARN"
    def _spr(a): return "PASS" if a.max() - a.min() >= 0.05 else "WARN"
    def _prm(b,v): return "PASS" if (v-b).max() >= 0.05 else "WARN"

    summary = [
        {"Metric": "Buy min (EUR/kWh)",
         "Weekday": f"{bwd.min():.4f}", "wd_st": _rng(bwd),
         "Weekend": f"{bwe.min():.4f}", "we_st": _rng(bwe),
         "wd_f": f"min(buy_wd[0..95]) = {bwd.min():.4f}   [valid: >= 0.01]",
         "we_f": f"min(buy_we[0..95]) = {bwe.min():.4f}   [valid: >= 0.01]"},

        {"Metric": "Buy max (EUR/kWh)",
         "Weekday": f"{bwd.max():.4f}", "wd_st": _rng(bwd),
         "Weekend": f"{bwe.max():.4f}", "we_st": _rng(bwe),
         "wd_f": f"max(buy_wd[0..95]) = {bwd.max():.4f}   [valid: <= 0.60]",
         "we_f": f"max(buy_we[0..95]) = {bwe.max():.4f}   [valid: <= 0.60]"},

        {"Metric": "Buy avg (EUR/kWh)",
         "Weekday": f"{bwd.mean():.4f}", "wd_st": "INFO",
         "Weekend": f"{bwe.mean():.4f}", "we_st": "INFO",
         "wd_f": f"mean(buy_wd) = sum(buy_wd)/96 = {bwd.mean():.4f}",
         "we_f": f"mean(buy_we) = sum(buy_we)/96 = {bwe.mean():.4f}"},

        {"Metric": "Daily spread (EUR/kWh)",
         "Weekday": f"{bwd.max()-bwd.min():.4f}", "wd_st": _spr(bwd),
         "Weekend": f"{bwe.max()-bwe.min():.4f}", "we_st": _spr(bwe),
         "wd_f": f"spread = {bwd.max():.4f} - {bwd.min():.4f} = {bwd.max()-bwd.min():.4f}   [need >= 0.05 for arbitrage]",
         "we_f": f"spread = {bwe.max():.4f} - {bwe.min():.4f} = {bwe.max()-bwe.min():.4f}   [need >= 0.05 for arbitrage]"},

        {"Metric": "V2G sell max (EUR/kWh)",
         "Weekday": f"{vwd.max():.4f}", "wd_st": "INFO",
         "Weekend": f"{vwe.max():.4f}", "we_st": "INFO",
         "wd_f": f"max(v2g_wd) = {vwd.max():.4f} = buy_max + FCR/aFRR premium",
         "we_f": f"max(v2g_we) = {vwe.max():.4f} = buy_max + FCR/aFRR premium"},

        {"Metric": "Max FCR premium (EUR/kWh)",
         "Weekday": f"{(vwd-bwd).max():.4f}", "wd_st": _prm(bwd,vwd),
         "Weekend": f"{(vwe-bwe).max():.4f}", "we_st": _prm(bwe,vwe),
         "wd_f": f"max(v2g_wd - buy_wd) = {(vwd-bwd).max():.4f}   [valid: >= 0.05]",
         "we_f": f"max(v2g_we - buy_we) = {(vwe-bwe).max():.4f}   [valid: >= 0.05]"},

        {"Metric": "Negative price slots",
         "Weekday": f"{int((bwd<0).sum())}/96",
         "wd_st": "INFO" if int((bwd<0).sum())==0 else "PASS",
         "Weekend": f"{int((bwe<0).sum())}/96",
         "we_st": "INFO" if int((bwe<0).sum())==0 else "PASS",
         "wd_f": f"count(buy_wd < 0) = {int((bwd<0).sum())} slots   [negative = charge free or get paid]",
         "we_f": f"count(buy_we < 0) = {int((bwe<0).sum())} slots"},
    ]

    hourly = []
    for i in range(0, 96, 4):
        h = i * 0.25
        hourly.append({
            "Time":    f"{int(h):02d}:00",
            "WD Buy":  f"{bwd[i]:.4f}",
            "WD V2G":  f"{vwd[i]:.4f}",
            "WD Prem": f"{vwd[i]-bwd[i]:.4f}",
            "WE Buy":  f"{bwe[i]:.4f}",
            "WE V2G":  f"{vwe[i]:.4f}",
            "WE Prem": f"{vwe[i]-bwe[i]:.4f}",
        })
    return summary, hourly, bwd, vwd


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════
SNAME = {"A":"A - Dumb","B":"B - Smart","C":"C - MILP","D":"D - MPC Pfct","E":"E - MPC Noisy"}


def scenario_comparison_rows(res, v2g):
    base = res.get("A", next(iter(res.values()))).cost_eur_day
    rows = []
    for k in ["A","B","C","D","E"]:
        if k not in res:
            continue
        r   = res[k]
        sm  = float(np.min(r.soc))
        mx  = int(np.sum((r.p_charge > 0.1) & (r.p_discharge > 0.1)))
        net    = round(r.cost_eur_day, 3)
        charge = round(r.charge_cost_eur_day, 3)
        v2gr   = round(r.v2g_revenue_eur_day, 3)
        deg    = round(r.deg_cost_eur_day, 3)
        vs_a   = round(net - base, 3)
        ann    = int(round(net * 365))
        annv   = int(round(v2gr * 365))
        rows.append({
            "key": k, "name": SNAME[k],
            "net": net, "vs_a": vs_a,
            "charge": charge, "v2g_rev": v2gr, "deg": deg,
            "v2g_kwh":   round(r.v2g_export_kwh_day, 1),
            "soc_min":   round(sm, 1),
            "soc_ok":    sm >= v2g.E_min - 0.05,
            "soc_final": round(float(r.soc[-1]), 1),
            "annual_net": ann, "annual_v2g": annv,
            "mutex_ok":  mx == 0,
            # formula strings (numbers substituted in)
            "f_net":     f"charge - v2g_rev + deg = {charge} - {v2gr} + {deg} = {net} EUR/day",
            "f_vs_a":    f"net - A_baseline = {net} - {round(base,3)} = {vs_a} EUR/day",
            "f_charge":  f"sum(p_charge[t] x buy[t] x 0.25h) over 96 slots = {charge} EUR/day",
            "f_v2g_rev": f"sum(p_discharge[t] x v2g_price[t] x 0.25h) over 96 slots = {v2gr} EUR/day",
            "f_deg":     f"sum((p_c[t]+p_d[t]) x {v2g.deg_cost_eur_kwh} x 0.25h) = {deg} EUR/day",
            "f_v2g_kwh": f"sum(p_discharge[t] x 0.25h) over 96 slots = {round(r.v2g_export_kwh_day,1)} kWh/day",
            "f_soc_min": f"min(SOC[0..96]) = {round(sm,1)} kWh   [E_min limit = {v2g.E_min:.2f} kWh]",
            "f_soc_fin": f"SOC[96] = {round(float(r.soc[-1]),1)} kWh   [target ~ {v2g.usable_capacity_kWh*0.8:.1f} kWh]",
            "f_ann_net": f"{net} EUR/day x 365 days = {ann} EUR/yr",
            "f_ann_v2g": f"{v2gr} EUR/day x 365 days = {annv} EUR/yr",
        })
    return rows


def ordering_check_rows(res):
    c = {k: res[k].cost_eur_day for k in res}
    rows = []
    for rule, k1, k2, question in [
        ("A >= B","A","B","Smart beats dumb?"),
        ("B >= C","B","C","V2G beats smart-only?"),
        ("C ~ D", "C","D","MPC tracks MILP?"),
        ("D <= E","D","E","Noise hurts MPC?"),
    ]:
        if k1 not in c or k2 not in c: continue
        if "~" in rule:
            gap = abs(c[k2]-c[k1])/(abs(c[k1])+1e-9)*100
            st  = "PASS" if gap <= 10 else "WARN"
            val = f"C={c[k1]:.3f}  D={c[k2]:.3f}  gap={gap:.1f}%"
        elif ">=" in rule:
            st  = "PASS" if c[k1] >= c[k2]-0.001 else "FAIL"
            val = f"{k1}={c[k1]:.3f}  {k2}={c[k2]:.3f}"
        else:
            st  = "PASS" if c[k1] <= c[k2]+0.001 else "WARN"
            val = f"{k1}={c[k1]:.3f}  {k2}={c[k2]:.3f}"
        rows.append({"rule":rule,"question":question,"values":val,"st":st})
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DEBUG RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def run_debug(cfg):
    out = {"steps":[],"error":None}
    if not IMPORT_OK:
        out["error"] = f"Cannot import run_optimisation.py:\n{IMPORT_ERROR}"
        return out

    season   = cfg.get("season","winter")
    soc_init = float(cfg.get("soc_init",45.0))
    soc_fin  = float(cfg.get("soc_fin",80.0))
    arr_h    = float(cfg.get("arrival_h",21.0))
    dep_h    = float(cfg.get("departure_h",7.0))
    run_scen = cfg.get("scenarios",[])
    steps    = cfg.get("steps",[])

    v2g = V2GParams()
    buy = v2g_p = tru = plugged = None

    def _ensure_prices():
        nonlocal buy, v2g_p
        if buy is None:
            _, _, buy, v2g_p = price_step_data(v2g, season)

    def _ensure_tru():
        nonlocal tru, plugged
        if tru is None:
            try:
                tru, plugged = build_load_and_availability(v2g, arrival_h=arr_h, departure_h=dep_h)
            except TypeError:
                tru, plugged = build_load_and_availability(v2g, dwell="NightOnly")

    try:
        if "battery" in steps:
            v2g, src = load_battery_params(V2GParams())
            out["steps"].append({
                "id":"battery","title":"Step 1 - Battery Parameters",
                "source":src, "rows":battery_rows(v2g)
            })

        if "prices" in steps:
            summary, hourly, buy, v2g_p = price_step_data(v2g, season)
            out["steps"].append({
                "id":"prices",
                "title":f"Step 2 - Price Profile  {season.capitalize()}  Weekday vs Weekend",
                "summary":summary,"hourly":hourly
            })

        if "tru" in steps:
            _ensure_tru()
            ph = float(np.sum(plugged)*v2g.dt_h)
            tt = float(np.sum(tru)*v2g.dt_h)
            tbl = [{"Time":f"{int(i*0.25):02d}:00",
                    "TRU (kW)":round(float(tru[i]),3),
                    "Plugged":"YES" if plugged[i]>0.5 else "-"}
                   for i in range(0,96,4)]
            out["steps"].append({
                "id":"tru","title":"Step 3 - TRU Load & Availability",
                "arrival":arr_h,"departure":dep_h,
                "plug_h":round(ph,1),"plug_st":"PASS" if ph>=8 else "WARN",
                "tru_kwh":round(tt,2),"tru_st":"PASS" if 30<=tt<=100 else "WARN",
                "tru_min":round(float(tru.min()),3),"tru_max":round(float(tru.max()),3),
                "f_plug":f"sum(plugged[t]) x 0.25h = {round(ph,1)}h   [valid: >= 8h]",
                "f_tru": f"sum(tru[t] x 0.25h) = {round(tt,2)} kWh/day   [valid: 30-100 kWh]",
                "table":tbl
            })

        if "energy_balance" in steps:
            _ensure_tru()
            Ei = v2g.usable_capacity_kWh * soc_init / 100
            Ef = v2g.usable_capacity_kWh * soc_fin  / 100
            tt = float(np.sum(tru)*v2g.dt_h)
            dE = Ef - Ei
            lk = (tt+dE)*0.10
            ep = tt+dE+lk
            rows = [
                {"Item":"Arrival energy",     "Value":f"{Ei:.2f} kWh",
                 "st":"PASS" if Ei>=v2g.E_min else "FAIL",
                 "formula":f"E_arrive = {soc_init}/100 x {v2g.usable_capacity_kWh} = {Ei:.2f} kWh   [must be >= E_min={v2g.E_min:.2f}]"},
                {"Item":"Departure target",   "Value":f"{Ef:.2f} kWh",
                 "st":"PASS" if Ef<=v2g.E_max else "FAIL",
                 "formula":f"E_depart = {soc_fin}/100 x {v2g.usable_capacity_kWh} = {Ef:.2f} kWh   [must be <= E_max={v2g.E_max:.2f}]"},
                {"Item":"TRU daily energy",   "Value":f"{tt:.2f} kWh",
                 "st":"PASS" if 30<=tt<=100 else "WARN",
                 "formula":f"sum(tru[t] x 0.25h) = {tt:.2f} kWh   [valid: 30-100 kWh]"},
                {"Item":"SOC top-up needed",  "Value":f"{dE:.2f} kWh",  "st":"INFO",
                 "formula":f"delta_E = E_depart - E_arrive = {Ef:.2f} - {Ei:.2f} = {dE:.2f} kWh"},
                {"Item":"Losses est. (~10%)", "Value":f"{lk:.2f} kWh",  "st":"INFO",
                 "formula":f"losses = (TRU + delta_E) x 0.10 = ({tt:.2f} + {dE:.2f}) x 0.10 = {lk:.2f} kWh"},
                {"Item":"Expected grid purchase","Value":f"{ep:.1f} kWh",
                 "st":"PASS" if 50<=ep<=150 else "WARN",
                 "formula":f"E_grid = {tt:.2f} + {dE:.2f} + {lk:.2f} = {ep:.1f} kWh   [valid: 50-150 kWh]"},
            ]
            out["steps"].append({"id":"energy_balance","title":"Step 4 - Energy Balance Pre-flight","rows":rows})

        if run_scen:
            _ensure_prices(); _ensure_tru()
            fns = {
                "A": lambda: run_dumb(v2g,buy,v2g_p,tru,plugged,soc_init,soc_fin),
                "B": lambda: run_smart_no_v2g(v2g,buy,v2g_p,tru,plugged,soc_init,soc_fin),
                "C": lambda: run_milp_day_ahead(v2g,buy,v2g_p,tru,plugged,soc_init,soc_fin),
                "D": lambda: run_mpc_day_ahead(v2g,buy,v2g_p,tru,plugged,soc_init,soc_fin,forecast_noise_std=0.0),
                "E": lambda: run_mpc_day_ahead(v2g,buy,v2g_p,tru,plugged,soc_init,soc_fin,forecast_noise_std=0.012),
            }
            res = {k:fns[k]() for k in ["A","B","C","D","E"] if k in run_scen}
            out["steps"].append({
                "id":"scenarios",
                "title":f"Step 5 - Scenario Comparison  {season.capitalize()}",
                "comparison":scenario_comparison_rows(res,v2g),
                "ordering":ordering_check_rows(res),
            })

    except Exception:
        out["error"] = traceback.format_exc()

    return out


# ═══════════════════════════════════════════════════════════════════════════
# HTML PAGE
# ═══════════════════════════════════════════════════════════════════════════
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V2G Debug Tool - S.KOe COOL</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#1a202c;min-height:100vh}
  header{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;padding:14px 26px;
    display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:1.2rem;font-weight:700}
  header p{font-size:.76rem;opacity:.8;margin-top:2px}
  .badge{background:rgba(255,255,255,.18);border-radius:5px;padding:3px 10px;
    font-size:.7rem;font-weight:700;letter-spacing:.5px}

  .layout{display:grid;grid-template-columns:270px 1fr;min-height:calc(100vh - 56px)}
  .sidebar{background:#fff;border-right:1px solid #e2e8f0;padding:13px;overflow-y:auto}
  .content{display:flex;flex-direction:column;overflow:hidden}

  /* sidebar cards */
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:7px;padding:11px;
    margin-bottom:9px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .card-title{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
    color:#64748b;margin-bottom:7px;padding-bottom:4px;border-bottom:1px solid #f1f5f9}
  .field{margin-bottom:7px}
  .field label{display:block;font-size:.74rem;font-weight:600;color:#6b7280;margin-bottom:2px}
  .field select,.field input[type=number]{width:100%;border:1px solid #d1d5db;
    border-radius:5px;padding:5px 8px;font-size:.82rem;color:#1a202c;background:#fafafa}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:7px}
  .cb-row{display:flex;align-items:center;gap:6px;padding:3px 0}
  .cb-row label{font-size:.82rem;color:#374151;cursor:pointer;user-select:none}
  input[type=checkbox]{width:14px;height:14px;accent-color:#2563eb;cursor:pointer;flex-shrink:0}
  .sel-all{font-size:.67rem;color:#2563eb;cursor:pointer;text-decoration:underline;
    background:none;border:none;padding:0 0 0 4px}

  /* formula toggle */
  .fm-row{display:flex;align-items:center;justify-content:space-between;
    background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;padding:8px 10px;margin-bottom:9px}
  .fm-label{font-size:.78rem;font-weight:700;color:#0c4a6e}
  .fm-sub{font-size:.68rem;color:#0369a1;margin-top:1px}
  .toggle-wrap{display:flex;align-items:center;gap:6px}
  .toggle{position:relative;width:40px;height:22px;flex-shrink:0}
  .toggle input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:#cbd5e1;border-radius:22px;
    cursor:pointer;transition:.2s}
  .slider:before{content:'';position:absolute;height:16px;width:16px;left:3px;bottom:3px;
    background:#fff;border-radius:50%;transition:.2s}
  input:checked+.slider{background:#2563eb}
  input:checked+.slider:before{transform:translateX(18px)}
  .toggle-lbl{font-size:.72rem;font-weight:700;color:#2563eb;min-width:36px}

  .run-btn{width:100%;padding:10px;background:linear-gradient(135deg,#2563eb,#1d4ed8);
    color:#fff;border:none;border-radius:7px;font-size:.9rem;font-weight:700;
    cursor:pointer;transition:.15s}
  .run-btn:hover{background:linear-gradient(135deg,#1d4ed8,#1e40af);transform:translateY(-1px)}
  .run-btn:disabled{background:#9ca3af;cursor:not-allowed;transform:none}

  /* tab bar */
  .tab-bar{display:flex;background:#fff;border-bottom:2px solid #e2e8f0;padding:0 18px;flex-shrink:0}
  .tab-btn{padding:11px 18px;font-size:.85rem;font-weight:600;color:#64748b;
    border:none;background:none;cursor:pointer;border-bottom:2px solid transparent;
    margin-bottom:-2px;transition:.15s}
  .tab-btn:hover{color:#2563eb}
  .tab-btn.active{color:#2563eb;border-bottom-color:#2563eb}

  /* main content area */
  .tab-panel{padding:16px;overflow-y:auto;flex:1}
  .tab-panel.hidden{display:none}

  /* result blocks */
  .step-block{background:#fff;border:1px solid #e2e8f0;border-radius:8px;
    margin-bottom:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05)}
  .step-hdr{padding:10px 15px;background:linear-gradient(90deg,#f8fafc,#f1f5f9);
    border-bottom:1px solid #e2e8f0;display:flex;align-items:center;gap:9px}
  .step-hdr h3{font-size:.88rem;font-weight:700;color:#1e3a5f}
  .src-tag{background:#dbeafe;color:#1e40af;border-radius:4px;padding:2px 7px;
    font-size:.66rem;font-weight:600}
  .step-body{padding:13px}
  .note{font-size:.76rem;color:#64748b;margin-bottom:7px}

  /* tables */
  .tw{overflow-x:auto;margin-top:5px}
  table{width:100%;border-collapse:collapse;font-size:.80rem}
  th{background:#f8fafc;text-align:left;padding:6px 9px;font-weight:600;
    color:#374151;border-bottom:2px solid #e2e8f0;white-space:nowrap}
  td{padding:5px 9px;border-bottom:1px solid #f1f5f9;color:#374151}
  tr:hover td{background:#f9fafb}
  td.r{text-align:right;font-family:'Courier New',monospace;color:#1e3a5f;font-weight:600}
  td.grn{color:#16a34a!important;font-weight:700}
  td.red{background:#fef2f2!important;color:#dc2626!important;font-weight:700}
  td.amb{color:#d97706!important;font-weight:600}

  /* formula mode — cells with formula data look different */
  .fm-active td[data-f]{background:#fffbeb;font-style:italic;color:#78350f!important;
    font-family:'Courier New',monospace;font-size:.75rem}

  /* KPI pills */
  .kpi-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
  .kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 12px}
  .kpi-val{font-size:1.1rem;font-weight:800;color:#1e3a5f}
  .kpi-lbl{font-size:.66rem;color:#64748b;margin-top:1px}
  .kpi.ok   .kpi-val{color:#16a34a}
  .kpi.warn .kpi-val{color:#d97706}

  .ord-box{margin-top:10px;background:#f0f9ff;border:1px solid #bae6fd;
    border-radius:6px;padding:10px}
  .ord-title{font-size:.72rem;font-weight:700;color:#0c4a6e;margin-bottom:5px}

  /* reference tab */
  .ref-intro{background:#eff6ff;border:1px solid #bfdbfe;border-radius:7px;
    padding:12px 15px;margin-bottom:14px;font-size:.82rem;color:#1e40af;line-height:1.6}
  .ref-section{background:#fff;border:1px solid #e2e8f0;border-radius:8px;
    margin-bottom:14px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  .ref-hdr{padding:10px 15px;background:linear-gradient(90deg,#f8fafc,#f1f5f9);
    border-bottom:1px solid #e2e8f0;font-size:.88rem;font-weight:700;color:#1e3a5f;
    display:flex;align-items:center;gap:8px}
  .ref-body{padding:13px}
  .ref-note{font-size:.74rem;color:#64748b;margin-bottom:7px;line-height:1.5}
  th.fm{background:#fef9c3;color:#713f12}
  td.fm{color:#92400e;font-style:italic;font-size:.78rem}
  td.rng{color:#1e40af;font-family:'Courier New',monospace;font-size:.78rem}
  td.src{color:#16a34a;font-size:.76rem}

  #placeholder{color:#94a3b8;text-align:center;padding:50px 20px}
  #placeholder .ico{font-size:2.5rem;margin-bottom:9px}
  .err{background:#fef2f2;border:1px solid #fca5a5;border-radius:7px;padding:13px;
    color:#991b1b;font-family:monospace;font-size:.74rem;white-space:pre-wrap}
  .spinner{text-align:center;padding:40px;color:#6b7280;font-size:.92rem;display:none}
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

  <!-- SIDEBAR -->
  <div class="sidebar">

    <div class="card">
      <div class="card-title">&#x2699;&#xFE0F; Simulation Settings</div>
      <div class="field">
        <label>Season</label>
        <select id="season">
          <option value="winter" selected>Winter</option>
          <option value="summer">Summer</option>
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

    <div class="card">
      <div class="card-title">&#x1F50C; Night Dwell Window</div>
      <div class="field row2">
        <div><label>Arrival (h)</label>
          <input type="number" id="arrival_h" value="21" min="0" max="23" step="0.5"></div>
        <div><label>Departure (h)</label>
          <input type="number" id="departure_h" value="7" min="0" max="23" step="0.5"></div>
      </div>
      <p style="font-size:.68rem;color:#94a3b8;margin-top:2px">Plugged arrival &#x2192; departure (wraps midnight)</p>
    </div>

    <!-- FORMULA TOGGLE -->
    <div class="fm-row">
      <div>
        <div class="fm-label">&#x1F9EE; Formula Mode</div>
        <div class="fm-sub">Replaces values with calculations</div>
      </div>
      <div class="toggle-wrap">
        <label class="toggle">
          <input type="checkbox" id="fmToggle">
          <span class="slider"></span>
        </label>
        <span class="toggle-lbl" id="fmLbl">OFF</span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">&#x1F4CB; Debug Steps
        <button class="sel-all" onclick="selAll('step')">all</button></div>
      <div class="cb-row"><input type="checkbox" id="step_battery" class="step" checked>
        <label for="step_battery">1 &middot; Battery parameters</label></div>
      <div class="cb-row"><input type="checkbox" id="step_prices" class="step" checked>
        <label for="step_prices">2 &middot; Price data (EPEX)</label></div>
      <div class="cb-row"><input type="checkbox" id="step_tru" class="step" checked>
        <label for="step_tru">3 &middot; TRU load &amp; availability</label></div>
      <div class="cb-row"><input type="checkbox" id="step_energy_balance" class="step" checked>
        <label for="step_energy_balance">4 &middot; Energy balance pre-flight</label></div>
    </div>

    <div class="card">
      <div class="card-title">&#x26A1; Scenarios
        <button class="sel-all" onclick="selAll('scen')">all</button></div>
      <div class="cb-row"><input type="checkbox" id="scen_A" class="scen" checked>
        <label for="scen_A">A &#x2014; Dumb (baseline)</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_B" class="scen" checked>
        <label for="scen_B">B &#x2014; Smart, no V2G</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_C" class="scen" checked>
        <label for="scen_C">C &#x2014; MILP Day-Ahead</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_D" class="scen">
        <label for="scen_D">D &#x2014; MPC Perfect</label></div>
      <div class="cb-row"><input type="checkbox" id="scen_E" class="scen">
        <label for="scen_E">E &#x2014; MPC Noisy</label></div>
    </div>

    <button class="run-btn" id="runBtn" onclick="runDebug()">
      &#x25B6; Run Selected Debug Steps
    </button>
  </div>

  <!-- CONTENT AREA -->
  <div class="content">
    <div class="tab-bar">
      <button class="tab-btn active" id="tab-dbg-btn" onclick="showTab('dbg')">
        &#x1F50D; Debug Results
      </button>
      <button class="tab-btn" id="tab-ref-btn" onclick="showTab('ref')">
        &#x1F4D6; Parameter Reference
      </button>
    </div>

    <!-- DEBUG PANEL -->
    <div class="tab-panel" id="tab-dbg">
      <div id="placeholder">
        <div class="ico">&#x1F52C;</div>
        <strong>Select steps and scenarios, then click Run.</strong><br><br>
        Toggle <em>Formula Mode</em> to see calculations behind every value.
      </div>
      <div class="spinner" id="spinner">&#x23F3; Running &#x2014; please wait&hellip;</div>
    </div>

    <!-- REFERENCE PANEL -->
    <div class="tab-panel hidden" id="tab-ref"></div>
  </div>
</div>

<script>
// ── globals ───────────────────────────────────────────────────────────────
let formulaMode = false;

// ── formula toggle ────────────────────────────────────────────────────────
document.getElementById('fmToggle').addEventListener('change', function() {
  formulaMode = this.checked;
  document.getElementById('fmLbl').textContent = formulaMode ? 'ON' : 'OFF';
  const dbg = document.getElementById('tab-dbg');
  if (formulaMode) {
    dbg.classList.add('fm-active');
  } else {
    dbg.classList.remove('fm-active');
  }
  // swap displayed text in all cells that have formula data
  document.querySelectorAll('td[data-f]').forEach(td => {
    td.textContent = formulaMode ? td.dataset.f : td.dataset.v;
  });
});

// ── tab switching ─────────────────────────────────────────────────────────
function showTab(name) {
  ['dbg','ref'].forEach(t => {
    document.getElementById('tab-'+t).classList.toggle('hidden', t !== name);
    document.getElementById('tab-'+t+'-btn').classList.toggle('active', t === name);
  });
  if (name === 'ref' && !document.getElementById('tab-ref').innerHTML.trim()) {
    document.getElementById('tab-ref').innerHTML = buildReference();
  }
}

// ── sidebar helpers ───────────────────────────────────────────────────────
function selAll(cls) {
  document.querySelectorAll('.'+cls).forEach(cb => cb.checked = true);
}
function getChecked(cls) {
  return [...document.querySelectorAll('.'+cls)]
    .filter(c => c.checked)
    .map(c => c.id.replace('step_','').replace('scen_',''));
}

// ── run ───────────────────────────────────────────────────────────────────
async function runDebug() {
  const btn = document.getElementById('runBtn');
  const sp  = document.getElementById('spinner');
  const dbg = document.getElementById('tab-dbg');
  const cfg = {
    season:      document.getElementById('season').value,
    soc_init:    document.getElementById('soc_init').value,
    soc_fin:     document.getElementById('soc_fin').value,
    arrival_h:   document.getElementById('arrival_h').value,
    departure_h: document.getElementById('departure_h').value,
    steps:       getChecked('step'),
    scenarios:   getChecked('scen'),
  };
  btn.disabled = true;
  dbg.innerHTML = '';
  sp.style.display = 'block';
  dbg.appendChild(sp);
  showTab('dbg');
  try {
    const resp = await fetch('/run', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(cfg)});
    const data = await resp.json();
    sp.style.display = 'none';
    renderDebug(data, dbg);
  } catch(e) {
    sp.style.display = 'none';
    dbg.innerHTML = `<div class="err">Network error: ${e.message}</div>`;
  }
  btn.disabled = false;
}

// ── icon map ──────────────────────────────────────────────────────────────
const ICO = {PASS:'✅', WARN:'⚠️', FAIL:'❌', INFO:'ℹ️'};

// ── cell helpers ──────────────────────────────────────────────────────────
// fCell: cell that participates in formula toggle
// val = display in value mode, f = display in formula mode
function fCell(val, f, extraCls) {
  const fStr = (f || val).replace(/"/g, '&quot;');
  const vStr = String(val).replace(/"/g, '&quot;');
  const displayed = formulaMode ? (f || val) : val;
  const cls = 'r' + (extraCls ? ' '+extraCls : '');
  return `<td class="${cls}" data-v="${vStr}" data-f="${fStr}">${displayed}</td>`;
}

// plain cell (no formula toggle)
function pCell(val, cls) {
  return `<td${cls?' class="'+cls+'"':''}>${val}</td>`;
}

// ── table builders ────────────────────────────────────────────────────────

// 2-col table: param | value+icon  (with formula support)
function twoColTable(rows, paramKey, valKey, stKey, fKey) {
  let h = `<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>`;
  rows.forEach(r => {
    const st = r[stKey] || 'INFO';
    const disp = r[valKey] + '  ' + (ICO[st]||'');
    const form = fKey ? (r[fKey] + '  ' + (ICO[st]||'')) : disp;
    h += `<tr>${pCell(r[paramKey])}${fCell(disp, form)}</tr>`;
  });
  return h + '</tbody></table>';
}

// price summary: Metric | WD value+icon | WE value+icon (with formula)
function priceSummaryTable(rows) {
  let h = `<table><thead><tr><th>Metric</th><th>Weekday</th><th>Weekend</th></tr></thead><tbody>`;
  rows.forEach(r => {
    const wdDisp = r.Weekday + '  ' + (ICO[r.wd_st]||'');
    const weDisp = r.Weekend + '  ' + (ICO[r.we_st]||'');
    h += `<tr>
      ${pCell(r.Metric)}
      ${fCell(wdDisp, r.wd_f + '  ' + (ICO[r.wd_st]||''))}
      ${fCell(weDisp, r.we_f + '  ' + (ICO[r.we_st]||''))}
    </tr>`;
  });
  return h + '</tbody></table>';
}

// generic data table (no formula — for hourly prices, TRU table)
function genericTable(rows) {
  if (!rows || !rows.length) return '';
  const keys = Object.keys(rows[0]);
  let h = '<div class="tw"><table><thead><tr>' +
    keys.map(k => `<th>${k}</th>`).join('') +
    '</tr></thead><tbody>';
  rows.forEach(r => {
    h += '<tr>' + keys.map(k => `<td class="r">${r[k]}</td>`).join('') + '</tr>';
  });
  return h + '</tbody></table></div>';
}

// unified scenario comparison table
function scenarioTable(rows) {
  let h = `<div class="tw"><table><thead><tr>
    <th>Scenario</th>
    <th>Net EUR/day</th>
    <th title="vs Scenario A baseline">vs A</th>
    <th>Charge EUR/day</th>
    <th>V2G Rev EUR/day</th>
    <th>Deg EUR/day</th>
    <th>V2G kWh/day</th>
    <th>SOC min kWh</th>
    <th>SOC final kWh</th>
    <th>Annual EUR/yr</th>
    <th>V2G EUR/yr</th>
    <th>Checks</th>
  </tr></thead><tbody>`;

  rows.forEach(r => {
    const netC  = r.net < 0 ? 'grn' : '';
    const vaC   = r.vs_a < 0 ? 'grn' : (r.vs_a > 0 ? 'amb' : '');
    const socC  = r.soc_ok ? '' : 'red';
    const annC  = r.annual_net < 0 ? 'grn' : '';
    const chk   = (r.soc_ok?'✅':'❌')+' SOC  '+(r.mutex_ok?'✅':'❌')+' MX';
    h += `<tr>
      ${pCell('<strong>'+r.name+'</strong>')}
      ${fCell(r.net,    r.f_net,     netC)}
      ${fCell(r.key==='A'?'—':(r.vs_a>0?'+':'')+r.vs_a, r.f_vs_a, vaC)}
      ${fCell(r.charge, r.f_charge,  '')}
      ${fCell(r.v2g_rev,r.f_v2g_rev,'grn')}
      ${fCell(r.deg,    r.f_deg,     '')}
      ${fCell(r.v2g_kwh,r.f_v2g_kwh,'')}
      ${fCell(r.soc_min,r.f_soc_min, socC)}
      ${fCell(r.soc_final,r.f_soc_fin,'')}
      ${fCell(r.annual_net,r.f_ann_net,annC)}
      ${fCell(r.annual_v2g,r.f_ann_v2g,'grn')}
      ${pCell(chk,'r')}
    </tr>`;
  });
  return h + '</tbody></table></div>';
}

// ordering check table
function orderingTable(rows) {
  let h = `<table><thead><tr>
    <th>Rule</th><th>Question</th><th>Values</th><th>Status</th>
  </tr></thead><tbody>`;
  rows.forEach(r => {
    h += `<tr>
      ${pCell('<strong>'+r.rule+'</strong>')}
      ${pCell(r.question)}
      <td style="font-family:monospace;font-size:.76rem">${r.values}</td>
      ${pCell(ICO[r.st]||r.st, 'r')}
    </tr>`;
  });
  return h + '</tbody></table>';
}

// wrapper
function wrap(title, body, badge) {
  const b = badge ? `<span class="src-tag">${badge.slice(0,55)}</span>` : '';
  return `<div class="step-block">
    <div class="step-hdr"><h3>${title}</h3>${b}</div>
    <div class="step-body">${body}</div>
  </div>`;
}

// ── main debug renderer ───────────────────────────────────────────────────
function renderDebug(data, area) {
  area.innerHTML = '';
  if (data.error) { area.innerHTML = `<div class="err">Error:\n\n${data.error}</div>`; return; }
  if (!data.steps || !data.steps.length) {
    area.innerHTML = '<div class="err">No steps ran - select at least one step.</div>'; return;
  }

  for (const s of data.steps) {
    let body = '', badge = '';

    if (s.id === 'battery') {
      badge = s.source || '';
      body  = twoColTable(s.rows, 'param', 'value', 'st', 'formula');
    }

    else if (s.id === 'prices') {
      body = `<p class="note">Buy = EPEX spot + Netzentgelt + levies + VAT.
        V2G adds FCR/aFRR premium. Weekend uses ~8% lower prices automatically.</p>
        ${priceSummaryTable(s.summary)}
        <p style="font-size:.71rem;font-weight:600;color:#6b7280;margin:10px 0 3px">
          Hourly snapshot (top-of-hour, every 4th 15-min slot):</p>
        ${genericTable(s.hourly)}`;
    }

    else if (s.id === 'tru') {
      const pc = s.plug_st==='PASS'?'ok':'warn';
      const tc = s.tru_st ==='PASS'?'ok':'warn';
      const pDisp = s.plug_h + 'h  ' + (ICO[s.plug_st]||'');
      const tDisp = s.tru_kwh + ' kWh  ' + (ICO[s.tru_st]||'');
      body = `<div class="kpi-row">
        <div class="kpi ${pc}">
          <div class="kpi-val" data-v="${pDisp}" data-f="${s.f_plug}  ${ICO[s.plug_st]||''}"
            >${formulaMode ? s.f_plug : pDisp}</div>
          <div class="kpi-lbl">Plugged in (${s.arrival}:00 -> ${s.departure}:00)</div>
        </div>
        <div class="kpi ${tc}">
          <div class="kpi-val" data-v="${tDisp}" data-f="${s.f_tru}  ${ICO[s.tru_st]||''}"
            >${formulaMode ? s.f_tru : tDisp}</div>
          <div class="kpi-lbl">TRU energy today</div>
        </div>
        <div class="kpi">
          <div class="kpi-val">${s.tru_min}-${s.tru_max} kW</div>
          <div class="kpi-lbl">TRU power range</div>
        </div>
      </div>
      ${genericTable(s.table)}`;
    }

    else if (s.id === 'energy_balance') {
      body = `<p class="note">Physical feasibility before running any solver.
        Scenario A total charge should match Expected grid purchase.</p>
        ${twoColTable(s.rows, 'Item', 'Value', 'st', 'formula')}`;
    }

    else if (s.id === 'scenarios') {
      body = `<p class="note">
        Required ordering: <strong>A >= B >= C ~ D <= E</strong>
        &nbsp;&#x00B7;&nbsp; Green net = daily profit
        &nbsp;&#x00B7;&nbsp; <em>vs A</em> = saving vs dumb baseline (green = better)
        &nbsp;&#x00B7;&nbsp; Toggle Formula Mode to see every calculation
      </p>
      ${scenarioTable(s.comparison)}
      ${s.ordering && s.ordering.length ? `<div class="ord-box">
        <div class="ord-title">&#x1F4D0; Cross-Scenario Ordering Validation</div>
        ${orderingTable(s.ordering)}
      </div>` : ''}`;
    }

    const div = document.createElement('div');
    div.innerHTML = wrap(s.title, body, badge);
    area.appendChild(div.firstElementChild);
  }

  // re-apply formula mode if currently on
  if (formulaMode) {
    area.classList.add('fm-active');
    area.querySelectorAll('td[data-f]').forEach(td => td.textContent = td.dataset.f);
    area.querySelectorAll('[data-f].kpi-val').forEach(el => el.textContent = el.dataset.f);
  }
}


// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE TAB — static complete documentation
// ═══════════════════════════════════════════════════════════════════════════
function buildReference() {
  function refSection(icon, title, note, tableHtml) {
    return `<div class="ref-section">
      <div class="ref-hdr">${icon} ${title}</div>
      <div class="ref-body">
        <p class="ref-note">${note}</p>
        <div class="tw">${tableHtml}</div>
      </div>
    </div>`;
  }

  function refTable(headers, rows) {
    const ths = headers.map(h => `<th${h.fm?' class="fm"':''}>${h.label}</th>`).join('');
    const trs = rows.map(r => '<tr>' + r.map((c,i) => {
      const cls = headers[i].cls || '';
      return `<td${cls?' class="'+cls+'"':''}>${c}</td>`;
    }).join('') + '</tr>').join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
  }

  const H = [
    {label:'Parameter'}, {label:'Formula / Derivation', fm:true, cls:'fm'},
    {label:'Valid Range', cls:'rng'}, {label:'Source', cls:'src'}, {label:'Why it matters'}
  ];

  const bat = refSection('🔋','Step 1 — Battery Parameters',
    'These 13 values define the entire physical model of the battery system. ' +
    'E_min and E_max are derived from the SOC limits and feed directly into every MILP constraint.',
    refTable(H, [
      ['battery_capacity_kWh','V2GParams.battery_capacity_kWh  (direct)','50-150 kWh','make_data.py / S.KOe spec sheet','Total pack size; sets the absolute energy ceiling'],
      ['usable_capacity_kWh','usable = battery_capacity x usable_fraction','50-90% of total','make_data.py','Operating window; protects cell longevity at both extremes'],
      ['soc_min_pct','V2GParams.soc_min_pct  (direct)','Design-dependent','make_data.py','Cold-chain hard floor — below this the refrigeration unit cannot be guaranteed'],
      ['soc_max_pct','V2GParams.soc_max_pct  (direct)','80-95%','make_data.py','Prevents overcharge; reduces calendar degradation'],
      ['E_min (kWh)','E_min = soc_min_pct / 100 x usable_capacity_kWh','> 0 kWh','Derived','Hard lower constraint in MILP and MPC at every timestep'],
      ['E_max (kWh)','E_max = soc_max_pct / 100 x usable_capacity_kWh','> E_min','Derived','Hard upper constraint in MILP and MPC at every timestep'],
      ['charge_power_kW','V2GParams.charge_power_kW  (direct)','10-50 kW','ISO 15118 / charger spec','Limits charging speed; ISO 15118 AC Mode 3 max = 22 kW'],
      ['discharge_power_kW','V2GParams.discharge_power_kW  (direct)','10-50 kW','ISO 15118-20 / inverter spec','Limits V2G export power'],
      ['eta_charge','V2GParams.eta_charge  (direct)','0.85-0.99','Charger datasheet','AC to DC conversion loss; multiplies p_charge in energy balance'],
      ['eta_discharge','V2GParams.eta_discharge  (direct)','0.85-0.99','Inverter datasheet','DC to AC conversion loss; divides p_discharge in energy balance'],
      ['Round-trip efficiency','RTE = eta_charge x eta_discharge','70-95%','Derived','Net loss per V2G cycle; directly limits arbitrage profit'],
      ['deg_cost_eur_kwh','V2GParams.deg_cost_eur_kwh  (direct)','EUR 0.01-0.20/kWh','Agora 2025: ~EUR 0.07','Tipping point: V2G only profitable if revenue > degradation cost per kWh cycled'],
      ['dt_h','V2GParams.dt_h  (direct)','0.25 h (fixed)','EPEX 15-min settlement','Timestep for all 96-slot arrays; energy = power x dt_h'],
    ]));

  const prc = refSection('💶','Step 2 — Price Data (EPEX)',
    'All prices are in EUR/kWh for 15-minute slots. ' +
    'Buy price = spot + network fees + levies + VAT. ' +
    'V2G sell price = buy price + FCR/aFRR capacity premium. ' +
    'Weekend prices are ~8% lower due to reduced industrial demand.',
    refTable(H, [
      ['buy_price[t]','EPEX spot[t] + Netzentgelt + EEG-Umlage + VAT','EUR 0.01-0.60/kWh','make_data.py / EPEX Intraday','Cost of charging from grid at each 15-min slot'],
      ['v2g_price[t]','buy_price[t] + FCR/aFRR_premium[t]','>= buy_price','make_data.py / Regelenergiemarkt','Revenue from V2G discharge; premium reflects ancillary service value'],
      ['FCR premium','Capacity payment for frequency reserve availability','EUR 0.05-0.20/kWh','Regelenergiemarkt (regelleistung.net)','Main V2G revenue driver beyond pure arbitrage'],
      ['Daily spread','max(buy) - min(buy) over 96 slots','Need >= EUR 0.05 for arbitrage','Derived from EPEX data','Low spread = little V2G arbitrage opportunity; FCR revenue compensates'],
      ['Weekend discount','buy_we ~ 0.92 x buy_wd (fallback)','~5-15% lower','Empirical EPEX data','Lower industrial demand on weekends suppresses prices'],
      ['Negative price slots','count(buy[t] < 0)','0-15 slots/day typical','EPEX actual data','Negative price = charging is free or the grid pays you to consume'],
    ]));

  const tru = refSection('🧊','Step 3 — TRU Load & Dwell Availability',
    'The Thermo-King/Carrier refrigeration unit (TRU) runs continuously regardless of price. ' +
    'It is the non-negotiable baseline load. The optimizer must satisfy TRU demand at all times — ' +
    'it takes absolute priority over all charging/discharging decisions.',
    refTable(H, [
      ['tru[t] (kW)','Sinusoidal profile around base load; peaks in afternoon','1-6 kW per slot','make_data.py / TRU spec','Continuous cooling draw; cannot be curtailed for any reason'],
      ['TRU daily energy','sum(tru[t] x dt_h) over all plugged slots','30-100 kWh/day','Derived','Must be covered by battery regardless of prices'],
      ['plugged[t] (binary)','1 if arrival_h <= t <= departure_h, else 0','1 = plugged in','User input (dwell window)','Defines when smart charging / V2G is physically possible'],
      ['Dwell hours','sum(plugged[t]) x dt_h','Need >= 8h for meaningful V2G','User input','Longer dwell = more V2G flexibility and revenue opportunity'],
      ['arrival_h / departure_h','User-set; dwell window wraps midnight if arrival > departure','Any window >= 8h','User input in UI','Represents depot parking schedule; critical for revenue calculations'],
    ]));

  const eb = refSection('⚖️','Step 4 — Energy Balance Pre-flight',
    'A sanity check run BEFORE the solver. If these pass, the problem is physically feasible. ' +
    'After Scenario A runs, compare its actual grid purchase against Expected grid purchase — ' +
    'they should match within ~10%.',
    refTable(H, [
      ['E_arrive','soc_init / 100 x usable_capacity_kWh','Must be >= E_min','User input (arrival SOC %)','Starting SOC must be above cold-chain floor or trailer arrives in violation'],
      ['E_depart','soc_fin / 100 x usable_capacity_kWh','Must be <= E_max','User input (departure SOC %)','Departure target must be physically reachable within the charge window'],
      ['TRU daily energy','sum(tru[t] x dt_h)','30-100 kWh/day','Derived in Step 3','Non-negotiable energy that must come from the battery'],
      ['delta_E (SOC top-up)','E_depart - E_arrive  (can be negative)','Any value','Derived','Net energy needed from grid just for SOC change'],
      ['Losses est.','(TRU + delta_E) x 0.10','~10% of throughput','Assumed 10% for RTE < 1','Approximation; actual losses depend on exact charge/discharge profile'],
      ['Expected grid purchase','TRU + delta_E + losses','50-150 kWh/day typical','Derived','Baseline check: Scenario A actual charge should be close to this'],
    ]));

  const sc = refSection('⚡','Step 5 — Scenario Metrics',
    'All financial metrics are per-day. Negative net cost = the trailer earns more from V2G than it spends on electricity. ' +
    'Annual figures are simple extrapolations (x365) for business case illustration.',
    refTable(H, [
      ['Net cost (EUR/day)','charge_cost - v2g_revenue + deg_cost','Negative = daily profit','Solver output','Primary optimisation objective; lower = better'],
      ['vs A (EUR/day)','net_cost - A.net_cost','Negative = improvement over dumb baseline','Derived','Shows how much each smarter strategy saves vs doing nothing'],
      ['Charge cost','sum(p_charge[t] x buy_price[t] x dt_h)','>= 0','Solver output','Grid electricity bill for the day'],
      ['V2G revenue','sum(p_discharge[t] x v2g_price[t] x dt_h)','>= 0','Solver output','Income from exporting to grid; includes FCR premium'],
      ['Deg cost','sum((p_charge[t]+p_discharge[t]) x deg_cost_eur_kwh x dt_h)','>= 0','Solver output','Battery wear penalty; reduces V2G net revenue'],
      ['V2G kWh/day','sum(p_discharge[t] x dt_h)','0-80 kWh typical','Solver output','Total energy exported to grid; x v2g_price gives revenue'],
      ['SOC min (kWh)','min(SOC[0..96])','Must be >= E_min','Solver output','Cold-chain compliance; any value below E_min = FAIL'],
      ['SOC final (kWh)','SOC[96]','Should be ~ soc_fin% x usable','Solver output','Departure readiness; must reach target before truck leaves'],
      ['Annual EUR/yr','net_cost x 365 (extrapolation)','Varies widely','Derived','Business case indicator; Agora 2025 benchmark: EUR 500/yr for car'],
      ['Annual V2G EUR/yr','v2g_revenue x 365','EUR 200-3000 for trailer','Derived','Compare to Agora 2025: EUR 500/yr for private car (trailer should be higher)'],
      ['SOC check (MX)','min(SOC) >= E_min - 0.05 (tolerance)','Must PASS','Validation','Cold-chain integrity constraint — must never fail'],
      ['Mutex check (MX)','count(p_charge > 0.1 AND p_discharge > 0.1) == 0','Must PASS','Validation','Physical impossibility to charge and discharge simultaneously'],
    ]));

  const ord = refSection('📐','Step 5 — Ordering Validation Rules',
    'These four inequalities must always hold. A violation means there is a bug in the optimiser logic. ' +
    'The C~D gap tolerance is 10% — larger gaps suggest the MPC prediction horizon is too short.',
    refTable([{label:'Rule'},{label:'Meaning'},{label:'Tolerance',cls:'rng'},{label:'What a violation means'},{label:'Action if violated'}], [
      ['A >= B','Smart charging must cost less than dumb charging','EUR 0.001','Smart logic is actively making things worse','Check V1G constraint logic in run_smart_no_v2g()'],
      ['B >= C','V2G must improve on smart-only (or be equal)','EUR 0.001','V2G not worth it at current deg cost — acceptable warning, not a bug','Lower deg_cost_eur_kwh or check FCR premium values'],
      ['C ~ D','MPC should track MILP within 10%','10% gap tolerance','MPC horizon too short or update frequency too low','Increase MPC prediction horizon; check rolling window logic'],
      ['D <= E','Forecast noise should hurt (or equal) MPC performance','EUR 0.001','Noisy MPC outperforming perfect — likely a random seed artifact','Re-run; if persistent, check noise_std and seed in run_mpc_day_ahead()'],
    ]));

  return `<div class="ref-intro">
    <strong>Parameter Reference</strong> — this tab documents every parameter, formula, limit, and data source used in the model.
    It is always available without running anything and can be shown directly to supervisors or the Schmitz Cargobull team.
    <br><br>
    <strong>Formula Mode</strong> (toggle in sidebar) replaces all values in the Debug Results tab with the actual
    calculation that produced them — e.g. <em>E_min = soc_min/100 x usable = 20/100 x 84 = 16.8 kWh</em>.
    Toggle it on before presenting results to make every number fully traceable.
  </div>
  ${bat}${prc}${tru}${eb}${sc}${ord}`;
}


// build reference immediately so it is ready
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('tab-ref').innerHTML = buildReference();
});
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════
class DebugHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        if urlparse(self.path).path != "/run":
            self.send_response(404); self.end_headers(); return
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)
        try:    config = json.loads(body)
        except: config = {}
        result  = run_debug(config)
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    port   = _free_port()
    server = HTTPServer(("localhost", port), DebugHandler)
    url    = f"http://localhost:{port}"
    print("\n" + "="*58)
    print("  V2G Debug Tool - S.KOe COOL")
    print("  Schmitz Cargobull AG x TU Dortmund | 2026")
    print("="*58)
    if not IMPORT_OK:
        print("\n  WARNING: Could not import run_optimisation.py")
        print("  Make sure both files are in the same folder.")
    print(f"\n  Opening browser at: {url}")
    print("  Press Ctrl+C to stop\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()