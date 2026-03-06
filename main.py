#!/usr/bin/env python3
"""
main.py  —  FastAPI backend for S.KOe COOL V2G Optimisation
Deploy this on Render.com (free tier).

Endpoints:
  GET  /          → health check
  POST /optimize  → run all 5 scenarios, return JSON results
"""

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass

warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="S.KOe COOL V2G API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input model ───────────────────────────────────────────────────────────────
class Config(BaseModel):
    battery_capacity_kWh:  float = 82.0
    usable_capacity_kWh:   float = 65.6
    soc_min_pct:           float = 20.0
    soc_max_pct:           float = 95.0
    charge_power_kW:       float = 22.0
    discharge_power_kW:    float = 11.0
    eta_charge:            float = 0.92
    eta_discharge:         float = 0.92
    deg_cost_eur_kwh:      float = 0.07
    soc_init_pct:          float = 45.0
    soc_final_pct:         float = 80.0
    dwell_profile:         str   = "Extended"
    mpc_price_noise_std:   float = 0.012
    network_fee:           float = 0.0663
    concession:            float = 0.01992
    offshore_levy:         float = 0.00816
    chp_levy:              float = 0.00277
    electricity_tax:       float = 0.0205
    nev19:                 float = 0.01558
    vat:                   float = 0.19
    fcr_premium_peak:      float = 0.132
    fcr_window_start:      int   = 16
    fcr_window_end:        int   = 20
    afrr_premium:          float = 0.040
    afrr_window_start:     int   = 7
    afrr_window_end:       int   = 9
    fleet_size:            int   = 1
    season:                str   = "winter"


# ── Battery params dataclass ──────────────────────────────────────────────────
@dataclass
class V2GParams:
    battery_capacity_kWh: float
    usable_capacity_kWh:  float
    soc_min_pct:          float
    soc_max_pct:          float
    charge_power_kW:      float
    discharge_power_kW:   float
    eta_charge:           float
    eta_discharge:        float
    deg_cost_eur_kwh:     float
    mpc_price_noise_std:  float
    dt_h:                 float = 0.25
    n_slots:              int   = 96

    @property
    def E_min(self): return self.usable_capacity_kWh * self.soc_min_pct / 100
    @property
    def E_max(self): return self.usable_capacity_kWh * self.soc_max_pct / 100
    @property
    def p_c_max(self): return self.charge_power_kW
    @property
    def p_d_max(self): return self.discharge_power_kW


# ── Price builder ─────────────────────────────────────────────────────────────
def build_prices(cfg: Config):
    fixed_net = (cfg.network_fee + cfg.concession + cfg.offshore_levy +
                 cfg.chp_levy + cfg.electricity_tax + cfg.nev19)
    h = np.arange(96) * 0.25

    if cfg.season == "summer":
        spot = np.select(
            [(h>=0)&(h<5), (h>=5)&(h<7), (h>=7)&(h<9), (h>=9)&(h<11),
             (h>=11)&(h<15), (h>=15)&(h<17), (h>=17)&(h<20), (h>=20)&(h<22)],
            [0.038, 0.055, 0.095, 0.088, 0.018, 0.072, 0.121, 0.085],
            default=0.038)
    else:
        spot = np.select(
            [(h>=0)&(h<5), (h>=5)&(h<7), (h>=7)&(h<9), (h>=9)&(h<12),
             (h>=12)&(h<14), (h>=14)&(h<16), (h>=16)&(h<19), (h>=19)&(h<21)],
            [0.052, 0.071, 0.148, 0.131, 0.108, 0.092, 0.154, 0.118],
            default=0.052)

    buy   = (spot + fixed_net) * (1 + cfg.vat)
    fcr   = np.where((h >= cfg.fcr_window_start)  & (h < cfg.fcr_window_end),
                     cfg.fcr_premium_peak, 0.0)
    afrr  = np.where((h >= cfg.afrr_window_start) & (h < cfg.afrr_window_end),
                     cfg.afrr_premium, 0.0)
    v2g_p = buy + fcr + afrr
    return buy, v2g_p


# ── TRU load + plug-in availability ──────────────────────────────────────────
def build_tru_and_plugged(cfg: Config):
    h   = np.arange(96) * 0.25
    tru = 2.8 + 1.2 * np.sin(2 * np.pi * np.arange(96) / 96 + np.pi)
    if cfg.dwell_profile == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    else:
        plugged = ((h >= 21) | (h < 7) | ((h >= 12) & (h < 18))).astype(float)
    return tru, plugged


# ── MILP solver ───────────────────────────────────────────────────────────────
def _solve_milp(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg):
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix, csc_matrix
    except ImportError:
        return np.zeros(96), np.zeros(96), np.full(96, E_init), False

    W  = len(buy)
    dt = v2g.dt_h
    idx_c = np.arange(W)
    idx_d = np.arange(W, 2*W)
    idx_e = np.arange(2*W, 3*W)
    nv    = 3 * W

    c_vec        = np.zeros(nv)
    c_vec[idx_c] =  buy   * dt + deg * dt
    c_vec[idx_d] = -v2g_p * dt + deg * dt

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[idx_c] = v2g.p_c_max * plugged
    ub[idx_d] = v2g.p_d_max * plugged
    lb[idx_e] = v2g.E_min
    ub[idx_e] = v2g.E_max

    n_rows = W + W + 1
    A  = lil_matrix((n_rows, nv))
    lo = np.zeros(n_rows)
    hi = np.zeros(n_rows)

    for t in range(W):
        A[t, idx_e[t]]  =  1.0
        A[t, idx_c[t]]  = -v2g.eta_charge * dt
        A[t, idx_d[t]]  =  (1.0 / v2g.eta_discharge) * dt
        rhs = -tru[t] * dt
        if t == 0:
            rhs += E_init
        else:
            A[t, idx_e[t-1]] = -1.0
        lo[t] = hi[t] = rhs

    max_p = max(v2g.p_c_max, v2g.p_d_max)
    for t in range(W):
        A[W+t, idx_c[t]] = 1.0
        A[W+t, idx_d[t]] = 1.0
        lo[W+t] = -np.inf
        hi[W+t] =  max_p

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


# ── Greedy fallback ───────────────────────────────────────────────────────────
def _greedy(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg):
    N, dt = len(buy), v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N)
    soc = E_init
    for t in range(N):
        soc = max(v2g.E_min, soc - tru[t] * dt)
        if plugged[t]:
            spread = (v2g_p[t] - deg) - (buy[t] + deg)
            if spread > 0.02 and soc > v2g.E_min + v2g.p_d_max * dt / v2g.eta_discharge:
                p = min(v2g.p_d_max, (soc - v2g.E_min) * v2g.eta_discharge / dt)
                P_d[t] = p
                soc = max(v2g.E_min, soc - p / v2g.eta_discharge * dt)
            elif buy[t] < 0.22 and soc < v2g.E_max - v2g.p_c_max * dt * v2g.eta_charge:
                p = min(v2g.p_c_max, (v2g.E_max - soc) / (v2g.eta_charge * dt))
                P_c[t] = p
                soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return P_c, P_d, e


# ── KPI calculator ────────────────────────────────────────────────────────────
def _kpis(v2g, P_c, P_d, e, buy, v2g_p, deg, fleet):
    dt  = v2g.dt_h
    chg = float(np.sum(P_c * buy)    * dt)
    rev = float(np.sum(P_d * v2g_p)  * dt)
    dgc = float(np.sum((P_c + P_d) * deg) * dt)
    return {
        "net_cost":    round((chg - rev + dgc) * fleet, 4),
        "charge_cost": round(chg  * fleet, 4),
        "v2g_revenue": round(rev  * fleet, 4),
        "deg_cost":    round(dgc  * fleet, 4),
        "v2g_kwh":     round(float(np.sum(P_d) * dt) * fleet, 3),
        "soc_profile": [round(float(x), 3) for x in e],
        "p_charge":    [round(float(x), 3) for x in P_c],
        "p_discharge": [round(float(x), 3) for x in P_d],
    }


# ── Scenarios ─────────────────────────────────────────────────────────────────
def run_a(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    N, dt = 96, v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N); soc = E_init
    for t in range(N):
        soc = max(v2g.E_min, soc - tru[t] * dt)
        if plugged[t] and soc < v2g.E_max:
            p = min(v2g.p_c_max, (v2g.E_max - soc) / (v2g.eta_charge * dt))
            P_c[t] = p
            soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return P_c, P_d, e


def run_b(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    P_c, P_d, e, ok = _solve_milp(v2g, buy, np.zeros_like(v2g_p),
                                   tru, plugged, E_init, E_fin, 0.0)
    if not ok:
        P_c, P_d, e = _greedy(v2g, buy, np.zeros_like(v2g_p),
                               tru, plugged, E_init, E_fin, 0.0)
    return P_c, P_d, e


def run_c(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    deg = v2g.deg_cost_eur_kwh
    P_c, P_d, e, ok = _solve_milp(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg)
    if not ok:
        P_c, P_d, e = _greedy(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg)
    return P_c, P_d, e


def run_d(v2g, buy_da, v2g_da, tru, plugged, E_init, E_fin,
          noise_std=0.0, seed=42):
    deg = v2g.deg_cost_eur_kwh
    N, dt = 96, v2g.dt_h
    rng = np.random.default_rng(seed)
    P_c_all = np.zeros(N); P_d_all = np.zeros(N)
    e_all   = np.zeros(N); soc = E_init

    for t in range(N):
        W  = N - t
        bf = buy_da[t:].copy()
        vf = v2g_da[t:].copy()
        if noise_std > 0:
            bf = np.maximum(0.01, bf + rng.normal(0, noise_std, W))
            vf = np.maximum(0.01, vf + rng.normal(0, noise_std, W))

        P_c_w, P_d_w, _, ok = _solve_milp(v2g, bf, vf, tru[t:],
                                           plugged[t:], soc, E_fin, deg)
        if not ok:
            P_c_w, P_d_w, _ = _greedy(v2g, bf, vf, tru[t:],
                                       plugged[t:], soc, E_fin, deg)

        pc_t = float(np.clip(P_c_w[0], 0, v2g.p_c_max * plugged[t]))
        pd_t = float(np.clip(P_d_w[0], 0, v2g.p_d_max * plugged[t]))

        if pc_t > 1e-6 and pd_t > 1e-6:
            if (v2g_da[t] - deg) > (buy_da[t] + deg):
                pc_t = 0.0
            else:
                pd_t = 0.0

        soc = float(np.clip(
            soc - tru[t] * dt + pc_t * v2g.eta_charge * dt
                             - pd_t / v2g.eta_discharge * dt,
            v2g.E_min, v2g.E_max))

        P_c_all[t] = pc_t
        P_d_all[t] = pd_t
        e_all[t]   = soc

    return P_c_all, P_d_all, e_all


def deg_sweep(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    rows = []
    for dv in np.linspace(0.02, 0.15, 14):
        dv = float(dv)
        P_c, P_d, e, ok = _solve_milp(v2g, buy, v2g_p, tru, plugged,
                                       E_init, E_fin, dv)
        if not ok:
            P_c, P_d, e = _greedy(v2g, buy, v2g_p, tru, plugged,
                                   E_init, E_fin, dv)
        dt  = v2g.dt_h
        rev = float(np.sum(P_d * v2g_p) * dt)
        kwh = float(np.sum(P_d) * dt)
        chg = float(np.sum(P_c * buy) * dt)
        dgc = float(np.sum((P_c + P_d) * dv) * dt)
        rows.append({
            "deg":      round(dv, 4),
            "net_cost": round(chg - rev + dgc, 4),
            "v2g_rev":  round(rev, 4),
            "v2g_kwh":  round(kwh, 3),
            "active":   kwh > 0.1,
        })
    return rows


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "S.KOe COOL V2G API"}


@app.post("/optimize")
def optimize(cfg: Config):
    v2g = V2GParams(
        battery_capacity_kWh = cfg.battery_capacity_kWh,
        usable_capacity_kWh  = cfg.usable_capacity_kWh,
        soc_min_pct          = cfg.soc_min_pct,
        soc_max_pct          = cfg.soc_max_pct,
        charge_power_kW      = cfg.charge_power_kW,
        discharge_power_kW   = cfg.discharge_power_kW,
        eta_charge           = cfg.eta_charge,
        eta_discharge        = cfg.eta_discharge,
        deg_cost_eur_kwh     = cfg.deg_cost_eur_kwh,
        mpc_price_noise_std  = cfg.mpc_price_noise_std,
    )

    buy, v2g_p   = build_prices(cfg)
    tru, plugged = build_tru_and_plugged(cfg)
    h            = np.arange(96) * 0.25
    fl           = cfg.fleet_size
    deg          = cfg.deg_cost_eur_kwh

    E_init = v2g.usable_capacity_kWh * cfg.soc_init_pct  / 100
    E_fin  = v2g.usable_capacity_kWh * cfg.soc_final_pct / 100

    # Run all 5 scenarios
    Ac, Ad, Ae = run_a(v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    Bc, Bd, Be = run_b(v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    Cc, Cd, Ce = run_c(v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    Dc, Dd, De = run_d(v2g, buy, v2g_p, tru, plugged, E_init, E_fin,
                       noise_std=0.0, seed=42)
    Ec, Ed, Ee = run_d(v2g, buy, v2g_p, tru, plugged, E_init, E_fin,
                       noise_std=cfg.mpc_price_noise_std, seed=42)

    scenarios = {
        "A": _kpis(v2g, Ac, Ad, Ae, buy, v2g_p, 0.0, fl),
        "B": _kpis(v2g, Bc, Bd, Be, buy, v2g_p, 0.0, fl),
        "C": _kpis(v2g, Cc, Cd, Ce, buy, v2g_p, deg, fl),
        "D": _kpis(v2g, Dc, Dd, De, buy, v2g_p, deg, fl),
        "E": _kpis(v2g, Ec, Ed, Ee, buy, v2g_p, deg, fl),
    }

    ref = scenarios["A"]["net_cost"]
    for s in scenarios.values():
        s["saving_vs_dumb"] = round(ref - s["net_cost"], 4)
        s["annual_saving"]  = round((ref - s["net_cost"]) * 365, 1)

    deg_data     = deg_sweep(v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    tipping      = max((d["deg"] for d in deg_data if d["active"]), default=0.0)

    fixed_net = (cfg.network_fee + cfg.concession + cfg.offshore_levy +
                 cfg.chp_levy + cfg.electricity_tax + cfg.nev19)

    return {
        "scenarios":     scenarios,
        "deg_sweep":     deg_data,
        "tipping_point": tipping,
        "hours":         [round(float(x), 2) for x in h],
        "buy_price":     [round(float(x), 4) for x in buy],
        "v2g_price":     [round(float(x), 4) for x in v2g_p],
        "tru_load":      [round(float(x), 3) for x in tru],
        "plugged":       [int(x) for x in plugged],
        "derived": {
            "E_min":         round(v2g.E_min, 3),
            "E_max":         round(v2g.E_max, 3),
            "E_init":        round(E_init, 3),
            "E_fin":         round(E_fin, 3),
            "rte_pct":       round(cfg.eta_charge * cfg.eta_discharge * 100, 2),
            "fixed_net":     round(fixed_net, 5),
            "plugged_hours": int(plugged.sum() * 0.25),
        },
        "fleet_size": fl,
        "season":     cfg.season,
    }


# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)