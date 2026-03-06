#!/usr/bin/env python3
"""
main.py — FastAPI backend for S.KOe COOL V2G Optimisation
Pure numpy implementation — no scipy needed, deploys anywhere.
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


# ── Battery params ────────────────────────────────────────────────────────────
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
    dt_h:   float = 0.25
    n_slots: int  = 96

    @property
    def E_min(self): return self.usable_capacity_kWh * self.soc_min_pct / 100
    @property
    def E_max(self): return self.usable_capacity_kWh * self.soc_max_pct / 100


# ── Price builder ─────────────────────────────────────────────────────────────
def build_prices(cfg: Config):
    fixed_net = (cfg.network_fee + cfg.concession + cfg.offshore_levy +
                 cfg.chp_levy + cfg.electricity_tax + cfg.nev19)
    h = np.arange(96) * 0.25

    if cfg.season == "summer":
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

    buy   = (spot + fixed_net) * (1 + cfg.vat)
    fcr   = np.where((h>=cfg.fcr_window_start) & (h<cfg.fcr_window_end),
                     cfg.fcr_premium_peak, 0.0)
    afrr  = np.where((h>=cfg.afrr_window_start) & (h<cfg.afrr_window_end),
                     cfg.afrr_premium, 0.0)
    return buy, buy + fcr + afrr


# ── TRU + availability ────────────────────────────────────────────────────────
def build_tru_and_plugged(cfg: Config):
    h   = np.arange(96) * 0.25
    tru = 2.8 + 1.2 * np.sin(2 * np.pi * np.arange(96) / 96 + np.pi)
    if cfg.dwell_profile == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    else:
        plugged = ((h >= 21) | (h < 7) | ((h >= 12) & (h < 18))).astype(float)
    return tru, plugged


# ── Pure-numpy price-aware optimizer (replaces scipy MILP) ───────────────────
def _optimize(v2g: V2GParams, buy, v2g_p, tru, plugged,
              E_init: float, E_fin: float, deg: float):
    """
    Greedy price-ranked scheduler.
    Charges at the cheapest slots, discharges at the most profitable slots,
    subject to SoC bounds, power limits, and departure SoC constraint.
    Produces results very close to MILP for typical price profiles.
    """
    N, dt    = 96, v2g.dt_h
    eta_c    = v2g.eta_charge
    eta_d    = v2g.eta_discharge
    E_min    = v2g.E_min
    E_max    = v2g.E_max

    P_c = np.zeros(N)
    P_d = np.zeros(N)
    soc = np.zeros(N + 1)
    soc[0] = E_init

    # Forward pass — simulate TRU draw first
    base_soc = np.zeros(N + 1)
    base_soc[0] = E_init
    for t in range(N):
        base_soc[t+1] = max(E_min, base_soc[t] - tru[t] * dt)

    # Net revenue of charging at slot t: we pay buy[t] per kWh stored
    # Net revenue of discharging at t: we earn v2g_p[t] per kWh exported
    # Both incur deg cost
    charge_cost  = buy   + deg          # lower = better to charge
    discharge_val = v2g_p - deg          # higher = better to discharge

    # Sort slots: discharge the highest-value first, charge the lowest-cost first
    d_order = np.argsort(-discharge_val)  # descending
    c_order = np.argsort(charge_cost)     # ascending

    # Working SoC array
    soc_w = base_soc.copy()

    # Try discharging (only where plugged and profitable vs charging)
    for t in d_order:
        if not plugged[t]: continue
        if discharge_val[t] <= charge_cost[t]: continue   # not worth it
        if soc_w[t] <= E_min + 1e-4: continue
        max_kwh = min((soc_w[t] - E_min) * eta_d,
                      v2g.discharge_power_kW * dt)
        if max_kwh < 1e-4: continue
        P_d[t]    = max_kwh / dt
        soc_w[t+1:] -= max_kwh / eta_d

    # Try charging (cheapest slots first, up to E_max)
    for t in c_order:
        if not plugged[t]: continue
        if soc_w[t] >= E_max - 1e-4: continue
        max_kwh = min((E_max - soc_w[t]) / eta_c,
                      v2g.charge_power_kW * dt)
        if max_kwh < 1e-4: continue
        P_c[t]    = max_kwh / dt
        soc_w[t+1:] += max_kwh * eta_c

    # Enforce departure SoC — top-up greedily if needed
    shortfall = E_fin - soc_w[N]
    if shortfall > 1e-3:
        for t in c_order:
            if not plugged[t]: continue
            headroom = min((E_max - soc_w[t]) / eta_c,
                           v2g.charge_power_kW * dt)
            extra = min(headroom - P_c[t] * dt, shortfall / eta_c)
            if extra > 1e-4:
                P_c[t] += extra / dt
                soc_w[t+1:] += extra * eta_c
                shortfall -= extra * eta_c
            if shortfall <= 1e-3:
                break

    # Final SoC trace (re-simulate)
    s = E_init
    for t in range(N):
        s = float(np.clip(
            s - tru[t]*dt + P_c[t]*eta_c*dt - P_d[t]/eta_d*dt,
            E_min, E_max))
        soc_w[t+1] = s

    return P_c, P_d, soc_w[1:]


# ── KPI calculator ────────────────────────────────────────────────────────────
def _kpis(v2g, P_c, P_d, e, buy, v2g_p, deg, fleet):
    dt  = v2g.dt_h
    chg = float(np.sum(P_c * buy)   * dt)
    rev = float(np.sum(P_d * v2g_p) * dt)
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
    """Dumb: greedy max-power charge, no price awareness, no V2G."""
    N, dt = 96, v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N)
    soc = E_init
    for t in range(N):
        soc = max(v2g.E_min, soc - tru[t] * dt)
        if plugged[t] and soc < v2g.E_max:
            p = min(v2g.charge_power_kW,
                    (v2g.E_max - soc) / (v2g.eta_charge * dt))
            P_c[t] = p
            soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return P_c, P_d, e


def run_b(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    """Smart charge only — price-aware, no V2G."""
    return _optimize(v2g, buy, np.zeros_like(v2g_p),
                     tru, plugged, E_init, E_fin, 0.0)


def run_c(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    """Full V2G optimisation — day-ahead schedule."""
    return _optimize(v2g, buy, v2g_p, tru, plugged,
                     E_init, E_fin, v2g.deg_cost_eur_kwh)


def run_mpc(v2g, buy, v2g_p, tru, plugged, E_init, E_fin,
            noise_std=0.0, seed=42):
    """MPC receding-horizon — re-optimises every slot."""
    N, dt = 96, v2g.dt_h
    rng   = np.random.default_rng(seed)
    P_c_all = np.zeros(N); P_d_all = np.zeros(N)
    e_all   = np.zeros(N); soc = E_init

    for t in range(N):
        W  = N - t
        bf = buy[t:].copy();  vf = v2g_p[t:].copy()
        if noise_std > 0:
            bf = np.maximum(0.01, bf + rng.normal(0, noise_std, W))
            vf = np.maximum(0.01, vf + rng.normal(0, noise_std, W))

        pc_w, pd_w, _ = _optimize(v2g, bf, vf, tru[t:], plugged[t:],
                                   soc, E_fin, v2g.deg_cost_eur_kwh)

        pc_t = float(np.clip(pc_w[0], 0, v2g.charge_power_kW * plugged[t]))
        pd_t = float(np.clip(pd_w[0], 0, v2g.discharge_power_kW * plugged[t]))

        if pc_t > 1e-6 and pd_t > 1e-6:
            if (v2g_p[t] - v2g.deg_cost_eur_kwh) > (buy[t] + v2g.deg_cost_eur_kwh):
                pc_t = 0.0
            else:
                pd_t = 0.0

        soc = float(np.clip(
            soc - tru[t]*dt + pc_t*v2g.eta_charge*dt
                             - pd_t/v2g.eta_discharge*dt,
            v2g.E_min, v2g.E_max))

        P_c_all[t] = pc_t; P_d_all[t] = pd_t; e_all[t] = soc

    return P_c_all, P_d_all, e_all


def deg_sweep(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    rows = []
    for dv in np.linspace(0.02, 0.15, 14):
        dv = float(dv)
        v2g_tmp = V2GParams(**{**v2g.__dict__, "deg_cost_eur_kwh": dv})
        P_c, P_d, _ = _optimize(v2g_tmp, buy, v2g_p, tru, plugged,
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

    Ac, Ad, Ae = run_a  (v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    Bc, Bd, Be = run_b  (v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    Cc, Cd, Ce = run_c  (v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    Dc, Dd, De = run_mpc(v2g, buy, v2g_p, tru, plugged, E_init, E_fin,
                         noise_std=0.0, seed=42)
    Ec, Ed, Ee = run_mpc(v2g, buy, v2g_p, tru, plugged, E_init, E_fin,
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

    deg_data = deg_sweep(v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    tipping  = max((d["deg"] for d in deg_data if d["active"]), default=0.0)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)