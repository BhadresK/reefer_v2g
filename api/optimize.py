from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from dataclasses import dataclass, asdict


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
    dt_h:    float = 0.25
    n_slots: int   = 96

    @property
    def E_min(self): return self.usable_capacity_kWh * self.soc_min_pct / 100
    @property
    def E_max(self): return self.usable_capacity_kWh * self.soc_max_pct / 100


def build_prices(cfg):
    fixed_net = (cfg["network_fee"] + cfg["concession"] + cfg["offshore_levy"] +
                 cfg["chp_levy"] + cfg["electricity_tax"] + cfg["nev19"])
    h = np.arange(96) * 0.25
    if cfg["season"] == "summer":
        spot = np.select(
            [(h>=0)&(h<5),(h>=5)&(h<7),(h>=7)&(h<9),(h>=9)&(h<11),
             (h>=11)&(h<15),(h>=15)&(h<17),(h>=17)&(h<20),(h>=20)&(h<22)],
            [0.038,0.055,0.095,0.088,0.018,0.072,0.121,0.085], default=0.038)
    else:
        spot = np.select(
            [(h>=0)&(h<5),(h>=5)&(h<7),(h>=7)&(h<9),(h>=9)&(h<12),
             (h>=12)&(h<14),(h>=14)&(h<16),(h>=16)&(h<19),(h>=19)&(h<21)],
            [0.052,0.071,0.148,0.131,0.108,0.092,0.154,0.118], default=0.052)
    buy   = (spot + fixed_net) * (1 + cfg["vat"])
    fcr   = np.where((h >= cfg["fcr_window_start"]) & (h < cfg["fcr_window_end"]),
                     cfg["fcr_premium_peak"], 0.0)
    afrr  = np.where((h >= cfg["afrr_window_start"]) & (h < cfg["afrr_window_end"]),
                     cfg["afrr_premium"], 0.0)
    return buy, buy + fcr + afrr


def build_tru_and_plugged(cfg):
    h   = np.arange(96) * 0.25
    tru = 2.8 + 1.2 * np.sin(2 * np.pi * np.arange(96) / 96 + np.pi)
    if cfg["dwell_profile"] == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    else:
        plugged = ((h >= 21) | (h < 7) | ((h >= 12) & (h < 18))).astype(float)
    return tru, plugged


def _optimize(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg):
    N, dt = 96, v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N)
    base_soc = np.zeros(N + 1); base_soc[0] = E_init
    for t in range(N):
        base_soc[t+1] = max(v2g.E_min, base_soc[t] - tru[t] * dt)

    charge_cost   = buy + deg
    discharge_val = v2g_p - deg
    d_order = np.argsort(-discharge_val)
    c_order = np.argsort(charge_cost)
    soc_w   = base_soc.copy()

    for t in d_order:
        if not plugged[t]: continue
        if discharge_val[t] <= charge_cost[t]: continue
        if soc_w[t] <= v2g.E_min + 1e-4: continue
        max_kwh = min((soc_w[t] - v2g.E_min) * v2g.eta_discharge,
                      v2g.discharge_power_kW * dt)
        if max_kwh < 1e-4: continue
        P_d[t] = max_kwh / dt
        soc_w[t+1:] -= max_kwh / v2g.eta_discharge

    for t in c_order:
        if not plugged[t]: continue
        if soc_w[t] >= v2g.E_max - 1e-4: continue
        max_kwh = min((v2g.E_max - soc_w[t]) / v2g.eta_charge,
                      v2g.charge_power_kW * dt)
        if max_kwh < 1e-4: continue
        P_c[t] = max_kwh / dt
        soc_w[t+1:] += max_kwh * v2g.eta_charge

    shortfall = E_fin - soc_w[N]
    if shortfall > 1e-3:
        for t in c_order:
            if not plugged[t]: continue
            headroom = min((v2g.E_max - soc_w[t]) / v2g.eta_charge,
                           v2g.charge_power_kW * dt)
            extra = min(headroom - P_c[t] * dt, shortfall / v2g.eta_charge)
            if extra > 1e-4:
                P_c[t] += extra / dt
                soc_w[t+1:] += extra * v2g.eta_charge
                shortfall -= extra * v2g.eta_charge
            if shortfall <= 1e-3: break

    s = E_init
    for t in range(N):
        s = float(np.clip(
            s - tru[t]*dt + P_c[t]*v2g.eta_charge*dt - P_d[t]/v2g.eta_discharge*dt,
            v2g.E_min, v2g.E_max))
        soc_w[t+1] = s
    return P_c, P_d, soc_w[1:]


def _kpis(v2g, P_c, P_d, e, buy, v2g_p, deg, fleet):
    dt  = v2g.dt_h
    chg = float(np.sum(P_c * buy)        * dt)
    rev = float(np.sum(P_d * v2g_p)      * dt)
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


def run_a(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    N, dt = 96, v2g.dt_h
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N); soc = E_init
    for t in range(N):
        soc = max(v2g.E_min, soc - tru[t] * dt)
        if plugged[t] and soc < v2g.E_max:
            p = min(v2g.charge_power_kW, (v2g.E_max - soc) / (v2g.eta_charge * dt))
            P_c[t] = p; soc = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return P_c, P_d, e


def run_mpc(v2g, buy, v2g_p, tru, plugged, E_init, E_fin, noise_std=0.0, seed=42):
    N, dt = 96, v2g.dt_h
    rng   = np.random.default_rng(seed)
    P_c_all = np.zeros(N); P_d_all = np.zeros(N)
    e_all   = np.zeros(N); soc = E_init
    for t in range(N):
        W  = N - t
        bf = buy[t:].copy(); vf = v2g_p[t:].copy()
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
            soc - tru[t]*dt + pc_t*v2g.eta_charge*dt - pd_t/v2g.eta_discharge*dt,
            v2g.E_min, v2g.E_max))
        P_c_all[t] = pc_t; P_d_all[t] = pd_t; e_all[t] = soc
    return P_c_all, P_d_all, e_all


def deg_sweep(v2g, buy, v2g_p, tru, plugged, E_init, E_fin):
    rows = []
    for dv in np.linspace(0.02, 0.15, 14):
        dv = float(dv)
        v2g_tmp = V2GParams(
            battery_capacity_kWh=v2g.battery_capacity_kWh,
            usable_capacity_kWh=v2g.usable_capacity_kWh,
            soc_min_pct=v2g.soc_min_pct, soc_max_pct=v2g.soc_max_pct,
            charge_power_kW=v2g.charge_power_kW,
            discharge_power_kW=v2g.discharge_power_kW,
            eta_charge=v2g.eta_charge, eta_discharge=v2g.eta_discharge,
            deg_cost_eur_kwh=dv, mpc_price_noise_std=v2g.mpc_price_noise_std)
        P_c, P_d, _ = _optimize(v2g_tmp, buy, v2g_p, tru, plugged, E_init, E_fin, dv)
        dt  = v2g.dt_h
        rev = float(np.sum(P_d * v2g_p) * dt)
        kwh = float(np.sum(P_d) * dt)
        chg = float(np.sum(P_c * buy) * dt)
        dgc = float(np.sum((P_c + P_d) * dv) * dt)
        rows.append({"deg": round(dv,4), "net_cost": round(chg-rev+dgc,4),
                     "v2g_rev": round(rev,4), "v2g_kwh": round(kwh,3),
                     "active": kwh > 0.1})
    return rows


def run_optimization(cfg):
    v2g = V2GParams(
        battery_capacity_kWh=cfg["battery_capacity_kWh"],
        usable_capacity_kWh=cfg["usable_capacity_kWh"],
        soc_min_pct=cfg["soc_min_pct"], soc_max_pct=cfg["soc_max_pct"],
        charge_power_kW=cfg["charge_power_kW"],
        discharge_power_kW=cfg["discharge_power_kW"],
        eta_charge=cfg["eta_charge"], eta_discharge=cfg["eta_discharge"],
        deg_cost_eur_kwh=cfg["deg_cost_eur_kwh"],
        mpc_price_noise_std=cfg["mpc_price_noise_std"])

    buy, v2g_p   = build_prices(cfg)
    tru, plugged = build_tru_and_plugged(cfg)
    h            = np.arange(96) * 0.25
    fl           = int(cfg["fleet_size"])
    deg          = cfg["deg_cost_eur_kwh"]
    E_init = v2g.usable_capacity_kWh * cfg["soc_init_pct"]  / 100
    E_fin  = v2g.usable_capacity_kWh * cfg["soc_final_pct"] / 100

    Ac,Ad,Ae = run_a  (v2g,buy,v2g_p,tru,plugged,E_init,E_fin)
    Bc,Bd,Be = _optimize(v2g,buy,np.zeros_like(v2g_p),tru,plugged,E_init,E_fin,0.0)
    Cc,Cd,Ce = _optimize(v2g,buy,v2g_p,tru,plugged,E_init,E_fin,deg)
    Dc,Dd,De = run_mpc(v2g,buy,v2g_p,tru,plugged,E_init,E_fin,noise_std=0.0,seed=42)
    Ec,Ed,Ee = run_mpc(v2g,buy,v2g_p,tru,plugged,E_init,E_fin,
                       noise_std=cfg["mpc_price_noise_std"],seed=42)

    scenarios = {
        "A": _kpis(v2g,Ac,Ad,Ae,buy,v2g_p,0.0,fl),
        "B": _kpis(v2g,Bc,Bd,Be,buy,v2g_p,0.0,fl),
        "C": _kpis(v2g,Cc,Cd,Ce,buy,v2g_p,deg,fl),
        "D": _kpis(v2g,Dc,Dd,De,buy,v2g_p,deg,fl),
        "E": _kpis(v2g,Ec,Ed,Ee,buy,v2g_p,deg,fl),
    }
    ref = scenarios["A"]["net_cost"]
    for s in scenarios.values():
        s["saving_vs_dumb"] = round(ref - s["net_cost"], 4)
        s["annual_saving"]  = round((ref - s["net_cost"]) * 365, 1)

    deg_data  = deg_sweep(v2g, buy, v2g_p, tru, plugged, E_init, E_fin)
    tipping   = max((d["deg"] for d in deg_data if d["active"]), default=0.0)
    fixed_net = (cfg["network_fee"]+cfg["concession"]+cfg["offshore_levy"]+
                 cfg["chp_levy"]+cfg["electricity_tax"]+cfg["nev19"])
    return {
        "scenarios": scenarios, "deg_sweep": deg_data, "tipping_point": tipping,
        "hours":     [round(float(x),2) for x in h],
        "buy_price": [round(float(x),4) for x in buy],
        "v2g_price": [round(float(x),4) for x in v2g_p],
        "tru_load":  [round(float(x),3) for x in tru],
        "plugged":   [int(x) for x in plugged],
        "derived": {
            "E_min": round(v2g.E_min,3), "E_max": round(v2g.E_max,3),
            "E_init": round(E_init,3),   "E_fin": round(E_fin,3),
            "rte_pct": round(cfg["eta_charge"]*cfg["eta_discharge"]*100,2),
            "fixed_net": round(fixed_net,5),
            "plugged_hours": int(plugged.sum()*0.25),
        },
        "fleet_size": fl, "season": cfg["season"],
    }


# ── Vercel serverless handler ─────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        body = json.dumps({"status": "ok", "service": "S.KOe COOL V2G API"}).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        cfg    = json.loads(self.rfile.read(length))
        result = run_optimization(cfg)
        body   = json.dumps(result).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")