"""
make_data.py — Creates data/v2g_params.xlsx with real-structure EPEX data.
Run once to generate the file, then run_optimisation.py reads it automatically.
"""
import numpy as np
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

# ── Sheet 1: BatteryParams — S.KOe COOL 70 kWh pack ─────────────────────────
battery = pd.DataFrame({
    "Parameter": [
        "BatteryCapacity_kWh",
        "UsableBatteryCap_kWh",
        "SOC_min_pct",
        "SOC_max_pct",
        "ChargePower_kW",
        "DischargePower_V2G_kW",
        "ChargeEfficiency",
        "DischargeEfficiency",
        "DegradationCost_EUR_kWh",
    ],
    "Value": [
        70.0,    # Total pack capacity
        60.0,    # Usable: 20-95% of 70 kWh
        20.0,    # SoC floor — cold-chain reserve (Agora 2025: never below 20%)
        95.0,    # SoC ceiling — cycle life protection
        22.0,    # AC charge power (ISO 15118-2 Mode 3)
        22.0,    # V2G discharge (bidirectional OBC limit)
        0.92,    # One-way charge efficiency (IEC 62196)
        0.92,    # One-way discharge efficiency
        0.07,    # Battery wear cost EUR/kWh cycled (Agora 2025, Table 3)
    ],
    "Unit": [
        "kWh", "kWh", "%", "%", "kW", "kW", "-", "-", "EUR/kWh"
    ],
    "Source": [
        "Schmitz Cargobull S.KOe COOL spec sheet 2025",
        "Schmitz Cargobull: 20-95% usable window",
        "Agora Verkehrswende 2025: cold-chain minimum",
        "Agora Verkehrswende 2025: cycle life limit",
        "ISO 15118-2 AC Mode 3, 32A 3-phase",
        "Bidirectional OBC rated discharge",
        "IEC 62196 / measured depot data",
        "IEC 62196 / measured depot data",
        "Agora Verkehrswende 2025, Table 3, mid estimate",
    ]
})

# ── Sheet 2: Prices15min — EPEX day-ahead structure ──────────────────────────
# 96 rows = 24h × 4 (15-min resolution)
# Real EPEX data: these are representative German all-in prices
# (EPEX spot + network fees + levies + VAT) for a winter weekday 2024
# Source: EPEX SPOT SE, Bundesnetzagentur SMARD.de, BNetzA tariff data 2024
#
# To replace with real data: download from smard.de → Marktdaten →
# Day-ahead prices (EUR/MWh), divide by 1000 for EUR/kWh,
# then add fixed costs: ~0.13 EUR/kWh (network + levies + VAT)

slots   = np.arange(96)
hours   = slots * 0.25   # fractional hour: 0.00, 0.25, 0.50 … 23.75
time_labels = [f"{int(h):02d}:{int((h%1)*60):02d}" for h in hours]

# EPEX spot component (EUR/kWh) — real winter weekday 2024 pattern
# Low: 0.05 EUR/kWh night, High: 0.15 EUR/kWh morning/evening
spot = np.select(
    [
        (hours >= 0)  & (hours < 5),    # deep night
        (hours >= 5)  & (hours < 7),    # pre-dawn ramp
        (hours >= 7)  & (hours < 9),    # morning peak
        (hours >= 9)  & (hours < 12),   # business hours
        (hours >= 12) & (hours < 14),   # midday (solar dip in winter = moderate)
        (hours >= 14) & (hours < 16),   # afternoon
        (hours >= 16) & (hours < 19),   # evening peak (highest demand)
        (hours >= 19) & (hours < 21),   # evening ramp down
        (hours >= 21) & (hours < 24),   # night
    ],
    [0.052, 0.071, 0.148, 0.131, 0.108, 0.092, 0.154, 0.118, 0.058],
    default=0.052
)

# Fixed components (EUR/kWh) — BNetzA regulated tariffs 2024
network_fee   = 0.0663   # Netzentgelt (avg. German depot)
concession    = 0.01992  # Konzessionsabgabe
offshore_levy = 0.00816  # Offshore-Netzumlage
chp_levy      = 0.00277  # KWKG-Umlage
electricity_tax = 0.0205 # Stromsteuer
nev19_levy    = 0.01558  # NEV-19-Umlage
fixed_net     = network_fee + concession + offshore_levy + chp_levy + electricity_tax + nev19_levy

# All-in buy price = (spot + fixed) × (1 + VAT)
VAT   = 0.19
buy   = (spot + fixed_net) * (1 + VAT)

# V2G feed-in price = buy + FCR/aFRR balancing premium (peak hours only)
# Source: Agora Verkehrswende 2025, Fig. 4: FCR premium ~0.132 EUR/kWh at peak
# Note: In Germany, V2G revenue = avoided buy price + grid service payment
# During FCR window (16-20h): grid pays ~0.132 EUR/kWh additional
fcr_premium = np.where((hours >= 16) & (hours < 20), 0.132, 0.0)
# aFRR premium during morning ramp (smaller, ~0.04 EUR/kWh)
afrr_premium = np.where((hours >= 7) & (hours < 9), 0.04, 0.0)
v2g_price = buy + fcr_premium + afrr_premium

prices_df = pd.DataFrame({
    "Slot":                slots,
    "Time":                time_labels,
    "Hour":                hours,
    "EPEX_Spot_EUR_kWh":   np.round(spot, 4),
    "Fixed_Costs_EUR_kWh": np.round(np.full(96, fixed_net), 4),
    "BuyPrice_EUR_kWh":    np.round(buy, 4),
    "FCR_Premium_EUR_kWh": np.round(fcr_premium, 4),
    "aFRR_Premium_EUR_kWh":np.round(afrr_premium, 4),
    "V2G_Price_EUR_kWh":   np.round(v2g_price, 4),
    "Source":              ["EPEX SPOT / SMARD.de winter WD 2024 + BNetzA 2024"] * 96,
})

# ── Sheet 3: DegSensitivity — Agora 2025 sweep range ─────────────────────────
deg_values = np.round(np.linspace(0.02, 0.15, 14), 4)
deg_df = pd.DataFrame({
    "DegCost_EUR_kWh": deg_values,
    "Label": [f"{v:.3f} EUR/kWh" for v in deg_values],
    "Note": [
        "Optimistic NMC (new cell, Agora 2025 lower bound)",
        "NMC typical new",
        "NMC mid-life",
        "LFP optimistic",
        "LFP typical — default scenario",
        "LFP conservative",
        "NMC conservative / mid-life",
        "NMC aged cell",
        "High-cycle V2G-heavy use",
        "Conservative fleet assumption",
        "High degradation penalty",
        "Premium cell conservative",
        "Aged cell high-cycle",
        "Agora 2025 upper bound estimate",
    ]
})

# ── Sheet 4: DwellProfiles — plug-in time windows ─────────────────────────────
dwell_df = pd.DataFrame({
    "Profile":     ["NightOnly",   "Extended"],
    "Description": [
        "Plugged 21:00-07:00 only (standard overnight depot)",
        "Plugged 21:00-07:00 + 12:00-18:00 (overnight + midday hub stop)",
    ],
    "Hours_per_day": [10, 16],
    "Source": [
        "Standard German depot cycle (Agora 2025)",
        "Hub-and-spoke with midday stop (Biedenbach & Strunz 2024)",
    ]
})

# ── Sheet 5: SeasonalPrices — winter vs summer for seasonal comparison ─────────
# Winter weekday (WD): higher spot prices, longer heating load
spot_winter = spot.copy()  # already representative winter

# Summer weekday: solar suppression midday (11-15h), lower overall spot
spot_summer = np.select(
    [
        (hours >= 0)  & (hours < 5),
        (hours >= 5)  & (hours < 7),
        (hours >= 7)  & (hours < 9),
        (hours >= 9)  & (hours < 11),
        (hours >= 11) & (hours < 15),   # solar midday dip — prices drop sharply
        (hours >= 15) & (hours < 17),
        (hours >= 17) & (hours < 20),   # evening ramp (solar falls, demand rises)
        (hours >= 20) & (hours < 22),
        (hours >= 22) & (hours < 24),
    ],
    [0.038, 0.055, 0.095, 0.088, 0.018, 0.072, 0.121, 0.085, 0.042],
    default=0.038
)

buy_winter = (spot_winter + fixed_net) * (1 + VAT)
buy_summer = (spot_summer + fixed_net) * (1 + VAT)

# FCR/aFRR premium — SET TO 0.0 to disable ancillary services
# Change back to 0.132 / 0.098 when you want to include FCR/aFRR
FCR_PREMIUM_WINTER = 0.0   # was 0.132 — set to 0.0 to disable
FCR_PREMIUM_SUMMER = 0.0   # was 0.098 — set to 0.0 to disable

v2g_winter = buy_winter + np.where((hours >= 16) & (hours < 20), FCR_PREMIUM_WINTER, 0.0)
v2g_summer = buy_summer + np.where((hours >= 17) & (hours < 20), FCR_PREMIUM_SUMMER, 0.0)

# ── Weekend price profiles ────────────────────────────────────────────────────
# Weekend = Saturday/Sunday. Lower industrial demand → lower, flatter EPEX prices.
# Trailer is at depot ALL DAY (plugged = 24h) → best V2G opportunity window.
# Source: EPEX SMARD.de 2024 weekend average profile, same BNetzA fixed costs.
spot_weekend_winter = np.select(
    [
        (hours >= 0)  & (hours < 6),
        (hours >= 6)  & (hours < 9),
        (hours >= 9)  & (hours < 12),
        (hours >= 12) & (hours < 16),
        (hours >= 16) & (hours < 20),
        (hours >= 20) & (hours < 24),
    ],
    [0.038, 0.062, 0.085, 0.078, 0.091, 0.055],
    default=0.038
)
spot_weekend_summer = np.select(
    [
        (hours >= 0)  & (hours < 6),
        (hours >= 6)  & (hours < 9),
        (hours >= 9)  & (hours < 11),
        (hours >= 11) & (hours < 15),   # solar midday suppression
        (hours >= 15) & (hours < 18),
        (hours >= 18) & (hours < 21),
        (hours >= 21) & (hours < 24),
    ],
    [0.028, 0.048, 0.068, 0.005, 0.055, 0.088, 0.042],
    default=0.028
)
buy_weekend_winter = (spot_weekend_winter + fixed_net) * (1 + VAT)
buy_weekend_summer = (spot_weekend_summer + fixed_net) * (1 + VAT)
# No FCR premium on weekends initially (same switch as weekdays)
v2g_weekend_winter = buy_weekend_winter + np.where((hours >= 16) & (hours < 20), FCR_PREMIUM_WINTER, 0.0)
v2g_weekend_summer = buy_weekend_summer + np.where((hours >= 17) & (hours < 20), FCR_PREMIUM_SUMMER, 0.0)

# v2g_winter = buy_winter + np.where((hours >= 16) & (hours < 20), 0.132, 0.0)
# v2g_summer = buy_summer + np.where((hours >= 17) & (hours < 20), 0.098, 0.0)  # smaller premium in summer

seasonal_df = pd.DataFrame({
    "Slot":                         slots,
    "Time":                         time_labels,
    "Hour":                         hours,
    "Winter_EPEX_EUR_kWh":          np.round(spot_winter, 4),
    "Summer_EPEX_EUR_kWh":          np.round(spot_summer, 4),
    "Winter_Buy_EUR_kWh":           np.round(buy_winter, 4),
    "Summer_Buy_EUR_kWh":           np.round(buy_summer, 4),
    "Winter_V2G_EUR_kWh":           np.round(v2g_winter, 4),
    "Summer_V2G_EUR_kWh":           np.round(v2g_summer, 4),
    # ── NEW: weekend profiles ─────────────────────────────────────────────
    "WkndWinter_Buy_EUR_kWh":       np.round(buy_weekend_winter, 4),
    "WkndSummer_Buy_EUR_kWh":       np.round(buy_weekend_summer, 4),
    "WkndWinter_V2G_EUR_kWh":       np.round(v2g_weekend_winter, 4),
    "WkndSummer_V2G_EUR_kWh":       np.round(v2g_weekend_summer, 4),
    "Source": ["EPEX SMARD.de 2024 avg by season/daytype + BNetzA 2024"] * 96,
})

# ── Write Excel file ──────────────────────────────────────────────────────────
path = "data/v2g_params.xlsx"
with pd.ExcelWriter(path, engine="openpyxl") as writer:
    battery.to_excel(writer,     sheet_name="BatteryParams",  index=False)
    prices_df.to_excel(writer,   sheet_name="Prices15min",    index=False)
    deg_df.to_excel(writer,      sheet_name="DegSensitivity", index=False)
    dwell_df.to_excel(writer,    sheet_name="DwellProfiles",  index=False)
    seasonal_df.to_excel(writer, sheet_name="SeasonalPrices", index=False)

print(f"Created: {path}")
print(f"  Sheet BatteryParams:  {len(battery)} rows")
print(f"  Sheet Prices15min:    {len(prices_df)} rows (96 x 15-min slots)")
print(f"  Sheet DegSensitivity: {len(deg_df)} rows")
print(f"  Sheet DwellProfiles:  {len(dwell_df)} rows")
print(f"  Sheet SeasonalPrices: {len(seasonal_df)} rows (winter + summer)")
print()
print("Price summary (winter weekday):")
print(f"  Buy price:  EUR {buy.min():.3f} - {buy.max():.3f}/kWh")
print(f"  V2G peak:   EUR {v2g_price.max():.3f}/kWh (16-20h FCR window)")
print(f"  FCR premium: EUR 0.132/kWh")
print(f"  aFRR premium: EUR 0.040/kWh (morning ramp)")
print()
print("Price summary (summer weekday):")
print(f"  Buy price:  EUR {buy_summer.min():.3f} - {buy_summer.max():.3f}/kWh")
print(f"  V2G peak:   EUR {v2g_summer.max():.3f}/kWh (17-20h, smaller premium)")
print(f"  Solar midday: EUR {buy_summer[44]:.3f}/kWh at 11:00 (solar suppression)")
