# pages/3_Telematics_Data.py
# ──────────────────────────────────────────────────────────────────────────────
# TrailerConnect Telematics Data Manager
# Upload real data → automatically used by all other pages
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Telematics Data", layout="wide")

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please log in from the main page first.")
    st.stop()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REAL_PATH = os.path.join(DATA_DIR, "telematics_real.xlsx")
TMPL_PATH = os.path.join(DATA_DIR, "telematics_template.xlsx")

st.title("📡 TrailerConnect Telematics Data")
st.markdown("""
Upload your real telematics export from **TrailerConnect** or use the synthetic template.
The uploaded file will be saved as `data/telematics_real.xlsx` and automatically used
by the Smart Charging and V2G pages.
""")

# ── Column spec card ──────────────────────────────────────────────────────────
with st.expander("📋 Required Column Format", expanded=True):
    st.markdown("""
| Column | Type | Description | Example |
|---|---|---|---|
| `Timestamp` | datetime | UTC or local, ISO 8601 | `2024-01-15 16:30:00` |
| `SoC_pct` | float | State of Charge in % | `72.5` |
| `TRU_Load_kW` | float | Reefer compressor draw (kW) | `3.2` |
| `Plugged_In` | int | 1 = at depot charger, 0 = not | `1` |
| `Ambient_Temp_C` | float | Optional: outside temp | `5.0` |
| `Setpoint_Temp_C` | float | Optional: cargo setpoint | `-18.0` |
| `TrailerID` | string | Optional: unit ID | `TRL-001` |

**Frequency**: 15-minute intervals preferred (app resamples if needed)
**Period**: Minimum 7 days; 90 days recommended for seasonal analysis
""")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### Upload Real Data")
uploaded = st.file_uploader(
    "Drop your TrailerConnect Excel export here",
    type=["xlsx", "xls"],
    help="Must contain at minimum: Timestamp, SoC_pct, TRU_Load_kW, Plugged_In"
)

if uploaded:
    try:
        df_up = pd.read_excel(uploaded, parse_dates=["Timestamp"])
        required = {"Timestamp", "SoC_pct", "TRU_Load_kW", "Plugged_In"}
        missing  = required - set(df_up.columns)
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            # Resample to 15 min if needed
            df_up = df_up.sort_values("Timestamp").set_index("Timestamp")
            freq_secs = (df_up.index[1] - df_up.index[0]).total_seconds()
            if freq_secs != 900:
                st.info(f"Data frequency detected: {freq_secs/60:.0f} min — resampling to 15 min")
                df_up = df_up.resample("15min").mean().interpolate()
            df_up = df_up.reset_index()
            df_up.to_excel(REAL_PATH, index=False)
            st.success(f"✅ Saved {len(df_up):,} rows to data/telematics_real.xlsx")
            st.session_state.telematics_df = df_up
    except Exception as e:
        st.error(f"Parse error: {e}")

# ── Load and preview whichever file is available ──────────────────────────────
st.markdown("### Data Preview")
if os.path.exists(REAL_PATH):
    df = pd.read_excel(REAL_PATH, parse_dates=["Timestamp"])
    badge = "🟢 **REAL** TrailerConnect data"
elif os.path.exists(TMPL_PATH):
    df = pd.read_excel(TMPL_PATH, parse_dates=["Timestamp"])
    badge = "🟡 **SYNTHETIC** template data — upload real data above"
else:
    st.warning("No data file found. Run `python generate_data.py` first.")
    st.stop()

st.markdown(badge)
st.markdown(f"`{len(df):,}` rows  |  "
            f"`{df['Timestamp'].min()}`  →  `{df['Timestamp'].max()}`  |  "
            f"Trailer: `{df.get('TrailerID', pd.Series(['N/A']))[0]}`")

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Avg TRU Load",   f"{df['TRU_Load_kW'].mean():.2f} kW")
with col2: st.metric("Mean SoC",       f"{df['SoC_pct'].mean():.1f}%")
with col3: st.metric("Plugged-in time",f"{df['Plugged_In'].mean()*100:.1f}%")
with col4:
    if "Ambient_Temp_C" in df.columns:
        st.metric("Avg Ambient Temp", f"{df['Ambient_Temp_C'].mean():.1f} °C")

# ── Plot ──────────────────────────────────────────────────────────────────────
st.markdown("### Visualisation")
days_avail = (df["Timestamp"].max() - df["Timestamp"].min()).days + 1
day_view = st.slider("Preview day (0 = first day)", 0, max(0, days_avail - 1), 0)

start_ts = df["Timestamp"].min() + pd.Timedelta(days=day_view)
end_ts   = start_ts + pd.Timedelta(days=1)
df_day   = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)]

if len(df_day) == 0:
    st.warning("No data for selected day.")
else:
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"TrailerConnect Data  —  {start_ts.date()}", fontsize=12)
    t = df_day["Timestamp"]

    # SoC
    axes[0].plot(t, df_day["SoC_pct"], color="#0055AA", lw=2)
    axes[0].fill_between(t, 20, df_day["SoC_pct"], alpha=0.15, color="#0055AA")
    axes[0].axhline(20, color="red",    lw=1, linestyle=":", label="Floor 20%")
    axes[0].axhline(95, color="orange", lw=1, linestyle=":", label="Ceiling 95%")
    axes[0].set_ylabel("SoC (%)"); axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    # TRU load
    axes[1].fill_between(t, df_day["TRU_Load_kW"], step="pre",
                          color="#AA0000", alpha=0.6)
    axes[1].set_ylabel("TRU Load (kW)"); axes[1].grid(True, alpha=0.3)
    if "Ambient_Temp_C" in df_day.columns:
        ax1b = axes[1].twinx()
        ax1b.plot(t, df_day["Ambient_Temp_C"], color="#888888",
                  lw=1, linestyle="--", label="Ambient °C")
        ax1b.set_ylabel("Temp (°C)")

    # Plug-in status
    axes[2].fill_between(t, df_day["Plugged_In"], step="pre",
                          color="#00AA44", alpha=0.7)
    axes[2].set_ylabel("Plugged In (1/0)")
    axes[2].set_xlabel("Time"); axes[2].grid(True, alpha=0.3)
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["Not Plugged", "Plugged"])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── Download template ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📥 Download Template")
st.markdown("Use this as a reference format for your TrailerConnect export:")
if os.path.exists(TMPL_PATH):
    with open(TMPL_PATH, "rb") as f:
        st.download_button(
            "⬇️ Download telematics_template.xlsx",
            f.read(),
            file_name="telematics_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
