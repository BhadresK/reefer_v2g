#!/usr/bin/env python3
"""
run_all.py
Master pipeline - calls each project script individually, in order.
Place this file in the same folder as your other 5 scripts and run:

    python run_all.py

Optional flags:
    python run_all.py --skip-data    # skip download if CSV already exists
    python run_all.py --skip-train   # skip ML training if model already exists
    python run_all.py --fast         # faster ML training (fewer trees)
    python run_all.py --fleet 5      # simulate 5 trailers
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class C:
    HEADER = "\033[95m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"


def banner(title, step, total):
    line = "=" * 65
    print(f"\n{C.BOLD}{C.CYAN}{line}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  STEP {step}/{total}  --  {title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{line}{C.RESET}")


def success(msg):
    print(f"\n{C.GREEN}{C.BOLD}  OK  {msg}{C.RESET}")


def failure(msg):
    print(f"\n{C.RED}{C.BOLD}  FAILED  {msg}{C.RESET}")


def info(msg):
    print(f"{C.YELLOW}  ->  {msg}{C.RESET}")


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_step(label, script, args, step, total, skip_if=None, skip_reason=""):
    """
    Runs one script as a separate subprocess.
    This does NOT combine code -- it calls the original file directly,
    exactly like typing: python script_name.py [args] in the terminal.
    """
    banner(label, step, total)

    # Skip if output already exists and user passed --skip flag
    if skip_if is not None and skip_if.exists():
        info(f"Already exists: {skip_if}")
        info(f"Skipping {skip_reason}  (delete the file to force re-run)")
        success(f"STEP {step} skipped")
        return True

    # Find the script in the same folder as this file
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        failure(f"Script not found: {script_path}")
        failure("Make sure run_all.py is in the same folder as all 5 project scripts.")
        return False

    cmd = [sys.executable, str(script_path)] + args
    info(f"Running: {' '.join(cmd)}")
    print()

    start  = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        success(f"STEP {step} completed in {fmt_time(elapsed)}")
        return True
    else:
        failure(f"STEP {step} FAILED (exit code {result.returncode})")
        failure("Pipeline stopped. Fix the error above and re-run.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Runs all V2G project scripts in the correct order"
    )
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip SMARD download if processed CSV already exists")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip ML training if model file already exists")
    parser.add_argument("--fast",       action="store_true",
                        help="Faster ML training with fewer trees (for testing)")
    parser.add_argument("--fleet",      type=int, default=1,
                        help="Number of trailers to simulate (default: 1)")
    args = parser.parse_args()

    wall_start = time.time()
    now        = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    print(f"\n{C.BOLD}{C.HEADER}{'=' * 65}{C.RESET}")
    print(f"{C.BOLD}{C.HEADER}  S.KOe COOL -- V2G FULL PIPELINE{C.RESET}")
    print(f"{C.BOLD}{C.HEADER}  Started : {now}{C.RESET}")
    print(f"{C.BOLD}{C.HEADER}{'=' * 65}{C.RESET}")
    print(f"\n  Real data     : 2022 - 2025  (downloaded from SMARD)")
    print(f"  Forecast year : 2026         (ML-predicted prices)")
    print(f"  Fleet size    : {args.fleet} trailer(s)")
    print(f"  Fast mode     : {'yes' if args.fast else 'no'}")
    print()

    TOTAL = 9
    step  = 0

    # ------------------------------------------------------------------
    # STEP 1 -- fetch_smard_data.py
    # Downloads 2022-2025 real electricity prices from SMARD.de
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label       = "Fetch SMARD Electricity Prices  (2022-2025)",
        script      = "fetch_smard_data.py",
        args        = ["--years", "2022", "2025"],
        step        = step,
        total       = TOTAL,
        skip_if     = Path("data/smard_prices_processed.csv") if args.skip_data else None,
        skip_reason = "SMARD download",
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 2 -- make_data.py
    # Builds the battery + depot config Excel file
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label  = "Build Battery & Depot Config  (make_data.py)",
        script = "make_data.py",
        args   = [],
        step   = step,
        total  = TOTAL,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3 -- train_forecaster.py
    # Trains the ML model on 2022-2025 data to forecast 2026 prices
    # ------------------------------------------------------------------
    step += 1
    train_args = ["--fast"] if args.fast else []
    ok = run_step(
        label       = "Train ML Price Forecaster  (Gradient Boosted Trees)",
        script      = "train_forecaster.py",
        args        = train_args,
        step        = step,
        total       = TOTAL,
        skip_if     = Path("data/price_model.pkl") if args.skip_train else None,
        skip_reason = "ML model training",
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 4 -- run_optimisation.py
    # Single-day sanity check: confirms optimizer is working correctly
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label  = "Single-Day Optimisation Sanity Check  (run_optimisation.py)",
        script = "run_optimisation.py",
        args   = [],
        step   = step,
        total  = TOTAL,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 5 -- run_year_simulation.py  --year 2023  (real prices)
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label  = "Full-Year Simulation: 2023  (Real SMARD Prices)",
        script = "run_year_simulation.py",
        args   = [
            "--year",     "2023",
            "--mode",     "real",
            "--fleet",    str(args.fleet),
            "--save-csv", "data/year_results_2023.csv",
        ],
        step  = step,
        total = TOTAL,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 6 -- run_year_simulation.py  --year 2024  (real prices)
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label  = "Full-Year Simulation: 2024  (Real SMARD Prices)",
        script = "run_year_simulation.py",
        args   = [
            "--year",     "2024",
            "--mode",     "real",
            "--fleet",    str(args.fleet),
            "--save-csv", "data/year_results_2024.csv",
        ],
        step  = step,
        total = TOTAL,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 7 -- run_year_simulation.py  --year 2025  (real prices)
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label  = "Full-Year Simulation: 2025  (Real SMARD Prices)",
        script = "run_year_simulation.py",
        args   = [
            "--year",     "2025",
            "--mode",     "real",
            "--fleet",    str(args.fleet),
            "--save-csv", "data/year_results_2025.csv",
        ],
        step  = step,
        total = TOTAL,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 8 -- run_year_simulation.py  --year 2026  (ML forecast)
    # ------------------------------------------------------------------
    step += 1
    ok = run_step(
        label  = "Full-Year Simulation: 2026  (ML-Forecasted Prices)",
        script = "run_year_simulation.py",
        args   = [
            "--year",     "2026",
            "--mode",     "forecast",
            "--fleet",    str(args.fleet),
            "--save-csv", "data/year_results_2026.csv",
        ],
        step  = step,
        total = TOTAL,
    )
    if not ok:
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 9 -- Print cross-year summary table
    # ------------------------------------------------------------------
    step += 1
    banner("Cross-Year Summary", step, TOTAL)
    _print_summary(args.fleet)
    success(f"STEP {step} completed")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_elapsed = time.time() - wall_start
    print(f"\n{C.BOLD}{C.GREEN}{'=' * 65}{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}  FULL PIPELINE COMPLETE{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}  Total time: {fmt_time(total_elapsed)}{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}{'=' * 65}{C.RESET}")
    print()
    print("  Files created:")
    print("    data/smard_prices_raw.csv        -- raw EPEX spot prices")
    print("    data/smard_prices_processed.csv  -- all-in prices + features")
    print("    data/v2g_params.xlsx             -- battery & depot config")
    print("    data/price_model.pkl             -- trained ML forecaster")
    print("    data/model_metrics.json          -- ML accuracy metrics")
    print("    data/year_results_2023.csv       -- 2023 daily results")
    print("    data/year_results_2024.csv       -- 2024 daily results")
    print("    data/year_results_2025.csv       -- 2025 daily results")
    print("    data/year_results_2026.csv       -- 2026 forecast results")
    print("    results.png                      -- single-day scenario chart")
    print("    results_year_summary.png         -- annual summary chart")
    print("    results_weekly_heatmap.png       -- 52-week heatmap")
    print()


def _print_summary(fleet_size=1):
    """Loads each year CSV and prints a simple comparison table."""
    try:
        import pandas as pd
    except ImportError:
        info("pandas not available -- skipping summary table")
        return

    rows = []
    for year in [2023, 2024, 2025, 2026]:
        csv = Path(f"data/year_results_{year}.csv")
        if not csv.exists():
            info(f"  {year}: result file not found, skipping")
            continue
        try:
            df      = pd.read_csv(csv)
            a_col   = next((c for c in df.columns if "A_cost" in c), None)
            c_col   = next((c for c in df.columns if "C_cost" in c), None)
            v_col   = next((c for c in df.columns if "v2g_rev" in c.lower()), None)
            s_col   = next((c for c in df.columns if "saving" in c.lower()), None)

            if a_col and c_col:
                saving = (df[a_col].sum() - df[c_col].sum()) * fleet_size
            elif s_col:
                saving = df[s_col].sum() * fleet_size
            else:
                info(f"  {year}: could not find cost columns in {csv.name}")
                continue

            v2g_rev = df[v_col].sum() * fleet_size if v_col else 0.0
            source  = "ML Forecast" if year == 2026 else "Real SMARD"
            rows.append((year, source, f"EUR {saving:,.0f}", f"EUR {v2g_rev:,.0f}"))
        except Exception as e:
            info(f"  {year}: error reading CSV -- {e}")

    if not rows:
        info("No result CSVs found -- check that all simulations completed.")
        return

    print()
    print(f"  {'Year':<8}  {'Source':<14}  {'Annual Saving':<18}  {'V2G Revenue'}")
    print(f"  {'----':<8}  {'--------------':<14}  {'------------------':<18}  {'-----------'}")
    for year, source, saving, v2g in rows:
        colour = C.CYAN if year == 2026 else C.RESET
        print(f"  {colour}{year:<8}  {source:<14}  {saving:<18}  {v2g}{C.RESET}")
    print()
    print(f"  {C.DIM}Fleet size: {fleet_size}.  "
          f"Saving = Scenario A (Dumb) minus Scenario C (MILP V2G).{C.RESET}")
    print()


if __name__ == "__main__":
    main()