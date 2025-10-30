import os
from dss_utils import (
    compile_feeder, set_all_reg_taps_to,
    set_time_to_hour, solve, collect_bus_voltages, plot_bus_profile_general, pv_debug_summary, generate_daily_loadshape_dss, inject_daily_loadshape
)

FEEDER_MASTER = r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
LOADSHAPE_DIR = r"C:\Users\pozam\.vscode\cagan\loadshapes"
os.makedirs(LOADSHAPE_DIR, exist_ok=True)
OUTDIR = r"C:\Users\pozam\.vscode\cagan\results"
os.makedirs(OUTDIR, exist_ok=True)

def run_baseline_simulation():
    """Run baseline simulation without PV to establish reference voltages"""
    print("Running baseline simulation (no PV)...")
    
    # Initialize OpenDSS
    compile_feeder(FEEDER_MASTER)
    
    inject_daily_loadshape(day=1, save_dir=LOADSHAPE_DIR)

    # Set all regulators to neutral position
    set_all_reg_taps_to(4)
    
    # Set simulation time to noon
    set_time_to_hour()
    
    # Solve power flow
    solve()
    pv_debug_summary()
    # Collect voltage data
    volts = collect_bus_voltages(per_phase=True, sort_by="distance")
    
    # Plot and save results
    plot_bus_profile_general(
        volts,
        title="Baseline Voltage Profile (No PV) - Time=12:00, Taps=4",
        band=(0.95, 1.05),
        show=True
    )
    return volts

if __name__ == "__main__":
    baseline_voltages = run_baseline_simulation()