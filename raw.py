import os
import opendssdirect as dss
from dss_utils import (
    compile_feeder, set_all_reg_taps_to,
    set_time_to_hour, solve, collect_bus_voltages, plot_bus_profile_general,
    add_pv, pv_debug_summary, pv_debug_summary, add_pvs_34, add_pvs_15, add_pvs_5, generate_daily_loadshape_dss, inject_daily_loadshape, create_all_irradiance_shapes,
    inject_irradiance_loadshapes
)

FEEDER_MASTER = r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
LOADSHAPE_DIR = r"C:\Users\pozam\.vscode\cagan\loadshapes"
os.makedirs(LOADSHAPE_DIR, exist_ok=True)
OUTDIR = r"C:\Users\pozam\.vscode\cagan\results"
os.makedirs(OUTDIR, exist_ok=True)

def run_pv_simulation():
    """Run simulation with PV integration to observe voltage issues"""
    print("Running PV integration simulation...")
    
    # Initialize OpenDSS
    compile_feeder(FEEDER_MASTER)
    
    inject_daily_loadshape(day=1, save_dir=LOADSHAPE_DIR)
    
    # Set all regulators to neutral position
    set_all_reg_taps_to(4)
    
    # Inject all irradiance loadshapes
    inject_irradiance_loadshapes(day=1, save_dir=LOADSHAPE_DIR)
    
    # Add PV systems
    add_pvs_15(day=1, save_dir=LOADSHAPE_DIR)

    # Set simulation time to noon
    set_time_to_hour()
    
    # Solve power flow
    solve()
    # PV debug summary
    pv_debug_summary()
    
    # Collect voltage data
    volts = collect_bus_voltages(per_phase=True, sort_by="distance")
    
    # Plot and save results
    plot_bus_profile_general(
        volts,
        title="Voltage Profile with 15 PV Integration - Time=12:00, Taps=4",
        band=(0.95, 1.05),
        show=True
    )
    return volts

if __name__ == "__main__":
    create_all_irradiance_shapes(day=1, save_dir=LOADSHAPE_DIR)
    pv_voltages = run_pv_simulation()