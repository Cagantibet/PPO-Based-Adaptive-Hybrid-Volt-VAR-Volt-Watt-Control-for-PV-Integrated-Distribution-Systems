import opendssdirect as dss
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def compile_feeder(master_path: str) -> None:
    dss.Basic.ClearAll()
    dss.Text.Command(f"Compile [{master_path}]")

def set_time_to_hour(hour: int = 12) -> None:
    """
    Set simulation to hourly mode and point to a specific hour.
    Must be used with 'daily' loadshape.
    """
    dss.Text.Command("Set Mode=Daily")
    dss.Text.Command("Set stepsize=1h")
    dss.Text.Command("Set number=1")
    dss.Text.Command(f"Set hour={hour}")
    
def generate_daily_loadshape_dss(day_num: int, save_dir: str = "loadshapes", name_prefix="LoadShape_Day", seed=None) -> str:
    if seed is not None:
        np.random.seed(seed + day_num)

    base_shape = np.array([
        0.20, 0.20, 0.20, 0.20,  # 12am–3am (2MW / 10MW peak = 0.2)
        0.25, 0.30, 0.40, 0.50,  # 4am–7am (rising to ~5MW = 0.5)
        0.60, 0.65, 0.70, 0.70,  # 8am–11am (rising to ~7MW = 0.7, then slight drop)
        0.65, 0.55, 0.50, 0.45,  # 12pm–3pm (noon dip to ~5MW = 0.5)
        0.50, 0.70, 0.90, 1.00,  # 4pm–7pm (evening peak to 10MW = 1.0)
        0.85, 0.70, 0.50, 0.30   # 8pm–11pm (decline to ~3MW = 0.3)
    ])

    # Adjusted variation scale based on typical load fluctuations
    variation_scale = np.array([
        0.02, 0.02, 0.02, 0.02,  # Night (small variations)
        0.03, 0.04, 0.05, 0.06,  # Morning ramp (increasing variability)
        0.04, 0.03, 0.03, 0.04,  # Late morning (moderate variability)
        0.05, 0.06, 0.05, 0.04,  # Noon dip (higher variability during transition)
        0.06, 0.08, 0.10, 0.12,  # Evening spike (highest variability at peak)
        0.08, 0.06, 0.04, 0.03   # Night drop (decreasing variability)
    ])

    noise = np.random.normal(0, variation_scale)
    shape = np.clip(base_shape + noise, 0.15, 1.15)  # Adjusted min/max based on actual range
    
    shape_str = " ".join([f"{v:.4f}" for v in shape])

    shape_name = f"{name_prefix}{day_num:03d}"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{shape_name}.dss")

    with open(filename, "w") as f:
        f.write(f"New LoadShape.{shape_name} npts=24 interval=1 mult=({shape_str})\n")

    return filename

def inject_daily_loadshape(day=1, save_dir="loadshapes"):
    shape_path = generate_daily_loadshape_dss(day_num=day, save_dir=save_dir)
    dss.Text.Command(f"Redirect [{shape_path}]")
    dss.Text.Command(f"BatchEdit Load..* daily=LoadShape_Day{day:03d}")
    
def create_irradiance_loadshape(day: int, pv_name: str, save_dir: str = None, seed=None) -> tuple[str, str]:
    if save_dir is None:
        save_dir = "loadshapes"  
    
    os.makedirs(save_dir, exist_ok=True)

    if seed is not None:
        np.random.seed(seed + day)

    base_shape = np.array([
        0.00, 0.00, 0.00, 0.00,  # Midnight–3am
        0.10, 0.30, 0.60, 0.80,  # 4am–7am
        1.00, 1.00, 1.00, 0.90,  # 8am–11am
        0.80, 0.70, 0.60, 0.50,  # 12pm–3pm
        0.40, 0.30, 0.20, 0.10,  # 4pm–7pm
        0.05, 0.00, 0.00, 0.00   # 8pm–11pm
    ])

    variation_scale = 0.05
    noise = np.random.normal(1.0, variation_scale, 24)
    shape = np.clip(base_shape * noise, 0.0, 1.0)

    shape_str = " ".join([f"{v:.4f}" for v in shape])
    shape_name = f"Irradiance_{pv_name}_Day{day:03d}"
    filename = os.path.join(save_dir, f"{shape_name}.dss")

    with open(filename, "w") as f:
        f.write(f"New LoadShape.{shape_name} npts=24 interval=1 mult=({shape_str})\n")

    return filename, shape_name

def create_all_irradiance_shapes(day=1, save_dir=None):
    """Create irradiance shapes for all PV systems before adding them"""
    if save_dir is None:
        save_dir = "loadshapes"
    
    pv_names = []
    phase_a_nodes = [1, 8, 10, 11, 20, 30, 33, 37, 46, 48, 71, 79, 88]
    phase_b_nodes = [22, 39, 43, 48, 59, 83, 90, 107]
    phase_c_nodes = [4, 6, 16, 17, 24, 32, 41, 48, 63, 75, 85, 92, 104]
    
    for node in phase_a_nodes:
        pv_names.append(f"PV_A_{node}")
    for node in phase_b_nodes:
        pv_names.append(f"PV_B_{node}")
    for node in phase_c_nodes:
        pv_names.append(f"PV_C_{node}")
    
    # Create all loadshape files first
    for pv_name in pv_names:
        create_irradiance_loadshape(day=day, pv_name=pv_name, save_dir=save_dir)
    
    return pv_names

def inject_irradiance_loadshapes(day=1, save_dir=None):
    """
    Inject all irradiance loadshapes for the specified day.
    """
    if save_dir is None:
        save_dir = "loadshapes"
    
    # Load ALL irradiance files
    pattern = os.path.join(save_dir, "Irradiance_*.dss")
    shape_files = glob.glob(pattern)

    for path in shape_files:
            dss.Text.Command(f"Redirect [{path}]")


def solve():
    dss.Solution.Solve()

def set_all_reg_taps_to(tapnum: int = 5) -> None:
    i = dss.RegControls.First()
    while i > 0:
        name = dss.RegControls.Name()
        dss.Text.Command(f"Edit RegControl.{name} TapNum={tapnum}")
        dss.Text.Command(f"Disable RegControl.{name}")
        i = dss.RegControls.Next()

def collect_bus_voltages(per_phase: bool = True,
                         sort_by: str = "index",
                         limit_to_idx: int = None):
    rows = []
    names = dss.Circuit.AllBusNames()
    for i, name in enumerate(names, start=1):
        dss.Circuit.SetActiveBus(name)
        pu = dss.Bus.puVmagAngle()
        mags = pu[0::2]
        if per_phase:
            mags = list(mags) + [np.nan] * (3 - len(mags))
            vals = np.array(mags[:3], dtype=float)
        else:
            vals = np.array([np.min(mags) if mags else np.nan], dtype=float)
        dist = dss.Bus.Distance()
        rows.append((i, name, dist, vals))

    if sort_by == "distance":
        rows.sort(key=lambda r: (r[2], r[0]))

    volts = [(k+1, name, vals) for k, (_, name, _, vals) in enumerate(rows)]
    if limit_to_idx is not None:
        volts = [t for t in volts if t[0] <= limit_to_idx]
    return volts

def plot_bus_profile_general(volts,
                             title="",
                             band=(0.95, 1.05),
                             per_phase=True,
                             colors=("C0","C2","C1"),
                             linewidth=1.4,
                             ylim=None,
                             save_path=None,
                             show=True):
    idx = np.array([i for i, _, _ in volts], dtype=float)
    V = np.array([v for _, _, v in volts], dtype=float)

    plt.figure(figsize=(10, 5))
    if per_phase:
        labels = ("Phase A", "Phase B", "Phase C")
        for k in range(3):
            mask = ~np.isnan(V[:, k])
            if np.any(mask):
                plt.plot(idx[mask], V[mask, k],
                         color=colors[k], linewidth=linewidth, label=labels[k])
    else:
        mask = ~np.isnan(V[:, 0])
        if np.any(mask):
            plt.plot(idx[mask], V[mask, 0], color="C0",
                     linewidth=linewidth, label="Worst phase")

    if band:
        plt.axhline(band[0], color="red", linestyle="--", linewidth=1.0)
        plt.axhline(band[1], color="red", linestyle="--", linewidth=1.0)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Node Number")
    plt.ylabel("Voltage (p.u.)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    if per_phase:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

# ----------------------- PV helpers ---------------------------

def add_pv(name, bus_dot_phase, kva=200.0, kv=2.4, phases=1, pf=0.90, day=1, save_dir=None):
    _, shape_name = create_irradiance_loadshape(day=day, pv_name=name, save_dir=save_dir)
    dss.Text.Command(
        f"New PVSystem.{name} bus1={bus_dot_phase} phases={phases} kv={kv} "
        f"kVA={kva} Pmpp={kva*pf} pf={pf} daily={shape_name}"
    )

def add_pvs_34(day=1, save_dir=None):
    # Phase A: 13 PVs (200 kVA each)
    phase_a_nodes = [1, 8, 10, 11, 20, 30, 33, 37, 46, 48, 71, 79, 88]
    for node in phase_a_nodes:
        add_pv(f"PV_A_{node}", f"{node}.1", day=day, save_dir=save_dir)
    
    # Phase B: 8 PVs (200 kVA each)
    phase_b_nodes = [22, 39, 43, 48, 59, 83, 90, 107]
    for node in phase_b_nodes:
        add_pv(f"PV_B_{node}", f"{node}.2", day=day, save_dir=save_dir)
    
    # Phase C: 13 PVs (200 kVA each)
    phase_c_nodes = [4, 6, 16, 17, 24, 32, 41, 48, 63, 75, 85, 92, 104]
    for node in phase_c_nodes:
        add_pv(f"PV_C_{node}", f"{node}.3", day=day, save_dir=save_dir)
 
def add_pvs_15(day=1, save_dir=None):  
    phase_a_nodes = [1, 8, 10, 11, 20]
    for node in phase_a_nodes:
        add_pv(f"PV_A_{node}", f"{node}.1", day=day, save_dir=save_dir)

    phase_b_nodes = [22, 39, 43, 48, 59]
    for node in phase_b_nodes:
        add_pv(f"PV_B_{node}", f"{node}.2", day=day, save_dir=save_dir)

    phase_c_nodes = [4, 6, 16, 17, 24]
    for node in phase_c_nodes:
        add_pv(f"PV_C_{node}", f"{node}.3", day=day, save_dir=save_dir)
        
def add_pvs_5(day=1, save_dir=None):
    # Phase A: 2 PVs (200 kVA each)
    phase_a_nodes = [1, 8]
    for node in phase_a_nodes:
        add_pv(f"PV_A_{node}", f"{node}.1", day=day, save_dir=save_dir)
    
    # Phase B: 1 PVs (200 kVA each)
    phase_b_nodes = [22]
    for node in phase_b_nodes:
        add_pv(f"PV_B_{node}", f"{node}.2", day=day, save_dir=save_dir)
    
    # Phase C: 2 PVs (200 kVA each)
    phase_c_nodes = [4, 6]
    for node in phase_c_nodes:
        add_pv(f"PV_C_{node}", f"{node}.3", day=day, save_dir=save_dir)

def pv_debug_summary() -> None:
    """Print PV count and total kW/kvar."""
    i = dss.PVsystems.First()
    count, P, Q = 0, 0.0, 0.0
    names = []
    while i > 0:
        names.append(dss.PVsystems.Name())
        pows = dss.CktElement.Powers()
        P += sum(pows[0::2])
        Q += sum(pows[1::2])
        count += 1
        i = dss.PVsystems.Next()
    print(f"[PV] count={count}, total_P_kW={P:.1f}, total_Q_kvar={Q:.1f}")
    if names:
        print("[PV] first few:", names[:8])
        
# ----------------------- Control helpers ---------------------------

def make_voltvar_curve(curve_name="VV1", vpoints=(0.97, 0.99, 1.01, 1.03), qpoints=(0.44, 0.0, 0.0, -0.44)):
    """Create a Volt-VAR curve"""
    x = " ".join(map(str, vpoints))
    y = " ".join(map(str, qpoints))
    dss.Text.Command(f"New XYCurve.{curve_name} npts={len(vpoints)} xarray=[{x}] yarray=[{y}]")

def make_voltwatt_curve(curve_name="VW1", vpoints=(1.00, 1.02, 1.04), ppoints=(1.0, 0.5, 0.2)):
    """Create a Volt-Watt curve"""
    x = " ".join(map(str, vpoints))
    y = " ".join(map(str, ppoints))
    dss.Text.Command(f"New XYCurve.{curve_name} npts={len(vpoints)} xarray=[{x}] yarray=[{y}]")
def existing_pv_names():
    """Get list of all PV system names"""
    pv_names = []
    dss.PVsystems.First()
    while True:
        name = dss.PVsystems.Name()
        if not name:
            break
        pv_names.append(name)
        if not dss.PVsystems.Next():
            break
    return pv_names

def enable_invcontrol_voltvar(curve_name="VV1"):
    """Enable Volt-VAR control"""
    pv_names = existing_pv_names()
    
    # Create individual InvControl for each PV system
    for pv_name in pv_names:
        dss.Text.Command(
            f"New InvControl.{pv_name}_VV mode=VOLTVAR vvc_curve1={curve_name} "
            f"PVSystemList={pv_name} "
            f"deltaQ_factor=0.3 voltagechangetolerance=0.005 varchangetolerance=0.02 "
            f"eventlog=yes"
        )

def enable_invcontrol_voltwatt(curve_name="VW1"):
    """Enable Volt-Watt"""
    pv_names = existing_pv_names()
    
    # Create individual InvControl for each PV system
    for pv_name in pv_names:
        dss.Text.Command(
            f"New InvControl.{pv_name}_VW mode=VOLTWATT voltwatt_curve={curve_name} "
            f"PVSystemList={pv_name} "
            f"deltaP_factor=0.3 voltagechangetolerance=0.005 "
            f"eventlog=yes"
        )

def enable_combined_controls(voltvar_curve="VV1", voltwatt_curve="VW1"):
    """Enable both controls"""
    pv_names = existing_pv_names()
    
    for pv_name in pv_names:
        # Volt-Watt control
        dss.Text.Command(
            f"New InvControl.{pv_name}_VW mode=VOLTWATT voltwatt_curve={voltwatt_curve} "
            f"PVSystemList={pv_name} "
            f"deltaP_factor=0.3 voltagechangetolerance=0.005 "
            f"eventlog=yes"
        )
        
        # Volt-VAR control
        dss.Text.Command(
            f"New InvControl.{pv_name}_VV mode=VOLTVAR vvc_curve1={voltvar_curve} "
            f"PVSystemList={pv_name} "
            f"deltaQ_factor=0.3 voltagechangetolerance=0.005 varchangetolerance=0.02 "
            f"eventlog=yes"
        )