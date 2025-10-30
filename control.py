import os
import opendssdirect as dss
from dss_utils import (
    compile_feeder, set_all_reg_taps_to,
    set_time_to_hour, solve, collect_bus_voltages, plot_bus_profile_general,
    add_pv, pv_debug_summary, pv_debug_summary, add_pvs_34, add_pvs_15, add_pvs_5, generate_daily_loadshape_dss,  make_voltvar_curve, make_voltwatt_curve,
    enable_invcontrol_voltvar, enable_invcontrol_voltwatt, enable_combined_controls, inject_daily_loadshape, create_all_irradiance_shapes,
    inject_irradiance_loadshapes
)

import numpy as np
import matplotlib.pyplot as plt 

FEEDER_MASTER = r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
LOADSHAPE_DIR = r"C:\Users\pozam\.vscode\cagan\loadshapes"
os.makedirs(LOADSHAPE_DIR, exist_ok=True)
OUTDIR = r"C:\Users\pozam\.vscode\cagan\results"
os.makedirs(OUTDIR, exist_ok=True)

def calculate_control_reward(voltages, p_actual, q_actual, num_pv_systems=15, pv_max_p=200.0, pv_max_q=88.0):
    """Calculate reward based on all phase voltage quality and power utilization"""
    reward = 0
    
    # 1. Voltage compliance for all phases
    voltage_penalty = 0
    ideal_bonus = 0
    violation_count = 0
    
    for v in voltages:
        if v < 0.95:
            voltage_penalty += (0.95 - v) * 100  # Heavy under-voltage penalty
            violation_count += 1
        elif v > 1.05:
            voltage_penalty += (v - 1.05) * 150  # Very heavy over-voltage penalty
            violation_count += 1
        elif 0.98 <= v <= 1.02:
            ideal_bonus += 2  # Bonus for ideal voltage
        elif 0.95 <= v <= 1.05:
            ideal_bonus += 0.5   # Base reward for acceptable voltage
    
    reward += ideal_bonus - voltage_penalty
    
    # Extra penalty for having any violations
    if violation_count > 0:
        reward -= violation_count * 5
    
    # 2. Power utilization (encourage maximum generation when safe)
    total_available = num_pv_systems * pv_max_p
    utilization = np.sum(np.abs(p_actual)) / total_available if total_available > 0 else 0
    
    # Only reward high utilization if ALL voltages are good
    max_v = np.max(voltages)
    min_v = np.min(voltages)
    
    
    if max_v <= 1.05 and min_v >= 0.95 and violation_count == 0:  # Good conditions
        power_reward = utilization * 100
    else:  # Poor conditions
        power_reward = utilization * 20
        
    reward += power_reward
    
    # 3. Voltage balance bonus (encourage balanced phases)
    voltage_std = np.std(voltages)
    if voltage_std < 0.01:
        reward += 3
    elif voltage_std < 0.02:
        reward += 1.5
    
    # 4. Reactive power efficiency
    q_penalty = np.sum(np.abs(q_actual)) / (num_pv_systems * pv_max_q) * 2.5
    reward -= q_penalty
    
    return reward

# Get PV powers for reward calculation
def get_pv_powers():
    """Get current P and Q from all PV systems"""
    p_values, q_values = [], []
        
    i = dss.PVsystems.First()
    while i > 0:
            name = dss.PVsystems.Name()
                
            dss.Circuit.SetActiveElement(f"PVSystem.{name}")
            p_kw = dss.PVsystems.kW()
            q_kvar = dss.PVsystems.kvar()
            
            p_values.append(p_kw)
            q_values.append(q_kvar)
            
            i = dss.PVsystems.Next()
    return np.sum(p_values), np.sum(q_values)

# Plot comparison (optional)
'''def plot_control_comparison(rewards_dict, filename="control_comparison.png"):
    """Plot comparison of different control strategies"""
    plt.figure(figsize=(10, 6))
    
    strategies = list(rewards_dict.keys())
    reward_values = list(rewards_dict.values())
    
    bars = plt.bar(strategies, reward_values, color=['blue', 'green', 'red', 'orange'])
    plt.title('Control Strategy Performance Comparison')
    plt.ylabel('Reward Score')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, reward_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename))
    plt.close()
    
    print(f"Control comparison saved to {filename}")'''

def run_pv_with_control_simulation(control_type):
    """
    Run PV simulation with inverter controls
    Returns: (voltages, reward, p_actual, q_actual)
    """
    print(f"Running PV + {control_type} control simulation...")
    
    # Initialize the system
    compile_feeder(FEEDER_MASTER)
    inject_daily_loadshape(day=1, save_dir=LOADSHAPE_DIR)
    
    # Set all regulators to neutral position
    set_all_reg_taps_to(4)
    
    # Inject all irradiance loadshapes
    inject_irradiance_loadshapes(day=1, save_dir=LOADSHAPE_DIR)
    
    # Add PV systems
    add_pvs_15(day=1, save_dir=LOADSHAPE_DIR)
    
    # Create control curves
    make_voltvar_curve("VV1")
    make_voltwatt_curve("VW1")
    
    # Enable appropriate controls
    if control_type == "voltvar":
        enable_invcontrol_voltvar("VV1")
    elif control_type == "voltwatt":
        enable_invcontrol_voltwatt("VW1")
    else:  # combined
        enable_combined_controls("VV1", "VW1")
    
    # Set simulation time and solve
    set_time_to_hour()
    
    dss.Text.Command("set maxiterations=300")
    dss.Text.Command("set maxcontroliter=150")
    dss.Text.Command("set controlmode=exponential")  
    
    try:
        solve()
        print("Solution converged successfully!")
    except Exception as e:
        print("Exception during solve:", e)
        # Try to continue even if there's an error
        
    # Show event log to check for control actions (optional)
    '''try:
        dss.Text.Command("Show Eventlog")
        event_log = dss.Text.Result()
        print("Event Log Path:", event_log)
        
        # Try to read the event log file
        event_log_path = event_log.strip()
        if os.path.exists(event_log_path):
            with open(event_log_path, 'r') as f:
                event_content = f.read()
                print("Event Log Content:")
                print(event_content[-1000:])  # Show last 1000 characters
        else:
            print("Event log file not found at:", event_log_path)
    except Exception as e:
        print("Error reading event log:", e)'''
    
    # Show PV summary (optional)
    '''pv_debug_summary()'''
    
    # Collect and plot voltages
    volts = collect_bus_voltages(per_phase=True, sort_by="distance")
    all_voltages = []
    for bus_data in volts:
        phase_voltages = bus_data[2]
        for v in phase_voltages:
            if not np.isnan(v):
                all_voltages.append(v)
    
    all_voltages = np.array(all_voltages)
    p_actual, q_actual = get_pv_powers()
    reward = calculate_control_reward(all_voltages, p_actual, q_actual)
    
    print(f"{control_type.upper()} Control Reward: {reward:.2f}")
    
    plot_bus_profile_general(
        volts,
        title=f"Voltage Profile with 15 PV + {control_type.upper()} Control - Time=12:00, Taps=4 - Reward: {reward:.2f}",
        band=(0.95, 1.05)
    )
    
    # Return voltages, reward, and power values
    return volts, reward, p_actual, q_actual

if __name__ == "__main__":
    rewards = {}  # Initialize rewards dictionary
    
    # Test control strategies
    print("Testing Improved Volt-VAR control...")
    volts_vv, reward_vv, p_vv, q_vv = run_pv_with_control_simulation("voltvar")
    rewards["Volt-VAR"] = reward_vv
    print(f"PV Generation (Volt-VAR): P={p_vv} kW, Q={q_vv} kVAR")
    
    
    print("\nTesting Improved Volt-Watt control...")
    volts_vw, reward_vw, p_vw, q_vw = run_pv_with_control_simulation("voltwatt")
    rewards["Volt-Watt"] = reward_vw
    print(f"PV Generation (Volt-Watt): P={p_vw} kW, Q={q_vw} kVAR")
    
    print("\nTesting Improved Combined control...")
    volts_combined, reward_combined, p_combined, q_combined = run_pv_with_control_simulation("combined")
    rewards["Combined"] = reward_combined
    print(f"PV Generation (Combined): P={p_combined} kW, Q={q_combined} kVAR")
    
    print("\n=== CONTROL STRATEGY COMPARISON ===")
    for strategy, reward in rewards.items():
        print(f"{strategy}: {reward:.2f}")
    
    print(f"\nBest performing strategy: {max(rewards, key=rewards.get)}")