# curve_daily_simulator.py
import os
import numpy as np
import opendssdirect as dss
from dss_utils import (
    compile_feeder, set_all_reg_taps_to, pv_debug_summary,
    set_time_to_hour, solve, collect_bus_voltages, plot_bus_profile_general,
    add_pvs_15, inject_daily_loadshape, create_all_irradiance_shapes, inject_irradiance_loadshapes
)
import torch
from curve_final import DynamicCurveOptimizationEnv, DynamicCurvePPO, PPOMemory

# Constants
FEEDER_MASTER = r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
LOADSHAPE_DIR = r"C:\Users\pozam\.vscode\cagan\loadshapes"
OUTDIR = r"C:\Users\pozam\.vscode\cagan\results"

def run_multi_day_simulation(model_path, num_days=7, target_hour=12):
    """
    Run simulation over multiple days using the trained model
    Returns average reward across all days
    
    Args:
        target_hour: Specific hour to evaluate (0-23)
    """
    print(f"Running {num_days}-day simulation with Dynamic Curve Optimization...")
    print(f"Evaluating at hour {target_hour}:00 each day")
    
    daily_rewards = []
    
    # Load the trained model once
    env = DynamicCurveOptimizationEnv(day=1)
    state_dim = env.num_voltage_points
    action_dim = env.action_dim
    ppo = DynamicCurvePPO(state_dim, action_dim)
    
    if os.path.exists(model_path):
        ppo.load(model_path)
        print("✅ Loaded trained dynamic curve model")
    else:
        print("❌ Model not found! Using standard curves.")
        return 0
    
    for day in range(1, num_days + 1):
        print(f"Day {day}...")
        
        # Set up environment for this day
        env = DynamicCurveOptimizationEnv(day=day)
        
        # Set the specific hour
        set_time_to_hour(target_hour)
        
        # Solve to update system state
        try:
            solve()
        except:
            pass
        
        # Run simulation for this day
        state = env.reset()
        memory = PPOMemory()
        
        for step in range(50):  # Run for 50 steps per day
            action = ppo.get_action(state, memory)
            state, reward, done, _ = env.step(action)
            final_reward = reward
            
            if done:
                break

        daily_rewards.append(final_reward)
        print(f"  Day {day} reward: {final_reward:.2f}")

    # Calculate average reward
    avg_reward = np.mean(daily_rewards)
    print(f"\n=== {num_days}-DAY SIMULATION RESULTS (Hour {target_hour}:00) ===")
    print(f"Average Daily Reward: {avg_reward:.2f}")
    print(f"Best Day: {np.max(daily_rewards):.2f}")
    print(f"Worst Day: {np.min(daily_rewards):.2f}")
    
    return avg_reward

def run_daily_simulation(model_path, target_day=1, target_hour=12):
    """
    Run a proper daily simulation with fixed load/irradiance profiles for the day
    using dynamic curve optimization
    """
    
    # Initialize environment for the specific day
    print(f"Setting up Day {target_day} simulation with Dynamic Curve Optimization...")
    env = DynamicCurveOptimizationEnv(day=target_day)
    state_dim = env.num_voltage_points
    action_dim = env.action_dim
    ppo = DynamicCurvePPO(state_dim, action_dim)
    
    # Load trained model
    if os.path.exists(model_path):
        ppo.load(model_path)
        print(f"✅ Loaded trained dynamic curve model from {model_path}")
    else:
        print("❌ Model not found! Using standard curves.")
    
    print("Starting 24-hour daily simulation with optimized curves...")
    print(f"Using fixed load/irradiance profiles for Day {target_day}")
    print("Profiles will remain constant throughout the simulation")
    
    # Storage for hourly results
    hourly_results = {}
    
    # Simulate each hour of the day
    for hour in range(24):
        print(f"\n--- Hour {hour}:00 ---")
        set_time_to_hour(hour)

        
        # Solve to update system state with new time
        solve()

        
        # Reset environment for this hour
        state = reset_for_hour(env)
        
        # Run control for this hour
        hour_voltages = []
        hour_actions = []
        hour_rewards = []
        
        done = False
        memory = PPOMemory()
        
        for timestep in range(50):  # Allow time to converge
            # Get curve parameters from trained policy
            action = ppo.get_action(state, memory)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store results for this timestep
            hour_voltages.append(next_state)
            hour_actions.append(action)
            hour_rewards.append(reward)
            
            state = next_state
            
            if done or timestep == 49:  # Converged or max timesteps
                # Take the final state as the result for this hour
                final_voltages = hour_voltages[-1]
                final_actions = hour_actions[-1]
                p_actuals, q_actuals = env.get_pv_powers()
                
                hourly_results[hour] = {
                    'voltages': final_voltages,
                    'actions': final_actions,
                    'pv_powers': (p_actuals, q_actuals),
                    'min_voltage': np.min(final_voltages),
                    'max_voltage': np.max(final_voltages),
                    'reward': (hour_rewards[-1],),
                    'voltvar_params': env.voltvar_params.copy(),
                    'voltwatt_params': env.voltwatt_params.copy()
                }
                
                print(f"  Converged after {timestep+1} steps")
                print(f"  Reward: {(hour_rewards)[-1]:.2f}")
                print(f"  Voltage range: {np.min(final_voltages):.3f} - {np.max(final_voltages):.3f} pu")
                print(f"  Total PV generation: {np.sum(p_actuals):.1f} kW")
                print(f"  Total reactive power: {np.sum(q_actuals):.1f} kVAR")
                break
    
    # Generate snapshot for target hour
    if target_hour in hourly_results:
        print(f"\n=== Generating snapshot for hour {target_hour}:00 ===")
        generate_hourly_snapshot(hourly_results[target_hour], target_hour, target_day)
    
    # Save all results
    save_daily_results(hourly_results, target_day)
    
    return hourly_results

def reset_for_hour(env):
    """Reset PV systems to initial state without recompiling profiles"""
    # Reset to standard curves
    env.voltvar_params = env.initialize_voltvar_curves()
    env.voltwatt_params = env.initialize_voltwatt_curves()
    
    # Apply initial curves
    env.apply_curve_control()
    
    # Solve power flow
    try:
        solve()
    except:
        pass
    
    return env.get_all_phase_voltages()

def generate_hourly_snapshot(hour_data, hour, day):
    """Generate voltage profile plot for a specific hour"""
    # Get current voltages from OpenDSS
    volts = collect_bus_voltages(per_phase=True, sort_by="index")
    
    plot_bus_profile_general(
        volts,
        title=f"Dynamic Curve Control - Day {day}, Hour {hour}:00",
        band=(0.95, 1.05),
        save_path=os.path.join(OUTDIR, f"dynamic_curve_day{day}_hour_{hour}_snapshot.png"),
        show=True
    )
    
    # Print detailed info for this hour
    print(f"Day {day}, Hour {hour}:00 Snapshot:")
    print(f"  Min voltage: {hour_data['min_voltage']:.4f} pu")
    print(f"  Max voltage: {hour_data['max_voltage']:.4f} pu")
    print(f"  Reward: {hour_data['reward'][0]:.2f}")
    print(f"  PV generation: {np.sum(hour_data['pv_powers'][0]):.1f} kW")
    print(f"  PV reactive: {np.sum(hour_data['pv_powers'][1]):.1f} kVAR")
    
    # Print average curve parameters
    avg_vv = hour_data['voltvar_params'].mean(axis=0)
    avg_vw = hour_data['voltwatt_params'].mean(axis=0)
    print(f"  Avg VV voltages: [{avg_vv[0]:.3f}, {avg_vv[1]:.3f}, {avg_vv[2]:.3f}, {avg_vv[3]:.3f}, {avg_vv[4]:.3f}]")
    print(f"  Avg VW voltages: [{avg_vw[0]:.3f}, {avg_vw[1]:.3f}, {avg_vw[2]:.3f}, {avg_vw[3]:.3f}, {avg_vw[4]:.3f}]")

def save_daily_results(hourly_results, day):
    """Save comprehensive daily results"""
    # Create summary arrays
    hours = list(hourly_results.keys())
    min_voltages = [hourly_results[h]['min_voltage'] for h in hours]
    max_voltages = [hourly_results[h]['max_voltage'] for h in hours]
    pv_generation = [np.sum(hourly_results[h]['pv_powers'][0]) for h in hours]
    pv_reactive = [np.sum(hourly_results[h]['pv_powers'][1]) for h in hours]
    
    # Save to files
    np.save(os.path.join(OUTDIR, f"dynamic_curve_day{day}_min_voltages.npy"), min_voltages)
    np.save(os.path.join(OUTDIR, f"dynamic_curve_day{day}_max_voltages.npy"), max_voltages)
    np.save(os.path.join(OUTDIR, f"dynamic_curve_day{day}_pv_generation.npy"), pv_generation)
    np.save(os.path.join(OUTDIR, f"dynamic_curve_day{day}_pv_reactive.npy"), pv_reactive)
    
    # Save final curve parameters
    final_hour = max(hours)
    np.save(os.path.join(OUTDIR, f"dynamic_curve_day{day}_final_voltvar.npy"), 
            hourly_results[final_hour]['voltvar_params'])
    np.save(os.path.join(OUTDIR, f"dynamic_curve_day{day}_final_voltwatt.npy"), 
            hourly_results[final_hour]['voltwatt_params'])
    
    # Print daily summary
    print("\n" + "="*60)
    print(f"DYNAMIC CURVE DAY {day} SIMULATION SUMMARY")
    print("="*60)
    print(f"Worst minimum voltage: {np.min(min_voltages):.4f} pu")
    print(f"Worst maximum voltage: {np.max(max_voltages):.4f} pu")
    print(f"Total daily reward: {np.sum([hourly_results[h]['reward'][0] for h in hours]):.2f}")
    print(f"Average hourly reward: {np.mean([hourly_results[h]['reward'][0] for h in hours]):.2f}")
    print(f"Best hourly reward: {np.max([hourly_results[h]['reward'][0] for h in hours]):.2f}")
    print(f"Worst hourly reward: {np.min([hourly_results[h]['reward'][0] for h in hours]):.2f}")
    print(f"Total daily PV energy: {np.sum(pv_generation):.1f} kW")
    print(f"Total daily reactive power: {np.sum(pv_reactive):.1f} kVAR")
    
    # Check for voltage violations
    violation_hours = [h for h in hours 
                      if hourly_results[h]['min_voltage'] < 0.95 
                      or hourly_results[h]['max_voltage'] > 1.05]
    
    if violation_hours:
        print(f"Voltage violations at hours: {violation_hours}")
    else:
        print("No voltage violations detected!")

def plot_daily_curves(hourly_results, day):
    """Plot how curves evolve throughout the day"""
    import matplotlib.pyplot as plt
    
    hours = list(hourly_results.keys())
    
    # Plot voltage breakpoint evolution
    plt.figure(figsize=(15, 10))
    
    # Volt-Var voltage breakpoints
    plt.subplot(2, 2, 1)
    for breakpoint_idx in range(5):
        breakpoints = [hourly_results[h]['voltvar_params'].mean(axis=0)[breakpoint_idx] for h in hours]
        plt.plot(hours, breakpoints, label=f'V{breakpoint_idx+1}', marker='o')
    plt.title('Volt-Var Voltage Breakpoints Evolution')
    plt.xlabel('Hour')
    plt.ylabel('Voltage (pu)')
    plt.legend()
    plt.grid(True)
    
    # Volt-Watt voltage breakpoints
    plt.subplot(2, 2, 2)
    for breakpoint_idx in range(5):
        breakpoints = [hourly_results[h]['voltwatt_params'].mean(axis=0)[breakpoint_idx] for h in hours]
        plt.plot(hours, breakpoints, label=f'V{breakpoint_idx+1}', marker='o')
    plt.title('Volt-Watt Voltage Breakpoints Evolution')
    plt.xlabel('Hour')
    plt.ylabel('Voltage (pu)')
    plt.legend()
    plt.grid(True)
    
    # Power generation
    plt.subplot(2, 2, 3)
    pv_generation = [np.sum(hourly_results[h]['pv_powers'][0]) for h in hours]
    pv_reactive = [np.sum(hourly_results[h]['pv_powers'][1]) for h in hours]
    plt.plot(hours, pv_generation, label='Active Power (kW)', marker='s')
    plt.plot(hours, pv_reactive, label='Reactive Power (kVAR)', marker='s')
    plt.title('PV Power Output')
    plt.xlabel('Hour')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    
    # Voltage range
    plt.subplot(2, 2, 4)
    min_voltages = [hourly_results[h]['min_voltage'] for h in hours]
    max_voltages = [hourly_results[h]['max_voltage'] for h in hours]
    plt.fill_between(hours, min_voltages, max_voltages, alpha=0.3, label='Voltage Range')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Voltage Limits')
    plt.axhline(y=1.05, color='r', linestyle='--')
    plt.title('System Voltage Range')
    plt.xlabel('Hour')
    plt.ylabel('Voltage (pu)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"dynamic_curve_day{day}_daily_evolution.png"))
    plt.close()

if __name__ == "__main__":
    # Find the best model
    best_model = os.path.join(OUTDIR, "dynamic_curve_ppo_best.pth")
    final_model = os.path.join(OUTDIR, "dynamic_curve_ppo_final.pth")
    
    model_path = best_model if os.path.exists(best_model) else final_model
    
    if not os.path.exists(model_path):
        print("No trained dynamic curve model found! Train first using curve_final.py")
        exit()
    
    print("Choose simulation mode:")
    print("1. Full 24-hour daily simulation with dynamic curves")
    print("2. Multi-day simulation (average performance)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        target_day = int(input("Which day to simulate? (default 1): ") or "1")
        target_hour = int(input("Which hour to generate snapshot for? (0-23, default 12): ") or "12")
        results = run_daily_simulation(model_path, target_day, target_hour)
        plot_daily_curves(results, target_day)
    
    elif choice == "2":
        num_days = int(input("How many days to simulate? (default 7): ") or "7")
        target_hour = int(input("Which hour to evaluate? (0-23, default 12): ") or "12")
        avg_reward = run_multi_day_simulation(model_path, num_days, target_hour)
        print(f"\nDynamic Curve Optimization - {num_days}-day average reward at hour {target_hour}:00: {avg_reward:.2f}")