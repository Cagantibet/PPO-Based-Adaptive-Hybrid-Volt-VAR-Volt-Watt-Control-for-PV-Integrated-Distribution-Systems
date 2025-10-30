import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from control import run_pv_with_control_simulation, calculate_control_reward
from curve_final import test_dynamic_curve_ppo, DynamicCurveOptimizationEnv

FEEDER_MASTER = r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
LOADSHAPE_DIR = r"C:\Users\pozam\.vscode\cagan\loadshapes"
OUTDIR = r"C:\Users\pozam\.vscode\cagan\results"
os.makedirs(OUTDIR, exist_ok=True)

def plot_comprehensive_comparison(metrics, results):
    """Create comprehensive comparison plots"""
    
    strategies = list(metrics.keys())
    traditional_strategies = [s for s in strategies if metrics[s]['type'] == 'traditional']
    ppo_strategies = [s for s in strategies if metrics[s]['type'] == 'ppo']
    
    colors = ['blue' if m['type'] == 'traditional' else 'red' for m in metrics.values()]
    
    # 1. Reward Comparison
    plt.figure(figsize=(10, 6))
    rewards = [metrics[s]['reward'] for s in strategies]
    
    bars = plt.bar(strategies, rewards, color=colors, alpha=0.7)
    plt.title('Control Strategy Reward Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Reward Score', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    max_reward = max(rewards)
    min_reward = min(rewards)
    plt.ylim(min_reward * 0.95, max_reward * 1.15)
    
    for bar, reward in zip(bars, rewards):
        bar_height = bar.get_height()
        y_position = bar_height + (max_reward * 0.02)
        va_position = 'bottom'
        
        if bar_height > max_reward * 0.8:
            y_position = bar_height * 0.95
            va_position = 'top'
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{reward:.1f}', ha='center', va=va_position, fontsize=10, 
                    color='white', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{reward:.1f}', ha='center', va=va_position, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'reward_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Voltage Range Comparison
    plt.figure(figsize=(10, 6))
    min_voltages = [metrics[s]['min_voltage'] for s in strategies]
    max_voltages = [metrics[s]['max_voltage'] for s in strategies]
    
    for i, strategy in enumerate(strategies):
        plt.plot([min_voltages[i], max_voltages[i]], [i, i], 'o-', 
                linewidth=3, markersize=8, 
                color='blue' if metrics[strategy]['type'] == 'traditional' else 'red',
                label='Traditional' if metrics[strategy]['type'] == 'traditional' and i == 0 else 
                      'PPO' if metrics[strategy]['type'] == 'ppo' and i == len(traditional_strategies) else "")
    
    plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.7, label='Lower Limit')
    plt.axvline(x=1.05, color='red', linestyle='--', alpha=0.7, label='Upper Limit')
    plt.axvspan(0.98, 1.02, alpha=0.2, color='green', label='Ideal Range')
    plt.title('Voltage Range Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Voltage (pu)', fontsize=12)
    plt.yticks(range(len(strategies)), strategies, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'voltage_range_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Violation Percentage
    plt.figure(figsize=(10, 6))
    violations = [metrics[s]['violation_percentage'] for s in strategies]
    
    bars = plt.bar(strategies, violations, color=colors, alpha=0.7)
    plt.title('Voltage Violation Percentage', fontsize=14, fontweight='bold')
    plt.ylabel('% of Measurements Violating', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    max_violation = max(violations)
    plt.ylim(0, max_violation * 1.15)
    
    for bar, violation in zip(bars, violations):
        bar_height = bar.get_height()
        y_position = bar_height + (max_violation * 0.02)
        va_position = 'bottom'
        
        if bar_height > max_violation * 0.8:
            y_position = bar_height * 0.95
            va_position = 'top'
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{violation:.1f}%', ha='center', va=va_position, fontsize=10, 
                    color='white', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{violation:.1f}%', ha='center', va=va_position, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'violation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Voltage Quality Score
    plt.figure(figsize=(10, 6))
    quality_scores = [metrics[s]['quality_score'] for s in strategies]
    
    bars = plt.bar(strategies, quality_scores, color=colors, alpha=0.7)
    plt.title('Voltage Quality Score', fontsize=14, fontweight='bold')
    plt.ylabel('Quality Score', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    max_quality = max(quality_scores)
    plt.ylim(0, max_quality * 1.15)
    
    for bar, score in zip(bars, quality_scores):
        bar_height = bar.get_height()
        y_position = bar_height + (max_quality * 0.02)
        va_position = 'bottom'
        
        if bar_height > max_quality * 0.8:
            y_position = bar_height * 0.95
            va_position = 'top'
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{score:.3f}', ha='center', va=va_position, fontsize=10, 
                    color='white', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{score:.3f}', ha='center', va=va_position, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'quality_score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Voltage Standard Deviation 
    plt.figure(figsize=(10, 6))
    voltage_stds = [metrics[s]['voltage_std'] for s in strategies]
    
    bars = plt.bar(strategies, voltage_stds, color=colors, alpha=0.7)
    plt.title('Voltage Standard Deviation', fontsize=14, fontweight='bold')
    plt.ylabel('Std Dev (pu)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    max_std = max(voltage_stds)
    plt.ylim(0, max_std * 1.15)
    
    for bar, std in zip(bars, voltage_stds):
        bar_height = bar.get_height()
        y_position = bar_height + (max_std * 0.02)
        va_position = 'bottom'
        
        if bar_height > max_std * 0.8:
            y_position = bar_height * 0.95
            va_position = 'top'
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{std:.4f}', ha='center', va=va_position, fontsize=10, 
                    color='white', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, y_position, 
                    f'{std:.4f}', ha='center', va=va_position, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'voltage_std_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def comprehensive_performance_comparison(day=1, num_trials=1):
    """Compare PPO vs traditional control strategies using same metrics - NO PLOTTING HERE"""
    
    print("=== PERFORMANCE COMPARISON: PPO vs TRADITIONAL CONTROLS ===")
    print(f"Running {num_trials} trial(s) for each strategy...")
    
    # Initialize results storage for multiple trials
    trial_results = {
        "Volt-VAR": [],
        "Volt-Watt": [],
        "Combined": [],
        "PPO Dynamic": []
    }
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        results = {}
        
        # Test traditional controls
        print("1. Testing Traditional Volt-VAR Control...")
        volts_vv, reward_vv, _, _= run_pv_with_control_simulation("voltvar")
        results["Volt-VAR"] = {
            'reward': reward_vv,
            'voltages': collect_all_voltages(volts_vv),
            'type': 'traditional'
        }
        
        print("2. Testing Traditional Volt-Watt Control...")
        volts_vw, reward_vw, _, _ = run_pv_with_control_simulation("voltwatt")
        results["Volt-Watt"] = {
            'reward': reward_vw,
            'voltages': collect_all_voltages(volts_vw),
            'type': 'traditional'
        }
        
        print("3. Testing Traditional Combined Control...")
        volts_comb, reward_comb, _, _ = run_pv_with_control_simulation("combined")
        results["Combined"] = {
            'reward': reward_comb,
            'voltages': collect_all_voltages(volts_comb),
            'type': 'traditional'
        }
        
        # Test PPO dynamic curves
        print("4. Testing PPO Dynamic Curve Control...")
        ppo_result = test_dynamic_curve_ppo(day=day, 
                                   model_path=os.path.join(OUTDIR, "dynamic_curve_ppo_best.pth"))
        
        # Handle the return value from test_dynamic_curve_ppo
        if isinstance(ppo_result, tuple):
            # If it returns a tuple, extract the environment and other values
            if len(ppo_result) >= 1:
                ppo_env = ppo_result[0]
                # Try to get voltages
                if hasattr(ppo_env, 'get_all_phase_voltages'):
                    ppo_voltages = ppo_env.get_all_phase_voltages()
                else:
                    # Fallback: try to extract from the tuple
                    ppo_voltages = extract_voltages_from_ppo_result(ppo_result)
                # Try to get powers
                if hasattr(ppo_env, 'get_pv_powers'):
                    p_actual, q_actual = ppo_env.get_pv_powers()
            else:
                # If can't extract properly, use fallback values
                print("Warning: Could not extract PPO results properly, using fallback")
                ppo_voltages = np.array([1.0])  # Default fallback
        else:
            # If it returns a single object (environment)
            ppo_env = ppo_result
            ppo_voltages = ppo_env.get_all_phase_voltages()
            p_actual, q_actual = ppo_env.get_pv_powers()
        
        # Use the SAME reward calculation as traditional controls for consistency
        ppo_reward = calculate_control_reward(ppo_voltages, p_actual, q_actual)
        
        results["PPO Dynamic"] = {
            'reward': ppo_reward,
            'voltages': ppo_voltages,
            'type': 'ppo'
        }
        
        # Store trial results
        for strategy in results:
            trial_results[strategy].append(results[strategy])
    
    return trial_results

def extract_voltages_from_ppo_result(ppo_result):
    """Extract voltages from PPO result tuple"""
    # Try different possible formats of the return tuple
    for item in ppo_result:
        if isinstance(item, np.ndarray) and item.size > 0:
            return item
        elif hasattr(item, 'get_all_phase_voltages'):
            return item.get_all_phase_voltages()
    # Fallback
    return np.array([1.0])

def extract_powers_from_ppo_result(ppo_result):
    """Extract powers from PPO result tuple"""
    # Try to find power values in the tuple
    for item in ppo_result:
        if isinstance(item, tuple) and len(item) == 2:
            return item
        elif hasattr(item, 'get_pv_powers'):
            return item.get_pv_powers()
    # Fallback
    return 0, 0

def collect_all_voltages(volts_data):
    """Extract all phase voltages from voltage collection data"""
    all_voltages = []
    for bus_data in volts_data:
        phase_voltages = bus_data[2]
        for v in phase_voltages:
            if not np.isnan(v):
                all_voltages.append(v)
    return np.array(all_voltages)

def analyze_performance_metrics(results):
    """Calculate detailed performance metrics"""
    metrics = {}
    
    for strategy, data in results.items():
        voltages = data['voltages']
        
        # Voltage compliance
        violation_count = np.sum(voltages < 0.95) + np.sum(voltages > 1.05)
        violation_percentage = (violation_count / len(voltages)) * 100
        
        # Voltage statistics
        min_voltage = np.min(voltages)
        max_voltage = np.max(voltages)
        avg_voltage = np.mean(voltages)
        voltage_std = np.std(voltages)
        
        within_ideal = np.sum((voltages >= 0.98) & (voltages <= 1.02))
        within_acceptable = np.sum((voltages >= 0.95) & (voltages <= 1.05))
        
        quality_score = (within_ideal * 2 + within_acceptable * 0.5) / len(voltages)
        
        # Power statistics
        total_power = data.get('total_power', 0)
        total_reactive = data.get('total_reactive', 0)
        
        metrics[strategy] = {
            'reward': data['reward'],
            'min_voltage': min_voltage,
            'max_voltage': max_voltage,
            'avg_voltage': avg_voltage,
            'voltage_std': voltage_std,
            'violation_count': violation_count,
            'violation_percentage': violation_percentage,
            'quality_score': quality_score,
            'total_power': total_power,
            'total_reactive': total_reactive,
            'type': data['type']
        }
    
    return metrics

def aggregate_trial_results(trial_results):
    """Aggregate results from multiple trials"""
    aggregated_results = {}
    
    for strategy, trials in trial_results.items():
        if not trials:  # Skip if no trials
            continue
            
        # Combine all voltages from all trials
        all_voltages = np.concatenate([trial['voltages'] for trial in trials])
        
        # Calculate average reward across trials
        avg_reward = np.mean([trial['reward'] for trial in trials])
        
        aggregated_results[strategy] = {
            'reward': avg_reward,
            'voltages': all_voltages,
            'type': trials[0]['type']  # All trials have same type
        }
    
    return aggregated_results

def main():
    """Run comprehensive performance comparison"""
    day = 1  # Use same day for all tests
    
    # Get number of trials from user
    try:
        num_trials = int(input("Enter number of trials to run for each strategy (default: 1): ") or "1")
    except ValueError:
        num_trials = 1
        print("Invalid input. Using default: 1 trial")
    
    print(f"Starting comprehensive performance comparison with {num_trials} trial(s)...")
    print("Note: All strategies evaluated using IDENTICAL reward function")
    print("NOTE: Internal plotting disabled during trials for cleaner output")
    
    # Set non-interactive backend during trials to prevent plotting
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    
    # Run all simulations with multiple trials
    trial_results = comprehensive_performance_comparison(day, num_trials)
    
    # Aggregate results across trials
    aggregated_results = aggregate_trial_results(trial_results)
    
    # Analyze metrics
    metrics = analyze_performance_metrics(aggregated_results)
    
    # Print summary table first
    print("\n=== PERFORMANCE SUMMARY (Aggregated over {} trials) ===".format(num_trials))
    print(f"{'Strategy':<15} {'Reward':<8} {'Min V':<8} {'Max V':<8} {'Avg V':<8} {'Std Dev':<8} {'Violations':<12} {'Quality':<8}")
    print("-" * 85)
    
    for strategy in metrics.keys():
        m = metrics[strategy]
        print(f"{strategy:<15} {m['reward']:<8.1f} {m['min_voltage']:<8.3f} {m['max_voltage']:<8.3f} "
              f"{m['avg_voltage']:<8.3f} {m['voltage_std']:<8.4f} {m['violation_count']:<4} ({m['violation_percentage']:<5.1f}%) "
              f"{m['quality_score']:<8.3f}")

    # Determine best performer
    best_strategy = max(metrics.items(), key=lambda x: x[1]['reward'])
    print(f"\n🎯 BEST PERFORMING STRATEGY: {best_strategy[0]} (Reward: {best_strategy[1]['reward']:.1f})")
    
    # Compare PPO vs best traditional
    traditional_rewards = [metrics[s]['reward'] for s in metrics if metrics[s]['type'] == 'traditional']
    if traditional_rewards:  # Check if there are traditional strategies
        best_traditional = max(traditional_rewards)
        ppo_reward = metrics['PPO Dynamic']['reward']
        
        improvement = ((ppo_reward - best_traditional) / abs(best_traditional)) * 100
        print(f"📈 PPO Improvement over Best Traditional: {improvement:+.1f}%")
    
    # Create comprehensive plots after all trials are complete
    print(f"\n📊 Generating comprehensive comparison plots (aggregated over {num_trials} trials)...")
    
    # Switch to interactive backend for final plots
    matplotlib.use(original_backend)
    
    # Generate final comparison plots
    plot_comprehensive_comparison(metrics, aggregated_results)
    print("Comparison plots generated and saved to output directory.")

if __name__ == "__main__":
    main()