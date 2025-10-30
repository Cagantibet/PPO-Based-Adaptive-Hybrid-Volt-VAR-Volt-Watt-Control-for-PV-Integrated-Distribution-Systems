import os
import numpy as np
import opendssdirect as dss
from dss_utils import (
    compile_feeder, set_all_reg_taps_to, pv_debug_summary,
    set_time_to_hour, solve, collect_bus_voltages, plot_bus_profile_general,
    add_pvs_34, add_pvs_15, add_pvs_5, inject_daily_loadshape, create_all_irradiance_shapes, inject_irradiance_loadshapes
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Constants
FEEDER_MASTER = r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss"
LOADSHAPE_DIR = r"C:\Users\pozam\.vscode\cagan\loadshapes"
OUTDIR = r"C:\Users\pozam\.vscode\cagan\results"
os.makedirs(OUTDIR, exist_ok=True)

class DynamicCurveOptimizationEnv:
    def __init__(self, day=1):
        self.day = day
        self.num_pv_systems = 15
        self.pv_max_p = 200.0  # kW per PV
        self.pv_max_q = 88.0   # kVAR per PV
        
        # Store PV information
        self.pv_phases = []
        self.pv_bus_names = []
        self.pv_names = []
        
        self.compile_system()
        
        # Dynamic curve parameters: 
        # For Volt-Var: [V1, V2, V3, V4, V5, Q1, Q2, Q3, Q4, Q5]
        # For Volt-Watt: [V1, V2, V3, V4, V5, P1, P2, P3, P4, P5]
        self.voltvar_params = self.initialize_voltvar_curves()
        self.voltwatt_params = self.initialize_voltwatt_curves()
        
        # Get ALL phase voltages for observation space
        self.num_voltage_points = len(self.get_all_phase_voltages())
        
        # Action space: 10 volt-var params + 10 volt-watt params = 20 parameters per PV
        self.action_dim = self.num_pv_systems * 20
        
        print(f"Dynamic Curve Environment: {self.num_voltage_points} phase voltages, {self.action_dim} curve parameters")
    
    def initialize_voltvar_curves(self):

        curves = []
        for i in range(self.num_pv_systems):
            # Standard IEEE 1547: [V1, V2, V3, V4, V5, Q1, Q2, Q3, Q4, Q5]
            curves.append([0.95, 0.98, 1.0, 1.02, 1.05,   # Voltage breakpoints
                          0.88, 0.44, 0.0, -0.44, -0.88])  # Q percentages
        return np.array(curves)
    
    def initialize_voltwatt_curves(self):
        curves = []
        for i in range(self.num_pv_systems):
            # Standard IEEE 1547: [V1, V2, V3, V4, V5, P1, P2, P3, P4, P5]
            curves.append([0.95, 0.98, 1.0, 1.02, 1.05,   # Voltage breakpoints
                          1.0, 1.0, 0.8, 0.4, 0.0])        # P percentages
        return np.array(curves)
    
    def compile_system(self):
        """Compile OpenDSS system"""
        compile_feeder(FEEDER_MASTER)
        inject_daily_loadshape(day=self.day, save_dir=LOADSHAPE_DIR)
        set_all_reg_taps_to(4)
        create_all_irradiance_shapes(day=self.day, save_dir=LOADSHAPE_DIR)
        inject_irradiance_loadshapes(day=self.day, save_dir=LOADSHAPE_DIR)
        add_pvs_15(day=self.day, save_dir=LOADSHAPE_DIR)
        set_time_to_hour(13)
        solve()
        
        # Store PV information
        self.pv_names = []
        self.pv_bus_names = []
        self.pv_phases = []
        
        dss.PVsystems.First()
        while True:
            name = dss.PVsystems.Name()
            if not name:
                break
            self.pv_names.append(name)
            
            # Get bus name and phase
            dss.Circuit.SetActiveElement(f"PVSystem.{name}")
            bus_info = dss.CktElement.BusNames()[0]
            bus_parts = bus_info.split('.')
            bus_name = bus_parts[0]
            phase = int(bus_parts[1]) if len(bus_parts) > 1 else 1
            
            self.pv_bus_names.append(bus_name)
            self.pv_phases.append(phase)
            
            if not dss.PVsystems.Next():
                break
    
    def get_pv_bus_voltages(self, bus_name):
        """Get ALL phase voltages for a specific bus"""
        dss.Circuit.SetActiveBus(bus_name)
        voltages = dss.Bus.puVmagAngle()
        num_phases = len(voltages) // 2
        phase_voltages = []
        for i in range(num_phases):                
            phase_voltages.append(voltages[i*2])
            
        return phase_voltages

    def get_all_phase_voltages(self):
        """Get ALL phase voltages for ALL buses (for observation)"""
        all_voltages = []
        volts = collect_bus_voltages(per_phase=True, sort_by="index")
        for bus_data in volts:
            # Each bus_data has [bus_idx, bus_name, [V_phaseA, V_phaseB, V_phaseC]]
            phase_voltages = bus_data[2]
            for v in phase_voltages:
                if not np.isnan(v):
                    all_voltages.append(v)
        return np.array(all_voltages)
    
    def get_critical_voltages(self, pv_idx):
        """Get relevant voltages for PV control decision"""
        bus_name = self.pv_bus_names[pv_idx]
        phase = self.pv_phases[pv_idx]
        
        # Get all phase voltages at PV bus
        bus_voltages = self.get_pv_bus_voltages(bus_name)
        
        if not bus_voltages:
            return 1.0, 1.0, 1.0
        
        # Use the PV's specific phase voltage for primary control
        pv_phase_voltage = bus_voltages[phase-1] if phase <= len(bus_voltages) else bus_voltages[0]
        
        # Also consider max voltage at this bus for conservative control
        max_bus_voltage = max(bus_voltages)
        
        # And consider min voltage for under-voltage support
        min_bus_voltage = min(bus_voltages)
        
        return pv_phase_voltage, max_bus_voltage, min_bus_voltage
    
    def get_pv_powers(self):
        """Get ACTUAL P and Q generation from PV systems using circuit measurements"""
        p_values, q_values = [], []
        
        i = dss.PVsystems.First()
        while i > 0:
            name = dss.PVsystems.Name()
            
            # Get directly from PVSystem object
            dss.PVsystems.Name(name)
            p_kw = dss.PVsystems.Pmpp()
            q_kvar = dss.PVsystems.kvar()

            p_values.append(p_kw)
            q_values.append(q_kvar)
            
            i = dss.PVsystems.Next()
        
        return np.array(p_values), np.array(q_values)

    def apply_curve_control(self):
        """Apply dynamic volt-var and volt-watt curve control using LOCAL voltages only"""
        for i, pv_name in enumerate(self.pv_names):
            # Get curve parameters for this PV
            vv_params = self.voltvar_params[i]
            vw_params = self.voltwatt_params[i]
            
            # Extract voltages and power percentages
            vv_voltages = vv_params[:5]
            vv_q_percent = vv_params[5:]
            vw_voltages = vw_params[:5]
            vw_p_percent = vw_params[5:]
            
            # Get the PV's LOCAL phase voltage only
            pv_voltage, _, _ = self.get_critical_voltages(i)
            
            # Use only the PV's local voltage for control decisions
            control_voltage = pv_voltage
            
            # Get available solar
            try:
                dss.PVsystems.Name(pv_name)
                irrad = dss.PVsystems.IrradianceNow()
                available_power = min(irrad * self.pv_max_p, self.pv_max_p)
            except:
                available_power = self.pv_max_p
            
            # VOLT-VAR CURVE: Piecewise linear interpolation using local voltage
            if control_voltage <= vv_voltages[0]:
                q_percent = vv_q_percent[0]
            elif control_voltage >= vv_voltages[4]:
                q_percent = vv_q_percent[4]
            else:
                # Find which segment we're in
                for seg in range(4):
                    if vv_voltages[seg] <= control_voltage <= vv_voltages[seg+1]:
                        # Linear interpolation
                        v_low, v_high = vv_voltages[seg], vv_voltages[seg+1]
                        q_low, q_high = vv_q_percent[seg], vv_q_percent[seg+1]
                        q_percent = q_low + (q_high - q_low) * (control_voltage - v_low) / (v_high - v_low)
                        break
                else:
                    q_percent = 0.0
            
            # VOLT-WATT CURVE: Piecewise linear interpolation using LOCAL voltage only
            watt_control_voltage = pv_voltage  # Use local voltage only
            
            if watt_control_voltage <= vw_voltages[0]:
                p_percent = vw_p_percent[0]
            elif watt_control_voltage >= vw_voltages[4]:
                p_percent = vw_p_percent[4]
            else:
                # Find which segment we're in
                for seg in range(4):
                    if vw_voltages[seg] <= watt_control_voltage <= vw_voltages[seg+1]:
                        # Linear interpolation
                        v_low, v_high = vw_voltages[seg], vw_voltages[seg+1]
                        p_low, p_high = vw_p_percent[seg], vw_p_percent[seg+1]
                        p_percent = p_low + (p_high - p_low) * (watt_control_voltage - v_low) / (v_high - v_low)
                        break
                else:
                    p_percent = 1.0
            
            # Calculate actual power setpoints
            q_setpoint = q_percent * self.pv_max_q  # kVAR
            
            # Apply inverter constraints
            p_percent = max(0.0, min(p_percent, 1.0))  # Ensure percentage is between 0 and 1
            
            p_setpoint = p_percent * available_power  # kW
            
            # Apply inverter capacity constraints
            s_available = np.sqrt(max(0, self.pv_max_p**2 - p_setpoint**2))
            q_setpoint = np.clip(q_setpoint, -s_available, s_available)
            
            # Set PV outputs - directly set Pmpp to the controlled power value
            dss.Text.Command(f"PVSystem.{pv_name}.Pmpp={p_setpoint:.3f}")
            dss.Text.Command(f"PVSystem.{pv_name}.kVAR={q_setpoint:.3f}")
    
    def calculate_reward(self, voltages, p_actual, q_actual):
        """Calculate reward based on ALL phase voltage quality and power utilization"""
        reward = 0
        
        # 1. Voltage compliance for ALL phases
        voltage_penalty = 0
        ideal_bonus = 0
        violation_count = 0
        
        for v in voltages:
            if v < 0.95:
                voltage_penalty += (0.95 - v) * 100
                violation_count += 1
            elif v > 1.05:
                voltage_penalty += (v - 1.05) * 150
                violation_count += 1
            elif 0.98 <= v <= 1.02:
                ideal_bonus += 2
            elif 0.95 <= v <= 1.05:
                ideal_bonus += 0.5
        
        reward += ideal_bonus - voltage_penalty
        
        # Extra penalty for having any violations
        if violation_count > 0:
            reward -= violation_count * 5
        
        # 2. Power utilization
        # p_actual is NEGATIVE for generation, so we use abs() for utilization calculation
        total_available = self.num_pv_systems * self.pv_max_p
        utilization = np.sum(np.abs(p_actual)) / total_available if total_available > 0 else 0
        
        # Only reward high utilization if ALL voltages are good
        max_v = np.max(voltages)
        min_v = np.min(voltages)
        
        if max_v <= 1.05 and min_v >= 0.95 and violation_count == 0:
            power_reward = utilization * 100
        else:
            power_reward = utilization * 20
            
        reward += power_reward
        
        # 3. Voltage balance bonus
        voltage_std = np.std(voltages)
        if voltage_std < 0.01:
            reward += 3
        elif voltage_std < 0.02:
            reward += 1.5
        
        # 4. Reactive power efficiency
        q_penalty = np.sum(np.abs(q_actual)) / (self.num_pv_systems * self.pv_max_q) * 2.5
        reward -= q_penalty
        
        return reward
    
    def denormalize_power_percentages(self, powers_norm, reactive=True):
        """Denormalize power percentages from [-1,1] to standard ranges"""
        if reactive:
            # Reactive power:
            return powers_norm * 0.88
        else:
            # Active power:
            p_denorm = (powers_norm + 1) / 2
            
            # For volt-watt, enforce that P1 >= P2 >= P3 >= P4 >= P5
            if not reactive:
                p_denorm = np.sort(p_denorm)[::-1]
            
            return p_denorm
    
    def update_curves_from_action(self, action):
        """Update volt-var and volt-watt curves from PPO action with standard constraints"""
        action_reshaped = action.reshape(self.num_pv_systems, 20)
        
        for i in range(self.num_pv_systems):
            pv_action = action_reshaped[i]
            
            # First 10: volt-var parameters [V1-V5, Q1-Q5]
            vv_action = pv_action[:10]
            # Last 10: volt-watt parameters [V1-V5, P1-P5]
            vw_action = pv_action[10:]
            
            # Denormalize volt-var parameters
            vv_voltages = self.denormalize_voltages(vv_action[:5], voltvar=True)
            vv_q = self.denormalize_power_percentages(vv_action[5:], reactive=True)
            
            # Denormalize volt-watt parameters  
            vw_voltages = self.denormalize_voltages(vw_action[:5], voltvar=False)
            vw_p = self.denormalize_power_percentages(vw_action[5:], reactive=False)
            
            # Ensure monotonic voltages and standard curve shapes
            vv_voltages = np.sort(vv_voltages)
            vw_voltages = np.sort(vw_voltages)
            
            vv_q_ordered = np.sort(vv_q)[::-1]    
            vv_q_ordered = np.array([vv_q_ordered[0], vv_q_ordered[1], vv_q_ordered[2], vv_q_ordered[3], vv_q_ordered[4]])
                        
            # For volt-watt: ensure P decreases as voltage increases
            vw_p_ordered = np.sort(vw_p)[::-1]
            
            self.voltvar_params[i] = np.concatenate([vv_voltages, vv_q_ordered])
            self.voltwatt_params[i] = np.concatenate([vw_voltages, vw_p_ordered])
    
    def denormalize_voltages(self, voltages_norm, voltvar=True):
        """Denormalize voltage breakpoints from [-1,1] to actual ranges"""
        if voltvar:
            # Volt-var voltage range: 0.95 to 1.05 pu
            min_v, max_v = 0.95, 1.05
        else:
            # Volt-watt voltage range: 0.95 to 1.05 pu
            min_v, max_v = 0.95, 1.05
        
        return min_v + (voltages_norm + 1) / 2 * (max_v - min_v)
    
    def step(self, action):
        """PPO step: action contains curve parameters"""
        # Denormalize and update curve parameters
        self.update_curves_from_action(action)
        
        # Apply curve-based control
        self.apply_curve_control()
        
        # Solve power flow
        solve()
        
        # Get new state (ALL phase voltages) and calculate reward
        state = self.get_all_phase_voltages()
        p_actual, q_actual = self.get_pv_powers()
        reward = self.calculate_reward(state, p_actual, q_actual)
        
        # Check termination (good voltage profile across ALL phases)
        done = np.all((state >= 0.95) & (state <= 1.05))
        
        return state, reward, done, {}
    
    def reset(self):
        """Reset environment with standard curves"""
        self.compile_system()
        self.voltvar_params = self.initialize_voltvar_curves()
        self.voltwatt_params = self.initialize_voltwatt_curves()

        
        # Apply initial curves
        self.apply_curve_control()
        
        solve()
        
        return self.get_all_phase_voltages()

class DynamicCurvePPO:
    def __init__(self, state_dim, action_dim, lr=3e-5, gamma=0.99, eps_clip=0.1, k_epochs=8):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # ActorCritic architecture
        self.policy = CurvePPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = CurvePPOActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):
        # Convert to tensors
        states = torch.FloatTensor(np.array(memory.states))
        actions = torch.FloatTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs))
        
        # Compute returns
        returns = []
        discounted_return = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Optimize policy
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(states, actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def evaluate(self, states, actions):
        dist, values = self.policy(states)
        logprobs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy()
        return logprobs, values, entropy
    
    def get_action(self, state, memory):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            dist, value = self.policy_old(state_tensor)
            
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(-1)
            
            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.state_values.append(value)
            
            return action.numpy().flatten()
    
    def save(self, filename):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())

class CurvePPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CurvePPOActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head - outputs curve parameters
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Outputs in [-1, 1]
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, x):
        shared_features = self.shared_net(x)
        
        value = self.critic(shared_features)
        mu = self.actor(shared_features)
        
        # Clamp mu and ensure reasonable std
        mu = torch.clamp(mu, -0.95, 0.95)
        std = torch.exp(self.log_std).clamp(min=1e-4, max=0.1)
        
        dist = Normal(mu, std)
        return dist, value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.state_values.clear()

def train_dynamic_curve_ppo(day=1, episodes=1000):
    """Train PPO to optimize dynamic volt-var/volt-watt curves"""
    env = DynamicCurveOptimizationEnv(day=day)
    state_dim = env.num_voltage_points
    action_dim = env.action_dim
    
    print(f"Training Dynamic Curve PPO: state_dim={state_dim}, action_dim={action_dim}")
    print("PPO will optimize 5-point volt-var and volt-watt curves")
    
    ppo = DynamicCurvePPO(state_dim, action_dim, lr=3e-5, gamma=0.995, eps_clip=0.15, k_epochs=8)
    
    # Training parameters
    max_episodes = episodes
    max_timesteps = 100
    update_interval = 2000
    
    memory = PPOMemory()
    running_reward = 0
    best_reward = -float('inf')
    time_step = 0
    
    reward_history = []
    
    print("Starting Dynamic Curve PPO training...")
    
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        final_reward = 0
        
        for t in range(max_timesteps):
            time_step += 1
            
            # Get curve parameters from PPO
            action = ppo.get_action(state, memory)
            
            # Step with curve parameters
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            final_reward = reward  # Track final single-step reward
            
            # Update policy
            if time_step % update_interval == 0:
                ppo.update(memory)
                memory.clear()
                time_step = 0
            
            if done:
                break
        
        # Update statistics
        running_reward = 0.95 * running_reward + 0.05 * final_reward
        reward_history.append(final_reward)
        
        # Save best model
        if final_reward > best_reward:
            best_reward = final_reward
            ppo.save(os.path.join(OUTDIR, "dynamic_curve_ppo_best.pth"))
            
            # Save optimized curves
            np.save(os.path.join(OUTDIR, "optimized_voltvar_curves.npy"), env.voltvar_params)
            np.save(os.path.join(OUTDIR, "optimized_voltwatt_curves.npy"), env.voltwatt_params)
            
            print(f"Episode {episode}: New best reward {best_reward:.2f}")
        
        # Log progress
        if episode % 10 == 0:
            # Print average curve parameters
            avg_vv = env.voltvar_params.mean(axis=0)
            avg_vw = env.voltwatt_params.mean(axis=0)

            print(f"Episode {episode:3d} | Final Single-step Reward: {final_reward:7.2f} | "
                  f"VV Voltages: [{avg_vv[0]:.3f},{avg_vv[1]:.3f},{avg_vv[2]:.3f},{avg_vv[3]:.3f},{avg_vv[4]:.3f}] | "
                  f"VW Voltages: [{avg_vw[0]:.3f},{avg_vw[1]:.3f},{avg_vw[2]:.3f},{avg_vw[3]:.3f},{avg_vw[4]:.3f}]")

    # Plot training results and optimized curves
    plot_optimized_curves(env, reward_history)
    
    ppo.save(os.path.join(OUTDIR, "dynamic_curve_ppo_final.pth"))
    print(f"Dynamic Curve PPO training completed! Best reward: {best_reward:.2f}")
    
    return ppo, env

def plot_optimized_curves(env, reward_history):
    """Plot the optimized volt-var and volt-watt curves"""
    plt.figure(figsize=(15, 10))
    
    # Plot training rewards
    plt.subplot(2, 3, 1)
    plt.plot(reward_history)
    plt.title('Training Rewards')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot optimized volt-var curves
    plt.subplot(2, 3, 2)
    voltages = np.linspace(0.85, 1.15, 200)

    for i in range(min(5, env.num_pv_systems)):  # Plot first 5 PVs
        vv_params = env.voltvar_params[i]
        vv_v = vv_params[:5]
        vv_q = vv_params[5:]
        
        # Get the actual bus name for this PV
        pv_bus = env.pv_bus_names[i]
        pv_phase = env.pv_phases[i]
        
        q_values = []
        for v in voltages:
            if v <= vv_v[0]:
                q = vv_q[0]
            elif v >= vv_v[4]:
                q = vv_q[4]
            else:
                for seg in range(4):
                    if vv_v[seg] <= v <= vv_v[seg+1]:
                        q_low, q_high = vv_q[seg], vv_q[seg+1]
                        v_low, v_high = vv_v[seg], vv_v[seg+1]
                        q = q_low + (q_high - q_low) * (v - v_low) / (v_high - v_low)
                        break
                else:
                    q = 0.0
            q_values.append(q * 88)  # Convert to kVAR
        
        # Use bus name and phase in the label
        plt.plot(voltages, q_values, label=f'Bus {pv_bus}.{pv_phase}')

    plt.axvspan(0.95, 1.05, alpha=0.2, color='green', label='Safe Range')
    plt.title('Optimized Volt-Var Curves')
    plt.xlabel('Voltage (pu)')
    plt.ylabel('Reactive Power (kVAR)')
    plt.legend()
    plt.grid(True)

    # Plot optimized volt-watt curves
    plt.subplot(2, 3, 3)

    for i in range(min(5, env.num_pv_systems)):  # Plot first 5 PVs
        vw_params = env.voltwatt_params[i]
        vw_v = vw_params[:5]
        vw_p = vw_params[5:]
        
        # Get the actual bus name for this PV
        pv_bus = env.pv_bus_names[i]
        pv_phase = env.pv_phases[i]
        
        p_values = []
        for v in voltages:
            if v <= vw_v[0]:
                p = vw_p[0]
            elif v >= vw_v[4]:
                p = vw_p[4]
            else:
                for seg in range(4):
                    if vw_v[seg] <= v <= vw_v[seg+1]:
                        p_low, p_high = vw_p[seg], vw_p[seg+1]
                        v_low, v_high = vw_v[seg], vw_v[seg+1]
                        p = p_low + (p_high - p_low) * (v - v_low) / (v_high - v_low)
                        break
                else:
                    p = 1.0
            p_values.append(p * 100)  # Convert to percentage
        
        # Use bus name and phase in the label
        plt.plot(voltages, p_values, label=f'Bus {pv_bus}.{pv_phase}')

    plt.axvspan(0.95, 1.05, alpha=0.2, color='green', label='Safe Range')
    plt.title('Optimized Volt-Watt Curves')
    plt.xlabel('Voltage (pu)')
    plt.ylabel('Active Power (%)')
    plt.legend()
    plt.grid(True)

    # Plot voltage breakpoint distributions
    plt.subplot(2, 3, 4)
    vv_voltages = env.voltvar_params[:, :5]
    plt.boxplot(vv_voltages)
    plt.title('Volt-Var Voltage Breakpoints')
    plt.ylabel('Voltage (pu)')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    vw_voltages = env.voltwatt_params[:, :5]
    plt.boxplot(vw_voltages)
    plt.title('Volt-Watt Voltage Breakpoints')
    plt.ylabel('Voltage (pu)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "dynamic_curve_optimization.png"))
    plt.close()

def test_dynamic_curve_ppo(day=1, model_path=None):
    """Test the optimized dynamic curves: Return single-step reward"""
    env = DynamicCurveOptimizationEnv(day=day)
    state_dim = env.num_voltage_points
    action_dim = env.action_dim
    
    ppo = DynamicCurvePPO(state_dim, action_dim)
    
    if model_path and os.path.exists(model_path):
        ppo.load(model_path)
        print("Loaded trained dynamic curve PPO model")
        
        # Load optimized curves
        vv_path = os.path.join(OUTDIR, "optimized_voltvar_curves.npy")
        vw_path = os.path.join(OUTDIR, "optimized_voltwatt_curves.npy")
        
        if os.path.exists(vv_path) and os.path.exists(vw_path):
            env.voltvar_params = np.load(vv_path)
            env.voltwatt_params = np.load(vw_path)
            print("Loaded optimized dynamic curves")
    else:
        print("Using standard curves")
    
    # Test performance
    state = env.reset()
    final_reward = 0
    convergence_step = 0
    
    for step in range(50):
        if model_path:
            action = ppo.get_action(state, PPOMemory())
        state, reward, done, _ = env.step(action)
        final_reward = reward  # Track the most recent reward
        
        print(f"Step {step+1}: Reward = {reward:.1f}, Voltage range = {np.min(state):.3f}-{np.max(state):.3f} pu")
        
        if done:
            convergence_step = step + 1
            print(f"✓ Converged after {convergence_step} steps")
            break
        elif step == 49:
            convergence_step = 50
            print(f"Reached maximum steps ({convergence_step})")
    
    # Final analysis
    voltages = env.get_all_phase_voltages()
    p_actual, q_actual = env.get_pv_powers()
    
    print(f"\n=== DYNAMIC CURVE PPO TEST RESULTS ===")
    print(f"Convergence Steps: {convergence_step}")
    print(f"Final Single-Step Reward: {final_reward:.2f}") 
    print(f"Voltage Range: {np.min(voltages):.3f} - {np.max(voltages):.3f} pu")
    print(f"Voltage Violations: {np.sum(voltages < 0.95) + np.sum(voltages > 1.05)}")
    print(f"Total PV Power: {np.sum(p_actual)} kW")
    print(f"Total Reactive Power: {np.sum(q_actual)} kVAR")

    # Print optimized curve parameters
    print(f"\n=== OPTIMIZED DYNAMIC CURVE PARAMETERS ===")
    avg_vv = env.voltvar_params.mean(axis=0)
    avg_vw = env.voltwatt_params.mean(axis=0)
    
    print("Volt-Var Curve:")
    print(f"  Voltages: [{avg_vv[0]:.3f}, {avg_vv[1]:.3f}, {avg_vv[2]:.3f}, {avg_vv[3]:.3f}, {avg_vv[4]:.3f}]")
    print(f"  Q %:      [{avg_vv[5]:.3f}, {avg_vv[6]:.3f}, {avg_vv[7]:.3f}, {avg_vv[8]:.3f}, {avg_vv[9]:.3f}]")
    
    print("Volt-Watt Curve:")
    print(f"  Voltages: [{avg_vw[0]:.3f}, {avg_vw[1]:.3f}, {avg_vw[2]:.3f}, {avg_vw[3]:.3f}, {avg_vw[4]:.3f}]")
    print(f"  P %:      [{avg_vw[5]:.3f}, {avg_vw[6]:.3f}, {avg_vw[7]:.3f}, {avg_vw[8]:.3f}, {avg_vw[9]:.3f}]")
    
    # Plot voltage profile
    volts = collect_bus_voltages(per_phase=True, sort_by="index")
    plot_bus_profile_general(
        volts,
        title=f"Dynamic Curve PPO Control - Reward: {final_reward:.1f} - "
              f"Voltage: {np.min(voltages):.3f}-{np.max(voltages):.3f} pu",
        band=(0.95, 1.05),
        per_phase=True,
        save_path=os.path.join(OUTDIR, "dynamic_curve_ppo_voltage_profile.png"),
        show=True
    )
    
    return env, final_reward

if __name__ == "__main__":
    day = int(input("Enter day number (default 1): ") or "1")
    episodes = int(input("Enter number of training episodes (default 1000): ") or "1000")
    print("Training PPO to optimize dynamic 5-point volt-var/volt-watt curves...")
    ppo_agent, environment = train_dynamic_curve_ppo(day=day, episodes=episodes)
    
    print("\nTesting optimized dynamic curves...")

    test_env = test_dynamic_curve_ppo(day=day, model_path=os.path.join(OUTDIR, "dynamic_curve_ppo_best.pth"))
