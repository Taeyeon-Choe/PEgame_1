"""
PEgame Environment with GA STM Integration using GASTMPropagator
Final implementation using the provided GASTMPropagator class
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal
from environment.pursuit_evasion_env import PursuitEvasionEnv
from orbital_mechanics.GASTMPropagator import GASTMPropagator
from scipy.integrate import solve_ivp
from orbital_mechanics.dynamics import relative_dynamics_evader_centered


class PursuitEvasionEnvGASTM(PursuitEvasionEnv):
    """
    Extended PursuitEvasionEnv with GA STM propagation option
    """
    
    def __init__(self, config: Dict, use_gastm: bool = False):
        """
        Initialize environment with optional GA STM dynamics
        
        Args:
            config: Configuration dictionary
            use_gastm: If True, use GA STM for propagation
        """
        super().__init__(config)
        
        self.use_gastm = use_gastm
        self.gastm_propagator = None
        
        # Initialize GA STM if requested
        if self.use_gastm:
            self._init_gastm_propagator()
            
    def _init_gastm_propagator(self):
        """Initialize the GA STM propagator"""
        if self.relative_state is not None:
            self.gastm_propagator = GASTMPropagator(
                chief_orbit=self.evader_orbit,
                initial_relative_state=self.relative_state,
                config=self.config
            )
    
    def reset(self) -> np.ndarray:
        """Reset environment and GA STM propagator if needed"""
        obs = super().reset()
        
        if self.use_gastm:
            # Reinitialize GA STM propagator with new initial conditions
            self._init_gastm_propagator()
            
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step function with optional GA STM dynamics
        
        Args:
            action: Combined action vector [evader_dv, pursuer_dv]
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Parse actions
        delta_v_evader = action[:3] * self.delta_v_emax
        delta_v_pursuer = action[3:] * self.delta_v_pmax
        
        # Store pre-step state for reward calculation
        pre_distance = np.linalg.norm(self.relative_state[:3])
        
        # Apply dynamics based on selected mode
        if self.use_gastm and self.gastm_propagator is not None:
            self._step_gastm(delta_v_evader, delta_v_pursuer)
        else:
            self._step_nonlinear(delta_v_evader, delta_v_pursuer)
        
        # Update time and step count
        self.time += self.timestep
        self.step_count += 1
        
        # Get observation, reward, done, info
        obs = self._get_obs()
        reward = self._compute_reward(pre_distance)
        done = self._check_done()
        info = self._get_info()
        
        return obs, reward, done, info
    
    def _step_gastm(self, delta_v_evader: np.ndarray, delta_v_pursuer: np.ndarray):
        """
        Step using GA STM propagation
        
        Args:
            delta_v_evader: Evader's delta-v in LVLH
            delta_v_pursuer: Pursuer's delta-v in LVLH
        """
        # Apply evader's maneuver
        if np.linalg.norm(delta_v_evader) > 1e-6:
            # Apply impulse to chief orbit
            self.evader_orbit.apply_impulse(delta_v_evader)
            
            # Reinitialize GA STM with new chief orbit
            self.gastm_propagator.reinitialize_with_new_chief_orbit(
                new_chief_orbit=self.evader_orbit,
                current_relative_state=self.relative_state
            )
        
        # Apply pursuer's control and propagate
        if np.linalg.norm(delta_v_pursuer) > 1e-6:
            # Apply control using GA STM
            self.relative_state = self.gastm_propagator.apply_pursuer_control(
                delta_v_p=delta_v_pursuer,
                dt=self.timestep
            )
        else:
            # Just propagate without control
            self.relative_state = self.gastm_propagator.propagate(dt=self.timestep)
        
        # Propagate chief orbit
        self.evader_orbit.propagate(self.timestep)
        
        # Update history
        self.relative_state_history[self.step_count] = self.relative_state.copy()
        
    def _step_nonlinear(self, delta_v_evader: np.ndarray, delta_v_pursuer: np.ndarray):
        """
        Step using nonlinear dynamics (original implementation)
        
        Args:
            delta_v_evader: Evader's delta-v
            delta_v_pursuer: Pursuer's delta-v
        """
        # Apply evader impulse
        self.evader_orbit.apply_impulse(delta_v_evader)
        
        # Apply pursuer impulse to relative state
        self.relative_state[3:6] += delta_v_pursuer
        
        # Propagate using nonlinear dynamics
        sol = solve_ivp(
            relative_dynamics_evader_centered,
            [self.time, self.time + self.timestep],
            self.relative_state,
            args=(self.evader_orbit,),
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        
        # Update states
        self.relative_state = sol.y[:, -1]
        self.evader_orbit.propagate(self.timestep)
        
        # Update history
        self.relative_state_history[self.step_count] = self.relative_state.copy()
    
    def compare_propagation_methods(self, test_duration: float = 300.0, 
                                  control_sequence: Optional[np.ndarray] = None) -> Dict:
        """
        Compare GA STM and nonlinear propagation methods
        
        Args:
            test_duration: Duration of test in seconds
            control_sequence: Optional predefined control sequence
            
        Returns:
            Dictionary with comparison results
        """
        # Save initial state
        initial_relative_state = self.relative_state.copy()
        initial_evader_state = self.evader_orbit.get_state(self.time)
        initial_time = self.time
        
        # Number of steps
        n_steps = int(test_duration / self.timestep)
        
        # Generate control sequence if not provided
        if control_sequence is None:
            # Simple pursuit strategy
            control_sequence = []
            for i in range(n_steps):
                # Pursuer tries to approach
                rel_pos = initial_relative_state[:3]
                pursuit_direction = -rel_pos / np.linalg.norm(rel_pos)
                pursuer_dv = pursuit_direction * 0.01  # 1 cm/s per step
                
                # Evader does nothing initially
                evader_dv = np.zeros(3)
                
                control = np.concatenate([evader_dv, pursuer_dv])
                control_sequence.append(control)
            control_sequence = np.array(control_sequence)
        
        # Run with nonlinear dynamics
        self.use_gastm = False
        nonlinear_states = [initial_relative_state.copy()]
        
        for i in range(n_steps):
            self.step(control_sequence[i])
            nonlinear_states.append(self.relative_state.copy())
        
        # Reset to initial state
        self.relative_state = initial_relative_state.copy()
        self.evader_orbit.set_state(initial_evader_state)
        self.time = initial_time
        self.step_count = 0
        
        # Run with GA STM
        self.use_gastm = True
        self._init_gastm_propagator()
        gastm_states = [initial_relative_state.copy()]
        
        for i in range(n_steps):
            self.step(control_sequence[i])
            gastm_states.append(self.relative_state.copy())
        
        # Convert to arrays
        nonlinear_states = np.array(nonlinear_states)
        gastm_states = np.array(gastm_states)
        
        # Compute errors
        position_errors = np.linalg.norm(
            nonlinear_states[:, :3] - gastm_states[:, :3], axis=1
        )
        velocity_errors = np.linalg.norm(
            nonlinear_states[:, 3:] - gastm_states[:, 3:], axis=1
        )
        
        # Create time array
        time_array = np.arange(len(position_errors)) * self.timestep
        
        return {
            'time': time_array,
            'nonlinear_states': nonlinear_states,
            'gastm_states': gastm_states,
            'position_errors': position_errors,
            'velocity_errors': velocity_errors,
            'max_position_error': np.max(position_errors),
            'mean_position_error': np.mean(position_errors),
            'final_position_error': position_errors[-1],
            'max_velocity_error': np.max(velocity_errors),
            'mean_velocity_error': np.mean(velocity_errors),
            'final_velocity_error': velocity_errors[-1]
        }


def visualize_comparison_results(results: Dict):
    """
    Visualize the comparison between GA STM and nonlinear dynamics
    
    Args:
        results: Dictionary from compare_propagation_methods
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time array in minutes
    time_min = results['time'] / 60
    
    # Plot trajectories in X-Y plane
    ax = axes[0, 0]
    ax.plot(results['nonlinear_states'][:, 0]/1000, 
            results['nonlinear_states'][:, 1]/1000, 
            'b-', label='Nonlinear', linewidth=2)
    ax.plot(results['gastm_states'][:, 0]/1000, 
            results['gastm_states'][:, 1]/1000, 
            'r--', label='GA STM', linewidth=2)
    ax.set_xlabel('Radial (km)')
    ax.set_ylabel('Along-track (km)')
    ax.set_title('Relative Trajectory (X-Y Plane)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Plot relative distance over time
    ax = axes[0, 1]
    nl_dist = np.linalg.norm(results['nonlinear_states'][:, :3], axis=1)
    ga_dist = np.linalg.norm(results['gastm_states'][:, :3], axis=1)
    ax.plot(time_min, nl_dist/1000, 'b-', label='Nonlinear', linewidth=2)
    ax.plot(time_min, ga_dist/1000, 'r--', label='GA STM', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Relative Distance (km)')
    ax.set_title('Separation Distance')
    ax.legend()
    ax.grid(True)
    
    # Plot position error
    ax = axes[1, 0]
    ax.semilogy(time_min, results['position_errors'], 'g-', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error: |Nonlinear - GA STM|')
    ax.grid(True)
    
    # Plot velocity error
    ax = axes[1, 1]
    ax.semilogy(time_min, results['velocity_errors'], 'm-', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Velocity Error (m/s)')
    ax.set_title('Velocity Error: |Nonlinear - GA STM|')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== GA STM vs Nonlinear Dynamics Comparison ===")
    print(f"Simulation duration: {results['time'][-1]/60:.1f} minutes")
    print(f"\nPosition errors:")
    print(f"  Maximum: {results['max_position_error']:.2f} m")
    print(f"  Mean: {results['mean_position_error']:.2f} m")
    print(f"  Final: {results['final_position_error']:.2f} m")
    print(f"\nVelocity errors:")
    print(f"  Maximum: {results['max_velocity_error']:.4f} m/s")
    print(f"  Mean: {results['mean_velocity_error']:.4f} m/s")
    print(f"  Final: {results['final_velocity_error']:.4f} m/s")
    
    return fig


# Example usage
if __name__ == "__main__":
    from config.settings import get_config
    
    # Get configuration
    config = get_config()
    config['dt'] = 10.0  # 10 second timestep
    
    # Create environment with GA STM
    print("Creating environment with GA STM...")
    env = PursuitEvasionEnvGASTM(config, use_gastm=True)
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial separation: {np.linalg.norm(env.relative_state[:3]):.1f} m")
    
    # Test single step
    print("\nTesting single step with GA STM...")
    action = np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.0])  # Pursuer approaches
    obs, reward, done, info = env.step(action)
    print(f"After step - Distance: {info['distance']:.2f} m")
    
    # Compare methods
    print("\nComparing GA STM with nonlinear dynamics...")
    env.reset()
    results = env.compare_propagation_methods(test_duration=600.0)  # 10 minutes
    
    # Visualize results
    fig = visualize_comparison_results(results)
    plt.savefig('gastm_nonlinear_comparison.png', dpi=150)
    plt.show()
