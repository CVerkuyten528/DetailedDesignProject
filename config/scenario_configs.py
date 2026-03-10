# config/scenario_configs.py
"""
Pre-configured settings for different scenarios.
Use these as starting points for team collaboration.
"""

import numpy as np
from config.spacecraft_config import load_config


def config_passive_orbit():
    """Passive orbit with aerodynamic torques."""
    cfg = load_config()
    cfg.sim.stop_time_s = 300.0
    cfg.aero.enabled = True
    cfg.passive.use_gravity_gradient = False
    
    # Optional: enable gravity gradient
    # cfg.passive.use_gravity_gradient = True
    
    return cfg


def config_nadir_direct():
    """Nadir pointing with unlimited direct torque control."""
    cfg = load_config()
    cfg.sim.stop_time_s = 300.0
    cfg.aero.enabled = True
    cfg.passive.use_gravity_gradient = False
    
    # Control gains
    cfg.control.K = 3.0
    cfg.control.P = 25.0
    cfg.control.Ki = -1.0  # Integral disabled
    
    return cfg


def config_nadir_mtb():
    """Nadir pointing with magnetorquer actuation."""
    cfg = load_config()
    cfg.sim.stop_time_s = 300.0
    
    # Enable magnetic components
    cfg.magnetic_field.enabled = True
    cfg.magnetic_field.epoch_year = 2025.5
    cfg.magnetometer.enabled = True
    cfg.magnetorquer.enabled = True
    cfg.earth_horizon_sensor.enabled = True
    
    # Optional: enable aero
    cfg.aero.enabled = True
    cfg.passive.use_gravity_gradient = False
    
    # Reduced gains for limited MTB authority
    cfg.control.K = 1.0
    cfg.control.P = 10.0
    cfg.control.Ki = -1.0
    
    # Magnetometer settings (adjust noise as needed)
    cfg.magnetometer.noise_std_T = np.array([100e-9, 100e-9, 100e-9])
    
    # MTB limits
    cfg.magnetorquer.max_dipole_Am2 = np.array([10.0, 10.0, 10.0])
    
    # EHS attitude noise
    cfg.earth_horizon_sensor.attitude_noise_rad = np.deg2rad(2.0)
    
    return cfg


def config_detumble():
    """B-dot detumbling from high initial rates."""
    cfg = load_config()
    cfg.sim.stop_time_s = 600.0  # 10 minutes
    cfg.sim.fsw_step_s = 1.0  # 1 Hz for B-dot
    
    # Enable magnetic components
    cfg.magnetic_field.enabled = True
    cfg.magnetic_field.epoch_year = 2025.5
    cfg.magnetometer.enabled = True
    cfg.magnetorquer.enabled = True
    
    # Disable other effects for pure detumbling test
    cfg.aero.enabled = False
    cfg.passive.use_gravity_gradient = False
    
    # Initial tumbling state
    cfg.ic.sigma_BN = np.array([0.3, -0.2, 0.1])
    cfg.ic.omega_BN_B_rps = np.array([0.5, -0.3, 0.2])  # High rates
    
    # B-dot gain (tune based on inertia and desired damping)
    cfg.bdot.k_bdot = 5.0e5
    
    # Magnetometer settings
    cfg.magnetometer.noise_std_T = np.array([100e-9, 100e-9, 100e-9])
    
    # MTB limits
    cfg.magnetorquer.max_dipole_Am2 = np.array([10.0, 10.0, 10.0])
    
    return cfg


def config_detumble_aggressive():
    """Faster detumbling with higher gain and lower noise."""
    cfg = config_detumble()
    
    # Higher gain for faster detumbling
    cfg.bdot.k_bdot = 1.0e6
    
    # Lower magnetometer noise for better performance
    cfg.magnetometer.noise_std_T = np.array([50e-9, 50e-9, 50e-9])
    
    # Can increase MTB authority if hardware allows
    cfg.magnetorquer.max_dipole_Am2 = np.array([15.0, 15.0, 15.0])
    
    return cfg


def config_combined_detumble_then_nadir():
    """
    Configuration for two-stage operation:
    1. Detumble with B-dot
    2. Switch to nadir pointing
    
    Note: Mode switching logic needs to be implemented in scenario.
    """
    cfg = config_nadir_mtb()
    
    # Extended time for both phases
    cfg.sim.stop_time_s = 900.0  # 15 minutes
    
    # Initial tumbling state
    cfg.ic.sigma_BN = np.array([0.2, -0.15, 0.1])
    cfg.ic.omega_BN_B_rps = np.array([0.3, -0.2, 0.15])
    
    # B-dot settings (for phase 1)
    cfg.bdot.k_bdot = 5.0e5
    
    # Nadir control settings (for phase 2)
    cfg.control.K = 1.0
    cfg.control.P = 10.0
    
    return cfg


# Dictionary for easy access
SCENARIO_CONFIGS = {
    "passive": config_passive_orbit,
    "nadir_direct": config_nadir_direct,
    "nadir_mtb": config_nadir_mtb,
    "detumble": config_detumble,
    "detumble_aggressive": config_detumble_aggressive,
    "combined": config_combined_detumble_then_nadir,
}


def get_config(scenario_name):
    """
    Get configuration for a specific scenario.
    
    Args:
        scenario_name: One of "passive", "nadir_direct", "nadir_mtb", 
                       "detumble", "detumble_aggressive", "combined"
    
    Returns:
        SpacecraftConfig object
    """
    if scenario_name not in SCENARIO_CONFIGS:
        available = ", ".join(SCENARIO_CONFIGS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    return SCENARIO_CONFIGS[scenario_name]()


if __name__ == "__main__":
    # Example: print all available configurations
    print("Available scenario configurations:")
    for name in SCENARIO_CONFIGS.keys():
        cfg = get_config(name)
        print(f"\n{name}:")
        print(f"  Stop time: {cfg.sim.stop_time_s}s")
        print(f"  FSW step: {cfg.sim.fsw_step_s}s")
        print(f"  Magnetic field: {cfg.magnetic_field.enabled}")
        print(f"  Magnetometer: {cfg.magnetometer.enabled}")
        print(f"  Magnetorquer: {cfg.magnetorquer.enabled}")
        print(f"  Aerodynamics: {cfg.aero.enabled}")
