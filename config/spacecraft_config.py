# config/spacecraft_config.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimulationConfig:
    dyn_step_s: float = 0.01
    sensor_step_s: float = 0.05
    fsw_step_s: float = 0.05
    stop_time_s: float = 300.0


@dataclass
class HubConfig:
    mass_kg: float = 0.75
    r_BcB_B_m: np.ndarray = field(default_factory=lambda: np.array([-0.0558, 0.0, 0.0]))
    IHubPntBc_B_kgm2: np.ndarray = field(default_factory=lambda: np.array([
        [0.015, 0.0, 0.0],
        [0.0, 0.062, 0.0],
        [0.0, 0.0, 0.062]
    ]))


@dataclass
class InitialConditionConfig:
    # sigma_BN: body attitude relative to inertial (N) frame (MRP)
    sigma_BN: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    omega_BN_B_rps: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5/60, 0.0]))
    # Optional: if provided, treated as body->Hill MRP. When present, thisg2
    # will be converted to an inertial initial MRP during dynamics build.
    sigma_BH: np.ndarray | None = None


@dataclass
class WindTunnelConfig:
    v_inf_N_mps: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    atmosphere_velocity_N_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class OrbitConfig:
    a_m: float = 6771000.0
    e: float = 0.00
    i_deg: float = 33.3
    Omega_deg: float = 48.2
    omega_deg: float = 347.8
    f_deg: float = 85.3


@dataclass
class AeroConfig:
    enabled: bool = True
    torque_table_file: str = "torque_table1.mat" # 1 for +x 2 for -x
    useNegativeRelativeWind: bool = False


@dataclass
class PassiveConfig:
    use_gravity_gradient: bool = False


@dataclass
class NadirConfig:
    enabled: bool = False
    sigma_R0R: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

@dataclass
class ControlConfig:
    K: float = 3.0
    P: float = 25.0
    Ki: float = -1.0
    integral_limit: float = 0.1

@dataclass
class NavConfig:
    use_simple_nav: bool = True

@dataclass
class MagnetometerConfig:
    enabled: bool = True
    noise_std_T: np.ndarray = field(default_factory=lambda: np.array([100e-9, 100e-9, 100e-9]))
    max_output_T: float = 65000e-9
    min_output_T: float = -65000e-9
    scale_factor: float = 1.0

@dataclass
class EarthHorizonSensorConfig:
    enabled: bool = True
    # Attitude noise in radians (1-sigma)
    attitude_noise_rad: float = np.deg2rad(2.0)

@dataclass
class MagnetorquerConfig:
    enabled: bool = True
    # Maximum dipole moment per axis [A·m²]
    max_dipole_Am2: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0]))
    # Torquer alignment matrix (body frame) - columns are dipole axes
    alignment_matrix_B: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))

@dataclass
class MagneticFieldConfig:
    enabled: bool = True
    epoch_year: float = 2025.5  # Fractional year for WMM model

@dataclass
class BdotConfig:
    k_bdot: float = 1.0e6  # [A·m²·s/T²] gain

@dataclass
class SpacecraftConfig:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    wind: WindTunnelConfig = field(default_factory=WindTunnelConfig)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    ic: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    aero: AeroConfig = field(default_factory=AeroConfig)
    passive: PassiveConfig = field(default_factory=PassiveConfig)
    nadir: NadirConfig = field(default_factory=NadirConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    magnetometer: MagnetometerConfig = field(default_factory=MagnetometerConfig)
    earth_horizon_sensor: EarthHorizonSensorConfig = field(default_factory=EarthHorizonSensorConfig)
    magnetorquer: MagnetorquerConfig = field(default_factory=MagnetorquerConfig)
    magnetic_field: MagneticFieldConfig = field(default_factory=MagneticFieldConfig)
    bdot: BdotConfig = field(default_factory=BdotConfig)


def load_config():
    return SpacecraftConfig()
