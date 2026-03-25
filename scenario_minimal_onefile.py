"""
Velocity-pointing ADCS simulation in Basilisk.

Purpose
-------
Full attitude determination and control system (ADCS) simulation with:
- Aerodynamic torque disturbances from lookup table
- Three-axis magnetometer (TAM) and magnetorquer (MTB) hardware
- Velocity-pointing reference frame generation
- MRP feedback control with attitude tracking
- Mode switching between B-dot detumbling and velocity-pointing control

Reference Frame Convention
--------------------------
Velocity-pointing reference frame R:
- +X_R: velocity direction (ram)
- +Z_R: orbit normal (h = r × v)
- +Y_R: completes right-handed frame (≈ radially inward for circular orbit)

The spacecraft starts tumbling and uses B-dot control to detumble, then switches
to velocity-pointing attitude control using magnetorquers for actuation.
"""

import os

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.utilities import simIncludeGravBody, orbitalMotion, RigidBodyKinematics as rbk, unitTestSupport
from Basilisk.simulation import spacecraft, magnetometer, MtbEffector, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import hillPoint, mrpFeedback, attTrackingError
from Basilisk.architecture import messaging, sysModel, bskLogging
import Basilisk.simulation.vizInterface as vizInterface
from Basilisk.utilities import vizSupport 
PROJECT_ROOT = Path(__file__).resolve().parent
from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


class NadirPDControl(sysModel.SysModel):
    """
    Nadir pointing control using cross product error.

    Calculates roll and pitch error by comparing a chosen body axis (e3) 
    with the Nadir vector (n_B) in the body frame.
    Control Law: tau = -Kp * (e3 x n_B) - Kd * omega_B
    """
    def __init__(self):
        super().__init__()
        self.ModelTag = "NadirPDControl"

        self.transNavInMsg = messaging.NavTransMsgReader()
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        self.Kp = 0.005  # Proportional gain
        self.Kd = 0.01   # Derivative gain
        self.align_axis = np.array([0.0, 0.0, 1.0])  # Body axis to point at Nadir

    def Reset(self, CurrentSimNanos):
        payload = self.cmdTorqueOutMsg.zeroMsgPayload
        self.cmdTorqueOutMsg.write(payload, CurrentSimNanos, self.moduleID)

    def UpdateState(self, CurrentSimNanos):
        if not self.transNavInMsg.isWritten() or not self.attNavInMsg.isWritten():
            return

        r_BN_N = np.array(self.transNavInMsg().r_BN_N, dtype=float)
        v_BN_N = np.array(self.transNavInMsg().v_BN_N, dtype=float)
        sigma_BN = np.array(self.attNavInMsg().sigma_BN, dtype=float)
        omega_BN_B = np.array(self.attNavInMsg().omega_BN_B, dtype=float)

        r_mag = np.linalg.norm(r_BN_N)
        if r_mag < 1e-6:
            return

        # Nadir vector in inertial frame (towards Earth center)
        nadir_N = -r_BN_N / r_mag

        # Convert Nadir vector to body frame
        C_BN = np.array(rbk.MRP2C(sigma_BN))
        nadir_B = C_BN @ nadir_N # [-sin roll cos pitch ; sin pitch; 0] change once earth horizo

        # Cross product error (e3 x n) gives the rotation axis to align e3 with n
        error_vec = np.cross(self.align_axis, nadir_B)

        # Calculate rotating frame rate to damp relative, not inertial, rotation
        h_N = np.cross(r_BN_N, v_BN_N)
        omega_RN_N = h_N / (r_mag**2) # Orbital rate vector
        omega_RN_B = C_BN @ omega_RN_N
        omega_err_B = omega_BN_B - omega_RN_B

        # Control torque (PD law)
        # Note: +Kp is used because e3 x n points in the direction we need to torque
        tau = self.Kp * error_vec - self.Kd * omega_err_B

        payload = self.cmdTorqueOutMsg.zeroMsgPayload
        payload.torqueRequestBody = tau.tolist()
        self.cmdTorqueOutMsg.write(payload, CurrentSimNanos, self.moduleID)


class SimpleMagneticField(sysModel.SysModel):
    """Very simple centered dipole magnetic field model in inertial frame N."""

    def __init__(self):
        super().__init__()
        self.mu_earth = 7.96e15  # T*m^3, simple dipole strength
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.magFieldOutMsg = messaging.MagneticFieldMsg()

    def Reset(self, currentTime):
        pass

    def UpdateState(self, currentTime):
        if not self.scStateInMsg.isWritten():
            return

        scState = self.scStateInMsg()
        r_N = np.array(scState.r_BN_N, dtype=float)
        r_mag = np.linalg.norm(r_N)

        if r_mag > 0.0:
            r_hat = r_N / r_mag
            m_unit = np.array([0.0, 0.0, -1.0])
            m_dot_r = np.dot(m_unit, r_hat)
            B_N = (self.mu_earth / r_mag**3) * (3.0 * m_dot_r * r_hat - m_unit)
        else:
            B_N = np.zeros(3)

        magFieldData = messaging.MagneticFieldMsgPayload()
        magFieldData.magField_N = B_N.tolist()
        self.magFieldOutMsg.write(magFieldData, self.moduleID, currentTime)


class SimpleBdot(sysModel.SysModel):
    """
    Simple B-dot controller for detumbling.

    Implements the B-dot control law: m = -k * dB/dt
    where m is the magnetic dipole moment and dB/dt is the rate of change of the magnetic field.
    This creates a damping torque to reduce spacecraft angular velocity.
    """

    def __init__(self):
        super().__init__()
        self.k_gain = 1200   # Gain for B-dot control [A·m²·s/T²]
        self.maxDipole = 0.016
        self.dt = 0.1
        self.B_prev = np.zeros(3)
        self.first = True

        self.tamInMsg = messaging.TAMSensorMsgReader()
        self.cmdOutMsg = messaging.MTBCmdMsg()

    def Reset(self, currentTime):
        self.B_prev = np.zeros(3)
        self.first = True
        cmdOut = messaging.MTBCmdMsgPayload()
        cmdOut.mtbDipoleCmds = [0.0, 0.0, 0.0]
        self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)

    def UpdateState(self, currentTime):
        if not self.tamInMsg.isWritten():
            return

        B = np.array(self.tamInMsg().tam_S, dtype=float)

        if self.first:
            self.B_prev = B
            self.first = False
            dipole = np.zeros(3)
        else:
            dBdt = (B - self.B_prev) / self.dt
            dipole = -self.k_gain * dBdt
            dipole = np.clip(dipole, -self.maxDipole, self.maxDipole)
            self.B_prev = B

        cmdOut = messaging.MTBCmdMsgPayload()
        cmdOut.mtbDipoleCmds = dipole.tolist()
        self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)


class MagneticMomentumManagement(sysModel.SysModel):
    """
    Magnetic torque allocation for commanded body torque.

    Allocates a desired control torque to magnetorquer dipoles using the cross product:
    τ = m × B  =>  m = (B × τ) / |B|²

    Note: Can only control 2 axes instantaneously (perpendicular to B-field).
    The activeAxes mask allows disabling specific magnetorquer axes.
    """

    def __init__(self):
        super().__init__()
        self.maxDipole = 0.016
        self.activeAxes = [True, True, True]  # [X, Y, Z] enable mask

        self.tamInMsg = messaging.TAMSensorMsgReader()
        self.cmdTorqueInMsg = messaging.CmdTorqueBodyMsgReader()
        self.cmdOutMsg = messaging.MTBCmdMsg()

    def Reset(self, currentTime):
        cmdOut = messaging.MTBCmdMsgPayload()
        cmdOut.mtbDipoleCmds = [0.0, 0.0, 0.0]
        self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)

    def UpdateState(self, currentTime):
        if not self.tamInMsg.isWritten() or not self.cmdTorqueInMsg.isWritten():
            return

        B_B = np.array(self.tamInMsg().tam_S, dtype=float)
        tau_des = np.array(self.cmdTorqueInMsg().torqueRequestBody, dtype=float)

        B_mag_sq = np.dot(B_B, B_B)
        if B_mag_sq > 1e-20:
            dipole = np.cross(B_B, tau_des) / B_mag_sq
            dipole = np.clip(dipole, -self.maxDipole, self.maxDipole)

            # Apply axis mask
            for i in range(3):
                if not self.activeAxes[i]:
                    dipole[i] = 0.0
        else:
            dipole = np.zeros(3)

        cmdOut = messaging.MTBCmdMsgPayload()
        cmdOut.mtbDipoleCmds = dipole.tolist()
        self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)


class ModeSwitch(sysModel.SysModel):
    """
    Mode switching logic between BDOT (detumbling) and NADIR (pointing) control.

    Switches from BDOT to NADIR mode when the spacecraft angular rate falls below
    a threshold, indicating successful detumbling. This allows transition from
    rate damping to attitude pointing control.
    """

    def __init__(self):
        super().__init__()
        self.rate_threshold = 0.5 * macros.D2R  # Switch at 0.5 deg/s
        self.mode = "BDOT"
        self.switched = False
        self.allowSwitching = False

        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.bdotCmdInMsg = messaging.MTBCmdMsgReader()
        self.nadirCmdInMsg = messaging.MTBCmdMsgReader()
        self.cmdOutMsg = messaging.MTBCmdMsg()

    def Reset(self, currentTime):
        self.mode = "BDOT"
        self.switched = False
        cmdOut = messaging.MTBCmdMsgPayload()
        cmdOut.mtbDipoleCmds = [0.0, 0.0, 0.0]
        self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)

    def UpdateState(self, currentTime):
        if not self.scStateInMsg.isWritten():
            return

        scState = self.scStateInMsg()
        omega_B = np.array(scState.omega_BN_B, dtype=float)
        omega_mag = np.linalg.norm(omega_B)

        if self.allowSwitching and self.mode == "BDOT" and omega_mag < self.rate_threshold:
            self.mode = "NADIR"
            if not self.switched:
                print(
                    f"✓ Switching to NADIR mode at "
                    f"t={currentTime*macros.NANO2SEC:.1f}s "
                    f"(ω={omega_mag*macros.R2D:.3f} deg/s)"
                )
                self.switched = True

        if self.mode == "BDOT" and self.bdotCmdInMsg.isWritten():
            cmd = self.bdotCmdInMsg()
        elif self.mode == "NADIR" and self.nadirCmdInMsg.isWritten():
            cmd = self.nadirCmdInMsg()
        else:
            cmdOut = messaging.MTBCmdMsgPayload()
            cmdOut.mtbDipoleCmds = [0.0, 0.0, 0.0]
            self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)
            return

        cmdOut = messaging.MTBCmdMsgPayload()
        cmdOut.mtbDipoleCmds = list(cmd.mtbDipoleCmds)
        self.cmdOutMsg.write(cmdOut, self.moduleID, currentTime)


class TorqueLookup:
    """
    Regular-grid aerodynamic torque table interpolator.

    Performs 2D linear interpolation of precomputed aerodynamic torques as a function of:
    - alpha: angle of attack (rotation about body Y axis) [-180, 180] degrees
    - beta: sideslip angle (rotation about body Z axis) [-180, 180] degrees

    The torque table should have shape (3, n_alpha, n_beta) containing body-frame torques.
    """

    def __init__(
        self,
        torque_table: np.ndarray,
        alpha_range: tuple = (-180.0, 180.0),
        beta_range: tuple = (-180.0, 180.0)
    ):
        self.torque_table = np.asarray(torque_table, dtype=float)
        if self.torque_table.ndim != 3 or self.torque_table.shape[0] != 3:
            raise ValueError(f"torque_table must have shape (3, n_alpha, n_beta), got {self.torque_table.shape}")

        _, n_alpha, n_beta = self.torque_table.shape
        self.alpha_vals = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        self.beta_vals = np.linspace(beta_range[0], beta_range[1], n_beta)

        self._interps = [
            RegularGridInterpolator(
                (self.alpha_vals, self.beta_vals),
                self.torque_table[i],
                method="linear",
                bounds_error=False,
                fill_value=None
            )
            for i in range(3)
        ]

    def angles_from_vbody(self, v_body):
        v = np.asarray(v_body, dtype=float).reshape(3,)
        u, vy, w = v
        speed = np.linalg.norm(v)

        if speed < 1e-12:
            return 0.0, 0.0, 0.0

        alpha_deg = np.degrees(np.arctan2(w, u))
        beta_deg = np.degrees(np.arctan2(vy, np.hypot(u, w)))

        alpha_deg = float(np.clip(alpha_deg, self.alpha_vals[0], self.alpha_vals[-1]))
        beta_deg = float(np.clip(beta_deg, self.beta_vals[0], self.beta_vals[-1]))
        return alpha_deg, beta_deg, speed

    def lookup_from_vbody(self, v_body):
        alpha_deg, beta_deg, speed = self.angles_from_vbody(v_body)
        if speed < 1e-12:
            return np.zeros(3)

        pt = np.array([alpha_deg, beta_deg])
        return np.array([f(pt).item() for f in self._interps])


class AeroTorqueFromTable(sysModel.SysModel):
    """Aerodynamic torque model using a precomputed lookup table."""

    def __init__(self, torque_table):
        super().__init__()
        self.ModelTag = "AeroTorqueFromTable"

        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        self.lookup = TorqueLookup(torque_table)

        self.useNegativeRelativeWind = False  # Table expects velocity direction, not wind
        self.assumeAtmosphereCorotates = False
        self.omegaPlanet_N = np.array([0.0, 0.0, 7.2921159e-5])

    def Reset(self, CurrentSimNanos):
        if not self.scStateInMsg.isLinked():
            bskLogging.bskLog(
                bskLogging.BSK_ERROR,
                "AeroTorqueFromTable.scStateInMsg is not linked."
            )

        payload = self.cmdTorqueOutMsg.zeroMsgPayload
        payload.torqueRequestBody = [0.0, 0.0, 0.0]
        self.cmdTorqueOutMsg.write(payload, CurrentSimNanos, self.moduleID)

    def compute_vrel_N(self, r_BN_N, v_BN_N):
        if self.assumeAtmosphereCorotates:
            v_atm_N = np.cross(self.omegaPlanet_N, r_BN_N)
        else:
            v_atm_N = np.zeros(3)
        return v_BN_N - v_atm_N

    def UpdateState(self, CurrentSimNanos):
        if not self.scStateInMsg.isWritten():
            return

        sc = self.scStateInMsg()

        sigma_BN = np.array(sc.sigma_BN, dtype=float)
        r_BN_N = np.array(sc.r_BN_N, dtype=float)
        v_BN_N = np.array(sc.v_BN_N, dtype=float)

        C_BN = np.array(rbk.MRP2C(sigma_BN))
        v_rel_N = self.compute_vrel_N(r_BN_N, v_BN_N)
        v_rel_B = C_BN @ v_rel_N
        v_query_B = -v_rel_B if self.useNegativeRelativeWind else v_rel_B

        torque_B = self.lookup.lookup_from_vbody(v_query_B)

        payload = self.cmdTorqueOutMsg.zeroMsgPayload
        payload.torqueRequestBody = torque_B.tolist()
        self.cmdTorqueOutMsg.write(payload, CurrentSimNanos, self.moduleID)


def run_adcs_sim():
    sim_dt = 0.1
    stop_time = 1*5400.0
    altitude = 300000.0  # 300 km altitude
    alpha = 30.0;  # Deployment angle for aerodynamic torque table selection choose from (0, 30, 60 ,90)

    scSim = SimulationBaseClass.SimBaseClass()
    simProcessName = "simProcess"
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simTaskName = "simTask"
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, macros.sec2nano(sim_dt)))

    # ------------------------------------------------------------------
    # SPACECRAFT
    # ------------------------------------------------------------------
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "sat"
    scObject.hub.mHub = 0.75  # Spacecraft mass [kg]
    # r_BcB_B and IHubPntBc_B will be set after loading the .mat files

    # Initial attitude: 
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    # Initial angular velocity: tumbling at 0.5 rad/s on all axes
    scObject.hub.omega_BN_BInit = [[0.5], [0.5], [0.5]]

    # ------------------------------------------------------------------
    # ORBIT
    # ------------------------------------------------------------------
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    mu = earth.mu

    # Define orbital elements for Sun-synchronous LEO orbit
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371000.0+altitude  # Semi-major axis [m] (~300 km altitude)
    oe.e = 0.0001  # Eccentricity (near-circular)
    oe.i = 97.0 * macros.D2R  # Inclination [rad] (sun-synchronous)
    oe.Omega = 0.0  # Right ascension of ascending node [rad]
    oe.omega = 0.0  # Argument of periapsis [rad]
    oe.f = 0.0  # True anomaly [rad]

    # Convert orbital elements to initial position and velocity
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    gravFactory.addBodiesTo(scObject)



    # ------------------------------------------------------------------
    # TORQUE TABLE SCALING (density and velocity)
    # ------------------------------------------------------------------
    def get_density_exp(h_km):
        # Simple exponential model, valid for 300-600 km
        # Reference: US Standard Atmosphere, scale height ~59 km
        rho0 = 2.3781e-11  # kg/m^3 at 300 km
        h0 = 300.0  # km
        H = 59.0    # km (scale height)
        return rho0 * np.exp(-(h_km - h0) / H)

    def get_orbital_velocity(a_m, mu=3.986004418e14):
        return np.sqrt(mu / a_m)

    # Reference values at 300 km
    ref_alt_km = 300.0
    ref_r_m = 6371000.0 + ref_alt_km * 1e3
    rho_ref = 2.3781e-11
    v_ref = get_orbital_velocity(ref_r_m, mu)

    # Current values
    alt_km = (oe.a - 6371000.0) / 1e3
    rho = get_density_exp(alt_km)
    v = get_orbital_velocity(oe.a, mu)

    scale = (rho / rho_ref) * (v / v_ref) ** 2

    # ------------------------------------------------------------------
    # AERODYNAMIC TORQUE TABLE
    # ------------------------------------------------------------------
    # Load precomputed aerodynamic torque table from MATLAB/ADBSat  
    # Table contains 3D torques as a function of angle-of-attack (alpha) and sideslip (beta)
    torque_table_file = f"torque_table_{int(alpha)}.mat"  # choose the right table file for the deployment angle
    torque_path = PROJECT_ROOT / "data" / torque_table_file
    mat_data = loadmat(torque_path, simplify_cells=True)
    torque_table = np.asarray(mat_data["torque_table"], dtype=float)
    # Scale the torque table for the current altitude
    torque_table = torque_table * scale

    # Load inertia and CoM from their own .mat files, using the correct slice for alpha
    data_I = loadmat(PROJECT_ROOT / "data" / "I_total_hist.mat", simplify_cells=True)
    data_CoM = loadmat(PROJECT_ROOT / "data" / "CoM_hist.mat", simplify_cells=True)
    I_total_hist = np.asarray(data_I["I_total_hist"], dtype=float)
    CoM_hist = np.asarray(data_CoM["CoM_hist"], dtype=float)
    alpha_idx = int(alpha)
    scObject.hub.IHubPntBc_B = I_total_hist[:, :, alpha_idx].tolist()
    scObject.hub.r_BcB_B = CoM_hist[:, alpha_idx].reshape((3,1)).tolist()

    aeroTorque = AeroTorqueFromTable(torque_table)
    aeroTorque.ModelTag = "AeroTorque"
    aeroTorque.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, aeroTorque, 95)

    extFT = extForceTorque.ExtForceTorque()
    extFT.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extFT)
    scSim.AddModelToTask(simTaskName, extFT)
    extFT.cmdTorqueInMsg.subscribeTo(aeroTorque.cmdTorqueOutMsg)

    # ------------------------------------------------------------------
    # MAGNETIC FIELD MODEL
    # ------------------------------------------------------------------
    magModule = SimpleMagneticField()
    magModule.ModelTag = "MagField"
    magModule.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, magModule, 100)

    # ------------------------------------------------------------------
    # MAGNETOMETER
    # ------------------------------------------------------------------
    TAM = magnetometer.Magnetometer()
    TAM.ModelTag = "TAM"
    TAM.scaleFactor = 1.0
    TAM.senNoiseStd = [15e-9, 15e-9, 15e-9]
    TAM.maxOutput = 800e-6
    TAM.minOutput = -800e-6
    TAM.dcm_SB = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    TAM.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    TAM.magInMsg.subscribeTo(magModule.magFieldOutMsg)
    scSim.AddModelToTask(simTaskName, TAM, 90)

    # ------------------------------------------------------------------
    # MAGNETORQUERS
    # ------------------------------------------------------------------
    mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
    mtbConfigParams.numMTB = 3
    mtbConfigParams.GtMatrix_B = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]
    mtbConfigParams.maxMtbDipoles = [0.016, 0.016, 0.016]
    mtbParamsMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)

    MTB = MtbEffector.MtbEffector()
    MTB.ModelTag = "MTB"
    scObject.addDynamicEffector(MTB)
    MTB.mtbParamsInMsg.subscribeTo(mtbParamsMsg)
    MTB.magInMsg.subscribeTo(magModule.magFieldOutMsg)
    scSim.AddModelToTask(simTaskName, MTB, 50)

    # ------------------------------------------------------------------
    # B-DOT
    # ------------------------------------------------------------------
    bdot = SimpleBdot()
    bdot.ModelTag = "Bdot"
    bdot.k_gain = 500
    bdot.maxDipole = 0.016
    bdot.dt = sim_dt
    bdot.tamInMsg.subscribeTo(TAM.tamDataOutMsg)
    scSim.AddModelToTask(simTaskName, bdot, 80)

    # ------------------------------------------------------------------
    # SIMPLE NAV
    # ------------------------------------------------------------------
    simpleNavModule = simpleNav.SimpleNav()
    simpleNavModule.ModelTag = "SimpleNav"
    simpleNavModule.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, simpleNavModule, 91)

    # ------------------------------------------------------------------
    # NADIR PD CONTROL
    # ------------------------------------------------------------------
    nadirControl = NadirPDControl()
    nadirControl.ModelTag = "NadirPDControl"
    nadirControl.Kp = 0.005  # Proportional gain [N*m]
    nadirControl.Kd = 0.01   # Derivative gain [N*m*s]
    nadirControl.align_axis = np.array([0.0, 0.0, 1.0])  # Body Z-axis pointing at Nadir
    nadirControl.transNavInMsg.subscribeTo(simpleNavModule.transOutMsg)
    nadirControl.attNavInMsg.subscribeTo(simpleNavModule.attOutMsg)
    scSim.AddModelToTask(simTaskName, nadirControl, 89)

    magMomentumCtrl = MagneticMomentumManagement()
    magMomentumCtrl.ModelTag = "MagMomentum"
    magMomentumCtrl.maxDipole = 0.016
    # Disable/enable dipoles
    magMomentumCtrl.activeAxes = [True, True, True]  # [X, Y, Z]
    magMomentumCtrl.tamInMsg.subscribeTo(TAM.tamDataOutMsg)
    magMomentumCtrl.cmdTorqueInMsg.subscribeTo(nadirControl.cmdTorqueOutMsg)
    scSim.AddModelToTask(simTaskName, magMomentumCtrl, 85)

    # ------------------------------------------------------------------
    # MODE SWITCH
    # ------------------------------------------------------------------
    modeSwitch = ModeSwitch()
    modeSwitch.ModelTag = "ModeSwitch"
    modeSwitch.rate_threshold = 0.1 * macros.D2R
    modeSwitch.allowSwitching = True  # Enable mode switching
    modeSwitch.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    modeSwitch.bdotCmdInMsg.subscribeTo(bdot.cmdOutMsg)
    modeSwitch.nadirCmdInMsg.subscribeTo(magMomentumCtrl.cmdOutMsg)
    scSim.AddModelToTask(simTaskName, modeSwitch, 75)

    MTB.mtbCmdInMsg.subscribeTo(modeSwitch.cmdOutMsg)

    scSim.AddModelToTask(simTaskName, scObject, 200)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------
    scLog = scObject.scStateOutMsg.recorder()
    tamLog = TAM.tamDataOutMsg.recorder()
    mtbLog = MTB.mtbOutMsg.recorder()
    bdotLog = bdot.cmdOutMsg.recorder()
    controlLog = nadirControl.cmdTorqueOutMsg.recorder()
    nadirLog = magMomentumCtrl.cmdOutMsg.recorder()
    modeLog = modeSwitch.cmdOutMsg.recorder()
    aeroLog = aeroTorque.cmdTorqueOutMsg.recorder()

    scSim.AddModelToTask(simTaskName, scLog)
    scSim.AddModelToTask(simTaskName, tamLog)
    scSim.AddModelToTask(simTaskName, mtbLog)
    scSim.AddModelToTask(simTaskName, bdotLog)
    scSim.AddModelToTask(simTaskName, controlLog)
    scSim.AddModelToTask(simTaskName, nadirLog)
    scSim.AddModelToTask(simTaskName, modeLog)
    scSim.AddModelToTask(simTaskName, aeroLog)



    viz = vizSupport.enableUnityVisualization(
        scSim,
        simTaskName,
        scObject,
        saveFile=fileName,
        liveStream=False
    )
    if  vizSupport.vizFound:
        print('vizfound')
        # Force these on by default in the .bin file
        viz.settings.showAngularVelocityVectors = True
        viz.settings.showMtbForceVectors = True  
        viz.settings.showMtbDipoleVectors = True 
        viz.settings.showMagneticFieldVectors = True
        viz.settings.showDataPanel = True        
        viz.settings.guiPlaybackSpeed = 100.0
        vizSupport.setActuatorGuiSetting(viz, viewRWPanel=True, viewRWHUD=True)

    # ------------------------------------------------------------------
    # This creates a .dot file named after your python script
    # ------------------------------------------------------------------
    
    # dot_file_name = fileName + ".dot"
    # unitTestSupport.writeToGraphviz(scSim, "adcs_simulation_map.dot")

    # print("Graphviz .dot file successfully generated as 'adcs_simulation_map.dot'")
    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(stop_time))
    scSim.ExecuteSimulation()

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------
    time_min = scLog.times() * macros.NANO2SEC / 60.0

    omega_BN_B = np.array(scLog.omega_BN_B)
    omega_BN_mag = np.linalg.norm(omega_BN_B, axis=1)

    sigma_BN = np.array(scLog.sigma_BN)
    r_BN_N = np.array(scLog.r_BN_N)

    # Calculate nadir tracking error (angle between body z-axis and literal moving nadir)
    body_z_to_nadir_deg = np.zeros(len(sigma_BN))
    for i in range(len(sigma_BN)):
        rN = r_BN_N[i]
        r_mag = np.linalg.norm(rN)
        if r_mag > 1e-6:
            nadir_N = -rN / r_mag
            C_BN = np.array(rbk.MRP2C(sigma_BN[i]))
            nadir_B = C_BN @ nadir_N
            body_z_to_nadir_deg[i] = np.degrees(np.arccos(np.clip(nadir_B[2], -1.0, 1.0)))

    B_tam_S_nT = np.array(tamLog.tam_S) * 1e9
    dipole_cmd = np.array(modeLog.mtbDipoleCmds)

    torque_mtb_uNm = np.array(mtbLog.mtbNetTorque_B) * 1e6
    torque_aero_uNm = np.array(aeroLog.torqueRequestBody) * 1e6

    # Calculate mean orbital rate for reference
    n_orbital = np.sqrt(mu / oe.a**3)

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(7, 1, figsize=(12, 26), sharex=True)

    # Plot 0: Body rate
    axes[0].plot(time_min, omega_BN_mag * macros.R2D, color='tab:purple', label='|ω_BN|')
    axes[0].axhline(n_orbital * macros.R2D, color='r', linestyle='--',
                    label=f'orbital rate ≈ {n_orbital * macros.R2D:.4f} deg/s')
    axes[0].set_ylabel('|ω_BN| [deg/s]')
    axes[0].set_title('Inertial body rate magnitude')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 1: Inertial attitude MRPs (body relative to inertial frame)
    axes[1].plot(time_min, sigma_BN[:, 0], label='σ1')
    axes[1].plot(time_min, sigma_BN[:, 1], label='σ2')
    axes[1].plot(time_min, sigma_BN[:, 2], label='σ3')
    axes[1].axhline(0.0, color='k', linestyle='--', linewidth=0.6)
    axes[1].set_ylabel('σ_BN [-]')
    axes[1].set_title('Inertial attitude MRPs')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 2: Nadir pointing error
    axes[2].plot(time_min, body_z_to_nadir_deg, color='tab:orange', label='Nadir Error')
    axes[2].set_ylabel('Angle [deg]')
    axes[2].set_title('Body +Z alignment offset from Nadir')
    axes[2].legend()
    axes[2].grid(True)

    # Plot 3: Magnetometer (TAM) measurement magnitude
    axes[3].plot(time_min, np.linalg.norm(B_tam_S_nT, axis=1), color='tab:green')
    axes[3].set_ylabel('|B| [nT]')
    axes[3].set_title('TAM magnitude')
    axes[3].grid(True)

    # Plot 4: Commanded magnetorquer (MTB) dipole moments
    axes[4].plot(time_min, dipole_cmd[:, 0], label='mx')
    axes[4].plot(time_min, dipole_cmd[:, 1], label='my')
    axes[4].plot(time_min, dipole_cmd[:, 2], label='mz')
    axes[4].axhline(0.0, color='k', linestyle='--', linewidth=0.6)
    axes[4].set_ylabel('Dipole [A·m²]')
    axes[4].set_title('Commanded MTB dipoles')
    axes[4].legend()
    axes[4].grid(True)

    # Plot 5: Applied magnetorquer torques on spacecraft body
    axes[5].plot(time_min, torque_mtb_uNm[:, 0], label='τx MTB')
    axes[5].plot(time_min, torque_mtb_uNm[:, 1], label='τy MTB')
    axes[5].plot(time_min, torque_mtb_uNm[:, 2], label='τz MTB')
    axes[5].axhline(0.0, color='k', linestyle='--', linewidth=0.6)
    axes[5].set_ylabel('MTB torque [µN·m]')
    axes[5].set_title('Applied MTB torques')
    axes[5].legend()
    axes[5].grid(True)

    # Plot 6: Applied aerodynamic torques on spacecraft body
    axes[6].plot(time_min, torque_aero_uNm[:, 0], label='τx aero')
    axes[6].plot(time_min, torque_aero_uNm[:, 1], label='τy aero')
    axes[6].plot(time_min, torque_aero_uNm[:, 2], label='τz aero')
    axes[6].axhline(0.0, color='k', linestyle='--', linewidth=0.6)
    axes[6].set_ylabel('Aero torque [µN·m]')
    axes[6].set_xlabel('Time [min]')
    axes[6].set_title('Applied aerodynamic torques')
    axes[6].legend()
    axes[6].grid(True)

    plt.tight_layout()
    plt.savefig('velocity_pointing_adcs_results.png', dpi=150)
    plt.show()

    return {
        "sim": scSim,
        "scLog": scLog,
        "tamLog": tamLog,
        "mtbLog": mtbLog,
        "bdotLog": bdotLog,
        "nadirLog": nadirLog,
        "modeLog": modeLog,
        "aeroLog": aeroLog,
        "controlLog": controlLog
    }


if __name__ == "__main__":
    run_adcs_sim()
