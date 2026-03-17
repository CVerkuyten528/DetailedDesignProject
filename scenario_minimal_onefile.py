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
from Basilisk.utilities import simIncludeGravBody, orbitalMotion, RigidBodyKinematics as rbk
from Basilisk.simulation import spacecraft, magnetometer, MtbEffector, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import hillPoint, mrpFeedback, attTrackingError
from Basilisk.architecture import messaging, sysModel, bskLogging


PROJECT_ROOT = Path(__file__).resolve().parent
from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


class VelocityPointingReference(sysModel.SysModel):
    """
    Velocity-pointing reference frame generator.

    Reference frame R:
    - +X_R: velocity direction (ram)
    - +Z_R: orbit normal (h = r × v)
    - +Y_R: Z_R × X_R (completes right-handed frame, ~radially inward)
    """

    def __init__(self):
        super().__init__()
        self.ModelTag = "VelocityPointing"

        self.transNavInMsg = messaging.NavTransMsgReader()
        self.attRefOutMsg = messaging.AttRefMsg()

    def Reset(self, currentTime):
        if not self.transNavInMsg.isLinked():
            bskLogging.bskLog(
                bskLogging.BSK_ERROR,
                f"{self.ModelTag}.transNavInMsg is not linked."
            )

        attRefOut = messaging.AttRefMsgPayload()
        attRefOut.sigma_RN = [0.0, 0.0, 0.0]
        attRefOut.omega_RN_N = [0.0, 0.0, 0.0]
        attRefOut.domega_RN_N = [0.0, 0.0, 0.0]
        self.attRefOutMsg.write(attRefOut, self.moduleID, currentTime)

    def UpdateState(self, currentTime):
        if not self.transNavInMsg.isWritten():
            return

        navData = self.transNavInMsg()

        r_BN_N = np.array(navData.r_BN_N, dtype=float)
        v_BN_N = np.array(navData.v_BN_N, dtype=float)

        r_mag = np.linalg.norm(r_BN_N)
        v_mag = np.linalg.norm(v_BN_N)

        if r_mag < 1e-6 or v_mag < 1e-6:
            attRefOut = messaging.AttRefMsgPayload()
            attRefOut.sigma_RN = [0.0, 0.0, 0.0]
            attRefOut.omega_RN_N = [0.0, 0.0, 0.0]
            attRefOut.domega_RN_N = [0.0, 0.0, 0.0]
            self.attRefOutMsg.write(attRefOut, self.moduleID, currentTime)
            return

        # Reference frame R with +X along velocity
        x_R_N = v_BN_N / v_mag  # velocity direction

        # Orbit normal
        h_vec = np.cross(r_BN_N, v_BN_N)
        h_mag = np.linalg.norm(h_vec)
        if h_mag < 1e-10:
            z_R_N = np.array([0.0, 0.0, 1.0])
        else:
            z_R_N = h_vec / h_mag

        # Complete right-handed frame
        y_R_N = np.cross(z_R_N, x_R_N)
        y_R_N = y_R_N / np.linalg.norm(y_R_N)

        # DCM from inertial to reference frame
        C_RN = np.vstack([x_R_N, y_R_N, z_R_N])
        sigma_RN = rbk.C2MRP(C_RN)

        # Angular velocity of reference frame
        # omega_RN = d(C_RN)/dt * C_NR^T
        # For circular orbit approximation: omega is along h direction
        omega_orb = v_mag / r_mag  # approximate orbital rate
        omega_RN_N = omega_orb * z_R_N

        attRefOut = messaging.AttRefMsgPayload()
        attRefOut.sigma_RN = sigma_RN.tolist()
        attRefOut.omega_RN_N = omega_RN_N.tolist()
        attRefOut.domega_RN_N = [0.0, 0.0, 0.0]

        self.attRefOutMsg.write(attRefOut, self.moduleID, currentTime)


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
        self.k_gain = 0.0
        self.maxDipole = 0.016
        self.dt = 1.0
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
    stop_time = 3*5400.0

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
    scObject.hub.r_BcB_B = [[-0.0558], [0.0], [0.0]]  # Center of mass offset from body origin [m]
    # Moment of inertia tensor about center of mass [kg*m^2]
    scObject.hub.IHubPntBc_B = [
        [0.0015, 0.0, 0.0],
        [0.0, 0.0062, 0.0],
        [0.0, 0.0, 0.0062]
    ]

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
    oe.a = 6771000.0  # Semi-major axis [m] (~400 km altitude)
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
    # AERODYNAMIC TORQUE TABLE
    # ------------------------------------------------------------------
    # Load precomputed aerodynamic torque table from MATLAB/ADBSat  
    # Table contains 3D torques as a function of angle-of-attack (alpha) and sideslip (beta)
    torque_table_file = "torque_table1.mat"   # 1 for +x velocity, 2 for -x velocity
    torque_path = PROJECT_ROOT / "data" / torque_table_file
    torque_table = loadmat(torque_path, simplify_cells=True)["torque_table"]
    torque_table = np.asarray(torque_table, dtype=float)

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
    TAM.senNoiseStd = [0.0, 0.0, 0.0]
    TAM.maxOutput = 100e-6
    TAM.minOutput = -100e-6
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
    bdot.k_gain = 1e6
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
    # VELOCITY-POINTING REFERENCE
    # ------------------------------------------------------------------
    velPointModule = VelocityPointingReference()
    velPointModule.ModelTag = "VelocityPointing"
    velPointModule.transNavInMsg.subscribeTo(simpleNavModule.transOutMsg)
    scSim.AddModelToTask(simTaskName, velPointModule, 89)

    trackingError = attTrackingError.attTrackingError()
    trackingError.ModelTag = "attTrackingError"
    trackingError.attRefInMsg.subscribeTo(velPointModule.attRefOutMsg)
    trackingError.attNavInMsg.subscribeTo(simpleNavModule.attOutMsg)
    scSim.AddModelToTask(simTaskName, trackingError, 88)

    # MRP feedback control for 3-axis attitude control
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    mrpControl.K = 0.01  # Derivative gain [N*m*s] (damping)
    mrpControl.P = 0.005  # Proportional gain [N*m] (stiffness)
    mrpControl.Ki = 0.0001  # Integral gain [N*m/s] (bias rejection)
    mrpControl.integralLimit = 2.0 / 180.0 * np.pi  # 2 deg integral windup limit [rad]
    mrpControl.guidInMsg.subscribeTo(trackingError.attGuidOutMsg)

    vcMsg = messaging.VehicleConfigMsg()
    vcData = messaging.VehicleConfigMsgPayload()
    vcData.ISCPntB_B = [
        0.0015, 0.0, 0.0,
        0.0, 0.0062, 0.0,
        0.0, 0.0, 0.0062
    ]
    vcMsg.write(vcData)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
    scSim.AddModelToTask(simTaskName, mrpControl, 87)

    magMomentumCtrl = MagneticMomentumManagement()
    magMomentumCtrl.ModelTag = "MagMomentum"
    magMomentumCtrl.maxDipole = 0.016
    # Disable X-axis dipole, enable Y+Z dipoles to avoid interfering with primary control axis
    magMomentumCtrl.activeAxes = [False, True, True]  # [X, Y, Z]
    magMomentumCtrl.tamInMsg.subscribeTo(TAM.tamDataOutMsg)
    magMomentumCtrl.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
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
    velRefLog = velPointModule.attRefOutMsg.recorder()
    trackingErrLog = trackingError.attGuidOutMsg.recorder()
    mrpLog = mrpControl.cmdTorqueOutMsg.recorder()
    nadirLog = magMomentumCtrl.cmdOutMsg.recorder()
    modeLog = modeSwitch.cmdOutMsg.recorder()
    aeroLog = aeroTorque.cmdTorqueOutMsg.recorder()

    scSim.AddModelToTask(simTaskName, scLog)
    scSim.AddModelToTask(simTaskName, tamLog)
    scSim.AddModelToTask(simTaskName, mtbLog)
    scSim.AddModelToTask(simTaskName, bdotLog)
    scSim.AddModelToTask(simTaskName, velRefLog)
    scSim.AddModelToTask(simTaskName, trackingErrLog)
    scSim.AddModelToTask(simTaskName, mrpLog)
    scSim.AddModelToTask(simTaskName, nadirLog)
    scSim.AddModelToTask(simTaskName, modeLog)
    scSim.AddModelToTask(simTaskName, aeroLog)

    from Basilisk.utilities import vizSupport

    viz = vizSupport.enableUnityVisualization(
        scSim,
        simTaskName,
        scObject,
        saveFile=fileName,
        liveStream=False
    )
    if vizSupport.vizFound:
        # 1. Enable Rate Visualization (Shows omega vectors in Vizard)
        viz.settings.showAngularVelocityVectors = True
        
        # 2. Set Default Playback Speed (How fast the clock runs in Vizard)
        # 1.0 is realtime; 10.0 is 10x faster.
        viz.settings.guiPlaybackSpeed = 5.0
        
        # 3. Optional: Add a CSS-style panel to see the rates in numbers
        viz.settings.showDataPanel = True
        
        # 4. Optional: Show the Magnetometer (TAM) and Magnetorquer (MTB) vectors
        # This helps you see the B-dot and control torque directions
        viz.settings.showMagneticFieldVectors = True
        viz.settings.showMtbForceVectors = True
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

    omega_BR_B = np.array(trackingErrLog.omega_BR_B)
    omega_BR_mag = np.linalg.norm(omega_BR_B, axis=1)

    sigma_BN = np.array(scLog.sigma_BN)
    sigma_BR = np.array(trackingErrLog.sigma_BR)

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

    # Plot 0: Body rate relative to velocity-pointing reference frame
    axes[0].plot(time_min, omega_BR_mag * macros.R2D, color='tab:purple', label='|ω_BR|')
    axes[0].axhline(n_orbital * macros.R2D, color='r', linestyle='--',
                    label=f'orbital rate ≈ {n_orbital * macros.R2D:.4f} deg/s')
    axes[0].set_ylabel('|ω_BR| [deg/s]')
    axes[0].set_title('Rate relative to velocity-pointing reference frame')
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

    # Plot 2: Attitude error relative to velocity-pointing reference
    axes[2].plot(time_min, sigma_BR[:, 0], label='σ1 error')
    axes[2].plot(time_min, sigma_BR[:, 1], label='σ2 error')
    axes[2].plot(time_min, sigma_BR[:, 2], label='σ3 error')
    axes[2].set_ylabel('σ_BR [-]')
    axes[2].set_title('Attitude error relative to velocity-pointing reference')
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
        "velRefLog": velRefLog,
        "trackingErrLog": trackingErrLog,
        "mrpLog": mrpLog
    }


if __name__ == "__main__":
    run_adcs_sim()
