import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GimAlfriendSTM:
    def __init__(self, initStruct):
        # Properties initialization
        self.Phi = None
        self.X = None
        self.Ak = None
        self.Bk = None
        self.B = None
        
        # Physical parameters
        self.Req = initStruct['params'][0]
        self.mu = initStruct['params'][1]
        self.J2 = initStruct['params'][2]
        self.tol = initStruct['params'][3]
        self.safetyAltitude = initStruct['params'][4]
        
        # Maneuver parameters
        self.samples = initStruct['maneuverParams'][0]
        self.B = initStruct['maneuverParams'][1]
        
        # Time parameters
        self.t0 = initStruct['timeParams']['t0']
        self.dt = initStruct['timeParams']['dt']
        self.tf = initStruct['timeParams']['tf']
        
        # Orbit descriptions
        self.chiefOrbitDescription = initStruct['initChiefDescription']
        self.deputyOrbitDescriptionInit = initStruct['initDeputyDescription']
        self.deputyOrbitDescriptionFinal = 'Cartesian'
        
        # Initialize chief elements
        method = self.chiefOrbitDescription
        if method == 'Classical':
            self.kepElemsInit = initStruct['Elements']
            self.ChiefElemsNSMean = COE_to_Nonsingular(self.kepElemsInit, self.tol)
            self.ChiefElemsNSMean = self.ChiefElemsNSMean.reshape(-1, 1).squeeze()
            _, self.ChiefOsc = MeanToOsculatingElements(self.J2, self.ChiefElemsNSMean, 
                                                       self.Req, self.mu)
        elif method == 'Nonsingular':
            self.ChiefElemsNSMean = initStruct['Elements']
            self.kepElemsInit = Nonsingular_to_COE(self.ChiefElemsNSMean)
            _, self.ChiefOsc = MeanToOsculatingElements(self.J2, self.ChiefElemsNSMean, 
                                                       self.Req, self.mu)
        
        # Initialize deputy elements
        method = self.deputyOrbitDescriptionInit
        if method == 'Cartesian':
            self.initialCondition = initStruct['RelInitState']
        elif method == 'Relative Nonsingular':
            self.DepElemsInitNS = self.ChiefElemsNSMean + initStruct['RelInitState']
            _, self.DepOsc = MeanToOsculatingElements(self.J2, self.DepElemsInitNS, 
                                                     self.Req, self.mu)
            deltaElems = self.DepOsc - self.ChiefOsc
            self.initialCondition = SigmaMatrix(self.J2, self.ChiefOsc, self.Req, 
                                              self.mu) @ deltaElems
        elif method == 'Relative Classical':
            self.DepElemsInit = self.kepElemsInit + initStruct['RelInitState']
            self.DepElemsInitNS = COE_to_Nonsingular(self.DepElemsInit, self.tol)
            _, self.DepOsc = MeanToOsculatingElements(self.J2, self.DepElemsInitNS, 
                                                     self.Req, self.mu)
            deltaElems = self.DepOsc - self.ChiefOsc
            self.initialCondition = SigmaMatrix(self.J2, self.ChiefOsc, self.Req, 
                                              self.mu) @ deltaElems
        
        # Terminal condition
        if 'RelFinalState' not in initStruct or initStruct['RelFinalState'] is None:
            self.terminalCondition = None
        else:
            method = self.deputyOrbitDescriptionInit
            if method == 'Cartesian':
                self.terminalCondition = initStruct['RelFinalState']
            elif method == 'Relative Nonsingular':
                self.DepElemsInitNS = self.ChiefElemsNSMean + initStruct['RelFinalState']
                _, self.DepOsc = MeanToOsculatingElements(self.J2, self.DepElemsInitNS, 
                                                         self.Req, self.mu)
                deltaElems = self.DepOsc - self.ChiefOsc
                self.terminalCondition = SigmaMatrix(self.J2, self.ChiefOsc, self.Req, 
                                                   self.mu) @ deltaElems
            elif method == 'Relative Classical':
                self.DepElemsInit = self.kepElemsInit + initStruct['RelFinalState']
                self.DepElemsInitNS = COE_to_Nonsingular(self.DepElemsInit, self.tol)
                _, self.DepOsc = MeanToOsculatingElements(self.J2, self.DepElemsInitNS, 
                                                         self.Req, self.mu)
                deltaElems = self.DepOsc - self.ChiefOsc
                self.terminalCondition = SigmaMatrix(self.J2, self.ChiefOsc, self.Req, 
                                                   self.mu) @ deltaElems
        
        self.makeTimeVector()
    
    def makeTimeVector(self):
        method = self.chiefOrbitDescription
        if method == 'Classical':
            n = np.sqrt(self.mu / self.kepElemsInit[0]**3)
            if self.t0 is None:
                self.t0 = 0
            self.period = 2 * np.pi / n
            self.time = np.arange(self.t0, self.tf + self.dt, self.dt)
        elif method == 'Nonsingular':
            n = np.sqrt(self.mu / self.ChiefElemsNSMean[0]**3)
            if self.t0 is None:
                self.t0 = 0
            self.period = 2 * np.pi / n
            self.time = np.arange(self.t0, self.tf + self.dt, self.dt)
    
    def propagateModel(self, t1=None, t2=None):
        if t1 is None:
            t1 = self.t0
        if t2 is None:
            t2 = self.tf
        t = np.arange(t1, t2 + self.dt, self.dt)
        self.Phi = PHI_GA_STM(t, self.J2, self.ChiefOsc, self.ChiefElemsNSMean, 
                             self.Req, self.mu, self.tol)
    
    def propagateState(self):
        self.propagateModel()
        self.X = np.zeros((6, len(self.time)))
        for ii in range(len(self.time)):
            self.X[:, ii] = self.Phi[:, :, ii] @ self.initialCondition
    
    def makeDiscreteMatrices(self):
        t = self.time
        if t is None or len(t) < 2:
            raise ValueError("Time vector must contain at least two samples to build discrete matrices")

        N = len(t) - 1  # number of propagation intervals
        self.Ak = np.zeros((6, 6, N))
        self.Bk = np.zeros((self.B.shape[0], self.B.shape[1], N))

        for k in range(N):
            t_start = t[k]
            t_end = t[k + 1]
            Tk = np.linspace(t_start, t_end, self.samples)
            if Tk.shape[0] < 2:
                raise ValueError("Discrete matrix integration requires at least two sample points")

            PhiJ2k = PHI_GA_STM(Tk, self.J2, self.ChiefOsc, self.ChiefElemsNSMean,
                               self.Req, self.mu, self.tol)
            phiend = PhiJ2k[:, :, -1]
            self.Ak[:, :, k] = phiend

            PhiBk = np.zeros((6, self.B.shape[1], len(Tk)))
            for ii in range(len(Tk)):
                PhiBk[:, :, ii] = phiend @ np.linalg.inv(PhiJ2k[:, :, ii]) @ self.B

            Bd = np.zeros_like(self.B)
            for ii in range(len(Tk) - 1):
                dt_segment = Tk[ii + 1] - Tk[ii]
                Bd = Bd + 0.5 * dt_segment * (PhiBk[:, :, ii + 1] + PhiBk[:, :, ii])

            self.Bk[:, :, k] = Bd
    
    def plotOrbit(self):
        x = 1e-3 * self.X[0, :]
        y = 1e-3 * self.X[2, :]
        z = 1e-3 * self.X[4, :]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'k', linewidth=2)
        ax.grid(True)
        ax.set_title('$J_2$-Perturbed Relative Motion')
        ax.set_xlabel('Radial, $x$, km')
        ax.set_ylabel('In-track, $y$, km')
        ax.set_zlabel('Cross-track, $z$, km')
        plt.tight_layout()
        plt.show()


def PHI_GA_STM(t, J2, ChiefOsc, ChiefNSMean, Req, mu, tol):
    """
    Function computes the Gim-Alfriend State Transition Matrix for the
    initial chief orbital elements over the time span t
    """
    t0 = t[0]
    SigmaInv = SigmaInverseMatrix(J2, ChiefOsc, Req, mu, tol)
    DJ20, _ = MeanToOsculatingElements(J2, ChiefNSMean, Req, mu)
    invDJ20 = np.linalg.inv(DJ20)
    
    PhiJ2 = np.zeros((6, 6, len(t)))
    for ii in range(len(t)):
        tii = np.array([t0, t[ii]])
        PhiBar, ChiefMeanPropNS = MeanElemsSTM(J2, tii, ChiefNSMean, Req, mu, tol)
        DJ2, ChiefOsc = MeanToOsculatingElements(J2, ChiefMeanPropNS, Req, mu)
        Sigma = SigmaMatrix(J2, ChiefOsc, Req, mu)
        PhiJ2[:, :, ii] = Sigma @ DJ2 @ PhiBar @ invDJ20 @ SigmaInv
    
    return PhiJ2


def MeanElemsSTM(J2, t, ICSc, Re, mu, tol):
    """
    Calculation of the state transition matrix
    for the mean non-singular variables with perturbation by J2
    """
    gamma = 3 * J2 * Re**2
    
    t0 = t[0]
    t = t[1]
    
    a0 = ICSc[0]
    argLat0 = ICSc[1]
    i0 = ICSc[2]
    q10 = ICSc[3]
    q20 = ICSc[4]
    RAAN0 = ICSc[5]
    
    n0 = np.sqrt(mu / a0**3)
    p0 = a0 * (1 - q10**2 - q20**2)
    R0 = p0 / (1 + q10 * np.cos(argLat0) + q20 * np.sin(argLat0))
    Vr0 = np.sqrt(mu / p0) * (q10 * np.sin(argLat0) - q20 * np.cos(argLat0))
    Vt0 = np.sqrt(mu / p0) * (1 + q10 * np.cos(argLat0) + q20 * np.sin(argLat0))
    eta0 = np.sqrt(1 - q10**2 - q20**2)
    
    lambda0 = theta2lam(a0, argLat0, q10, q20)
    
    # Secular Variations by J2
    aDot = 0
    incDot = 0
    argPerDot = gamma * 0.25 * (n0 / p0**2) * (5 * np.cos(i0)**2 - 1)
    
    sDot = np.sin(argPerDot * (t - t0))
    cDot = np.cos(argPerDot * (t - t0))
    
    lamDot = n0 + gamma * 0.25 * (n0 / p0**2) * ((5 + 3 * eta0) * np.cos(i0)**2 - (1 + eta0))
    RAANDot = -gamma * 0.5 * (n0 / p0**2) * np.cos(i0)
    
    # Perturbed orbital elements
    a = a0 + aDot * (t - t0)
    i = i0 + incDot * (t - t0)
    Omega = RAAN0 + RAANDot * (t - t0)
    q1 = q10 * np.cos(argPerDot * (t - t0)) - q20 * np.sin(argPerDot * (t - t0))
    q2 = q10 * np.sin(argPerDot * (t - t0)) + q20 * np.cos(argPerDot * (t - t0))
    
    lambda_val = lambda0 + lamDot * (t - t0)
    
    theta, _ = lam2theta(lambda_val, q1, q2, tol)
    
    n = np.sqrt(mu / a**3)
    p = a * (1 - q1**2 - q2**2)
    R = p / (1 + q1 * np.cos(theta) + q2 * np.sin(theta))
    Vr = np.sqrt(mu / p) * (q1 * np.sin(theta) - q2 * np.cos(theta))
    Vt = np.sqrt(mu / p) * (1 + q1 * np.cos(theta) + q2 * np.sin(theta))
    eta = np.sqrt(1 - q1**2 - q2**2)
    
    # Partial derivatives
    G_theta = n * R / Vt
    G_theta0 = -n0 * R0 / Vt0
    G_q1 = (q1 * Vr) / (eta * Vt) + q2 / (eta * (1 + eta)) - eta * R * (a + R) * (q2 + np.sin(theta)) / (p**2)
    G_q10 = -(q10 * Vr0) / (eta0 * Vt0) - q20 / (eta0 * (1 + eta0)) + eta0 * R0 * (a0 + R0) * (q20 + np.sin(argLat0)) / (p0**2)
    G_q2 = (q2 * Vr) / (eta * Vt) - q1 / (eta * (1 + eta)) + eta * R * (a + R) * (q1 + np.cos(theta)) / (p**2)
    G_q20 = -(q20 * Vr0) / (eta0 * Vt0) + q10 / (eta0 * (1 + eta0)) - eta0 * R0 * (a0 + R0) * (q10 + np.cos(argLat0)) / (p0**2)
    K = 1 + G_q1 * (q10 * sDot + q20 * cDot) - G_q2 * (q10 * cDot - q20 * sDot)
    
    # Transformation Matrix A
    phi11 = 1
    phi12 = 0
    phi13 = 0
    phi14 = 0
    phi15 = 0
    phi16 = 0
    
    phi21 = -((t - t0) / G_theta) * ((3 * n0 / (2 * a0)) + 
            (7 * gamma / 8) * (n0 / (a0 * p0**2)) * (eta0 * (3 * np.cos(i0)**2 - 1) + K * (5 * np.cos(i0)**2 - 1)))
    phi22 = -(G_theta0 / G_theta)
    phi23 = -((t - t0) / G_theta) * (gamma / 2) * (n0 * np.sin(i0) * np.cos(i0) / p0**2) * (3 * eta0 + 5 * K)
    phi24 = -((G_q10 + cDot * G_q1 + sDot * G_q2) / G_theta) + \
            ((t - t0) / G_theta) * (gamma / 4) * (n0 * a0 * q10 / p0**3) * (3 * eta0 * (3 * np.cos(i0)**2 - 1) + 4 * K * (5 * np.cos(i0)**2 - 1))
    phi25 = -((G_q20 - sDot * G_q1 + cDot * G_q2) / G_theta) + \
            ((t - t0) / G_theta) * (gamma / 4) * (n0 * a0 * q20 / p0**3) * (3 * eta0 * (3 * np.cos(i0)**2 - 1) + 4 * K * (5 * np.cos(i0)**2 - 1))
    phi26 = 0
    
    phi31 = 0
    phi32 = 0
    phi33 = 1
    phi34 = 0
    phi35 = 0
    phi36 = 0
    
    phi41 = (7 * gamma / 8) * (n0 * (q10 * sDot + q20 * cDot) * (5 * np.cos(i0)**2 - 1) / (a0 * p0**2)) * (t - t0)
    phi42 = 0
    phi43 = (5 * gamma / 2) * (n0 * (q10 * sDot + q20 * cDot) * (np.sin(i0) * np.cos(i0)) / p0**2) * (t - t0)
    phi44 = cDot - gamma * (n0 * a0 * q10 * (q10 * sDot + q20 * cDot) * (5 * np.cos(i0)**2 - 1) / p0**3) * (t - t0)
    phi45 = -sDot - gamma * (n0 * a0 * q20 * (q10 * sDot + q20 * cDot) * (5 * np.cos(i0)**2 - 1) / p0**3) * (t - t0)
    phi46 = 0
    
    phi51 = -(7 * gamma / 8) * (n0 * (q10 * cDot - q20 * sDot) * (5 * np.cos(i0)**2 - 1) / (a0 * p0**2)) * (t - t0)
    phi52 = 0
    phi53 = -(5 * gamma / 2) * (n0 * (q10 * cDot - q20 * sDot) * (np.sin(i0) * np.cos(i0)) / p0**2) * (t - t0)
    phi54 = sDot + gamma * (n0 * a0 * q10 * (q10 * cDot - q20 * sDot) * (5 * np.cos(i0)**2 - 1) / p0**3) * (t - t0)
    phi55 = cDot + gamma * (n0 * a0 * q20 * (q10 * cDot - q20 * sDot) * (5 * np.cos(i0)**2 - 1) / p0**3) * (t - t0)
    phi56 = 0
    
    phi61 = (7 * gamma / 4) * (n0 * np.cos(i0) / (a0 * p0**2)) * (t - t0)
    phi62 = 0
    phi63 = (gamma / 2) * (n0 * np.sin(i0) / p0**2) * (t - t0)
    phi64 = -(2 * gamma) * (n0 * a0 * q10 * np.cos(i0) / p0**3) * (t - t0)
    phi65 = -(2 * gamma) * (n0 * a0 * q20 * np.cos(i0) / p0**3) * (t - t0)
    phi66 = 1
    
    # state transition matrix, phi_J2
    phi_J2 = np.array([
        [phi11, phi12, phi13, phi14, phi15, phi16],
        [phi21, phi22, phi23, phi24, phi25, phi26],
        [phi31, phi32, phi33, phi34, phi35, phi36],
        [phi41, phi42, phi43, phi44, phi45, phi46],
        [phi51, phi52, phi53, phi54, phi55, phi56],
        [phi61, phi62, phi63, phi64, phi65, phi66]
    ])
    
    # Orbital elements
    cond_c = np.array([a, theta, i, q1, q2, Omega])
    
    return phi_J2, cond_c


def SigmaMatrix(J2, elems, Re, mu):
    """
    System_matrix, Sigma in osculating element with perturbation by J2
    """
    gamma = 3 * J2 * Re**2
    a = elems[0]
    argLat = elems[1]
    inc = elems[2]
    q1 = elems[3]
    q2 = elems[4]
    
    # Evaluations from the inputs
    p = a * (1 - q1**2 - q2**2)
    R = p / (1 + q1 * np.cos(argLat) + q2 * np.sin(argLat))
    Vr = np.sqrt(mu / p) * (q1 * np.sin(argLat) - q2 * np.cos(argLat))
    Vt = np.sqrt(mu / p) * (1 + q1 * np.cos(argLat) + q2 * np.sin(argLat))
    
    # Transformation Matrix A
    A11 = R / a
    A12 = R * Vr / Vt
    A13 = 0
    A14 = -(2 * a * R * q1 / p) - (R**2 / p) * np.cos(argLat)
    A15 = -(2 * a * R * q2 / p) - (R**2 / p) * np.sin(argLat)
    A16 = 0
    
    A21 = -0.5 * Vr / a
    A22 = np.sqrt(mu / p) * ((p / R) - 1)
    A23 = 0
    A24 = (Vr * a * q1 / p) + np.sqrt(mu / p) * np.sin(argLat)
    A25 = (Vr * a * q2 / p) - np.sqrt(mu / p) * np.cos(argLat)
    A26 = 0
    
    A31 = 0
    A32 = R
    A33 = 0
    A34 = 0
    A35 = 0
    A36 = R * np.cos(inc)
    
    A41 = -1.5 * Vt / a
    A42 = -Vr
    A43 = 0
    A44 = (3 * Vt * a * q1 / p) + 2 * np.sqrt(mu / p) * np.cos(argLat)
    A45 = (3 * Vt * a * q2 / p) + 2 * np.sqrt(mu / p) * np.sin(argLat)
    A46 = Vr * np.cos(inc)
    
    A51 = 0
    A52 = 0
    A53 = R * np.sin(argLat)
    A54 = 0
    A55 = 0
    A56 = -R * np.cos(argLat) * np.sin(inc)
    
    A61 = 0
    A62 = 0
    A63 = Vt * np.cos(argLat) + Vr * np.sin(argLat)
    A64 = 0
    A65 = 0
    A66 = (Vt * np.sin(argLat) - Vr * np.cos(argLat)) * np.sin(inc)
    
    A = np.array([
        [A11, A12, A13, A14, A15, A16],
        [A21, A22, A23, A24, A25, A26],
        [A31, A32, A33, A34, A35, A36],
        [A41, A42, A43, A44, A45, A46],
        [A51, A52, A53, A54, A55, A56],
        [A61, A62, A63, A64, A65, A66]
    ])
    
    # Transformation Matrix B
    B11 = 0
    B12 = 0
    B13 = 0
    B14 = 0
    B15 = 0
    B16 = 0
    
    B21 = 0
    B22 = 0
    B23 = 0
    B24 = 0
    B25 = 0
    B26 = 0
    
    B31 = 0
    B32 = 0
    B33 = 0
    B34 = 0
    B35 = 0
    B36 = 0
    
    B41 = 0
    B42 = 0
    B43 = -Vt * np.sin(inc) * np.cos(inc) * np.sin(argLat)**2 / (p * R)
    B44 = 0
    B45 = 0
    B46 = Vt * np.sin(inc)**2 * np.cos(inc) * np.sin(argLat) * np.cos(argLat) / (p * R)
    
    B51 = 0
    B52 = 0
    B53 = 0
    B54 = 0
    B55 = 0
    B56 = 0
    
    B61 = 0
    B62 = Vt * np.sin(inc) * np.cos(inc) * np.sin(argLat) / (p * R)
    B63 = 0
    B64 = 0
    B65 = 0
    B66 = Vt * np.sin(inc) * np.cos(inc)**2 * np.sin(argLat) / (p * R)
    
    B = np.array([
        [B11, B12, B13, B14, B15, B16],
        [B21, B22, B23, B24, B25, B26],
        [B31, B32, B33, B34, B35, B36],
        [B41, B42, B43, B44, B45, B46],
        [B51, B52, B53, B54, B55, B56],
        [B61, B62, B63, B64, B65, B66]
    ])
    
    # System Matrix
    Sigma = A + gamma * B
    
    return Sigma


def SigmaInverseMatrix(J2, elems, Re, mu, tol):
    """
    Calculation of the inverse of system_matrix at t0
    """
    gamma = 3 * J2 * Re**2
    a = elems[0]
    argLat = elems[1]
    inc = elems[2]
    q1 = elems[3]
    q2 = elems[4]
    RAAN = elems[5]
    
    Hamiltonian = -mu / (2 * a)
    p = a * (1 - q1**2 - q2**2)
    R = p / (1 + q1 * np.cos(argLat) + q2 * np.sin(argLat))
    Vr = np.sqrt(mu / p) * (q1 * np.sin(argLat) - q2 * np.cos(argLat))
    Vt = np.sqrt(mu / p) * (1 + q1 * np.cos(argLat) + q2 * np.sin(argLat))
    
    # New non-singular elements
    q1tilde = q1 * np.cos(RAAN) - q2 * np.sin(RAAN)
    q2tilde = q1 * np.sin(RAAN) + q2 * np.cos(RAAN)
    p1 = np.tan(inc / 2) * np.cos(RAAN)
    p2 = np.tan(inc / 2) * np.sin(RAAN)
    
    if (p1 == 0) and (p2 == 0):
        p1p2 = p1**2 + p2**2 + tol
    else:
        p1p2 = p1**2 + p2**2
    
    # Inverse Matrix of T for non-singular elements
    InvT11 = 1
    InvT12 = 0
    InvT13 = 0
    InvT14 = 0
    InvT15 = 0
    InvT16 = 0
    
    InvT21 = 0
    InvT22 = 1
    InvT23 = 0
    InvT24 = 0
    InvT25 = p2 / p1p2
    InvT26 = -p1 / p1p2
    
    InvT31 = 0
    InvT32 = 0
    InvT33 = 0
    InvT34 = 0
    InvT35 = 2 * p1 / (np.sqrt(p1p2) * (1 + p1p2))
    InvT36 = 2 * p2 / (np.sqrt(p1p2) * (1 + p1p2))
    
    InvT41 = 0
    InvT42 = 0
    InvT43 = p1 / np.sqrt(p1p2)
    InvT44 = p2 / np.sqrt(p1p2)
    InvT45 = -p2 * (p1 * q2tilde - p2 * q1tilde) / (p1p2**(3/2))
    InvT46 = p1 * (p1 * q2tilde - p2 * q1tilde) / (p1p2**(3/2))
    
    InvT51 = 0
    InvT52 = 0
    InvT53 = -p2 / np.sqrt(p1p2)
    InvT54 = p1 / np.sqrt(p1p2)
    InvT55 = p2 * (p1 * q1tilde + p2 * q2tilde) / (p1p2**(3/2))
    InvT56 = -p1 * (p1 * q1tilde + p2 * q2tilde) / (p1p2**(3/2))
    
    InvT61 = 0
    InvT62 = 0
    InvT63 = 0
    InvT64 = 0
    InvT65 = -p2 / p1p2
    InvT66 = p1 / p1p2
    
    InvT = np.array([
        [InvT11, InvT12, InvT13, InvT14, InvT15, InvT16],
        [InvT21, InvT22, InvT23, InvT24, InvT25, InvT26],
        [InvT31, InvT32, InvT33, InvT34, InvT35, InvT36],
        [InvT41, InvT42, InvT43, InvT44, InvT45, InvT46],
        [InvT51, InvT52, InvT53, InvT54, InvT55, InvT56],
        [InvT61, InvT62, InvT63, InvT64, InvT65, InvT66]
    ])
    
    # T*Inverse(A)
    InvTA11 = (1 / (R * Hamiltonian)) * ((mu / R) * (3 * a - 2 * R) - a * (2 * Vr**2 + 3 * Vt**2))
    InvTA12 = -a * Vr / Hamiltonian
    InvTA13 = -(Vr / Hamiltonian) * ((Vt / p) * (2 * a - R) - (a / (R * Vt)) * (Vr**2 + 2 * Vt**2))
    InvTA14 = (R / Hamiltonian) * ((Vt / p) * (2 * a - R) - (a / (R * Vt)) * (Vr**2 + 2 * Vt**2))
    InvTA15 = 0
    InvTA16 = 0
    
    InvTA21 = 0
    InvTA22 = 0
    InvTA23 = 1 / R
    InvTA24 = 0
    InvTA25 = -((Vr * np.sin(argLat) + Vt * np.cos(argLat)) / (R * Vt)) * (np.sin(inc) / (1 + np.cos(inc)))
    InvTA26 = (np.sin(argLat) / Vt) * (np.sin(inc) / (1 + np.cos(inc)))
    
    InvTA31 = p * (np.cos(RAAN) * (2 * Vr * np.sin(argLat) + 3 * Vt * np.cos(argLat)) + 
                  np.sin(RAAN) * (2 * Vr * np.cos(argLat) - 3 * Vt * np.sin(argLat))) / (R**2 * Vt)
    InvTA32 = np.sqrt(p / mu) * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat))
    InvTA33 = (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) * ((1 / R) - (Vr**2 + Vt**2) / mu) - \
              (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) * (Vr * Vt / mu)
    InvTA34 = 2 * np.sqrt(p / mu) * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) + \
              (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) * (R * Vr / mu)
    InvTA35 = ((q1 * np.sin(RAAN) + q2 * np.cos(RAAN)) * (q1 + np.cos(argLat)) * np.sin(inc)) / (p * (1 + np.cos(inc)))
    InvTA36 = -((q1 * np.sin(RAAN) + q2 * np.cos(RAAN)) * np.sin(argLat) * np.sin(inc)) / (Vt * (1 + np.cos(inc)))
    
    InvTA41 = p * (np.sin(RAAN) * (2 * Vr * np.sin(argLat) + 3 * Vt * np.cos(argLat)) - 
                  np.cos(RAAN) * (2 * Vr * np.cos(argLat) - 3 * Vt * np.sin(argLat))) / (R**2 * Vt)
    InvTA42 = -np.sqrt(p / mu) * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat))
    InvTA43 = -(np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) * ((1 / R) - (Vr**2 + Vt**2) / mu) - \
               (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) * (Vr * Vt / mu)
    InvTA44 = 2 * np.sqrt(p / mu) * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) - \
              (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) * (R * Vr / mu)
    InvTA45 = -((q1 * np.cos(RAAN) - q2 * np.sin(RAAN)) * (q1 + np.cos(argLat)) * np.sin(inc)) / (p * (1 + np.cos(inc)))
    InvTA46 = ((q1 * np.cos(RAAN) - q2 * np.sin(RAAN)) * np.sin(argLat) * np.sin(inc)) / (Vt * (1 + np.cos(inc)))
    
    InvTA51 = 0
    InvTA52 = 0
    InvTA53 = 0
    InvTA54 = 0
    InvTA55 = -(np.cos(RAAN) * (Vr * np.cos(argLat) - Vt * np.sin(argLat)) - 
                np.sin(RAAN) * (Vr * np.sin(argLat) + Vt * np.cos(argLat))) / (R * Vt * (1 + np.cos(inc)))
    InvTA56 = (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) / (Vt * (1 + np.cos(inc)))
    
    InvTA61 = 0
    InvTA62 = 0
    InvTA63 = 0
    InvTA64 = 0
    InvTA65 = -(np.sin(RAAN) * (Vr * np.cos(argLat) - Vt * np.sin(argLat)) + 
                np.cos(RAAN) * (Vr * np.sin(argLat) + Vt * np.cos(argLat))) / (R * Vt * (1 + np.cos(inc)))
    InvTA66 = (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) / (Vt * (1 + np.cos(inc)))
    
    InvTA = np.array([
        [InvTA11, InvTA12, InvTA13, InvTA14, InvTA15, InvTA16],
        [InvTA21, InvTA22, InvTA23, InvTA24, InvTA25, InvTA26],
        [InvTA31, InvTA32, InvTA33, InvTA34, InvTA35, InvTA36],
        [InvTA41, InvTA42, InvTA43, InvTA44, InvTA45, InvTA46],
        [InvTA51, InvTA52, InvTA53, InvTA54, InvTA55, InvTA56],
        [InvTA61, InvTA62, InvTA63, InvTA64, InvTA65, InvTA66]
    ])
    
    # The rest of inverse Matrix
    InvTD11 = 0
    InvTD12 = 0
    InvTD13 = 0
    InvTD14 = 0
    InvTD15 = (np.sin(inc) * np.cos(inc) * np.sin(argLat) / (Hamiltonian * p * R**2)) * \
              ((mu / R) * (2 * a - R) - a * (Vr**2 + 2 * Vt**2))
    InvTD16 = 0
    
    InvTD21 = 0
    InvTD22 = 0
    InvTD23 = -np.cos(inc) * (1 - np.cos(inc)) * np.sin(argLat)**2 / (p * R**2)
    InvTD24 = 0
    InvTD25 = 0
    InvTD26 = 0
    
    InvTD31 = 0
    InvTD32 = 0
    InvTD33 = (np.cos(inc)**2 * np.sin(argLat)**2 / (Vt**2 * R**3)) * \
              (Vr * Vt * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) - 
               Vt**2 * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat))) + \
              (np.cos(inc) * np.sin(argLat)**2 / (p * R**2)) * \
              (np.cos(inc) * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) + 
               (q1 * np.sin(RAAN) + q2 * np.cos(RAAN)))
    InvTD34 = 0
    InvTD35 = (np.sin(inc) * np.cos(inc) * np.sin(argLat) / (Vt * R**3)) * \
              (Vr * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) + 
               2 * Vt * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)))
    InvTD36 = 0
    
    InvTD41 = 0
    InvTD42 = 0
    InvTD43 = (np.cos(inc)**2 * np.sin(argLat)**2 / (Vt**2 * R**3)) * \
              (Vr * Vt * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)) + 
               Vt**2 * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat))) - \
              (np.cos(inc) * np.sin(argLat)**2 / (p * R**2)) * \
              (np.cos(inc) * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) + 
               (q1 * np.cos(RAAN) - q2 * np.sin(RAAN)))
    InvTD44 = 0
    InvTD45 = -(np.sin(inc) * np.cos(inc) * np.sin(argLat) / (Vt * R**3)) * \
               (Vr * (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat)) - 
                2 * Vt * (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat)))
    InvTD46 = 0
    
    InvTD51 = 0
    InvTD52 = 0
    InvTD53 = -(np.sin(inc) * np.cos(inc) * np.sin(argLat) / ((1 + np.cos(inc)) * p * R**2)) * \
               (np.cos(RAAN) * np.cos(argLat) - np.sin(RAAN) * np.sin(argLat))
    InvTD54 = 0
    InvTD55 = 0
    InvTD56 = 0
    
    InvTD61 = 0
    InvTD62 = 0
    InvTD63 = -(np.sin(inc) * np.cos(inc) * np.sin(argLat) / ((1 + np.cos(inc)) * p * R**2)) * \
               (np.sin(RAAN) * np.cos(argLat) + np.cos(RAAN) * np.sin(argLat))
    InvTD64 = 0
    InvTD65 = 0
    InvTD66 = 0
    
    InvTD = np.array([
        [InvTD11, InvTD12, InvTD13, InvTD14, InvTD15, InvTD16],
        [InvTD21, InvTD22, InvTD23, InvTD24, InvTD25, InvTD26],
        [InvTD31, InvTD32, InvTD33, InvTD34, InvTD35, InvTD36],
        [InvTD41, InvTD42, InvTD43, InvTD44, InvTD45, InvTD46],
        [InvTD51, InvTD52, InvTD53, InvTD54, InvTD55, InvTD56],
        [InvTD61, InvTD62, InvTD63, InvTD64, InvTD65, InvTD66]
    ])
    
    # Inverse of System Matrix
    SigmaInverse = InvT @ (InvTA + gamma * InvTD)
    
    return SigmaInverse


def MeanToOsculatingElements(J2, meanElems, Re, mu):
    """
    Transformation matrix D_J2 in closed form
    between mean and osculating new set of elements
    with the perturbation by only J2
    """
    gamma = -J2 * Re**2
    a = meanElems[0]
    argLat = meanElems[1]
    inc = meanElems[2]
    q1 = meanElems[3]
    q2 = meanElems[4]
    RAAN = meanElems[5]
    
    si = np.sin(inc)
    ci = np.cos(inc)
    s2i = np.sin(2 * inc)
    c2i = np.cos(2 * inc)
    sth = np.sin(argLat)
    cth = np.cos(argLat)
    s2th = np.sin(2 * argLat)
    c2th = np.cos(2 * argLat)
    s3th = np.sin(3 * argLat)
    c3th = np.cos(3 * argLat)
    s4th = np.sin(4 * argLat)
    c4th = np.cos(4 * argLat)
    s5th = np.sin(5 * argLat)
    c5th = np.cos(5 * argLat)
    
    p = a * (1 - (q1**2 + q2**2))
    R = p / (1 + q1 * cth + q2 * sth)
    Vr = np.sqrt(mu / p) * (q1 * sth - q2 * cth)
    Vt = np.sqrt(mu / p) * (1 + q1 * cth + q2 * sth)
    
    Ttheta = 1 / (1 - 5 * ci**2)
    eta = np.sqrt(1 - (q1**2 + q2**2))
    eps1 = np.sqrt(q1**2 + q2**2)
    eps2 = q1 * cth + q2 * sth
    eps3 = q1 * sth - q2 * cth
    lambda_val = theta2lam(a, argLat, q1, q2)
    argLatLam = argLat - lambda_val
    lam_q1 = (q1 * Vr) / (eta * Vt) + q2 / (eta * (1 + eta)) - eta * R * (a + R) * (q2 + np.sin(argLat)) / (p**2)
    lam_q2 = (q2 * Vr) / (eta * Vt) - q1 / (eta * (1 + eta)) + eta * R * (a + R) * (q1 + np.cos(argLat)) / (p**2)
    
    # Long period part, D_lp
    lamLp = (si**2 / (8 * a**2 * eta**2 * (1 + eta))) * (1 - 10 * Ttheta * ci**2) * q1 * q2 + \
            (q1 * q2 / (16 * a**2 * eta**4)) * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6)
    
    aLp = 0
    argLatLp = lamLp - (si**2 / (16 * a**2 * eta**4)) * (1 - 10 * Ttheta * ci**2) * \
               ((3 + 2 * eta**2 / (1 + eta)) * q1 * q2 + 2 * q1 * sth + 2 * q2 * cth + 0.5 * (q1**2 + q2**2) * s2th)
    incLp = (s2i / (32 * a**2 * eta**4)) * (1 - 10 * Ttheta * ci**2) * (q1**2 - q2**2)
    q1Lp = -(q1 * si**2 / (16 * a**2 * eta**2)) * (1 - 10 * Ttheta * ci**2) - \
           (q1 * q2**2 / (16 * a**2 * eta**4)) * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6)
    q2Lp = (q2 * si**2 / (16 * a**2 * eta**2)) * (1 - 10 * Ttheta * ci**2) + \
           (q1**2 * q2 / (16 * a**2 * eta**4)) * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6)
    RAANLp = (q1 * q2 * ci / (8 * a**2 * eta**4)) * (11 + 80 * Ttheta * ci**2 + 200 * Ttheta**2 * ci**4)
    
    # Create DLP matrix
    DLP11 = -(1 / a) * aLp
    DLP12 = 0
    DLP13 = 0
    DLP14 = 0
    DLP15 = 0
    DLP16 = 0
    
    DLP21 = -(2 / a) * argLatLp
    DLP22 = -(si**2 / (16 * a**2 * eta**4)) * (1 - 10 * Ttheta * ci**2) * (2 * (q1 * cth - q2 * sth) + eps1 * c2th)
    DLP23 = (s2i / (16 * a**2 * eta**4)) * (5 * q1 * q2 * (11 + 112 * Ttheta * ci**2 + 520 * Ttheta**2 * ci**4 + 800 * Ttheta**3 * ci**6) - 
            (2 * q1 * q2 + (2 + eps2) * (q1 * sth + q2 * cth)) * ((1 - 10 * Ttheta * ci**2) + 10 * Ttheta * si**2 * (1 + 5 * Ttheta * ci**2)))
    DLP24 = (1 / (16 * a**2 * eta**6)) * ((eta**2 + 4 * q1**2) * (q2 * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6) - 
            si**2 * (1 - 10 * Ttheta * ci**2) * (3 * q2 + 2 * sth)) - 2 * si**2 * (1 - 10 * Ttheta * ci**2) * (4 * q2 + sth * (1 + eps1)) * q1 * cth)
    DLP25 = (1 / (16 * a**2 * eta**6)) * ((eta**2 + 4 * q2**2) * (q1 * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6) - 
            si**2 * (1 - 10 * Ttheta * ci**2) * (3 * q1 + 2 * cth)) - 2 * si**2 * (1 - 10 * Ttheta * ci**2) * (4 * q1 + cth * (1 + eps1)) * q2 * sth)
    DLP26 = 0
    
    DLP31 = -(2 / a) * incLp
    DLP32 = 0
    DLP33 = ((q1**2 - q2**2) / (16 * a**2 * eta**4)) * (c2i * (1 - 10 * Ttheta * ci**2) + 5 * Ttheta * s2i**2 * (1 + 5 * Ttheta * ci**2))
    DLP34 = (q1 * s2i / (16 * a**2 * eta**6)) * (1 - 10 * Ttheta * ci**2) * (eta**2 + 2 * (q1**2 - q2**2))
    DLP35 = -(q2 * s2i / (16 * a**2 * eta**6)) * (1 - 10 * Ttheta * ci**2) * (eta**2 - 2 * (q1**2 - q2**2))
    DLP36 = 0
    
    DLP41 = -(2 / a) * q1Lp
    DLP42 = 0
    DLP43 = -(q1 * s2i / (16 * a**2 * eta**4)) * (eta**2 * ((1 - 10 * Ttheta * ci**2) + 10 * Ttheta * si**2 * (1 + 5 * Ttheta * ci**2)) + 
            5 * q2**2 * (11 + 112 * Ttheta * ci**2 + 520 * Ttheta**2 * ci**4 + 800 * Ttheta**3 * ci**6))
    DLP44 = -(1 / (16 * a**2 * eta**6)) * (eta**2 * si**2 * (1 - 10 * Ttheta * ci**2) * (eta**2 + 2 * q1**2) + 
            q2**2 * (eta**2 + 4 * q1**2) * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6))
    DLP45 = -(q1 * q2 / (8 * a**2 * eta**6)) * (eta**2 * si**2 * (1 - 10 * Ttheta * ci**2) + 
            (eta**2 + 2 * q2**2) * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6))
    DLP46 = 0
    
    DLP51 = -(2 / a) * q2Lp
    DLP52 = 0
    DLP53 = (q2 * s2i / (16 * a**2 * eta**4)) * (eta**2 * (1 - 10 * Ttheta * ci**2) + 10 * Ttheta * eta**2 * si**2 * (1 + 5 * Ttheta * ci**2) + 
            5 * q1**2 * (11 + 112 * Ttheta * ci**2 + 520 * Ttheta**2 * ci**4 + 800 * Ttheta**3 * ci**6))
    DLP54 = (q1 * q2 / (8 * a**2 * eta**6)) * (eta**2 * si**2 * (1 - 10 * Ttheta * ci**2) + 
            (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6) * (eta**2 + 2 * q1**2))
    DLP55 = (1 / (16 * a**2 * eta**6)) * (eta**2 * si**2 * (1 - 10 * Ttheta * ci**2) * (eta**2 + 2 * q2**2) + 
            q1**2 * (3 - 55 * ci**2 - 280 * Ttheta * ci**4 - 400 * Ttheta**2 * ci**6) * (eta**2 + 4 * q2**2))
    DLP56 = 0
    
    DLP61 = -(2 / a) * RAANLp
    DLP62 = 0
    DLP63 = -(q1 * q2 * si / (8 * a**2 * eta**4)) * ((11 + 80 * Ttheta * ci**2 + 200 * Ttheta**2 * ci**4) + 
            160 * Ttheta * ci**2 * (1 + 5 * Ttheta * ci**2)**2)
    DLP64 = (q2 * ci / (8 * a**2 * eta**6)) * (eta**2 + 4 * q1**2) * (11 + 80 * Ttheta * ci**2 + 200 * Ttheta**2 * ci**4)
    DLP65 = (q1 * ci / (8 * a**2 * eta**6)) * (eta**2 + 4 * q2**2) * (11 + 80 * Ttheta * ci**2 + 200 * Ttheta**2 * ci**4)
    DLP66 = 0
    
    DLP = np.array([
        [DLP11, DLP12, DLP13, DLP14, DLP15, DLP16],
        [DLP21, DLP22, DLP23, DLP24, DLP25, DLP26],
        [DLP31, DLP32, DLP33, DLP34, DLP35, DLP36],
        [DLP41, DLP42, DLP43, DLP44, DLP45, DLP46],
        [DLP51, DLP52, DLP53, DLP54, DLP55, DLP56],
        [DLP61, DLP62, DLP63, DLP64, DLP65, DLP66]
    ])
    
    # First short period part, D_sp1
    lamSp1 = (eps3 * (1 - 3 * ci**2) / (4 * a**2 * eta**4 * (1 + eta))) * ((1 + eps2)**2 + (1 + eps2) + eta**2) + \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * (argLatLam + eps3)
    
    aSp1 = ((1 - 3 * ci**2) / (2 * a * eta**6)) * ((1 + eps2)**3 - eta**3)
    argLatSp1 = lamSp1 - (eps3 * (1 - 3 * ci**2) / (4 * a**2 * eta**4 * (1 + eta))) * ((1 + eps2)**2 + eta * (1 + eta))
    IncSp1 = 0
    q1Sp1 = -(3 * q2 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * (argLatLam + eps3) + \
            ((1 - 3 * ci**2) / (4 * a**2 * eta**4 * (1 + eta))) * (((1 + eps2)**2 + eta**2) * (q1 + (1 + eta) * cth) + 
            (1 + eps2) * ((1 + eta) * cth + q1 * (eta - eps2)))
    q2Sp1 = (3 * q1 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * (argLatLam + eps3) + \
            ((1 - 3 * ci**2) / (4 * a**2 * eta**4 * (1 + eta))) * (((1 + eps2)**2 + eta**2) * (q2 + (1 + eta) * sth) + 
            (1 + eps2) * ((1 + eta) * sth + q2 * (eta - eps2)))
    RAANSp1 = (3 * ci / (2 * a**2 * eta**4)) * (argLatLam + eps3)
    
    # Create DSP1 matrix
    DSP111 = -(1 / a) * aSp1
    DSP112 = -(3 * eps3 / (2 * a * eta**6)) * (1 - 3 * ci**2) * (1 + eps2)**2
    DSP113 = (3 * s2i / (2 * a * eta**6)) * ((1 + eps2)**3 - eta**3)
    DSP114 = (3 * (1 - 3 * ci**2) / (2 * a * eta**8)) * (2 * q1 * (1 + eps2)**3 + eta**2 * (1 + eps2)**2 * cth - eta**3 * q1)
    DSP115 = (3 * (1 - 3 * ci**2) / (2 * a * eta**8)) * (2 * q2 * (1 + eps2)**3 + eta**2 * (1 + eps2)**2 * sth - eta**3 * q2)
    DSP116 = 0
    
    DSP121 = -(2 / a) * argLatSp1
    DSP122 = ((1 - 3 * ci**2) / (4 * a**2 * eta**4 * (1 + eta))) * (eps2 * (1 + eps2 - eta) - eps3**2) + \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**4 * (1 + eps2)**2)) * ((1 + eps2)**3 - eta**3)
    DSP123 = (3 * eps3 * s2i / (4 * a**2 * eta**4 * (1 + eta))) * ((1 + eps2) + (5 + 4 * eta)) + \
             (15 * s2i / (4 * a**2 * eta**4)) * argLatLam
    DSP124 = ((1 - 3 * ci**2) / (4 * a**2 * eta**6 * (1 + eta)**2)) * (eta**2 * (eps1 * sth + (1 + eta) * (eps2 * sth + eps3 * cth)) + 
             q1 * eps3 * (4 * (eps1 + eps2) + eta * (2 + 5 * eps2))) + \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (4 * q1 * (argLatLam + eps3) + eta**2 * sth) - \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * lam_q1
    DSP125 = -((1 - 3 * ci**2) / (4 * a**2 * eta**6 * (1 + eta)**2)) * (eta**2 * (eps1 * cth + (1 + eta) * (eps2 * cth - eps3 * sth)) - 
             q2 * eps3 * (4 * (eps1 + eps2) + eta * (2 + 5 * eps2))) + \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (4 * q2 * (argLatLam + eps3) - eta**2 * cth) - \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * lam_q2
    DSP126 = 0
    
    DSP131 = -(2 / a) * IncSp1
    DSP132 = 0
    DSP133 = 0
    DSP134 = 0
    DSP135 = 0
    DSP136 = 0
    
    DSP141 = -(2 / a) * q1Sp1
    DSP142 = -((1 - 3 * ci**2) / (4 * a**2 * eta**4)) * ((1 + eps2) * (2 * sth + eps2 * sth + 2 * eps3 * cth) + eps3 * (q1 + cth) + eta**2 * sth) - \
             (3 * q2 * (1 - 5 * ci**2) / (4 * a**2 * eta**4 * (1 + eps2)**2)) * ((1 + eps2)**3 - eta**3)
    DSP143 = (3 * q1 * s2i / (4 * a**2 * eta**2 * (1 + eta))) + \
             (3 * s2i / (4 * a**2 * eta**4)) * ((1 + eps2) * (q1 + (2 + eps2) * cth) - 5 * q2 * eps3 + eta**2 * cth) - \
             (15 * q2 * s2i / (4 * a**2 * eta**4)) * argLatLam
    DSP144 = ((1 - 3 * ci**2) / (4 * a**2 * eta**2 * (1 + eta))) + ((1 - 3 * ci**2) * q1**2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta)**2)) + \
             ((1 - 3 * ci**2) / (8 * a**2 * eta**6)) * (eta**2 * (5 + 2 * (5 * q1 * cth + 2 * q2 * sth) + (3 + 2 * eps2) * c2th) + 
             2 * q1 * (4 * (1 + eps2) * (2 + eps2) * cth + (3 * eta + 4 * eps2) * q1)) - \
             (3 * q2 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (4 * q1 * eps3 + eta**2 * sth) - \
             (3 * q1 * q2 * (1 - 5 * ci**2) / (a**2 * eta**6)) * argLatLam + \
             (3 * q2 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * lam_q1
    DSP145 = ((1 - 3 * ci**2) / (8 * a**2 * eta**6)) * (eta**2 * (2 * (q1 * sth + 2 * q2 * cth) + (3 + 2 * eps2) * s2th) + 
             2 * q2 * (4 * (1 + eps2) * (2 + eps2) * cth + (3 * eta + 4 * eps2) * q1)) + \
             ((1 - 3 * ci**2) * q1 * q2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta)**2)) - \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (eps3 * (eta**2 + 4 * q2**2) - eta**2 * q2 * cth) - \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (argLatLam * (eta**2 + 4 * q2**2)) + \
             (3 * q2 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * lam_q2
    DSP146 = 0
    
    DSP151 = -(2 / a) * q2Sp1
    DSP152 = ((1 - 3 * ci**2) / (4 * a**2 * eta**4)) * ((1 + eps2) * (2 * cth + eps2 * cth - 2 * eps3 * sth) - eps3 * (q2 + sth) + eta**2 * cth) + \
             (3 * q1 * (1 - 5 * ci**2) / (4 * a**2 * eta**4 * (1 + eps2)**2)) * ((1 + eps2)**3 - eta**3)
    DSP153 = (3 * q2 * s2i / (4 * a**2 * eta**2 * (1 + eta))) + \
             (3 * s2i / (4 * a**2 * eta**4)) * ((1 + eps2) * (q2 + (2 + eps2) * sth) + 5 * q1 * eps3 + eta**2 * sth) - \
             (15 * q1 * s2i / (4 * a**2 * eta**4)) * argLatLam
    DSP154 = ((1 - 3 * ci**2) / (8 * a**2 * eta**6)) * (eta**2 * (2 * (2 * q1 * sth + q2 * cth) + (3 + 2 * eps2) * s2th) + 
             2 * q1 * (4 * (1 + eps2) * (2 + eps2) * sth + (3 * eta + 4 * eps2) * q2)) + \
             ((1 - 3 * ci**2) * q1 * q2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta)**2)) + \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (eps3 * (eta**2 + 4 * q1**2) + eta**2 * q1 * sth) + \
             (3 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (argLatLam * (eta**2 + 4 * q1**2)) - \
             (3 * q1 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * lam_q1
    DSP155 = ((1 - 3 * ci**2) / (4 * a**2 * eta**2 * (1 + eta))) + ((1 - 3 * ci**2) * q2**2 * (4 + 5 * eta) / (4 * a**2 * eta**6 * (1 + eta)**2)) + \
             ((1 - 3 * ci**2) / (8 * a**2 * eta**6)) * (eta**2 * (5 + 2 * (2 * q1 * cth + 5 * q2 * sth) - (3 + 2 * eps2) * c2th) + 
             2 * q2 * (4 * (1 + eps2) * (2 + eps2) * sth + (3 * eta + 4 * eps2) * q2)) + \
             (3 * q1 * (1 - 5 * ci**2) / (4 * a**2 * eta**6)) * (4 * q2 * eps3 - eta**2 * cth) + \
             (3 * q1 * q2 * (1 - 5 * ci**2) / (a**2 * eta**6)) * argLatLam - \
             (3 * q1 * (1 - 5 * ci**2) / (4 * a**2 * eta**4)) * lam_q2
    DSP156 = 0
    
    DSP161 = -(2 / a) * RAANSp1
    DSP162 = (3 * ci / (2 * a**2 * eta**4 * (1 + eps2)**2)) * ((1 + eps2)**3 - eta**3)
    DSP163 = -(3 * eps3 * si / (2 * a**2 * eta**4)) - (3 * si / (2 * a**2 * eta**4)) * argLatLam
    DSP164 = (3 * ci / (2 * a**2 * eta**6)) * (4 * q1 * eps3 + eta**2 * sth) + (6 * q1 * ci / (a**2 * eta**6)) * argLatLam - \
             (3 * ci / (2 * a**2 * eta**4)) * lam_q1
    DSP165 = (3 * ci / (2 * a**2 * eta**6)) * (4 * q2 * eps3 - eta**2 * cth) + (6 * q2 * ci / (a**2 * eta**6)) * argLatLam - \
             (3 * ci / (2 * a**2 * eta**4)) * lam_q2
    DSP166 = 0
    
    D_sp1 = np.array([
        [DSP111, DSP112, DSP113, DSP114, DSP115, DSP116],
        [DSP121, DSP122, DSP123, DSP124, DSP125, DSP126],
        [DSP131, DSP132, DSP133, DSP134, DSP135, DSP136],
        [DSP141, DSP142, DSP143, DSP144, DSP145, DSP146],
        [DSP151, DSP152, DSP153, DSP154, DSP155, DSP156],
        [DSP161, DSP162, DSP163, DSP164, DSP165, DSP166]
    ])
    
    # Second short period part, D_sp2
    lamSp2 = -(3 * eps3 * si**2 * c2th / (4 * a**2 * eta**4 * (1 + eta))) * (1 + eps2) * (2 + eps2) - \
             (si**2 / (8 * a**2 * eta**2 * (1 + eta))) * (3 * (q1 * sth + q2 * cth) + (q1 * s3th - q2 * c3th)) - \
             ((3 - 5 * ci**2) / (8 * a**2 * eta**4)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
    
    aSp2 = -(3 * si**2 / (2 * a * eta**6)) * (1 + eps2)**3 * c2th
    argLatSp2 = lamSp2 - (si**2 / (32 * a**2 * eta**4 * (1 + eta))) * (36 * q1 * q2 - 4 * (3 * eta**2 + 5 * eta - 1) * (q1 * sth + q2 * cth) + 
                12 * eps2 * q1 * q2 - 32 * (1 + eta) * s2th - (eta**2 + 12 * eta + 39) * (q1 * s3th - q2 * c3th) + 
                36 * q1 * q2 * c4th - 18 * (q1**2 - q2**2) * s4th + 3 * q2 * (3 * q1**2 - q2**2) * c5th - 3 * q1 * (q1**2 - 3 * q2**2) * s5th)
    incSp2 = -(s2i / (8 * a**2 * eta**4)) * (3 * (q1 * cth - q2 * sth) + 3 * c2th + (q1 * c3th + q2 * s3th))
    q1Sp2 = (q2 * (3 - 5 * ci**2) / (8 * a**2 * eta**4)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th)) + \
            (si**2 / (8 * a**2 * eta**4)) * (3 * (eta**2 - q1**2) * cth + 3 * q1 * q2 * sth - (eta**2 + 3 * q1**2) * c3th - 3 * q1 * q2 * s3th) - \
            (3 * si**2 * c2th / (16 * a**2 * eta**4)) * (10 * q1 + (8 + 3 * q1**2 + q2**2) * cth + 2 * q1 * q2 * sth + 
            6 * (q1 * c2th + q2 * s2th) + (q1**2 - q2**2) * c3th + 2 * q1 * q2 * s3th)
    q2Sp2 = -(q1 * (3 - 5 * ci**2) / (8 * a**2 * eta**4)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th)) - \
            (si**2 / (8 * a**2 * eta**4)) * (3 * (eta**2 - q2**2) * sth + 3 * q1 * q2 * cth + (eta**2 + 3 * q2**2) * s3th + 3 * q1 * q2 * c3th) - \
            (3 * si**2 * c2th / (16 * a**2 * eta**4)) * (10 * q2 + (8 + q1**2 + 3 * q2**2) * sth + 2 * q1 * q2 * cth + 
            6 * (q1 * s2th - q2 * c2th) + (q1**2 - q2**2) * s3th - 2 * q1 * q2 * c3th)
    RAANSp2 = -(ci / (4 * a**2 * eta**4)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
    
    # Create DSP2 matrix
    DSP211 = -(1 / a) * aSp2
    DSP212 = (3 * si**2 / (2 * a * eta**6)) * (1 + eps2)**2 * (3 * eps3 * c2th + 2 * (1 + eps2) * s2th)
    DSP213 = -(3 * s2i * c2th / (2 * a * eta**6)) * (1 + eps2)**3
    DSP214 = -(9 * si**2 * c2th / (2 * a * eta**8)) * (1 + eps2)**2 * (2 * q1 * (1 + eps2) + eta**2 * cth)
    DSP215 = -(9 * si**2 * c2th / (2 * a * eta**8)) * (1 + eps2)**2 * (2 * q2 * (1 + eps2) + eta**2 * sth)
    DSP216 = 0
    
    DSP221 = -(2 / a) * argLatSp2
    DSP222 = -(1 / (8 * a**2 * eta**4)) * (3 * (3 - 5 * ci**2) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th)) - 
            si**2 * (5 * (q1 * cth - q2 * sth) + 16 * c2th + 9 * (q1 * c3th + q2 * s3th)))
    DSP223 = -(s2i / (8 * a**2 * eta**4)) * (10 * (q1 * sth + q2 * cth) + 7 * s2th + 2 * (q1 * s3th - q2 * c3th))
    DSP224 = -((3 - 5 * ci**2) / (8 * a**2 * eta**6)) * (4 * q1 * (3 * s2th + q2 * (3 * cth - c3th)) + (eta**2 + 4 * q1**2) * (3 * sth + s3th)) - \
             (si**2 * (3 * sth + s3th) / (8 * a**2 * eta**2 * (1 + eta))) - \
             (si**2 / (32 * a**2 * eta**4 * (1 + eta))) * (36 * q2 - 4 * (2 + 3 * eta) * sth - (eta * (12 + eta) + 39) * s3th + 9 * eps1 * s5th + 
             12 * q2 * (2 * q1 * cth + q2 * sth) + 9 * q1 * (q1 * s3th - q2 * c3th) + 18 * (3 * q1 * s4th + 2 * q2 * c4th) - 
             3 * q1 * (q1 * s5th - 11 * q2 * c5th) + 24 * ((1 + eps2) * (2 + eps2) * sth + eps3 * (3 + 2 * eps2) * cth) * c2th) - \
             (3 * si**2 / (32 * a**2 * eta**4 * (1 + eta)**2)) * (4 * sth - 6 * q1 * s4th - q1 * (q1 * s5th + q2 * c5th)) + \
             (q1 * si**2 / (8 * a**2 * eta**6 * (1 + eta))) * (20 * (1 + eta) * (q1 * sth + q2 * cth) + 32 * (1 + eta) * s2th + 3 * (4 + 3 * eta) * (q1 * s3th - q2 * c3th)) - \
             (q1 * si**2 * (4 + 5 * eta) / (32 * a**2 * eta**6 * (1 + eta)**2)) * (24 * (q1 * sth + q2 * cth) + 24 * eps3 * (1 + eps2) * (2 + eps2) * c2th - 
             (27 + 3 * eta) * (q1 * s3th - q2 * c3th) - 18 * s4th - 3 * (q1 * s5th + q2 * c5th) + 
             12 * q2 * ((3 + eps2) * q1 + 3 * (q1 * c4th + q2 * s4th) + q1 * (q1 * c5th + q2 * s5th)))
    DSP225 = -((3 - 5 * ci**2) / (8 * a**2 * eta**6)) * (4 * q2 * (3 * s2th + q1 * (3 * sth + s3th)) + (eta**2 + 4 * q2**2) * (3 * cth - c3th)) - \
             (si**2 * (3 * cth - c3th) / (8 * a**2 * eta**2 * (1 + eta))) - \
             (si**2 / (32 * a**2 * eta**4 * (1 + eta))) * (36 * q1 - 4 * (2 + 3 * eta) * cth + (eta * (12 + eta) + 39) * c3th + 9 * eps1 * c5th + 
             12 * q1 * (q1 * cth + 2 * q2 * sth) + 9 * q2 * (q1 * s3th - q2 * c3th) + 18 * (2 * q1 * c4th + 7 * q2 * s4th) + 
             3 * q2 * (11 * q1 * s5th - q2 * c5th) + 24 * (eps3 * (3 + 2 * eps2) * sth - (1 + eps2) * (2 + eps2) * cth) * c2th) - \
             (3 * si**2 / (32 * a**2 * eta**4 * (1 + eta)**2)) * (4 * cth - 6 * q2 * s4th - q2 * (q1 * s5th + q2 * c5th)) + \
             (q2 * si**2 / (8 * a**2 * eta**6 * (1 + eta))) * (20 * (1 + eta) * (q1 * sth + q2 * cth) + 32 * (1 + eta) * s2th + 3 * (4 + 3 * eta) * (q1 * s3th - q2 * c3th)) - \
             (q2 * si**2 * (4 + 5 * eta) / (32 * a**2 * eta**6 * (1 + eta)**2)) * (24 * (q1 * sth + q2 * cth) + 24 * eps3 * (1 + eps2) * (2 + eps2) * c2th - 
             (27 + 3 * eta) * (q1 * s3th - q2 * c3th) - 18 * s4th - 3 * (q1 * s5th + q2 * c5th) + 
             12 * q2 * ((3 + eps2) * q1 + 3 * (q1 * c4th + q2 * s4th) + q1 * (q1 * c5th + q2 * s5th)))
    DSP226 = 0
    
    DSP231 = -(2 / a) * incSp2
    DSP232 = (3 * s2i / (8 * a**2 * eta**4)) * ((q1 * sth + q2 * cth) + 2 * s2th + (q1 * s3th - q2 * c3th))
    DSP233 = -(c2i / (4 * a**2 * eta**4)) * (3 * (q1 * cth - q2 * sth) + 3 * c2th + (q1 * c3th + q2 * s3th))
    DSP234 = -(s2i / (8 * a**2 * eta**6)) * (4 * q1 * (3 * c2th - q2 * (3 * sth - s3th)) + (eta**2 + 4 * q1**2) * (3 * cth + c3th))
    DSP235 = -(s2i / (8 * a**2 * eta**6)) * (4 * q2 * (3 * c2th + q1 * (3 * cth + c3th)) - (eta**2 + 4 * q2**2) * (3 * sth - s3th))
    DSP236 = 0
    
    DSP241 = -(2 / a) * q1Sp2
    DSP242 = (3 * q2 * (3 - 5 * ci**2) / (8 * a**2 * eta**4)) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th)) + \
             (3 * si**2 / (16 * a**2 * eta**4)) * ((2 * eps2 * q2 - 9 * q2 * (q1 * c3th + q2 * s3th) + 12 * (q1 * s4th - q2 * c4th) - 5 * q2 * (q1 * c5th + q2 * s5th)) + 
             0.5 * (4 * (1 + 3 * q1**2) * sth + 40 * q1 * s2th + (28 + 17 * eps1) * s3th + 5 * eps1 * s5th))
    DSP243 = -(s2i / (16 * a**2 * eta**4)) * ((36 * q1 * (q1 * cth - q2 * sth) + 30 * (q1 * c2th - q2 * s2th) - q2 * (q1 * s3th - q2 * c3th) + 
             9 * (q1 * c4th + q2 * s4th) + 3 * q2 * (q1 * s5th - q2 * c5th)) + 
             0.5 * (6 * q1 * (3 + 2 * q1 * cth) + 12 * (1 - 4 * eps1) * cth + (28 + 17 * eps1) * c3th + 3 * eps1 * c5th))
    DSP244 = (q2 * (3 - 5 * ci**2) / (8 * a**2 * eta**6)) * (4 * q1 * (3 * s2th + q2 * (3 * cth - c3th)) + (eta**2 + 4 * q1**2) * (3 * sth + s3th)) - \
             (si**2 / (8 * a**2 * eta**4)) * ((8 * q1 * c3th - 3 * q2 * (sth - s3th)) + 3 * (5 + eps2 + 3 * c2th + 3 * (q1 * c3th + q2 * s3th)) * c2th) - \
             (3 * q1 * si**2 / (4 * a**2 * eta**6)) * (2 * q1 * ((q1 * cth - q2 * sth) + (q1 * c3th + q2 * s3th)) + 
             (9 * cth - c3th + 2 * q1 * (5 + eps2) + 6 * (q1 * c2th + q2 * s2th) + 2 * q1 * (q1 * c3th + q2 * s3th)) * c2th)
    DSP245 = ((3 - 5 * ci**2) / (8 * a**2 * eta**6)) * ((eta**2 + 4 * q2**2) * (3 * s2th + q1 * (3 * sth + s3th)) + 
             2 * (eta**2 + 2 * q2**2) * q2 * (3 * cth - c3th)) + \
             (si**2 / (16 * a**2 * eta**4)) * (6 * (q1 * sth + 2 * q2 * cth) - (9 * q1 * s3th + q2 * c3th) - 9 * s4th - 3 * (q1 * s5th + q2 * c5th)) - \
             (3 * q2 * si**2 / (8 * a**2 * eta**6)) * (2 * q1 * (3 + 2 * (2 * q1 * cth - q2 * sth) + 10 * c2th + 3 * (q1 * c3th + q2 * s3th) + (q1 * c5th + q2 * s5th)) + 
             (8 * cth + 9 * c3th + 6 * (q1 * c4th + q2 * s4th) - c5th))
    DSP246 = 0
    
    DSP251 = -(2 / a) * q2Sp2
    DSP252 = -(3 * q1 * (3 - 5 * ci**2) / (8 * a**2 * eta**4)) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th)) + \
             (3 * si**2 / (16 * a**2 * eta**4)) * ((2 * eps2 * q1 + 9 * q1 * (q1 * c3th + q2 * s3th) - 12 * (q1 * c4th + q2 * s4th) - 5 * q1 * (q1 * c5th + q2 * s5th)) + 
             0.5 * (4 * (1 + 3 * q2**2) * cth + 40 * q2 * s2th - (28 + 17 * eps1) * c3th + 5 * eps1 * c5th))
    DSP253 = -(s2i / (16 * a**2 * eta**4)) * ((36 * q1 * (q1 * sth + q2 * cth) + 30 * (q1 * s2th + q2 * c2th) + q1 * (q1 * s3th - q2 * c3th) + 
             9 * (q1 * s4th - q2 * c4th) + 3 * q1 * (q1 * s5th - q2 * c5th)) - 
             0.5 * (6 * q2 * (3 + 2 * q2 * sth) + 12 * (1 + 2 * eps1) * sth - (28 + 17 * eps1) * s3th + 3 * eps1 * s5th))
    DSP254 = -((3 - 5 * ci**2) / (8 * a**2 * eta**6)) * ((eta**2 + 4 * q1**2) * (3 * s2th + q2 * (3 * cth - c3th)) + 
             2 * (eta**2 + 2 * q1**2) * q1 * (3 * sth + s3th)) - \
             (si**2 / (16 * a**2 * eta**4)) * (6 * (2 * q1 * sth + q2 * cth) + (q1 * s3th + 9 * q2 * c3th) + 9 * s4th - 3 * (q1 * s5th + q2 * c5th)) + \
             (3 * q1 * si**2 / (8 * a**2 * eta**6)) * (2 * q2 * (3 - 2 * (q1 * cth - 2 * q2 * sth) - 10 * c2th - 3 * (q1 * c3th + q2 * s3th) + (q1 * c5th + q2 * s5th)) + 
             (8 * sth - 9 * s3th - 6 * (q1 * s4th - q2 * c4th) - s5th))
    DSP255 = -(q1 * (3 - 5 * ci**2) / (8 * a**2 * eta**6)) * ((eta**2 + 4 * q2**2) * (3 * cth - c3th) + 4 * q2 * (3 * s2th + q1 * (3 * sth + s3th))) - \
             (si**2 / (8 * a**2 * eta**4)) * (8 * q2 * s3th + 3 * q1 * (cth + c3th) + 3 * (5 + eps2 - 3 * c2th - (q1 * c3th - q2 * s3th)) * c2th) - \
             (3 * si**2 * q2 * c2th / (4 * a**2 * eta**6)) * (9 * sth - s3th + 2 * q2 * (5 + eps2) + 6 * (q1 * s2th - q2 * c2th) + 2 * q1 * (q1 * s3th - q2 * c3th))
    DSP256 = 0
    
    DSP261 = -(2 / a) * RAANSp2
    DSP262 = -(3 * ci / (4 * a**2 * eta**4)) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th))
    DSP263 = (si / (4 * a**2 * eta**4)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
    DSP264 = -(ci / (4 * a**2 * eta**6)) * (4 * q1 * (3 * s2th + q2 * (3 * cth - c3th)) + (eta**2 + 4 * q1**2) * (3 * sth + s3th))
    DSP265 = -(ci / (4 * a**2 * eta**6)) * (4 * q2 * (3 * s2th + q1 * (3 * sth + s3th)) + (eta**2 + 4 * q2**2) * (3 * cth - c3th))
    DSP266 = 0
    
    D_sp2 = np.array([
        [DSP211, DSP212, DSP213, DSP214, DSP215, DSP216],
        [DSP221, DSP222, DSP223, DSP224, DSP225, DSP226],
        [DSP231, DSP232, DSP233, DSP234, DSP235, DSP236],
        [DSP241, DSP242, DSP243, DSP244, DSP245, DSP246],
        [DSP251, DSP252, DSP253, DSP254, DSP255, DSP256],
        [DSP261, DSP262, DSP263, DSP264, DSP265, DSP266]
    ])
    
    # Evaluating Osculating Elements
    aOsc = a + gamma * (aLp + aSp1 + aSp2)
    argLatOsc = argLat + gamma * (argLatLp + argLatSp1 + argLatSp2)
    iOsc = inc + gamma * (incLp + IncSp1 + incSp2)
    q1Osc = q1 + gamma * (q1Lp + q1Sp1 + q1Sp2)
    q2Osc = q2 + gamma * (q2Lp + q2Sp1 + q2Sp2)
    OmegaOsc = RAAN + gamma * (RAANLp + RAANSp1 + RAANSp2)
    
    # Transformation Matrix D_J2
    DJ2 = np.eye(6) + gamma * (DLP + D_sp1 + D_sp2)
    
    # Osculating Elements from Mean elements
    osc_c = np.array([aOsc, argLatOsc, iOsc, q1Osc, q2Osc, OmegaOsc])
    
    return DJ2, osc_c


def theta2lam(a, theta, q1, q2):
    """
    Calculation of mean longitude lambda = M + w
    from true longitude theta = f + w
    """
    eta = np.sqrt(1 - q1**2 - q2**2)
    beta = 1 / (eta * (1 + eta))
    R = (a * eta**2) / (1 + q1 * np.cos(theta) + q2 * np.sin(theta))
    
    num = R * (1 + beta * q1**2) * np.sin(theta) - beta * R * q1 * q2 * np.cos(theta) + a * q2
    den = R * (1 + beta * q2**2) * np.cos(theta) - beta * R * q1 * q2 * np.sin(theta) + a * q1
    
    F = np.arctan2(num, den)
    lambda_val = F - q1 * np.sin(F) + q2 * np.cos(F)
    
    while lambda_val < 0:
        lambda_val = lambda_val + 2 * np.pi
    while lambda_val >= 2 * np.pi:
        lambda_val = lambda_val - 2 * np.pi
    
    if theta < 0:
        kk_plus = 0
        quad_plus = 0
        while theta < 0:
            kk_plus = kk_plus + 1
            theta = theta + 2 * np.pi
        if theta < np.pi / 2 and lambda_val > np.pi:
            quad_plus = 1
        elif lambda_val < np.pi / 2 and theta > np.pi:
            quad_plus = -1
        lambda_val = lambda_val - (kk_plus + quad_plus) * 2 * np.pi
    else:
        kk_minus = 0
        quad_minus = 0
        while theta >= 2 * np.pi:
            kk_minus = kk_minus + 1
            theta = theta - 2 * np.pi
        if theta < np.pi / 2 and lambda_val > np.pi:
            quad_minus = -1
        elif lambda_val < np.pi / 2 and theta > np.pi:
            quad_minus = 1
        lambda_val = lambda_val + (kk_minus + quad_minus) * 2 * np.pi
    
    return lambda_val


def lam2theta(lambda_val, q1, q2, Tol):
    """
    Calculation of true longitude theta = f + w
    from mean longitude lambda = M + w
    """
    eta = np.sqrt(1 - q1**2 - q2**2)
    
    # Modified Kepler's Equation
    F = lambda_val
    FF = 1
    while abs(FF) > Tol:
        FF = lambda_val - (F - q1 * np.sin(F) + q2 * np.cos(F))
        dFFdF = -(1 - q1 * np.cos(F) - q2 * np.sin(F))
        del_F = -FF / dFFdF
        F = F + del_F
    
    # True Longitude
    num = (1 + eta) * (eta * np.sin(F) - q2) + q2 * (q1 * np.cos(F) + q2 * np.sin(F))
    den = (1 + eta) * (eta * np.cos(F) - q1) + q1 * (q1 * np.cos(F) + q2 * np.sin(F))
    theta = np.arctan2(num, den)
    
    while theta < 0:
        theta = theta + 2 * np.pi
    while theta >= 2 * np.pi:
        theta = theta - 2 * np.pi
    
    if lambda_val < 0:
        kk_plus = 0
        quad_plus = 0
        while lambda_val < 0:
            kk_plus = kk_plus + 1
            lambda_val = lambda_val + 2 * np.pi
        if lambda_val < np.pi / 2 and theta > np.pi:
            quad_plus = 1
        elif theta < np.pi / 2 and lambda_val > np.pi:
            quad_plus = -1
        theta = theta - (kk_plus + quad_plus) * 2 * np.pi
    else:
        kk_minus = 0
        quad_minus = 0
        while lambda_val >= 2 * np.pi:
            kk_minus = kk_minus + 1
            lambda_val = lambda_val - 2 * np.pi
        if lambda_val < np.pi / 2 and theta > np.pi:
            quad_minus = -1
        elif theta < np.pi / 2 and lambda_val > np.pi:
            quad_minus = 1
        theta = theta + (kk_minus + quad_minus) * 2 * np.pi
    
    return theta, F


def COE_to_Nonsingular(kepElems, tol):
    """
    Convert COE to Nonsingular Elements
    """
    E, f = keplerSolve(kepElems[1], kepElems[5], tol)
    nsElems = np.zeros(6)
    nsElems[0] = kepElems[0]
    nsElems[1] = kepElems[4] + f
    nsElems[2] = kepElems[2]
    nsElems[3] = kepElems[1] * np.cos(kepElems[4])
    nsElems[4] = kepElems[1] * np.sin(kepElems[4])
    nsElems[5] = kepElems[3]
    return nsElems


def Nonsingular_to_COE(nsElems):
    """
    Convert Nonsingular Elements to COE
    """
    kepElems = np.zeros(6)
    kepElems[0] = nsElems[0]
    kepElems[1] = np.sqrt(nsElems[3]**2 + nsElems[4]**2)
    kepElems[2] = nsElems[2]
    kepElems[3] = nsElems[5]
    kepElems[4] = np.arccos(nsElems[3] / kepElems[1])
    kepElems[5] = nsElems[1] - kepElems[4]
    return kepElems


def keplerSolve(e, M, Tol1):
    # Kepler's Equation
    E = M
    FF = 1
    while abs(FF) > Tol1:
        FF = M - (E - e * np.sin(E))
        dFFdE = -(1 - e * np.cos(E))
        del_E = -FF / dFFdE
        E = E + del_E
    
    while E < 0:
        E = E + 2 * np.pi
    while E >= 2 * np.pi:
        E = E - 2 * np.pi
    
    kk_plus = 0
    while M < 0:
        kk_plus = kk_plus + 1
        M = M + 2 * np.pi
    
    kk_minus = 0
    while M >= 2 * np.pi:
        kk_minus = kk_minus + 1
        M = M - 2 * np.pi
    
    # True Anomaly
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    if 0 <= E <= np.pi:
        f = abs(f)
    else:
        f = 2 * np.pi - abs(f)
    
    f = f - kk_plus * 2 * np.pi + kk_minus * 2 * np.pi
    
    return E, f
