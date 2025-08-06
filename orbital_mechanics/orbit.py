"""
궤도 요소 및 궤도 계산 클래스
"""

import numpy as np
from scipy.optimize import fsolve, newton
from typing import Dict, Tuple
from utils.constants import MU_EARTH
# state_to_orbital_elements를 파일 상단으로 이동
from orbital_mechanics.coordinate_transforms import state_to_orbital_elements


class ChiefOrbit:
    """기준 궤도(Chief Orbit) 클래스"""

    def __init__(
        self,
        a: float,
        e: float,
        i: float,
        RAAN: float,
        omega: float,
        M0: float,
        mu: float = MU_EARTH,
        epoch_time=0
    ):
        self.a = a
        self.e = e
        self.i = i
        self.RAAN = RAAN
        self.omega = omega
        self.epoch_time = epoch_time  # M0가 정의된 시각
        self.M0 = M0
        self.mu = mu
        self.n = np.sqrt(mu / a**3)
        self.period = 2 * np.pi / self.n
        self._state_cache = {}
        
    def kepler_equation(self, E: float, M: float) -> float:
        return E - self.e * np.sin(E) - M

    def kepler_equation_derivative(self, E: float) -> float:
        return 1 - self.e * np.cos(E)

    def get_M(self, t: float) -> float:
        dt = t - self.epoch_time  # epoch으로부터의 경과 시간
        
        # 래핑된 Mean Anomaly
        M = (self.M0 + self.n * dt) % (2*np.pi)
        
        return M

    def get_E(self, t: float) -> float:
        M = self.get_M(t)
        M = M % (2 * np.pi)
        
        if self.e < 1e-8:
            return M
        elif self.e < 0.2:
            E0 = M
            return self._solve_kepler_newton_raphson(M, E0)
        elif self.e < 0.9:
            E0 = M + self.e * np.sin(M) / (1 - np.sin(M + self.e) + np.sin(M))
            return self._solve_kepler_newton_raphson(M, E0)
        else:
            E0 = M if M < np.pi else np.pi
            return self._solve_kepler_halley(M, E0)

    def _solve_kepler_newton_raphson(
        self, M: float, E0: float, max_iter: int = 50, tol: float = 1e-12
    ) -> float:
        E = E0
        for i in range(max_iter):
            f = E - self.e * np.sin(E) - M
            df = 1 - self.e * np.cos(E)
            if abs(df) < 1e-14:
                return self._solve_kepler_halley(M, E, max_iter=max_iter-i)
            E_new = E - f / df
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return self._solve_kepler_bisection(M)

    def _solve_kepler_halley(
        self, M: float, E0: float, max_iter: int = 50, tol: float = 1e-12
    ) -> float:
        E = E0
        for i in range(max_iter):
            sin_E, cos_E = np.sin(E), np.cos(E)
            f = E - self.e * sin_E - M
            df = 1 - self.e * cos_E
            ddf = self.e * sin_E
            if abs(df) < 1e-14:
                return self._solve_kepler_bisection(M)
            numerator = 2 * f * df
            denominator = 2 * df * df - f * ddf
            E_new = E - numerator / denominator if abs(denominator) > 1e-14 else E - f / df
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return self._solve_kepler_bisection(M)

    def _solve_kepler_bisection(self, M: float, tol: float = 1e-12) -> float:
        a, b = (M - self.e, M + self.e) if M < np.pi else (M - self.e, M + self.e)
        a, b = max(0, a), min(2 * np.pi, b)
        for _ in range(100):
            E = (a + b) / 2
            f = E - self.e * np.sin(E) - M
            if abs(f) < tol: return E
            if f < 0: a = E
            else: b = E
        return (a + b) / 2

    def get_f(self, t: float) -> float:
        E = self.get_E(t)
        if self.e < 1e-8: return E
        else:
            sin_half_E, cos_half_E = np.sin(E / 2), np.cos(E / 2)
            if abs(cos_half_E) < 1e-10: return np.pi
            return 2 * np.arctan2(np.sqrt(1 + self.e) * sin_half_E, np.sqrt(1 - self.e) * cos_half_E)

    def get_state(self, t: float) -> Dict[str, float]:
        if t in self._state_cache:
            return self._state_cache[t]

        f = self.get_f(t)
        E = self.get_E(t)
        r0 = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f))
        dot_r0 = np.sqrt(self.mu / (self.a * (1 - self.e**2))) * self.e * np.sin(f)
        theta0 = self.omega + f
        dot_theta0 = (np.sqrt(self.mu / (self.a**3 * (1 - self.e**2) ** 3)) * (1 + self.e * np.cos(f)) ** 2)
        ddot_theta0 = -2 * dot_r0 * dot_theta0 / r0

        state = {"r0": r0, "dot_r0": dot_r0, "theta0": theta0, "dot_theta0": dot_theta0, "ddot_theta0": ddot_theta0, "f": f, "E": E}
        if len(self._state_cache) > 1000: self._state_cache.pop(next(iter(self._state_cache)))
        self._state_cache[t] = state
        return state

    def get_position_velocity(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        f = self.get_f(t)
        r_mag = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f))
        r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0])
        p = self.a * (1 - self.e**2)
        v_mag = np.sqrt(self.mu / p)
        v_pf = np.array([-v_mag * np.sin(f), v_mag * (self.e + np.cos(f)), 0])
        R = self._get_rotation_matrix()
        r_eci = R @ r_pf
        v_eci = R @ v_pf
        return r_eci, v_eci

    def _get_rotation_matrix(self) -> np.ndarray:
        cos_RAAN, sin_RAAN = np.cos(self.RAAN), np.sin(self.RAAN)
        cos_i, sin_i = np.cos(self.i), np.sin(self.i)
        cos_omega, sin_omega = np.cos(self.omega), np.sin(self.omega)
        R11 = cos_RAAN * cos_omega - sin_RAAN * sin_omega * cos_i
        R12 = -cos_RAAN * sin_omega - sin_RAAN * cos_omega * cos_i
        R13 = sin_RAAN * sin_i
        R21 = sin_RAAN * cos_omega + cos_RAAN * sin_omega * cos_i
        R22 = -sin_RAAN * sin_omega + cos_RAAN * cos_omega * cos_i
        R23 = -cos_RAAN * sin_i
        R31 = sin_omega * sin_i
        R32 = cos_omega * sin_i
        R33 = cos_i
        return np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    def get_orbital_elements(self) -> Dict[str, float]:
        return {"a": self.a, "e": self.e, "i": self.i, "RAAN": self.RAAN, "omega": self.omega, "M0": self.M0, "period": self.period, "n": self.n}

    def update_orbit(self, new_elements: Dict[str, float]):
        if "a" in new_elements:
            self.a = new_elements["a"]
            self.n = np.sqrt(self.mu / self.a**3)
            self.period = 2 * np.pi / self.n
        for key in ["e", "i", "RAAN", "omega", "M0"]:
            if key in new_elements:
                setattr(self, key, new_elements[key])

    def copy(self) -> "ChiefOrbit":
        return ChiefOrbit(self.a, self.e, self.i, self.RAAN, self.omega, self.M0, self.mu)

    def apply_impulse(self, delta_v_lvlh: np.ndarray, t: float):
        """
        LVLH 좌표계 기준의 delta-v를 궤도에 적용합니다.
        
        Args:
            delta_v_lvlh: LVLH 좌표계에서의 속도 변화량 [vx, vy, vz]
            t: 임펄스가 적용되는 시간
        """
        if not np.any(delta_v_lvlh):
            return
        
        # 현재 시간에서의 위치와 속도
        r_eci, v_eci = self.get_position_velocity(t)
        
        # 각운동량 계산
        h_vec = np.cross(r_eci, v_eci)
        h_norm = np.linalg.norm(h_vec)
        r_norm = np.linalg.norm(r_eci)
    
        if h_norm < 1e-10 or r_norm < 1e-10:
            print(f"WARNING: 작은 각운동량 또는 위치 노름 감지: h_norm={h_norm}, r_norm={r_norm}")
            return
        
        # LVLH to ECI 변환 행렬
        x_lvlh = r_eci / r_norm
        z_lvlh = h_vec / h_norm
        y_lvlh = np.cross(z_lvlh, x_lvlh)
        y_lvlh = y_lvlh / np.linalg.norm(y_lvlh)
        
        R_lvlh_to_eci = np.vstack((x_lvlh, y_lvlh, z_lvlh)).T
        
        # Delta-v를 ECI 좌표계로 변환
        delta_v_eci = R_lvlh_to_eci @ delta_v_lvlh
        
        # 새로운 속도
        v_eci_new = v_eci + delta_v_eci
        
        # 새로운 궤도 요소 계산
        new_elements = state_to_orbital_elements(r_eci, v_eci_new, self.mu)
        
        # 궤도 요소 업데이트
        self.a, self.e, self.i, self.RAAN, self.omega, self.M0 = new_elements
        
        # 평균 운동과 주기 재계산
        self.n = np.sqrt(self.mu / self.a**3)
        self.period = 2 * np.pi / self.n
        
        # 상태 캐시 초기화
        self._state_cache.clear()
        
        # 안정성 검사
        if self.e >= 1.0 or self.a <= 0:
            print(f"WARNING: 궤도가 불안정해졌습니다 (e={self.e}, a={self.a})")

    def __str__(self) -> str:
        return (f"ChiefOrbit(a={self.a/1000:.2f}km, e={self.e:.6f}, "
                f"i={self.i*180/np.pi:.2f}°, RAAN={self.RAAN*180/np.pi:.2f}°, "
                f"ω={self.omega*180/np.pi:.2f}°, M0={self.M0*180/np.pi:.2f}°)")

    def __repr__(self) -> str:
        return self.__str__()
