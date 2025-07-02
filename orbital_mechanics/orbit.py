"""
궤도 요소 및 궤도 계산 클래스
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, Tuple
from utils.constants import MU_EARTH


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
    ):
        """
        궤도 요소로 궤도 초기화

        Args:
            a: 반장축 (m)
            e: 이심률
            i: 경사각 (rad)
            RAAN: 승교점 적경 (rad)
            omega: 근점 편각 (rad)
            M0: 초기 평균 근점이각 (rad)
            mu: 중력 상수 (m^3/s^2)
        """
        self.a = a
        self.e = e
        self.i = i
        self.RAAN = RAAN
        self.omega = omega
        self.M0 = M0
        self.mu = mu
        self.n = np.sqrt(mu / a**3)  # 평균 운동 (rad/s)

        # 궤도 주기 계산
        self.period = 2 * np.pi / self.n
        self._state_cache = {}

    def kepler_equation(self, E: float, M: float) -> float:
        """케플러 방정식: E - e*sin(E) = M"""
        return E - self.e * np.sin(E) - M

    def get_M(self, t: float) -> float:
        """시간 t에서의 평균 근점이각 계산"""
        return self.M0 + self.n * t

    def get_E(self, t: float) -> float:
        """시간 t에서의 이심근점이각(Eccentric Anomaly) 계산"""
        M = self.get_M(t)

        # 초기값 개선
        if self.e < 0.8:
            E0 = M  # 이심률이 낮을 때는 M이 좋은 초기값
        else:
            E0 = np.pi  # 이심률이 높을 때는 π가 더 나은 초기값

        try:
            E = fsolve(self.kepler_equation, E0, args=(M,))[0]
        except:
            # fsolve 실패 시 Newton-Raphson 방법 사용
            E = self._solve_kepler_newton_raphson(M, E0)

        return E

    def _solve_kepler_newton_raphson(
        self, M: float, E0: float, max_iter: int = 20, tol: float = 1e-12
    ) -> float:
        """Newton-Raphson 방법으로 케플러 방정식 해결"""
        E = E0
        for _ in range(max_iter):
            f = E - self.e * np.sin(E) - M
            df = 1 - self.e * np.cos(E)

            if abs(df) < 1e-14:  # 분모가 0에 가까운 경우
                break

            E_new = E - f / df

            if abs(E_new - E) < tol:
                break

            E = E_new

        return E

    def get_f(self, t: float) -> float:
        """시간 t에서의 참 근점이각(True Anomaly) 계산"""
        E = self.get_E(t)

        # 수치적 안정성 개선
        return 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(E / 2), np.sqrt(1 - self.e) * np.cos(E / 2)
        )

    def get_state(self, t: float) -> Dict[str, float]:
        """시간 t에서의 궤도 상태 계산"""
        if t in self._state_cache:
            return self._state_cache[t]

        f = self.get_f(t)
        E = self.get_E(t)

        # 거리 및 각속도 계산
        r0 = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f))
        dot_r0 = np.sqrt(self.mu / (self.a * (1 - self.e**2))) * self.e * np.sin(f)
        theta0 = self.omega + f
        dot_theta0 = (
            np.sqrt(self.mu / (self.a**3 * (1 - self.e**2) ** 3))
            * (1 + self.e * np.cos(f)) ** 2
        )
        ddot_theta0 = -2 * dot_r0 * dot_theta0 / r0

        state = {
            "r0": r0,
            "dot_r0": dot_r0,
            "theta0": theta0,
            "dot_theta0": dot_theta0,
            "ddot_theta0": ddot_theta0,
            "f": f,
            "E": E,
        }

        if len(self._state_cache) > 1000:
            self._state_cache.pop(next(iter(self._state_cache)))
        self._state_cache[t] = state

        return state

    def get_position_velocity(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """시간 t에서의 관성 좌표계 위치와 속도"""
        f = self.get_f(t)
        r_mag = self.a * (1 - self.e**2) / (1 + self.e * np.cos(f))

        # 궤도면 내에서의 위치 (페리포컬 좌표계)
        r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0])

        # 궤도면 내에서의 속도
        p = self.a * (1 - self.e**2)
        v_mag = np.sqrt(self.mu / p)
        v_pf = np.array([-v_mag * np.sin(f), v_mag * (self.e + np.cos(f)), 0])

        # 회전 행렬 계산 (페리포컬 -> 적도 좌표계)
        R = self._get_rotation_matrix()

        # 관성 좌표계에서의 위치와 속도
        r_eci = R @ r_pf
        v_eci = R @ v_pf

        return r_eci, v_eci

    def _get_rotation_matrix(self) -> np.ndarray:
        """페리포컬 좌표계에서 관성 좌표계로의 회전 행렬"""
        cos_RAAN, sin_RAAN = np.cos(self.RAAN), np.sin(self.RAAN)
        cos_i, sin_i = np.cos(self.i), np.sin(self.i)
        cos_omega, sin_omega = np.cos(self.omega), np.sin(self.omega)

        # 회전 행렬 계산
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
        """현재 궤도 요소 반환"""
        return {
            "a": self.a,
            "e": self.e,
            "i": self.i,
            "RAAN": self.RAAN,
            "omega": self.omega,
            "M0": self.M0,
            "period": self.period,
            "n": self.n,
        }

    def update_orbit(self, new_elements: Dict[str, float]):
        """궤도 요소 업데이트"""
        if "a" in new_elements:
            self.a = new_elements["a"]
            self.n = np.sqrt(self.mu / self.a**3)
            self.period = 2 * np.pi / self.n

        for key in ["e", "i", "RAAN", "omega", "M0"]:
            if key in new_elements:
                setattr(self, key, new_elements[key])

    def copy(self) -> "ChiefOrbit":
        """궤도 복사본 생성"""
        return ChiefOrbit(
            self.a, self.e, self.i, self.RAAN, self.omega, self.M0, self.mu
        )

    def __str__(self) -> str:
        """문자열 표현"""
        return (
            f"ChiefOrbit(a={self.a/1000:.2f}km, e={self.e:.6f}, "
            f"i={self.i*180/np.pi:.2f}°, RAAN={self.RAAN*180/np.pi:.2f}°, "
            f"ω={self.omega*180/np.pi:.2f}°, M0={self.M0*180/np.pi:.2f}°)"
        )

    def __repr__(self) -> str:
        """표현 문자열"""
        return self.__str__()
