"""
ga_stm_propagator.py
Gim-Alfriend State Transition Matrix를 사용한 상대 궤도 전파기
"""

import numpy as np
from .orbit import ChiefOrbit
from .GimAlfriendSTM import GimAlfriendSTM  # 기존 GimAlfriendSTM.py 파일 임포트
from utils.constants import MU_EARTH, R_EARTH, J2_EARTH


class GASTMPropagator:
    """
    Gim-Alfriend State Transition Matrix를 사용한 상대 궤도 전파기
    """
    def __init__(self, chief_orbit: ChiefOrbit, initial_relative_state: np.ndarray, config):
        """
        전파기 초기화

        Args:
            chief_orbit (ChiefOrbit): 기준 궤도(Chief) 객체
            initial_relative_state (np.ndarray): 초기 상대 상태 [x,y,z,vx,vy,vz]
            config: 환경 설정 객체
        """
        self.chief_orbit = chief_orbit
        self.relative_state = initial_relative_state
        self.config = config
        self.t = 0
        
        # GimAlfriendSTM 클래스를 초기화합니다.
        self._initialize_ga_stm()

    def _initialize_ga_stm(self):
        """GimAlfriendSTM 클래스를 위한 초기 구조를 생성하고 초기화합니다."""
        # PEgame의 궤도 요소를 GimAlfriendSTM 입력 형식에 맞게 변환
        chief_elements_ns = self._convert_elements_to_ns(self.chief_orbit.get_orbital_elements())

        init_struct = {
            'params': [R_EARTH, MU_EARTH, J2_EARTH, 1e-12, 0],
            'maneuverParams': [10, np.vstack((np.zeros((3,3)), np.eye(3)))],
            'timeParams': {'t0': self.t, 'dt': self.config.dt, 'tf': self.t + self.config.dt},
            'initChiefDescription': 'Nonsingular',
            'initDeputyDescription': 'Cartesian',
            'Elements': chief_elements_ns,
            'RelInitState': self.relative_state
        }
        
        self.ga_stm = GimAlfriendSTM(init_struct)

    def _convert_elements_to_ns(self, coe: dict) -> np.ndarray:
        """
        전통적인 궤도 요소(COE)를 비특이(Nonsingular) 궤도 요소로 변환합니다.
        a, e, i, RAAN, omega, M -> a, theta, i, q1, q2, RAAN
        """
        # 이심근점 이각(E) 및 진근점이각(f) 계산
        E = self.chief_orbit.get_E(self.t)
        f = self.chief_orbit.get_f(self.t)
        
        ns_elements = np.zeros(6)
        ns_elements[0] = coe['a']
        ns_elements[1] = coe['omega'] + f  # 진경도 (True Longitude)
        ns_elements[2] = coe['i']
        ns_elements[3] = coe['e'] * np.cos(coe['omega']) # q1
        ns_elements[4] = coe['e'] * np.sin(coe['omega']) # q2
        ns_elements[5] = coe['RAAN']
        
        return ns_elements

    def propagate(self, dt: float) -> np.ndarray:
        """
        주어진 시간(dt)만큼 상태를 전파합니다.

        Args:
            dt (float): 전파할 시간 간격 (초)

        Returns:
            np.ndarray: 전파된 후의 새로운 상대 상태
        """
        # 다음 스텝까지의 상태 전이 행렬(STM) 계산
        self.ga_stm.propagateModel(t1=self.t, t2=self.t + dt)
        
        # STM을 사용하여 상태 업데이트
        phi_dt = self.ga_stm.Phi[:, :, -1]
        self.relative_state = phi_dt @ self.relative_state
        
        # 내부 시간 업데이트
        self.t += dt
        
        return self.relative_state

    def apply_pursuer_control(self, delta_v_p: np.ndarray, dt: float) -> np.ndarray:
        """
        추격자(Deputy)의 제어 입력(delta-v)을 적용합니다.
        이 때, LVLH 좌표계의 회전을 고려한 입력 행렬 Bk를 사용합니다.

        Args:
            delta_v_p (np.ndarray): 추격자의 delta-v 벡터 (LVLH)
            dt (float): 제어가 적용될 시간 간격

        Returns:
            np.ndarray: 제어 입력이 적용된 후의 상대 상태
        """
        # 현재 시간에 맞는 이산 상태 및 입력 행렬 계산
        self.ga_stm.dt = dt
        self.ga_stm.makeDiscreteMatrices()
        
        # 마지막으로 계산된 행렬 사용
        Ak = self.ga_stm.Ak[:, :, -1]
        Bk = self.ga_stm.Bk[:, :, -1]
        
        # 제어 입력 적용
        # x_k+1 = Ak * x_k + Bk * u_k
        self.relative_state = Ak @ self.relative_state + Bk @ delta_v_p
        
        # 내부 시간 업데이트
        self.t += dt
        
        return self.relative_state

    def reinitialize_with_new_chief_orbit(self, new_chief_orbit: ChiefOrbit, current_relative_state: np.ndarray):
        """
        회피자(Chief)의 궤도가 변경되었을 때 전파기를 재초기화합니다.
        
        Args:
            new_chief_orbit (ChiefOrbit): 새로운 기준 궤도
            current_relative_state (np.ndarray): 현재의 상대 상태 벡터
        """
        self.chief_orbit = new_chief_orbit
        self.relative_state = current_relative_state
        self._initialize_ga_stm()
