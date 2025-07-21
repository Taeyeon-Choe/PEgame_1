"""
ga_stm_propagator.py
Gim-Alfriend State Transition Matrix를 사용한 상대 궤도 전파기
시간 동기화 문제 해결 버전
"""

import numpy as np
from .orbit import ChiefOrbit
from .GimAlfriendSTM import GimAlfriendSTM
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
        
        # GimAlfriendSTM 클래스를 초기화합니다.
        self._initialize_ga_stm(0.0)

    def _initialize_ga_stm(self, current_time: float):
        """
        GimAlfriendSTM 클래스를 위한 초기 구조를 생성하고 초기화합니다.
        
        Args:
            current_time: 현재 시뮬레이션 시간
        """
        # 현재 시간에서의 궤도 요소를 NS 형식으로 변환
        chief_elements_ns = self._convert_elements_to_ns(
            self.chief_orbit.get_orbital_elements(), 
            current_time
        )

        init_struct = {
            'params': [R_EARTH, MU_EARTH, J2_EARTH, 1e-12, 0],
            'maneuverParams': [10, np.vstack((np.zeros((3,3)), np.eye(3)))],
            'timeParams': {
                't0': current_time, 
                'dt': self.config.dt, 
                'tf': current_time + self.config.dt
            },
            'initChiefDescription': 'Nonsingular',
            'initDeputyDescription': 'Cartesian',
            'Elements': chief_elements_ns,
            'RelInitState': self.relative_state
        }
        
        self.ga_stm = GimAlfriendSTM(init_struct)

    def _convert_elements_to_ns(self, coe: dict, current_time: float) -> np.ndarray:
        """
        전통적인 궤도 요소(COE)를 비특이(Nonsingular) 궤도 요소로 변환합니다.
        
        Args:
            coe: 궤도 요소 딕셔너리
            current_time: 변환 시점의 시간
            
        Returns:
            비특이 궤도 요소 배열
        """
        # 현재 시간에서의 이심근점 이각과 진근점이각 계산
        E = self.chief_orbit.get_E(current_time)
        f = self.chief_orbit.get_f(current_time)
        
        ns_elements = np.zeros(6)
        ns_elements[0] = coe['a']
        ns_elements[1] = coe['omega'] + f  # 진경도 (True Longitude)
        ns_elements[2] = coe['i']
        ns_elements[3] = coe['e'] * np.cos(coe['omega'])  # q1
        ns_elements[4] = coe['e'] * np.sin(coe['omega'])  # q2
        ns_elements[5] = coe['RAAN']
        
        return ns_elements

    def propagate(self, dt: float, current_time: float) -> np.ndarray:
        """
        주어진 시간(dt)만큼 상태를 전파합니다.

        Args:
            dt: 전파할 시간 간격 (초)
            current_time: 현재 시뮬레이션 시간

        Returns:
            전파된 후의 새로운 상대 상태
        """
        # 현재 시간부터 다음 스텝까지의 STM 계산
        self.ga_stm.propagateModel(t1=current_time, t2=current_time + dt)
        
        # STM을 사용하여 상태 업데이트
        phi_dt = self.ga_stm.Phi[:, :, -1]
        self.relative_state = phi_dt @ self.relative_state
        
        return self.relative_state

    def apply_pursuer_control(self, delta_v_p: np.ndarray, dt: float, current_time: float) -> np.ndarray:
        """
        추격자(Deputy)의 제어 입력(delta-v)을 적용합니다.

        Args:
            delta_v_p: 추격자의 delta-v 벡터 (LVLH)
            dt: 제어가 적용될 시간 간격
            current_time: 현재 시뮬레이션 시간

        Returns:
            제어 입력이 적용된 후의 상대 상태
        """
        # 현재 시간에 맞는 이산 상태 및 입력 행렬 계산
        self.ga_stm.dt = dt
        self.ga_stm.timeParams = {
            't0': current_time, 
            'dt': dt, 
            'tf': current_time + dt
        }
        self.ga_stm.makeTimeVector()
        self.ga_stm.makeDiscreteMatrices()
        
        # 마지막으로 계산된 행렬 사용
        Ak = self.ga_stm.Ak[:, :, -1]
        Bk = self.ga_stm.Bk[:, :, -1]
        
        # 제어 입력 적용
        self.relative_state = Ak @ self.relative_state + Bk @ delta_v_p
        
        return self.relative_state

    def reinitialize_with_new_chief_orbit(self, new_chief_orbit: ChiefOrbit, 
                                        current_relative_state: np.ndarray, 
                                        current_time: float):
        """
        회피자(Chief)의 궤도가 변경되었을 때 전파기를 재초기화합니다.
        
        Args:
            new_chief_orbit: 새로운 기준 궤도
            current_relative_state: 현재의 상대 상태 벡터
            current_time: 현재 시뮬레이션 시간
        """
        self.chief_orbit = new_chief_orbit
        self.relative_state = current_relative_state
        self._initialize_ga_stm(current_time)
