"""
ga_stm_propagator.py
Gim-Alfriend State Transition Matrix
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
        """전파기 초기화"""
        self.chief_orbit = chief_orbit
        self.relative_state = initial_relative_state
        self.config = config
        
        # 디버그 모드
        self.debug_mode = getattr(config, 'debug_mode', False)
        
        # GimAlfriendSTM 클래스 초기화
        self._initialize_ga_stm(0.0)
        
        # 행렬 캐시 (성능 최적화)
        self._matrix_cache = {}
        self._cache_time = None

    def _initialize_ga_stm(self, current_time: float):
        """GimAlfriendSTM 클래스 초기화"""
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
        """COE를 Nonsingular 요소로 변환"""
        E = self.chief_orbit.get_E(current_time)
        f = self.chief_orbit.get_f(current_time)
        
        ns_elements = np.zeros(6)
        ns_elements[0] = coe['a']
        ns_elements[1] = coe['omega'] + f  # True Longitude
        ns_elements[2] = coe['i']
        ns_elements[3] = coe['e'] * np.cos(coe['omega'])  # q1
        ns_elements[4] = coe['e'] * np.sin(coe['omega'])  # q2
        ns_elements[5] = coe['RAAN']
        
        return ns_elements

    def _compute_discrete_matrices(self, dt: float, current_time: float) -> tuple:
        """이산 상태 및 입력 행렬 계산 (캐싱 포함)"""
        # 캐시 확인
        cache_key = (current_time, dt)
        if cache_key in self._matrix_cache:
            if self.debug_mode:
                print(f"Using cached matrices for t={current_time}, dt={dt}")
            return self._matrix_cache[cache_key]
        
        # 새로운 행렬 계산
        self.ga_stm.dt = dt
        self.ga_stm.timeParams = {
            't0': current_time, 
            'dt': dt, 
            'tf': current_time + dt
        }
        self.ga_stm.makeTimeVector()
        self.ga_stm.makeDiscreteMatrices()
        
        Ak = self.ga_stm.Ak[:, :, -1]
        Bk = self.ga_stm.Bk[:, :, -1]
        
        # 캐시 저장 (최대 10개 유지)
        if len(self._matrix_cache) > 10:
            self._matrix_cache.clear()
        self._matrix_cache[cache_key] = (Ak, Bk)
        
        return Ak, Bk

    def propagate_with_control(self, control: np.ndarray, dt: float, 
                              current_time: float, current_state: np.ndarray = None) -> np.ndarray:
        """
        통합된 전파 메서드 - 제어 입력 포함
        
        Args:
            control: 제어 입력 벡터 (추격자 delta-v) [3x1]
            dt: 시간 간격
            current_time: 현재 시뮬레이션 시간
            current_state: 외부 상태 (동기화용, optional)
            
        Returns:
            업데이트된 상대 상태 벡터 [6x1]
        """
        # 외부 상태와 동기화
        if current_state is not None:
            self.relative_state = current_state.copy()
        
        # 이산 행렬 계산 (캐싱 활용)
        Ak, Bk = self._compute_discrete_matrices(dt, current_time)
        
        # 상태 업데이트: x_{k+1} = Ak * x_k + Bk * u_k
        self.relative_state = Ak @ self.relative_state + Bk @ control
        
        if self.debug_mode and np.linalg.norm(control) > 0:
            print(f"Applied control: {control}")
            print(f"Control effect (Bk*u): {Bk @ control}")
        
        return self.relative_state

    def propagate(self, dt: float, current_time: float) -> np.ndarray:
        """
        제어 없이 상태 전파 (하위 호환성 유지)
        내부적으로 propagate_with_control을 호출
        """
        return self.propagate_with_control(np.zeros(3), dt, current_time)

    def apply_pursuer_control(self, delta_v_p: np.ndarray, dt: float, 
                             current_time: float) -> np.ndarray:
        """
        추격자 제어 적용 (하위 호환성 유지)
        내부적으로 propagate_with_control을 호출
        """
        return self.propagate_with_control(delta_v_p, dt, current_time)

    def reinitialize_with_new_chief_orbit(self, new_chief_orbit: ChiefOrbit, 
                                        current_relative_state: np.ndarray, 
                                        current_time: float):
        """Chief 궤도 변경 시 재초기화"""
        self.chief_orbit = new_chief_orbit
        self.relative_state = current_relative_state
        self._matrix_cache.clear()  # 캐시 초기화
        self._initialize_ga_stm(current_time)

    def get_state_transition_matrix(self, dt: float, current_time: float) -> np.ndarray:
        """현재 STM 반환 (디버깅/분석용)"""
        Ak, _ = self._compute_discrete_matrices(dt, current_time)
        return Ak

    def get_control_influence_matrix(self, dt: float, current_time: float) -> np.ndarray:
        """현재 제어 영향 행렬 반환 (디버깅/분석용)"""
        _, Bk = self._compute_discrete_matrices(dt, current_time)
        return Bk

    def propagate_with_impulse(self, delta_v: np.ndarray, dt: float, current_time: float, current_state: np.ndarray=None):
        """
        Propagate one GA-STM step with an instantaneous Δv (impulse) at the BEGINNING of the step.
        x_plus  = x + [0,0,0, Δv]^T
        x_next  = Ak @ x_plus
        Notes
        -----
        - Ak maps t_k → t_k+dt under zero-input.
        - Bk is not used in the impulse model.
        - delta_v must be in the same RTN/LVLH frame as the relative state velocity components.
        """
        if current_state is not None:
            self.relative_state = np.asarray(current_state, dtype=float)
        assert dt > 0.0, "dt must be > 0"
        # Build Ak for this step (Bk returned but unused)
        Ak, _Bk = self._compute_discrete_matrices(dt, current_time)
        # Optional: stability log
        try:
            import numpy as _np
            rho = _np.max(_np.abs(_np.linalg.eigvals(Ak)))
            if rho > 1.05:
                print(f"[GA-STM WARN] t={current_time:.3f}s dt={dt:.3f}s spectral_radius(Ak)={rho:.4f}")
        except Exception:
            pass
        dv = np.asarray(delta_v, dtype=float).reshape(-1)
        assert dv.shape[0] == 3, "delta_v must be 3-dim"
        # Impulse at the beginning of the step
        x_plus = self.relative_state.copy()
        x_plus[3:6] += dv  # instantaneous velocity jump
        self.relative_state = Ak @ x_plus
        if not np.isfinite(self.relative_state).all():
            raise FloatingPointError("GA-STM state became non-finite after impulse propagation")
        return self.relative_state

