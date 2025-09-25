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
        # TV-LQR용 시퀀스 캐시 (슬라이딩 윈도우)
        self._tvlqr_cache = {
            "t0": None,
            "dt": None,
            "H": None,
            "A_seq": None,
            "B_seq": None,
        }

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
        ns_elements[1] = coe['omega'] + f  # true longitude θ = f + ω
        ns_elements[2] = coe['i']
        ns_elements[3] = coe['e'] * np.cos(coe['omega'])  # q1
        ns_elements[4] = coe['e'] * np.sin(coe['omega'])  # q2
        ns_elements[5] = coe['RAAN']
        
        return ns_elements

    def _compute_discrete_matrices(self, dt: float, current_time: float) -> tuple:
        """이산 상태 및 입력 행렬 계산 (캐싱 포함)"""
        # 캐시 확인
        cache_key = (current_time, dt, getattr(self.ga_stm, "samples", None))
        if cache_key in self._matrix_cache:
            if self.debug_mode:
                print(f"Using cached matrices for t={current_time}, dt={dt}")
            return self._matrix_cache[cache_key]
        
        # 새로운 행렬 계산
        self.ga_stm.dt = float(dt)
        self.ga_stm.t0 = float(current_time)
        self.ga_stm.tf = float(current_time + dt)
        self.ga_stm.makeTimeVector()
        self.ga_stm.makeDiscreteMatrices()
        
        Ak_int = self.ga_stm.Ak[:, :, -1]
        Bk_int = self.ga_stm.Bk[:, :, -1]
        # Permutation: std [x,y,z,vx,vy,vz]  <->  int [x,vx,y,vy,z,vz]
        P = np.array([
            [1,0,0,0,0,0],
            [0,0,0,1,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,1,0,0,0],
            [0,0,0,0,0,1],
        ], dtype=float)
        Ak = P.T @ Ak_int @ P
        Bk = P.T @ Bk_int    # accel-input용일 때        
        # 캐시 저장 (최대 10개 유지)
        if len(self._matrix_cache) > 10:
            self._matrix_cache.clear()
        self._matrix_cache[cache_key] = (Ak, Bk)
        
        return Ak, Bk

    def _reset_tvlqr_cache(self):
        self._tvlqr_cache = {
            "t0": None,
            "dt": None,
            "H": None,
            "A_seq": None,
            "B_seq": None,
        }

    def get_impulsive_AB_sequence(self, horizon: int, dt: float, current_time: float):
        """Impulse 기반 TV-LQR를 위한 (Ak, B~k) 시퀀스를 생성한다.

        Impulse는 각 스텝 시작에 속도 성분에 더해지며, 이후 Ak로 전파된다:
            x_{k+1} = Ak x_k + (Ak G) v_k,  G = [0; I].

        Args:
            horizon: TV-LQR 스텝 수 (양수)
            dt: 시간 간격 [s]
            current_time: 시퀀스 시작 시간 [s]

        Returns:
            tuple (A_seq, B_seq)
                A_seq: (H, 6, 6) array
                B_seq: (H, 6, 3) array where B_seq[k] = Ak @ G
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        H = int(horizon)
        dt = float(dt)
        t0 = float(current_time)

        cache = self._tvlqr_cache
        same_grid = (
            cache["A_seq"] is not None
            and cache["B_seq"] is not None
            and cache["H"] == H
            and cache["dt"] == dt
            and cache["t0"] is not None
            and abs((cache["t0"] + dt) - t0) < 1e-9
        )

        perm = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        G = np.zeros((6, 3), dtype=float)
        G[3:, :] = np.eye(3, dtype=float)

        if same_grid:
            A_seq_prev = cache["A_seq"]
            B_seq_prev = cache["B_seq"]
            A_seq = np.empty_like(A_seq_prev)
            B_seq = np.empty_like(B_seq_prev)
            A_seq[:-1] = A_seq_prev[1:]
            B_seq[:-1] = B_seq_prev[1:]

            self.ga_stm.dt = dt
            self.ga_stm.t0 = t0 + (H - 1) * dt
            self.ga_stm.tf = t0 + H * dt
            self.ga_stm.makeTimeVector()
            self.ga_stm.makeDiscreteMatrices()

            Ak_last_int = self.ga_stm.Ak
            if Ak_last_int is None or Ak_last_int.shape[2] == 0:
                raise RuntimeError("Failed to build GA-STM matrix for TV-LQR")
            Ak_last_std = perm.T @ Ak_last_int[:, :, -1] @ perm
            A_seq[-1] = Ak_last_std
            B_seq[-1] = Ak_last_std @ G
        else:
            self.ga_stm.dt = dt
            self.ga_stm.t0 = t0
            self.ga_stm.tf = t0 + H * dt
            self.ga_stm.makeTimeVector()
            self.ga_stm.makeDiscreteMatrices()

            Ak_int = self.ga_stm.Ak
            if Ak_int is None or Ak_int.shape[2] < H:
                raise RuntimeError("GA-STM discrete matrices are not available for requested horizon")

            A_seq = np.zeros((H, 6, 6), dtype=float)
            for k in range(H):
                A_seq[k] = perm.T @ Ak_int[:, :, k] @ perm
            B_seq = np.einsum('kij,jm->kim', A_seq, G, optimize=True)

        self._tvlqr_cache.update({
            "t0": t0,
            "dt": dt,
            "H": H,
            "A_seq": A_seq,
            "B_seq": B_seq,
        })

        return A_seq, B_seq

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
        self._reset_tvlqr_cache()
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
        # std 순서에서 속도 성분에 임펄스 적용
        x_plus[3:6] += dv
        # 표준 순서용 Ak로 전파
        self.relative_state = Ak @ x_plus
        
        if not np.isfinite(self.relative_state).all():
            raise FloatingPointError("GA-STM state became non-finite after impulse propagation")
        return self.relative_state
