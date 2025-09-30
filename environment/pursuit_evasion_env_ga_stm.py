"""
pursuit_evasion_env_ga_stm.py
PEgame 환경: GASTMPropagator를 사용한 STM 통합
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from environment.pursuit_evasion_env import PursuitEvasionEnv
from orbital_mechanics.ga_stm_propagator import GASTMPropagator
from orbital_mechanics.orbit import ChiefOrbit
from scipy.integrate import solve_ivp
from orbital_mechanics.dynamics import relative_dynamics_evader_centered
from controllers.tvlqr import tvlqr_action


class PursuitEvasionEnvGASTM(PursuitEvasionEnv):
    """
    Gim-Alfriend STM 전파 옵션을 포함한 확장된 PursuitEvasionEnv.
    """
    
    def __init__(self, config=None, use_gastm: bool = True):
        """
        환경 초기화

        Args:
            config: 환경 설정 객체
            use_gastm: True이면 GA STM을, False이면 수치 적분을 사용합니다.
        """
        super().__init__(config)
        
        self.use_gastm = use_gastm
        self.gastm_propagator = None
        self._pending_pursuer_impulse = np.zeros(3, dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """환경을 리셋하고, 필요 시 GASTMPropagator를 재초기화합니다."""
        obs, info = super().reset(seed=seed, options=options)

        if self.use_gastm:
            # 새로운 초기 조건으로 GASTMPropagator 초기화
            self.gastm_propagator = GASTMPropagator(
                chief_orbit=self.evader_orbit,
                initial_relative_state=self.state,
                config=self.config
            )
        self._pending_pursuer_impulse.fill(0.0)
        return obs, info

    def _simulate_relative_motion(self):
        """상대 운동 시뮬레이션 (모드에 따라 다른 방법 사용)"""
        if self.use_gastm and self.gastm_propagator:
            dv = np.asarray(self._pending_pursuer_impulse, dtype=np.float32)
            try:
                self.state = self.gastm_propagator.propagate_with_impulse(
                    delta_v=dv,
                    dt=self.dt,
                    current_time=self.t,
                    current_state=self.state,
                )
            except AttributeError:
                # 구 버전 호환: impulse 기능이 없으면 기본 propagate 사용 + 속도 갱신
                if np.any(dv != 0):
                    self.state[3:] += dv
                self.state = self.gastm_propagator.propagate(self.dt, self.t)
            finally:
                self._pending_pursuer_impulse.fill(0.0)
        else:
            # 기존의 비선형 동역학 수치 적분 사용
            super()._simulate_relative_motion()

    def _apply_pursuer_delta_v(self, delta_v_p: np.ndarray):
        """추격자 임펄스를 저장하여 GA-STM 전파에 반영"""
        delta_v_p = np.asarray(delta_v_p, dtype=np.float32)
        if not np.any(delta_v_p != 0):
            self._pending_pursuer_impulse.fill(0.0)
            if not (self.use_gastm and self.gastm_propagator):
                super()._apply_pursuer_delta_v(delta_v_p)
            return

        if self.use_gastm and self.gastm_propagator:
            self._pending_pursuer_impulse = delta_v_p.copy()
        else:
            super()._apply_pursuer_delta_v(delta_v_p)

    def compute_interception_strategy(self, state: np.ndarray) -> np.ndarray:
        policy = getattr(self.config, "pursuer_policy", "heuristic")
        if policy != "tvlqr" or not (self.use_gastm and self.gastm_propagator):
            return super().compute_interception_strategy(state)

        H = max(1, int(getattr(self.config, "lqr_horizon", 10)))
        try:
            A_seq, B_seq = self.gastm_propagator.get_impulsive_AB_sequence(
                horizon=H,
                dt=self.dt,
                current_time=self.t,
            )
        except Exception as exc:
            if self.debug_mode:
                print(f"[TVLQR] Fallback to heuristic due to GA-STM error: {exc}")
            return super().compute_interception_strategy(state)

        Q_diag = np.asarray(getattr(self.config, "lqr_Q_diag", [1, 1, 1, 0.05, 0.05, 0.05]), dtype=float)
        QN_diag = np.asarray(getattr(self.config, "lqr_QN_diag", [5, 5, 5, 0.1, 0.1, 0.1]), dtype=float)
        R_diag = np.asarray(getattr(self.config, "lqr_R_diag", [1e-2, 1e-2, 1e-2]), dtype=float)

        try:
            dv_cmd = tvlqr_action(
                x=np.asarray(state, dtype=float).reshape(-1),
                A_seq=A_seq,
                B_seq=B_seq,
                Q_seq=Q_diag,
                R_seq=R_diag,
                QN=QN_diag,
                dv_max=self.delta_v_pmax,
            )
        except Exception as exc:
            if self.debug_mode:
                print(f"[TVLQR] Gain computation failed, fallback to heuristic: {exc}")
            return super().compute_interception_strategy(state)

        dv_cmd = np.nan_to_num(dv_cmd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return dv_cmd

    def _apply_evader_delta_v(self, delta_v_e: np.ndarray):
        """
        회피자의 delta-v를 적용하고, STM 사용 시 전파기를 재초기화합니다.
        """
        # ChiefOrbit 클래스에 추가한 apply_impulse 메서드 호출
        self.evader_orbit.apply_impulse(delta_v_e, self.t)
        
        # STM 모드일 경우, 변경된 궤도를 기준으로 전파기를 재초기화
        if self.use_gastm and self.gastm_propagator:
            self.gastm_propagator.reinitialize_with_new_chief_orbit(
                self.evader_orbit, self.state, self.t
            )

    def compare_propagation_methods(self, test_duration: float = 300.0, 
                                  control_sequence: Optional[np.ndarray] = None) -> Dict:
        """
        GA STM과 비선형 전파 방법을 비교합니다.
        
        Args:
            test_duration: 테스트 지속 시간 (초)
            control_sequence: 선택적 제어 시퀀스
            
        Returns:
            비교 결과 딕셔너리
        """
        # 초기 상태와 궤도 요소 저장
        initial_state = self.state.copy()
        initial_orbit_elements = self.evader_orbit.get_orbital_elements()
        initial_time = self.t
        initial_step = self.step_count
        
        # 스텝 수 계산
        n_steps = int(test_duration / self.dt)
        
        # 제어 시퀀스 생성 (제공되지 않은 경우)
        if control_sequence is None:
            control_sequence = []
            for i in range(n_steps):
                # 간단한 추격 전략
                rel_pos = self.state[:3]
                pursuit_direction = -rel_pos / np.linalg.norm(rel_pos)
                action = pursuit_direction * 0.01  # 정규화된 액션
                control_sequence.append(action)
            control_sequence = np.array(control_sequence)
        
        # 비선형 동역학으로 실행
        self.use_gastm = False
        nonlinear_states = [initial_state.copy()]
        
        for i in range(n_steps):
            obs, _, _, _, _ = self.step(control_sequence[i])
            nonlinear_states.append(self.state.copy())
        
        # 초기 상태와 궤도 요소로 복원
        self.state = initial_state.copy()
        self.evader_orbit = ChiefOrbit(
            a=initial_orbit_elements["a"],
            e=initial_orbit_elements["e"],
            i=initial_orbit_elements["i"],
            RAAN=initial_orbit_elements["RAAN"],
            omega=initial_orbit_elements["omega"],
            M0=initial_orbit_elements["M0"],
            mu=self.evader_orbit.mu,
        )
        self.t = initial_time
        self.step_count = initial_step
        
        # GA STM으로 실행
        self.use_gastm = True
        self.gastm_propagator = GASTMPropagator(
            chief_orbit=self.evader_orbit,
            initial_relative_state=self.state,
            config=self.config
        )
        gastm_states = [initial_state.copy()]
        
        for i in range(n_steps):
            obs, _, _, _, _ = self.step(control_sequence[i])
            gastm_states.append(self.state.copy())
        
        # 배열로 변환
        nonlinear_states = np.array(nonlinear_states)
        gastm_states = np.array(gastm_states)
        
        # 오차 계산
        position_errors = np.linalg.norm(
            nonlinear_states[:, :3] - gastm_states[:, :3], axis=1
        )
        velocity_errors = np.linalg.norm(
            nonlinear_states[:, 3:] - gastm_states[:, 3:], axis=1
        )
        
        # 시간 배열 생성
        time_array = np.arange(len(position_errors)) * self.dt
        
        return {
            'time': time_array,
            'nonlinear_states': nonlinear_states,
            'gastm_states': gastm_states,
            'position_errors': position_errors,
            'velocity_errors': velocity_errors,
            'max_position_error': np.max(position_errors),
            'mean_position_error': np.mean(position_errors),
            'final_position_error': position_errors[-1],
            'max_velocity_error': np.max(velocity_errors),
            'mean_velocity_error': np.mean(velocity_errors),
            'final_velocity_error': velocity_errors[-1]
        }


# 사용 예시
if __name__ == "__main__":
    from config.settings import get_config
    
    # 설정 로드
    config = get_config()
    config.environment.dt = 10.0  # 10초 시간 간격
    
    # GA STM 사용 환경 생성
    print("GA STM 환경 생성 중...")
    env = PursuitEvasionEnvGASTM(config, use_gastm=True)
    
    # 환경 리셋
    obs, _ = env.reset()
    print(f"초기 관측값 형상: {obs.shape}")
    print(f"초기 상대 거리: {np.linalg.norm(env.state[:3]):.1f} m")
    
    # 단일 스텝 테스트
    print("\nGA STM으로 단일 스텝 테스트...")
    action = np.array([-0.1, 0.0, 0.0])  # 회피자 액션만 (추격자는 자동)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"스텝 후 - 거리: {info['relative_distance_m']:.2f} m")
    print(f"동역학 모드: {info['dynamics_mode']}")
    
    # 방법 비교
    print("\nGA STM과 비선형 동역학 비교 중...")
    env.reset()
    results = env.compare_propagation_methods(test_duration=600.0)  # 10분
    
    # 결과 출력
    print(f"\n위치 오차:")
    print(f"  최대: {results['max_position_error']:.2f} m")
    print(f"  평균: {results['mean_position_error']:.2f} m")
    print(f"  최종: {results['final_position_error']:.2f} m")
