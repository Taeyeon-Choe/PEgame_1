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
        return obs, info

    def _simulate_relative_motion(self):
        """상대 운동 시뮬레이션 (모드에 따라 다른 방법 사용)"""
        if self.use_gastm and self.gastm_propagator:
            # GASTMPropagator를 사용하여 상태 전파 (현재 시간 전달)
            self.state = self.gastm_propagator.propagate(self.dt, self.t)
        else:
            # 기존의 비선형 동역학 수치 적분 사용
            super()._simulate_relative_motion()

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

    def step(
        self, normalized_action_e: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행"""
        # 1. 회피자(Evader) 액션 처리
        action_e = self._denormalize_action(normalized_action_e)
        delta_v_e = np.clip(action_e, -self.delta_v_emax, self.delta_v_emax)
        delta_v_e_mag = float(np.linalg.norm(delta_v_e))
        self.total_delta_v_e += delta_v_e_mag
    
        # 회피자 기동 적용 (궤도 변경)
        if np.any(delta_v_e):
            self._apply_evader_delta_v(delta_v_e)
    
        # 2. 추격자(Pursuer) 액션 계산
        if self.step_count % self.k == 0:
            delta_v_p = self.compute_interception_strategy(self.state)
            delta_v_p = np.asarray(delta_v_p, dtype=np.float32)
            self.pursuer_last_action = delta_v_p
        else:
            delta_v_p = np.zeros(3, dtype=np.float32)
    
        # 3. 상태 전파 - 통합된 방식
        if self.use_gastm and self.gastm_propagator:
            # GA-STM 모드: 항상 propagate_with_control 사용
            # GA-STM with impulse at the BEGINNING of the step
            self.state = self.gastm_propagator.propagate_with_impulse(
                delta_v=delta_v_p,
                dt=self.dt,
                current_time=self.t,
                current_state=self.state
            )
        else:
            # 비선형 모드: 기존 방식 유지
            if np.any(delta_v_p):
                self.state[3:] += delta_v_p  # 순간 임펄스
            self._simulate_relative_motion()
        self.state = np.asarray(self.state, dtype=np.float32)
    
        # 4. 시간 및 스텝 업데이트
        self.t += self.dt
        self.step_count += 1
        
        # 5. 종료 조건 및 보상 계산
        terminated, truncated, termination_info = self.check_orbital_period_termination()
        done = terminated or truncated
        evader_reward, pursuer_reward, info = self._calculate_rewards(
            done, termination_info, delta_v_e_mag
        )

        # [중요] 기본 환경과 동일하게 보상 히스토리 기록 (analyze_results 안전성 보장)
        self.reward_history.append({"evader": evader_reward, "pursuer": pursuer_reward})

        # 6. 관측값 생성 및 반환
        observed_state = self.observe(self.state)
        normalized_obs = self._normalize_obs(observed_state, self.pursuer_last_action)
        
        # 정보 딕셔너리 업데이트
        info.update({
            "relative_distance_m": float(np.linalg.norm(self.state[:3])),
            "evader_dv_magnitude": delta_v_e_mag,
            "pursuer_dv_magnitude": float(np.linalg.norm(delta_v_p)),
            "total_evader_delta_v": self.total_delta_v_e,
            "nash_metric": self.nash_metric,
            "dynamics_mode": "GA_STM" if self.use_gastm else "Nonlinear",
            "control_applied": np.any(delta_v_p)  # 디버깅용
        })
        
        if "outcome" in termination_info:
            info["outcome"] = termination_info["outcome"]
        # 콜백 호환을 위한 거리 키 보강
        if 'initial_relative_distance' not in info and self.initial_relative_distance is not None:
            info['initial_relative_distance'] = float(self.initial_relative_distance)
        final_dist = float(np.linalg.norm(self.state[:3]))
        info.setdefault('final_relative_distance', final_dist)
        info.setdefault('final_distance', final_dist)
    
        return normalized_obs.astype(np.float32, copy=False), evader_reward, terminated, truncated, info

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
