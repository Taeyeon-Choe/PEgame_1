"""
PEgame 환경: GASTMPropagator를 사용한 STM 통합 버전
"""

import numpy as np
from typing import Dict, Tuple, Optional
from environment.pursuit_evasion_env import PursuitEvasionEnv
from orbital_mechanics.ga_stm_propagator import GASTMPropagator

class PursuitEvasionEnvGASTM(PursuitEvasionEnv):
    """
    Gim-Alfriend STM 전파 옵션을 포함한 확장된 PursuitEvasionEnv.
    기존 RL 트레이너와 호환되도록 수정되었습니다.
    """
    
    def __init__(self, config=None, use_gastm: bool = True):
        """
        환경 초기화

        Args:
            config: 환경 설정 객체
            use_gastm (bool): True이면 GA STM을, False이면 수치 적분을 사용합니다.
        """
        super().__init__(config)
        
        self.use_gastm = use_gastm
        self.gastm_propagator = None
        
        if self.use_gastm:
            print("INFO: Gim-Alfriend STM 전파기를 사용합니다.")
        else:
            print("INFO: 비선형 동역학 수치 적분기를 사용합니다.")

    def reset(self) -> np.ndarray:
        """환경을 리셋하고, 필요 시 GASTMPropagator를 재초기화합니다."""
        obs = super().reset()
        
        if self.use_gastm:
            # 새로운 초기 조건으로 GASTMPropagator 초기화
            self.gastm_propagator = GASTMPropagator(
                chief_orbit=self.evader_orbit,
                initial_relative_state=self.state,
                config=self.config
            )
        return obs

    def _simulate_relative_motion(self):
        """상대 운동 시뮬레이션 (모드에 따라 다른 방법 사용)"""
        if self.use_gastm:
            # GASTMPropagator를 사용하여 상태 전파
            if self.gastm_propagator:
                self.state = self.gastm_propagator.propagate(self.dt)
            else:
                # 안전 장치: Propagator가 초기화되지 않은 경우
                super()._simulate_relative_motion()
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
                self.evader_orbit, self.state
            )

    def step(self, normalized_action_e: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 스텝 실행 (기존 PEgame과 동일한 인터페이스)
        """
        # 1. 회피자(Evader) 액션 처리
        action_e = self._denormalize_action(normalized_action_e)
        delta_v_e = np.clip(action_e, -self.delta_v_emax, self.delta_v_emax)
        delta_v_e_mag = np.linalg.norm(delta_v_e)
        self.total_delta_v_e += delta_v_e_mag

        # 회피자 기동 적용 (궤도 변경)
        if np.any(delta_v_e):
            self._apply_evader_delta_v(delta_v_e)

        # 2. 추격자(Pursuer) 액션 처리 (내부적으로 계산)
        if self.step_count % self.k == 0:
            delta_v_p = self.compute_interception_strategy(self.state)
            self.pursuer_last_action = delta_v_p
        else:
            delta_v_p = np.zeros(3)

        # 3. 상태 전파
        if self.use_gastm and self.gastm_propagator:
            # STM 모드: 추격자 제어 적용 후 전파
            if np.any(delta_v_p):
                self.state = self.gastm_propagator.apply_pursuer_control(delta_v_p, self.dt)
            else:
                self.state = self.gastm_propagator.propagate(self.dt)
        else:
            # 비선형 모드: 추격자 제어 적용 후 전파
            self.state[3:] += delta_v_p
            self._simulate_relative_motion()

        # 4. 시간 및 스텝 업데이트
        self.t += self.dt
        self.step_count += 1
        
        # 5. 종료 조건 및 보상 계산
        done, termination_info = self.check_termination_conditions()
        evader_reward, pursuer_reward, info = self._calculate_rewards(
            done, termination_info, delta_v_e_mag
        )
        
        # 6. 관측값 생성 및 반환
        observed_state = self.observe(self.state)
        normalized_obs = self._normalize_obs(observed_state, self.pursuer_last_action)
        
        # 정보 딕셔너리 업데이트
        info.update({
            "relative_distance_m": np.linalg.norm(self.state[:3]),
            "evader_dv_magnitude": delta_v_e_mag,
            "pursuer_dv_magnitude": np.linalg.norm(delta_v_p),
            "total_evader_delta_v": self.total_delta_v_e,
            "nash_metric": self.nash_metric,
        })
        if "outcome" in termination_info:
            info["outcome"] = termination_info["outcome"]

        return normalized_obs, evader_reward, done, info
