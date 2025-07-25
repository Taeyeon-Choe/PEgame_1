"""
추격-회피 게임 환경 구현 (1궤도: 게임, 2궤도: 관찰, 3궤도: 게임)
"""

import numpy as np
import math
from scipy.integrate import solve_ivp
from collections import deque
import copy
import gym
from gym import spaces
from typing import Dict, Tuple, Any, Optional

from utils.constants import MU_EARTH, ENV_PARAMS, BUFFER_PARAMS, SAFETY_THRESHOLDS
from orbital_mechanics.orbit import ChiefOrbit
from orbital_mechanics.dynamics import relative_dynamics_evader_centered, safe_relative_dynamics
from orbital_mechanics.coordinate_transforms import (
    state_to_orbital_elements,
    orbital_elements_to_state,
    convert_orbital_elements_to_relative_state,
)


class CircularBuffer:
    """Simple circular buffer for limited history storage."""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = []
        self.index = 0

    def append(self, item):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
            self.index = (self.index + 1) % self.maxlen

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def get_all(self):
        return self.buffer.copy()


class PursuitEvasionEnv(gym.Env):
    """추격-회피 게임 환경"""

    def __init__(self, config=None):
        """
        환경 초기화

        Args:
            config: 환경 설정 객체
        """
        # 설정 로드
        if config is not None:
            self.config = config.environment
            self.debug_mode = config.debug_mode
            self.use_rk4 = self.config.use_rk4
        else:
            # 기본 설정 사용
            from config.settings import default_config

            self.config = default_config.environment
            self.debug_mode = default_config.debug_mode
            self.use_rk4 = self.config.use_rk4

        # 환경 파라미터 설정
        self._init_parameters()

        # 기준 궤도 설정
        self._init_chief_orbit()

        # 액션 및 관측 공간 정의
        self._init_spaces()

        # 상태 추적 변수 초기화
        self._init_tracking_variables()

        # 지능형 추격자 관련 변수
        self._init_pursuer_strategy()

        # 상태 변수
        self.step_count = 0
        self.t = 0
        self.state = None
        self.pursuer_last_action = np.zeros(3)
        self.reward_history = []

        # Zero-Sum 게임 관련
        self.zero_sum = True
        self.nash_metric = 0.0

    def _init_parameters(self):
        """환경 파라미터 초기화"""
        self.dt = self.config.dt
        self.k = self.config.k
        self.delta_v_emax = self.config.delta_v_emax
        self.delta_v_pmax = self.config.delta_v_pmax
        self.sigma_noise = self.config.sigma_noise
        self.sensor_noise_sigma = self.config.sensor_noise_sigma
        self.sensor_range = self.config.sensor_range
        self.capture_distance = self.config.capture_distance
        self.evasion_distance = self.config.evasion_distance
        self.c = self.config.c
        self.max_steps = self.config.max_steps
        self.max_delta_v_budget = self.config.max_delta_v_budget
        self.max_initial_separation = self.config.max_initial_separation

        # 정규화 스케일
        self.pos_scale = self.max_initial_separation
        self.vel_scale = self.delta_v_emax * 2

        # 버퍼 설정 - 3궤도 주기 기반으로 조정
        self.capture_buffer_steps = self.config.capture_buffer_steps
        self.evasion_buffer_steps = self.config.evasion_buffer_steps
        self.safety_buffer_steps = self.config.safety_buffer_steps

    def _init_chief_orbit(self, randomize: bool = False):
        """기준 궤도 초기화

        Args:
            randomize: True이면 반장축을 7000~8500km, 이심률을 0~0.5 범위에서
                무작위로 샘플링한다.
        """
        from config.settings import default_config

        orbit_config = default_config.orbit

        a = orbit_config.a
        e = orbit_config.e

        if randomize:
            # 매 에피소드마다 넓은 범위에서 궤도 매개변수를 샘플링
            a = np.random.uniform(7000e3, 8500e3)  # 7000~8500 km
            e = np.random.uniform(0.0, 0.5)

        self.evader_orbit = ChiefOrbit(
            a=a,
            e=e,
            i=orbit_config.i,
            RAAN=orbit_config.RAAN,
            omega=orbit_config.omega,
            M0=orbit_config.M0,
            mu=MU_EARTH,
        )
        
        # 궤도 주기 계산 및 3궤도 주기 버퍼 설정
        self.orbital_period = self.evader_orbit.period
        self.three_orbit_time = 3 * self.orbital_period
        self.three_orbit_steps = int(self.three_orbit_time / self.dt)
        
        # 각 궤도 주기의 스텝 수
        self.steps_per_orbit = int(self.orbital_period / self.dt)
        
        # 궤도별 모드 설정 (1궤도: 게임, 2궤도: 관찰, 3궤도: 게임)
        self.orbit_modes = ['game', 'observe', 'game']
        
        # 궤도별 버퍼 크기 재설정
        self.orbital_buffer_capture = max(int(0.5 * self.orbital_period / self.dt), 10)  # 0.5 궤도
        self.orbital_buffer_evasion = max(int(1.0 * self.orbital_period / self.dt), 20)  # 1 궤도
        self.orbital_buffer_safety = max(int(1.5 * self.orbital_period / self.dt), 30)   # 1.5 궤도
        
        if self.debug_mode:
            print(f"궤도 주기: {self.orbital_period:.1f}초 ({self.orbital_period/60:.1f}분)")
            print(f"3궤도 주기: {self.three_orbit_time:.1f}초 ({self.three_orbit_time/60:.1f}분)")
            print(f"각 궤도당 스텝: {self.steps_per_orbit}")
            print(f"버퍼 크기 - 포획: {self.orbital_buffer_capture} 스텝, "
                  f"회피: {self.orbital_buffer_evasion} 스텝, "
                  f"안전: {self.orbital_buffer_safety} 스텝")

    def _init_spaces(self):
        """액션 및 관측 공간 초기화"""
        # 정규화된 액션 사용
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 정규화된 관측 공간 (현재 궤도 모드 추가)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),  # 정규화된 위치(3), 속도(3), 추격자 최근 행동(3), 궤도 모드(1)
            dtype=np.float32,
        )

    def _init_tracking_variables(self):
        """상태 추적 변수 초기화"""
        # 추진제 사용량
        self.total_delta_v_e = 0

        # 버퍼 시간 관련
        self.capture_state_duration = 0
        self.evasion_state_duration = 0
        self.high_safety_duration = 0
        self.medium_safety_duration = 0

        # 상태 기록 큐 - 3궤도 주기 기반
        self.safety_score_history = deque(maxlen=self.orbital_buffer_safety)
        self.capture_status_history = deque(maxlen=self.orbital_buffer_capture)
        self.evasion_status_history = deque(maxlen=self.orbital_buffer_evasion)

        # 종료 조건 상세 기록
        self.termination_details = {}

        # 초기 조건 정보
        self.initial_evader_orbital_elements = None
        self.initial_pursuer_orbital_elements = None
        self.initial_relative_distance = None
        self.final_relative_distance = None

        # 물리적 궤도 업데이트 추적
        self.evader_impulse_history = []
        self.delta_v_e_sum = np.zeros(3)
        
        # 3궤도 주기 추적
        self.orbit_time_tracker = 0
        self.complete_orbits = 0
        self.current_orbit_mode = 'game'  # 현재 궤도 모드

    def _init_pursuer_strategy(self):
        """지능형 추격자 전략 초기화"""
        self.pursuer_strategy_history = deque(maxlen=100)
        self.pursuer_action_history = deque(maxlen=100)
        self.pursuer_success_history = deque(maxlen=100)
        self.cached_projection_dirs = []

    def _normalize_obs(
        self, state: np.ndarray, pursuer_action: np.ndarray
    ) -> np.ndarray:
        """관측값 정규화"""
        norm_pos = np.clip(state[:3] / self.pos_scale, -1.0, 1.0)
        norm_vel = np.clip(state[3:6] / self.vel_scale, -1.0, 1.0)
        norm_action = np.clip(pursuer_action / self.delta_v_pmax, -1.0, 1.0)
        
        # 현재 궤도 모드를 숫자로 변환 (game: 1, observe: 0)
        mode_value = 1.0 if self.current_orbit_mode == 'game' else 0.0

        return np.concatenate((norm_pos, norm_vel, norm_action, [mode_value]))

    def _denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """정규화된 액션을 실제 delta-v로 변환"""
        return normalized_action * self.delta_v_emax

    def initialize_with_accurate_dynamics(self) -> np.ndarray:
        """
        회피자(Evader) 궤도를 기준으로, 물리적으로 일관된
        상대 상태(relative_state)를 5 km 이내에서 샘플링
        """
        max_sep = 5_000.0
        a_e = self.evader_orbit.a
        e_e = self.evader_orbit.e
        i_e = self.evader_orbit.i
        RAAN_e = self.evader_orbit.RAAN
        omega_e = self.evader_orbit.omega
        M_e = self.evader_orbit.M0
        deg2rad = np.pi / 180.0

        # 재샘플링 루프
        for _ in range(100):
            # deputy COE 샘플
            a_p = a_e + np.random.uniform(-50.0, 50.0)
            e_p = max(1e-4, e_e + np.random.uniform(-7e-4, 7e-4))
            i_p = i_e + np.random.uniform(-0.05, 0.05) * deg2rad
            RAAN_p = RAAN_e + np.random.uniform(-0.05, 0.05) * deg2rad
            omega_p = omega_e + np.random.uniform(-0.05, 0.05) * deg2rad

            # 원하는 along-track 오프셋
            delta_s = np.random.uniform(-2000.0, 2000.0)
            delta_M = delta_s / a_e
            M_p = M_e + delta_M

            # COE → LVLH 변환
            rel = convert_orbital_elements_to_relative_state(
                a_e, e_e, i_e, RAAN_e, omega_e, M_e, a_p, e_p, i_p, RAAN_p, omega_p, M_p
            )

            if np.linalg.norm(rel[:3]) <= max_sep:
                relative_state = rel
                break
        else:
            # 실패 시 기본값
            print("WARNING: 5km 이내 궤도 샘플링 실패, 기본값 사용")
            relative_state = np.array([2000.0, 1000.0, 500.0, 0.0, 0.0, 0.0])

        # 기록용 속성 저장
        self.initial_evader_orbital_elements = dict(
            a=a_e, e=e_e, i=i_e, RAAN=RAAN_e, omega=omega_e, M=M_e
        )
        self.initial_pursuer_orbital_elements = dict(
            a=a_p, e=e_p, i=i_p, RAAN=RAAN_p, omega=omega_p, M=M_p
        )
        self.initial_relative_distance = np.linalg.norm(relative_state[:3])

        return relative_state

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        # 기준 궤도 재설정 (매 에피소드 무작위화)
        self._init_chief_orbit(randomize=True)

        # 무작위화된 궤도에 맞춰 버퍼 크기 재계산
        self.orbital_period = self.evader_orbit.period
        self.three_orbit_time = 3 * self.orbital_period
        self.three_orbit_steps = int(self.three_orbit_time / self.dt)
        self.steps_per_orbit = int(self.orbital_period / self.dt)

        self.orbital_buffer_capture = max(int(0.5 * self.orbital_period / self.dt), 10)
        self.orbital_buffer_evasion = max(int(1.0 * self.orbital_period / self.dt), 20)
        self.orbital_buffer_safety = max(int(1.5 * self.orbital_period / self.dt), 30)

        # 상태 초기화
        self.state = self.initialize_with_accurate_dynamics()
        self.pursuer_last_action = np.zeros(3)
        self.step_count = 0
        self.t = 0
        self.total_delta_v_e = 0
        self.reward_history = []
        self.final_relative_distance = None

        # Nash Equilibrium 관련 변수 초기화
        self.nash_metric = 0.0

        # 물리적 궤도 업데이트 추적 변수 초기화
        self.evader_impulse_history = []
        self.delta_v_e_sum = np.zeros(3)

        # 버퍼 시간 관련 변수 초기화
        self.capture_state_duration = 0
        self.evasion_state_duration = 0
        self.high_safety_duration = 0
        self.medium_safety_duration = 0
        
        # 3궤도 주기 추적 초기화
        self.orbit_time_tracker = 0
        self.complete_orbits = 0
        self.current_orbit_mode = 'game'  # 첫 궤도는 게임 모드

        # 히스토리 초기화
        self.safety_score_history = deque(maxlen=self.orbital_buffer_safety)
        self.capture_status_history = deque(maxlen=self.orbital_buffer_capture)
        self.evasion_status_history = deque(maxlen=self.orbital_buffer_evasion)
        self.termination_details = {}

        # 센서 노이즈 추가
        observed_state = self.observe(self.state)

        # 관측값 정규화
        normalized_obs = self._normalize_obs(observed_state, self.pursuer_last_action)

        return normalized_obs

    def observe(self, state: np.ndarray) -> np.ndarray:
        """센서 모델: 노이즈와 관측 제한이 있는 상태 반환"""
        if np.linalg.norm(state[:3]) > self.sensor_range:
            # 센서 범위 밖이면 노이즈가 큰 추정치 반환
            observed_state = state + np.random.normal(0, self.sensor_noise_sigma * 3, 6)
        else:
            # 센서 범위 내면 노이즈가 있는 측정값 반환
            observed_state = state + np.random.normal(0, self.sensor_noise_sigma, 6)

        return observed_state

    def compute_interception_strategy(self, state: np.ndarray) -> np.ndarray:
        """지능형 추격자의 추격 전략 계산 (개선된 버전)"""
        rho = state[:3].copy()
        v_rel = state[3:6].copy()
        rho_mag = np.linalg.norm(rho)

        if rho_mag < 1e-10:
            return np.zeros(3)

        prediction_time = np.clip(rho_mag / 1000.0, 1.0, 10.0) * self.dt

        if not hasattr(self, "_cached_directions"):
            self._cached_directions = []
            self._cache_counter = 0

        if self._cache_counter % 10 == 0:
            self._cached_directions = self.get_optimal_interception_directions(
                rho, v_rel, prediction_time
            )
        self._cache_counter += 1

        best_action = None
        min_future_distance = float("inf")

        for direction in self._cached_directions:
            for scale in [0.6, 0.8, 1.0]:
                test_action = scale * self.delta_v_pmax * direction
                future_vel = v_rel + test_action
                future_pos = rho + future_vel * prediction_time
                future_distance = np.linalg.norm(future_pos)

                cost = future_distance + 0.1 * scale * self.delta_v_pmax

                if cost < min_future_distance:
                    min_future_distance = cost
                    best_action = test_action

        if best_action is None:
            best_action = -self.delta_v_pmax * rho / rho_mag

        noise_scale = np.clip(
            0.05 + 0.1 * (rho_mag / self.capture_distance - 1), 0.05, 0.3
        )
        noise = np.random.normal(0, noise_scale * self.delta_v_pmax, 3)

        action_with_noise = best_action + noise
        action_with_noise = np.clip(
            action_with_noise, -self.delta_v_pmax, self.delta_v_pmax
        )

        return action_with_noise

    def get_optimal_interception_directions(
        self, rho: np.ndarray, v_rel: np.ndarray, prediction_time: float
    ) -> list:
        """최적 방향 세트 계산"""
        # 1. 직접 추격
        direct_dir = -rho / (np.linalg.norm(rho) + 1e-10)

        # 2. 예측 위치 방향
        future_pos = rho + v_rel * prediction_time
        predictive_dir = -future_pos / (np.linalg.norm(future_pos) + 1e-10)

        # 3. 상대 속도 반대 방향
        vel_dir = -v_rel / (np.linalg.norm(v_rel) + 1e-10)

        # 4. HCW 방향
        hcw_dir = np.zeros(3)
        rho_xy_norm = np.linalg.norm(rho[:2])
        if rho_xy_norm > 0:
            theta = np.arctan2(rho[1], rho[0])
            lead_angle = np.pi / 4
            hcw_dir[0] = -np.cos(theta + lead_angle)
            hcw_dir[1] = -np.sin(theta + lead_angle)
            if np.abs(rho[2]) > 0:
                hcw_dir[2] = -np.sign(rho[2]) * 0.3
            hcw_dir = hcw_dir / (np.linalg.norm(hcw_dir) + 1e-10)

        # 5. 이전 성공 전략
        history_dir = None
        if len(self.pursuer_success_history) > 0 and np.random.random() < 0.3:
            similar_states = []
            for i, (old_rho, old_success) in enumerate(self.pursuer_success_history):
                if (
                    old_success
                    and np.linalg.norm(rho - old_rho) < self.capture_distance * 3
                ):
                    similar_states.append(i)

            if similar_states:
                idx = np.random.choice(similar_states)
                history_dir = self.pursuer_action_history[idx].copy()
                history_dir = history_dir / (np.linalg.norm(history_dir) + 1e-10)

        all_directions = [direct_dir, predictive_dir, vel_dir, hcw_dir]
        if history_dir is not None:
            all_directions.append(history_dir)

        return all_directions

    def get_current_orbit_mode(self) -> str:
        """현재 궤도 주기에 따른 모드 반환"""
        # 현재 어느 궤도인지 계산 (0, 1, 2)
        current_orbit_index = self.complete_orbits % 3
        return self.orbit_modes[current_orbit_index]

    def step(self, normalized_action_e: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 실행 - 수정된 버전"""
        # 현재 궤도 모드 확인
        self.current_orbit_mode = self.get_current_orbit_mode()
        
        # 스텝 시작 시 현재 연료 사용량 저장 (delta-v 계산용)
        prev_total_delta_v_e = self.total_delta_v_e
        prev_pursuer_action = self.pursuer_last_action.copy() if hasattr(self, 'pursuer_last_action') else np.zeros(3)
        
        # 정규화된 액션을 실제 delta-v로 변환
        action_e = self._denormalize_action(normalized_action_e)
    
        # NaN 체크
        if np.isnan(action_e).any():
            if self.debug_mode:
                print(f"WARNING: action_e에 NaN 값 감지됨: {action_e}")
            action_e = np.zeros_like(action_e)
    
        # 현재 모드에 따라 delta-v 적용 여부 결정
        if self.current_orbit_mode == 'game':
            # 게임 모드: delta-v 정상 적용
            delta_v_e = np.clip(action_e, -self.delta_v_emax, self.delta_v_emax)
            delta_v_e_mag = np.linalg.norm(delta_v_e)
            self.total_delta_v_e += delta_v_e_mag
            
            # 추격자 행동 결정
            if self.step_count % self.k == 0:
                delta_v_p = self.compute_interception_strategy(self.state)
                self.pursuer_last_action = delta_v_p
                self.pursuer_action_history.append(delta_v_p.copy())
            else:
                delta_v_p = np.zeros(3)
        else:
            # 관찰 모드: delta-v 없이 propagate만
            delta_v_e = np.zeros(3)
            delta_v_e_mag = 0.0
            delta_v_p = np.zeros(3)
            
            if self.debug_mode and self.step_count % 100 == 0:
                print(f"관찰 모드 (궤도 {self.complete_orbits + 1}): delta-v 적용하지 않음")
    
        # 실제 이번 스텝의 delta-v 기록 (중요!)
        actual_evader_dv = delta_v_e_mag  # 실제 적용된 값
        actual_pursuer_dv = np.linalg.norm(delta_v_p)  # 실제 적용된 값
    
        # 궤도 업데이트 및 시뮬레이션 (기존 코드와 동일)
        if self.current_orbit_mode == 'game' and np.any(delta_v_e != 0):
            self._apply_evader_delta_v(delta_v_e)
    
        if self.current_orbit_mode == 'game' and np.any(delta_v_p != 0):
            self.state[3:] += delta_v_p
    
        self._simulate_relative_motion()
    
        # 시간 및 스텝 업데이트
        self.t += self.dt
        self.step_count += 1
        
        # 궤도 주기 추적
        self.orbit_time_tracker += self.dt
        if self.orbit_time_tracker >= self.orbital_period:
            self.complete_orbits += 1
            self.orbit_time_tracker -= self.orbital_period
            if self.debug_mode:
                new_mode = self.get_current_orbit_mode()
                print(f"궤도 {self.complete_orbits} 완료, 다음 모드: {new_mode}")
    
        # 종료 조건 확인
        done, termination_info = self.check_orbital_period_termination()
    
        # 보상 계산
        evader_reward, pursuer_reward, info = self._calculate_rewards(
            done, termination_info, delta_v_e_mag
        )
    
        # Nash Equilibrium 메트릭 업데이트
        self._update_nash_metric()
    
        # 보상 히스토리 저장
        self.reward_history.append({"evader": evader_reward, "pursuer": pursuer_reward})
    
        # 관측값 생성
        observed_state = self.observe(self.state)
        normalized_obs = self._normalize_obs(observed_state, self.pursuer_last_action)
    
        # ========== 수정된 부분: 실제 delta-v 값 사용 ==========
        # 현재 상태 정보
        current_relative_distance = np.linalg.norm(self.state[:3])
        
        # 기존 info 업데이트
        info.update({
            # 콜백이 기대하는 키들 - 실제 적용된 delta-v 사용
            "relative_distance_m": current_relative_distance,
            "evader_dv_magnitude": actual_evader_dv,  # 수정: 실제 값 사용
            "pursuer_dv_magnitude": actual_pursuer_dv,  # 수정: 실제 값 사용
            
            # 추가 유용한 정보
            "total_evader_delta_v": self.total_delta_v_e,
            "remaining_fuel": self.max_delta_v_budget - self.total_delta_v_e,
            "fuel_fraction_used": self.total_delta_v_e / self.max_delta_v_budget,
            
            # 기존 정보들도 유지
            "nash_metric": self.nash_metric,
            "evader_delta_v_sum": self.delta_v_e_sum.copy(),
            "evader_impulse_count": len(self.evader_impulse_history),
            "complete_orbits": self.complete_orbits,
            "orbital_phase": self.orbit_time_tracker / self.orbital_period,
            "current_orbit_mode": self.current_orbit_mode,
            
            # 추가: 모드별 delta-v 정보
            "is_game_mode": self.current_orbit_mode == 'game',
            "pursuer_action_step": self.step_count % self.k == 0,  # 추격자가 행동한 스텝인지
        })
    
        # 종료 시 추가 정보
        if "outcome" in termination_info:
            info["termination_type"] = termination_info["outcome"]
            info["outcome"] = termination_info["outcome"]  # 콜백 호환성
            if "buffer_time" in termination_info:
                info["buffer_time"] = termination_info["buffer_time"]
            if "orbit_consistency" in termination_info:
                info["orbit_consistency"] = termination_info["orbit_consistency"]

        if done:
            # 최종 상대거리 저장
            self.final_relative_distance = current_relative_distance
            
            # 종료 정보 업데이트
            info.update({
                # 기본 종료 정보
                "outcome": termination_info.get("outcome", "unknown"),
                "termination_details": termination_info,
                
                # 궤도 정보
                "initial_evader_orbital_elements": self.initial_evader_orbital_elements,
                "initial_pursuer_orbital_elements": self.initial_pursuer_orbital_elements,
                "initial_relative_distance": self.initial_relative_distance,
                "final_relative_distance": self.final_relative_distance,
                
                # 보상 정보
                "evader_reward": evader_reward,
                "pursuer_reward": pursuer_reward,
                "episode": {
                    "r": evader_reward,  # 총 보상
                    "l": self.step_count,  # 에피소드 길이
                }
            })
    
        return normalized_obs, evader_reward, done, info

    def _apply_evader_delta_v(self, delta_v_e: np.ndarray):
        """회피자 델타-V 적용 및 궤도 업데이트"""
        if not np.any(delta_v_e != 0):
            return

        # 현재 회피자 절대 상태
        r_evader, v_evader = self.evader_orbit.get_position_velocity(self.t)

        # LVLH → ECI 변환 행렬 계산
        h_evader = np.cross(r_evader, v_evader)
        h_evader_norm = np.linalg.norm(h_evader)
        r_evader_norm = np.linalg.norm(r_evader)

        x_lvlh = r_evader / r_evader_norm
        z_lvlh = h_evader / h_evader_norm
        y_lvlh = np.cross(z_lvlh, x_lvlh)
        y_lvlh = y_lvlh / np.linalg.norm(y_lvlh)

        R_lvlh_to_eci = np.vstack((x_lvlh, y_lvlh, z_lvlh)).T
        delta_v_e_eci = R_lvlh_to_eci @ delta_v_e

        # 누적 델타-V 기록
        self.delta_v_e_sum += delta_v_e

        # 임펄스 기록
        self.evader_impulse_history.append(
            {
                "t": self.t,
                "delta_v_lvlh": delta_v_e.copy(),
                "delta_v_eci": delta_v_e_eci.copy(),
                "r_evader": r_evader.copy(),
                "v_evader": v_evader.copy(),
                "orbit_number": self.complete_orbits + 1,
                "orbit_mode": self.current_orbit_mode,
            }
        )

        # 새 회피자 속도
        v_evader_new = v_evader + delta_v_e_eci

        # 새 궤도 요소 계산
        a_e_new, e_e_new, i_e_new, RAAN_e_new, omega_e_new, M_e_new = (
            state_to_orbital_elements(r_evader, v_evader_new)
        )

        # 추격자 절대 상태 계산
        r_pursuer = r_evader + R_lvlh_to_eci @ self.state[:3]

        # 새 LVLH 프레임 계산
        h_evader_new = np.cross(r_evader, v_evader_new)
        omega_lvlh_new = h_evader_new / (r_evader_norm**2 + 1e-12)

        z_lvlh_new = h_evader_new / (np.linalg.norm(h_evader_new) + 1e-12)
        y_lvlh_new = np.cross(z_lvlh_new, x_lvlh)
        y_lvlh_new /= np.linalg.norm(y_lvlh_new) + 1e-12
        R_lvlh_to_eci_new = np.vstack((x_lvlh, y_lvlh_new, z_lvlh_new)).T

        # 추격자 속도 계산
        v_pursuer = (
            v_evader_new
            + R_lvlh_to_eci_new @ self.state[3:6]
            + np.cross(omega_lvlh_new, R_lvlh_to_eci_new @ self.state[:3])
        )

        # 회피자 궤도 업데이트
        self.evader_orbit = ChiefOrbit(
            a=a_e_new,
            e=e_e_new,
            i=i_e_new,
            RAAN=RAAN_e_new,
            omega=omega_e_new,
            M0=M_e_new,
            mu=MU_EARTH,
        )

        # 새 LVLH에서 추격자 상대 상태 계산
        a_p, e_p, i_p, RAAN_p, omega_p, M_p = state_to_orbital_elements(
            r_pursuer, v_pursuer
        )
        self.state = convert_orbital_elements_to_relative_state(
            a_e_new,
            e_e_new,
            i_e_new,
            RAAN_e_new,
            omega_e_new,
            M_e_new,
            a_p,
            e_p,
            i_p,
            RAAN_p,
            omega_p,
            M_p,
        )

    def _rk4_step(self):
        """RK4 방법으로 상태 업데이트 (개선된 버전)"""
        dt = self.dt
        s = self.state.copy()
        t = self.t

        try:
            max_steps = 4
            dt_sub = dt / max_steps

            for _ in range(max_steps):
                k1 = np.array(relative_dynamics_evader_centered(t, s, self.evader_orbit))
                if np.isnan(k1).any():
                    if self.debug_mode:
                        print("WARNING: RK4 k1에서 NaN 발생")
                    break

                k2 = np.array(
                    relative_dynamics_evader_centered(
                        t + 0.5 * dt_sub, s + 0.5 * dt_sub * k1, self.evader_orbit
                    )
                )
                if np.isnan(k2).any():
                    if self.debug_mode:
                        print("WARNING: RK4 k2에서 NaN 발생")
                    break

                k3 = np.array(
                    relative_dynamics_evader_centered(
                        t + 0.5 * dt_sub, s + 0.5 * dt_sub * k2, self.evader_orbit
                    )
                )
                if np.isnan(k3).any():
                    if self.debug_mode:
                        print("WARNING: RK4 k3에서 NaN 발생")
                    break

                k4 = np.array(
                    relative_dynamics_evader_centered(
                        t + dt_sub, s + dt_sub * k3, self.evader_orbit
                    )
                )
                if np.isnan(k4).any():
                    if self.debug_mode:
                        print("WARNING: RK4 k4에서 NaN 발생")
                    break

                s = s + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                t += dt_sub

            if not np.isnan(s).any():
                self.state = s
            else:
                # 간단한 선형 업데이트로 대체
                self.state[0:3] += self.state[3:6] * self.dt

        except Exception as e:
            if self.debug_mode:
                print(f"RK4 오류: {e}")
            self.state[0:3] += self.state[3:6] * self.dt

    def _simulate_relative_motion(self):
        """상대 운동 시뮬레이션 (개선된 버전)"""
        if np.isnan(self.state).any():
            if self.debug_mode:
                print("WARNING: 시뮬레이션 전 state에 NaN 값 감지됨")
            # NaN을 안전한 값으로 대체
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=1e6, neginf=-1e6)
            # 초기 분리 거리의 절반으로 위치 재설정
            if np.linalg.norm(self.state[:3]) < 1e-6:
                self.state[:3] = np.array([self.max_initial_separation * 0.5, 0, 0])

        try:
            if self.use_rk4:
                self._rk4_step()
            else:
                # ODE 솔버 사용
                max_attempts = 3
                dt_factor = 1.0

                for attempt in range(max_attempts):
                    try:
                        # 안전한 동역학 함수 사용
                        sol = solve_ivp(
                            safe_relative_dynamics,
                            [self.t, self.t + self.dt * dt_factor],
                            self.state,
                            args=(self.evader_orbit,),
                            method="DOP853",
                            rtol=1e-6,
                            atol=1e-6,
                            dense_output=True,
                        )

                        if sol.success and not np.isnan(sol.y[:, -1]).any():
                            self.state = sol.y[:, -1]
                            break
                        else:
                            dt_factor *= 0.5
                            if self.debug_mode:
                                print(f"적분 실패, 시간 간격 감소: {dt_factor}")
                    except Exception as e:
                        dt_factor *= 0.5
                        if self.debug_mode:
                            print(f"적분 오류 (시도 {attempt+1}): {e}")
                else:
                    if self.debug_mode:
                        print("WARNING: 적분 실패, 선형 근사 사용")
                    self.state[0:3] += self.state[3:6] * self.dt

        except Exception as e:
            if self.debug_mode:
                print(f"시뮬레이션 오류 발생: {e}")
            self.state[0:3] += self.state[3:6] * self.dt

        # 최종 NaN 체크
        if np.isnan(self.state).any():
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=1e6, neginf=-1e6)

    def check_orbital_period_termination(self) -> Tuple[bool, Dict]:
        """3궤도 주기를 고려한 종료 조건 확인"""
        rho_mag = np.linalg.norm(self.state[:3])

        # 1. 포획 상태 추적 (게임 모드에서만)
        if self.current_orbit_mode == 'game':
            is_captured = rho_mag < self.capture_distance
            self.capture_status_history.append(is_captured)

            # 포획 상태가 충분히 지속되었는지 확인
            if len(self.capture_status_history) >= self.orbital_buffer_capture:
                capture_ratio = sum(self.capture_status_history) / len(self.capture_status_history)
                
                # 80% 이상의 시간 동안 포획 상태 유지 시 종료
                if capture_ratio > 0.8:
                    self.termination_details = {
                        "outcome": "captured",
                        "evader_reward": -10,
                        "pursuer_reward": 10,
                        "buffer_time": self.orbital_buffer_capture * self.dt,
                        "relative_distance": rho_mag,
                        "orbit_consistency": f"{capture_ratio:.1%} over {self.complete_orbits} orbits",
                        "game_orbits": sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game'),
                    }
                    return True, self.termination_details

        # 2. 추진제 소진
        if self.total_delta_v_e > self.max_delta_v_budget:
            self.termination_details = {
                "outcome": "fuel_depleted",
                "evader_reward": -5,
                "pursuer_reward": 5,
                "relative_distance": rho_mag,
                "delta_v_used": self.total_delta_v_e,
                "complete_orbits": self.complete_orbits,
                "game_orbits": sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game'),
            }
            return True, self.termination_details

        # 3. 최대 단계 초과 (최소 3궤도는 보장)
        if self.step_count >= self.max_steps or self.complete_orbits >= 3:
            norm_distance = min(rho_mag / self.evasion_distance, 1.0)
            self.termination_details = {
                "outcome": "max_steps_reached",
                "evader_reward": 5 * norm_distance,
                "pursuer_reward": -5 * norm_distance,
                "relative_distance": rho_mag,
                "complete_orbits": self.complete_orbits,
                "game_orbits": sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game'),
            }
            return True, self.termination_details

        # 4. 회피 분석 (게임 모드에서의 성과를 중심으로)
        if rho_mag > self.evasion_distance and self.current_orbit_mode == 'game':
            self.evasion_status_history.append(True)

            if len(self.evasion_status_history) >= self.orbital_buffer_evasion:
                evasion_ratio = sum(self.evasion_status_history) / len(self.evasion_status_history)
                
                # 90% 이상의 시간 동안 회피 거리 유지 시
                if evasion_ratio > 0.9 and self.complete_orbits >= 1:
                    # 안전도 분석 수행
                    safety_analysis = self._perform_safety_analysis()
                    safety_score = safety_analysis["overall_safety_score"]

                    self.safety_score_history.append(safety_score)

                    if len(self.safety_score_history) >= self.orbital_buffer_safety:
                        min_safety_score = min(self.safety_score_history)
                        avg_safety_score = sum(self.safety_score_history) / len(self.safety_score_history)

                        # 1.5궤도 이상 안전도 점수 유지 시
                        if min_safety_score > SAFETY_THRESHOLDS["permanent_evasion"] and self.complete_orbits >= 2:
                            self.termination_details = {
                                "outcome": "permanent_evasion",
                                "evader_reward": 10,
                                "pursuer_reward": -10,
                                "safety_score": min_safety_score,
                                "avg_safety_score": avg_safety_score,
                                "buffer_time": self.orbital_buffer_safety * self.dt,
                                "safety_analysis": safety_analysis,
                                "orbit_consistency": f"{evasion_ratio:.1%} over {self.complete_orbits} orbits",
                                "game_orbits": sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game'),
                            }
                            return True, self.termination_details

                        elif min_safety_score > SAFETY_THRESHOLDS["conditional_evasion"] and self.complete_orbits >= 1:
                            self.termination_details = {
                                "outcome": "conditional_evasion",
                                "evader_reward": 7,
                                "pursuer_reward": -7,
                                "safety_score": min_safety_score,
                                "avg_safety_score": avg_safety_score,
                                "buffer_time": self.orbital_buffer_safety * self.dt,
                                "safety_analysis": safety_analysis,
                                "orbit_consistency": f"{evasion_ratio:.1%} over {self.complete_orbits} orbits",
                                "game_orbits": sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game'),
                            }
                            return True, self.termination_details
        else:
            # 회피 거리 미달 시 히스토리 일부만 제거 (부드러운 전환)
            if len(self.evasion_status_history) > 0 and self.current_orbit_mode == 'game':
                self.evasion_status_history.popleft()
            if len(self.safety_score_history) > 0 and self.current_orbit_mode == 'game':
                self.safety_score_history.popleft()

        return False, {}

    def _perform_safety_analysis(self) -> Dict:
        """안전도 분석 수행 (간단한 버전)"""
        # 여기서는 간단한 분석만 수행
        # 실제로는 더 복잡한 궤도 분석이 필요

        rho_mag = np.linalg.norm(self.state[:3])
        distance_safety = min(1.0, rho_mag / (self.capture_distance * 10))

        remaining_dv = self.max_delta_v_budget - self.total_delta_v_e
        fuel_safety = min(1.0, remaining_dv / 50.0)  # 50 m/s 기준

        overall_safety = 0.6 * distance_safety + 0.4 * fuel_safety

        return {
            "overall_safety_score": overall_safety,
            "distance_safety": distance_safety,
            "fuel_safety": fuel_safety,
            "remaining_delta_v": remaining_dv,
            "complete_orbits": self.complete_orbits,
            "game_orbits": sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game'),
        }

    def _calculate_rewards(self, done: bool, termination_info: Dict, delta_v_e_mag: float) -> Tuple[float, float, Dict]:
        """보상 계산 - 연료 효율성 강조"""
        if done:
            evader_reward = termination_info["evader_reward"]
            pursuer_reward = termination_info["pursuer_reward"]
            info = termination_info
            self.final_relative_distance = np.linalg.norm(self.state[:3])
            info['final_distance'] = self.final_relative_distance
        else:
            rho_mag = np.linalg.norm(self.state[:3])
            
            if self.current_orbit_mode == 'game':
                # 거리 보상
                normalized_distance = min(rho_mag / self.capture_distance, 100)
                distance_reward = 0.01 * normalized_distance
                
                # 속도 방향 보상
                v_rel = self.state[3:6]
                v_rel_mag = np.linalg.norm(v_rel)
                if v_rel_mag > 0 and rho_mag > 0:
                    angle = np.arccos(
                        np.clip(np.dot(v_rel, self.state[:3]) / (v_rel_mag * rho_mag), -1.0, 1.0)
                    )
                    tangential_reward = 0.005 * np.sin(angle)
                else:
                    tangential_reward = 0
                
                # 제어 비용 (증가)
                control_cost = self.c * delta_v_e_mag
                
                # 연료 보존 보너스 추가
                fuel_remaining = self.max_delta_v_budget - self.total_delta_v_e
                fuel_bonus = 0.001 * fuel_remaining if fuel_remaining > 0 else 0
                
                # 회피 보너스
                dodge_bonus = 0
                if self.step_count % self.k == 0 and rho_mag > self.capture_distance * 3:
                    dodge_bonus = 0.1
                
                evader_reward = (
                    distance_reward + tangential_reward - control_cost + 
                    fuel_bonus + dodge_bonus
                )
                pursuer_reward = -evader_reward
            else:
                # 관찰 모드: 거리 유지에 대한 작은 보너스
                evader_reward = 0.001 * min(rho_mag / self.capture_distance, 10)
                pursuer_reward = -evader_reward
            
            info = {
                "evader_reward": evader_reward, 
                "pursuer_reward": pursuer_reward,
                "orbit_mode": self.current_orbit_mode,
            }
        
        return evader_reward, pursuer_reward, info

    def _update_nash_metric(self):
        """Nash Equilibrium 메트릭 업데이트"""
        if len(self.reward_history) >= 100:
            recent_rewards = [r.get("evader", 0) for r in self.reward_history[-100:]]
            reward_std = np.std(recent_rewards)
            self.nash_metric = 1.0 / (1.0 + reward_std)
        else:
            self.nash_metric = 0.5

    def analyze_results(
        self, states: np.ndarray, actions_e: np.ndarray, actions_p: np.ndarray
    ) -> Dict:
        """결과 분석"""
        final_distance = np.linalg.norm(states[-1, :3])
        total_e_delta_v = np.sum([np.linalg.norm(a) for a in actions_e])
        total_p_delta_v = np.sum(
            [np.linalg.norm(a) for a in actions_p if np.any(a != 0)]
        )

        # 보상 계산
        if isinstance(self.reward_history[-1], dict):
            evader_rewards = [r.get("evader", 0) for r in self.reward_history]
            pursuer_rewards = [r.get("pursuer", 0) for r in self.reward_history]
            mean_evader_reward = np.mean(evader_rewards)
            mean_pursuer_reward = np.mean(pursuer_rewards)
        else:
            mean_evader_reward = np.mean(self.reward_history)
            mean_pursuer_reward = -mean_evader_reward

        success = final_distance > self.capture_distance

        # 게임 궤도 수 계산
        game_orbits = sum(1 for i in range(self.complete_orbits + 1) if self.orbit_modes[i % 3] == 'game')

        result = {
            "final_distance_m": final_distance,
            "evader_total_delta_v_ms": total_e_delta_v,
            "pursuer_total_delta_v_ms": total_p_delta_v,
            "mean_evader_reward": mean_evader_reward,
            "mean_pursuer_reward": mean_pursuer_reward,
            "nash_metric": self.nash_metric,
            "success": success,
            "trajectory_length": len(states),
            "complete_orbits": self.complete_orbits,
            "game_orbits": game_orbits,
            "observation_orbits": self.complete_orbits - game_orbits,
        }

        return result
