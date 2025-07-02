"""
추격-회피 게임 환경 구현
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
from orbital_mechanics.dynamics import relative_dynamics_evader_centered
from orbital_mechanics.coordinate_transforms import (
    state_to_orbital_elements,
    orbital_elements_to_state,
    convert_orbital_elements_to_relative_state,
)


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

        # 버퍼 설정
        self.capture_buffer_steps = self.config.capture_buffer_steps
        self.evasion_buffer_steps = self.config.evasion_buffer_steps
        self.safety_buffer_steps = self.config.safety_buffer_steps

    def _init_chief_orbit(self):
        """기준 궤도 초기화"""
        from config.settings import default_config

        orbit_config = default_config.orbit

        self.evader_orbit = ChiefOrbit(
            a=orbit_config.a,
            e=orbit_config.e,
            i=orbit_config.i,
            RAAN=orbit_config.RAAN,
            omega=orbit_config.omega,
            M0=orbit_config.M0,
            mu=MU_EARTH,
        )

    def _init_spaces(self):
        """액션 및 관측 공간 초기화"""
        # 정규화된 액션 사용
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 정규화된 관측 공간
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),  # 정규화된 위치(3), 속도(3), 추격자 최근 행동(3)
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

        # 상태 기록 큐
        self.safety_score_history = deque(maxlen=self.safety_buffer_steps)
        self.capture_status_history = deque(maxlen=self.capture_buffer_steps)
        self.evasion_status_history = deque(maxlen=self.evasion_buffer_steps)

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

        return np.concatenate((norm_pos, norm_vel, norm_action))

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
        # 기준 궤도 재설정
        from config.settings import default_config

        orbit_config = default_config.orbit

        self.evader_orbit = ChiefOrbit(
            a=orbit_config.a,
            e=orbit_config.e,
            i=orbit_config.i,
            RAAN=orbit_config.RAAN,
            omega=orbit_config.omega,
            M0=orbit_config.M0,
            mu=MU_EARTH,
        )

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

        # 히스토리 초기화
        self.safety_score_history = deque(maxlen=self.safety_buffer_steps)
        self.capture_status_history = deque(maxlen=self.capture_buffer_steps)
        self.evasion_status_history = deque(maxlen=self.evasion_buffer_steps)
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
        """지능형 추격자의 추격 전략 계산"""
        rho = state[:3]  # 상대 위치
        v_rel = state[3:6]  # 상대 속도
        rho_mag = np.linalg.norm(rho)

        # 적응형 예측 시간
        prediction_time = min(20, max(1, int(rho_mag / 500))) * self.dt

        # 다양한 추격 방향 생성
        candidate_directions = self.get_optimal_interception_directions(
            rho, v_rel, prediction_time
        )

        # 각 방향에 대한 평가
        best_action = None
        min_distance = float("inf")

        for direction in candidate_directions:
            for scale in [0.7, 0.85, 1.0]:
                test_action = scale * self.delta_v_pmax * direction

                # 간단한 선형 예측
                rel_speed = v_rel + test_action
                future_pos = rho + rel_speed * prediction_time
                future_distance = np.linalg.norm(future_pos)

                if future_distance < min_distance:
                    min_distance = future_distance
                    best_action = test_action

        # 전략 기록
        self.pursuer_strategy_history.append((rho.copy(), best_action.copy()))

        # 노이즈 추가
        noise_scale = max(0.1, min(0.3, rho_mag / self.capture_distance * 0.1))
        action_with_noise = best_action + np.random.normal(
            0, noise_scale * self.delta_v_pmax, 3
        )
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

    def step(
        self, normalized_action_e: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 실행"""
        # 정규화된 액션을 실제 delta-v로 변환
        action_e = self._denormalize_action(normalized_action_e)

        # NaN 체크
        if np.isnan(action_e).any():
            if self.debug_mode:
                print(f"WARNING: action_e에 NaN 값 감지됨: {action_e}")
            action_e = np.zeros_like(action_e)

        # 회피자 행동 제한
        delta_v_e = np.clip(action_e, -self.delta_v_emax, self.delta_v_emax)

        # 추진제 예산 업데이트
        delta_v_e_mag = np.linalg.norm(delta_v_e)
        self.total_delta_v_e += delta_v_e_mag

        # 추격자 행동 결정
        if self.step_count % self.k == 0:
            delta_v_p = self.compute_interception_strategy(self.state)
            self.pursuer_last_action = delta_v_p
            self.pursuer_action_history.append(delta_v_p.copy())
        else:
            delta_v_p = np.zeros(3)

        # 궤도 업데이트 (회피자 델타-V 적용)
        self._apply_evader_delta_v(delta_v_e)

        # 추격자 델타-V 적용 (상대 속도에 직접)
        if np.any(delta_v_p != 0):
            self.state[3:] += delta_v_p

        # 상대 운동 시뮬레이션
        self._simulate_relative_motion()

        # 시간 및 스텝 업데이트
        self.t += self.dt
        self.step_count += 1

        # 종료 조건 확인
        done, termination_info = self.check_time_buffered_termination()

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

        # 추가 정보
        info.update(
            {
                "nash_metric": self.nash_metric,
                "evader_delta_v_sum": self.delta_v_e_sum.copy(),
                "evader_impulse_count": len(self.evader_impulse_history),
            }
        )

        if "outcome" in termination_info:
            info["termination_type"] = termination_info["outcome"]
            if "buffer_time" in termination_info:
                info["buffer_time"] = termination_info["buffer_time"]

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
        """RK4 방법으로 상태 업데이트"""
        dt = self.dt
        s = self.state
        t = self.t

        k1 = np.array(relative_dynamics_evader_centered(t, s, self.evader_orbit))
        k2 = np.array(
            relative_dynamics_evader_centered(
                t + 0.5 * dt, s + 0.5 * dt * k1, self.evader_orbit
            )
        )
        k3 = np.array(
            relative_dynamics_evader_centered(
                t + 0.5 * dt, s + 0.5 * dt * k2, self.evader_orbit
            )
        )
        k4 = np.array(
            relative_dynamics_evader_centered(t + dt, s + dt * k3, self.evader_orbit)
        )

        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _simulate_relative_motion(self):
        """상대 운동 시뮬레이션"""
        # NaN 체크
        if np.isnan(self.state).any():
            if self.debug_mode:
                print(f"WARNING: 시뮬레이션 전 state에 NaN 값 감지됨: {self.state}")
            self.state = np.zeros_like(self.state)
            self.state[0] = self.max_initial_separation * 0.5

        try:
            if self.use_rk4:
                self._rk4_step()
            else:
                sol = solve_ivp(
                    relative_dynamics_evader_centered,
                    [self.t, self.t + self.dt],
                    self.state,
                    args=(self.evader_orbit,),
                    method="RK45",
                    rtol=1e-6,
                    atol=1e-6,
                )

                if len(sol.y[0]) == 0 or np.isnan(sol.y[:, -1]).any():
                    if self.debug_mode:
                        print(f"WARNING: 시뮬레이션 결과가 비정상")
                    # 선형 변화만 적용
                    self.state[0:3] += self.state[3:6] * self.dt
                else:
                    self.state = sol.y[:, -1]

        except Exception as e:
            if self.debug_mode:
                print(f"시뮬레이션 오류 발생: {e}")
            self.state[0:3] += self.state[3:6] * self.dt

    def check_time_buffered_termination(self) -> Tuple[bool, Dict]:
        """버퍼 시간을 고려한 종료 조건 확인"""
        rho_mag = np.linalg.norm(self.state[:3])

        # 1. 포획 상태 추적
        is_captured = rho_mag < self.capture_distance
        self.capture_status_history.append(is_captured)

        if len(self.capture_status_history) >= self.capture_buffer_steps and all(
            self.capture_status_history
        ):
            self.termination_details = {
                "outcome": "captured",
                "evader_reward": -10,
                "pursuer_reward": 10,
                "buffer_time": self.capture_buffer_steps * self.dt,
                "relative_distance": rho_mag,
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
            }
            return True, self.termination_details

        # 3. 최대 단계 초과
        if self.step_count >= self.max_steps:
            norm_distance = min(rho_mag / self.evasion_distance, 1.0)
            self.termination_details = {
                "outcome": "max_steps_reached",
                "evader_reward": 5 * norm_distance,
                "pursuer_reward": -5 * norm_distance,
                "relative_distance": rho_mag,
            }
            return True, self.termination_details

        # 4. 회피 분석
        if rho_mag > self.evasion_distance:
            self.evasion_status_history.append(True)

            if len(self.evasion_status_history) >= self.evasion_buffer_steps and all(
                self.evasion_status_history
            ):

                # 안전도 분석 수행
                safety_analysis = self._perform_safety_analysis()
                safety_score = safety_analysis["overall_safety_score"]

                self.safety_score_history.append(safety_score)

                if len(self.safety_score_history) >= self.safety_buffer_steps:
                    min_safety_score = min(self.safety_score_history)

                    if min_safety_score > SAFETY_THRESHOLDS["permanent_evasion"]:
                        self.termination_details = {
                            "outcome": "permanent_evasion",
                            "evader_reward": 10,
                            "pursuer_reward": -10,
                            "safety_score": min_safety_score,
                            "buffer_time": self.safety_buffer_steps * self.dt,
                            "safety_analysis": safety_analysis,
                        }
                        return True, self.termination_details

                    elif min_safety_score > SAFETY_THRESHOLDS["conditional_evasion"]:
                        self.termination_details = {
                            "outcome": "conditional_evasion",
                            "evader_reward": 7,
                            "pursuer_reward": -7,
                            "safety_score": min_safety_score,
                            "buffer_time": self.safety_buffer_steps * self.dt,
                            "safety_analysis": safety_analysis,
                        }
                        return True, self.termination_details
        else:
            self.evasion_status_history.clear()
            self.safety_score_history.clear()

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
        }

    def _calculate_rewards(
        self, done: bool, termination_info: Dict, delta_v_e_mag: float
    ) -> Tuple[float, float, Dict]:
        """보상 계산"""
        if done:
            evader_reward = termination_info["evader_reward"]
            pursuer_reward = termination_info["pursuer_reward"]
            info = termination_info
            self.final_relative_distance = np.linalg.norm(self.state[:3])
        else:
            # 단계별 보상
            rho_mag = np.linalg.norm(self.state[:3])

            # 거리 보상
            normalized_distance = min(rho_mag / self.capture_distance, 100)
            distance_reward = 0.01 * normalized_distance

            # 속도 방향 보상
            v_rel = self.state[3:6]
            v_rel_mag = np.linalg.norm(v_rel)
            if v_rel_mag > 0 and rho_mag > 0:
                angle = np.arccos(
                    np.clip(
                        np.dot(v_rel, self.state[:3]) / (v_rel_mag * rho_mag), -1.0, 1.0
                    )
                )
                tangential_reward = 0.005 * np.sin(angle)
            else:
                tangential_reward = 0

            control_cost = self.c * delta_v_e_mag

            # 회피 보너스
            dodge_bonus = 0
            if self.step_count % self.k == 0 and rho_mag > self.capture_distance * 3:
                dodge_bonus = 0.1

            evader_reward = (
                distance_reward + tangential_reward - control_cost + dodge_bonus
            )
            pursuer_reward = -evader_reward

            info = {"evader_reward": evader_reward, "pursuer_reward": pursuer_reward}

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

        result = {
            "final_distance_m": final_distance,
            "evader_total_delta_v_ms": total_e_delta_v,
            "pursuer_total_delta_v_ms": total_p_delta_v,
            "mean_evader_reward": mean_evader_reward,
            "mean_pursuer_reward": mean_pursuer_reward,
            "nash_metric": self.nash_metric,
            "success": success,
            "trajectory_length": len(states),
        }

        return result
