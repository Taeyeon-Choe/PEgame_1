# environment/pursuit_evasion_env_improved.py
"""
개선된 추격-회피 게임 환경 (주요 수정사항만)
"""

# 기존 코드의 _simulate_relative_motion 메서드 개선
def _simulate_relative_motion(self):
    """상대 운동 시뮬레이션 (개선된 버전)"""
    # NaN 체크 및 복구
    if np.isnan(self.state).any():
        if self.debug_mode:
            print(f"WARNING: 시뮬레이션 전 state에 NaN 값 감지됨")
        # 안전한 기본값으로 복구
        self.state = np.array([
            self.max_initial_separation * 0.5, 0, 0,  # 위치
            0, 0, 0  # 속도
        ])

    try:
        if self.use_rk4:
            self._rk4_step()
        else:
            # 적응적 시간 간격 사용
            max_attempts = 3
            dt_factor = 1.0
            
            for attempt in range(max_attempts):
                try:
                    sol = solve_ivp(
                        relative_dynamics_evader_centered,
                        [self.t, self.t + self.dt * dt_factor],
                        self.state,
                        args=(self.evader_orbit,),
                        method='DOP853',  # 더 안정적인 적분기
                        rtol=1e-6,
                        atol=1e-6,
                        dense_output=True
                    )
                    
                    if sol.success and not np.isnan(sol.y[:, -1]).any():
                        self.state = sol.y[:, -1]
                        break
                    else:
                        dt_factor *= 0.5  # 시간 간격 감소
                        if self.debug_mode:
                            print(f"적분 실패, 시간 간격 감소: {dt_factor}")
                except Exception as e:
                    dt_factor *= 0.5
                    if self.debug_mode:
                        print(f"적분 오류: {e}")
            else:
                # 모든 시도 실패시 선형 근사
                if self.debug_mode:
                    print("WARNING: 적분 실패, 선형 근사 사용")
                self.state[0:3] += self.state[3:6] * self.dt

    except Exception as e:
        if self.debug_mode:
            print(f"시뮬레이션 오류 발생: {e}")
        # 안전한 선형 업데이트
        self.state[0:3] += self.state[3:6] * self.dt
    
    # 최종 NaN 체크
    if np.isnan(self.state).any():
        self.state = np.nan_to_num(self.state, nan=0.0, posinf=1e6, neginf=-1e6)


# 개선된 RK4 메서드
def _rk4_step(self):
    """RK4 방법으로 상태 업데이트 (개선된 버전)"""
    dt = self.dt
    s = self.state.copy()  # 복사본 사용
    t = self.t
    
    try:
        # 적응적 시간 간격
        max_steps = 4
        dt_sub = dt / max_steps
        
        for _ in range(max_steps):
            k1 = np.array(relative_dynamics_evader_centered(t, s, self.evader_orbit))
            
            # NaN 체크
            if np.isnan(k1).any():
                break
                
            k2 = np.array(relative_dynamics_evader_centered(
                t + 0.5 * dt_sub, s + 0.5 * dt_sub * k1, self.evader_orbit
            ))
            
            if np.isnan(k2).any():
                break
                
            k3 = np.array(relative_dynamics_evader_centered(
                t + 0.5 * dt_sub, s + 0.5 * dt_sub * k2, self.evader_orbit
            ))
            
            if np.isnan(k3).any():
                break
                
            k4 = np.array(relative_dynamics_evader_centered(
                t + dt_sub, s + dt_sub * k3, self.evader_orbit
            ))
            
            if np.isnan(k4).any():
                break
            
            # RK4 업데이트
            s = s + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt_sub
        
        # 성공적으로 완료된 경우만 상태 업데이트
        if not np.isnan(s).any():
            self.state = s
        else:
            # 실패시 선형 업데이트
            self.state[0:3] += self.state[3:6] * self.dt
            
    except Exception as e:
        if self.debug_mode:
            print(f"RK4 오류: {e}")
        # 선형 업데이트 폴백
        self.state[0:3] += self.state[3:6] * self.dt


# 개선된 지능형 추격자 전략
def compute_interception_strategy(self, state: np.ndarray) -> np.ndarray:
    """지능형 추격자의 추격 전략 계산 (개선된 버전)"""
    rho = state[:3].copy()  # 상대 위치
    v_rel = state[3:6].copy()  # 상대 속도
    rho_mag = np.linalg.norm(rho)
    
    # 안전성 체크
    if rho_mag < 1e-10:
        return np.zeros(3)
    
    # 적응형 예측 시간 (거리 기반)
    prediction_time = np.clip(rho_mag / 1000.0, 1.0, 10.0) * self.dt
    
    # 캐시된 방향 사용 (성능 향상)
    if not hasattr(self, '_cached_directions'):
        self._cached_directions = []
        self._cache_counter = 0
    
    # 주기적으로 캐시 갱신
    if self._cache_counter % 10 == 0:
        self._cached_directions = self.get_optimal_interception_directions(
            rho, v_rel, prediction_time
        )
    self._cache_counter += 1
    
    # 최적 방향 선택
    best_action = None
    min_future_distance = float("inf")
    
    for direction in self._cached_directions:
        for scale in [0.6, 0.8, 1.0]:  # 다양한 추력 크기
            test_action = scale * self.delta_v_pmax * direction
            
            # 미래 위치 예측 (간단한 선형 모델)
            future_vel = v_rel + test_action
            future_pos = rho + future_vel * prediction_time
            future_distance = np.linalg.norm(future_pos)
            
            # 거리와 연료 효율성 고려
            cost = future_distance + 0.1 * scale * self.delta_v_pmax
            
            if cost < min_future_distance:
                min_future_distance = cost
                best_action = test_action
    
    if best_action is None:
        # 기본 전략: 직접 추격
        best_action = -self.delta_v_pmax * rho / rho_mag
    
    # 지능적 노이즈 추가 (거리 기반)
    noise_scale = np.clip(0.05 + 0.1 * (rho_mag / self.capture_distance - 1), 0.05, 0.3)
    noise = np.random.normal(0, noise_scale * self.delta_v_pmax, 3)
    
    action_with_noise = best_action + noise
    action_with_noise = np.clip(action_with_noise, -self.delta_v_pmax, self.delta_v_pmax)
    
    return action_with_noise


# 메모리 효율적인 히스토리 관리
class CircularBuffer:
    """순환 버퍼 구현"""
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
