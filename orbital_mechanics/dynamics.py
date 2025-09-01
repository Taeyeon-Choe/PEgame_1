"""
궤도 동역학 함수들 (Numba JIT 및 NaN 처리 개선 버전)
"""

import numpy as np
from typing import List, Tuple
from numba import jit
from utils.constants import MU_EARTH, R_EARTH, J2_EARTH


@jit(nopython=True, cache=True)
def _relative_dynamics_core(state: np.ndarray, r0: float, dot_theta0: float, ddot_theta0: float) -> np.ndarray:
    """Core computation for relative dynamics without J2 perturbation."""
    x, y, z, vx, vy, vz = state

    # NaN 체크
    if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isnan(vx) or np.isnan(vy) or np.isnan(vz):
        return np.zeros(6)
    
    # 안전한 거리 계산
    rho_squared = (r0 + x) ** 2 + y**2 + z**2
    rho_norm = np.sqrt(max(rho_squared, 1e-10))
    
    # 중력 가속도 계산
    grav_factor = MU_EARTH / (rho_norm ** 3)
    
    # 가속도 계산
    ddot_x = (
        2 * dot_theta0 * vy
        + ddot_theta0 * y
        + dot_theta0**2 * x
        - grav_factor * (r0 + x)
        + MU_EARTH / r0**2
    )
    ddot_y = (
        -2 * dot_theta0 * vx
        - ddot_theta0 * x
        + dot_theta0**2 * y
        - grav_factor * y
    )
    ddot_z = -grav_factor * z

    return np.array([vx, vy, vz, ddot_x, ddot_y, ddot_z])


@jit(nopython=True, cache=True)
def _j2_diff_accel_core(relative_pos: np.ndarray, r_evader: np.ndarray, v_evader: np.ndarray) -> np.ndarray:
    """Core computation of differential J2 acceleration in the LVLH frame."""
    x, y, z = relative_pos

    # 안전한 노름 계산
    r_evader_squared = np.dot(r_evader, r_evader)
    r_evader_norm = np.sqrt(max(r_evader_squared, 1e-10))

    h_evader = np.cross(r_evader, v_evader)
    h_evader_squared = np.dot(h_evader, h_evader)
    h_evader_norm = np.sqrt(max(h_evader_squared, 1e-10))

    # LVLH 좌표계 정의
    x_lvlh = r_evader / r_evader_norm
    z_lvlh = h_evader / h_evader_norm
    y_lvlh = np.cross(z_lvlh, x_lvlh)
    y_lvlh_norm = np.sqrt(np.dot(y_lvlh, y_lvlh))
    if y_lvlh_norm > 1e-10:
        y_lvlh = y_lvlh / y_lvlh_norm
    else:
        y_lvlh = np.array([0., 1., 0.])

    R_lvlh_to_eci = np.vstack((x_lvlh, y_lvlh, z_lvlh)).T

    # 추격자 위치 계산
    rel_vec = np.array([x, y, z])
    r_pursuer_eci = r_evader + R_lvlh_to_eci @ rel_vec
    r_pursuer_squared = np.dot(r_pursuer_eci, r_pursuer_eci)
    r_pursuer_norm = np.sqrt(max(r_pursuer_squared, 1e-10))

    # J2 가속도 계산
    j2_factor_evader = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / (r_evader_norm**5)
    z_ratio_evader = r_evader[2] / r_evader_norm
    term_evader = 5 * z_ratio_evader ** 2 - 1
    
    j2_accel_evader = j2_factor_evader * np.array(
        [
            r_evader[0] * term_evader,
            r_evader[1] * term_evader,
            r_evader[2] * (5 * z_ratio_evader ** 2 - 3),
        ]
    )

    j2_factor_pursuer = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / (r_pursuer_norm**5)
    z_ratio_pursuer = r_pursuer_eci[2] / r_pursuer_norm
    term_pursuer = 5 * z_ratio_pursuer ** 2 - 1
    
    j2_accel_pursuer = j2_factor_pursuer * np.array(
        [
            r_pursuer_eci[0] * term_pursuer,
            r_pursuer_eci[1] * term_pursuer,
            r_pursuer_eci[2] * (5 * z_ratio_pursuer ** 2 - 3),
        ]
    )

    # 차등 가속도
    j2_diff_accel_eci = j2_accel_pursuer - j2_accel_evader
    j2_diff_accel_lvlh = R_lvlh_to_eci.T @ j2_diff_accel_eci

    return j2_diff_accel_lvlh


def relative_dynamics_evader_centered(
    t: float, state: List[float], evader_orbit
) -> List[float]:
    """
    Evader 중심 상대 동역학 (pursuer의 상대 운동)
    개선: NaN 처리 및 수치 안정성 향상

    Args:
        t: 시간
        state: 상태 벡터 [x, y, z, vx, vy, vz]
        evader_orbit: 회피자 궤도 객체

    Returns:
        상태 도함수 [vx, vy, vz, ax, ay, az]
    """
    # 상태 벡터를 numpy 배열로 변환 및 NaN 체크
    state_array = np.array(state, dtype=np.float64)
    
    # NaN 체크 및 처리
    if np.any(np.isnan(state_array)):
        print(f"WARNING: NaN 값 감지됨! state = {state}")
        # NaN 값을 0으로 대체
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 극단적으로 큰 값 제한
    max_position = 1e6  # 1000 km
    max_velocity = 1e4  # 10 km/s
    
    state_array[:3] = np.clip(state_array[:3], -max_position, max_position)
    state_array[3:6] = np.clip(state_array[3:6], -max_velocity, max_velocity)
    
    x, y, z, vx, vy, vz = state_array

    try:
        # ChiefOrbit 객체에서 필요한 값들 추출
        evader_state = evader_orbit.get_state(t)
        r0 = float(evader_state["r0"])
        dot_theta0 = float(evader_state["dot_theta0"])
        ddot_theta0 = float(evader_state["ddot_theta0"])
        
        # NaN 체크
        if np.isnan(r0) or np.isnan(dot_theta0) or np.isnan(ddot_theta0):
            print(f"WARNING: 궤도 상태에 NaN 값: r0={r0}, dot_theta0={dot_theta0}, ddot_theta0={ddot_theta0}")
            return [0, 0, 0, 0, 0, 0]

        # JIT 컴파일된 핵심 계산 함수 호출
        core_result = _relative_dynamics_core(
            state_array,
            r0,
            dot_theta0,
            ddot_theta0,
        )

        # J2 섭동 효과 추가
        try:
            j2_accel = compute_j2_differential_acceleration(t, state_array.tolist(), evader_orbit)
            core_result[3:] += j2_accel
        except Exception as e:
            print(f"WARNING: J2 계산 오류, 무시함: {e}")

        # 결과 NaN 체크
        if np.any(np.isnan(core_result)):
            print(f"WARNING: 동역학 계산 결과에 NaN")
            return [0, 0, 0, 0, 0, 0]
        
        return core_result.tolist()
        
    except Exception as e:
        print(f"ERROR in relative_dynamics: {e}")
        return [0, 0, 0, 0, 0, 0]


def compute_j2_differential_acceleration(
    t: float, state: List[float], evader_orbit
) -> np.ndarray:
    """
    J2 섭동의 차등 가속도 계산
    개선: 예외 처리 및 안정성 향상

    Args:
        t: 시간
        state: 상대 상태 벡터
        evader_orbit: 회피자 궤도

    Returns:
        J2 차등 가속도 벡터 (LVLH 좌표계)
    """
    try:
        # 궤도 객체에서 필요한 위치/속도 벡터 추출
        r_evader, v_evader = evader_orbit.get_position_velocity(t)
        
        # NaN 체크
        if np.any(np.isnan(r_evader)) or np.any(np.isnan(v_evader)):
            print("WARNING: J2 계산을 위한 위치/속도에 NaN")
            return np.zeros(3)
        
        relative_pos = np.array(state[:3], dtype=np.float64)
        
        # JIT 컴파일된 함수 호출
        j2_accel = _j2_diff_accel_core(relative_pos, r_evader.astype(np.float64), v_evader.astype(np.float64))
        
        # 결과 검증
        if np.any(np.isnan(j2_accel)):
            return np.zeros(3)
        
        max_accel = 1e-6  # 0.001 mm/s^2
        j2_accel = np.clip(j2_accel, -max_accel, max_accel)
        
        return j2_accel
        
    except Exception as e:
        print(f"WARNING: J2 가속도 계산 오류: {e}")
        return np.zeros(3)




def atmospheric_drag_acceleration(
    r_eci: np.ndarray, v_eci: np.ndarray, cd_area_mass: float = 1e-3
) -> np.ndarray:
    """
    대기 항력 가속도 계산 (간단한 모델)

    Args:
        r_eci: ECI 위치 벡터
        v_eci: ECI 속도 벡터
        cd_area_mass: CD * A / m (drag coefficient * area / mass)

    Returns:
        항력 가속도 벡터
    """
    r_norm = np.linalg.norm(r_eci)
    altitude = r_norm - R_EARTH

    # 간단한 지수 대기 모델
    if altitude > 1000e3:  # 1000km 이상에서는 무시
        return np.zeros(3)

    # 대기 밀도 (매우 간단한 모델)
    h0 = 7000  # 스케일 높이 (m)
    rho0 = 1.225  # 해수면 밀도 (kg/m^3)
    rho = rho0 * np.exp(-altitude / h0)

    # 속도 크기
    v_norm = np.linalg.norm(v_eci)

    if v_norm < 1e-10:
        return np.zeros(3)

    # 항력 가속도
    drag_accel = -0.5 * rho * cd_area_mass * v_norm * v_eci

    return drag_accel


def solar_radiation_pressure(
    r_eci: np.ndarray, r_sun_eci: np.ndarray, cr_area_mass: float = 1e-3
) -> np.ndarray:
    """
    태양 복사압 가속도 계산

    Args:
        r_eci: 위성 ECI 위치
        r_sun_eci: 태양 ECI 위치
        cr_area_mass: CR * A / m (radiation coefficient * area / mass)

    Returns:
        복사압 가속도 벡터
    """
    # 태양-위성 벡터
    r_sat_sun = r_eci - r_sun_eci
    r_sat_sun_norm = np.linalg.norm(r_sat_sun)

    if r_sat_sun_norm < 1e-10:
        return np.zeros(3)

    # 태양 복사압 상수 (1 AU에서의 압력)
    P_solar = 4.56e-6  # N/m^2
    AU = 149.6e9  # m

    # 거리 제곱 법칙
    pressure = P_solar * (AU / r_sat_sun_norm) ** 2

    # 복사압 가속도 (태양 방향)
    srp_accel = pressure * cr_area_mass * (r_sat_sun / r_sat_sun_norm)

    return srp_accel


def third_body_acceleration(
    r_eci: np.ndarray, r_body_eci: np.ndarray, mu_body: float
) -> np.ndarray:
    """
    제3천체 중력 가속도 계산

    Args:
        r_eci: 위성 ECI 위치
        r_body_eci: 제3천체 ECI 위치
        mu_body: 제3천체 중력 상수

    Returns:
        제3천체 중력 가속도 벡터
    """
    # 제3천체-위성 벡터
    r_sat_body = r_eci - r_body_eci
    r_sat_body_norm = np.linalg.norm(r_sat_body)
    r_body_norm = np.linalg.norm(r_body_eci)

    if r_sat_body_norm < 1e-10 or r_body_norm < 1e-10:
        return np.zeros(3)

    # 제3천체 중력 가속도
    accel_direct = -mu_body * r_sat_body / r_sat_body_norm**3
    accel_indirect = -mu_body * r_body_eci / r_body_norm**3

    return accel_direct + accel_indirect


# 추가: 안전한 동역학 함수 (fallback)
def safe_relative_dynamics(t: float, state: List[float], evader_orbit) -> List[float]:
    """
    안전한 상대 동역학 함수 (JIT 없이)
    Numba 오류 발생시 fallback으로 사용
    """
    try:
        return relative_dynamics_evader_centered(t, state, evader_orbit)
    except Exception as e:
        print(f"동역학 계산 오류: {e}")
        # 간단한 선형 동역학으로 fallback
        x, y, z, vx, vy, vz = state
        return [vx, vy, vz, 0.0, 0.0, 0.0]
