"""
궤도 동역학 함수들
"""

import numpy as np
from typing import List, Tuple
from numba import jit
from utils.constants import MU_EARTH, R_EARTH, J2_EARTH


@jit(nopython=True, cache=True)
def _relative_dynamics_core(state: np.ndarray, r0: float, dot_theta0: float, ddot_theta0: float) -> np.ndarray:
    """Core computation for relative dynamics without J2 perturbation."""
    x, y, z, vx, vy, vz = state

    rho_norm = np.sqrt((r0 + x) ** 2 + y**2 + z**2)
    if rho_norm < 1e-10:
        rho_norm = 1e-10

    ddot_x = (
        2 * dot_theta0 * vy
        + ddot_theta0 * y
        + dot_theta0**2 * x
        - MU_EARTH * (r0 + x) / rho_norm**3
        + MU_EARTH / r0**2
    )
    ddot_y = (
        -2 * dot_theta0 * vx
        - ddot_theta0 * x
        + dot_theta0**2 * y
        - MU_EARTH * y / rho_norm**3
    )
    ddot_z = -MU_EARTH * z / rho_norm**3

    return np.array([vx, vy, vz, ddot_x, ddot_y, ddot_z])


@jit(nopython=True, cache=True)
def _j2_diff_accel_core(relative_pos: np.ndarray, r_evader: np.ndarray, v_evader: np.ndarray) -> np.ndarray:
    """Core computation of differential J2 acceleration in the LVLH frame."""
    x, y, z = relative_pos

    r_evader_norm = np.sqrt(np.dot(r_evader, r_evader))
    if r_evader_norm < 1e-10:
        r_evader_norm = 1e-10

    h_evader = np.cross(r_evader, v_evader)
    h_evader_norm = np.sqrt(np.dot(h_evader, h_evader))
    if h_evader_norm < 1e-10:
        h_evader_norm = 1e-10

    x_lvlh = r_evader / r_evader_norm
    z_lvlh = h_evader / h_evader_norm
    y_lvlh = np.cross(z_lvlh, x_lvlh)
    y_lvlh_norm = np.sqrt(np.dot(y_lvlh, y_lvlh))
    y_lvlh = y_lvlh / y_lvlh_norm

    R_lvlh_to_eci = np.vstack((x_lvlh, y_lvlh, z_lvlh)).T

    rel_vec = np.array([x, y, z])
    r_pursuer_eci = r_evader + R_lvlh_to_eci @ rel_vec
    r_pursuer_norm = np.sqrt(np.dot(r_pursuer_eci, r_pursuer_eci))
    if r_pursuer_norm < 1e-10:
        r_pursuer_norm = 1e-10

    j2_factor_evader = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / r_evader_norm**4
    term_evader = 5 * (r_evader[2] / r_evader_norm) ** 2 - 1
    j2_accel_evader = j2_factor_evader * np.array(
        [
            r_evader[0] * term_evader,
            r_evader[1] * term_evader,
            r_evader[2] * (5 * (r_evader[2] / r_evader_norm) ** 2 - 3),
        ]
    )

    j2_factor_pursuer = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / r_pursuer_norm**4
    term_pursuer = 5 * (r_pursuer_eci[2] / r_pursuer_norm) ** 2 - 1
    j2_accel_pursuer = j2_factor_pursuer * np.array(
        [
            r_pursuer_eci[0] * term_pursuer,
            r_pursuer_eci[1] * term_pursuer,
            r_pursuer_eci[2] * (5 * (r_pursuer_eci[2] / r_pursuer_norm) ** 2 - 3),
        ]
    )

    j2_diff_accel_eci = j2_accel_pursuer - j2_accel_evader
    j2_diff_accel_lvlh = R_lvlh_to_eci.T @ j2_diff_accel_eci

    return j2_diff_accel_lvlh


def relative_dynamics_evader_centered(
    t: float, state: List[float], evader_orbit
) -> List[float]:
    """
    Evader 중심 상대 동역학 (pursuer의 상대 운동)

    Args:
        t: 시간
        state: 상태 벡터 [x, y, z, vx, vy, vz]
        evader_orbit: 회피자 궤도 객체

    Returns:
        상태 도함수 [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state

    if (
        np.isnan(x)
        or np.isnan(y)
        or np.isnan(z)
        or np.isnan(vx)
        or np.isnan(vy)
        or np.isnan(vz)
    ):
        print(f"WARNING: NaN 값 감지됨! state = {state}")
        return [0, 0, 0, 0, 0, 0]

    evader_state = evader_orbit.get_state(t)
    r0 = float(evader_state["r0"])
    dot_theta0 = float(evader_state["dot_theta0"])
    ddot_theta0 = float(evader_state["ddot_theta0"])

    core_result = _relative_dynamics_core(
        np.array([x, y, z, vx, vy, vz], dtype=np.float64),
        r0,
        dot_theta0,
        ddot_theta0,
    )

    j2_accel = compute_j2_differential_acceleration(t, state, evader_orbit)
    core_result[3:] += j2_accel

    return core_result.tolist()


def compute_j2_differential_acceleration(
    t: float, state: List[float], evader_orbit
) -> np.ndarray:
    """
    J2 섭동의 차등 가속도 계산

    Args:
        t: 시간
        state: 상대 상태 벡터
        evader_orbit: 회피자 궤도

    Returns:
        J2 차등 가속도 벡터 (LVLH 좌표계)
    """
    r_evader, v_evader = evader_orbit.get_position_velocity(t)
    relative_pos = np.array(state[:3], dtype=np.float64)

    return _j2_diff_accel_core(relative_pos, r_evader.astype(np.float64), v_evader.astype(np.float64))


def hcw_dynamics(t: float, state: List[float], n: float) -> List[float]:
    """
    Hill-Clohessy-Wiltshire (HCW) 선형 상대 동역학

    Args:
        t: 시간
        state: 상태 벡터 [x, y, z, vx, vy, vz]
        n: 평균 운동 (rad/s)

    Returns:
        상태 도함수
    """
    x, y, z, vx, vy, vz = state

    # HCW 방정식
    ddot_x = 3 * n**2 * x + 2 * n * vy
    ddot_y = -2 * n * vx
    ddot_z = -(n**2) * z

    return [vx, vy, vz, ddot_x, ddot_y, ddot_z]


def clohessy_wiltshire_stm(t: float, n: float) -> np.ndarray:
    """
    Clohessy-Wiltshire 상태 전이 행렬 (STM)

    Args:
        t: 시간
        n: 평균 운동

    Returns:
        6x6 상태 전이 행렬
    """
    nt = n * t
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)

    # 위치 블록
    phi_rr = np.array(
        [[4 - 3 * cos_nt, 0, 0], [6 * (sin_nt - nt), 1, 0], [0, 0, cos_nt]]
    )

    # 위치-속도 블록
    phi_rv = np.array(
        [
            [sin_nt / n, 2 * (1 - cos_nt) / n, 0],
            [2 * (cos_nt - 1) / n, (4 * sin_nt - 3 * nt) / n, 0],
            [0, 0, sin_nt / n],
        ]
    )

    # 속도-위치 블록
    phi_vr = np.array(
        [[3 * n * sin_nt, 0, 0], [6 * n * (cos_nt - 1), 0, 0], [0, 0, -n * sin_nt]]
    )

    # 속도-속도 블록
    phi_vv = np.array(
        [[cos_nt, 2 * sin_nt, 0], [-2 * sin_nt, 4 * cos_nt - 3, 0], [0, 0, cos_nt]]
    )

    # 전체 STM 조립
    stm = np.zeros((6, 6))
    stm[:3, :3] = phi_rr
    stm[:3, 3:] = phi_rv
    stm[3:, :3] = phi_vr
    stm[3:, 3:] = phi_vv

    return stm


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
