"""
좌표 변환 및 궤도 요소 변환 함수들
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, Dict
from utils.constants import MU_EARTH


def state_to_orbital_elements(r_vec: np.ndarray, v_vec: np.ndarray, 
                            mu: float = MU_EARTH) -> Tuple[float, float, float, float, float, float]:
    """
    위치 벡터와 속도 벡터로부터 궤도 요소 계산
    
    Args:
        r_vec: 위치 벡터 (x, y, z) [m]
        v_vec: 속도 벡터 (vx, vy, vz) [m/s]
        mu: 중력 상수 [m^3/s^2]
        
    Returns:
        (a, e, i, RAAN, omega, M): 궤도 요소
    """
    # 궤도 각운동량 벡터
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # 위치 크기
    r = np.linalg.norm(r_vec)
    
    # 궤도 에너지에서 반장축 계산
    v2 = np.sum(v_vec**2)
    a = mu / (2.0 * mu/r - v2)
    
    # 이심률 벡터 및 이심률
    e_vec = np.cross(v_vec, h_vec)/mu - r_vec/r
    e = np.linalg.norm(e_vec)
    
    # 궤도 경사각 (궤도면과 적도면 사이 각)
    i = np.arccos(np.clip(h_vec[2]/h, -1.0, 1.0))
    
    # 승교점 벡터
    n_vec = np.cross(np.array([0, 0, 1]), h_vec)
    n = np.linalg.norm(n_vec)
    
    # 승교점 적경 (RAAN)
    if n < 1e-10:  # 적도 궤도의 경우
        RAAN = 0.0
    else:
        cos_RAAN = n_vec[0]/n
        if n_vec[1] >= 0:
            RAAN = np.arccos(np.clip(cos_RAAN, -1.0, 1.0))
        else:
            RAAN = 2*np.pi - np.arccos(np.clip(cos_RAAN, -1.0, 1.0))
    
    # 근지점 인수 (omega)
    if n < 1e-10:  # 적도 궤도
        if e < 1e-10:  # 원 궤도
            omega = 0.0
        else:
            cos_omega = e_vec[0]/e
            if e_vec[1] >= 0:
                omega = np.arccos(np.clip(cos_omega, -1.0, 1.0))
            else:
                omega = 2*np.pi - np.arccos(np.clip(cos_omega, -1.0, 1.0))
    else:
        if e < 1e-10:  # 원 궤도
            omega = 0.0
        else:
            cos_omega = np.dot(n_vec, e_vec)/(n*e)
            cos_omega = np.clip(cos_omega, -1.0, 1.0)
            if e_vec[2] >= 0:
                omega = np.arccos(cos_omega)
            else:
                omega = 2*np.pi - np.arccos(cos_omega)
    
    # 진근점이각 (f)
    if e < 1e-10:  # 원 궤도
        if i < 1e-10:  # 적도 원궤도
            f = np.arctan2(r_vec[1], r_vec[0])
        else:
            if n > 1e-10:
                cos_f = np.dot(n_vec, r_vec)/(n*r)
                cos_f = np.clip(cos_f, -1.0, 1.0)
                if r_vec[2] >= 0:
                    f = np.arccos(cos_f)
                else:
                    f = 2*np.pi - np.arccos(cos_f)
            else:
                f = 0.0
    else:
        cos_f = np.dot(e_vec, r_vec)/(e*r)
        cos_f = np.clip(cos_f, -1.0, 1.0)
        if np.dot(r_vec, v_vec) >= 0:
            f = np.arccos(cos_f)
        else:
            f = 2*np.pi - np.arccos(cos_f)
    
    # 이심근점이각 (E)
    if e > 1e-10:
        E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(f/2))
        if E < 0:
            E += 2*np.pi
    else:
        E = f
    
    # 평균근점이각 (M)
    M = E - e * np.sin(E)
    if M < 0:
        M += 2*np.pi
    
    return a, e, i, RAAN, omega, M


def orbital_elements_to_state(a: float, e: float, i: float, RAAN: float, 
                            omega: float, M: float, mu: float = MU_EARTH) -> np.ndarray:
    """
    궤도 요소에서 위치/속도 벡터로 변환
    
    Args:
        a, e, i, RAAN, omega, M: 궤도 요소
        mu: 중력 상수
        
    Returns:
        상태 벡터 [x, y, z, vx, vy, vz]
    """
    # 케플러 방정식 해결
    def kepler_eq(E):
        return E - e * np.sin(E) - M
    
    try:
        E = fsolve(kepler_eq, M)[0]
    except:
        # 단순한 반복법 사용
        E = M
        for _ in range(10):
            E = M + e * np.sin(E)
    
    # 진근점이각 계산
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), 
                       np.sqrt(1 - e) * np.cos(E / 2))
    
    # 궤도면 내에서의 위치 계산
    r_mag = a * (1 - e**2) / (1 + e * np.cos(f))
    r_orbit = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0])
    
    # 궤도면 내에서의 속도 계산
    p = a * (1 - e**2)
    v_orbit_mag = np.sqrt(mu / p)
    v_orbit = np.array([-v_orbit_mag * np.sin(f), 
                       v_orbit_mag * (e + np.cos(f)), 0])
    
    # 회전 행렬 계산 (페리포컬 -> 적도 좌표계)
    R = compute_rotation_matrix(RAAN, i, omega)
    
    # 관성 좌표계에서의 위치와 속도
    r_eci = R @ r_orbit
    v_eci = R @ v_orbit
    
    return np.concatenate((r_eci, v_eci))


def compute_rotation_matrix(RAAN: float, i: float, omega: float) -> np.ndarray:
    """
    페리포컬 좌표계에서 ECI 좌표계로의 회전 행렬 계산
    
    Args:
        RAAN: 승교점 적경 [rad]
        i: 경사각 [rad]
        omega: 근점 편각 [rad]
        
    Returns:
        3x3 회전 행렬
    """
    cos_RAAN, sin_RAAN = np.cos(RAAN), np.sin(RAAN)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    
    # 회전 행렬 계산
    R11 = cos_RAAN * cos_omega - sin_RAAN * sin_omega * cos_i
    R12 = -cos_RAAN * sin_omega - sin_RAAN * cos_omega * cos_i
    R13 = sin_RAAN * sin_i
    
    R21 = sin_RAAN * cos_omega + cos_RAAN * sin_omega * cos_i
    R22 = -sin_RAAN * sin_omega + cos_RAAN * cos_omega * cos_i
    R23 = -cos_RAAN * sin_i
    
    R31 = sin_omega * sin_i
    R32 = cos_omega * sin_i
    R33 = cos_i
    
    return np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])


def eci_to_lvlh(r_chief: np.ndarray, v_chief: np.ndarray, 
                r_deputy: np.ndarray, v_deputy: np.ndarray) -> np.ndarray:
    """
    ECI 좌표계에서 LVLH 좌표계로 변환
    
    Args:
        r_chief, v_chief: 기준 위성(chief)의 위치와 속도
        r_deputy, v_deputy: 부 위성(deputy)의 위치와 속도
        
    Returns:
        LVLH 좌표계에서의 상대 상태 [x, y, z, vx, vy, vz]
    """
    # LVLH 기준 프레임 계산
    h_chief = np.cross(r_chief, v_chief)
    h_chief_norm = np.linalg.norm(h_chief)
    r_chief_norm = np.linalg.norm(r_chief)
    
    # 수치 안정성 검사
    if h_chief_norm < 1e-10 or r_chief_norm < 1e-10:
        h_chief_norm = max(h_chief_norm, 1e-10)
        r_chief_norm = max(r_chief_norm, 1e-10)
    
    # LVLH 좌표계 정의
    x_lvlh = r_chief / r_chief_norm  # Radial 방향 (+x)
    z_lvlh = h_chief / h_chief_norm  # 각운동량 벡터 방향 (+z)
    y_lvlh = np.cross(z_lvlh, x_lvlh)  # Along-track 방향 (+y)
    y_lvlh = y_lvlh / np.linalg.norm(y_lvlh)
    
    # 회전 행렬 생성 (ECI -> LVLH)
    R_eci_to_lvlh = np.array([x_lvlh, y_lvlh, z_lvlh])
    
    # 상대 위치 계산
    r_rel = R_eci_to_lvlh @ (r_deputy - r_chief)
    
    # LVLH 프레임의 회전 속도 계산
    omega_lvlh = h_chief / r_chief_norm**2
    
    # 회전 효과를 고려한 상대 속도 계산
    v_rel = R_eci_to_lvlh @ (v_deputy - v_chief) - np.cross(omega_lvlh, r_rel)
    
    return np.concatenate((r_rel, v_rel))


def lvlh_to_eci(r_chief: np.ndarray, v_chief: np.ndarray, 
                state_lvlh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LVLH 좌표계에서 ECI 좌표계로 변환
    
    Args:
        r_chief, v_chief: 기준 위성의 ECI 위치와 속도
        state_lvlh: LVLH 상대 상태 [x, y, z, vx, vy, vz]
        
    Returns:
        ECI 좌표계에서의 부 위성 위치와 속도
    """
    # LVLH 기준 프레임 계산
    h_chief = np.cross(r_chief, v_chief)
    h_chief_norm = np.linalg.norm(h_chief)
    r_chief_norm = np.linalg.norm(r_chief)
    
    # 수치 안정성 검사
    if h_chief_norm < 1e-10 or r_chief_norm < 1e-10:
        h_chief_norm = max(h_chief_norm, 1e-10)
        r_chief_norm = max(r_chief_norm, 1e-10)
    
    # LVLH 좌표계 정의
    x_lvlh = r_chief / r_chief_norm
    z_lvlh = h_chief / h_chief_norm
    y_lvlh = np.cross(z_lvlh, x_lvlh)
    y_lvlh = y_lvlh / np.linalg.norm(y_lvlh)
    
    # 회전 행렬 (LVLH -> ECI)
    R_lvlh_to_eci = np.vstack((x_lvlh, y_lvlh, z_lvlh)).T
    
    # 상대 위치와 속도 추출
    r_rel_lvlh = state_lvlh[:3]
    v_rel_lvlh = state_lvlh[3:6]
    
    # ECI 위치 계산
    r_deputy = r_chief + R_lvlh_to_eci @ r_rel_lvlh
    
    # LVLH 프레임 회전 속도
    omega_lvlh = h_chief / r_chief_norm**2
    
    # ECI 속도 계산
    v_deputy = (v_chief + R_lvlh_to_eci @ v_rel_lvlh + 
               np.cross(omega_lvlh, R_lvlh_to_eci @ r_rel_lvlh))

    return r_deputy, v_deputy


def convert_orbital_elements_to_relative_state(
    a1: float, e1: float, i1: float, RAAN1: float, omega1: float, M1: float,
    a2: float, e2: float, i2: float, RAAN2: float, omega2: float, M2: float,
    mu: float = MU_EARTH) -> np.ndarray:
    """
    두 위성의 궤도 요소를 이용하여 상대 상태(LVLH 프레임)를 계산
    
    Args:
        a1, e1, i1, RAAN1, omega1, M1: 기준 위성의 궤도 요소
        a2, e2, i2, RAAN2, omega2, M2: 부 위성의 궤도 요소
        mu: 중력 상수
        
    Returns:
        LVLH 프레임에서의 상대 상태 [x, y, z, vx, vy, vz]
    """
    # 두 위성의 절대 상태 계산
    state1 = orbital_elements_to_state(a1, e1, i1, RAAN1, omega1, M1, mu)
    state2 = orbital_elements_to_state(a2, e2, i2, RAAN2, omega2, M2, mu)
    
    # 관성 좌표계에서의 위치 및 속도
    r1, v1 = state1[:3], state1[3:]
    r2, v2 = state2[:3], state2[3:]
    
    # ECI -> LVLH 변환
    return eci_to_lvlh(r1, v1, r2, v2)


def roe_to_cartesian(roe: np.ndarray, chief_orbit) -> np.ndarray:
    """
    상대 궤도 요소(ROE)를 직교 좌표로 변환
    
    Args:
        roe: 상대 궤도 요소 [da, de_x, de_y, di_x, di_y, dM]
        chief_orbit: 기준 궤도 객체
        
    Returns:
        직교 좌표계에서의 상대 상태
    """
    # ROE 요소 추출
    da, de_x, de_y, di_x, di_y, dM = roe
    
    # 기준 궤도 요소
    a_c = chief_orbit.a
    e_c = chief_orbit.e
    i_c = chief_orbit.i
    
    # 근사적 변환 (ROE -> 직교좌표)
    # 이는 간단한 선형 변환으로, 더 정확한 변환을 위해서는
    # 완전한 ROE 변환 공식을 사용해야 함
    
    x = da
    y = a_c * dM
    z = a_c * di_x
    
    vx = 0  # 속도는 더 복잡한 계산 필요
    vy = 0
    vz = 0
    
    return np.array([x, y, z, vx, vy, vz])


def cartesian_to_roe(state: np.ndarray, chief_orbit) -> np.ndarray:
    """
    직교 좌표를 상대 궤도 요소(ROE)로 변환
    
    Args:
        state: 직교 좌표계에서의 상대 상태
        chief_orbit: 기준 궤도 객체
        
    Returns:
        상대 궤도 요소 [da, de_x, de_y, di_x, di_y, dM]
    """
    x, y, z, vx, vy, vz = state
    
    # 기준 궤도 요소
    a_c = chief_orbit.a
    
    # 근사적 변환 (직교좌표 -> ROE)
    da = x
    dM = y / a_c
    di_x = z / a_c
    
    # 다른 요소들은 속도 정보가 필요
    de_x = 0  # 간단화
    de_y = 0
    di_y = 0
    
    return np.array([da, de_x, de_y, di_x, di_y, dM])
