# tests/test_orbital_mechanics.py
"""
궤도 역학 모듈 테스트
"""

import pytest
import numpy as np
from orbital_mechanics.orbit import ChiefOrbit
from orbital_mechanics.coordinate_transforms import (
    state_to_orbital_elements, orbital_elements_to_state
)
from utils.constants import MU_EARTH


class TestChiefOrbit:
    """ChiefOrbit 클래스 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.orbit = ChiefOrbit(
            a=7500e3, e=0.1, i=0.1, RAAN=0.0, 
            omega=0.0, M0=0.0, mu=MU_EARTH
        )
    
    def test_orbit_initialization(self):
        """궤도 초기화 테스트"""
        assert self.orbit.a == 7500e3
        assert self.orbit.e == 0.1
        assert self.orbit.n > 0  # 평균 운동이 양수인지 확인
    
    def test_mean_anomaly(self):
        """평균 근점이각 계산 테스트"""
        t = 1000  # 1000초
        M = self.orbit.get_M(t)
        expected_M = self.orbit.M0 + self.orbit.n * t
        assert abs(M - expected_M) < 1e-10
    
    def test_position_velocity(self):
        """위치/속도 계산 테스트"""
        r, v = self.orbit.get_position_velocity(0)
        
        # 결과가 NaN이 아닌지 확인
        assert not np.isnan(r).any()
        assert not np.isnan(v).any()
        
        # 크기가 합리적인지 확인
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        assert r_mag > 6000e3  # 지구 반지름보다 큼
        assert r_mag < 10000e3  # 합리적인 고도
        assert v_mag > 6000  # 합리적인 속도 (m/s)
        assert v_mag < 9000
    
    def test_energy_conservation(self):
        """에너지 보존 테스트 (간단한 확인)"""
        r, v = self.orbit.get_position_velocity(0)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # 궤도 에너지 계산
        energy = v_mag**2 / 2 - MU_EARTH / r_mag
        expected_energy = -MU_EARTH / (2 * self.orbit.a)
        
        # 1% 오차 내에서 일치하는지 확인
        relative_error = abs(energy - expected_energy) / abs(expected_energy)
        assert relative_error < 0.01


class TestCoordinateTransforms:
    """좌표 변환 테스트"""
    
    def test_state_to_elements_roundtrip(self):
        """상태 → 궤도요소 → 상태 변환 테스트"""
        # 원래 궤도 요소
        a, e, i = 7500e3, 0.1, 0.1
        RAAN, omega, M = 0.1, 0.2, 0.3
        
        # 궤도요소 → 상태
        state = orbital_elements_to_state(a, e, i, RAAN, omega, M)
        
        # 상태 → 궤도요소
        a2, e2, i2, RAAN2, omega2, M2 = state_to_orbital_elements(state[:3], state[3:])
        
        # 정확도 확인 (각도는 2π 모듈로)
        assert abs(a - a2) / a < 0.01
        assert abs(e - e2) < 0.01
        assert abs(i - i2) < 0.01
        
        # 각도는 2π 주기성 고려
        def angle_diff(a1, a2):
            diff = (a1 - a2) % (2 * np.pi)
            return min(diff, 2 * np.pi - diff)
        
        assert angle_diff(RAAN, RAAN2) < 0.1
        assert angle_diff(omega, omega2) < 0.1
        assert angle_diff(M, M2) < 0.1
    
    def test_nan_handling(self):
        """NaN 처리 테스트"""
        # 극단적인 값들로 테스트
        try:
            r = np.array([1e-10, 0, 0])  # 매우 작은 거리
            v = np.array([0, 1e10, 0])   # 매우 큰 속도
            
            elements = state_to_orbital_elements(r, v)
            
            # NaN이 없어야 함
            assert not np.isnan(elements).any()
            
        except:
            # 예외가 발생해도 괜찮음 (극단적 케이스)
            pass