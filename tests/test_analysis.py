# tests/test_analysis.py
"""
분석 모듈 테스트
"""

import pytest
import numpy as np
from analysis.metrics import (
    calculate_performance_metrics,
    calculate_distance_metrics,
    analyze_trajectory_quality
)


class TestMetrics:
    """메트릭 계산 테스트"""
    
    def setup_method(self):
        """테스트 데이터 설정"""
        # 샘플 궤적 데이터 생성
        n_steps = 100
        self.states = np.random.randn(n_steps, 6) * 1000  # 위치, 속도
        self.actions_e = np.random.randn(n_steps, 3) * 5   # 회피자 액션
        self.actions_p = np.random.randn(n_steps, 3) * 8   # 추격자 액션
        
        self.rewards = {
            'evader': np.random.randn(n_steps).tolist(),
            'pursuer': np.random.randn(n_steps).tolist()
        }
        
        self.info = {
            'safety_score': 0.5,
            'outcome': 'captured'
        }
    
    def test_distance_metrics(self):
        """거리 메트릭 테스트"""
        metrics = calculate_distance_metrics(self.states)
        
        # 기본 메트릭들이 계산되었는지 확인
        assert 'min_distance' in metrics
        assert 'max_distance' in metrics
        assert 'mean_distance' in metrics
        assert 'final_distance' in metrics
        
        # 값들이 합리적인지 확인
        assert metrics['min_distance'] >= 0
        assert metrics['max_distance'] >= metrics['min_distance']
        assert not np.isnan(metrics['mean_distance'])
    
    def test_performance_metrics(self):
        """종합 성능 메트릭 테스트"""
        metrics = calculate_performance_metrics(
            self.states, self.actions_e, self.actions_p, 
            self.rewards, self.info
        )
        
        # 주요 메트릭들이 있는지 확인
        assert 'total_evader_delta_v' in metrics
        assert 'mean_evader_reward' in metrics
        assert 'safety_score' in metrics
        assert 'zero_sum_compliance' in metrics
        
        # NaN이 없는지 확인
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"NaN found in metric: {key}"
    
    def test_trajectory_quality(self):
        """궤적 품질 분석 테스트"""
        metrics = analyze_trajectory_quality(self.states, self.actions_e)
        
        assert 'trajectory_smoothness' in metrics
        assert 'path_efficiency' in metrics
        assert 'trajectory_complexity' in metrics
        
        # 값들이 합리적인 범위에 있는지 확인
        if 'path_efficiency' in metrics:
            assert 0 <= metrics['path_efficiency'] <= 1
