# tests/test_integration.py
"""
통합 테스트
"""

import pytest
import numpy as np
from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv
from training.trainer import create_trainer


class TestIntegration:
    """통합 테스트"""
    
    def test_full_pipeline_short(self):
        """전체 파이프라인 짧은 테스트"""
        # 설정
        config = get_config(debug_mode=True)
        config.training.total_timesteps = 50  # 매우 짧게
        config.environment.max_steps = 10
        
        # 환경 생성
        env = PursuitEvasionEnv(config)
        
        # 트레이너 생성 및 설정
        trainer = create_trainer(env, config)
        trainer.setup_model()
        
        # 짧은 학습
        trainer.train(total_timesteps=50)
        
        # 평가
        results = trainer.evaluate(n_episodes=2)
        
        # 결과 검증
        assert 'mean_reward' in results
        assert 'success_rate' in results
        assert not np.isnan(results['mean_reward'])
        
        # 정리
        env.close()
    
    def test_environment_model_compatibility(self):
        """환경과 모델 호환성 테스트"""
        config = get_config(debug_mode=True)
        env = PursuitEvasionEnv(config)
        
        # 환경 리셋
        obs, _ = env.reset()
        
        # 액션 공간 확인
        action = env.action_space.sample()
        
        # 스텝 실행
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 기본 호환성 확인
        assert obs.shape == env.observation_space.shape
        assert action.shape == env.action_space.shape
        assert next_obs.shape == env.observation_space.shape

        env.close()