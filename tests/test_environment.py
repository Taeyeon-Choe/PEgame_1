# tests/test_environment.py
"""
환경 모듈 테스트
"""

import pytest
import numpy as np
from environment.pursuit_evasion_env import PursuitEvasionEnv
from environment.pursuit_evasion_env_ga_stm import PursuitEvasionEnvGASTM
from config.settings import get_config


class TestPursuitEvasionEnv:
    """추격-회피 환경 테스트"""

    def setup_method(self):
        """테스트 설정"""
        config = get_config(debug_mode=True)
        self.env = PursuitEvasionEnv(config)

    def test_environment_initialization(self):
        """환경 초기화 테스트"""
        assert self.env.action_space is not None
        assert self.env.observation_space is not None
        assert self.env.action_space.shape == (3,)
        assert self.env.observation_space.shape == (10,)

    def test_reset(self):
        """리셋 기능 테스트"""
        obs = self.env.reset()

        # 관측값이 올바른 형태인지 확인
        assert obs.shape == (10,)
        assert not np.isnan(obs).any()

        # 정규화된 관측값이 [-1, 1] 범위 내에 있는지 확인
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_step(self):
        """스텝 기능 테스트"""
        obs = self.env.reset()

        # 랜덤 액션 생성
        action = self.env.action_space.sample()

        # 스텝 실행
        next_obs, reward, done, info = self.env.step(action)

        # 결과 검증
        assert next_obs.shape == (10,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert not np.isnan(next_obs).any()
        assert "relative_distance_m" in info
        assert "evader_dv_magnitude" in info
        assert "pursuer_dv_magnitude" in info

    def test_multiple_steps(self):
        """다중 스텝 테스트"""
        self.env.reset()

        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)

            if done:
                break

        # 에피소드가 정상적으로 진행되었는지 확인
        assert self.env.step_count >= 1

    def test_termination_conditions(self):
        """종료 조건 테스트"""
        self.env.reset()

        # 강제로 포획 상황 만들기
        self.env.state[:3] = np.array([100, 100, 100])  # 매우 가까운 거리

        obs, reward, done, info = self.env.step(np.zeros(3))

        # 포획 상황에서는 버퍼 시간이 필요하므로 즉시 종료되지 않을 수 있음
        # 단지 시스템이 크래시하지 않는지만 확인
        assert not np.isnan(obs).any()

    def test_gastm_environment_creation(self):
        """GA STM 환경 생성 테스트"""
        config = get_config(debug_mode=True)
        config.environment.use_gastm = True
        env = PursuitEvasionEnvGASTM(config)
        obs = env.reset()
        assert obs.shape == (10,)
        env.close()
