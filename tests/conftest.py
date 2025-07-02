# tests/conftest.py
"""
pytest 설정 파일
"""

import pytest
import numpy as np
import warnings

# 경고 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@pytest.fixture(autouse=True)
def setup_random_seed():
    """랜덤 시드 고정"""
    np.random.seed(42)

@pytest.fixture
def sample_trajectory():
    """샘플 궤적 데이터"""
    n_steps = 50
    states = np.random.randn(n_steps, 6) * 1000
    actions = np.random.randn(n_steps, 3) * 5
    return states, actions

@pytest.fixture
def mock_environment():
    """모의 환경"""
    from unittest.mock import Mock
    env = Mock()
    env.action_space.shape = (3,)
    env.observation_space.shape = (9,)
    env.action_space.sample.return_value = np.zeros(3)
    return env
