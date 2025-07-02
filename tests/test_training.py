# tests/test_training.py
"""
학습 모듈 테스트
"""

import pytest
import tempfile
import os
from unittest.mock import Mock
from training.trainer import SACTrainer
from environment.pursuit_evasion_env import PursuitEvasionEnv
from config.settings import get_config


class TestSACTrainer:
    """SAC 트레이너 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        config = get_config(debug_mode=True)
        self.env = PursuitEvasionEnv(config)
        self.trainer = SACTrainer(self.env, config)
    
    def test_trainer_initialization(self):
        """트레이너 초기화 테스트"""
        assert self.trainer.env is not None
        assert self.trainer.config is not None
        assert self.trainer.model is None  # 아직 설정되지 않음
    
    def test_model_setup(self):
        """모델 설정 테스트"""
        model = self.trainer.setup_model()
        
        assert model is not None
        assert self.trainer.model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'learn')
    
    def test_short_training(self):
        """짧은 학습 테스트"""
        self.trainer.setup_model()
        
        # 매우 짧은 학습 (테스트용)
        self.trainer.train(total_timesteps=100)
        
        # 학습이 완료되었는지 확인
        assert self.trainer.model is not None
    
    def test_model_save_load(self):
        """모델 저장/로드 테스트"""
        self.trainer.setup_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.zip")
            
            # 모델 저장
            self.trainer.save_model(model_path, save_replay_buffer=False)
            assert os.path.exists(model_path)
            
            # 모델 로드
            loaded_model = self.trainer.load_model(model_path, load_replay_buffer=False)
            assert loaded_model is not None
