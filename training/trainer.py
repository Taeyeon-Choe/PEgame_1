"""
SAC 모델 트레이너 클래스
"""

import os
import datetime
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from typing import Optional, List, Dict, Any

from config.settings import ProjectConfig
from training.callbacks import (
    EvasionTrackingCallback, PerformanceCallback, 
    ModelSaveCallback, EarlyStoppingCallback
)


class SACTrainer:
    """SAC 모델 트레이너"""
    
    def __init__(self, env, config: Optional[ProjectConfig] = None):
        """
        트레이너 초기화
        
        Args:
            env: 학습 환경
            config: 프로젝트 설정
        """
        if config is None:
            from config.settings import default_config
            config = default_config
            
        self.env = env
        self.config = config
        self.training_config = config.training
        self.paths_config = config.paths
        
        # 모델 및 로깅 설정
        self.model = None
        self.logger = None
        self.callbacks = []
        
        # 학습 기록
        self.training_history = {}
        
        # 로그 디렉토리 설정
        self.log_dir = self._setup_log_directory()
        
    def _setup_log_directory(self) -> str:
        """로그 디렉토리 설정"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = f"{self.paths_config.logs}/sac_training_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # 서브 디렉토리들 생성
        os.makedirs(f"{log_dir}/plots", exist_ok=True)
        os.makedirs(f"{log_dir}/models", exist_ok=True)
        os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
        
        return log_dir
    
    def setup_model(self, model_params: Optional[Dict[str, Any]] = None):
        """SAC 모델 설정"""
        if model_params is None:
            model_params = {}
        
        # 기본 파라미터 설정
        default_params = {
            "learning_rate": self.training_config.learning_rate,
            "buffer_size": self.training_config.buffer_size,
            "batch_size": self.training_config.batch_size,
            "tau": self.training_config.tau,
            "gamma": self.training_config.gamma,
            "ent_coef": "auto",
            "policy_kwargs": {"net_arch": self.training_config.net_arch},
            "verbose": self.training_config.verbose,
            "device": self.training_config.device,
            "tensorboard_log": f"{self.log_dir}/tensorboard"
        }
        
        # 사용자 파라미터로 덮어쓰기
        default_params.update(model_params)
        
        print(f"SAC 모델 초기화 중... (Device: {self.training_config.device})")
        self.model = SAC("MlpPolicy", self.env, **default_params)
        
        # 로거 설정
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(self.logger)
        
        return self.model
    
    def setup_callbacks(self, custom_callbacks: Optional[List] = None):
        """콜백 설정"""
        self.callbacks = []
        
        # 기본 콜백들
        # 1. 회피 성공률 추적
        evasion_callback = EvasionTrackingCallback(
            verbose=1, 
            log_dir=f"{self.log_dir}/plots"
        )
        self.callbacks.append(evasion_callback)
        
        # 2. 모델 체크포인트 저장
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_config.save_freq,
            save_path=f"{self.log_dir}/models/",
            name_prefix="sac_checkpoint",
            save_replay_buffer=True
        )
        self.callbacks.append(checkpoint_callback)
        
        # 3. 성능 모니터링
        performance_callback = PerformanceCallback(
            eval_freq=1000,
            verbose=0
        )
        self.callbacks.append(performance_callback)
        
        # 4. 추가 모델 저장
        model_save_callback = ModelSaveCallback(
            save_freq=self.training_config.save_freq * 2,
            save_path=f"{self.log_dir}/models/",
            name_prefix="sac_model",
            verbose=1
        )
        self.callbacks.append(model_save_callback)
        
        # 5. 조기 종료 (선택적)
        if hasattr(self.training_config, 'early_stopping') and self.training_config.early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                target_success_rate=0.8,
                patience=10000,
                verbose=1
            )
            self.callbacks.append(early_stopping_callback)
        
        # 사용자 정의 콜백 추가
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)
        
        return CallbackList(self.callbacks)
    
    def train(self, total_timesteps: Optional[int] = None, 
             reset_num_timesteps: bool = True,
             tb_log_name: str = "sac_run",
             callback_list: Optional[List] = None) -> 'SACTrainer':
        """
        모델 학습 실행
        
        Args:
            total_timesteps: 총 학습 스텝 수
            reset_num_timesteps: 타임스텝 카운터 리셋 여부
            tb_log_name: Tensorboard 로그 이름
            callback_list: 사용자 정의 콜백 리스트
            
        Returns:
            self (메서드 체이닝용)
        """
        if self.model is None:
            print("모델이 설정되지 않았습니다. setup_model()을 먼저 호출하세요.")
            return self
        
        if total_timesteps is None:
            total_timesteps = self.training_config.total_timesteps
        
        # 콜백 설정
        if callback_list is None:
            callback_list = self.setup_callbacks()
        
        print(f"SAC 모델 학습 시작... (총 {total_timesteps:,} 스텝)")
        print(f"로그 디렉토리: {self.log_dir}")
        
        # 학습 실행
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=self.training_config.log_interval,
            tb_log_name=tb_log_name,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False  # Rich 프로그레스 바 비활성화
        )
        
        print("학습 완료!")
        
        # 학습 히스토리 저장
        self._save_training_history()
        
        return self
    
    def save_model(self, save_path: str, save_replay_buffer: bool = True):
        """모델 저장"""
        if self.model is None:
            print("저장할 모델이 없습니다.")
            return
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 모델 저장
        self.model.save(save_path)
        print(f"모델 저장됨: {save_path}")
        
        # 리플레이 버퍼 저장 (선택적)
        if save_replay_buffer and hasattr(self.model, 'replay_buffer'):
            buffer_path = save_path.replace('.zip', '_replay_buffer.pkl')
            self.model.save_replay_buffer(buffer_path)
            print(f"리플레이 버퍼 저장됨: {buffer_path}")
    
    def load_model(self, load_path: str, load_replay_buffer: bool = True):
        """모델 로드"""
        try:
            self.model = SAC.load(load_path, env=self.env)
            print(f"모델 로드됨: {load_path}")
            
            # 리플레이 버퍼 로드 (선택적)
            if load_replay_buffer:
                buffer_path = load_path.replace('.zip', '_replay_buffer.pkl')
                if os.path.exists(buffer_path):
                    self.model.load_replay_buffer(buffer_path)
                    print(f"리플레이 버퍼 로드됨: {buffer_path}")
            
            return self.model
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """모델 평가"""
        if self.model is None:
            print("평가할 모델이 없습니다.")
            return {}
        
        print(f"모델 평가 중... ({n_episodes} 에피소드)")
        
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0
        }
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            results['episode_rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)
            
            # 성공 여부 판단
            if 'outcome' in info:
                success = info['outcome'] in ['permanent_evasion', 'conditional_evasion', 'max_steps_reached']
                if success:
                    results['success_count'] += 1
            
            print(f"에피소드 {episode+1}: 보상={episode_reward:.2f}, 길이={episode_length}")
        
        # 평가 결과 요약
        results['mean_reward'] = sum(results['episode_rewards']) / len(results['episode_rewards'])
        results['std_reward'] = torch.tensor(results['episode_rewards']).std().item()
        results['mean_length'] = sum(results['episode_lengths']) / len(results['episode_lengths'])
        results['success_rate'] = results['success_count'] / n_episodes
        
        print(f"\n평가 결과:")
        print(f"  평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  평균 길이: {results['mean_length']:.1f}")
        print(f"  성공률: {results['success_rate']:.1%}")
        
        return results
    
    def get_training_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {
            'log_dir': self.log_dir,
            'model_params': self.model.get_parameters() if self.model else None,
            'training_config': self.training_config.__dict__,
        }
        
        # 콜백에서 통계 수집
        for callback in self.callbacks:
            if isinstance(callback, EvasionTrackingCallback):
                stats.update({
                    'episodes_completed': callback.episode_count,
                    'success_rates': callback.success_rates,
                    'outcomes': callback.outcomes,
                    'nash_metrics': callback.nash_equilibrium_metrics,
                })
                break
        
        return stats
    
    def _save_training_history(self):
        """학습 히스토리 저장"""
        import json
        
        history = self.get_training_stats()
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, (list, dict, str, int, float, bool)):
                serializable_history[key] = value
            elif hasattr(value, '__dict__'):
                serializable_history[key] = value.__dict__
            else:
                serializable_history[key] = str(value)
        
        # 파일 저장
        history_path = f"{self.log_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"학습 히스토리 저장됨: {history_path}")
    
    def continue_training(self, additional_timesteps: int, 
                         tb_log_name: str = "sac_continue") -> 'SACTrainer':
        """학습 계속하기"""
        if self.model is None:
            print("계속할 모델이 없습니다.")
            return self
        
        print(f"학습 계속... (추가 {additional_timesteps:,} 스텝)")
        
        return self.train(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=False,
            tb_log_name=tb_log_name
        )
    
    def fine_tune(self, new_env, fine_tune_timesteps: int, 
                 learning_rate_multiplier: float = 0.1) -> 'SACTrainer':
        """파인 튜닝"""
        if self.model is None:
            print("파인 튜닝할 모델이 없습니다.")
            return self
        
        print(f"파인 튜닝 시작... ({fine_tune_timesteps:,} 스텝)")
        
        # 학습률 조정
        original_lr = self.model.learning_rate
        self.model.learning_rate = original_lr * learning_rate_multiplier
        
        # 새 환경 설정
        old_env = self.env
        self.env = new_env
        self.model.set_env(new_env)
        
        # 파인 튜닝 실행
        self.train(
            total_timesteps=fine_tune_timesteps,
            reset_num_timesteps=False,
            tb_log_name="sac_fine_tune"
        )
        
        # 설정 복원
        self.model.learning_rate = original_lr
        self.env = old_env
        
        print("파인 튜닝 완료!")
        return self
    
    def cleanup(self):
        """리소스 정리"""
        if self.model and hasattr(self.model, 'replay_buffer'):
            del self.model.replay_buffer
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("리소스 정리 완료")


def create_trainer(env, config: Optional[ProjectConfig] = None, 
                  experiment_name: Optional[str] = None) -> SACTrainer:
    """트레이너 생성 헬퍼 함수"""
    if config is None:
        from config.settings import get_config
        config = get_config(experiment_name=experiment_name)
    
    trainer = SACTrainer(env, config)
    return trainer