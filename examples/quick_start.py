 # examples/quick_start.py
"""
빠른 시작 예시 (최소한의 코드)
"""

from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv
from training.trainer import create_trainer


def quick_demo():
    """최소한의 코드로 빠른 데모"""
    # 디버그 모드로 빠른 설정
    config = get_config(debug_mode=True)
    config.training.total_timesteps = 1000  # 매우 짧은 학습
    
    # 환경 및 트레이너 생성
    env = PursuitEvasionEnv(config)
    trainer = create_trainer(env, config)
    trainer.setup_model()
    
    # 학습 및 평가
    print("빠른 학습 시작...")
    trainer.train()
    
    print("평가 중...")
    results = trainer.evaluate(n_episodes=3)
    print(f"결과: 성공률 {results['success_rate']:.1%}")
    
    # 정리
    env.close()
    print("완료!")


if __name__ == "__main__":
    quick_demo()
