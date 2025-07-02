 # examples/basic_training.py
"""
기본 학습 예시
"""

from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv
from training.trainer import create_trainer


def main():
    """기본 학습 실행"""
    print("=== 기본 SAC 학습 예시 ===")
    
    # 1. 설정 로드
    config = get_config(
        experiment_name="basic_example",
        debug_mode=False
    )
    
    # 학습 파라미터 조정
    config.training.total_timesteps = 50000
    config.training.learning_rate = 0.0003
    
    print(f"설정: {config.experiment_name}")
    print(f"총 학습 스텝: {config.training.total_timesteps}")
    
    # 2. 환경 생성
    env = PursuitEvasionEnv(config)
    print("환경 생성 완료")
    
    # 3. 트레이너 생성 및 설정
    trainer = create_trainer(env, config)
    trainer.setup_model()
    print("트레이너 설정 완료")
    
    # 4. 학습 실행
    print("학습 시작...")
    trainer.train()
    
    # 5. 모델 저장
    model_path = f"models/{config.experiment_name}_final.zip"
    trainer.save_model(model_path)
    print(f"모델 저장: {model_path}")
    
    # 6. 간단한 평가
    print("모델 평가 중...")
    eval_results = trainer.evaluate(n_episodes=5)
    print(f"평가 결과: 성공률 {eval_results['success_rate']:.1%}")
    
    # 7. 정리
    env.close()
    print("학습 완료!")


if __name__ == "__main__":
    main()
