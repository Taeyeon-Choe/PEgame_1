 # examples/nash_equilibrium_training.py
"""
Nash Equilibrium 학습 예시
"""

from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv
from training.nash_equilibrium import train_nash_equilibrium_model


def main():
    """Nash Equilibrium 학습 실행"""
    print("=== Nash Equilibrium 학습 예시 ===")
    
    # 1. 설정 로드
    config = get_config(
        experiment_name="nash_example",
        debug_mode=False
    )
    
    # Nash 학습 파라미터 조정
    config.training.nash_total_timesteps = 100000
    
    print(f"설정: {config.experiment_name}")
    print(f"총 Nash 학습 스텝: {config.training.nash_total_timesteps}")
    
    # 2. 환경 생성
    env = PursuitEvasionEnv(config)
    print("환경 생성 완료")
    
    # 3. Nash Equilibrium 학습 실행
    print("Nash Equilibrium 학습 시작...")
    trainer = train_nash_equilibrium_model(env, config)
    
    # 4. 학습 통계 출력
    stats = trainer.get_nash_training_stats()
    print(f"최종 Nash 메트릭: {stats['final_nash_metric']:.4f}")
    print(f"수렴 달성: {stats['convergence_achieved']}")
    
    # 5. 정리
    env.close()
    print("Nash Equilibrium 학습 완료!")


if __name__ == "__main__":
    main()

