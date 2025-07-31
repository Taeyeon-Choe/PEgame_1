 # examples/custom_configuration.py
"""
사용자 정의 설정 예시
"""

from config.settings import get_config, ProjectConfig
from environment.pursuit_evasion_env import PursuitEvasionEnv
from training.trainer import create_trainer
from utils.constants import R_EARTH


def create_custom_config() -> ProjectConfig:
    """사용자 정의 설정 생성"""
    
    # 기본 설정에서 시작
    config = get_config(experiment_name="custom_config_example")
    
    # 궤도 파라미터 수정
    config.orbit.a = 8000e3  # 더 높은 궤도
    config.orbit.e = 0.05    # 더 낮은 이심률
    
    # 환경 파라미터 수정
    config.environment.max_steps = 15000  # 더 긴 에피소드
    config.environment.capture_distance = 800.0  # 더 짧은 포획 거리
    config.environment.delta_v_emax = 10.0  # 더 높은 회피자 성능
    
    # 학습 파라미터 수정
    config.training.learning_rate = 0.0005
    config.training.batch_size = 1024
    config.training.total_timesteps = 75000
    
    # 시각화 설정 수정
    config.visualization.figure_size_2d = (14, 8)
    config.visualization.save_plots = True
    
    return config


def main():
    """사용자 정의 설정으로 학습"""
    print("=== 사용자 정의 설정 예시 ===")
    
    # 1. 사용자 정의 설정 생성
    config = create_custom_config()
    print("사용자 정의 설정 생성 완료")
    
    # 2. 설정 정보 출력
    print(f"실험 이름: {config.experiment_name}")
    print(f"궤도 고도: {(config.orbit.a - R_EARTH)/1000:.0f} km")
    print(f"최대 스텝: {config.environment.max_steps}")
    print(f"학습률: {config.training.learning_rate}")
    
    # 3. 환경 생성
    env = PursuitEvasionEnv(config)
    
    # 4. 설정 저장 (재사용을 위해)
    config_path = f"config/{config.experiment_name}_config.json"
    config.save_to_file(config_path)
    print(f"설정 저장: {config_path}")
    
    # 5. 트레이너 생성 및 학습
    trainer = create_trainer(env, config)
    trainer.setup_model()
    
    print("학습 시작...")
    trainer.train()
    
    # 6. 결과 저장
    model_path = f"models/{config.experiment_name}_final.zip"
    trainer.save_model(model_path)
    
    # 7. 정리
    env.close()
    print("사용자 정의 설정 학습 완료!")


if __name__ == "__main__":
    main()
