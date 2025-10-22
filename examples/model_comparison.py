 # examples/model_comparison.py
"""
다중 모델 비교 예시
"""

from stable_baselines3 import SAC
from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv
from analysis.evaluator import create_evaluator


def main():
    """다중 모델 비교 실행"""
    print("=== 다중 모델 비교 예시 ===")
    
    # 비교할 모델 경로들
    model_paths = [
        "models/standard_model.zip",
        "models/variant_model.zip",
        "models/fine_tuned_model.zip",
    ]
    
    model_names = [
        "Standard SAC",
        "Variant SAC",
        "Fine-tuned Model",
    ]
    
    # 1. 설정 및 환경 생성
    config = get_config(experiment_name="model_comparison")
    env = PursuitEvasionEnv(config)
    
    # 2. 모델들 로드
    models = []
    valid_names = []
    
    for path, name in zip(model_paths, model_names):
        try:
            model = SAC.load(path, env=env)
            models.append(model)
            valid_names.append(name)
            print(f"모델 로드 성공: {name}")
        except Exception as e:
            print(f"모델 로드 실패 ({name}): {e}")
    
    if not models:
        print("로드할 수 있는 모델이 없습니다.")
        return
    
    # 3. 비교 실행
    evaluator = create_evaluator(models[0], env, config)
    
    print(f"\n{len(models)}개 모델 비교 중...")
    comparison_results = evaluator.compare_models(models, valid_names, n_tests_per_model=10)
    
    # 4. 결과 출력
    print("\n=== 비교 결과 ===")
    print(f"{'모델':<20} {'성공률':<10} {'평균거리':<12}")
    print('-' * 45)
    
    for name, results in comparison_results.items():
        print(f"{name:<20} {results['success_rate']:<10.1f}% {results['avg_final_distance']:<12.0f}")
    
    # 5. 정리
    env.close()
    print("\n모델 비교 완료!")


if __name__ == "__main__":
    main()
