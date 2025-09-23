 # examples/model_evaluation.py
"""
모델 평가 예시
"""

import sys
from stable_baselines3 import SAC
from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv
from analysis.evaluator import create_evaluator


def main(model_path: str):
    """모델 평가 실행"""
    print(f"=== 모델 평가 예시: {model_path} ===")
    
    # 1. 설정 로드
    config = get_config(experiment_name="evaluation_example")
    
    # 2. 환경 생성
    env = PursuitEvasionEnv(config)
    print("환경 생성 완료")
    
    # 3. 모델 로드
    try:
        model = SAC.load(model_path, env=env)
        print("모델 로드 성공")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 4. 평가자 생성
    evaluator = create_evaluator(model, env, config)
    
    # 5. 다중 시나리오 평가
    print("다중 시나리오 평가 중...")
    results = evaluator.evaluate_multiple_scenarios(n_tests=20)
    
    # 6. 결과 출력
    summary = results['summary']
    print(f"\n평가 결과:")
    print(f"  성공률: {summary['success_rate']:.1f}%")
    print(f"  평균 최종 거리: {summary['avg_final_distance']:.2f} m")
    print(f"  평균 회피자 delta-v: {summary['avg_evader_delta_v']:.2f} m/s")
    print(f"  Zero-Sum 검증: {summary['zero_sum_verification']:.6f}")
    
    # 7. 데모 실행
    print("\n데모 실행 중...")
    demo_result = evaluator.run_demonstration()
    demo_metrics = demo_result['metrics']
    print(f"데모 결과: {'성공' if demo_metrics['success'] else '실패'}")
    
    # 8. 정리
    env.close()
    print("평가 완료!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python model_evaluation.py <model_path>")
        sys.exit(1)
    
    main(sys.argv[1])
