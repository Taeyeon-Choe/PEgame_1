"""
위성 추격-회피 게임 메인 실행 스크립트
"""

import argparse
import sys
import os
from typing import Optional

# 프로젝트 모듈 임포트
from config.settings import get_config, ProjectConfig
from environment.pursuit_evasion_env import PursuitEvasionEnv
from training.trainer import SACTrainer, create_trainer
from training.nash_equilibrium import NashEquilibriumTrainer, train_nash_equilibrium_model
from analysis.evaluator import ModelEvaluator, create_evaluator
from stable_baselines3 import SAC


def setup_environment(config: ProjectConfig) -> PursuitEvasionEnv:
    """환경 설정 및 생성"""
    print("추격-회피 환경 초기화 중...")
    env = PursuitEvasionEnv(config)
    print(f"환경 초기화 완료")
    print(f"  - 관측 공간: {env.observation_space}")
    print(f"  - 액션 공간: {env.action_space}")
    return env


def train_standard_model(env: PursuitEvasionEnv, config: ProjectConfig) -> SACTrainer:
    """표준 SAC 모델 학습"""
    print("\n=== 표준 SAC 모델 학습 시작 ===")
    
    # 트레이너 생성
    trainer = create_trainer(env, config, experiment_name="standard_sac")
    
    # 모델 설정
    trainer.setup_model()
    
    # 학습 실행
    trainer.train(total_timesteps=config.training.total_timesteps)
    
    # 모델 저장
    model_path = f"{trainer.log_dir}/models/standard_sac_final.zip"
    trainer.save_model(model_path)
    
    print(f"표준 SAC 학습 완료. 모델 저장: {model_path}")
    return trainer


def train_nash_model(env: PursuitEvasionEnv, config: ProjectConfig) -> NashEquilibriumTrainer:
    """Nash Equilibrium 모델 학습"""
    print("\n=== Nash Equilibrium 모델 학습 시작 ===")
    
    trainer = train_nash_equilibrium_model(env, config, experiment_name="nash_equilibrium")
    
    print("Nash Equilibrium 학습 완료")
    return trainer


def load_trained_model(model_path: str, env: PursuitEvasionEnv) -> SAC:
    """저장된 모델 로드"""
    print(f"모델 로드 중: {model_path}")
    
    try:
        model = SAC.load(model_path, env=env)
        print("모델 로드 성공")
        return model
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None


def evaluate_model(model: SAC, env: PursuitEvasionEnv, 
                  config: ProjectConfig, n_tests: int = 10):
    """모델 평가"""
    print(f"\n=== 모델 평가 시작 ({n_tests} 시나리오) ===")
    
    evaluator = create_evaluator(model, env, config)
    results = evaluator.evaluate_multiple_scenarios(n_tests=n_tests)
    
    # 결과 출력
    summary = results['summary']
    print(f"\n평가 결과 요약:")
    print(f"  성공률: {summary['success_rate']:.1f}%")
    print(f"  평균 최종 거리: {summary['avg_final_distance']:.2f} m")
    print(f"  평균 회피자 delta-v: {summary['avg_evader_delta_v']:.2f} m/s")
    print(f"  평균 Nash 메트릭: {summary['avg_nash_metric']:.4f}")
    print(f"  Zero-Sum 검증: {summary['zero_sum_verification']:.6f}")
    
    return evaluator


def run_demonstration(model: SAC, env: PursuitEvasionEnv, config: ProjectConfig):
    """데모 실행"""
    print("\n=== 데모 실행 ===")
    
    evaluator = create_evaluator(model, env, config)
    demo_result = evaluator.run_demonstration()
    
    metrics = demo_result['metrics']
    print(f"데모 결과:")
    print(f"  최종 거리: {metrics['final_distance_m']:.2f} m")
    print(f"  회피자 총 delta-v: {metrics['evader_total_delta_v_ms']:.2f} m/s")
    print(f"  성공 여부: {'성공' if metrics['success'] else '실패'}")
    
    return demo_result


def compare_models(model_paths: list, env: PursuitEvasionEnv, config: ProjectConfig):
    """다중 모델 비교"""
    print(f"\n=== 모델 비교 ({len(model_paths)}개 모델) ===")
    
    models = []
    model_names = []
    
    for i, path in enumerate(model_paths):
        model = load_trained_model(path, env)
        if model:
            models.append(model)
            model_names.append(f"Model_{i+1}")
    
    if not models:
        print("로드할 수 있는 모델이 없습니다.")
        return
    
    # 첫 번째 모델로 평가자 생성
    evaluator = create_evaluator(models[0], env, config)
    
    # 모델 비교 실행
    comparison_results = evaluator.compare_models(models, model_names)
    
    # 결과 출력
    print("\n모델 비교 결과:")
    for name, results in comparison_results.items():
        print(f"{name}: 성공률 {results['success_rate']:.1f}%")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="위성 추격-회피 게임")
    
    # 모드 선택
    parser.add_argument('--mode', type=str, default='train_standard',
                       choices=['train_standard', 'train_nash', 'evaluate', 'demo', 'compare'],
                       help='실행 모드')
    
    # 학습 관련 인자
    parser.add_argument('--timesteps', type=int, default=None,
                       help='학습 스텝 수')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='실험 이름')
    
    # 평가 관련 인자  
    parser.add_argument('--model-path', type=str, default=None,
                       help='평가할 모델 경로')
    parser.add_argument('--n-tests', type=int, default=10,
                       help='평가 시나리오 수')
    
    # 비교 관련 인자
    parser.add_argument('--model-paths', type=str, nargs='+', default=None,
                       help='비교할 모델 경로들')
    
    # 설정 관련 인자
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 모드')
    parser.add_argument('--gpu', action='store_true',
                       help='GPU 사용 강제')
    parser.add_argument('--cpu', action='store_true',
                       help='CPU 사용 강제')
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = ProjectConfig.load_from_file(args.config)
    else:
        custom_config = {}
        
        # 타임스텝 설정
        if args.timesteps:
            custom_config['training'] = {'total_timesteps': args.timesteps}
        
        # GPU/CPU 설정
        if args.gpu:
            custom_config['training'] = custom_config.get('training', {})
            custom_config['training']['use_gpu'] = True
        elif args.cpu:
            custom_config['training'] = custom_config.get('training', {})
            custom_config['training']['use_gpu'] = False
        
        config = get_config(
            experiment_name=args.experiment_name,
            debug_mode=args.debug,
            custom_config=custom_config
        )
    
    # 설정 출력
    print("=== 설정 정보 ===")
    print(f"실행 모드: {args.mode}")
    print(f"실험 이름: {config.experiment_name}")
    print(f"GPU 사용: {config.training.use_gpu}")
    print(f"디버그 모드: {config.debug_mode}")
    
    # 환경 생성
    env = setup_environment(config)
    
    try:
        # 모드별 실행
        if args.mode == 'train_standard':
            trainer = train_standard_model(env, config)
            
            # 학습 후 간단한 평가
            print("\n학습된 모델 평가 중...")
            evaluator = create_evaluator(trainer.model, env, config)
            evaluator.evaluate_multiple_scenarios(n_tests=3)
            
        elif args.mode == 'train_nash':
            trainer = train_nash_model(env, config)
            
            # 학습 후 간단한 평가
            print("\n학습된 모델 평가 중...")
            evaluator = create_evaluator(trainer.model, env, config)
            evaluator.evaluate_multiple_scenarios(n_tests=3)
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                print("평가 모드에서는 --model-path가 필요합니다.")
                sys.exit(1)
            
            model = load_trained_model(args.model_path, env)
            if model:
                evaluate_model(model, env, config, args.n_tests)
            
        elif args.mode == 'demo':
            if not args.model_path:
                print("데모 모드에서는 --model-path가 필요합니다.")
                sys.exit(1)
            
            model = load_trained_model(args.model_path, env)
            if model:
                run_demonstration(model, env, config)
            
        elif args.mode == 'compare':
            if not args.model_paths:
                print("비교 모드에서는 --model-paths가 필요합니다.")
                sys.exit(1)
            
            compare_models(args.model_paths, env, config)
        
        print("\n프로그램 정상 종료")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # 리소스 정리
        env.close()


def quick_train():
    """빠른 학습 함수 (스크립트 내에서 직접 호출용)"""
    config = get_config(experiment_name="quick_test", debug_mode=True)
    env = setup_environment(config)
    
    trainer = train_standard_model(env, config)
    evaluator = create_evaluator(trainer.model, env, config)
    evaluator.evaluate_multiple_scenarios(n_tests=3)
    
    env.close()
    return trainer


def quick_demo(model_path: str):
    """빠른 데모 함수"""
    config = get_config(experiment_name="quick_demo")
    env = setup_environment(config)
    
    model = load_trained_model(model_path, env)
    if model:
        run_demonstration(model, env, config)
    
    env.close()


if __name__ == "__main__":
    main()