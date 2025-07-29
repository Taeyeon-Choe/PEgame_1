#!/usr/bin/env python3
"""
위성 추격-회피 게임 메인 실행 스크립트
"""

import argparse
import sys
import os
import warnings
from typing import Optional

# 경고 필터링 (프로덕션 모드)
if not os.environ.get('DEBUG'):
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

# 프로젝트 모듈 임포트
from config.settings import get_config, ProjectConfig
from environment.pursuit_evasion_env import PursuitEvasionEnv
from environment.pursuit_evasion_env_ga_stm import PursuitEvasionEnvGASTM
from training.trainer import SACTrainer, create_trainer
from training.nash_equilibrium import NashEquilibriumTrainer, train_nash_equilibrium_model
from analysis.evaluator import ModelEvaluator, create_evaluator
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch


def setup_environment(config: ProjectConfig) -> PursuitEvasionEnv:
    """환경 설정 및 생성"""
    print("추격-회피 환경 초기화 중...")
    
    # GPU 정보 출력
    if config.training.use_gpu and torch.cuda.is_available():
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CPU 모드로 실행")
    
    if config.environment.use_gastm:
        env = PursuitEvasionEnvGASTM(config)
    else:
        env = PursuitEvasionEnv(config)
    print(f"환경 초기화 완료")
    print(f"  - 관측 공간: {env.observation_space}")
    print(f"  - 액션 공간: {env.action_space}")
    print(f"  - 최대 스텝: {env.max_steps}")
    return env


def create_parallel_env(config: ProjectConfig):
    """병렬 환경 생성"""
    n_envs = config.training.n_envs
    
    def make_env():
        if config.environment.use_gastm:
            return PursuitEvasionEnvGASTM(config)
        return PursuitEvasionEnv(config)
    
    # Windows에서는 DummyVecEnv 사용
    if sys.platform == 'win32' and n_envs > 1:
        print(f"Windows 환경: DummyVecEnv 사용 ({n_envs}개 환경)")
        return DummyVecEnv([make_env for _ in range(n_envs)])
    elif n_envs > 1:
        print(f"SubprocVecEnv 사용 ({n_envs}개 병렬 환경)")
        return SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        print("단일 환경 사용")
        return make_env()


def train_standard_model(config: ProjectConfig, save_path: Optional[str] = None) -> SACTrainer:
    """표준 SAC 모델 학습"""
    print("\n=== 표준 SAC 모델 학습 시작 ===")
    
    # 병렬 환경 생성
    env = create_parallel_env(config)
    
    # 트레이너 생성
    trainer = create_trainer(env, config, experiment_name=config.experiment_name)
    
    # 모델 설정
    trainer.setup_model()
    
    # 학습 실행
    trainer.train(total_timesteps=config.training.total_timesteps)
    
    # 모델 저장
    if save_path is None:
        save_path = f"{trainer.log_dir}/models/standard_sac_final.zip"
    trainer.save_model(save_path)
    
    print(f"표준 SAC 학습 완료. 모델 저장: {save_path}")
    
    # 환경 정리
    if hasattr(env, 'close'):
        env.close()
    
    return trainer


def train_nash_model(config: ProjectConfig, save_path: Optional[str] = None) -> NashEquilibriumTrainer:
    """Nash Equilibrium 모델 학습"""
    print("\n=== Nash Equilibrium 모델 학습 시작 ===")
    
    # Nash 학습은 단일 환경 사용
    env = PursuitEvasionEnv(config)
    
    trainer = train_nash_equilibrium_model(env, config, experiment_name=config.experiment_name)
    
    if save_path:
        trainer.save_model(save_path)
    
    print("Nash Equilibrium 학습 완료")
    env.close()
    
    return trainer


def evaluate_model(model_path: str, config: ProjectConfig, n_tests: int = 10):
    """모델 평가"""
    print(f"\n=== 모델 평가 시작 ({n_tests} 시나리오) ===")
    
    # 평가용 단일 환경
    env = PursuitEvasionEnv(config)
    
    # 모델 로드
    try:
        model = SAC.load(model_path, env=env)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        env.close()
        return None
    
    evaluator = create_evaluator(model, env, config)
    results = evaluator.evaluate_multiple_scenarios(n_tests=n_tests)
    
    # 결과 출력
    summary = results['summary']
    print(f"\n평가 결과 요약:")
    print(f"  성공률: {summary.get('success_rate', 0):.1f}%")
    
    # avg_final_distance 키 확인 후 출력
    if 'avg_final_distance' in summary:
        print(f"  평균 최종 거리: {summary['avg_final_distance']:.2f} m")
    else:
        # 대체 키 확인
        for key in ['average_final_distance', 'mean_final_distance', 'final_distance']:
            if key in summary:
                print(f"  평균 최종 거리: {summary[key]:.2f} m")
                break
        else:
            # 개별 결과에서 직접 계산
            if 'individual_results' in results:
                distances = [r.get('final_distance_m', 0) for r in results['individual_results']]
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    print(f"  평균 최종 거리: {avg_distance:.2f} m")
                else:
                    print(f"  평균 최종 거리: 데이터 없음")
            else:
                print(f"  평균 최종 거리: 계산 불가")
    
    print(f"  평균 회피자 delta-v: {summary.get('avg_evader_delta_v', 0):.2f} m/s")
    print(f"  평균 Nash 메트릭: {summary.get('avg_nash_metric', 0):.4f}")
    print(f"  Zero-Sum 검증: {summary.get('zero_sum_verification', 0):.6f}")
    
    env.close()
    return evaluator


def run_demonstration(model_path: str, config: ProjectConfig):
    """데모 실행"""
    print("\n=== 데모 실행 ===")
    
    env = PursuitEvasionEnv(config)
    
    try:
        model = SAC.load(model_path, env=env)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        env.close()
        return None
    
    evaluator = create_evaluator(model, env, config)
    demo_result = evaluator.run_demonstration()
    
    metrics = demo_result['metrics']
    print(f"\n데모 결과:")
    print(f"  최종 거리: {metrics.get('final_distance_m', 0):.2f} m")
    print(f"  회피자 총 delta-v: {metrics.get('evader_total_delta_v_ms', 0):.2f} m/s")
    print(f"  성공 여부: {'성공' if metrics.get('success', False) else '실패'}")
    
    env.close()
    return demo_result


def interactive_mode():
    """대화형 모드 - GA-STM 옵션 추가 버전"""
    print("\n=== 위성 추격-회피 게임 대화형 모드 ===")
    print("\n사용 가능한 명령:")
    print("1. train - 새 모델 학습")
    print("2. evaluate - 기존 모델 평가")
    print("3. demo - 데모 실행")
    print("4. quick - 빠른 테스트 (5000 스텝)")
    print("5. exit - 종료")
    
    while True:
        choice = input("\n명령을 선택하세요 (1-5): ").strip()
        
        if choice == '1' or choice == 'train':
            print("\n=== 학습 설정 ===")
            
            # 타임스텝 설정
            timesteps = input("학습 스텝 수 (기본값: 50000): ").strip()
            timesteps = int(timesteps) if timesteps else 50000
            
            # GA-STM 사용 여부
            use_gastm = input("GA-STM 사용? (y/n, 기본값: n): ").strip().lower()
            use_gastm = use_gastm == 'y'
            
            # 고급 설정 여부
            advanced = input("고급 설정을 하시겠습니까? (y/n, 기본값: n): ").strip().lower()
            
            config = get_config(experiment_name="interactive_training")
            config.training.total_timesteps = timesteps
            config.environment.use_gastm = use_gastm
            
            if advanced == 'y':
                # c 파라미터
                c_value = input(f"c 파라미터 (기본값: {config.environment.c}): ").strip()
                if c_value:
                    config.environment.c = float(c_value)
                
                # 최대 스텝
                max_steps = input(f"최대 스텝 수 (기본값: {config.environment.max_steps}): ").strip()
                if max_steps:
                    config.environment.max_steps = int(max_steps)
                
                # Delta-V 설정
                delta_v_emax = input(f"회피자 최대 Delta-V (기본값: {config.environment.delta_v_emax} m/s): ").strip()
                if delta_v_emax:
                    config.environment.delta_v_emax = float(delta_v_emax)
                
                delta_v_pmax = input(f"추격자 최대 Delta-V (기본값: {config.environment.delta_v_pmax} m/s): ").strip()
                if delta_v_pmax:
                    config.environment.delta_v_pmax = float(delta_v_pmax)
                
                # 병렬 환경 수
                n_envs = input(f"병렬 환경 수 (기본값: {config.training.n_envs}): ").strip()
                if n_envs:
                    config.training.n_envs = int(n_envs)
            
            print(f"\n학습 설정:")
            print(f"  - 타임스텝: {config.training.total_timesteps:,}")
            print(f"  - GA-STM 사용: {config.environment.use_gastm}")
            print(f"  - c 파라미터: {config.environment.c}")
            print(f"  - 병렬 환경: {config.training.n_envs}")
            
            confirm = input("\n이 설정으로 학습을 시작하시겠습니까? (y/n): ").strip().lower()
            if confirm == 'y':
                train_standard_model(config)
            else:
                print("학습이 취소되었습니다.")
            
        elif choice == '2' or choice == 'evaluate':
            model_path = input("모델 경로 (기본값: models/standard_sac_final.zip): ").strip()
            model_path = model_path if model_path else "models/standard_sac_final.zip"
            
            if not os.path.exists(model_path):
                print(f"모델 파일을 찾을 수 없습니다: {model_path}")
                continue
            
            n_tests = input("테스트 수 (기본값: 10): ").strip()
            n_tests = int(n_tests) if n_tests else 10

            # 평가 시에도 GA-STM 옵션 제공
            use_gastm = input("평가 시 GA-STM 사용? (y/n, 기본값: n): ").strip().lower()

            config = get_config(experiment_name="interactive_evaluation")
            config.environment.use_gastm = use_gastm == 'y'

            # Delta-V 설정 입력
            delta_v_emax = input(
                f"회피자 최대 Delta-V (기본값: {config.environment.delta_v_emax} m/s): "
            ).strip()
            if delta_v_emax:
                config.environment.delta_v_emax = float(delta_v_emax)

            delta_v_pmax = input(
                f"추격자 최대 Delta-V (기본값: {config.environment.delta_v_pmax} m/s): "
            ).strip()
            if delta_v_pmax:
                config.environment.delta_v_pmax = float(delta_v_pmax)

            print("\n평가 설정:")
            print(f"  GA-STM 사용: {config.environment.use_gastm}")
            print(f"  회피자 최대 Delta-V: {config.environment.delta_v_emax} m/s")
            print(f"  추격자 최대 Delta-V: {config.environment.delta_v_pmax} m/s")

            evaluate_model(model_path, config, n_tests)
            
        elif choice == '3' or choice == 'demo':
            model_path = input("모델 경로 (기본값: models/standard_sac_final.zip): ").strip()
            model_path = model_path if model_path else "models/standard_sac_final.zip"
            
            if not os.path.exists(model_path):
                print(f"모델 파일을 찾을 수 없습니다: {model_path}")
                continue
            
            config = get_config(experiment_name="interactive_demo")
            run_demonstration(model_path, config)
            
        elif choice == '4' or choice == 'quick':
            print("\n빠른 테스트 실행 중...")
            
            # 빠른 테스트에서도 GA-STM 옵션
            use_gastm = input("GA-STM으로 테스트? (y/n, 기본값: n): ").strip().lower()
            
            config = get_config(experiment_name="quick_test", debug_mode=True)
            config.training.total_timesteps = 5000
            config.training.n_envs = 2
            config.environment.use_gastm = use_gastm == 'y'
            
            print(f"GA-STM {'사용' if config.environment.use_gastm else '미사용'}으로 테스트 시작...")
            train_standard_model(config)
            
        elif choice == '5' or choice == 'exit':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="위성 추격-회피 게임 강화학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 대화형 모드 (권장)
  python main.py
  
  # 빠른 학습 테스트
  python main.py --mode train_standard --timesteps 10000
  
  # 전체 학습
  python main.py --mode train_standard --timesteps 100000 --experiment-name my_experiment
  
  # 모델 평가
  python main.py --mode evaluate --model-path models/standard_sac_final.zip
  
  # GPU 사용 강제
  python main.py --mode train_standard --gpu
  
  # 환경 변수 조정 예시
  python main.py --mode train_standard --use-gastm --c 0.01 --max-steps 800
        """
    )
    
    # 모드 선택
    parser.add_argument('--mode', type=str, default=None,
                       choices=['train_standard', 'train_nash', 'evaluate', 'demo', 'compare'],
                       help='실행 모드 (기본값: 대화형 모드)')
    
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
    
    # 설정 관련 인자
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 모드')
    parser.add_argument('--gpu', action='store_true',
                       help='GPU 사용 강제')
    parser.add_argument('--cpu', action='store_true',
                       help='CPU 사용 강제')
    parser.add_argument('--n-envs', type=int, default=None,
                       help='병렬 환경 수')
    parser.add_argument('--save-freq', type=int, default=None,
                       help='모델 저장 주기')
    
    # 환경 변수 관련 인자 추가
    parser.add_argument('--use-gastm', action='store_true',
                       help='GA-STM 사용')
    parser.add_argument('--c', type=float, default=None,
                       help='c 파라미터 (기본값: 0.001)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='에피소드 최대 스텝 수')
    parser.add_argument('--max-delta-v-budget', type=float, default=None,
                       help='최대 Delta-V 예산 (m/s)')
    parser.add_argument('--delta-v-emax', type=float, default=None,
                       help='회피자 최대 Delta-V (m/s)')
    parser.add_argument('--delta-v-pmax', type=float, default=None,
                       help='추격자 최대 Delta-V (m/s)')
    
    args = parser.parse_args()
    
    # 모드가 지정되지 않으면 대화형 모드
    if args.mode is None:
        interactive_mode()
        return
    
    # 설정 로드
    if args.config:
        config = ProjectConfig.load_from_file(args.config)
    else:
        custom_config = {}
        
        # 타임스텝 설정
        if args.timesteps:
            custom_config['training'] = {'total_timesteps': args.timesteps}
        
        # 병렬 환경 수 설정
        if args.n_envs:
            custom_config['training'] = custom_config.get('training', {})
            custom_config['training']['n_envs'] = args.n_envs
            
        # 저장 주기 설정
        if args.save_freq:
            custom_config['training'] = custom_config.get('training', {})
            custom_config['training']['save_freq'] = args.save_freq
        
        # GPU/CPU 설정
        if args.gpu:
            custom_config['training'] = custom_config.get('training', {})
            custom_config['training']['use_gpu'] = True
        elif args.cpu:
            custom_config['training'] = custom_config.get('training', {})
            custom_config['training']['use_gpu'] = False
        
        # 환경 변수 설정
        if args.use_gastm:
            custom_config['environment'] = custom_config.get('environment', {})
            custom_config['environment']['use_gastm'] = True
        
        if args.c is not None:
            custom_config['environment'] = custom_config.get('environment', {})
            custom_config['environment']['c'] = args.c
            
        if args.max_steps is not None:
            custom_config['environment'] = custom_config.get('environment', {})
            custom_config['environment']['max_steps'] = args.max_steps
            
        if args.max_delta_v_budget is not None:
            custom_config['environment'] = custom_config.get('environment', {})
            custom_config['environment']['max_delta_v_budget'] = args.max_delta_v_budget
            
        if args.delta_v_emax is not None:
            custom_config['environment'] = custom_config.get('environment', {})
            custom_config['environment']['delta_v_emax'] = args.delta_v_emax
            
        if args.delta_v_pmax is not None:
            custom_config['environment'] = custom_config.get('environment', {})
            custom_config['environment']['delta_v_pmax'] = args.delta_v_pmax
        
        config = get_config(
            experiment_name=args.experiment_name or f"{args.mode}_experiment",
            debug_mode=args.debug,
            custom_config=custom_config
        )
    
    # 설정 출력
    print("\n=== 설정 정보 ===")
    print(f"실행 모드: {args.mode}")
    print(f"실험 이름: {config.experiment_name}")
    print(f"GPU 사용: {config.training.use_gpu}")
    print(f"디버그 모드: {config.debug_mode}")
    print(f"GA-STM 사용: {config.environment.use_gastm}")
    
    if args.mode.startswith('train'):
        print(f"학습 스텝: {config.training.total_timesteps:,}")
        print(f"병렬 환경 수: {config.training.n_envs}")
        print(f"저장 주기: {config.training.save_freq}")
        
    print(f"\n환경 설정:")
    print(f"  c 파라미터: {config.environment.c}")
    print(f"  최대 스텝: {config.environment.max_steps}")
    print(f"  최대 Delta-V 예산: {config.environment.max_delta_v_budget} m/s")
    print(f"  회피자 최대 Delta-V: {config.environment.delta_v_emax} m/s")
    print(f"  추격자 최대 Delta-V: {config.environment.delta_v_pmax} m/s")
    
    try:
        # 모드별 실행
        if args.mode == 'train_standard':
            trainer = train_standard_model(config)
            
            # 학습 후 간단한 평가
            if input("\n학습된 모델을 평가하시겠습니까? (y/n): ").lower() == 'y':
                evaluate_model(f"{trainer.log_dir}/models/standard_sac_final.zip", config, 5)
            
        elif args.mode == 'train_nash':
            trainer = train_nash_model(config)
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                print("평가 모드에서는 --model-path가 필요합니다.")
                sys.exit(1)
            
            evaluate_model(args.model_path, config, args.n_tests)
            
        elif args.mode == 'demo':
            if not args.model_path:
                print("데모 모드에서는 --model-path가 필요합니다.")
                sys.exit(1)
            
            run_demonstration(args.model_path, config)
        
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


if __name__ == "__main__":
    main()
