"""
Nash Equilibrium 학습을 위한 모듈
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from training.trainer import SACTrainer
from config.settings import ProjectConfig


class NashEquilibriumTrainer(SACTrainer):
    """Nash Equilibrium을 위한 미니맥스 SAC 트레이너"""
    
    def __init__(self, env, config: Optional[ProjectConfig] = None):
        super().__init__(env, config)
        
        # Nash Equilibrium 관련 설정
        self.minimax_iterations = 2
        self.pursuer_optimization_episodes = 200
        
        # 전략 평가 기록
        self.strategy_evaluation_history = []
        self.nash_convergence_history = []
    
    def train_nash_equilibrium(self, total_timesteps: Optional[int] = None, 
                              minimax_iterations: Optional[int] = None) -> 'NashEquilibriumTrainer':
        """
        Nash Equilibrium을 위한 미니맥스 학습
        
        Args:
            total_timesteps: 총 학습 스텝 수
            minimax_iterations: 미니맥스 반복 횟수
            
        Returns:
            self
        """
        if total_timesteps is None:
            total_timesteps = self.training_config.nash_total_timesteps
        
        if minimax_iterations is None:
            minimax_iterations = self.minimax_iterations
        
        print(f"Nash Equilibrium 미니맥스 학습 시작...")
        print(f"총 스텝: {total_timesteps:,}, 미니맥스 반복: {minimax_iterations}")
        
        # 모델 설정 (아직 안된 경우)
        if self.model is None:
            self.setup_model()
        
        # 반복당 스텝 수 계산
        steps_per_iteration = total_timesteps // minimax_iterations
        
        for iteration in range(minimax_iterations):
            print(f"\n=== 미니맥스 반복 {iteration+1}/{minimax_iterations} ===")
            
            # 1. 회피자 정책 학습
            print("회피자 정책 학습 중...")
            self.train(
                total_timesteps=steps_per_iteration,
                reset_num_timesteps=(iteration == 0),
                tb_log_name=f"nash_evader_iter{iteration+1}"
            )
            
            # 2. 추격자 전략 최적화
            print("추격자 전략 최적화 중...")
            pursuer_stats = self.optimize_pursuer_strategy()
            
            # 3. Nash Equilibrium 수렴도 평가
            nash_metric = self.evaluate_nash_convergence()
            self.nash_convergence_history.append(nash_metric)
            
            print(f"Nash Equilibrium 메트릭: {nash_metric:.4f}")
            print(f"추격자 최적화 결과: {pursuer_stats}")
            
            # 4. 중간 모델 저장
            self.save_model(f"{self.log_dir}/models/nash_iter{iteration+1}.zip")
            
            # 5. 수렴 조건 체크
            if self.check_nash_convergence():
                print(f"Nash Equilibrium 수렴 달성! (반복 {iteration+1})")
                break
        
        print("Nash Equilibrium 학습 완료!")
        return self
    
    def optimize_pursuer_strategy(self, episodes: Optional[int] = None) -> Dict:
        """
        추격자 전략 최적화
        
        Args:
            episodes: 최적화에 사용할 에피소드 수
            
        Returns:
            최적화 결과 통계
        """
        if episodes is None:
            episodes = self.pursuer_optimization_episodes
        
        print(f"추격자 전략 최적화 ({episodes} 에피소드)...")
        
        # 결과 추적
        outcomes = {
            'captured': 0,
            'permanent_evasion': 0,
            'conditional_evasion': 0,
            'temporary_evasion': 0,
            'fuel_depleted': 0,
            'max_steps_reached': 0
        }
        
        # 성공 전략 수집
        successful_strategies = []
        strategy_performance = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            # 에피소드 실행
            while not done:
                # 회피자 액션 (현재 정책 사용)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # 결과 기록
            outcome = info.get('outcome', 'unknown')
            if outcome in outcomes:
                outcomes[outcome] += 1
            
            # 추격자 성공 시 전략 기록
            if outcome in ['captured', 'fuel_depleted']:
                if hasattr(self.env, 'pursuer_last_action'):
                    successful_strategies.append({
                        'state': self.env.state[:3].copy(),
                        'action': self.env.pursuer_last_action.copy(),
                        'outcome': outcome,
                        'reward': episode_reward
                    })
            
            # 성능 기록
            strategy_performance.append({
                'episode': episode,
                'outcome': outcome,
                'reward': episode_reward,
                'success': outcome in ['captured', 'fuel_depleted']
            })
            
            # 진행 상황 출력
            if (episode + 1) % max(1, episodes // 10) == 0:
                capture_rate = outcomes['captured'] / (episode + 1) * 100
                evade_rate = (outcomes['permanent_evasion'] + outcomes['conditional_evasion']) / (episode + 1) * 100
                print(f"진행률: {(episode+1)/episodes*100:.1f}% - 포획률: {capture_rate:.1f}%, 회피률: {evade_rate:.1f}%")
        
        # 결과 분석
        total_episodes = sum(outcomes.values())
        stats = {
            'total_episodes': total_episodes,
            'outcomes': outcomes,
            'capture_rate': outcomes['captured'] / total_episodes if total_episodes > 0 else 0,
            'evasion_rate': (outcomes['permanent_evasion'] + outcomes['conditional_evasion']) / total_episodes if total_episodes > 0 else 0,
            'successful_strategies_count': len(successful_strategies),
            'strategy_diversity': self.analyze_strategy_diversity(successful_strategies)
        }
        
        # 전략 평가 히스토리에 추가
        self.strategy_evaluation_history.append({
            'stats': stats,
            'strategies': successful_strategies,
            'performance': strategy_performance
        })
        
        return stats
    
    def analyze_strategy_diversity(self, strategies: List[Dict]) -> Dict:
        """전략 다양성 분석"""
        if not strategies:
            return {'diversity_score': 0.0, 'clusters': 0}
        
        # 전략 위치 기반 클러스터링 (간단한 버전)
        positions = np.array([s['state'] for s in strategies])
        actions = np.array([s['action'] for s in strategies])
        
        # 거리 기반 다양성 점수 계산
        if len(positions) > 1:
            position_diversity = np.mean(np.std(positions, axis=0))
            action_diversity = np.mean(np.std(actions, axis=0))
            diversity_score = (position_diversity + action_diversity) / 2
        else:
            diversity_score = 0.0
        
        # 간단한 클러스터 수 추정
        unique_outcomes = len(set(s['outcome'] for s in strategies))
        
        return {
            'diversity_score': diversity_score,
            'clusters': unique_outcomes,
            'position_std': np.std(positions, axis=0).tolist() if len(positions) > 0 else [0, 0, 0],
            'action_std': np.std(actions, axis=0).tolist() if len(actions) > 0 else [0, 0, 0]
        }
    
    def evaluate_nash_convergence(self) -> float:
        """Nash Equilibrium 수렴도 평가"""
        # 최근 성능 데이터 수집
        if not self.strategy_evaluation_history:
            return 0.0
        
        recent_stats = self.strategy_evaluation_history[-1]['stats']
        
        # 1. 전략 안정성 (포획률의 변화)
        if len(self.strategy_evaluation_history) >= 2:
            prev_capture_rate = self.strategy_evaluation_history[-2]['stats']['capture_rate']
            curr_capture_rate = recent_stats['capture_rate']
            capture_stability = 1.0 - abs(curr_capture_rate - prev_capture_rate)
        else:
            capture_stability = 0.5
        
        # 2. 전략 다양성 (추격자가 다양한 전략을 사용하는지)
        diversity = recent_stats['strategy_diversity']['diversity_score']
        diversity_score = min(1.0, diversity / 10.0)  # 10.0으로 정규화
        
        # 3. 성능 균형 (회피자와 추격자의 성능 균형)
        capture_rate = recent_stats['capture_rate']
        evasion_rate = recent_stats['evasion_rate']
        balance_score = 1.0 - abs(capture_rate - evasion_rate)
        
        # 종합 Nash 수렴 메트릭
        nash_metric = 0.4 * capture_stability + 0.3 * diversity_score + 0.3 * balance_score
        
        return nash_metric
    
    def check_nash_convergence(self, threshold: float = 0.8, 
                              consecutive_checks: int = 3) -> bool:
        """Nash Equilibrium 수렴 여부 확인"""
        if len(self.nash_convergence_history) < consecutive_checks:
            return False
        
        # 최근 메트릭들이 모두 임계값 이상인지 확인
        recent_metrics = self.nash_convergence_history[-consecutive_checks:]
        return all(metric > threshold for metric in recent_metrics)
    
    def get_nash_training_stats(self) -> Dict:
        """Nash Equilibrium 학습 통계 반환"""
        base_stats = self.get_training_stats()
        
        nash_stats = {
            'nash_convergence_history': self.nash_convergence_history,
            'strategy_evaluation_count': len(self.strategy_evaluation_history),
            'final_nash_metric': self.nash_convergence_history[-1] if self.nash_convergence_history else 0.0,
            'convergence_achieved': self.check_nash_convergence() if self.nash_convergence_history else False
        }
        
        # 최신 전략 분석 결과 추가
        if self.strategy_evaluation_history:
            latest_evaluation = self.strategy_evaluation_history[-1]
            nash_stats.update({
                'latest_capture_rate': latest_evaluation['stats']['capture_rate'],
                'latest_evasion_rate': latest_evaluation['stats']['evasion_rate'],
                'latest_strategy_diversity': latest_evaluation['stats']['strategy_diversity']
            })
        
        base_stats.update(nash_stats)
        return base_stats


def train_nash_equilibrium_model(env, config: Optional[ProjectConfig] = None,
                                experiment_name: str = "nash_equilibrium") -> NashEquilibriumTrainer:
    """
    Nash Equilibrium 모델 학습 헬퍼 함수
    
    Args:
        env: 학습 환경
        config: 프로젝트 설정
        experiment_name: 실험 이름
        
    Returns:
        학습된 Nash Equilibrium 트레이너
    """
    if config is None:
        from config.settings import get_config
        config = get_config(experiment_name=experiment_name)
    
    # Nash Equilibrium 트레이너 생성
    trainer = NashEquilibriumTrainer(env, config)
    
    # 모델 설정
    trainer.setup_model()
    
    # Nash Equilibrium 학습 실행
    trainer.train_nash_equilibrium()
    
    # 최종 모델 저장
    final_model_path = f"{trainer.log_dir}/models/nash_equilibrium_final.zip"
    trainer.save_model(final_model_path)
    
    print(f"\nNash Equilibrium 학습 완료!")
    print(f"최종 모델: {final_model_path}")
    
    # 학습 통계 출력
    final_stats = trainer.get_nash_training_stats()
    print(f"최종 Nash 메트릭: {final_stats['final_nash_metric']:.4f}")
    print(f"수렴 달성: {final_stats['convergence_achieved']}")
    
    return trainer


class SelfPlayTrainer(NashEquilibriumTrainer):
    """셀프 플레이 기반 트레이너 (미래 확장용)"""
    
    def __init__(self, env, config: Optional[ProjectConfig] = None):
        super().__init__(env, config)
        self.opponent_models = []  # 상대방 모델들 저장
        self.self_play_history = []
    
    def train_self_play(self, total_timesteps: int, opponent_update_freq: int = 10000):
        """
        셀프 플레이 학습 (미래 구현용 스텁)
        
        Args:
            total_timesteps: 총 학습 스텝
            opponent_update_freq: 상대방 모델 업데이트 주기
        """
        # TODO: 셀프 플레이 로직 구현
        print("셀프 플레이 학습은 아직 구현되지 않았습니다.")
        pass
    
    def update_opponent_pool(self):
        """상대방 모델 풀 업데이트 (미래 구현용)"""
        # TODO: 상대방 모델 풀 관리 로직 구현
        pass


def create_nash_trainer(env, config: Optional[ProjectConfig] = None) -> NashEquilibriumTrainer:
    """Nash Equilibrium 트레이너 생성 헬퍼"""
    return NashEquilibriumTrainer(env, config)