"""
학습 콜백 함수들 - 벡터 환경 지원 버전
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import copy
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import torch

from utils.constants import ANALYSIS_PARAMS
from analysis.visualization import plot_training_progress


class EvasionTrackingCallback(BaseCallback):
    """회피 결과 추적 콜백 (벡터 환경 지원)"""
    
    def __init__(self, verbose=0, window_size=100, log_dir=None):
        super().__init__(verbose)
        self.outcomes = []
        self.success_rates = []
        self.episode_count = 0
        
        # 이동 평균을 위한 큐
        self.success_window = deque(maxlen=window_size)
        
        # 결과별 카운트
        self.captures = 0
        self.evasions = 0
        self.fuel_depleted = 0
        self.max_steps = 0
        
        # 세분화된 회피 결과
        self.permanent_evasions = 0
        self.conditional_evasions = 0
        self.temporary_evasions = 0
        self.buffer_time_stats = []
        
        # Zero-Sum 게임 메트릭
        self.evader_rewards = []
        self.pursuer_rewards = []
        self.nash_equilibrium_metrics = []
        
        # 그래프 저장 디렉토리
        self.log_dir = log_dir or f"./training_plots/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 초기 조건과 결과 기록
        self.episodes_info = []
        
        # Nash Equilibrium 평가를 위한 정책 히스토리
        self.policy_history = []
        self.eval_frequency = ANALYSIS_PARAMS['eval_frequency']
    
    def _on_step(self):
        """에피소드 종료 시 처리 - 벡터 환경 지원"""
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        
        if dones is None or infos is None:
            return True
        
        # 벡터 환경인지 확인
        is_vectorized = isinstance(dones, np.ndarray)
        
        if is_vectorized:
            # 벡터 환경 처리
            for i, done in enumerate(dones):
                if done:
                    self._process_episode_end(infos[i], i)
        else:
            # 단일 환경 처리
            if dones:
                self._process_episode_end(infos, 0)
        
        return True
    
    def _process_episode_end(self, info, env_idx):
        """에피소드 종료 처리"""
        # 환경 접근 (벡터 환경과 단일 환경 모두 지원)
        if hasattr(self.training_env, 'envs'):
            # 벡터 환경
            if hasattr(self.training_env.envs[env_idx], 'unwrapped'):
                env = self.training_env.envs[env_idx].unwrapped
            else:
                env = self.training_env.envs[env_idx]
        else:
            # 단일 환경
            env = self.training_env.unwrapped if hasattr(self.training_env, 'unwrapped') else self.training_env
        
        if 'outcome' in info:
            outcome = info['outcome']
            self.outcomes.append(outcome)
            self.episode_count += 1
            
            # 결과별 카운트 증가
            self._update_outcome_counts(outcome)
            
            # 성공 여부 계산
            success = outcome in ['permanent_evasion', 'conditional_evasion', 'evaded', 'max_steps_reached']
            self.success_window.append(float(success))
            
            # 버퍼 시간 통계 추적
            if 'buffer_time' in info:
                self.buffer_time_stats.append(info['buffer_time'])
            
            # Zero-Sum 게임 보상 기록
            evader_reward = info.get('evader_reward', 0)
            pursuer_reward = info.get('pursuer_reward', -evader_reward)
            self.evader_rewards.append(evader_reward)
            self.pursuer_rewards.append(pursuer_reward)
            
            # Nash Equilibrium 메트릭 계산
            if self.episode_count % self.eval_frequency == 0:
                nash_metric = self.evaluate_nash_equilibrium(env)
                self.nash_equilibrium_metrics.append(nash_metric)
                
                # 정책 저장
                if len(self.policy_history) < 5:
                    self.policy_history.append(
                        copy.deepcopy(self.model.policy.state_dict())
                    )
            
            # 이동 평균 성공률 계산
            if len(self.success_window) > 0:
                success_rate = sum(self.success_window) / len(self.success_window)
                self.success_rates.append(success_rate)
                
                # 에피소드 정보 기록
                self._record_episode_info(env, outcome, success, evader_reward, pursuer_reward, info)
                
                # 로그 출력
                if self.episode_count % 100 == 0:
                    self._print_progress_log(success_rate, env)
                    self.plot_interim_results()
            
            # Tensorboard 로깅
            self._log_to_tensorboard(success_rate if 'success_rate' in locals() else 0.0)
    
    def _update_outcome_counts(self, outcome):
        """결과별 카운트 업데이트"""
        if outcome == 'captured':
            self.captures += 1
        elif outcome == 'permanent_evasion':
            self.permanent_evasions += 1
            self.evasions += 1
        elif outcome == 'conditional_evasion':
            self.conditional_evasions += 1
            self.evasions += 1
        elif outcome == 'temporary_evasion':
            self.temporary_evasions += 1
        elif outcome == 'evaded':
            self.evasions += 1
        elif outcome == 'fuel_depleted':
            self.fuel_depleted += 1
        elif outcome == 'max_steps_reached':
            self.max_steps += 1
    
    def _record_episode_info(self, env, outcome, success, evader_reward, pursuer_reward, info):
        """에피소드 정보 기록"""
        episode_info = {
            'episode': self.episode_count,
            'outcome': outcome,
            'success': success,
            'evader_reward': evader_reward,
            'pursuer_reward': pursuer_reward,
            'nash_metric': self.nash_equilibrium_metrics[-1] if self.nash_equilibrium_metrics else 0,
            'buffer_time': info.get('buffer_time', 0)
        }
        
        # 환경의 초기 조건 정보가 있으면 추가
        if hasattr(env, 'initial_evader_orbital_elements'):
            episode_info.update({
                'evader_elements': env.initial_evader_orbital_elements,
                'pursuer_elements': env.initial_pursuer_orbital_elements,
                'initial_distance': env.initial_relative_distance,
                'final_distance': getattr(env, 'final_relative_distance', None),
            })
        
        self.episodes_info.append(episode_info)
    
    def _print_progress_log(self, success_rate, env):
        """진행 상황 로그 출력"""
        print(f"\n===== 에피소드 {self.episode_count} - 성공률(최근 {len(self.success_window)}): {success_rate:.2%} =====")
        print(f"  - 결과 분포: 포획={self.captures}, 영구회피={self.permanent_evasions}, "
              f"조건부회피={self.conditional_evasions}, 임시회피={self.temporary_evasions}, "
              f"연료소진={self.fuel_depleted}, 최대스텝={self.max_steps}")
        
        # 버퍼 시간 통계
        if self.buffer_time_stats:
            print(f"  - 평균 버퍼 시간: {np.mean(self.buffer_time_stats):.2f}초")
        
        # Nash Equilibrium 메트릭 출력
        if len(self.nash_equilibrium_metrics) > 0:
            print(f"  - Nash Equilibrium 메트릭: {self.nash_equilibrium_metrics[-1]:.4f}")
            print(f"  - 최근 회피자/추격자 보상 평균: {np.mean(self.evader_rewards[-100:]):.4f}/"
                  f"{np.mean(self.pursuer_rewards[-100:]):.4f}")
            print(f"  - Zero-Sum 검증: {np.mean(self.evader_rewards[-100:]) + np.mean(self.pursuer_rewards[-100:]):.6f}")
        
        # 최근 에피소드의 초기 조건 출력 (선택적)
        if hasattr(env, 'initial_evader_orbital_elements'):
            self._print_orbital_elements(env)
    
    def _print_orbital_elements(self, env):
        """궤도 요소 정보 출력"""
        if not self.episodes_info or not hasattr(env, 'initial_evader_orbital_elements'):
            return
        
        latest_info = self.episodes_info[-1]
        if 'evader_elements' not in latest_info:
            return
            
        evader_elements = latest_info['evader_elements']
        pursuer_elements = latest_info['pursuer_elements']
        
        print("\n초기 조건:")
        print("  회피자 궤도 요소:")
        print(f"    - 반장축(a): {evader_elements['a']/1000:.2f} km")
        print(f"    - 이심률(e): {evader_elements['e']:.6f}")
        print(f"    - 경사각(i): {evader_elements['i']*180/np.pi:.4f} deg")
        
        print("  추격자 궤도 요소:")
        print(f"    - 반장축(a): {pursuer_elements['a']/1000:.2f} km")
        print(f"    - 이심률(e): {pursuer_elements['e']:.6f}")
        print(f"    - 경사각(i): {pursuer_elements['i']*180/np.pi:.4f} deg")
        
        print(f"\n  초기 상대 거리: {latest_info.get('initial_distance', 0):.2f} m")
        print(f"  최종 상대 거리: {latest_info.get('final_distance', 'N/A')}")
        print(f"  결과: {latest_info['outcome'].upper()}")
        print("\n" + "="*70)
    
    def _log_to_tensorboard(self, success_rate):
        """Tensorboard에 로깅"""
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record("evasion/success_rate", success_rate)
            self.logger.record("evasion/capture_rate", self.captures / max(self.episode_count, 1))
            self.logger.record("evasion/evade_rate", self.evasions / max(self.episode_count, 1))
            
            # 세분화된 회피 결과 로깅
            self.logger.record("evasion/permanent_evasion_rate", self.permanent_evasions / max(self.episode_count, 1))
            self.logger.record("evasion/conditional_evasion_rate", self.conditional_evasions / max(self.episode_count, 1))
            self.logger.record("evasion/temporary_evasion_rate", self.temporary_evasions / max(self.episode_count, 1))
            
            # Zero-Sum 게임 메트릭 로깅
            if len(self.evader_rewards) > 0:
                self.logger.record("zero_sum/evader_reward", self.evader_rewards[-1])
                self.logger.record("zero_sum/pursuer_reward", self.pursuer_rewards[-1])
            if len(self.nash_equilibrium_metrics) > 0:
                self.logger.record("zero_sum/nash_metric", self.nash_equilibrium_metrics[-1])
            
            # 버퍼 시간 로깅
            if self.buffer_time_stats:
                self.logger.record("termination/buffer_time", self.buffer_time_stats[-1])
    
    def evaluate_nash_equilibrium(self, env):
        """Nash Equilibrium 평가 메트릭"""
        if not hasattr(self, 'model') or len(self.policy_history) < 2:
            return 0.0
        
        # 1. 정책 안정성 평가
        current_policy = self.model.policy.state_dict()
        previous_policy = self.policy_history[-1]
        
        policy_distance = 0.0
        for key in current_policy:
            if key in previous_policy:
                policy_distance += torch.norm(
                    current_policy[key] - previous_policy[key]
                ).item() ** 2
        policy_distance = np.sqrt(policy_distance)
        
        policy_stability = 1.0 / (1.0 + policy_distance)
        
        # 2. Zero-Sum 특성 검증
        if len(self.evader_rewards) >= 100:
            recent_evader = np.mean(self.evader_rewards[-100:])
            recent_pursuer = np.mean(self.pursuer_rewards[-100:])
            zero_sum_metric = 1.0 / (1.0 + abs(recent_evader + recent_pursuer))
        else:
            zero_sum_metric = 0.5
        
        # 3. 보상 변동성 감소
        if len(self.evader_rewards) >= 100:
            evader_std = np.std(self.evader_rewards[-100:])
            pursuer_std = np.std(self.pursuer_rewards[-100:])
            reward_stability = 1.0 / (1.0 + evader_std + pursuer_std)
        else:
            reward_stability = 0.5
        
        # 종합 Nash Equilibrium 메트릭
        nash_metric = 0.4 * policy_stability + 0.4 * zero_sum_metric + 0.2 * reward_stability
        
        return nash_metric
    
    def plot_interim_results(self):
        """중간 결과 시각화"""
        plot_training_progress(
            self.success_rates,
            [self.captures, self.permanent_evasions, self.conditional_evasions, 
             self.fuel_depleted, self.max_steps],
            self.evader_rewards,
            self.pursuer_rewards,
            self.nash_equilibrium_metrics,
            self.buffer_time_stats,
            self.episode_count,
            self.log_dir
        )
    
    def plot_success_rate(self):
        """학습 완료 후 최종 결과 시각화"""
        self.plot_interim_results()
        
        # 추가적인 최종 분석 그래프들
        self._plot_final_analysis()
    
    def _plot_final_analysis(self):
        """최종 분석 그래프들"""
        # 에피소드별 성공/실패 추세
        episodes = list(range(1, len(self.outcomes) + 1))
        outcomes_binary = [
            1 if o in ['permanent_evasion', 'conditional_evasion', 'evaded', 'max_steps_reached'] 
            else 0 for o in self.outcomes
        ]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(episodes, outcomes_binary, alpha=0.5, c=outcomes_binary, cmap='RdYlGn')
        plt.xlabel('에피소드')
        plt.ylabel('결과 (1=성공, 0=실패)')
        plt.title('에피소드별 성공/실패')
        plt.grid(True)
        plt.savefig(f'{self.log_dir}/episode_outcomes.png')
        plt.close()
        
        # 최종 분포 파이 차트
        labels = ['Captured', 'Permanent Evasion', 'Conditional Evasion', 'Fuel Depleted', 'Max Steps']
        sizes = [self.captures, self.permanent_evasions, self.conditional_evasions, 
                self.fuel_depleted, self.max_steps]
        
        # 0이 아닌 값들만 필터링
        filtered_sizes = []
        filtered_labels = []
        for size, label in zip(sizes, labels):
            if size > 0:
                filtered_sizes.append(size)
                filtered_labels.append(label)
        
        if filtered_sizes:
            plt.figure(figsize=(10, 10))
            plt.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%')
            plt.title('학습 완료 시 결과 분포')
            plt.savefig(f'{self.log_dir}/final_outcome_distribution.png')
            plt.close()


class PerformanceCallback(BaseCallback):
    """성능 모니터링 콜백 - 벡터 환경 지원"""
    
    def __init__(self, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.fps_history = []
        self.loss_history = []
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # FPS 계산
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                if 'time/fps' in self.model.logger.name_to_value:
                    fps = self.model.logger.name_to_value['time/fps']
                    self.fps_history.append(fps)
                    
        return True


class ModelSaveCallback(BaseCallback):
    """모델 저장 콜백 - 벡터 환경 지원"""
    
    def __init__(self, save_freq=10000, save_path="./models/", name_prefix="model", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        # 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            save_file = f"{self.save_path}/{self.name_prefix}_step_{self.n_calls}.zip"
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"모델 저장됨: {save_file}")
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """조기 종료 콜백 - 벡터 환경 지원"""
    
    def __init__(self, target_success_rate=0.8, patience=5000, verbose=0):
        super().__init__(verbose)
        self.target_success_rate = target_success_rate
        self.patience = patience
        self.best_success_rate = 0.0
        self.steps_since_improvement = 0
        self.callback_tracker = None
        
    def _on_training_start(self) -> None:
        """학습 시작 시 EvasionTrackingCallback 찾기"""
        # 다른 콜백들 중에서 EvasionTrackingCallback 찾기
        if hasattr(self.model, '_callback'):
            if hasattr(self.model._callback, 'callbacks'):
                for callback in self.model._callback.callbacks:
                    if isinstance(callback, EvasionTrackingCallback):
                        self.callback_tracker = callback
                        break
        
    def _on_step(self):
        # EvasionTrackingCallback에서 성공률 정보 가져오기
        if self.callback_tracker and len(self.callback_tracker.success_rates) > 0:
            current_success_rate = self.callback_tracker.success_rates[-1]
            
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1
            
            # 목표 달성 시 조기 종료
            if current_success_rate >= self.target_success_rate:
                if self.verbose > 0:
                    print(f"목표 성공률 달성: {current_success_rate:.2%}")
                return False
            
            # 개선이 없을 시 조기 종료
            if self.steps_since_improvement >= self.patience:
                if self.verbose > 0:
                    print(f"성능 개선 없음으로 조기 종료. 최고 성공률: {self.best_success_rate:.2%}")
                return False
        
        return True
        
class LearningRateScheduler(BaseCallback):
    """학습률 감소 스케줄러"""
    def __init__(self, initial_lr=0.0001, decay_rate=0.95, decay_steps=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def _on_step(self):
        if self.n_calls % self.decay_steps == 0:
            new_lr = self.initial_lr * (self.decay_rate ** (self.n_calls // self.decay_steps))
            self.model.lr_schedule = lambda _: new_lr
            if self.verbose > 0:
                print(f"학습률 업데이트: {new_lr:.6f}")
        return True
