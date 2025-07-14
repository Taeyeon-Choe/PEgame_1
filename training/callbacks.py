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
    def __init__(self, initial_lr=0.0001, decay_rate=0.95, decay_steps=10000, verbose=0):
        super().__init__(verbose)  # BaseCallback에 verbose 전달
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def _on_step(self):
        if self.n_calls % self.decay_steps == 0 and self.n_calls > 0:
            new_lr = self.initial_lr * (self.decay_rate ** (self.n_calls // self.decay_steps))
            
            # 학습률 업데이트
            self.model.lr_schedule = lambda _: new_lr
            
            # verbose가 설정되어 있으면 출력
            if self.verbose > 0:
                print(f"[Step {self.n_calls}] 학습률 업데이트: {new_lr:.6f}")
                
            # 로거가 있으면 기록
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.record("train/learning_rate", new_lr)
        
        return True


class DetailedAnalysisCallback(BaseCallback):
    """벡터 환경을 지원하는 상세 분석 콜백"""
    
    def __init__(self, plot_freq=1000, save_dir="./analysis_plots", 
                 episode_plot_freq=100, verbose=1):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.save_dir = save_dir
        self.episode_plot_freq = episode_plot_freq
        os.makedirs(save_dir, exist_ok=True)
        
        # 벡터 환경 지원을 위한 환경별 데이터 저장
        self.env_data = {}  # 각 환경별 데이터
        self.all_episodes_data = []  # 모든 환경의 모든 에피소드
        self.total_episode_count = 0  # 모든 환경의 총 에피소드 수
        
    def _init_env_tracking(self, env_idx):
        """환경별 추적 데이터 초기화"""
        if env_idx not in self.env_data:
            self.env_data[env_idx] = {
                'current_steps': [],
                'current_distances': [],
                'current_evader_dv': [],
                'current_pursuer_dv': [],
                'episode_count': 0,
                'episode_start_step': 0
            }
    
    def _on_step(self):
        # 환경 정보 가져오기
        obs = self.locals.get('new_obs', self.locals.get('obs'))
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])
        
        # 벡터 환경인지 확인
        is_vectorized = isinstance(dones, np.ndarray)
        n_envs = len(dones) if is_vectorized else 1
        
        # 각 환경 처리
        for env_idx in range(n_envs):
            self._init_env_tracking(env_idx)
            
            # 현재 환경의 정보
            info = infos[env_idx] if is_vectorized else infos[0] if infos else {}
            done = dones[env_idx] if is_vectorized else dones
            
            # 상대 거리와 Delta-V 정보 추출
            relative_distance = info.get('relative_distance_m', 0)
            evader_dv_mag = info.get('evader_dv_magnitude', 0)
            pursuer_dv_mag = info.get('pursuer_dv_magnitude', 0)
            
            # 현재 환경의 데이터에 추가
            env_data = self.env_data[env_idx]
            current_step = len(env_data['current_steps'])
            env_data['current_steps'].append(current_step)
            env_data['current_distances'].append(relative_distance)
            env_data['current_evader_dv'].append(evader_dv_mag)
            env_data['current_pursuer_dv'].append(pursuer_dv_mag)
            
            # 에피소드 종료 시
            if done and 'outcome' in info:
                self.total_episode_count += 1
                env_data['episode_count'] += 1
                
                # 에피소드 데이터 저장
                episode_data = {
                    'env_id': env_idx,
                    'episode_num': self.total_episode_count,
                    'env_episode_num': env_data['episode_count'],
                    'start_step': env_data['episode_start_step'],
                    'end_step': self.n_calls,
                    'steps': env_data['current_steps'].copy(),
                    'outcome': info['outcome'],
                    'final_distance': relative_distance,
                    'total_evader_dv': sum(env_data['current_evader_dv']),
                    'total_pursuer_dv': sum(env_data['current_pursuer_dv']),
                    'distances': env_data['current_distances'].copy(),
                    'evader_dvs': env_data['current_evader_dv'].copy(),
                    'pursuer_dvs': env_data['current_pursuer_dv'].copy()
                }
                
                self.all_episodes_data.append(episode_data)
                
                # 100번째 전체 에피소드마다 해당 에피소드의 상세 플롯 생성
                if self.total_episode_count % self.episode_plot_freq == 0:
                    self._plot_single_episode(episode_data)
                
                # 현재 환경의 데이터 초기화
                env_data['current_steps'] = []
                env_data['current_distances'] = []
                env_data['current_evader_dv'] = []
                env_data['current_pursuer_dv'] = []
                env_data['episode_start_step'] = self.n_calls
        
        # plot_freq마다 전체 학습 분석 플롯 생성
        if self.n_calls % self.plot_freq == 0 and self.n_calls > 0:
            self._generate_overall_plots()
        
        return True
    
    def _plot_single_episode(self, episode_data):
        """단일 에피소드의 상세 플롯 생성"""
        episode_num = episode_data['episode_num']
        env_id = episode_data['env_id']
        
        plt.figure(figsize=(15, 10))
        
        # 1. 거리 변화
        plt.subplot(3, 1, 1)
        steps = episode_data['steps']
        distances = episode_data['distances']
        
        plt.plot(steps, distances, 'b-', linewidth=2)
        plt.axhline(y=1000, color='r', linestyle='--', label='Capture Distance', alpha=0.7)
        plt.axhline(y=50000, color='g', linestyle='--', label='Evasion Distance', alpha=0.7)
        plt.xlabel('Episode Steps')
        plt.ylabel('Relative Distance (m)')
        plt.title(f'Episode {episode_num} (Env {env_id}): Distance Over Time (Outcome: {episode_data["outcome"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 2. Delta-V 사용량 (순간)
        plt.subplot(3, 1, 2)
        plt.plot(steps, episode_data['evader_dvs'], 'g-', label='Evader ΔV', linewidth=2, alpha=0.7)
        plt.plot(steps, episode_data['pursuer_dvs'], 'r-', label='Pursuer ΔV', linewidth=2, alpha=0.7)
        plt.xlabel('Episode Steps')
        plt.ylabel('Instantaneous ΔV (m/s)')
        plt.title('Delta-V Usage per Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 누적 Delta-V
        plt.subplot(3, 1, 3)
        cumulative_evader = np.cumsum(episode_data['evader_dvs'])
        cumulative_pursuer = np.cumsum(episode_data['pursuer_dvs'])
        
        plt.plot(steps, cumulative_evader, 'g-', label=f'Evader (Total: {cumulative_evader[-1]:.1f} m/s)', linewidth=2)
        plt.plot(steps, cumulative_pursuer, 'r-', label=f'Pursuer (Total: {cumulative_pursuer[-1]:.1f} m/s)', linewidth=2)
        plt.xlabel('Episode Steps')
        plt.ylabel('Cumulative ΔV (m/s)')
        plt.title('Cumulative Delta-V Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/episode_{episode_num}_details.png', dpi=150)
        plt.close()
        
        if self.verbose:
            print(f"\n[Episode {episode_num} (Env {env_id})] 상세 플롯 저장")
            print(f"  - 결과: {episode_data['outcome']}")
            print(f"  - 최종 거리: {episode_data['final_distance']:.1f} m")
            print(f"  - 총 Delta-V - 회피자: {episode_data['total_evader_dv']:.1f} m/s, 추격자: {episode_data['total_pursuer_dv']:.1f} m/s")
    
    def _generate_overall_plots(self):
        """전체 학습 과정 분석 플롯"""
        if len(self.all_episodes_data) < 2:
            return
        
        # 1. 에피소드별 결과 요약
        self._plot_episode_summary()
        
        # 2. 환경별 통계
        self._plot_env_statistics()
        
        if self.verbose:
            print(f"\n[Step {self.n_calls}] 전체 분석 플롯 생성 완료")
            print(f"  - 총 에피소드: {self.total_episode_count}")
            print(f"  - 환경 수: {len(self.env_data)}")
    
    def _plot_episode_summary(self):
        """모든 환경의 에피소드별 요약 통계"""
        episodes = [ep['episode_num'] for ep in self.all_episodes_data]
        env_ids = [ep['env_id'] for ep in self.all_episodes_data]
        outcomes = [ep['outcome'] for ep in self.all_episodes_data]
        final_distances = [ep['final_distance'] for ep in self.all_episodes_data]
        total_evader_dvs = [ep['total_evader_dv'] for ep in self.all_episodes_data]
        total_pursuer_dvs = [ep['total_pursuer_dv'] for ep in self.all_episodes_data]
        
        # 색상 맵 (환경별로 다른 마커)
        env_markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D', 'd', 'P', 'X', '8', '|', '_']
        
        plt.figure(figsize=(15, 10))
        
        # 1. 최종 거리 분포 (환경별로 다른 마커)
        plt.subplot(2, 2, 1)
        for env_id in sorted(set(env_ids)):
            env_episodes = [(i, ep) for i, ep in enumerate(self.all_episodes_data) if ep['env_id'] == env_id]
            if env_episodes:
                indices = [i for i, _ in env_episodes]
                distances = [ep['final_distance'] for _, ep in env_episodes]
                plt.scatter([episodes[i] for i in indices], distances, 
                           marker=env_markers[env_id % len(env_markers)], 
                           alpha=0.6, label=f'Env {env_id}', s=30)
        
        plt.axhline(y=1000, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=50000, color='g', linestyle='--', alpha=0.5)
        plt.xlabel('Total Episode Number')
        plt.ylabel('Final Distance (m)')
        plt.title('Final Distance per Episode (All Environments)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        if len(set(env_ids)) <= 8:  # 환경이 8개 이하일 때만 범례 표시
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 회피자 연료 사용량
        plt.subplot(2, 2, 2)
        plt.scatter(episodes, total_evader_dvs, c=env_ids, cmap='tab20', alpha=0.6, s=30)
        plt.xlabel('Total Episode Number')
        plt.ylabel('Total Evader Delta-V (m/s)')
        plt.title('Evader Fuel Usage per Episode')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Environment ID')
        
        # 3. 추격자 연료 사용량
        plt.subplot(2, 2, 3)
        plt.scatter(episodes, total_pursuer_dvs, c=env_ids, cmap='tab20', alpha=0.6, s=30)
        plt.xlabel('Total Episode Number')
        plt.ylabel('Total Pursuer Delta-V (m/s)')
        plt.title('Pursuer Fuel Usage per Episode')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Environment ID')
        
        # 4. 결과 분포
        plt.subplot(2, 2, 4)
        outcome_types, counts = np.unique(outcomes, return_counts=True)
        colors = ['red' if 'CAPTURED' in o else 'green' if 'EVADED' in o else 'orange' for o in outcome_types]
        plt.bar(outcome_types, counts, color=colors)
        plt.xlabel('Outcome')
        plt.ylabel('Count (All Environments)')
        plt.title(f'Episode Outcome Distribution (Total: {self.total_episode_count})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/episode_summary_step_{self.n_calls}.png', dpi=150)
        plt.close()
    
    def _plot_env_statistics(self):
        """환경별 통계 비교"""
        if len(self.env_data) <= 1:
            return
        
        plt.figure(figsize=(12, 8))
        
        env_ids = sorted(self.env_data.keys())
        env_episode_counts = [self.env_data[env_id]['episode_count'] for env_id in env_ids]
        
        # 각 환경의 결과 통계
        env_outcomes = {env_id: {'captured': 0, 'evaded': 0, 'fuel_depleted': 0, 'other': 0} 
                       for env_id in env_ids}
        
        for ep in self.all_episodes_data:
            outcome = ep['outcome']
            env_id = ep['env_id']
            if 'CAPTURED' in outcome.upper():
                env_outcomes[env_id]['captured'] += 1
            elif 'EVADED' in outcome.upper():
                env_outcomes[env_id]['evaded'] += 1
            elif 'FUEL' in outcome.upper():
                env_outcomes[env_id]['fuel_depleted'] += 1
            else:
                env_outcomes[env_id]['other'] += 1
        
        # 1. 환경별 에피소드 수
        plt.subplot(2, 2, 1)
        plt.bar(env_ids, env_episode_counts, color='skyblue')
        plt.xlabel('Environment ID')
        plt.ylabel('Episode Count')
        plt.title('Episodes per Environment')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 2. 환경별 결과 분포 (stacked bar)
        plt.subplot(2, 2, 2)
        captured = [env_outcomes[env_id]['captured'] for env_id in env_ids]
        evaded = [env_outcomes[env_id]['evaded'] for env_id in env_ids]
        fuel_depleted = [env_outcomes[env_id]['fuel_depleted'] for env_id in env_ids]
        other = [env_outcomes[env_id]['other'] for env_id in env_ids]
        
        plt.bar(env_ids, captured, label='Captured', color='red', alpha=0.8)
        plt.bar(env_ids, evaded, bottom=captured, label='Evaded', color='green', alpha=0.8)
        plt.bar(env_ids, fuel_depleted, bottom=np.array(captured)+np.array(evaded), 
                label='Fuel Depleted', color='orange', alpha=0.8)
        plt.bar(env_ids, other, bottom=np.array(captured)+np.array(evaded)+np.array(fuel_depleted), 
                label='Other', color='gray', alpha=0.8)
        
        plt.xlabel('Environment ID')
        plt.ylabel('Outcome Count')
        plt.title('Outcome Distribution per Environment')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 3. 환경별 평균 Delta-V
        plt.subplot(2, 2, 3)
        avg_evader_dv = {}
        avg_pursuer_dv = {}
        
        for env_id in env_ids:
            env_episodes = [ep for ep in self.all_episodes_data if ep['env_id'] == env_id]
            if env_episodes:
                avg_evader_dv[env_id] = np.mean([ep['total_evader_dv'] for ep in env_episodes])
                avg_pursuer_dv[env_id] = np.mean([ep['total_pursuer_dv'] for ep in env_episodes])
        
        x = np.arange(len(env_ids))
        width = 0.35
        
        plt.bar(x - width/2, [avg_evader_dv.get(env_id, 0) for env_id in env_ids], 
                width, label='Evader', color='green', alpha=0.8)
        plt.bar(x + width/2, [avg_pursuer_dv.get(env_id, 0) for env_id in env_ids], 
                width, label='Pursuer', color='red', alpha=0.8)
        
        plt.xlabel('Environment ID')
        plt.ylabel('Average Delta-V (m/s)')
        plt.title('Average Fuel Usage per Environment')
        plt.xticks(x, env_ids)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 요약 통계
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
Total Episodes: {self.total_episode_count}
Active Environments: {len(self.env_data)}
Avg Episodes/Env: {self.total_episode_count / max(len(self.env_data), 1):.1f}

Overall Success Rate: {sum(evaded) / max(self.total_episode_count, 1):.1%}
Overall Capture Rate: {sum(captured) / max(self.total_episode_count, 1):.1%}
Overall Fuel Depletion: {sum(fuel_depleted) / max(self.total_episode_count, 1):.1%}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', 
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/env_statistics_step_{self.n_calls}.png', dpi=150)
        plt.close()
    
    def on_training_end(self):
        """학습 종료 시 최종 분석"""
        self._generate_overall_plots()
        self._save_data()
        
        if self.verbose:
            print(f"\n=== 학습 완료 통계 ===")
            print(f"총 에피소드: {self.total_episode_count}")
            print(f"활성 환경 수: {len(self.env_data)}")
            for env_id, data in self.env_data.items():
                print(f"  - 환경 {env_id}: {data['episode_count']} 에피소드")
    
    def _save_data(self):
        """수집된 데이터 저장"""
        import pickle
        data = {
            'all_episodes_data': self.all_episodes_data,
            'env_data': self.env_data,
            'total_episode_count': self.total_episode_count,
            'total_steps': self.n_calls
        }
        
        with open(f'{self.save_dir}/analysis_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        if self.verbose:
            print(f"\n분석 데이터 저장 완료: {self.save_dir}/analysis_data.pkl")
