"""
학습 콜백 함수들 - 벡터 환경 지원 버전
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import copy
import json
from collections import deque, Counter
from stable_baselines3.common.callbacks import BaseCallback
import torch
from orbital_mechanics.coordinate_transforms import lvlh_to_eci
from analysis.visualization import plot_eci_trajectories
from utils.constants import ANALYSIS_PARAMS
from analysis.visualization import plot_training_progress, plot_delta_v_per_episode


class EvasionTrackingCallback(BaseCallback):
    """회피 결과 추적 콜백 (벡터 환경 완전 지원)"""
    
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
        self.evader_delta_vs = []
        self.delta_v_window = deque(maxlen=window_size)
        
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
        """에피소드 종료 시 처리 - 벡터 환경 완전 지원 (수정된 버전)"""
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])

        if not isinstance(dones, (list, np.ndarray)):
            dones = [dones]
        # SB3 2.x returns infos as Tuple[Dict]; DummyVecEnv uses list
        # → treat both uniformly, wrapping only non-sequence objects
        if not isinstance(infos, (list, tuple)):
            infos = [infos]

        # 각 환경 처리
        for env_idx, done in enumerate(dones):
            if done:
                # Robust extraction across SubprocVecEnv/DummyVecEnv & Gymnasium:
                # - Guard against empty/short `infos`
                # - Clamp `env_idx` to valid range
                # - Handle Gymnasium sometimes returning list for `final_info`
                if not isinstance(infos, (list, tuple)) or len(infos) == 0:
                    # Nothing to log this step (can happen due to async/timing); just continue
                    return True
                # clamp env_idx into [0, len(infos)-1]
                env_idx = max(0, min(env_idx, len(infos) - 1))
                _fi = infos[env_idx].get('final_info')
                if isinstance(_fi, (list, tuple)):
                    _fi = _fi[0] if _fi else None
                final_info = _fi or infos[env_idx]
                if final_info:
                    self._process_episode_end(final_info, env_idx)
        
        return True
    
    def _process_episode_end(self, info, env_idx=0):
        """에피소드 종료 처리"""
        self.episode_count += 1
        
        # 결과 정보 추출
        outcome = info.get('outcome', 'unknown').lower()
        termination_details = info.get('termination_details', {})
        
        # 성공 여부 판단
        success = outcome in ['permanent_evasion', 'conditional_evasion', 'temporary_evasion', 'max_steps_reached']
        self.success_window.append(1 if success else 0)

        total_evader_dv = self._extract_total_delta_v(info, termination_details)
        if total_evader_dv is None:
            total_evader_dv = 0.0
        self.evader_delta_vs.append(total_evader_dv)
        self.delta_v_window.append(total_evader_dv)
        
        # 보상 정보
        evader_reward = termination_details.get('evader_reward', info.get('evader_reward', 0))
        pursuer_reward = termination_details.get('pursuer_reward', info.get('pursuer_reward', 0))
        
        # 각종 메트릭 업데이트
        self._update_outcome_counts(outcome)
        self.evader_rewards.append(evader_reward)
        self.pursuer_rewards.append(pursuer_reward)
        self.nash_equilibrium_metrics.append(info.get('nash_metric', 0))
        
        # 버퍼 시간 기록
        buffer_time = termination_details.get('buffer_time', info.get('buffer_time', 0))
        if buffer_time > 0:
            self.buffer_time_stats.append(buffer_time)
        
        # 성공률 계산
        if len(self.success_window) > 0:
            success_rate = sum(self.success_window) / len(self.success_window)
            self.success_rates.append(success_rate)
        
        # 에피소드 정보 기록
        episode_info = {
            'episode': self.episode_count,
            'env_idx': env_idx,
            'outcome': outcome,
            'success': success,
            'evader_reward': evader_reward,
            'pursuer_reward': pursuer_reward,
            'nash_metric': self.nash_equilibrium_metrics[-1] if self.nash_equilibrium_metrics else 0,
            'buffer_time': buffer_time,
            'evader_elements': info.get('initial_evader_orbital_elements', {}),
            'pursuer_elements': info.get('initial_pursuer_orbital_elements', {}),
            'initial_distance': (
                info.get('initial_relative_distance') or
                info.get('initial_distance') or
                info.get('initial_distance_m') or 0
            ),
            'final_distance': (
                info.get('final_relative_distance') or
                info.get('final_distance') or
                info.get('relative_distance_m') or
                info.get('relative_distance') or 0
            ),
            'total_evader_delta_v': total_evader_dv,
        }
        self.episodes_info.append(episode_info)
        
        # 주기적 로그 출력
        if self.episode_count % 100 == 0:
            self._print_progress_log()
            self.plot_interim_results()
        
        # Tensorboard 로깅
        self._log_to_tensorboard(success_rate if 'success_rate' in locals() else 0.0)
    
    def _update_outcome_counts(self, outcome):
        """결과별 카운트 업데이트"""
        outcome = outcome.lower()
        if 'captured' in outcome:
            self.captures += 1
        elif 'permanent_evasion' in outcome:
            self.permanent_evasions += 1
            self.evasions += 1
        elif 'conditional_evasion' in outcome:
            self.conditional_evasions += 1
            self.evasions += 1
        elif 'temporary_evasion' in outcome:
            self.temporary_evasions += 1
        elif 'fuel_depleted' in outcome:
            self.fuel_depleted += 1
        elif 'max_steps' in outcome:
            self.max_steps += 1

    def _extract_total_delta_v(self, info, termination_details):
        """에피소드 전체 Delta-V 추출"""
        candidates = [
            info.get('total_evader_delta_v'),
            termination_details.get('delta_v_used') if termination_details else None,
            info.get('evader_total_delta_v_ms'),
            termination_details.get('evader_total_delta_v_ms') if termination_details else None,
        ]

        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _print_progress_log(self):
        """진행 상황 로그 출력"""
        success_rate = self.success_rates[-1] if self.success_rates else 0
        
        print(f"\n{'='*70}")
        print(f"에피소드 {self.episode_count} - 성공률(최근 {len(self.success_window)}): {success_rate:.2%}")
        print(f"{'='*70}")
        print(f"  - 결과 분포: 포획={self.captures}, 영구회피={self.permanent_evasions}, "
              f"조건부회피={self.conditional_evasions}, 임시회피={self.temporary_evasions}, "
              f"연료소진={self.fuel_depleted}, 최대스텝={self.max_steps}")

        # 버퍼 시간 통계
        if self.buffer_time_stats:
            print(f"  - 평균 버퍼 시간: {np.mean(self.buffer_time_stats):.2f}초")

        if self.evader_delta_vs:
            window_values = list(self.delta_v_window) if self.delta_v_window else self.evader_delta_vs
            print(f"  - 최근 평균 ΔV 사용량: {np.mean(window_values):.2f} m/s")
            print(f"  - 누적 ΔV 평균: {np.mean(self.evader_delta_vs):.2f} m/s")

        # Nash Equilibrium 메트릭 출력
        if len(self.nash_equilibrium_metrics) > 0:
            print(f"  - Nash Equilibrium 메트릭: {self.nash_equilibrium_metrics[-1]:.4f}")
            print(f"  - 최근 회피자/추격자 보상 평균: {np.mean(self.evader_rewards[-100:]):.4f}/"
                  f"{np.mean(self.pursuer_rewards[-100:]):.4f}")
            print(f"  - Zero-Sum 검증: {np.mean(self.evader_rewards[-100:]) + np.mean(self.pursuer_rewards[-100:]):.6f}")
        
        # 최근 에피소드 정보 출력
        if self.episodes_info:
            latest_info = self.episodes_info[-1]
            print("\n초기 조건:")
            
            # 회피자 궤도 요소
            evader_elements = latest_info.get('evader_elements', {})
            if evader_elements:
                print("  회피자 궤도 요소:")
                print(f"    - 반장축(a): {evader_elements.get('a', 0)/1000:.2f} km")
                print(f"    - 이심률(e): {evader_elements.get('e', 0):.6f}")
                print(f"    - 경사각(i): {evader_elements.get('i', 0)*180/np.pi:.4f} deg")
            
            # 추격자 궤도 요소
            pursuer_elements = latest_info.get('pursuer_elements', {})
            if pursuer_elements:
                print("  추격자 궤도 요소:")
                print(f"    - 반장축(a): {pursuer_elements.get('a', 0)/1000:.2f} km")
                print(f"    - 이심률(e): {pursuer_elements.get('e', 0):.6f}")
                print(f"    - 경사각(i): {pursuer_elements.get('i', 0)*180/np.pi:.4f} deg")
            
            print(f"  초기 상대 거리: {latest_info.get('initial_distance', 0):.2f} m")
            print(f"  최종 상대 거리: {latest_info.get('final_distance', 0):.2f} m")
            print(f"  결과: {latest_info.get('outcome', 'UNKNOWN').upper()}")
        
        print(f"{'='*70}")
    
    def _log_to_tensorboard(self, success_rate):
        """Tensorboard 로깅"""
        # SB3의 logger 사용
        if self.logger is not None:
            self.logger.record("evasion/success_rate", success_rate)
            self.logger.record("evasion/capture_rate", self.captures / max(1, self.episode_count))
            self.logger.record("evasion/evade_rate", self.evasions / max(1, self.episode_count))
            self.logger.record("evasion/permanent_evasion_rate", 
                             self.permanent_evasions / max(1, self.episode_count))
            self.logger.record("evasion/conditional_evasion_rate", 
                             self.conditional_evasions / max(1, self.episode_count))
            self.logger.record("evasion/temporary_evasion_rate", 
                             self.temporary_evasions / max(1, self.episode_count))
            
            # Zero-Sum 메트릭
            if self.evader_rewards:
                self.logger.record("zero_sum/evader_reward", np.mean(self.evader_rewards[-100:]))
                self.logger.record("zero_sum/pursuer_reward", np.mean(self.pursuer_rewards[-100:]))
                self.logger.record("zero_sum/nash_metric", self.nash_equilibrium_metrics[-1])
    
    def plot_interim_results(self):
        """중간 결과 플롯 생성"""
        if self.episode_count == 0:
            return
        
        # 플롯 데이터 준비
        outcome_counts = [
            self.captures,
            self.permanent_evasions,
            self.conditional_evasions,
            self.fuel_depleted,
            self.max_steps
        ]
        
        try:
            plot_training_progress(
                success_rates=self.success_rates,
                outcome_counts=outcome_counts,
                evader_rewards=self.evader_rewards,
                pursuer_rewards=self.pursuer_rewards,
                nash_metrics=self.nash_equilibrium_metrics,
                buffer_times=self.buffer_time_stats,
                episode_count=self.episode_count,
                save_dir=self.log_dir
            )
        except Exception as e:
            if self.verbose > 0:
                print(f"플롯 생성 중 오류: {e}")

        try:
            plot_delta_v_per_episode(self.evader_delta_vs, self.log_dir)
        except Exception as e:
            if self.verbose > 0:
                print(f"Delta-V 플롯 생성 중 오류: {e}")
    
    def on_training_end(self):
        """학습 종료 시 최종 결과 저장"""
        if self.verbose > 0:
            print("\n=== 학습 종료 - 최종 통계 ===")
            print(f"총 에피소드: {self.episode_count}")
            print(f"최종 성공률: {self.success_rates[-1]:.2%}" if self.success_rates else "N/A")
            print(f"포획: {self.captures}, 회피: {self.evasions}")
            print(f"영구회피: {self.permanent_evasions}, 조건부회피: {self.conditional_evasions}")
        
        # 최종 플롯 생성
        self.plot_interim_results()
        
        # 학습 통계 저장
        stats = {
            "total_episodes": self.episode_count,
            "final_success_rate": self.success_rates[-1] if self.success_rates else 0,
            "captures": self.captures,
            "evasions": self.evasions,
            "permanent_evasions": self.permanent_evasions,
            "conditional_evasions": self.conditional_evasions,
            "temporary_evasions": self.temporary_evasions,
            "fuel_depleted": self.fuel_depleted,
            "max_steps": self.max_steps,
            "episodes_info": self.episodes_info[-100:],  # 마지막 100개만 저장
            "evader_delta_v": self.evader_delta_vs,
        }
        
        stats_path = os.path.join(self.log_dir, "evasion_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if self.verbose > 0:
            print(f"학습 통계 저장: {stats_path}")


class PerformanceCallback(BaseCallback):
    """성능 추적 및 시각화 콜백"""
    
    def __init__(self, log_dir: str, plot_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        self.plot_dir = f"{log_dir}/plots"
        
        # 디렉토리 생성
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 추적 데이터
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.outcome_counter = Counter()
        self.evader_rewards = []
        self.pursuer_rewards = []
        self.nash_metrics = []
        self.buffer_times = []
        self.episode_count = 0
        
        # 윈도우 사이즈
        self.window_size = 100
        self.success_window = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if not isinstance(dones, (list, tuple, np.ndarray)):
            dones = [dones]
        if not isinstance(infos, (list, tuple)):
            infos = [infos]

        for env_idx, done in enumerate(dones):
            if not done:
                continue

            info_index = env_idx if env_idx < len(infos) else -1
            env_info = infos[info_index] if infos else {}

            final_info = env_info.get('final_info') if isinstance(env_info, dict) else None
            if isinstance(final_info, (list, tuple)):
                final_info = final_info[0] if final_info else None
            if final_info is None:
                final_info = env_info if isinstance(env_info, dict) else {}

            if not isinstance(final_info, dict):
                final_info = {}

            self.episode_count += 1

            episode_data = final_info.get("episode", {})
            episode_reward = episode_data.get("r", final_info.get("evader_reward", 0.0))
            episode_length = episode_data.get("l", final_info.get("episode_length", 0))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            evader_reward = final_info.get("evader_reward", episode_reward)
            pursuer_reward = final_info.get("pursuer_reward", -evader_reward)
            self.evader_rewards.append(evader_reward)
            self.pursuer_rewards.append(pursuer_reward)
            self.nash_metrics.append(final_info.get("nash_metric", 0.0))
            self.buffer_times.append(final_info.get("buffer_time", 0.0))

            outcome = str(final_info.get("outcome", "unknown")).lower()
            self.outcome_counter[outcome] += 1

            success = outcome in [
                'permanent_evasion',
                'conditional_evasion',
                'temporary_evasion',
                'max_steps_reached',
            ]
            self.success_window.append(1 if success else 0)
            success_rate = float(np.mean(self.success_window)) if self.success_window else 0.0
            self.success_rates.append(success_rate)

            if self.verbose > 0 and self.episode_count % 10 == 0:
                print(f"\n에피소드 {self.episode_count}:")
                print(f"  보상: {episode_reward:.2f}")
                print(f"  성공률(최근 {len(self.success_window)}): {success_rate:.1%}")
                print(f"  Nash 메트릭: {self.nash_metrics[-1]:.3f}")
                print(f"  타임스텝: {self.num_timesteps}")

            if self.episode_count % self.plot_freq == 0:
                self._save_plots()

        return True

    def _save_plots(self):
        """플롯 저장"""
        try:
            plot_training_progress(
                success_rates=self.success_rates,
                outcome_counts=[
                    self.outcome_counter.get('captured', 0),
                    self.outcome_counter.get('permanent_evasion', 0),
                    self.outcome_counter.get('conditional_evasion', 0),
                    self.outcome_counter.get('fuel_depleted', 0),
                    self.outcome_counter.get('max_steps_reached', 0),
                ],
                evader_rewards=self.evader_rewards,
                pursuer_rewards=self.pursuer_rewards,
                nash_metrics=self.nash_metrics,
                buffer_times=self.buffer_times,
                episode_count=self.episode_count,
                save_dir=self.plot_dir
            )
            
            if self.verbose > 0:
                print(f"플롯 저장 완료: {self.plot_dir}")
                
        except Exception as e:
            print(f"플롯 저장 중 오류 발생: {e}")
    
    def save_final_stats(self):
        """최종 통계 저장"""
        stats = {
            "episodes_completed": self.episode_count,
            "final_success_rate": self.success_rates[-1] if self.success_rates else 0,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "final_nash_metric": self.nash_metrics[-1] if self.nash_metrics else 0,
        }
        
        # JSON으로 저장
        import json
        stats_path = f"{self.log_dir}/training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        # 최종 플롯 저장
        self._save_plots()
        
        if self.verbose > 0:
            print(f"최종 통계 저장: {stats_path}")


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


class EphemerisLoggerCallback(BaseCallback):
    """최종 에피소드의 ECI 궤적을 기록하는 콜백"""
    
    def __init__(self, log_dir: str, verbose: int = 0, record_last_n_episodes: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.record_last_n_episodes = record_last_n_episodes
        
        # 임시 저장용
        self.current_episode_data = {
            'times': [],
            'evader_states': [],
            'pursuer_states': []
        }
        
        # 최종 에피소드들 저장
        self.final_episodes = []
        self.is_recording = False
        self.total_timesteps = None
        
    def _on_training_start(self) -> None:
        """학습 시작 시 환경 참조 저장"""
        if hasattr(self.training_env, 'envs'):
            self.env = self.training_env.envs[0]
        else:
            self.env = self.training_env
            
        # 총 타임스텝 수 저장
        self.total_timesteps = self.model._total_timesteps
        
    def _on_step(self) -> bool:
        """각 스텝에서 호출"""
        # 남은 타임스텝 계산
        if self.total_timesteps:
            remaining_timesteps = self.total_timesteps - self.model.num_timesteps
            # 대략 1000 스텝 이하 남았을 때부터 기록 시작
            if remaining_timesteps < 1000 and not self.is_recording:
                self.is_recording = True
                if self.verbose > 0:
                    print(f"최종 에피소드 ECI 궤적 기록 시작 (남은 스텝: {remaining_timesteps})")
        
        # 기록 중일 때만 데이터 수집
        if self.is_recording:
        # ECI 좌표 계산 (VecEnv 안전 접근)
            try:
                # 벡터 환경(Subproc/Dummy)인 경우: env_method 사용
                if hasattr(self.training_env, 'env_method'):
                    t, r_e, v_e, r_p, v_p = self.training_env.env_method('get_absolute_states', indices=0)[0]
                else:
                    # 단일 환경인 경우 직접 접근
                    t = getattr(self.env, 't', 0.0)
                    r_e, v_e = self.env.evader_orbit.get_position_velocity(t)
                    r_p, v_p = lvlh_to_eci(r_e, v_e, self.env.state)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[EphemerisLogger] 상태 조회 실패: {e}")
                return True

            self.current_episode_data['times'].append(t)
            self.current_episode_data['evader_states'].append(np.concatenate((r_e, v_e)))
            self.current_episode_data['pursuer_states'].append(np.concatenate((r_p, v_p)))
        
        
        # 에피소드 종료 확인
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])

        if not isinstance(dones, (list, tuple, np.ndarray)):
            dones = [dones]
        if not isinstance(infos, (list, tuple)):
            infos = [infos]

        for env_idx, done in enumerate(dones):
            if not done or not self.is_recording:
                continue

            info_index = env_idx if env_idx < len(infos) else -1
            env_info = infos[info_index] if infos else {}
            final_info = env_info.get('final_info') if isinstance(env_info, dict) else None
            if isinstance(final_info, (list, tuple)):
                final_info = final_info[0] if final_info else None
            if final_info is None:
                final_info = env_info if isinstance(env_info, dict) else {}

            if self.current_episode_data['times']:
                episode_data = {
                    't': np.array(self.current_episode_data['times']),
                    'evader': np.array(self.current_episode_data['evader_states']),
                    'pursuer': np.array(self.current_episode_data['pursuer_states']),
                    'outcome': final_info.get('outcome', 'unknown')
                }
                self.final_episodes.append(episode_data)

                if len(self.final_episodes) > self.record_last_n_episodes:
                    self.final_episodes.pop(0)

                if self.verbose > 0:
                    print(f"에피소드 ECI 데이터 저장 (총 {len(self.final_episodes)}개)")

            self.current_episode_data = {
                'times': [],
                'evader_states': [],
                'pursuer_states': []
            }
        
        return True
    
    def _on_training_end(self) -> None:
        """학습 종료 시 최종 에피소드 저장 및 플롯"""
        if not self.final_episodes:
            if self.verbose > 0:
                print("저장된 최종 에피소드가 없습니다.")
            return
        
        # 가장 마지막 에피소드 선택
        final_episode = self.final_episodes[-1]
        
        # NumPy 파일로 저장
        path = os.path.join(self.log_dir, 'final_episode_ephemeris.npz')
        np.savez(path,
                 t=final_episode['t'],
                 evader=final_episode['evader'],
                 pursuer=final_episode['pursuer'],
                 outcome=final_episode['outcome'])
        
        if self.verbose > 0:
            print(f"최종 에피소드 ECI 데이터 저장: {path}")
            print(f"  - 시뮬레이션 시간: {final_episode['t'][-1]/60:.1f} 분")
            print(f"  - 데이터 포인트: {len(final_episode['t'])}")
            print(f"  - 결과: {final_episode['outcome']}")
        
        # 시각화
        plot_eci_trajectories(
            final_episode['t'],
            final_episode['pursuer'],
            final_episode['evader'],
            save_path=os.path.join(self.log_dir, 'final_episode'),
            title=f"Final Training Episode - {final_episode['outcome']}"
        )
        
        # 모든 최종 에피소드들도 저장 (선택사항)
        if len(self.final_episodes) > 1:
            all_path = os.path.join(self.log_dir, 'last_episodes_ephemeris.pkl')
            import pickle
            with open(all_path, 'wb') as f:
                pickle.dump(self.final_episodes, f)
            if self.verbose > 0:
                print(f"마지막 {len(self.final_episodes)}개 에피소드 저장: {all_path}")

        
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
            
            # 모드 정보 추출
            is_game_mode = info.get('is_game_mode', True)
            current_orbit_mode = info.get('current_orbit_mode', 'unknown')
            
            # 현재 환경의 데이터에 추가
            env_data = self.env_data[env_idx]
            current_step = len(env_data['current_steps'])
            env_data['current_steps'].append(current_step)
            env_data['current_distances'].append(relative_distance)
            
            # 게임 모드일 때만 delta-v 기록 (또는 모든 스텝 기록)
            env_data['current_evader_dv'].append(evader_dv_mag)
            env_data['current_pursuer_dv'].append(pursuer_dv_mag)
            
            # 모드 정보도 추가로 기록
            if 'orbit_modes' not in env_data:
                env_data['orbit_modes'] = []
            env_data['orbit_modes'].append(current_orbit_mode)
            
            # 에피소드 종료 시
            if done and 'outcome' in info:
                self.total_episode_count += 1
                env_data['episode_count'] += 1
                
                # 게임 모드에서의 실제 delta-v만 계산
                game_mode_indices = [i for i, mode in enumerate(env_data['orbit_modes']) if mode == 'game']
                
                if game_mode_indices:
                    game_evader_dvs = [env_data['current_evader_dv'][i] for i in game_mode_indices]
                    game_pursuer_dvs = [env_data['current_pursuer_dv'][i] for i in game_mode_indices]
                    total_evader_dv = sum(game_evader_dvs)
                    total_pursuer_dv = sum(game_pursuer_dvs)
                else:
                    total_evader_dv = sum(env_data['current_evader_dv'])
                    total_pursuer_dv = sum(env_data['current_pursuer_dv'])
                
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
                    'total_evader_dv': total_evader_dv,  # 게임 모드에서의 총 delta-v
                    'total_pursuer_dv': total_pursuer_dv,  # 게임 모드에서의 총 delta-v
                    'distances': env_data['current_distances'].copy(),
                    'evader_dvs': env_data['current_evader_dv'].copy(),
                    'pursuer_dvs': env_data['current_pursuer_dv'].copy(),
                    'orbit_modes': env_data['orbit_modes'].copy(),  # 모드 정보 추가
                    'game_mode_steps': len(game_mode_indices),  # 게임 모드 스텝 수
                    'total_steps': len(env_data['current_steps'])  # 전체 스텝 수
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
                env_data['orbit_modes'] = []
                env_data['episode_start_step'] = self.n_calls
        
        # plot_freq마다 전체 학습 분석 플롯 생성
        if self.n_calls % self.plot_freq == 0 and self.n_calls > 0:
            self._generate_overall_plots()
        
        return True
    
    def _plot_single_episode(self, episode_data):
        """단일 에피소드의 상세 플롯 생성 - 모드 정보 포함"""
        episode_num = episode_data['episode_num']
        env_id = episode_data['env_id']
        
        plt.figure(figsize=(15, 12))
        
        # 1. 거리 변화 (모드별 색상 구분)
        plt.subplot(4, 1, 1)
        steps = episode_data['steps']
        distances = episode_data['distances']
        orbit_modes = episode_data.get('orbit_modes', ['game'] * len(steps))
        
        # 모드별로 색상 구분하여 플롯
        game_indices = [i for i, mode in enumerate(orbit_modes) if mode == 'game']
        observe_indices = [i for i, mode in enumerate(orbit_modes) if mode == 'observe']
        
        if game_indices:
            plt.scatter([steps[i] for i in game_indices], 
                       [distances[i] for i in game_indices], 
                       c='blue', s=10, alpha=0.6, label='Game Mode')
        if observe_indices:
            plt.scatter([steps[i] for i in observe_indices], 
                       [distances[i] for i in observe_indices], 
                       c='gray', s=10, alpha=0.3, label='Observe Mode')
        
        plt.plot(steps, distances, 'k-', linewidth=0.5, alpha=0.3)  # 연결선
        plt.axhline(y=1000, color='r', linestyle='--', label='Capture Distance', alpha=0.7)
        plt.axhline(y=50000, color='g', linestyle='--', label='Evasion Distance', alpha=0.7)
        plt.xlabel('Episode Steps')
        plt.ylabel('Relative Distance (m)')
        plt.title(f'Episode {episode_num} (Env {env_id}): Distance Over Time (Outcome: {episode_data["outcome"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 2. Delta-V 사용량 (게임 모드만 표시)
        plt.subplot(4, 1, 2)
        evader_dvs = episode_data['evader_dvs']
        pursuer_dvs = episode_data['pursuer_dvs']
        
        # 게임 모드 스텝만 표시
        if game_indices:
            game_steps = [steps[i] for i in game_indices]
            game_evader_dvs = [evader_dvs[i] for i in game_indices]
            game_pursuer_dvs = [pursuer_dvs[i] for i in game_indices]
            
            plt.stem(game_steps, game_evader_dvs, 'g-', markerfmt='go', basefmt=' ', 
                    label='Evader ΔV (Game Mode)', alpha=0.7)
            plt.stem(game_steps, game_pursuer_dvs, 'r-', markerfmt='ro', basefmt=' ', 
                    label='Pursuer ΔV (Game Mode)', alpha=0.7)
        
        plt.xlabel('Episode Steps')
        plt.ylabel('Instantaneous ΔV (m/s)')
        plt.title('Delta-V Usage per Step (Game Mode Only)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 누적 Delta-V (게임 모드만 누적)
        plt.subplot(4, 1, 3)
        
        # 게임 모드에서만 누적
        cumulative_evader = np.zeros(len(steps))
        cumulative_pursuer = np.zeros(len(steps))
        evader_sum = 0
        pursuer_sum = 0
        
        for i in range(len(steps)):
            if i in game_indices:
                evader_sum += evader_dvs[i]
                pursuer_sum += pursuer_dvs[i]
            cumulative_evader[i] = evader_sum
            cumulative_pursuer[i] = pursuer_sum
        
        plt.plot(steps, cumulative_evader, 'g-', 
                label=f'Evader (Total: {evader_sum:.1f} m/s)', linewidth=2)
        plt.plot(steps, cumulative_pursuer, 'r-', 
                label=f'Pursuer (Total: {pursuer_sum:.1f} m/s)', linewidth=2)
        
        # 모드 전환 시점 표시
        for i in range(1, len(orbit_modes)):
            if orbit_modes[i] != orbit_modes[i-1]:
                plt.axvline(x=steps[i], color='gray', linestyle=':', alpha=0.5)
        
        plt.xlabel('Episode Steps')
        plt.ylabel('Cumulative ΔV (m/s)')
        plt.title('Cumulative Delta-V Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 궤도 모드 타임라인
        plt.subplot(4, 1, 4)
        mode_values = [1 if mode == 'game' else 0 for mode in orbit_modes]
        plt.fill_between(steps, 0, mode_values, alpha=0.3, step='post', 
                        label='Game Mode (1) / Observe Mode (0)')
        plt.xlabel('Episode Steps')
        plt.ylabel('Orbit Mode')
        plt.title('Orbit Mode Timeline')
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/episode_{episode_num}_details.png', dpi=150)
        plt.close()
        
        if self.verbose:
            print(f"\n[Episode {episode_num} (Env {env_id})] 상세 플롯 저장")
            print(f"  - 결과: {episode_data['outcome']}")
            print(f"  - 최종 거리: {episode_data['final_distance']:.1f} m")
            print(f"  - 게임 모드 스텝: {episode_data.get('game_mode_steps', 'N/A')} / {episode_data.get('total_steps', 'N/A')}")
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
        """환경별 통계 비교 - 게임 모드 delta-v만 고려"""
        if len(self.env_data) <= 1:
            return
        
        plt.figure(figsize=(14, 10))
        
        env_ids = sorted(self.env_data.keys())
        env_episode_counts = [self.env_data[env_id]['episode_count'] for env_id in env_ids]
        
        # 각 환경의 결과 통계
        env_outcomes = {env_id: {'captured': 0, 'evaded': 0, 'fuel_depleted': 0, 'other': 0} 
                       for env_id in env_ids}
        
        for ep in self.all_episodes_data:
            outcome = ep['outcome']
            env_id = ep['env_id']
            if 'captured' in outcome.lower():
                env_outcomes[env_id]['captured'] += 1
            elif 'evaded' in outcome.lower() or 'evasion' in outcome.lower():
                env_outcomes[env_id]['evaded'] += 1
            elif 'fuel' in outcome.lower():
                env_outcomes[env_id]['fuel_depleted'] += 1
            else:
                env_outcomes[env_id]['other'] += 1
        
        # 1. 환경별 에피소드 수
        plt.subplot(3, 2, 1)
        plt.bar(env_ids, env_episode_counts, color='skyblue')
        plt.xlabel('Environment ID')
        plt.ylabel('Episode Count')
        plt.title('Episodes per Environment')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 2. 환경별 결과 분포 (stacked bar)
        plt.subplot(3, 2, 2)
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
        
        # 3. 환경별 평균 Delta-V (게임 모드만)
        plt.subplot(3, 2, 3)
        avg_evader_dv = {}
        avg_pursuer_dv = {}
        
        for env_id in env_ids:
            env_episodes = [ep for ep in self.all_episodes_data if ep['env_id'] == env_id]
            if env_episodes:
                # 게임 모드 delta-v만 사용
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
        plt.title('Average Fuel Usage per Environment (Game Mode Only)')
        plt.xticks(x, env_ids)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 환경별 게임 모드 비율
        plt.subplot(3, 2, 4)
        game_mode_ratios = {}
        
        for env_id in env_ids:
            env_episodes = [ep for ep in self.all_episodes_data if ep['env_id'] == env_id]
            if env_episodes:
                total_steps = sum(ep.get('total_steps', 0) for ep in env_episodes)
                game_steps = sum(ep.get('game_mode_steps', 0) for ep in env_episodes)
                game_mode_ratios[env_id] = game_steps / total_steps if total_steps > 0 else 0
        
        plt.bar(env_ids, [game_mode_ratios.get(env_id, 0) for env_id in env_ids], 
                color='purple', alpha=0.7)
        plt.xlabel('Environment ID')
        plt.ylabel('Game Mode Ratio')
        plt.title('Proportion of Steps in Game Mode')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Delta-V 효율성 (성공률 vs 평균 Delta-V)
        plt.subplot(3, 2, 5)
        success_rates = []
        avg_dvs = []
        
        for env_id in env_ids:
            env_episodes = [ep for ep in self.all_episodes_data if ep['env_id'] == env_id]
            if env_episodes:
                successes = sum(1 for ep in env_episodes if 'evaded' in ep['outcome'].lower() or 'evasion' in ep['outcome'].lower())
                success_rate = successes / len(env_episodes) if env_episodes else 0
                success_rates.append(success_rate)
                avg_dvs.append(avg_evader_dv.get(env_id, 0))
        
        if success_rates and avg_dvs:
            scatter = plt.scatter(avg_dvs, success_rates, c=env_ids, cmap='viridis', s=100)
            plt.xlabel('Average Evader Delta-V (m/s)')
            plt.ylabel('Success Rate')
            plt.title('Fuel Efficiency vs Success Rate')
            plt.colorbar(scatter, label='Environment ID')
            plt.grid(True, alpha=0.3)
        
        # 6. 요약 통계
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        # 전체 통계 계산
        total_game_steps = sum(ep.get('game_mode_steps', 0) for ep in self.all_episodes_data)
        total_all_steps = sum(ep.get('total_steps', 0) for ep in self.all_episodes_data)
        game_mode_ratio = total_game_steps / total_all_steps if total_all_steps > 0 else 0
        
        summary_text = f"""
    Total Episodes: {self.total_episode_count}
    Active Environments: {len(self.env_data)}
    Avg Episodes/Env: {self.total_episode_count / max(len(self.env_data), 1):.1f}
    
    Overall Success Rate: {sum(evaded) / max(self.total_episode_count, 1):.1%}
    Overall Capture Rate: {sum(captured) / max(self.total_episode_count, 1):.1%}
    Overall Fuel Depletion: {sum(fuel_depleted) / max(self.total_episode_count, 1):.1%}
    
    Game Mode Ratio: {game_mode_ratio:.1%}
    Total Game Steps: {total_game_steps:,}
    Total All Steps: {total_all_steps:,}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
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
