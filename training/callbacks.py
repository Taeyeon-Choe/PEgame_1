"""
학습 콜백 함수들 - 벡터 환경 지원 버전
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import copy
import json
from typing import Any, Dict, Optional, List, Callable
from collections import deque, Counter
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import torch
from orbital_mechanics.coordinate_transforms import lvlh_to_eci
from analysis.visualization import (
    plot_eci_trajectories,
    plot_training_progress,
    plot_delta_v_per_episode,
    aggregate_outcome_counts,
    plot_sac_training_metrics,
)


def _json_ready(value):
    """Convert numpy / float32 values to JSON-safe Python types."""

    if isinstance(value, (np.floating, float)):
        result = float(value)
        if np.isnan(result):
            return None
        return result

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}

    return value


class LogStdClampCallback(BaseCallback):
    """Keep gSDE log_std finite to avoid NaN exploration noise."""

    def __init__(
        self,
        min_value: float = -8.0,
        max_value: float = 2.0,
        reset_value: float = -3.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.reset_value = float(reset_value)

    def _on_step(self) -> bool:
        policy = getattr(self.model, "policy", None)
        actor = getattr(policy, "actor", None)
        log_std = getattr(actor, "log_std", None)

        if actor is None or not getattr(actor, "use_sde", False) or log_std is None:
            return True

        with torch.no_grad():
            if torch.isnan(log_std).any():
                torch.nan_to_num_(
                    log_std,
                    nan=self.reset_value,
                    posinf=self.max_value,
                    neginf=self.min_value,
                )
                if self.verbose > 0:
                    print("[LogStdClamp] NaN detected in gSDE log_std; values reset")
            log_std.clamp_(self.min_value, self.max_value)
        return True


class EvasionTrackingCallback(BaseCallback):
    """회피 결과 추적 콜백 (벡터 환경 완전 지원)"""
    
    def __init__(self, verbose=0, window_size=100, log_dir=None,
                 resume: bool = False, resume_data: Optional[Dict[str, Any]] = None):
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
        
        # 회피 결과
        self.permanent_evasions = 0
        self.buffer_time_stats = []
        self.evader_delta_vs = []
        self.delta_v_window = deque(maxlen=window_size)
        
        # Zero-Sum 게임 메트릭
        self.evader_rewards = []
        self.pursuer_rewards = []
        self.episode_rewards = []
        self.reward_breakdowns: List[Dict[str, float]] = []
        
        # 그래프 저장 디렉토리
        self.log_dir = log_dir or f"./training_plots/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 초기 조건과 결과 기록
        self.episodes_info = []
        
        self._resume_enabled = resume
        if resume and resume_data:
            self._restore_from_resume(resume_data)
    
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
        normalized_outcome = outcome.replace(' ', '_')
        if 'conditional_evasion' in normalized_outcome or 'temporary_evasion' in normalized_outcome:
            normalized_outcome = 'permanent_evasion'
        success = normalized_outcome in ['permanent_evasion', 'max_steps_reached']
        self.success_window.append(1 if success else 0)
        outcome = normalized_outcome

        total_evader_dv = self._extract_total_delta_v(info, termination_details)
        if total_evader_dv is None:
            total_evader_dv = 0.0
        self.evader_delta_vs.append(total_evader_dv)
        self.delta_v_window.append(total_evader_dv)
        
        # 보상 정보
        evader_reward = termination_details.get('evader_reward', info.get('evader_reward', 0))
        pursuer_reward = termination_details.get('pursuer_reward', info.get('pursuer_reward', 0))
        episode_data = info.get('episode')
        episode_reward = None
        if isinstance(episode_data, dict):
            episode_reward = episode_data.get('r')
        if episode_reward is None:
            episode_reward = termination_details.get('evader_total_reward') if termination_details else None
        if episode_reward is None:
            episode_reward = info.get('evader_total_reward')
        if episode_reward is None:
            episode_reward = evader_reward
        try:
            episode_reward = float(episode_reward)
        except (TypeError, ValueError):
            episode_reward = float(evader_reward)

        # 각종 메트릭 업데이트
        self._update_outcome_counts(outcome)
        self.evader_rewards.append(evader_reward)
        self.pursuer_rewards.append(pursuer_reward)
        self.episode_rewards.append(episode_reward)

        breakdown = termination_details.get('evader_reward_breakdown') or info.get('evader_reward_breakdown') or {}
        normalized_breakdown: Dict[str, float] = {}
        if isinstance(breakdown, dict):
            for key, value in breakdown.items():
                try:
                    normalized_breakdown[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        # keep length aligned with episodes
        self.reward_breakdowns.append(normalized_breakdown)
        
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
            'reward_breakdown': normalized_breakdown,
        }
        self.episodes_info.append(episode_info)
        
        # 주기적 로그 출력
        if self.episode_count % 100 == 0:
            self._print_progress_log()
            self.plot_interim_results()
        
        # Tensorboard 로깅
        self._log_to_tensorboard(success_rate if 'success_rate' in locals() else 0.0)

        # 이어학습을 위한 상태 저장 (주기적으로)
        if self._resume_enabled and (self.episode_count % 20 == 0 or self.episode_count < 20):
            self._write_resume_state()

    def _update_outcome_counts(self, outcome):
        """결과별 카운트 업데이트"""
        outcome = outcome.lower()
        if 'captured' in outcome:
            self.captures += 1
        elif any(keyword in outcome for keyword in (
            'permanent_evasion',
            'conditional_evasion',
            'temporary_evasion',
            'evaded',
        )):
            self.permanent_evasions += 1
            self.evasions += 1
        elif 'fuel_depleted' in outcome:
            self.fuel_depleted += 1
        elif 'max_steps' in outcome:
            self.max_steps += 1
            self.evasions += 1

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
        print(
            "  - 결과 분포: 포획={0}, 회피={1} (영구={2}, 최대스텝={3}), 연료소진={4}".format(
                self.captures,
                self.evasions,
                self.permanent_evasions,
                self.max_steps,
                self.fuel_depleted,
            )
        )

        summary_counts, evaded_breakdown = aggregate_outcome_counts(
            {
                'captured': self.captures,
                'permanent_evasion': self.permanent_evasions,
                'fuel_depleted': self.fuel_depleted,
                'max_steps_reached': self.max_steps,
            },
            include_breakdown=True,
        )
        if summary_counts:
            total_macro = max(sum(summary_counts.values()), 1)
            captured_macro = summary_counts.get('Captured', 0)
            evaded_macro = summary_counts.get('Evaded', 0)
            fuel_macro = summary_counts.get('Fuel Depleted', 0)
            other_macro = summary_counts.get('Other', 0)
            print(
                "  - Captured={0} ({1:.1%}), Evaded={2} ({3:.1%}), Fuel Depleted={4} ({5:.1%})".format(
                    captured_macro,
                    captured_macro / total_macro,
                    evaded_macro,
                    evaded_macro / total_macro,
                    fuel_macro,
                    fuel_macro / total_macro,
                )
            )
            if other_macro:
                print(
                    "    · Other={0} ({1:.1%})".format(
                        other_macro,
                        other_macro / total_macro,
                    )
                )

            if evaded_breakdown:
                breakdown_total = max(sum(evaded_breakdown.values()), 1)
                breakdown_str = ", ".join(
                    f"{label}={count} ({count / breakdown_total:.1%})"
                    for label, count in evaded_breakdown.items()
                )
                print(f"    · 회피 세부: {breakdown_str}")

        # 버퍼 시간 통계
        if self.buffer_time_stats:
            print(f"  - 평균 버퍼 시간: {np.mean(self.buffer_time_stats):.2f}초")

        if self.evader_delta_vs:
            window_values = list(self.delta_v_window) if self.delta_v_window else self.evader_delta_vs
            print(f"  - 최근 평균 ΔV 사용량: {np.mean(window_values):.2f} m/s")
            print(f"  - 누적 ΔV 평균: {np.mean(self.evader_delta_vs):.2f} m/s")

        if self.evader_rewards:
            recent_evader = np.mean(self.evader_rewards[-100:])
            recent_pursuer = np.mean(self.pursuer_rewards[-100:])
            print(f"  - 최근 회피자/추격자 보상 평균: {recent_evader:.4f}/{recent_pursuer:.4f}")
            print(f"  - Zero-Sum 검증: {recent_evader + recent_pursuer:.6f}")
        
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
            self.logger.record(
                "evasion/permanent_evasion_rate",
                self.permanent_evasions / max(1, self.episode_count),
            )
            
            # Zero-Sum 메트릭
            if self.evader_rewards:
                self.logger.record("zero_sum/evader_reward", np.mean(self.evader_rewards[-100:]))
                self.logger.record("zero_sum/pursuer_reward", np.mean(self.pursuer_rewards[-100:]))
    
    def plot_interim_results(self):
        """중간 결과 플롯 생성"""
        if self.episode_count == 0:
            return
        
        # 플롯 데이터 준비
        outcome_counts = {
            'captured': self.captures,
            'permanent_evasion': self.permanent_evasions,
            'fuel_depleted': self.fuel_depleted,
            'max_steps_reached': self.max_steps,
        }
        summary_counts, evaded_breakdown = aggregate_outcome_counts(
            outcome_counts,
            include_breakdown=True,
        )

        try:
            plot_training_progress(
                success_rates=self.success_rates,
                outcome_counts=outcome_counts,
                evader_rewards=self.evader_rewards,
                pursuer_rewards=self.pursuer_rewards,
                buffer_times=[_json_ready(bt) for bt in self.buffer_time_stats],
                episode_count=self.episode_count,
                save_dir=self.log_dir,
                macro_counts=summary_counts,
                episode_rewards=self.episode_rewards,
                evaded_breakdown=evaded_breakdown,
                reward_breakdowns=self.reward_breakdowns,
            )

            progress_csv = os.path.join(self.log_dir, 'progress.csv')
            plot_sac_training_metrics(
                progress_path=progress_csv,
                save_dir=self.log_dir,
            )
        except Exception as e:
            if self.verbose > 0:
                print(f"플롯 생성 중 오류: {e}")

        try:
            plot_delta_v_per_episode([float(x) for x in self.evader_delta_vs], self.log_dir)
        except Exception as e:
            if self.verbose > 0:
                print(f"Delta-V 플롯 생성 중 오류: {e}")

        if self._resume_enabled:
            self._write_resume_state()

    def on_training_end(self):
        """학습 종료 시 최종 결과 저장"""
        if self.verbose > 0:
            print("\n=== 학습 종료 - 최종 통계 ===")
            print(f"총 에피소드: {self.episode_count}")
            print(f"최종 성공률: {self.success_rates[-1]:.2%}" if self.success_rates else "N/A")
            print(
                "포획: {0}, 회피: {1} (영구={2}, 최대스텝={3}), 연료 소진: {4}".format(
                    self.captures,
                    self.evasions,
                    self.permanent_evasions,
                    self.max_steps,
                    self.fuel_depleted,
                )
            )
        
        # 최종 플롯 생성
        self.plot_interim_results()
        
        # 학습 통계 저장
        stats = {
            "total_episodes": self.episode_count,
            "final_success_rate": self.success_rates[-1] if self.success_rates else 0,
            "captures": self.captures,
            "evasions": self.evasions,
            "permanent_evasions": self.permanent_evasions,
            "fuel_depleted": self.fuel_depleted,
            "max_steps": self.max_steps,
            "episodes_info": self.episodes_info[-100:],  # 마지막 100개만 저장
            "evader_delta_v": self.evader_delta_vs,
        }

        if self.reward_breakdowns:
            stats["reward_breakdowns"] = self.reward_breakdowns[-100:]

        _, evaded_breakdown = aggregate_outcome_counts(
            {
                'permanent_evasion': self.permanent_evasions,
                'max_steps_reached': self.max_steps,
            },
            include_breakdown=True,
        )
        if evaded_breakdown:
            stats["evaded_breakdown"] = evaded_breakdown
        
        stats_path = os.path.join(self.log_dir, "evasion_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(_json_ready(stats), f, indent=2)

        if self.verbose > 0:
            print(f"학습 통계 저장: {stats_path}")

        if self._resume_enabled:
            self._write_resume_state()

    def _restore_from_resume(self, data: Dict[str, Any]) -> None:
        """저장된 상태로부터 내부 지표를 복원"""
        try:
            self.episode_count = int(data.get("episode_count", self.episode_count) or 0)
        except (TypeError, ValueError):
            self.episode_count = self.episode_count

        self.success_rates = [float(x) for x in data.get("success_rates", self.success_rates)]
        self.evader_rewards = [float(x) for x in data.get("evader_rewards", self.evader_rewards)]
        self.pursuer_rewards = [float(x) for x in data.get("pursuer_rewards", self.pursuer_rewards)]
        self.episode_rewards = [float(x) for x in data.get("episode_rewards", self.episode_rewards)]
        self.buffer_time_stats = [float(x) for x in data.get("buffer_times", self.buffer_time_stats)]
        self.evader_delta_vs = [float(x) for x in data.get("evader_delta_vs", self.evader_delta_vs)]

        self.episodes_info = list(data.get("episodes_info", self.episodes_info))
        self.outcomes = [info.get('outcome', 'unknown') for info in self.episodes_info]

        stored_breakdowns = data.get("reward_breakdowns")
        self.reward_breakdowns = []
        if isinstance(stored_breakdowns, list):
            for entry in stored_breakdowns:
                if isinstance(entry, dict):
                    normalized: Dict[str, float] = {}
                    for key, value in entry.items():
                        try:
                            normalized[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
                    self.reward_breakdowns.append(normalized)
                else:
                    self.reward_breakdowns.append({})
        
        n_eps = len(self.episode_rewards)
        if len(self.reward_breakdowns) < n_eps:
            self.reward_breakdowns.extend({} for _ in range(n_eps - len(self.reward_breakdowns)))
        elif len(self.reward_breakdowns) > n_eps and n_eps > 0:
            self.reward_breakdowns = self.reward_breakdowns[-n_eps:]

        self.captures = int(data.get("captures", self.captures) or self.captures)
        self.evasions = int(data.get("evasions", self.evasions) or self.evasions)

        base_permanent = int(
            data.get("permanent_evasions", self.permanent_evasions) or self.permanent_evasions
        )
        legacy_conditional = int(data.get("conditional_evasions", 0) or 0)
        legacy_temporary = int(data.get("temporary_evasions", 0) or 0)
        self.permanent_evasions = base_permanent + legacy_conditional + legacy_temporary

        self.fuel_depleted = int(data.get("fuel_depleted", self.fuel_depleted) or self.fuel_depleted)
        self.max_steps = int(data.get("max_steps", self.max_steps) or self.max_steps)

        outcome_counts = data.get("outcome_counts")
        if isinstance(outcome_counts, dict):
            self.captures = int(outcome_counts.get("captured", self.captures))
            permanent_from_counts = int(
                outcome_counts.get("permanent_evasion", self.permanent_evasions)
            )
            legacy_conditional = int(outcome_counts.get("conditional_evasion", 0) or 0)
            legacy_temporary = int(outcome_counts.get("temporary_evasion", 0) or 0)
            self.permanent_evasions = permanent_from_counts + legacy_conditional + legacy_temporary
            self.fuel_depleted = int(outcome_counts.get("fuel_depleted", self.fuel_depleted))
            self.max_steps = int(outcome_counts.get("max_steps", self.max_steps))

        self.evasions = max(self.evasions, self.permanent_evasions)

        # 최근 성공/ΔV 기록 복원
        recent_successes = data.get("recent_successes")
        if not recent_successes and self.episodes_info:
            recent_successes = [1 if info.get("success") else 0 for info in self.episodes_info][-self.success_window.maxlen:]

        self.success_window = deque(maxlen=self.success_window.maxlen)
        if recent_successes:
            for flag in recent_successes[-self.success_window.maxlen:]:
                try:
                    self.success_window.append(int(flag))
                except (TypeError, ValueError):
                    continue
        elif self.success_rates:
            default_flag = 1 if self.success_rates[-1] >= 0.5 else 0
            self.success_window.extend([default_flag] * min(len(self.success_rates), self.success_window.maxlen))

        recent_delta_v = data.get("recent_delta_v")
        if not recent_delta_v and self.evader_delta_vs:
            recent_delta_v = self.evader_delta_vs[-self.delta_v_window.maxlen:]

        self.delta_v_window = deque(maxlen=self.delta_v_window.maxlen)
        if recent_delta_v:
            for value in recent_delta_v[-self.delta_v_window.maxlen:]:
                try:
                    self.delta_v_window.append(float(value))
                except (TypeError, ValueError):
                    continue

    def _write_resume_state(self) -> None:
        """현재 상태를 JSON으로 저장하여 이어학습 시 활용"""
        resume_data = {
            "episode_count": self.episode_count,
            "success_rates": self.success_rates,
            "recent_successes": list(self.success_window),
            "captures": self.captures,
            "evasions": self.evasions,
            "permanent_evasions": self.permanent_evasions,
            "fuel_depleted": self.fuel_depleted,
            "max_steps": self.max_steps,
            "evader_rewards": self.evader_rewards,
            "pursuer_rewards": self.pursuer_rewards,
            "buffer_times": self.buffer_time_stats,
            "episode_rewards": self.episode_rewards[-self.success_window.maxlen:] if self.episode_rewards else [],
            "evader_delta_vs": self.evader_delta_vs,
            "recent_delta_v": list(self.delta_v_window),
            "episodes_info": self.episodes_info[-100:],
            "reward_breakdowns": self.reward_breakdowns[-100:] if self.reward_breakdowns else [],
            "outcome_counts": {
                "captured": self.captures,
                "permanent_evasion": self.permanent_evasions,
                "fuel_depleted": self.fuel_depleted,
                "max_steps": self.max_steps,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        resume_path = os.path.join(self.log_dir, "resume_state.json")
        try:
            with open(resume_path, 'w') as f:
                json.dump(_json_ready(resume_data), f, indent=2)
        except Exception as exc:
            if self.verbose > 0:
                print(f"resume_state 저장 실패: {exc}")


class PerformanceCallback(BaseCallback):
    """성능 추적 및 시각화 콜백"""
    
    def __init__(self, log_dir: str, plot_freq: int = 100, verbose: int = 0,
                 resume: bool = False, resume_data: Optional[Dict[str, Any]] = None):
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
        self.buffer_times = []
        self.episode_count = 0
        
        # 윈도우 사이즈
        self.window_size = 100
        self.success_window = deque(maxlen=self.window_size)

        # 보상 분해 데이터 (선택적으로 외부 공급자 사용)
        self.reward_breakdowns: Optional[List[Dict[str, float]]] = None
        self._reward_breakdown_provider: Optional[Callable[[], Optional[List[Dict[str, float]]]]] = None

        if resume and resume_data:
            self._restore_from_resume(resume_data)

    def set_reward_breakdown_source(self, provider: Callable[[], Optional[List[Dict[str, float]]]]) -> None:
        """EvasionTrackingCallback 등에서 보상 분해 리스트를 공급받기 위한 설정."""
        self._reward_breakdown_provider = provider

    def _get_reward_breakdowns(self) -> Optional[List[Dict[str, float]]]:
        if self._reward_breakdown_provider is not None:
            try:
                data = self._reward_breakdown_provider()
                if data is not None:
                    self.reward_breakdowns = data
            except Exception:
                pass
        return self.reward_breakdowns

    def _summary_outcome_counts(self) -> Dict[str, int]:
        if not self.outcome_counter:
            return {}
        raw = {str(key): int(value) for key, value in self.outcome_counter.items()}
        return aggregate_outcome_counts(raw)

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
            episode_reward = episode_data.get(
                "r",
                final_info.get(
                    "evader_total_reward",
                    final_info.get("evader_reward", 0.0),
                ),
            )
            episode_reward = float(episode_reward)
            episode_length = int(episode_data.get("l", final_info.get("episode_length", 0)))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            evader_reward = float(final_info.get("evader_total_reward", final_info.get("evader_reward", episode_reward)))
            pursuer_reward = float(final_info.get("pursuer_total_reward", final_info.get("pursuer_reward", -evader_reward)))
            self.evader_rewards.append(evader_reward)
            self.pursuer_rewards.append(pursuer_reward)
            self.buffer_times.append(final_info.get("buffer_time", 0.0))

            outcome = str(final_info.get("outcome", "unknown")).lower()
            if outcome in ("conditional_evasion", "temporary_evasion"):
                outcome = "permanent_evasion"
            self.outcome_counter[outcome] += 1

            success = outcome in [
                'permanent_evasion',
                'max_steps_reached',
            ]
            self.success_window.append(1 if success else 0)
            success_rate = float(np.mean(self.success_window)) if self.success_window else 0.0
            self.success_rates.append(success_rate)

            if self.logger is not None:
                recent_window = min(len(self.episode_rewards), self.window_size)
                reward_ma = float(np.mean(self.episode_rewards[-recent_window:])) if recent_window else float(episode_reward)
                self.logger.record("performance/episode_reward", float(episode_reward))
                self.logger.record("performance/episode_reward_ma", reward_ma)
                self.logger.record("performance/episode_length", float(episode_length))
                self.logger.record("performance/success_rate_ma", success_rate)
                self.logger.record("performance/evader_total_reward", float(evader_reward))
                self.logger.record("performance/pursuer_total_reward", float(pursuer_reward))

            if self.verbose > 0 and self.episode_count % 10 == 0:
                print(f"\n에피소드 {self.episode_count}:")
                print(f"  보상: {episode_reward:.2f}")
                print(f"  성공률(최근 {len(self.success_window)}): {success_rate:.1%}")
                print(f"  타임스텝: {self.num_timesteps}")
                summary_counts = self._summary_outcome_counts()
                if summary_counts:
                    total_macro = max(sum(summary_counts.values()), 1)
                    captured_macro = summary_counts.get('Captured', 0)
                    evaded_macro = summary_counts.get('Evaded', 0)
                    fuel_macro = summary_counts.get('Fuel Depleted', 0)
                    other_macro = summary_counts.get('Other', 0)
                    print(
                        "  누적 결과: Captured={0} ({1:.1%}), Evaded={2} ({3:.1%}), Fuel Depleted={4} ({5:.1%})".format(
                            captured_macro,
                            captured_macro / total_macro,
                            evaded_macro,
                            evaded_macro / total_macro,
                            fuel_macro,
                            fuel_macro / total_macro,
                        )
                    )
                    if other_macro:
                        print(
                            "    · Other={0} ({1:.1%})".format(
                                other_macro,
                                other_macro / total_macro,
                            )
                        )

            if self.episode_count % self.plot_freq == 0:
                self._save_plots()

        return True

    def _save_plots(self):
        """플롯 저장"""
        try:
            summary_counts, evaded_breakdown = aggregate_outcome_counts(
                {
                    key: int(value)
                    for key, value in self.outcome_counter.items()
                },
                include_breakdown=True,
            )

            plot_training_progress(
                success_rates=self.success_rates,
                outcome_counts={
                    'captured': self.outcome_counter.get('captured', 0),
                    'permanent_evasion': sum(
                        self.outcome_counter.get(key, 0)
                        for key in (
                            'permanent_evasion',
                            'conditional_evasion',
                            'temporary_evasion',
                            'evaded',
                        )
                    ),
                    'fuel_depleted': self.outcome_counter.get('fuel_depleted', 0),
                    'max_steps_reached': self.outcome_counter.get('max_steps_reached', 0),
                },
                evader_rewards=self.evader_rewards,
                pursuer_rewards=self.pursuer_rewards,
                buffer_times=self.buffer_times,
                episode_count=self.episode_count,
                save_dir=self.plot_dir,
                macro_counts=summary_counts,
                episode_rewards=self.episode_rewards,
                evaded_breakdown=evaded_breakdown,
                reward_breakdowns=self._get_reward_breakdowns(),
            )

            progress_csv = os.path.join(self.log_dir, 'progress.csv')
            plot_sac_training_metrics(
                progress_path=progress_csv,
                save_dir=self.plot_dir,
            )
            
            if self.verbose > 0:
                print(f"플롯 저장 완료: {self.plot_dir}")
                
        except Exception as e:
            print(f"플롯 저장 중 오류 발생: {e}")
    
    def save_final_stats(self):
        """최종 통계 저장"""
        stats = {
            "episodes_completed": int(self.episode_count),
            "final_success_rate": float(self.success_rates[-1]) if self.success_rates else 0.0,
            "average_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
        }

        summary_counts, evaded_breakdown = aggregate_outcome_counts(
            {
                key: int(value)
                for key, value in self.outcome_counter.items()
            },
            include_breakdown=True,
        )
        if summary_counts:
            stats["macro_outcomes"] = {key: int(value) for key, value in summary_counts.items()}
        if evaded_breakdown:
            stats["evaded_breakdown"] = {key: int(value) for key, value in evaded_breakdown.items()}
        breakdowns = self._get_reward_breakdowns()
        if breakdowns:
            stats["reward_breakdowns"] = breakdowns[-100:]

        # JSON으로 저장
        import json
        stats_path = f"{self.log_dir}/training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        # 최종 플롯 저장
        self._save_plots()
        
        if self.verbose > 0:
            print(f"최종 통계 저장: {stats_path}")

    def _restore_from_resume(self, data: Dict[str, Any]) -> None:
        """이전 학습 세션의 기록을 복원"""
        self.success_rates = [float(x) for x in data.get("success_rates", self.success_rates)]
        try:
            self.episode_count = int(data.get("episode_count", len(self.success_rates)) or self.episode_count)
        except (TypeError, ValueError):
            self.episode_count = len(self.success_rates)

        self.evader_rewards = [float(x) for x in data.get("evader_rewards", self.evader_rewards)]
        self.pursuer_rewards = [float(x) for x in data.get("pursuer_rewards", self.pursuer_rewards)]
        self.buffer_times = [float(x) for x in data.get("buffer_times", self.buffer_times)]

        outcome_counts = data.get("outcome_counts")
        if isinstance(outcome_counts, dict):
            normalized = Counter()
            for key, value in outcome_counts.items():
                try:
                    normalized[key] = int(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                self.outcome_counter = normalized

        stored_breakdowns = data.get("reward_breakdowns")
        if isinstance(stored_breakdowns, list):
            parsed: List[Dict[str, float]] = []
            for entry in stored_breakdowns:
                if isinstance(entry, dict):
                    normalized_entry: Dict[str, float] = {}
                    for key, value in entry.items():
                        try:
                            normalized_entry[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
                    parsed.append(normalized_entry)
                else:
                    parsed.append({})
            self.reward_breakdowns = parsed

        recent_successes = data.get("recent_successes")
        if not recent_successes and data.get("episodes_info"):
            recent_successes = [1 if info.get("success") else 0 for info in data["episodes_info"]][-self.window_size:]

        self.success_window = deque(maxlen=self.window_size)
        if recent_successes:
            for flag in recent_successes[-self.window_size:]:
                try:
                    self.success_window.append(int(flag))
                except (TypeError, ValueError):
                    continue
        elif self.success_rates:
            default_flag = 1 if self.success_rates[-1] >= 0.5 else 0
            self.success_window.extend([default_flag] * min(len(self.success_rates), self.window_size))


class ModelSaveCallback(BaseCallback):
    """모델 저장 콜백 - 벡터 환경 지원"""
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./models/",
        name_prefix: str = "model",
        verbose: int = 0,
        save_replay_buffer: bool = True,
        overwrite_latest: bool = False,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.overwrite_latest = overwrite_latest
        
        # 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            if self.overwrite_latest:
                save_file = os.path.join(self.save_path, f"{self.name_prefix}.zip")
            else:
                save_file = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_step_{self.n_calls}.zip",
                )

            self.model.save(save_file)

            if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
                buffer_path = save_file.replace(".zip", "_replay_buffer.pkl")
                self.model.save_replay_buffer(buffer_path)

            if self.verbose > 0:
                print(f"모델 체크포인트 저장됨: {save_file}")
        
        return True


class EvalCallbackWithReplayBuffer(EvalCallback):
    """평가 시 최적 모델과 리플레이 버퍼를 함께 저장."""

    def __init__(self, *args, save_replay_buffer: bool = True, replay_buffer_suffix: str = "_replay_buffer.pkl", **kwargs):
        super().__init__(*args, **kwargs)
        self.save_replay_buffer = save_replay_buffer
        self.replay_buffer_suffix = replay_buffer_suffix

    def _save_model(self) -> None:
        super()._save_model()

        if not self.save_replay_buffer:
            return

        if self.best_model_save_path is None:
            return

        if not hasattr(self.model, "save_replay_buffer"):
            return

        buffer_path = os.path.join(
            self.best_model_save_path,
            f"{self.name_prefix}{self.replay_buffer_suffix}",
        )

        try:
            self.model.save_replay_buffer(buffer_path)
        except Exception as exc:
            if self.verbose > 0:
                print(f"리플레이 버퍼 저장 실패: {buffer_path} ({exc})")
        else:
            if self.verbose > 0:
                print(f"리플레이 버퍼 저장됨: {buffer_path}")


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

        # MATLAB 호환을 위한 MAT 파일 저장 시도
        try:
            from scipy.io import savemat

            matlab_path = os.path.join(self.log_dir, 'final_episode_ephemeris.mat')
            savemat(matlab_path, {
                't': final_episode['t'],
                'evader': final_episode['evader'],
                'pursuer': final_episode['pursuer'],
                'outcome': final_episode['outcome'],
            })

            if self.verbose > 0:
                print(f"MATLAB 호환 데이터 저장: {matlab_path}")
        except Exception as exc:
            if self.verbose > 0:
                print(f"MAT 파일 저장 실패: {exc}")
        
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
                 episode_plot_freq=100, verbose=1,
                 resume_data: Optional[Dict[str, Any]] = None):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.save_dir = save_dir
        self.episode_plot_freq = episode_plot_freq
        os.makedirs(save_dir, exist_ok=True)
        
        # 벡터 환경 지원을 위한 환경별 데이터 저장
        self.env_data = {}  # 각 환경별 데이터
        self.all_episodes_data = []  # 모든 환경의 모든 에피소드
        self.total_episode_count = 0  # 모든 환경의 총 에피소드 수

        if resume_data:
            self._restore_from_resume(resume_data)
        
    def _init_env_tracking(self, env_idx):
        """환경별 추적 데이터 초기화"""
        if env_idx not in self.env_data:
            self.env_data[env_idx] = {
                'current_steps': [],
                'current_times': [],
                'current_distances': [],
                'current_evader_dv': [],
                'current_pursuer_dv': [],
                'current_evader_dv_vec': [],
                'current_pursuer_dv_vec': [],
                'current_positions': [],
                'current_velocities': [],
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
            env_data['current_times'].append(float(info.get('simulation_time_s', current_step)))
            env_data['current_distances'].append(relative_distance)
            
            # 게임 모드일 때만 delta-v 기록 (또는 모든 스텝 기록)
            env_data['current_evader_dv'].append(evader_dv_mag)
            env_data['current_pursuer_dv'].append(pursuer_dv_mag)

            evader_vec = info.get('evader_delta_v_vector')
            pursuer_vec = info.get('pursuer_delta_v_vector')

            if evader_vec is not None:
                evader_vec = np.asarray(evader_vec, dtype=np.float32).flatten()
            if pursuer_vec is not None:
                pursuer_vec = np.asarray(pursuer_vec, dtype=np.float32).flatten()

            env_data['current_evader_dv_vec'].append(
                evader_vec[:3].astype(np.float64).tolist() if isinstance(evader_vec, np.ndarray) and evader_vec.size >= 3
                else [0.0, 0.0, 0.0]
            )
            env_data['current_pursuer_dv_vec'].append(
                pursuer_vec[:3].astype(np.float64).tolist() if isinstance(pursuer_vec, np.ndarray) and pursuer_vec.size >= 3
                else [0.0, 0.0, 0.0]
            )
            
            # 상대 위치/속도 기록 (있다면)
            rel_state = info.get('relative_state')
            if rel_state is not None:
                rel_state = np.asarray(rel_state, dtype=np.float32).flatten()
            else:
                rel_state = None

            if rel_state is not None and rel_state.size >= 6:
                env_data['current_positions'].append(rel_state[:3].astype(np.float64).tolist())
                env_data['current_velocities'].append(rel_state[3:6].astype(np.float64).tolist())
            else:
                rel_pos = info.get('relative_position_m')
                rel_vel = info.get('relative_velocity_mps')
                if rel_pos is not None:
                    rel_pos = np.asarray(rel_pos, dtype=np.float32).flatten()
                if rel_vel is not None:
                    rel_vel = np.asarray(rel_vel, dtype=np.float32).flatten()
                env_data['current_positions'].append(
                    rel_pos[:3].astype(np.float64).tolist() if isinstance(rel_pos, np.ndarray) and rel_pos.size >= 3
                    else [float('nan')]*3
                )
                env_data['current_velocities'].append(
                    rel_vel[:3].astype(np.float64).tolist() if isinstance(rel_vel, np.ndarray) and rel_vel.size >= 3
                    else [float('nan')]*3
                )
            
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
                    'times': env_data['current_times'].copy(),
                    'outcome': info['outcome'],
                    'final_distance': relative_distance,
                    'total_evader_dv': total_evader_dv,  # 게임 모드에서의 총 delta-v
                    'total_pursuer_dv': total_pursuer_dv,  # 게임 모드에서의 총 delta-v
                    'distances': env_data['current_distances'].copy(),
                    'evader_dvs': env_data['current_evader_dv'].copy(),
                    'pursuer_dvs': env_data['current_pursuer_dv'].copy(),
                    'evader_dv_vectors': [vec[:] for vec in env_data['current_evader_dv_vec']],
                    'pursuer_dv_vectors': [vec[:] for vec in env_data['current_pursuer_dv_vec']],
                    'orbit_modes': env_data['orbit_modes'].copy(),  # 모드 정보 추가
                    'positions': [pos[:] for pos in env_data['current_positions']],
                    'velocities': [vel[:] for vel in env_data['current_velocities']],
                    'game_mode_steps': len(game_mode_indices),  # 게임 모드 스텝 수
                    'total_steps': len(env_data['current_steps'])  # 전체 스텝 수
                }
                
                self.all_episodes_data.append(episode_data)
                
                # 100번째 전체 에피소드마다 해당 에피소드의 상세 플롯 생성
                if self.total_episode_count % self.episode_plot_freq == 0:
                    self._plot_single_episode(episode_data)
                
                # 현재 환경의 데이터 초기화
                env_data['current_steps'] = []
                env_data['current_times'] = []
                env_data['current_distances'] = []
                env_data['current_evader_dv'] = []
                env_data['current_pursuer_dv'] = []
                env_data['current_evader_dv_vec'] = []
                env_data['current_pursuer_dv_vec'] = []
                env_data['current_positions'] = []
                env_data['current_velocities'] = []
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
        filename = f'episode_env_{env_id}_details.png'
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()
        
        if self.verbose:
            print(f"\n[Episode {episode_num} (Env {env_id})] 상세 플롯 저장")
            print(f"  - 결과: {episode_data['outcome']}")
            print(f"  - 최종 거리: {episode_data['final_distance']:.1f} m")
            print(f"  - 게임 모드 스텝: {episode_data.get('game_mode_steps', 'N/A')} / {episode_data.get('total_steps', 'N/A')}")
            print(f"  - 총 Delta-V - 회피자: {episode_data['total_evader_dv']:.1f} m/s, 추격자: {episode_data['total_pursuer_dv']:.1f} m/s")
    
    def _restore_from_resume(self, data: Dict[str, Any]) -> None:
        """기존 분석 데이터를 복원"""
        env_data = data.get('env_data')
        if isinstance(env_data, dict):
            restored_env = {}
            for key, value in env_data.items():
                try:
                    env_id = int(key)
                except (TypeError, ValueError):
                    env_id = key
                restored_env[env_id] = value
            self.env_data = restored_env

        episodes = data.get('all_episodes_data')
        if isinstance(episodes, list):
            self.all_episodes_data = episodes

        try:
            self.total_episode_count = int(data.get('total_episode_count', self.total_episode_count) or self.total_episode_count)
        except (TypeError, ValueError):
            pass

        try:
            self.n_calls = int(data.get('total_steps', self.n_calls) or self.n_calls)
        except (TypeError, ValueError):
            pass

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

        self._save_data()
    
    def _plot_episode_summary(self):
        """모든 환경의 에피소드별 요약 통계"""
        episodes = [ep['episode_num'] for ep in self.all_episodes_data]
        env_ids = [ep['env_id'] for ep in self.all_episodes_data]
        outcomes = [ep['outcome'] for ep in self.all_episodes_data]
        final_distances = [ep['final_distance'] for ep in self.all_episodes_data]
        total_evader_dvs = [ep['total_evader_dv'] for ep in self.all_episodes_data]
        total_pursuer_dvs = [ep['total_pursuer_dv'] for ep in self.all_episodes_data]

        def _auto_ylim(ax, values, *, log_scale=False):
            finite_vals = [float(v) for v in values if np.isfinite(v)]
            if not finite_vals:
                return

            vmin = min(finite_vals)
            vmax = max(finite_vals)

            if log_scale:
                vmin = max(vmin, 1e-6)
                if vmin == vmax:
                    vmin *= 0.9
                    vmax *= 1.1
                lower = max(vmin / 1.2, 1e-6)
                upper = max(lower * 1.001, vmax * 1.2)
                ax.set_ylim(lower, upper)
            else:
                if vmin == vmax:
                    delta = abs(vmin) * 0.15 or 1.0
                else:
                    delta = (vmax - vmin) * 0.1
                    if delta == 0:
                        delta = max(abs(vmax), abs(vmin), 1.0) * 0.1
                ax.set_ylim(vmin - delta, vmax + delta)

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
        _auto_ylim(plt.gca(), final_distances, log_scale=True)
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
        _auto_ylim(plt.gca(), total_evader_dvs)

        # 3. 추격자 연료 사용량
        plt.subplot(2, 2, 3)
        plt.scatter(episodes, total_pursuer_dvs, c=env_ids, cmap='tab20', alpha=0.6, s=30)
        plt.xlabel('Total Episode Number')
        plt.ylabel('Total Pursuer Delta-V (m/s)')
        plt.title('Pursuer Fuel Usage per Episode')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Environment ID')
        _auto_ylim(plt.gca(), total_pursuer_dvs)
        
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
        plt.savefig(f'{self.save_dir}/episode_summary.png', dpi=150)
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
            outcome_lower = outcome.lower()
            if 'captur' in outcome_lower:
                env_outcomes[env_id]['captured'] += 1
            elif any(tag in outcome_lower for tag in ('evaded', 'evasion', 'max_steps')):
                env_outcomes[env_id]['evaded'] += 1
            elif 'fuel' in outcome_lower:
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
        
        # 4. Delta-V 효율성 (성공률 vs 평균 Delta-V)
        plt.subplot(3, 2, 4)
        success_rates = []
        avg_dvs = []
        scatter_env_ids = []

        for env_id in env_ids:
            env_episodes = [ep for ep in self.all_episodes_data if ep['env_id'] == env_id]
            if env_episodes:
                successes = sum(
                    1
                    for ep in env_episodes
                    if any(tag in ep['outcome'].lower() for tag in ('evaded', 'evasion', 'max_steps'))
                )
                success_rate = successes / len(env_episodes) if env_episodes else 0
                success_rates.append(success_rate)
                avg_dvs.append(avg_evader_dv.get(env_id, 0))
                scatter_env_ids.append(env_id)

        if scatter_env_ids:
            color_values = np.array(scatter_env_ids, dtype=float)
            scatter = plt.scatter(avg_dvs, success_rates, c=color_values, cmap='viridis', s=100)
            plt.xlabel('Average Evader Delta-V (m/s)')
            plt.ylabel('Success Rate')
            plt.title('Fuel Efficiency vs Success Rate')
            cbar = plt.colorbar(scatter, label='Environment ID')
            cbar.set_ticks(color_values)
            cbar.set_ticklabels([str(env_id) for env_id in scatter_env_ids])
            plt.grid(True, alpha=0.3)

        # 5. 요약 통계
        plt.subplot(3, 2, 5)
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
        plt.savefig(f'{self.save_dir}/env_statistics.png', dpi=150)
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

        # JSON 포맷으로도 저장 (MATLAB 등 타 도구 호환)
        json_path = f'{self.save_dir}/analysis_data.json'

        try:
            with open(json_path, 'w') as f:
                json.dump(_json_ready(data), f, indent=2)
            if self.verbose:
                print(f"JSON 데이터 저장 완료: {json_path}")
        except Exception as exc:
            if self.verbose:
                print(f"JSON 저장 실패: {exc}")
