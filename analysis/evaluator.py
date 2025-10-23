"""
모델 평가 및 테스트 모듈
"""

import numpy as np
import os
import datetime
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import SAC
from analysis.visualization import (
    visualize_trajectory,
    plot_test_results,
    plot_orbital_elements_comparison,
    create_summary_dashboard,
    plot_eci_trajectories,
    plot_delta_v_components,
    aggregate_outcome_counts,
)
from analysis.metrics import calculate_performance_metrics, analyze_trajectory_quality
from orbital_mechanics.coordinate_transforms import lvlh_to_eci
from utils.matlab_templates import render_matlab_script


class ModelEvaluator:
    """모델 평가자 클래스"""
    
    def __init__(self, model, env, config=None):
        """
        평가자 초기화
        
        Args:
            model: 평가할 모델
            env: 테스트 환경
            config: 설정 객체
        """
        self.model = model
        self.env = env
        self.config = config
        
        # 결과 저장
        self.evaluation_results = []
        self.trajectories = []
        self.detailed_stats = {}
        
    def evaluate_multiple_scenarios(self, n_tests: int = 10, 
                                  deterministic: bool = True,
                                  save_results: bool = True) -> Dict[str, Any]:
        """
        다중 시나리오 평가
        
        Args:
            n_tests: 테스트 시나리오 수
            deterministic: 결정적 정책 사용 여부
            save_results: 결과 저장 여부
            
        Returns:
            평가 결과 딕셔너리
        """
        print(f"다중 시나리오 평가 시작... ({n_tests} 시나리오)")
        
        results = []
        trajectories = []
        zero_sum_metrics = {
            'evader_rewards': [],
            'pursuer_rewards': [],
            'evader_impulse_counts': [],
            'safety_scores': [],
            'buffer_times': []
        }
        
        outcome_types = {
            'captured': 0,
            'permanent_evasion': 0,
            'fuel_depleted': 0,
            'max_steps_reached': 0
        }
        
        # 결과 저장 디렉토리
        if save_results:
            results_dir = f"./test_results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(results_dir, exist_ok=True)
            self._write_matlab_evaluation_script(results_dir)
        else:
            results_dir = None
        
        success_count = 0
        
        for i in range(n_tests):
            print(f"테스트 {i+1}/{n_tests} 실행 중...")
            
            # 단일 시나리오 실행
            scenario_result = self.run_single_scenario(
                deterministic=deterministic,
                scenario_id=i+1,
                save_trajectory=save_results,
                save_dir=results_dir
            )
            
            results.append(scenario_result['metrics'])
            trajectories.append(scenario_result['trajectory'])
            
            # 성공 카운트
            if scenario_result['metrics']['success']:
                success_count += 1
            
            # 결과별 카운트
            outcome = scenario_result['info'].get('termination_type', 'unknown')
            normalized_outcome = str(outcome).lower()
            if normalized_outcome in ('conditional_evasion', 'temporary_evasion'):
                normalized_outcome = 'permanent_evasion'
            if normalized_outcome in outcome_types:
                outcome_types[normalized_outcome] += 1
            
            # Zero-Sum 메트릭 수집
            self._collect_zero_sum_metrics(scenario_result, zero_sum_metrics)
        
        # 종합 결과 계산
        comprehensive_results = self._calculate_comprehensive_results(
            results, zero_sum_metrics, outcome_types, success_count, n_tests
        )
        
        # 결과 저장 및 시각화
        if save_results:
            self._save_evaluation_results(
                comprehensive_results, results, trajectories, 
                zero_sum_metrics, outcome_types, results_dir
            )
        
        self.evaluation_results = results
        self.trajectories = trajectories
        self.detailed_stats = comprehensive_results
        
        return comprehensive_results
    
    def run_single_scenario(self, deterministic: bool = True,
                          scenario_id: Optional[int] = None,
                          save_trajectory: bool = False,
                          save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        단일 시나리오 실행
        
        Args:
            deterministic: 결정적 정책 사용 여부
            scenario_id: 시나리오 ID
            save_trajectory: 궤적 저장 여부
            save_dir: 저장 디렉토리
            
        Returns:
            시나리오 실행 결과
        """
        obs, _ = self.env.reset()
        done = False

        # 데이터 수집
        states = []
        actions_e = []
        actions_p = []
        command_actions_e = []
        times = []
        evader_eci = []
        pursuer_eci = []
        evader_impulse_steps = []
        evader_impulse_times = []
        evader_impulse_values = []
        pursuer_impulse_steps = []
        pursuer_impulse_times = []
        pursuer_impulse_values = []
        rewards = {'evader': [], 'pursuer': []}
        step_count = 0
        
        while not done:
            # 정규화된 액션 예측
            normalized_action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = self.env.step(normalized_action)
            done = terminated or truncated
            step_count += 1

            # 보상 기록
            evader_reward = info.get('evader_reward', reward)
            pursuer_reward = info.get('pursuer_reward', -reward)
            rewards['evader'].append(evader_reward)
            rewards['pursuer'].append(pursuer_reward)
            
            # 상태 및 액션 기록
            phys_state = self.env.state.copy()
            
            command_action_e = self.env._denormalize_action(normalized_action)
            evader_action_step = bool(info.get('evader_action_step', False))
            pursuer_action_step = bool(info.get('pursuer_action_step', False))

            evader_delta_v = np.array(
                info.get('evader_delta_v_vector', np.zeros(3)), dtype=np.float32
            )
            pursuer_delta_v = np.array(
                info.get('pursuer_delta_v_vector', np.zeros(3)), dtype=np.float32
            )
            
            states.append(phys_state.copy())
            command_actions_e.append(command_action_e.copy())
            actions_e.append(evader_delta_v.copy())
            actions_p.append(pursuer_delta_v.copy())

            step_index = step_count - 1
            current_time = float(self.env.t)

            if evader_action_step and np.linalg.norm(evader_delta_v) > 0.0:
                evader_impulse_steps.append(step_index)
                evader_impulse_times.append(current_time)
                evader_impulse_values.append(evader_delta_v.copy())

            if pursuer_action_step and np.linalg.norm(pursuer_delta_v) > 0.0:
                pursuer_impulse_steps.append(step_index)
                pursuer_impulse_times.append(current_time)
                pursuer_impulse_values.append(pursuer_delta_v.copy())

            r_e, v_e = self.env.evader_orbit.get_position_velocity(self.env.t)
            r_p, v_p = lvlh_to_eci(r_e, v_e, self.env.state)
            times.append(self.env.t)
            evader_eci.append(np.concatenate((r_e, v_e)))
            pursuer_eci.append(np.concatenate((r_p, v_p)))
        
        # 배열 변환
        states = np.array(states)
        actions_e = np.array(actions_e)
        actions_p = np.array(actions_p)
        command_actions_e = np.array(command_actions_e)
        times = np.array(times)
        evader_eci = np.array(evader_eci)
        pursuer_eci = np.array(pursuer_eci)
        
        # 결과 분석
        metrics = self.env.analyze_results(states, actions_e, actions_p)

        # 추가 메트릭 계산
        additional_metrics = calculate_performance_metrics(
            states, actions_e, actions_p, rewards, info
        )
        metrics.update(additional_metrics)

        termination_type = info.get('termination_type') or info.get('outcome')
        if termination_type:
            termination_str = str(termination_type)
            outcome_lower = termination_str.lower()
            normalized_outcome = outcome_lower
            if normalized_outcome in ('conditional_evasion', 'temporary_evasion'):
                normalized_outcome = 'permanent_evasion'

            metrics['outcome'] = normalized_outcome
            metrics['termination_type'] = normalized_outcome

            reached_max_steps = 'max_step' in outcome_lower
            if reached_max_steps:
                metrics['max_steps_reached'] = True

            evaded_keywords = ('evasion', 'evaded')
            is_evaded = any(keyword in normalized_outcome for keyword in evaded_keywords)
            if reached_max_steps or is_evaded:
                metrics['success'] = True

            metrics['success_category'] = 'evaded' if metrics.get('success') else 'failure'

        # 시뮬레이션 진행 정보
        elapsed_time_sec = float(times[-1]) if len(times) > 0 else step_count * getattr(self.env, "dt", 0.0)
        metrics['total_steps'] = step_count
        metrics['elapsed_time_seconds'] = elapsed_time_sec
        metrics['elapsed_time_minutes'] = elapsed_time_sec / 60.0 if elapsed_time_sec else 0.0

        # 궤적 품질 분석
        trajectory_quality = analyze_trajectory_quality(states, actions_e)
        metrics.update(trajectory_quality)

        # 초기 궤도 요소 저장 (평가 로그용)
        evader_elements = info.get('initial_evader_orbital_elements')
        if evader_elements:
            metrics['initial_evader_orbital_elements'] = dict(evader_elements)

        pursuer_elements = info.get('initial_pursuer_orbital_elements')
        if pursuer_elements:
            metrics['initial_pursuer_orbital_elements'] = dict(pursuer_elements)

        # 시나리오 결과 패키징
        scenario_result = {
            'metrics': metrics,
            'trajectory': (states, actions_e, actions_p),
            'ephemeris_eci': (times, evader_eci, pursuer_eci),
            'rewards': rewards,
            'info': info,
            'step_count': step_count,
            'commands': {
                'evader': command_actions_e,
            },
            'impulses': {
                'evader': {
                    'step_indices': evader_impulse_steps,
                    'times': evader_impulse_times,
                    'delta_v': evader_impulse_values,
                },
                'pursuer': {
                    'step_indices': pursuer_impulse_steps,
                    'times': pursuer_impulse_times,
                    'delta_v': pursuer_impulse_values,
                },
            }
        }

        if hasattr(self.env, "tvlqr_fallback_events"):
            events = getattr(self.env, "tvlqr_fallback_events", [])
            scenario_result['tvlqr_fallback_events'] = [
                {
                    "index": int(event.get("index", 0)),
                    "step": int(event.get("step", 0)),
                    "time": float(event.get("time", 0.0)),
                    "reason": str(event.get("reason", "")),
                    "error": str(event.get("error", "")),
                }
                for event in events
            ]
        else:
            scenario_result['tvlqr_fallback_events'] = []

        summary = None
        if hasattr(self.env, "get_tvlqr_fallback_summary"):
            try:
                summary = self.env.get_tvlqr_fallback_summary()
            except Exception:
                summary = None
        scenario_result['tvlqr_fallback_summary'] = summary or {}
        
        # 궤적 시각화 및 저장
        if save_trajectory and save_dir and scenario_id:
            self._save_scenario_trajectory(
                scenario_result, scenario_id, save_dir
            )
        
        return scenario_result
    
    def _collect_zero_sum_metrics(self, scenario_result: Dict, zero_sum_metrics: Dict):
        """Zero-Sum 메트릭 수집"""
        rewards = scenario_result['rewards']
        info = scenario_result['info']
        
        zero_sum_metrics['evader_rewards'].append(np.mean(rewards['evader']))
        zero_sum_metrics['pursuer_rewards'].append(np.mean(rewards['pursuer']))
        zero_sum_metrics['evader_impulse_counts'].append(info.get('evader_impulse_count', 0))
        
        if 'safety_score' in info:
            zero_sum_metrics['safety_scores'].append(info['safety_score'])
        if 'buffer_time' in info:
            zero_sum_metrics['buffer_times'].append(info['buffer_time'])
    
    def _calculate_comprehensive_results(self, results: List[Dict], 
                                       zero_sum_metrics: Dict,
                                       outcome_types: Dict,
                                       success_count: int,
                                       n_tests: int) -> Dict[str, Any]:
        """종합 결과 계산"""
        # 기본 통계
        success_rate = success_count / n_tests * 100 if n_tests else 0.0

        def _valid(values):
            filtered = []
            for value in values:
                if value is None:
                    continue
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(val):
                    filtered.append(val)
            return filtered

        # 최종 거리는 환경/메트릭 버전에 따라 키가 다를 수 있어 순차적으로 확인한다.
        distance_keys = (
            'final_distance_m',
            'final_distance',
            'final_relative_distance',
        )
        final_distances = []
        for key in distance_keys:
            final_distances = _valid([r.get(key) for r in results])
            if final_distances:
                break
        avg_distance = float(np.mean(final_distances)) if final_distances else 0.0

        evader_delta_vs = _valid([r.get('evader_total_delta_v_ms') for r in results])
        avg_evader_dv = float(np.mean(evader_delta_vs)) if evader_delta_vs else 0.0

        evader_rewards = _valid(zero_sum_metrics['evader_rewards'])
        pursuer_rewards = _valid(zero_sum_metrics['pursuer_rewards'])
        avg_evader_reward = float(np.mean(evader_rewards)) if evader_rewards else 0.0
        avg_pursuer_reward = float(np.mean(pursuer_rewards)) if pursuer_rewards else 0.0

        impulse_counts = _valid(zero_sum_metrics['evader_impulse_counts'])
        avg_impulse_count = float(np.mean(impulse_counts)) if impulse_counts else 0.0

        # 안전도 및 버퍼 시간 통계
        safety_scores = _valid(zero_sum_metrics['safety_scores'])
        avg_safety_score = float(np.mean(safety_scores)) if safety_scores else 0.0

        buffer_times = _valid(zero_sum_metrics['buffer_times'])
        avg_buffer_time = float(np.mean(buffer_times)) if buffer_times else 0.0

        # 변동성 통계
        distance_std = float(np.std(final_distances)) if len(final_distances) > 1 else 0.0
        dv_std = float(np.std(evader_delta_vs)) if len(evader_delta_vs) > 1 else 0.0
        reward_std = float(np.std(evader_rewards)) if len(evader_rewards) > 1 else 0.0

        comprehensive_results = {
            'summary': {
                'total_tests': n_tests,
                'success_count': success_count,
                'success_rate': success_rate,
                'avg_final_distance': avg_distance,
                'avg_evader_delta_v': avg_evader_dv,
                'avg_evader_reward': avg_evader_reward,
                'avg_pursuer_reward': avg_pursuer_reward,
                'zero_sum_verification': avg_evader_reward + avg_pursuer_reward,
                'avg_impulse_count': avg_impulse_count,
                'avg_safety_score': avg_safety_score,
                'avg_buffer_time': avg_buffer_time
            },
            'variability': {
                'distance_std': distance_std,
                'delta_v_std': dv_std,
                'reward_std': reward_std
            },
            'outcome_distribution': outcome_types,
            'zero_sum_metrics': zero_sum_metrics,
            'individual_results': results
        }

        macro_counts, evaded_breakdown = aggregate_outcome_counts(
            outcome_types,
            include_breakdown=True,
        )
        if macro_counts:
            comprehensive_results['macro_outcome_distribution'] = macro_counts
            if evaded_breakdown:
                comprehensive_results['evaded_breakdown'] = evaded_breakdown

            comprehensive_results['summary']['captured_macro'] = macro_counts.get('Captured', 0)
            comprehensive_results['summary']['evaded_macro'] = macro_counts.get('Evaded', 0)
            comprehensive_results['summary']['fuel_depleted_macro'] = macro_counts.get('Fuel Depleted', 0)

            # 세부 회피 통계 (최대 스텝 포함)
            max_steps_cases = evaded_breakdown.get('Max Steps', outcome_types.get('max_steps_reached', 0))
            comprehensive_results['summary']['max_steps_cases'] = max_steps_cases

        return comprehensive_results
    
    def _save_scenario_trajectory(self, scenario_result: Dict, 
                                scenario_id: int, save_dir: str):
        """시나리오 궤적 저장"""
        states, actions_e, actions_p = scenario_result['trajectory']
        metrics = scenario_result['metrics']
        info = scenario_result['info']
        
        # 3D 궤적 시각화
        title = f"Test {scenario_id}: {'Success' if metrics['success'] else 'Failure'}"
        if 'termination_type' in info:
            title += f" - {info['termination_type'].replace('_', ' ').title()}"
        
        visualize_trajectory(
            states, actions_e, actions_p,
            title=title,
            save_path=f"{save_dir}/test_{scenario_id}_trajectory",
            safety_info=info.get('safety_score'),
            buffer_time=info.get('buffer_time')
        )

        if 'ephemeris_eci' in scenario_result:
            t, e_eci, p_eci = scenario_result['ephemeris_eci']
            plot_eci_trajectories(
                t, p_eci, e_eci,
                save_path=f"{save_dir}/test_{scenario_id}_eci",
                title=f"Test {scenario_id} ECI Trajectory"
            )

        # Delta-V 기록 저장
        if 'ephemeris_eci' in scenario_result:
            times = np.asarray(scenario_result['ephemeris_eci'][0], dtype=float)
        else:
            times = np.arange(len(actions_e), dtype=float) * self.env.dt

        step_indices = np.arange(actions_e.shape[0], dtype=int)

        def _extract_impulse_data(
            actions: np.ndarray, time_series: np.ndarray, stored: Optional[Dict[str, Any]]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            mask = np.linalg.norm(actions, axis=1) > 1e-8
            if stored:
                stored_steps = stored.get('step_indices', [])
                stored_times = stored.get('times', [])
                stored_values = stored.get('delta_v', [])
                if stored_values:
                    values = np.asarray(stored_values, dtype=np.float32)
                    if values.ndim == 1:
                        values = values.reshape(-1, 3)
                else:
                    values = np.zeros((len(stored_steps), 3), dtype=np.float32)
                if values.size == 0:
                    values = values.reshape(0, 3)
                steps = np.asarray(stored_steps, dtype=int)
                impulse_times = np.asarray(stored_times, dtype=float)
                if steps.size and impulse_times.size == 0:
                    impulse_times = time_series[steps]
                return steps, impulse_times, values
            steps = np.nonzero(mask)[0]
            values = actions[mask]
            if values.size == 0:
                values = values.reshape(0, 3)
            impulse_times = time_series[mask]
            return steps, impulse_times, values

        impulses = scenario_result.get('impulses', {})
        evader_impulse_steps, evader_impulse_times, evader_impulses = _extract_impulse_data(
            actions_e,
            times,
            impulses.get('evader') if impulses else None,
        )
        pursuer_impulse_steps, pursuer_impulse_times, pursuer_impulses = _extract_impulse_data(
            actions_p,
            times,
            impulses.get('pursuer') if impulses else None,
        )

        delta_v_data = {
            'step_index': step_indices.tolist(),
            'time': times.tolist(),
            'evader_delta_v_full': actions_e.tolist(),
            'evader_delta_v_full_norm': np.linalg.norm(actions_e, axis=1).tolist(),
            'pursuer_delta_v_full': actions_p.tolist(),
            'pursuer_delta_v_full_norm': np.linalg.norm(actions_p, axis=1).tolist(),
            'evader_impulses': {
                'step_index': evader_impulse_steps.tolist(),
                'time': evader_impulse_times.tolist(),
                'delta_v': evader_impulses.tolist(),
                'delta_v_norm': np.linalg.norm(evader_impulses, axis=1).tolist() if evader_impulses.size else [],
            },
            'pursuer_impulses': {
                'step_index': pursuer_impulse_steps.tolist(),
                'time': pursuer_impulse_times.tolist(),
                'delta_v': pursuer_impulses.tolist(),
                'delta_v_norm': np.linalg.norm(pursuer_impulses, axis=1).tolist() if pursuer_impulses.size else [],
            },
        }

        with open(f"{save_dir}/test_{scenario_id}_delta_v.json", "w") as f:
            import json
            json.dump(delta_v_data, f, indent=2)

        # Delta-V 성분별 그래프 저장
        plot_delta_v_components(
            evader_impulses,
            pursuer_impulses,
            f"{save_dir}/test_{scenario_id}",
            evader_steps=evader_impulse_steps,
            pursuer_steps=pursuer_impulse_steps,
        )

        fallback_events = scenario_result.get('tvlqr_fallback_events') or []
        if fallback_events:
            fallback_csv_path = f"{save_dir}/test_{scenario_id}_tvlqr_fallback.csv"
            fieldnames = ['index', 'step', 'time', 'reason', 'error']
            with open(fallback_csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for event in fallback_events:
                    row = {key: event.get(key, "") for key in fieldnames}
                    writer.writerow(row)

        fallback_summary = scenario_result.get('tvlqr_fallback_summary') or {}
        if fallback_summary:
            import json
            summary_path = f"{save_dir}/test_{scenario_id}_tvlqr_fallback_summary.json"
            with open(summary_path, "w") as f:
                json.dump(fallback_summary, f, indent=2)

    def _save_evaluation_results(self, comprehensive_results: Dict,
                               results: List[Dict],
                               trajectories: List[Tuple],
                               zero_sum_metrics: Dict,
                               outcome_types: Dict,
                               save_dir: str):
        """평가 결과 저장"""
        # 1. 텍스트 요약 저장
        self._save_text_summary(comprehensive_results, save_dir)
        
        # 2. 상세 결과 저장
        self._save_detailed_results(results, save_dir)
        
        # 3. Zero-Sum 메트릭 저장
        self._save_zero_sum_metrics(zero_sum_metrics, save_dir)
        
        # 4. 시각화 생성
        macro_counts = comprehensive_results.get('macro_outcome_distribution')
        evaded_breakdown = comprehensive_results.get('evaded_breakdown')
        plot_test_results(
            results,
            zero_sum_metrics,
            outcome_types,
            save_dir,
            macro_counts=macro_counts,
            evaded_breakdown=evaded_breakdown,
        )
        
        # 5. 대시보드 생성
        training_stats = getattr(self, 'training_stats', {})
        create_summary_dashboard(training_stats, results, save_dir)
        
        print(f"평가 결과 저장 완료: {save_dir}")

    def _write_matlab_evaluation_script(self, results_dir: str) -> None:
        """평가 결과 폴더에 MATLAB 분석 스크립트를 생성."""
        resolved = Path(results_dir).resolve()
        batch_name = resolved.name
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        destination = resolved / "Analysis_evaluation.m"

        try:
            render_matlab_script(
                "analysis_evaluation_template.m",
                destination,
                {
                    "RUN_NAME": batch_name,
                    "GENERATED_AT": generated_at,
                },
            )
        except FileNotFoundError as exc:
            print(f"[경고] MATLAB 평가 분석 스크립트 생성 실패: {exc}")
    
    def _save_text_summary(self, results: Dict, save_dir: str):
        """텍스트 요약 저장"""
        with open(f"{save_dir}/summary.txt", "w") as f:
            summary = results['summary']
            
            f.write("=== 모델 평가 결과 요약 ===\n\n")
            f.write(f"총 테스트 수: {summary['total_tests']}\n")
            f.write(f"성공 횟수: {summary['success_count']}\n")
            f.write(f"성공률: {summary['success_rate']:.1f}%\n\n")

            macro = results.get('macro_outcome_distribution', {})
            evaded_breakdown = results.get('evaded_breakdown', {})
            if macro:
                captured_macro = macro.get('Captured', 0)
                evaded_macro = macro.get('Evaded', 0)
                fuel_macro = macro.get('Fuel Depleted', 0)
                other_macro = macro.get('Other', 0)

                f.write("=== 대분류 결과 ===\n")
                f.write(f"Captured: {captured_macro}\n")
                f.write(f"Evaded: {evaded_macro}\n")
                f.write(f"Fuel Depleted: {fuel_macro}\n")
                if other_macro:
                    f.write(f"Other: {other_macro}\n")
                if evaded_breakdown:
                    for label, count in evaded_breakdown.items():
                        f.write(f" └ {label}: {count}\n")
                f.write("\n")

            f.write("=== 성능 지표 ===\n")
            f.write(f"평균 최종 거리: {summary['avg_final_distance']:.2f} m\n")
            f.write(f"평균 회피자 delta-v: {summary['avg_evader_delta_v']:.2f} m/s\n")
            f.write(f"평균 회피자 보상: {summary['avg_evader_reward']:.4f}\n")
            f.write(f"평균 추격자 보상: {summary['avg_pursuer_reward']:.4f}\n")
            f.write(f"Zero-Sum 검증: {summary['zero_sum_verification']:.6f}\n")
            f.write(f"평균 궤도 변경 횟수: {summary['avg_impulse_count']:.1f}\n")
            f.write(f"평균 안전도 점수: {summary['avg_safety_score']:.4f}\n")
            f.write(f"평균 버퍼 시간: {summary['avg_buffer_time']:.2f} 초\n\n")
            
            f.write("=== 변동성 지표 ===\n")
            variability = results['variability']
            f.write(f"거리 표준편차: {variability['distance_std']:.2f} m\n")
            f.write(f"Delta-V 표준편차: {variability['delta_v_std']:.2f} m/s\n")
            f.write(f"보상 표준편차: {variability['reward_std']:.4f}\n\n")
            
            f.write("=== 종료 조건 분포 ===\n")
            for outcome, count in results['outcome_distribution'].items():
                if count > 0:
                    percentage = count / summary['total_tests'] * 100
                    f.write(f"{outcome.replace('_', ' ').title()}: {count}회 ({percentage:.1f}%)\n")
    
    def _save_detailed_results(self, results: List[Dict], save_dir: str):
        """상세 결과 저장"""
        with open(f"{save_dir}/detailed_results.txt", "w") as f:
            for i, result in enumerate(results):
                f.write(f"\n=== 테스트 {i+1} ===\n")
                seen = set()
                for key, value in result.items():
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(f"{key}: {value}\n")
    
    def _save_zero_sum_metrics(self, zero_sum_metrics: Dict, save_dir: str):
        """Zero-Sum 메트릭 CSV 저장"""
        with open(f"{save_dir}/zero_sum_metrics.csv", "w") as f:
            f.write("episode,evader_reward,pursuer_reward,impulse_count,safety_score,buffer_time\n")
            
            n_episodes = len(zero_sum_metrics['evader_rewards'])
            for i in range(n_episodes):
                evader_reward = zero_sum_metrics['evader_rewards'][i]
                pursuer_reward = zero_sum_metrics['pursuer_rewards'][i]
                impulse_count = zero_sum_metrics['evader_impulse_counts'][i]
                
                safety_score = (zero_sum_metrics['safety_scores'][i] 
                                  if i < len(zero_sum_metrics['safety_scores']) else 0)
                buffer_time = (zero_sum_metrics['buffer_times'][i] 
                             if i < len(zero_sum_metrics['buffer_times']) else 0)
                
                f.write(f"{i+1},{evader_reward:.4f},{pursuer_reward:.4f},"
                       f"{impulse_count},{safety_score:.4f},{buffer_time:.2f}\n")
    
    def run_demonstration(self, save_dir: Optional[str] = None,
                         deterministic: bool = True) -> Dict[str, Any]:
        """
        데모 실행
        
        Args:
            save_dir: 저장 디렉토리
            deterministic: 결정적 정책 사용 여부
            
        Returns:
            데모 결과
        """
        if save_dir is None:
            save_dir = f"./demo_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(save_dir, exist_ok=True)
        
        print("지능형 추격자를 사용한 데모 실행 중...")
        
        demo_result = self.run_single_scenario(
            deterministic=deterministic,
            scenario_id=1,
            save_trajectory=True,
            save_dir=save_dir
        )
        
        # 결과 저장
        metrics = demo_result['metrics']
        info = demo_result['info']
        rewards = demo_result['rewards']
        
        with open(f"{save_dir}/demo_result.txt", "w") as f:
            f.write("지능형 추격자 vs. 학습된 회피자 데모 결과\n\n")
            
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\n회피자 평균 보상: {np.mean(rewards['evader']):.4f}\n")
            f.write(f"추격자 평균 보상: {np.mean(rewards['pursuer']):.4f}\n")
            f.write(f"Zero-Sum 검증: {np.mean(rewards['evader']) + np.mean(rewards['pursuer']):.6f}\n")
            f.write(f"종료 조건: {info.get('termination_type', 'unknown')}\n")
            f.write(f"버퍼 시간: {info.get('buffer_time', 0):.2f} 초\n")
            f.write(f"안전도 점수: {info.get('safety_score', 0):.4f}\n")
        
        print(f"데모 결과 저장 완료: {save_dir}")
        return demo_result
    
    def compare_models(self, models: List[SAC], model_names: List[str],
                      n_tests_per_model: int = 5) -> Dict[str, Any]:
        """
        다중 모델 비교 평가
        
        Args:
            models: 비교할 모델들
            model_names: 모델 이름들
            n_tests_per_model: 모델당 테스트 수
            
        Returns:
            비교 결과
        """
        print(f"다중 모델 비교 평가 시작... ({len(models)} 모델)")
        
        comparison_results = {}
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            print(f"\n모델 {i+1}/{len(models)} 평가 중: {name}")
            
            # 현재 모델로 교체
            original_model = self.model
            self.model = model
            
            # 평가 실행
            results = self.evaluate_multiple_scenarios(
                n_tests=n_tests_per_model,
                save_results=False
            )
            
            comparison_results[name] = results['summary']
            
            # 원래 모델 복원
            self.model = original_model
        
        # 비교 결과 저장
        comparison_dir = f"./model_comparison_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(comparison_dir, exist_ok=True)
        
        self._save_model_comparison(comparison_results, comparison_dir)
        
        return comparison_results
    
    def _save_model_comparison(self, comparison_results: Dict, save_dir: str):
        """모델 비교 결과 저장"""
        with open(f"{save_dir}/model_comparison.txt", "w") as f:
            f.write("=== 모델 비교 결과 ===\n\n")
            
            # 성능 지표별 비교
            metrics = ['success_rate', 'avg_final_distance', 'avg_evader_delta_v']
            
            for metric in metrics:
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                for model_name, results in comparison_results.items():
                    value = results.get(metric, 0)
                    f.write(f"  {model_name}: {value:.4f}\n")
        
        print(f"모델 비교 결과 저장 완료: {save_dir}")


def create_evaluator(model, env, config=None) -> ModelEvaluator:
    """평가자 생성 헬퍼 함수"""
    return ModelEvaluator(model, env, config)
