"""
시각화 및 그래프 생성 모듈 (데이터 저장 개선)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Tuple, Any
import os
import json
import csv

from utils.constants import PLOT_PARAMS, SAFETY_THRESHOLDS, R_EARTH
from scipy.io import savemat

def _json_ready(value):
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

def setup_matplotlib():
    """Matplotlib 설정"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = PLOT_PARAMS['figure_size_2d']
    plt.rcParams['figure.dpi'] = PLOT_PARAMS['dpi']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1


def save_data_to_csv(data: Dict[str, List], filepath: str):
    """데이터를 CSV 파일로 저장"""
    if not data:
        return
    
    # 키를 헤더로 사용
    headers = list(data.keys())
    
    # 모든 리스트의 길이 확인
    max_length = max(len(data[key]) for key in headers)
    
    # CSV 작성
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for i in range(max_length):
            row = []
            for header in headers:
                if i < len(data[header]):
                    row.append(data[header][i])
                else:
                    row.append('')  # 빈 값
            writer.writerow(row)


def plot_training_progress(success_rates: List[float],
                          outcome_counts: List[int],
                          evader_rewards: List[float],
                          pursuer_rewards: List[float],
                          nash_metrics: List[float],
                          buffer_times: List[float],
                          episode_count: int,
                          save_dir: str):
    """학습 진행 상황 시각화"""
    setup_matplotlib()
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 저장을 위한 딕셔너리
    training_data = {
        'episode': list(range(len(success_rates))),
        'success_rate': success_rates,
        'evader_reward': evader_rewards[-len(success_rates):] if len(evader_rewards) >= len(success_rates) else evader_rewards,
        'pursuer_reward': pursuer_rewards[-len(success_rates):] if len(pursuer_rewards) >= len(success_rates) else pursuer_rewards,
    }
    
    # 1. 성공률 그래프
    if success_rates:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        plt.plot(range(len(success_rates)), success_rates, 
                label='Success Rate (Moving Average)', color=PLOT_PARAMS['colors']['evader'])
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title(f'Training Success Rate (Episode {episode_count})')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/success_rate.png', dpi=PLOT_PARAMS['dpi'])
        plt.close()
        
        # 성공률 데이터 저장
        save_data_to_csv({'episode': list(range(len(success_rates))), 'success_rate': success_rates}, 
                        f'{save_dir}/success_rate_data.csv')
    
    # 2. 결과 분포 파이 차트
    if outcome_counts and sum(outcome_counts) > 0:
        labels = ['Captured', 'Permanent Evasion', 'Conditional Evasion', 'Fuel Depleted', 'Max Steps']
        # 0이 아닌 값들만 필터링
        filtered_counts = []
        filtered_labels = []
        for i, count in enumerate(outcome_counts):
            if count > 0:
                filtered_counts.append(count)
                filtered_labels.append(labels[i])
        
        if filtered_counts:
            plt.figure(figsize=(10, 10))
            colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_counts)))
            wedges, texts, autotexts = plt.pie(filtered_counts, labels=filtered_labels, 
                                               autopct='%1.1f%%', colors=colors,
                                               startangle=90)
            plt.title(f'Outcome Distribution (Episode {episode_count})')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/outcome_distribution.png', dpi=PLOT_PARAMS['dpi'])
            plt.close()
            
            # 결과 분포 데이터 저장
            outcome_data = {label: count for label, count in zip(filtered_labels, filtered_counts)}
            with open(f'{save_dir}/outcome_distribution.json', 'w') as f:
                json.dump(_json_ready(outcome_data), f, indent=2)
    
    # 3. Zero-Sum 게임 보상 그래프
    if len(evader_rewards) > 100:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        episodes = range(len(evader_rewards))
        
        # 원시 데이터 (투명하게)
        plt.plot(episodes, evader_rewards, label='Evader Reward', 
                color=PLOT_PARAMS['colors']['evader'], alpha=0.3, linewidth=0.5)
        plt.plot(episodes, pursuer_rewards, label='Pursuer Reward', 
                color=PLOT_PARAMS['colors']['pursuer'], alpha=0.3, linewidth=0.5)
        
        # 이동 평균 추가
        window_size = 100
        if len(evader_rewards) >= window_size:
            evader_ma = np.convolve(evader_rewards, np.ones(window_size)/window_size, mode='valid')
            pursuer_ma = np.convolve(pursuer_rewards, np.ones(window_size)/window_size, mode='valid')
            ma_episodes = range(window_size-1, len(evader_rewards))
            plt.plot(ma_episodes, evader_ma, label='Evader Reward (MA)', 
                    color='darkgreen', linewidth=2)
            plt.plot(ma_episodes, pursuer_ma, label='Pursuer Reward (MA)', 
                    color='darkred', linewidth=2)
        
        # Zero-Sum 검증
        reward_sum = [evader_rewards[i] + pursuer_rewards[i] for i in range(len(evader_rewards))]
        plt.plot(episodes, reward_sum, label='Reward Sum (Zero-Sum Verification)', 
                color='black', linestyle='--', alpha=0.7)
        
        plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Zero-Sum Game Reward Trend')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/zero_sum_rewards.png', dpi=PLOT_PARAMS['dpi'])
        plt.close()
        
        # 보상 데이터 저장
        rewards_data = {
            'episode': list(range(len(evader_rewards))),
            'evader_reward': evader_rewards,
            'pursuer_reward': pursuer_rewards,
            'reward_sum': reward_sum
        }
        save_data_to_csv(rewards_data, f'{save_dir}/rewards_data.csv')
    
    # 4. Nash Equilibrium 메트릭 그래프
    if nash_metrics:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        eval_episodes = list(range(0, episode_count, 10))  # 10 에피소드마다 평가 가정

        # eval_episodes와 nash_metrics 길이를 맞추기 위한 처리
        min_len = min(len(eval_episodes), len(nash_metrics))
        episodes_to_plot = eval_episodes[:min_len]
        nash_to_plot = nash_metrics[:min_len]

        plt.plot(episodes_to_plot, nash_to_plot, 'b-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Nash Equilibrium Metric')
        plt.title('Nash Equilibrium Convergence')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/nash_metric.png', dpi=PLOT_PARAMS['dpi'])
        plt.close()
        
        # Nash 메트릭 데이터 저장
        nash_data = {
            'episode': episodes_to_plot,
            'nash_metric': nash_to_plot
        }
        save_data_to_csv(nash_data, f'{save_dir}/nash_metrics.csv')
    
    # 5. 버퍼 시간 통계 그래프
    if buffer_times:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        plt.hist(buffer_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=np.mean(buffer_times), color='r', linestyle='--',
                   label=f'Average: {np.mean(buffer_times):.2f}s')
        plt.xlabel('Buffer Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Termination Condition Buffer Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/buffer_time_stats.png', dpi=PLOT_PARAMS['dpi'])
        plt.close()
    
    # 모든 학습 데이터를 하나의 JSON 파일로 저장
    all_training_data = {
        'episode_count': episode_count,
        'success_rates': success_rates,
        'outcome_counts': outcome_counts,
        'evader_rewards': evader_rewards[-1000:] if len(evader_rewards) > 1000 else evader_rewards,  # 최근 1000개만
        'pursuer_rewards': pursuer_rewards[-1000:] if len(pursuer_rewards) > 1000 else pursuer_rewards,
        'nash_metrics': nash_metrics,
        'buffer_times': buffer_times
    }
    
    with open(f'{save_dir}/training_progress.json', 'w') as f:
        json.dump(_json_ready(all_training_data), f, indent=2)


def plot_delta_v_per_episode(delta_v_values: List[float], save_dir: str, window: int = 50):
    """에피소드별 회피자 Delta-V 사용량을 선 그래프로 저장"""
    if not delta_v_values:
        return

    setup_matplotlib()
    os.makedirs(save_dir, exist_ok=True)

    episodes = np.arange(1, len(delta_v_values) + 1)

    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(episodes, delta_v_values, label='Per-Episode ΔV', color='tab:green', linewidth=1.2)

    if len(delta_v_values) >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        rolling = np.convolve(delta_v_values, kernel, mode='valid')
        plt.plot(episodes[window - 1:], rolling, label=f'{window}-Episode Moving Avg', color='tab:blue', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total ΔV (m/s)')
    plt.title('Evader Delta-V Usage per Episode')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    latest_path = os.path.join(save_dir, 'evader_delta_v_trend.png')
    plt.savefig(latest_path, dpi=PLOT_PARAMS['dpi'])
    plt.close()

    data = {
        'episode': episodes.tolist(),
        'delta_v': delta_v_values,
    }
    save_data_to_csv(data, os.path.join(save_dir, 'evader_delta_v.csv'))


def plot_delta_v_components(
    actions_e: np.ndarray, actions_p: np.ndarray, save_path: str
) -> None:
    """Evader와 Pursuer의 Delta-V 성분 그래프를 저장.

    각 에이전트의 vx, vy, vz 성분을 스텝에 따라 플로팅한다.

    Args:
        actions_e: 회피자의 Delta-V 배열 (step x 3)
        actions_p: 추격자의 Delta-V 배열 (step x 3)
        save_path: 저장할 파일 경로 (확장자 제외)
    """
    setup_matplotlib()

    # Ensure the output directory exists so that savefig doesn't fail
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    
    steps = np.arange(actions_e.shape[0])
    labels = ['vx', 'vy', 'vz']

    # Evader plot
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    for i, label in enumerate(labels):
        plt.plot(steps, actions_e[:, i], label=label)
    plt.xlabel('Step')
    plt.ylabel('Delta-V (m/s)')
    plt.title('Evader Delta-V Components')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}_evader_delta_v.png", dpi=PLOT_PARAMS['dpi'])
    plt.close()

    # Pursuer plot
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    for i, label in enumerate(labels):
        plt.plot(steps, actions_p[:, i], label=label)
    plt.xlabel('Step')
    plt.ylabel('Delta-V (m/s)')
    plt.title('Pursuer Delta-V Components')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}_pursuer_delta_v.png", dpi=PLOT_PARAMS['dpi'])
    plt.close()


def visualize_trajectory(states: np.ndarray,
                        actions_e: Optional[np.ndarray] = None,
                        actions_p: Optional[np.ndarray] = None,
                        title: str = "3D Trajectory",
                        save_path: Optional[str] = None,
                        nash_info: Optional[float] = None,
                        safety_info: Optional[float] = None,
                        buffer_time: Optional[float] = None,
                        show_evader_actions: bool = False,
                        arrow_length: Optional[float] = None):
    """3D 궤적 시각화

    Args:
        states: 상대 좌표계에서의 상태 배열
        actions_e: 회피자의 delta-v 기록
        actions_p: 추격자의 delta-v 기록
        title: 그래프 제목
        save_path: 저장 경로 (확장자 제외)
        nash_info: 내쉬 메트릭
        safety_info: 안전도 메트릭
        buffer_time: 버퍼 시간
        show_evader_actions: 회피자 화살표 표시 여부
        arrow_length: 화살표 길이 (normalize=True 기준). None이면 데이터 범위 기반 자동 결정
    """
    setup_matplotlib()
    
    fig = plt.figure(figsize=PLOT_PARAMS['figure_size_3d'])
    ax = fig.add_subplot(111, projection='3d')
    
    # 궤적 그리기
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 
           color=PLOT_PARAMS['colors']['trajectory'], linewidth=2, label='Trajectory')
    ax.scatter(states[0, 0], states[0, 1], states[0, 2], 
              color=PLOT_PARAMS['colors']['evader'], s=100, marker='o', label='Start')
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], 
              color=PLOT_PARAMS['colors']['pursuer'], s=100, marker='x', label='End')
    
    # Evader의 위치 (원점)
    ax.scatter(0, 0, 0, color='k', s=150, marker='*', label='Evader (Chief)')
    
    # 제목에 정보 추가
    enhanced_title = title
    if nash_info is not None:
        enhanced_title += f"\nNash: {nash_info:.2f}"
    if safety_info is not None:
        enhanced_title += f", Safety: {safety_info:.2f}"
    if buffer_time is not None:
        enhanced_title += f", Buffer: {buffer_time:.1f}s"
    
    # 축 레이블
    ax.set_xlabel('x (m) - Radial Direction')
    ax.set_ylabel('y (m) - Along-Track Direction')
    ax.set_zlabel('z (m) - Cross-Track Direction')
    ax.set_title(enhanced_title)
    
    # 축 범위를 데이터에 맞게 설정
    x_min, x_max = states[:, 0].min(), states[:, 0].max()
    y_min, y_max = states[:, 1].min(), states[:, 1].max()
    z_min, z_max = states[:, 2].min(), states[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    max_range = max(x_range, y_range, z_range)

    # Delta-v 화살표가 축 스케일 대비 지나치게 작아지지 않도록 자동 길이 설정
    if arrow_length is None:
        scale_reference = max(max_range, 1.0)
        arrow_length = 0.05 * scale_reference

        # 궤적 스텝 크기를 참고하여 화살표가 과도하게 길어지지 않도록 보정
        if states.shape[0] > 1:
            step_norms = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
            if np.any(step_norms > 0):
                typical_step = np.percentile(step_norms[step_norms > 0], 75)
                arrow_length = min(arrow_length, typical_step * 0.8)

    #if max_range == 0:
    #    max_range = 1.0
    #margin_ratio = 0.05  # 그래프가 너무 꽉 차지 않도록 약간의 여백
    #ax.set_xlim(x_min - x_range * margin_ratio, x_max + x_range * margin_ratio)
    #ax.set_ylim(y_min - y_range * margin_ratio, y_max + y_range * margin_ratio)
    #ax.set_zlim(z_min - z_range * margin_ratio, z_max + z_range * margin_ratio)
    
    # 범례
    ax.legend(loc='best')
    
    # 그리드
    ax.grid(True, alpha=0.3)
    
    # 행동 벡터 표시 - 실제 impulsive delta-v가 적용된 스텝마다 표시
    if actions_e is not None and actions_p is not None:
        impulse_p_indices = np.where(np.linalg.norm(actions_p, axis=1) > 0)[0]

        if show_evader_actions:
            impulse_e_indices = np.where(np.linalg.norm(actions_e, axis=1) > 0)[0]
            for i in impulse_e_indices:
                ax.quiver(
                    0,
                    0,
                    0,
                    actions_e[i, 0],
                    actions_e[i, 1],
                    actions_e[i, 2],
                    color=PLOT_PARAMS["colors"]["evader"],
                    length=arrow_length,
                    normalize=True,
                    alpha=0.5,
                )

        # 추격자 액션 (상대 좌표 위치에서)
        for i in impulse_p_indices:
            ax.quiver(
                states[i, 0],
                states[i, 1],
                states[i, 2],
                actions_p[i, 0],
                actions_p[i, 1],
                actions_p[i, 2],
                color=PLOT_PARAMS["colors"]["pursuer"],
                length=arrow_length,
                normalize=True,
                alpha=0.5,
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_3d_trajectory.png", dpi=PLOT_PARAMS['dpi'])
        
        # 궤적 데이터 저장
        trajectory_data = {
            'x': states[:, 0].tolist(),
            'y': states[:, 1].tolist(),
            'z': states[:, 2].tolist(),
            'vx': states[:, 3].tolist() if states.shape[1] > 3 else [],
            'vy': states[:, 4].tolist() if states.shape[1] > 4 else [],
            'vz': states[:, 5].tolist() if states.shape[1] > 5 else [],
        }
        with open(f"{save_path}_trajectory_data.json", 'w') as f:
            json.dump(_json_ready(trajectory_data), f, indent=2)
        # MATLAB 호환 형식 저장
        mat_data = {
            'x': states[:, 0],
            'y': states[:, 1],
            'z': states[:, 2],
        }
        # 속도 데이터가 있는 경우만 추가
        if states.shape[1] > 3:
            mat_data['vx'] = states[:, 3]
        if states.shape[1] > 4:
            mat_data['vy'] = states[:, 4]
        if states.shape[1] > 5:
            mat_data['vz'] = states[:, 5]
        savemat(f"{save_path}_trajectory_data.mat", mat_data)
    plt.close()
    
    # 거리 변화 그래프
    if len(states) > 1:
        plot_distance_evolution(states, save_path)


def plot_distance_evolution(states: np.ndarray, save_path: Optional[str] = None):
    """거리 변화 그래프"""
    distances = np.sqrt(np.sum(states[:, :3]**2, axis=1))
    time_steps = np.arange(len(distances))
    
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(time_steps, distances, color=PLOT_PARAMS['colors']['trajectory'], linewidth=2)
    plt.axhline(y=1000, color=PLOT_PARAMS['colors']['pursuer'], 
               linestyle='--', label='Capture Distance (1000m)')
    plt.axhline(y=5000, color=PLOT_PARAMS['colors']['evader'], 
               linestyle='--', label='Evasion Distance (5000m)')
    plt.xlabel('Time Steps')
    plt.ylabel('Distance (m)')
    plt.title('Distance Between Pursuer and Evader')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_distance.png", dpi=PLOT_PARAMS['dpi'])
        
        # 거리 데이터 저장
        distance_data = {
            'time_step': time_steps.tolist(),
            'distance': distances.tolist()
        }
        save_data_to_csv(distance_data, f"{save_path}_distance_data.csv")
    
    plt.close()


def plot_test_results(results: List[Dict], 
                     zero_sum_metrics: Dict,
                     outcome_types: Dict,
                     save_dir: Optional[str] = None):
    """테스트 결과 시각화"""
    setup_matplotlib()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    success = [r['success'] for r in results]
    distances = [r['final_distance_m'] for r in results]
    evader_dvs = [r['evader_total_delta_v_ms'] for r in results]
    
    # 1. 최종 거리와 Delta-V 사용량 그래프
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 최종 거리
    colors = [PLOT_PARAMS['colors']['success'] if s else PLOT_PARAMS['colors']['failure'] 
             for s in success]
    axes[0].bar(range(len(distances)), distances, color=colors, alpha=0.7)
    axes[0].axhline(y=1000, color='r', linestyle='--', label='Capture Distance (1000m)')
    axes[0].set_xlabel('Test Run')
    axes[0].set_ylabel('Final Distance (m)')
    axes[0].set_title('Final Distance by Test Run')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Delta-V 사용량
    axes[1].bar(range(len(evader_dvs)), evader_dvs, color=colors, alpha=0.7)
    axes[1].set_xlabel('Test Run')
    axes[1].set_ylabel('Evader Total Delta-V (m/s)')
    axes[1].set_title('Evader Propellant Usage by Test Run')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/results_summary.png", dpi=PLOT_PARAMS['dpi'])
        
        # 결과 데이터 저장
        test_results_data = {
            'test_run': list(range(len(results))),
            'success': success,
            'final_distance': distances,
            'evader_delta_v': evader_dvs
        }
        save_data_to_csv(test_results_data, f"{save_dir}/test_results.csv")
    plt.close()
    
    # 2. 종료 조건 분포 파이 차트
    plot_outcome_distribution(outcome_types, save_dir)
    
    # 3. Zero-Sum 게임 메트릭 그래프들
    if zero_sum_metrics:
        plot_zero_sum_analysis(zero_sum_metrics, success, save_dir)


def plot_outcome_distribution(outcome_types: Dict, save_dir: Optional[str] = None):
    """결과 분포 파이 차트"""
    # 비어있는 항목 필터링
    filtered_counts = []
    filtered_labels = []
    for label, count in outcome_types.items():
        if count > 0:
            filtered_counts.append(count)
            filtered_labels.append(label.replace('_', ' ').title())
    
    if filtered_counts:
        plt.figure(figsize=(10, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_counts)))
        wedges, texts, autotexts = plt.pie(filtered_counts, labels=filtered_labels, 
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90)
        plt.title('Test Outcome Distribution')
        
        if save_dir:
            plt.savefig(f"{save_dir}/outcome_distribution_pie.png", dpi=PLOT_PARAMS['dpi'])
            
            # 분포 데이터 저장
            outcome_data = {label: count for label, count in zip(filtered_labels, filtered_counts)}
            with open(f"{save_dir}/outcome_distribution.json", 'w') as f:
                json.dump(_json_ready(outcome_data), f, indent=2)
        plt.close()


def plot_zero_sum_analysis(zero_sum_metrics: Dict, success: List[bool], 
                          save_dir: Optional[str] = None):
    """Zero-Sum 게임 분석 그래프들"""
    episodes = range(1, len(zero_sum_metrics['evader_rewards']) + 1)
    
    # 1. 보상 비교 그래프
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(episodes, zero_sum_metrics['evader_rewards'], 
            color=PLOT_PARAMS['colors']['evader'], label='Evader Rewards', linewidth=2)
    plt.plot(episodes, zero_sum_metrics['pursuer_rewards'], 
            color=PLOT_PARAMS['colors']['pursuer'], label='Pursuer Rewards', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Test Run')
    plt.ylabel('Average Reward')
    plt.title('Zero-Sum Game Rewards by Test Run')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/zero_sum_rewards.png", dpi=PLOT_PARAMS['dpi'])
    plt.close()
    
    # 2. Nash Equilibrium 메트릭
    if 'nash_metrics' in zero_sum_metrics and zero_sum_metrics['nash_metrics']:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        plt.plot(episodes, zero_sum_metrics['nash_metrics'], 'b-', linewidth=2)
        plt.xlabel('Test Run')
        plt.ylabel('Nash Equilibrium Metric')
        plt.title('Nash Equilibrium Metric by Test Run')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/nash_metrics.png", dpi=PLOT_PARAMS['dpi'])
        plt.close()
    
    # 3. Zero-Sum 검증 그래프
    rewards_sum = [zero_sum_metrics['evader_rewards'][i] + zero_sum_metrics['pursuer_rewards'][i]
                  for i in range(len(zero_sum_metrics['evader_rewards']))]
    
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(episodes, rewards_sum, 'k-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Zero-Sum')
    plt.xlabel('Test Run')
    plt.ylabel('Rewards Sum (Evader + Pursuer)')
    plt.title('Zero-Sum Game Verification')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/zero_sum_verification.png", dpi=PLOT_PARAMS['dpi'])
        
        # Zero-Sum 메트릭 데이터 저장
        zero_sum_data = {
            'test_run': list(episodes),
            'evader_reward': zero_sum_metrics['evader_rewards'],
            'pursuer_reward': zero_sum_metrics['pursuer_rewards'],
            'reward_sum': rewards_sum
        }
        if 'nash_metrics' in zero_sum_metrics:
            zero_sum_data['nash_metric'] = zero_sum_metrics['nash_metrics']
        save_data_to_csv(zero_sum_data, f"{save_dir}/zero_sum_metrics.csv")
    plt.close()
    
    # 4. 안전도 점수 그래프
    if 'safety_scores' in zero_sum_metrics and zero_sum_metrics['safety_scores']:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        safety_scores = zero_sum_metrics['safety_scores']
        colors = ['g' if s > SAFETY_THRESHOLDS['permanent_evasion'] 
                 else ('b' if s > SAFETY_THRESHOLDS['conditional_evasion'] else 'r') 
                 for s in safety_scores]
        
        plt.bar(range(1, len(safety_scores) + 1), safety_scores, color=colors, alpha=0.7)
        plt.axhline(y=SAFETY_THRESHOLDS['permanent_evasion'], color='g', 
                   linestyle='--', label='Permanent Evasion Threshold')
        plt.axhline(y=SAFETY_THRESHOLDS['conditional_evasion'], color='b', 
                   linestyle='--', label='Conditional Evasion Threshold')
        plt.xlabel('Test Run')
        plt.ylabel('Safety Score')
        plt.title('Safety Score by Test Run')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/safety_scores.png", dpi=PLOT_PARAMS['dpi'])
        plt.close()


def plot_orbital_elements_comparison(evader_elements: Dict, pursuer_elements: Dict,
                                   save_path: Optional[str] = None):
    """궤도 요소 비교 그래프"""
    elements = ['a', 'e', 'i', 'RAAN', 'omega', 'M']
    evader_vals = [evader_elements[e] for e in elements]
    pursuer_vals = [pursuer_elements[e] for e in elements]
    
    # 단위 정규화
    normalized_evader = []
    normalized_pursuer = []
    labels = []
    
    for i, elem in enumerate(elements):
        if elem == 'a':  # 반장축은 km 단위로
            normalized_evader.append(evader_vals[i] / 1000)
            normalized_pursuer.append(pursuer_vals[i] / 1000)
            labels.append('a (km)')
        elif elem in ['i', 'RAAN', 'omega', 'M']:  # 각도는 도 단위로
            normalized_evader.append(evader_vals[i] * 180 / np.pi)
            normalized_pursuer.append(pursuer_vals[i] * 180 / np.pi)
            labels.append(f'{elem} (deg)')
        else:  # 이심률은 그대로
            normalized_evader.append(evader_vals[i])
            normalized_pursuer.append(pursuer_vals[i])
            labels.append(elem)
    
    # 그래프 생성
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, normalized_evader, width, 
           label='Evader', color=PLOT_PARAMS['colors']['evader'], alpha=0.7)
    plt.bar(x + width/2, normalized_pursuer, width, 
           label='Pursuer', color=PLOT_PARAMS['colors']['pursuer'], alpha=0.7)
    
    plt.xlabel('Orbital Elements')
    plt.ylabel('Values')
    plt.title('Orbital Elements Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_orbital_elements.png", dpi=PLOT_PARAMS['dpi'])
    plt.close()


def plot_eci_trajectories(
    times: np.ndarray,
    pursuer_states: np.ndarray,
    evader_states: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ECI Trajectories",
    show_earth: bool = True,
    show_stats: bool = True,
    use_plotly: bool = True,
    animate: bool = False,
):
    """ECI 프레임 궤적 시각화

    각 시각별 위치를 점으로 표시하며, 시간에 따라 색상이 변한다.
    Pursuer는 적색 스펙트럼, Evader는 청색 스펙트럼을 사용한다.
    ``use_plotly``가 ``True``이면 Plotly 기반 인터랙티브 HTML을 생성한다.
    ``animate``가 ``True``이면 시간 순서대로 점이 나타나는 애니메이션을 반환한다.
    """
    if save_path:
        ephemeris_data = {
            't': times.astype(float),  # float 타입 보장
            'evader_x': evader_states[:, 0],
            'evader_y': evader_states[:, 1],
            'evader_z': evader_states[:, 2],
            'pursuer_x': pursuer_states[:, 0],
            'pursuer_y': pursuer_states[:, 1],
            'pursuer_z': pursuer_states[:, 2],
        }
        # MAT 파일 저장
        savemat(f"{save_path}_eci.mat", ephemeris_data)
        # JSON 파일 저장 (리스트 변환)
        json_data = {k: v.tolist() for k, v in ephemeris_data.items()}
        with open(f"{save_path}_eci.json", 'w') as f:
            json.dump(_json_ready(json_data), f, indent=2)

    if use_plotly:
        if animate:
            return _plot_eci_trajectories_plotly_live(
                times,
                pursuer_states,
                evader_states,
                save_path=save_path,
                title=title,
                show_earth=show_earth,
            )
        return _plot_eci_trajectories_plotly(
            times,
            pursuer_states,
            evader_states,
            save_path=save_path,
            title=title,
            show_earth=show_earth,
            show_stats=show_stats,
        )

    setup_matplotlib()
    fig = plt.figure(figsize=PLOT_PARAMS['figure_size_3d'])
    ax = fig.add_subplot(111, projection='3d')

    # 시간에 따른 색상 매핑 준비
    norm = plt.Normalize(times.min(), times.max())
    cmap_evader = plt.get_cmap('Blues')
    cmap_pursuer = plt.get_cmap('Reds')

    # 각 시점의 실제 데이터를 점으로 표시 (원 모양으로 통일)
    sc_evader = ax.scatter(
        evader_states[:, 0], evader_states[:, 1], evader_states[:, 2],
        c=times, cmap=cmap_evader, norm=norm, s=8, marker='o', alpha=0.8  # marker='o'로 변경, 크기 조정
    )
    sc_pursuer = ax.scatter(
        pursuer_states[:, 0], pursuer_states[:, 1], pursuer_states[:, 2],
        c=times, cmap=cmap_pursuer, norm=norm, s=8, marker='o', alpha=0.8  # marker='o'로 변경, 크기 조정
    )
    
    # 시작점과 끝점 표시 - 시각적 구분 강화
    # Evader 시작점 - 크고 진한 파란색 사각형
    ax.scatter(
        evader_states[0, 0], evader_states[0, 1], evader_states[0, 2],
        c='darkblue', s=100, marker='s', edgecolors='white', linewidth=2  # 크기 증가, 색상 변경, 테두리 추가
    )
    # Evader 끝점 - 크고 연한 파란색 X
    ax.scatter(
        evader_states[-1, 0], evader_states[-1, 1], evader_states[-1, 2],
        c='lightblue', s=100, marker='X', edgecolors='darkblue', linewidth=2  # 크기 증가, 색상 변경, 테두리 추가
    )
    # Pursuer 시작점 - 크고 진한 빨간색 다이아몬드
    ax.scatter(
        pursuer_states[0, 0], pursuer_states[0, 1], pursuer_states[0, 2],
        c='darkred', s=100, marker='D', edgecolors='white', linewidth=2  # 크기 증가, 색상 변경, 테두리 추가
    )
    # Pursuer 끝점 - 크고 연한 빨간색 별
    ax.scatter(
        pursuer_states[-1, 0], pursuer_states[-1, 1], pursuer_states[-1, 2],
        c='lightcoral', s=100, marker='*', edgecolors='darkred', linewidth=2  # 크기 증가, 색상 변경, 테두리 추가
    )
    
    # 지구 표시
    if show_earth:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))  # 미터 단위
        y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z_earth = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_earth, y_earth, z_earth, color='grey', alpha=0.3)
    
    # 축 설정
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 축 범위 설정
    combined = np.vstack((evader_states[:, :3], pursuer_states[:, :3]))
    max_range = np.max(np.abs(combined)) * 1.1  # 10% 여유
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # 컬러바 및 범례
    cbar = fig.colorbar(sc_evader, ax=ax, pad=0.1)
    cbar.set_label('Time (s)')

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap_evader(0.8),
               markersize=8, label='Evader Start'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=cmap_evader(0.2),
               markersize=8, label='Evader End'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=cmap_pursuer(0.8),
               markersize=8, label='Pursuer Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=cmap_pursuer(0.2),
               markersize=10, label='Pursuer End'),
    ]
    ax.legend(handles=handles, loc='best')
    
    # 통계 정보 표시
    if show_stats:
        # 초기/최종 거리
        initial_dist = np.linalg.norm(evader_states[0, :3] - pursuer_states[0, :3])
        final_dist = np.linalg.norm(evader_states[-1, :3] - pursuer_states[-1, :3])
        
        # 평균 고도
        evader_alt = np.mean(np.linalg.norm(evader_states[:, :3], axis=1)) - R_EARTH
        pursuer_alt = np.mean(np.linalg.norm(pursuer_states[:, :3], axis=1)) - R_EARTH
        
        textstr = f'Initial Distance: {initial_dist/1000:.1f} km\n'
        textstr += f'Final Distance: {final_dist/1000:.1f} km\n'
        textstr += f'Duration: {times[-1]/60:.1f} min\n'
        textstr += f'Avg Altitude: E={evader_alt/1000:.0f} km, P={pursuer_alt/1000:.0f} km'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, 
                  fontsize=10, verticalalignment='top', bbox=props)
    
    # 뷰 각도 설정 (더 나은 시각화를 위해)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(f"{save_path}_eci.png", dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
      
        plt.close()
    else:
        plt.show()  # save_path가 없으면 화면에 표시


def _plot_eci_trajectories_plotly(times: np.ndarray,
                                  pursuer_states: np.ndarray,
                                  evader_states: np.ndarray,
                                  save_path: Optional[str] = None,
                                  title: str = "ECI Trajectories",
                                  show_earth: bool = True,
                                  show_stats: bool = True):
    """Plotly를 이용한 인터랙티브 ECI 궤적 시각화

    Pursuer는 적색, Evader는 청색 스펙트럼을 사용하여 점을 표시한다.
    시작과 끝 지점은 각각 별도의 모양으로 강조된다.
    """
    import plotly.graph_objects as go

    # Plotly에서는 km 단위로 시각화하기 위해 좌표를 변환한다
    evader_states_km = evader_states / 1000.0
    pursuer_states_km = pursuer_states / 1000.0
    # 시간 정보는 그대로 사용하며 색상 범위는 전체 시간 구간에 맞춘다
    time_values_sec = times.astype(float)
    time_values_min = time_values_sec / 60.0

    fig = go.Figure()

    # 기본 궤적 - 원 모양의 점들로 표시 (km 단위)
    fig.add_trace(
        go.Scatter3d(
            x=evader_states_km[:, 0],
            y=evader_states_km[:, 1],
            z=evader_states_km[:, 2],
            mode="markers",
            marker=dict(
                size=2,  # 크기 약간 감소
                color=time_values_min,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Time (min)", x=1.0),
                cmin=time_values_min.min(),
                cmax=time_values_min.max(),
                symbol="circle"  # 명시적으로 원 모양 지정
            ),
            name="Evader",
            customdata=np.column_stack((evader_states_km[:, 0], evader_states_km[:, 1], evader_states_km[:, 2], time_values_sec)),
            hovertemplate=(
                "x: %{customdata[0]:.2f} km<br>" +
                "y: %{customdata[1]:.2f} km<br>" +
                "z: %{customdata[2]:.2f} km<br>" +
                "t: %{customdata[3]:.2f}s<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=pursuer_states_km[:, 0],
            y=pursuer_states_km[:, 1],
            z=pursuer_states_km[:, 2],
            mode="markers",
            marker=dict(
                size=2,  # 크기 약간 감소
                color=time_values_min,
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Time (min)", x=1.05),
                cmin=time_values_min.min(),
                cmax=time_values_min.max(),
                symbol="triangle"  # 삼각형 모양 지정
            ),
            name="Pursuer",
            customdata=np.column_stack((pursuer_states_km[:, 0], pursuer_states_km[:, 1], pursuer_states_km[:, 2], time_values_sec)),
            hovertemplate=(
                "x: %{customdata[0]:.2f} km<br>" +
                "y: %{customdata[1]:.2f} km<br>" +
                "z: %{customdata[2]:.2f} km<br>" +
                "t: %{customdata[3]:.2f}s<extra></extra>"
            ),
        )
    )

    # 시작과 끝 표시 - 시각적 구분 강화
    # Evader 시작점 - 크고 진한 파란색 사각형
    fig.add_trace(
        go.Scatter3d(
            x=[evader_states_km[0, 0]],
            y=[evader_states_km[0, 1]],
            z=[evader_states_km[0, 2]],
            mode="markers",
            marker=dict(
                size=6,  # 크기
                color="darkblue",  # 진한 파란색
                symbol="square",
                line=dict(color="white", width=2)  # 흰색 테두리 추가
            ),
            name="Evader Start",
            customdata=[[evader_states_km[0, 0], evader_states_km[0, 1], evader_states_km[0, 2], time_values_sec[0]]],
            hovertemplate=(
                "x: %{customdata[0]:.2f} km<br>" +
                "y: %{customdata[1]:.2f} km<br>" +
                "z: %{customdata[2]:.2f} km<br>" +
                "t: %{customdata[3]:.2f}s<extra></extra>"
            ),
        )
    )
    
    # Evader 끝점 - 크고 연한 파란색 X 표시
    fig.add_trace(
        go.Scatter3d(
            x=[evader_states_km[-1, 0]],
            y=[evader_states_km[-1, 1]],
            z=[evader_states_km[-1, 2]],
            mode="markers",
            marker=dict(
                size=6,  # 크기
                color="lightblue",  # 연한 파란색
                symbol="x",
                line=dict(color="darkblue", width=2)  # 진한 파란색 테두리
            ),
            name="Evader End",
            customdata=[[evader_states_km[-1, 0], evader_states_km[-1, 1], evader_states_km[-1, 2], time_values_sec[-1]]],
            hovertemplate=(
                "x: %{customdata[0]:.2f} km<br>" +
                "y: %{customdata[1]:.2f} km<br>" +
                "z: %{customdata[2]:.2f} km<br>" +
                "t: %{customdata[3]:.2f}s<extra></extra>"
            ),
        )
    )
    
    # Pursuer 시작점 - 크고 진한 빨간색 다이아몬드
    fig.add_trace(
        go.Scatter3d(
            x=[pursuer_states_km[0, 0]],
            y=[pursuer_states_km[0, 1]],
            z=[pursuer_states_km[0, 2]],
            mode="markers",
            marker=dict(
                size=6,  # 크기
                color="darkred",  # 진한 빨간색
                symbol="diamond",
                line=dict(color="white", width=2)  # 흰색 테두리 추가
            ),
            name="Pursuer Start",
            customdata=[[pursuer_states_km[0, 0], pursuer_states_km[0, 1], pursuer_states_km[0, 2], time_values_sec[0]]],
            hovertemplate=(
                "x: %{customdata[0]:.2f} km<br>" +
                "y: %{customdata[1]:.2f} km<br>" +
                "z: %{customdata[2]:.2f} km<br>" +
                "t: %{customdata[3]:.2f}s<extra></extra>"
            ),
        )
    )
    
    # Pursuer 끝점 - 크고 연한 빨간색 십자가 표시
    fig.add_trace(
        go.Scatter3d(
            x=[pursuer_states_km[-1, 0]],
            y=[pursuer_states_km[-1, 1]],
            z=[pursuer_states_km[-1, 2]],
            mode="markers",
            marker=dict(
                size=6,  # 크기
                color="lightcoral",  # 연한 빨간색
                symbol="cross",  # 'star' 대신 'cross' 사용
                line=dict(color="darkred", width=2)  # 진한 빨간색 테두리
            ),
            name="Pursuer End",
            customdata=[[pursuer_states_km[-1, 0], pursuer_states_km[-1, 1], pursuer_states_km[-1, 2], time_values_sec[-1]]],
            hovertemplate=(
                "x: %{customdata[0]:.2f} km<br>" +
                "y: %{customdata[1]:.2f} km<br>" +
                "z: %{customdata[2]:.2f} km<br>" +
                "t: %{customdata[3]:.2f}s<extra></extra>"
            ),
        )
    )

    if show_earth:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        # 지구 반지름을 km 단위로 사용
        x_earth = (R_EARTH / 1000) * np.outer(np.cos(u), np.sin(v))
        y_earth = (R_EARTH / 1000) * np.outer(np.sin(u), np.sin(v))
        z_earth = (R_EARTH / 1000) * np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(
            go.Surface(
                x=x_earth, 
                y=y_earth, 
                z=z_earth, 
                opacity=0.3, 
                showscale=False, 
                colorscale=[[0, "lightblue"], [1, "lightblue"]],
                name="Earth"
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        legend=dict(
            itemsizing="constant",
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    if show_stats:
        initial_dist = np.linalg.norm(evader_states[0, :3] - pursuer_states[0, :3])
        final_dist = np.linalg.norm(evader_states[-1, :3] - pursuer_states[-1, :3])
        evader_alt = np.mean(np.linalg.norm(evader_states[:, :3], axis=1)) - R_EARTH
        pursuer_alt = np.mean(np.linalg.norm(pursuer_states[:, :3], axis=1)) - R_EARTH
        
        textstr = f"Initial Distance: {initial_dist/1000:.1f} km<br>"
        textstr += f"Final Distance: {final_dist/1000:.1f} km<br>"
        textstr += f"Duration: {times[-1]/60:.1f} min<br>"
        textstr += f"Avg Altitude: E={evader_alt/1000:.0f} km, P={pursuer_alt/1000:.0f} km"
        
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=textstr,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        )

    if save_path:
        fig.write_html(f"{save_path}_eci.html")
        
    return fig


def _plot_eci_trajectories_plotly_live(
    times: np.ndarray,
    pursuer_states: np.ndarray,
    evader_states: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ECI Trajectories",
    show_earth: bool = True,
):
    """Plotly 애니메이션으로 ECI 궤적을 실시간 재생"""

    import plotly.graph_objects as go

    evader_km = evader_states / 1000.0
    pursuer_km = pursuer_states / 1000.0

    frames = []
    for k in range(len(times)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=evader_km[: k + 1, 0],
                        y=evader_km[: k + 1, 1],
                        z=evader_km[: k + 1, 2],
                        mode="lines",
                        line=dict(color="blue"),
                    ),
                    go.Scatter3d(
                        x=pursuer_km[: k + 1, 0],
                        y=pursuer_km[: k + 1, 1],
                        z=pursuer_km[: k + 1, 2],
                        mode="lines",
                        line=dict(color="red"),
                    ),
                    go.Scatter3d(
                        x=[evader_km[k, 0]],
                        y=[evader_km[k, 1]],
                        z=[evader_km[k, 2]],
                        mode="markers",
                        marker=dict(color="blue", size=4),
                        name="Evader",
                    ),
                    go.Scatter3d(
                        x=[pursuer_km[k, 0]],
                        y=[pursuer_km[k, 1]],
                        z=[pursuer_km[k, 2]],
                        mode="markers",
                        marker=dict(color="red", size=4),
                        name="Pursuer",
                    ),
                ],
                name=str(k),
            )
        )

    data = frames[0].data
    if show_earth:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_earth = (R_EARTH / 1000) * np.outer(np.cos(u), np.sin(v))
        y_earth = (R_EARTH / 1000) * np.outer(np.sin(u), np.sin(v))
        z_earth = (R_EARTH / 1000) * np.outer(np.ones(np.size(u)), np.cos(v))
        data = list(data) + [
            go.Surface(
                x=x_earth,
                y=y_earth,
                z=z_earth,
                opacity=0.3,
                showscale=False,
                colorscale=[[0, "lightblue"], [1, "lightblue"]],
                name="Earth",
            )
        ]

    fig = go.Figure(data=data, frames=frames)

    steps = [
        dict(
            method="animate",
            args=[[f.name], {"mode": "immediate", "frame": {"duration": 50, "redraw": True}}],
            label=str(idx),
        )
        for idx, f in enumerate(frames)
    ]
    sliders = [dict(active=0, steps=steps, x=0.1, y=0, len=0.9)]
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)", aspectmode="data"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
                x=0,
                y=1.05,
            )
        ],
        sliders=sliders,
        margin=dict(l=0, r=0, b=0, t=50),
    )

    if save_path:
        fig.write_html(f"{save_path}_eci_live.html")

    return fig

def numpy_to_python(obj):
    """NumPy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

def create_summary_dashboard(training_stats: Dict, test_results: List[Dict],
                           save_dir: str):
    """종합 대시보드 생성"""
    setup_matplotlib()
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training and Testing Summary Dashboard', fontsize=16)

    def _safe_hist(ax, data, bins=10, color='blue', **kwargs):
        """작은 표본/제로 분산 데이터를 위해 안전하게 히스토그램을 그린다."""
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
            return

        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            ax.text(0.5, 0.5, '유효 데이터 없음', ha='center', va='center')
            return

        data = finite_data
        value_range = np.max(data) - np.min(data)
        # 표본 개수보다 많은 bin은 불필요하므로 제한
        max_bins = max(1, min(bins, data.size))

        if value_range <= 0:
            # 단일 값만 있을 때는 수동 범위를 지정해 0 폭 문제 방지
            center = data[0]
            span = max(1e-3, abs(center) * 0.1)
            hist_range = (center - span, center + span)
            ax.hist(data, bins=1, range=hist_range, color=color, **kwargs)
        else:
            ax.hist(data, bins=max_bins, color=color, **kwargs)

    # 1. 성공률 추이
    if 'success_rates' in training_stats and training_stats['success_rates']:
        axes[0, 0].plot(training_stats['success_rates'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Success Rate')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
    
    # 2. 테스트 결과 분포
    success_count = 0
    failure_count = 0
    if test_results:
        success_count = sum(1 for r in test_results if r.get('success'))
        failure_count = len(test_results) - success_count
        axes[0, 1].pie([success_count, failure_count], labels=['Success', 'Failure'], 
                       autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        axes[0, 1].set_title('Test Results Distribution')
    
    # 3. 평균 보상 추이
    if 'nash_metrics' in training_stats and training_stats['nash_metrics']:
        axes[0, 2].plot(training_stats['nash_metrics'], 'g-', linewidth=2)
        axes[0, 2].set_title('Nash Equilibrium Convergence')
        axes[0, 2].set_xlabel('Evaluations')
        axes[0, 2].set_ylabel('Nash Metric')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
    
    # 4. Delta-V 사용량 분포
    delta_vs = []
    if test_results:
        delta_vs = [r['evader_total_delta_v_ms'] for r in test_results]
        _safe_hist(axes[1, 0], delta_vs, bins=10, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Delta-V Usage Distribution')
        axes[1, 0].set_xlabel('Delta-V (m/s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

    # 5. 최종 거리 분포
    distances = []
    if test_results:
        distances = [r['final_distance_m'] for r in test_results]
        _safe_hist(axes[1, 1], distances, bins=10, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=1000, color='r', linestyle='--', label='Capture Threshold')
        axes[1, 1].set_title('Final Distance Distribution')
        axes[1, 1].set_xlabel('Distance (m)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 성능 요약 텍스트
    axes[1, 2].axis('off')
    success_rate = success_count / len(test_results) if test_results else 0
    avg_delta_v = np.mean(delta_vs) if delta_vs else 0
    avg_distance = np.mean(distances) if distances else 0
    nash_metric = (
        training_stats.get('nash_metrics', [0])[-1]
        if 'nash_metrics' in training_stats and training_stats['nash_metrics']
        else 0
    )

    summary_text = (
        "Training Episodes: {episodes}\n"
        "Test Success Rate: {success_rate}\n"
        "Avg Delta-V: {avg_delta_v}\n"
        "Avg Distance: {avg_distance}\n"
        "Nash Metric: {nash_metric:.3f}"
    ).format(
        episodes=training_stats.get('episodes_completed', 'N/A'),
        success_rate=f"{success_rate:.1%}" if test_results else 'N/A',
        avg_delta_v=f"{avg_delta_v:.1f} m/s" if delta_vs else 'N/A',
        avg_distance=f"{avg_distance:.0f} m" if distances else 'N/A',
        nash_metric=nash_metric,
    )
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 2].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/summary_dashboard.png", dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    plt.close()
    
    # 대시보드 데이터 저장
    dashboard_data = {
        'training_stats': {
            'episodes_completed': int(training_stats.get('episodes_completed', 0)),
            'final_success_rate': float(training_stats.get('success_rates', [0])[-1]) if 'success_rates' in training_stats and training_stats['success_rates'] else 0,
            'final_nash_metric': float(training_stats.get('nash_metrics', [0])[-1]) if 'nash_metrics' in training_stats and training_stats['nash_metrics'] else 0,
        },
        'test_stats': {
            'total_tests': int(len(test_results)) if test_results else 0,
            'success_count': int(success_count) if test_results else 0,
            'success_rate': float(success_count/len(test_results)) if test_results else 0,
            'avg_delta_v': float(np.mean(delta_vs)) if test_results and delta_vs else 0,
            'avg_distance': float(np.mean(distances)) if test_results and distances else 0,
        }
    }
    
    # NumPy 타입을 Python 타입으로 변환
    dashboard_data = numpy_to_python(dashboard_data)
    
    try:
        with open(f"{save_dir}/dashboard_summary.json", 'w') as f:
            json.dump(_json_ready(dashboard_data), f, indent=2)
    except TypeError as e:
        print(f"JSON 저장 오류: {e}")
        # 문제가 있는 데이터 타입 출력
        for key, value in dashboard_data.items():
            print(f"{key}: {type(value)}")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {type(v)}")


# 모듈 초기화 시 matplotlib 설정
setup_matplotlib()
