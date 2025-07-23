"""
시각화 및 그래프 생성 모듈 (데이터 저장 개선)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple, Any
import os
import json
import csv

from utils.constants import PLOT_PARAMS, SAFETY_THRESHOLDS


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
        plt.savefig(f'{save_dir}/success_rate_ep{episode_count}.png', dpi=PLOT_PARAMS['dpi'])
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
            plt.savefig(f'{save_dir}/outcome_distribution_ep{episode_count}.png', dpi=PLOT_PARAMS['dpi'])
            plt.close()
            
            # 결과 분포 데이터 저장
            outcome_data = {label: count for label, count in zip(filtered_labels, filtered_counts)}
            with open(f'{save_dir}/outcome_distribution.json', 'w') as f:
                json.dump(outcome_data, f, indent=2)
    
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
        plt.savefig(f'{save_dir}/zero_sum_rewards_ep{episode_count}.png', dpi=PLOT_PARAMS['dpi'])
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
        eval_episodes = range(0, episode_count, 10)  # 10 에피소드마다 평가 가정
        plt.plot(eval_episodes[:len(nash_metrics)], nash_metrics, 'b-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Nash Equilibrium Metric')
        plt.title('Nash Equilibrium Convergence')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/nash_metric_ep{episode_count}.png', dpi=PLOT_PARAMS['dpi'])
        plt.close()
        
        # Nash 메트릭 데이터 저장
        nash_data = {
            'episode': list(eval_episodes[:len(nash_metrics)]),
            'nash_metric': nash_metrics
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
        plt.savefig(f'{save_dir}/buffer_time_stats_ep{episode_count}.png', dpi=PLOT_PARAMS['dpi'])
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
        json.dump(all_training_data, f, indent=2)


def visualize_trajectory(states: np.ndarray, 
                        actions_e: Optional[np.ndarray] = None,
                        actions_p: Optional[np.ndarray] = None, 
                        title: str = "3D Trajectory",
                        save_path: Optional[str] = None, 
                        nash_info: Optional[float] = None,
                        safety_info: Optional[float] = None, 
                        buffer_time: Optional[float] = None):
    """3D 궤적 시각화"""
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
    
    # 동일한 스케일
    max_range = np.max([
        np.abs(states[:, 0]).max(),
        np.abs(states[:, 1]).max(),
        np.abs(states[:, 2]).max()
    ])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # 범례
    ax.legend(loc='best')
    
    # 그리드
    ax.grid(True, alpha=0.3)
    
    # 행동 벡터 표시 (선택적)
    if actions_e is not None and actions_p is not None:
        # 일부 행동만 표시 (가독성을 위해)
        step = max(1, len(states) // 20)
        
        for i in range(0, len(states), step):
            if i < len(actions_e) and np.any(actions_e[i] != 0):
                # 회피자 액션 (원점에서)
                ax.quiver(
                    0, 0, 0,
                    actions_e[i, 0], actions_e[i, 1], actions_e[i, 2],
                    color=PLOT_PARAMS['colors']['evader'], 
                    length=max_range/10, normalize=True, alpha=0.5
                )
            
            if i < len(actions_p) and np.any(actions_p[i] != 0):
                # 추격자 액션
                ax.quiver(
                    states[i, 0], states[i, 1], states[i, 2],
                    actions_p[i, 0], actions_p[i, 1], actions_p[i, 2],
                    color=PLOT_PARAMS['colors']['pursuer'], 
                    length=max_range/10, normalize=True, alpha=0.5
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
            json.dump(trajectory_data, f, indent=2)
    
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
    plt.axhline(y=50000, color=PLOT_PARAMS['colors']['evader'], 
               linestyle='--', label='Evasion Distance (50000m)')
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
                json.dump(outcome_data, f, indent=2)
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


def plot_eci_trajectories(times: np.ndarray,
                          pursuer_states: np.ndarray,
                          evader_states: np.ndarray,
                          save_path: Optional[str] = None,
                          title: str = "ECI Trajectories",
                          show_earth: bool = True, 
                          show_stats: bool = True):
    """ECI 프레임 궤적 시각화"""
    setup_matplotlib()
    fig = plt.figure(figsize=PLOT_PARAMS['figure_size_3d'])
    ax = fig.add_subplot(111, projection='3d')
    
    # 궤적 플롯
    ax.plot(evader_states[:, 0], evader_states[:, 1], evader_states[:, 2],
            color=PLOT_PARAMS['colors']['evader'], label='Evader', linewidth=2)
    ax.plot(pursuer_states[:, 0], pursuer_states[:, 1], pursuer_states[:, 2],
            color=PLOT_PARAMS['colors']['pursuer'], label='Pursuer', linewidth=2)
    
    # 시작점과 끝점 표시
    ax.scatter(evader_states[0, 0], evader_states[0, 1], evader_states[0, 2], 
               c='green', s=100, marker='o', label='Start')
    ax.scatter(evader_states[-1, 0], evader_states[-1, 1], evader_states[-1, 2], 
               c='red', s=100, marker='*', label='End')
    
    # 지구 표시
    if show_earth:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_earth = 6371e3 * np.outer(np.cos(u), np.sin(v))  # 미터 단위
        y_earth = 6371e3 * np.outer(np.sin(u), np.sin(v))
        z_earth = 6371e3 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)
    
    # 축 설정
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 축 범위 설정
    combined = np.vstack((evader_states[:, :3], pursuer_states[:, :3]))
    max_range = np.max(np.abs(combined)) * 1.1  # 10% 여유
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # 통계 정보 표시
    if show_stats:
        # 초기/최종 거리
        initial_dist = np.linalg.norm(evader_states[0, :3] - pursuer_states[0, :3])
        final_dist = np.linalg.norm(evader_states[-1, :3] - pursuer_states[-1, :3])
        
        # 평균 고도
        evader_alt = np.mean(np.linalg.norm(evader_states[:, :3], axis=1)) - 6371e3
        pursuer_alt = np.mean(np.linalg.norm(pursuer_states[:, :3], axis=1)) - 6371e3
        
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
        
        # 데이터도 저장
        ephemeris_data = {
            't': times.tolist(),
            'evader': evader_states.tolist(),
            'pursuer': pursuer_states.tolist()
        }
        with open(f"{save_path}_eci.json", 'w') as f:
            json.dump(ephemeris_data, f, indent=2)
        
        plt.close()
    else:
        plt.show()  # save_path가 없으면 화면에 표시

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
    
    # 1. 성공률 추이
    if 'success_rates' in training_stats and training_stats['success_rates']:
        axes[0, 0].plot(training_stats['success_rates'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Success Rate')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
    
    # 2. 테스트 결과 분포
    if test_results:
        success_count = sum(1 for r in test_results if r['success'])
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
    if test_results:
        delta_vs = [r['evader_total_delta_v_ms'] for r in test_results]
        axes[1, 0].hist(delta_vs, bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_title('Delta-V Usage Distribution')
        axes[1, 0].set_xlabel('Delta-V (m/s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 최종 거리 분포
    if test_results:
        distances = [r['final_distance_m'] for r in test_results]
        axes[1, 1].hist(distances, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(x=1000, color='r', linestyle='--', label='Capture Threshold')
        axes[1, 1].set_title('Final Distance Distribution')
        axes[1, 1].set_xlabel('Distance (m)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 성능 요약 텍스트
    axes[1, 2].axis('off')
    summary_text = f"""
    Training Episodes: {training_stats.get('episodes_completed', 'N/A')}
    Test Success Rate: {success_count/len(test_results):.1%} if test_results else 'N/A'
    Avg Delta-V: {np.mean(delta_vs):.1f} m/s if test_results else 'N/A'
    Avg Distance: {np.mean(distances):.0f} m if test_results else 'N/A'
    Nash Metric: {training_stats.get('nash_metrics', [0])[-1] if 'nash_metrics' in training_stats and training_stats['nash_metrics'] else 0:.3f}
    """
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
            json.dump(dashboard_data, f, indent=2)
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
