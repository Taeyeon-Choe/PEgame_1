"""
시각화 및 그래프 생성 모듈
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple, Any
import os

from utils.constants import PLOT_PARAMS, SAFETY_THRESHOLDS


def setup_matplotlib():
    """Matplotlib 설정"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = PLOT_PARAMS['figure_size_2d']
    plt.rcParams['figure.dpi'] = PLOT_PARAMS['dpi']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True


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
    
    # 1. 성공률 그래프
    if success_rates:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        plt.plot(range(len(success_rates)), success_rates, 
                label='성공률 (이동 평균)', color=PLOT_PARAMS['colors']['evader'])
        plt.xlabel('에피소드')
        plt.ylabel('성공률')
        plt.title(f'학습 과정 성공률 (에피소드 {episode_count})')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f'{save_dir}/success_rate_ep{episode_count}.png')
        plt.close()
    
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
            plt.pie(filtered_counts, labels=filtered_labels, autopct='%1.1f%%')
            plt.title(f'결과 분포 (에피소드 {episode_count})')
            plt.savefig(f'{save_dir}/outcome_distribution_ep{episode_count}.png')
            plt.close()
    
    # 3. Zero-Sum 게임 보상 그래프
    if len(evader_rewards) > 100:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        episodes = range(len(evader_rewards))
        plt.plot(episodes, evader_rewards, label='회피자 보상', 
                color=PLOT_PARAMS['colors']['evader'], alpha=0.4)
        plt.plot(episodes, pursuer_rewards, label='추격자 보상', 
                color=PLOT_PARAMS['colors']['pursuer'], alpha=0.4)
        
        # 이동 평균 추가
        window_size = 100
        if len(evader_rewards) >= window_size:
            evader_ma = np.convolve(evader_rewards, np.ones(window_size)/window_size, mode='valid')
            pursuer_ma = np.convolve(pursuer_rewards, np.ones(window_size)/window_size, mode='valid')
            ma_episodes = range(window_size-1, len(evader_rewards))
            plt.plot(ma_episodes, evader_ma, label='회피자 보상 (MA)', 
                    color='darkgreen', linewidth=2)
            plt.plot(ma_episodes, pursuer_ma, label='추격자 보상 (MA)', 
                    color='darkred', linewidth=2)
        
        # Zero-Sum 검증
        reward_sum = [evader_rewards[i] + pursuer_rewards[i] for i in range(len(evader_rewards))]
        plt.plot(episodes, reward_sum, label='보상 합계 (Zero-Sum 검증)', 
                color='black', linestyle='--')
        
        plt.axhline(y=0, color='k', linestyle=':')
        plt.xlabel('에피소드')
        plt.ylabel('보상')
        plt.title('Zero-Sum 게임 보상 추이')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/zero_sum_rewards_ep{episode_count}.png')
        plt.close()
    
    # 4. Nash Equilibrium 메트릭 그래프
    if nash_metrics:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        eval_episodes = range(0, episode_count, 10)  # 10 에피소드마다 평가 가정
        plt.plot(eval_episodes[:len(nash_metrics)], nash_metrics)
        plt.xlabel('에피소드')
        plt.ylabel('Nash Equilibrium 메트릭')
        plt.title('Nash Equilibrium 수렴도')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.savefig(f'{save_dir}/nash_metric_ep{episode_count}.png')
        plt.close()
    
    # 5. 버퍼 시간 통계 그래프
    if buffer_times:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        plt.hist(buffer_times, bins=20, alpha=0.7)
        plt.axvline(x=np.mean(buffer_times), color='r', linestyle='--',
                   label=f'평균: {np.mean(buffer_times):.2f}초')
        plt.xlabel('버퍼 시간 (초)')
        plt.ylabel('빈도')
        plt.title('종료 조건 버퍼 시간 분포')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/buffer_time_stats_ep{episode_count}.png')
        plt.close()


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
              color=PLOT_PARAMS['colors']['evader'], s=100, label='Start')
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], 
              color=PLOT_PARAMS['colors']['pursuer'], s=100, label='End')
    
    # Evader의 위치 (원점)
    ax.scatter(0, 0, 0, color='k', s=150, label='Evader (Chief)')
    
    # 제목에 정보 추가
    enhanced_title = title
    if nash_info is not None:
        enhanced_title += f" - Nash: {nash_info:.2f}"
    if safety_info is not None:
        enhanced_title += f" - Safety: {safety_info:.2f}"
    if buffer_time is not None:
        enhanced_title += f" - Buffer: {buffer_time:.1f}s"
    
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
    ax.legend()
    
    # 행동 벡터 표시 (선택적)
    if actions_e is not None and actions_p is not None:
        # 일부 행동만 표시 (가독성을 위해)
        step = max(1, len(states) // 20)
        
        for i in range(0, len(states), step):
            if np.any(actions_e[i] != 0):
                ax.quiver(
                    0, 0, 0,  # Evader는 원점
                    actions_e[i, 0], actions_e[i, 1], actions_e[i, 2],
                    color=PLOT_PARAMS['colors']['evader'], 
                    length=max_range/10, normalize=True, alpha=0.6
                )
            
            if np.any(actions_p[i] != 0):
                ax.quiver(
                    states[i, 0], states[i, 1], states[i, 2],
                    actions_p[i, 0], actions_p[i, 1], actions_p[i, 2],
                    color=PLOT_PARAMS['colors']['pursuer'], 
                    length=max_range/10, normalize=True, alpha=0.6
                )
    
    if save_path:
        plt.savefig(f"{save_path}_3d_trajectory.png")
    
    plt.close()
    
    # 거리 변화 그래프
    if len(states) > 1:
        plot_distance_evolution(states, save_path)


def plot_distance_evolution(states: np.ndarray, save_path: Optional[str] = None):
    """거리 변화 그래프"""
    distances = np.sqrt(np.sum(states[:, :3]**2, axis=1))
    
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(distances, color=PLOT_PARAMS['colors']['trajectory'])
    plt.axhline(y=1000, color=PLOT_PARAMS['colors']['pursuer'], 
               linestyle='--', label='Capture Distance (1000m)')
    plt.axhline(y=50000, color=PLOT_PARAMS['colors']['evader'], 
               linestyle='--', label='Evasion Distance (50000m)')
    plt.xlabel('Time Steps')
    plt.ylabel('Distance (m)')
    plt.title('Distance Between Pursuer and Evader')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(f"{save_path}_distance.png")
    
    plt.close()


def plot_test_results(results: List[Dict], 
                     zero_sum_metrics: Dict,
                     outcome_types: Dict,
                     save_dir: Optional[str] = None):
    """테스트 결과 시각화"""
    setup_matplotlib()
    
    success = [r['success'] for r in results]
    distances = [r['final_distance_m'] for r in results]
    evader_dvs = [r['evader_total_delta_v_ms'] for r in results]
    
    # 1. 최종 거리와 Delta-V 사용량 그래프
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 최종 거리
    colors = [PLOT_PARAMS['colors']['success'] if s else PLOT_PARAMS['colors']['failure'] 
             for s in success]
    axes[0].bar(range(len(distances)), distances, color=colors)
    axes[0].axhline(y=1000, color='r', linestyle='--', label='Capture Distance (1000m)')
    axes[0].set_xlabel('Test Run')
    axes[0].set_ylabel('Final Distance (m)')
    axes[0].set_title('Final Distance by Test Run')
    axes[0].legend()
    
    # Delta-V 사용량
    axes[1].bar(range(len(evader_dvs)), evader_dvs, color=colors)
    axes[1].set_xlabel('Test Run')
    axes[1].set_ylabel('Evader Total Delta-V (m/s)')
    axes[1].set_title('Evader Propellant Usage by Test Run')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/results_summary.png")
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
        plt.pie(filtered_counts, labels=filtered_labels, autopct='%1.1f%%', colors=colors)
        plt.title('Test Outcome Distribution')
        
        if save_dir:
            plt.savefig(f"{save_dir}/outcome_distribution_pie.png")
        plt.close()


def plot_zero_sum_analysis(zero_sum_metrics: Dict, success: List[bool], 
                          save_dir: Optional[str] = None):
    """Zero-Sum 게임 분석 그래프들"""
    episodes = range(1, len(zero_sum_metrics['evader_rewards']) + 1)
    
    # 1. 보상 비교 그래프
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(episodes, zero_sum_metrics['evader_rewards'], 
            color=PLOT_PARAMS['colors']['evader'], label='Evader Rewards')
    plt.plot(episodes, zero_sum_metrics['pursuer_rewards'], 
            color=PLOT_PARAMS['colors']['pursuer'], label='Pursuer Rewards')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Test Run')
    plt.ylabel('Average Reward')
    plt.title('Zero-Sum Game Rewards by Test Run')
    plt.grid(True)
    plt.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}/zero_sum_rewards.png")
    plt.close()
    
    # 2. Nash Equilibrium 메트릭
    if 'nash_metrics' in zero_sum_metrics and zero_sum_metrics['nash_metrics']:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        plt.plot(episodes, zero_sum_metrics['nash_metrics'], 'b-')
        plt.xlabel('Test Run')
        plt.ylabel('Nash Equilibrium Metric')
        plt.title('Nash Equilibrium Metric by Test Run')
        plt.grid(True)
        plt.ylim(0, 1)
        
        if save_dir:
            plt.savefig(f"{save_dir}/nash_metrics.png")
        plt.close()
    
    # 3. Zero-Sum 검증 그래프
    rewards_sum = [zero_sum_metrics['evader_rewards'][i] + zero_sum_metrics['pursuer_rewards'][i]
                  for i in range(len(zero_sum_metrics['evader_rewards']))]
    
    plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
    plt.plot(episodes, rewards_sum, 'k-')
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Zero-Sum')
    plt.xlabel('Test Run')
    plt.ylabel('Rewards Sum (Evader + Pursuer)')
    plt.title('Zero-Sum Game Verification')
    plt.grid(True)
    plt.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}/zero_sum_verification.png")
    plt.close()
    
    # 4. 안전도 점수 그래프
    if 'safety_scores' in zero_sum_metrics and zero_sum_metrics['safety_scores']:
        plt.figure(figsize=PLOT_PARAMS['figure_size_2d'])
        safety_scores = zero_sum_metrics['safety_scores']
        colors = ['g' if s > SAFETY_THRESHOLDS['permanent_evasion'] 
                 else ('b' if s > SAFETY_THRESHOLDS['conditional_evasion'] else 'r') 
                 for s in safety_scores]
        
        plt.bar(range(1, len(safety_scores) + 1), safety_scores, color=colors)
        plt.axhline(y=SAFETY_THRESHOLDS['permanent_evasion'], color='g', 
                   linestyle='--', label='Permanent Evasion Threshold')
        plt.axhline(y=SAFETY_THRESHOLDS['conditional_evasion'], color='b', 
                   linestyle='--', label='Conditional Evasion Threshold')
        plt.xlabel('Test Run')
        plt.ylabel('Safety Score')
        plt.title('Safety Score by Test Run')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)
        
        if save_dir:
            plt.savefig(f"{save_dir}/safety_scores.png")
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
           label='Evader', color=PLOT_PARAMS['colors']['evader'])
    plt.bar(x + width/2, normalized_pursuer, width, 
           label='Pursuer', color=PLOT_PARAMS['colors']['pursuer'])
    
    plt.xlabel('Orbital Elements')
    plt.ylabel('Values')
    plt.title('Orbital Elements Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}_orbital_elements.png")
    plt.close()


def create_summary_dashboard(training_stats: Dict, test_results: List[Dict],
                           save_dir: str):
    """종합 대시보드 생성"""
    setup_matplotlib()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training and Testing Summary Dashboard', fontsize=16)
    
    # 1. 성공률 추이
    if 'success_rates' in training_stats:
        axes[0, 0].plot(training_stats['success_rates'])
        axes[0, 0].set_title('Training Success Rate')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True)
    
    # 2. 테스트 결과 분포
    success_count = sum(1 for r in test_results if r['success'])
    failure_count = len(test_results) - success_count
    axes[0, 1].pie([success_count, failure_count], labels=['Success', 'Failure'], 
                   autopct='%1.1f%%', colors=['green', 'red'])
    axes[0, 1].set_title('Test Results Distribution')
    
    # 3. 평균 보상 추이
    if 'nash_metrics' in training_stats:
        axes[0, 2].plot(training_stats['nash_metrics'])
        axes[0, 2].set_title('Nash Equilibrium Convergence')
        axes[0, 2].set_xlabel('Evaluations')
        axes[0, 2].set_ylabel('Nash Metric')
        axes[0, 2].grid(True)
    
    # 4. Delta-V 사용량 분포
    delta_vs = [r['evader_total_delta_v_ms'] for r in test_results]
    axes[1, 0].hist(delta_vs, bins=10, alpha=0.7)
    axes[1, 0].set_title('Delta-V Usage Distribution')
    axes[1, 0].set_xlabel('Delta-V (m/s)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # 5. 최종 거리 분포
    distances = [r['final_distance_m'] for r in test_results]
    axes[1, 1].hist(distances, bins=10, alpha=0.7)
    axes[1, 1].axvline(x=1000, color='r', linestyle='--', label='Capture Threshold')
    axes[1, 1].set_title('Final Distance Distribution')
    axes[1, 1].set_xlabel('Distance (m)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. 성능 요약 텍스트
    axes[1, 2].axis('off')
    summary_text = f"""
    Training Episodes: {training_stats.get('episodes_completed', 'N/A')}
    Test Success Rate: {success_count/len(test_results):.1%}
    Avg Delta-V: {np.mean(delta_vs):.1f} m/s
    Avg Distance: {np.mean(distances):.0f} m
    Nash Metric: {training_stats.get('nash_metrics', [0])[-1]:.3f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 2].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/summary_dashboard.png")
    plt.close()


# 모듈 초기화 시 matplotlib 설정
setup_matplotlib()