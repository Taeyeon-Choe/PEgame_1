"""
성능 메트릭 계산 모듈
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import signal
from utils.constants import SAFETY_THRESHOLDS


def calculate_performance_metrics(states: np.ndarray, 
                                actions_e: np.ndarray,
                                actions_p: np.ndarray,
                                rewards: Dict[str, List[float]],
                                info: Dict[str, Any]) -> Dict[str, float]:
    """
    종합 성능 메트릭 계산
    
    Args:
        states: 상태 궤적 [N, 6]
        actions_e: 회피자 액션 [N, 3]  
        actions_p: 추격자 액션 [N, 3]
        rewards: 보상 히스토리
        info: 에피소드 정보
        
    Returns:
        성능 메트릭 딕셔너리
    """
    metrics = {}
    
    # 기본 거리 메트릭
    metrics.update(calculate_distance_metrics(states))
    
    # 효율성 메트릭
    metrics.update(calculate_efficiency_metrics(actions_e, actions_p))
    
    # 제어 품질 메트릭
    metrics.update(calculate_control_quality_metrics(actions_e, actions_p))
    
    # 보상 기반 메트릭
    metrics.update(calculate_reward_metrics(rewards))
    
    # 안전성 메트릭
    metrics.update(calculate_safety_metrics(states, info))
    
    # Zero-Sum 게임 메트릭
    metrics.update(calculate_zero_sum_metrics(rewards, info))
    
    return metrics


def calculate_distance_metrics(states: np.ndarray) -> Dict[str, float]:
    """거리 관련 메트릭 계산"""
    positions = states[:, :3]
    distances = np.linalg.norm(positions, axis=1)
    
    metrics = {
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'mean_distance': np.mean(distances),
        'final_distance': distances[-1],
        'initial_distance': distances[0],
        'distance_variance': np.var(distances),
        'distance_range': np.max(distances) - np.min(distances)
    }
    
    # 거리 변화 추세
    if len(distances) > 1:
        distance_trend = np.polyfit(range(len(distances)), distances, 1)[0]
        metrics['distance_trend'] = distance_trend  # 양수면 증가, 음수면 감소
    else:
        metrics['distance_trend'] = 0.0
    
    # 임계 거리 교차 횟수
    capture_threshold = 1000.0
    evasion_threshold = 50000.0
    
    metrics['capture_threshold_crossings'] = count_threshold_crossings(distances, capture_threshold)
    metrics['evasion_threshold_crossings'] = count_threshold_crossings(distances, evasion_threshold)
    
    # 위험 구간에서의 시간 비율
    danger_zone_steps = np.sum(distances < capture_threshold * 5)  # 5km 이내를 위험 구간으로 정의
    metrics['danger_zone_time_ratio'] = danger_zone_steps / len(distances)
    
    return metrics


def calculate_efficiency_metrics(actions_e: np.ndarray, actions_p: np.ndarray) -> Dict[str, float]:
    """효율성 메트릭 계산"""
    # Delta-V 사용량
    evader_delta_vs = np.linalg.norm(actions_e, axis=1)
    pursuer_delta_vs = np.linalg.norm(actions_p, axis=1)
    
    metrics = {
        'total_evader_delta_v': np.sum(evader_delta_vs),
        'total_pursuer_delta_v': np.sum(pursuer_delta_vs),
        'mean_evader_delta_v': np.mean(evader_delta_vs),
        'mean_pursuer_delta_v': np.mean(pursuer_delta_vs),
        'max_evader_delta_v': np.max(evader_delta_vs),
        'max_pursuer_delta_v': np.max(pursuer_delta_vs),
        'evader_fuel_efficiency': np.sum(evader_delta_vs > 0.1),  # 유의미한 기동 횟수
        'pursuer_fuel_efficiency': np.sum(pursuer_delta_vs > 0.1)
    }
    
    # 연료 효율성 비율
    if metrics['total_pursuer_delta_v'] > 0:
        metrics['fuel_usage_ratio'] = metrics['total_evader_delta_v'] / metrics['total_pursuer_delta_v']
    else:
        metrics['fuel_usage_ratio'] = float('inf')
    
    # 기동 빈도
    total_steps = len(actions_e)
    metrics['evader_maneuver_frequency'] = metrics['evader_fuel_efficiency'] / total_steps
    metrics['pursuer_maneuver_frequency'] = metrics['pursuer_fuel_efficiency'] / total_steps
    
    return metrics


def calculate_control_quality_metrics(actions_e: np.ndarray, actions_p: np.ndarray) -> Dict[str, float]:
    """제어 품질 메트릭 계산"""
    metrics = {}
    
    # 액션 변동성 (제어 부드러움)
    if len(actions_e) > 1:
        evader_action_changes = np.diff(actions_e, axis=0)
        evader_smoothness = np.mean(np.linalg.norm(evader_action_changes, axis=1))
        metrics['evader_control_smoothness'] = evader_smoothness
        
        # 제어 일관성 (방향 변화)
        evader_direction_changes = calculate_direction_changes(actions_e)
        metrics['evader_direction_consistency'] = 1.0 / (1.0 + evader_direction_changes)
    else:
        metrics['evader_control_smoothness'] = 0.0
        metrics['evader_direction_consistency'] = 1.0
    
    if len(actions_p) > 1:
        pursuer_action_changes = np.diff(actions_p, axis=0)
        pursuer_smoothness = np.mean(np.linalg.norm(pursuer_action_changes, axis=1))
        metrics['pursuer_control_smoothness'] = pursuer_smoothness
        
        pursuer_direction_changes = calculate_direction_changes(actions_p)
        metrics['pursuer_direction_consistency'] = 1.0 / (1.0 + pursuer_direction_changes)
    else:
        metrics['pursuer_control_smoothness'] = 0.0
        metrics['pursuer_direction_consistency'] = 1.0
    
    # 제어 효율성 (작은 액션으로 큰 효과)
    metrics.update(calculate_control_efficiency(actions_e, actions_p))
    
    return metrics


def calculate_direction_changes(actions: np.ndarray) -> float:
    """액션 방향 변화 횟수 계산"""
    if len(actions) < 2:
        return 0.0
    
    # 0이 아닌 액션들만 고려
    nonzero_actions = actions[np.linalg.norm(actions, axis=1) > 1e-6]
    
    if len(nonzero_actions) < 2:
        return 0.0
    
    # 정규화된 방향 벡터들 계산
    normalized_actions = nonzero_actions / np.linalg.norm(nonzero_actions, axis=1, keepdims=True)
    
    # 연속된 방향 간의 각도 계산
    direction_changes = 0
    for i in range(len(normalized_actions) - 1):
        cos_angle = np.clip(np.dot(normalized_actions[i], normalized_actions[i+1]), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle > np.pi / 2:  # 90도 이상 변화를 방향 변화로 간주
            direction_changes += 1
    
    return direction_changes / max(len(nonzero_actions) - 1, 1)


def calculate_control_efficiency(actions_e: np.ndarray, actions_p: np.ndarray) -> Dict[str, float]:
    """제어 효율성 계산"""
    metrics = {}
    
    # 액션 크기 분포 분석
    evader_action_norms = np.linalg.norm(actions_e, axis=1)
    pursuer_action_norms = np.linalg.norm(actions_p, axis=1)
    
    # 제어 집중도 (큰 액션 vs 작은 액션 비율)
    evader_large_actions = np.sum(evader_action_norms > np.mean(evader_action_norms))
    pursuer_large_actions = np.sum(pursuer_action_norms > np.mean(pursuer_action_norms))
    
    total_steps = len(actions_e)
    metrics['evader_control_concentration'] = evader_large_actions / total_steps
    metrics['pursuer_control_concentration'] = pursuer_large_actions / total_steps
    
    # 제어 변동성
    metrics['evader_control_variability'] = np.std(evader_action_norms)
    metrics['pursuer_control_variability'] = np.std(pursuer_action_norms)
    
    return metrics


def calculate_reward_metrics(rewards: Dict[str, List[float]]) -> Dict[str, float]:
    """보상 기반 메트릭 계산"""
    evader_rewards = rewards['evader']
    pursuer_rewards = rewards['pursuer']
    
    metrics = {
        'total_evader_reward': sum(evader_rewards),
        'total_pursuer_reward': sum(pursuer_rewards),
        'mean_evader_reward': np.mean(evader_rewards),
        'mean_pursuer_reward': np.mean(pursuer_rewards),
        'evader_reward_variance': np.var(evader_rewards),
        'pursuer_reward_variance': np.var(pursuer_rewards),
        'cumulative_reward_difference': sum(evader_rewards) - sum(pursuer_rewards)
    }
    
    # 보상 추세 분석
    if len(evader_rewards) > 1:
        evader_trend = np.polyfit(range(len(evader_rewards)), evader_rewards, 1)[0]
        pursuer_trend = np.polyfit(range(len(pursuer_rewards)), pursuer_rewards, 1)[0]
        
        metrics['evader_reward_trend'] = evader_trend
        metrics['pursuer_reward_trend'] = pursuer_trend
    
    # 보상 안정성
    metrics['reward_stability'] = 1.0 / (1.0 + metrics['evader_reward_variance'])
    
    return metrics


def calculate_safety_metrics(states: np.ndarray, info: Dict[str, Any]) -> Dict[str, float]:
    """안전성 메트릭 계산"""
    metrics = {}
    
    # 기본 안전성 점수
    safety_score = info.get('safety_score', 0.0)
    metrics['safety_score'] = safety_score
    
    # 안전성 카테고리 분류
    if safety_score >= SAFETY_THRESHOLDS['permanent_evasion']:
        metrics['safety_category'] = 3  # 안정적인 회피
    elif safety_score >= SAFETY_THRESHOLDS.get('evaded', SAFETY_THRESHOLDS['permanent_evasion']):
        metrics['safety_category'] = 2  # 회피 유지
    else:
        metrics['safety_category'] = 1  # 위험
    
    # 거리 기반 안전성
    distances = np.linalg.norm(states[:, :3], axis=1)
    capture_distance = 1000.0
    
    # 안전 마진
    min_safety_margin = np.min(distances) - capture_distance
    mean_safety_margin = np.mean(distances) - capture_distance
    
    metrics['min_safety_margin'] = min_safety_margin
    metrics['mean_safety_margin'] = mean_safety_margin
    metrics['safety_margin_normalized'] = min_safety_margin / capture_distance
    
    # 위험도 분석
    risk_levels = calculate_risk_levels(distances, capture_distance)
    metrics.update(risk_levels)
    
    return metrics


def calculate_risk_levels(distances: np.ndarray, capture_distance: float) -> Dict[str, float]:
    """위험도 레벨 분석"""
    total_steps = len(distances)
    
    # 위험 구간 정의
    critical_zone = capture_distance * 1.5    # 1.5km
    danger_zone = capture_distance * 3.0      # 3km  
    warning_zone = capture_distance * 5.0     # 5km
    
    critical_steps = np.sum(distances < critical_zone)
    danger_steps = np.sum(distances < danger_zone)
    warning_steps = np.sum(distances < warning_zone)
    
    return {
        'critical_risk_ratio': critical_steps / total_steps,
        'high_risk_ratio': danger_steps / total_steps,
        'medium_risk_ratio': warning_steps / total_steps,
        'low_risk_ratio': (total_steps - warning_steps) / total_steps
    }


def calculate_zero_sum_metrics(rewards: Dict[str, List[float]], info: Dict[str, Any]) -> Dict[str, float]:
    """Zero-Sum 게임 메트릭 계산"""
    evader_rewards = rewards['evader']
    pursuer_rewards = rewards['pursuer']
    
    metrics = {}
    
    # Zero-Sum 특성 검증
    reward_sum = sum(evader_rewards) + sum(pursuer_rewards)
    metrics['zero_sum_violation'] = abs(reward_sum)
    metrics['zero_sum_compliance'] = 1.0 / (1.0 + abs(reward_sum))
    
    # 게임 이론적 성능
    evader_advantage = sum(evader_rewards) / (sum(evader_rewards) + sum(pursuer_rewards) + 1e-10)
    metrics['evader_advantage'] = evader_advantage
    metrics['game_balance'] = 1.0 - abs(evader_advantage - 0.5) * 2  # 0.5에 가까울수록 균형
    
    # 전략적 다양성
    if len(evader_rewards) > 10:
        evader_diversity = calculate_strategy_diversity(evader_rewards)
        pursuer_diversity = calculate_strategy_diversity(pursuer_rewards)
        
        metrics['evader_strategy_diversity'] = evader_diversity
        metrics['pursuer_strategy_diversity'] = pursuer_diversity
        metrics['overall_strategy_diversity'] = (evader_diversity + pursuer_diversity) / 2
    
    return metrics


def calculate_strategy_diversity(rewards: List[float], window_size: int = 10) -> float:
    """전략 다양성 계산"""
    if len(rewards) < window_size:
        return 0.0
    
    # 슬라이딩 윈도우로 보상 패턴 분석
    patterns = []
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i + window_size]
        pattern_signature = (np.mean(window), np.std(window), np.max(window) - np.min(window))
        patterns.append(pattern_signature)
    
    # 패턴 다양성 계산 (유클리드 거리 기반)
    if len(patterns) < 2:
        return 0.0
    
    total_distance = 0.0
    comparisons = 0
    
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            distance = np.linalg.norm(np.array(patterns[i]) - np.array(patterns[j]))
            total_distance += distance
            comparisons += 1
    
    return total_distance / comparisons if comparisons > 0 else 0.0


def analyze_trajectory_quality(states: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    """궤적 품질 분석"""
    metrics = {}
    
    # 궤적 부드러움 (속도 변화율)
    velocities = states[:, 3:6]
    if len(velocities) > 1:
        velocity_changes = np.diff(velocities, axis=0)
        trajectory_smoothness = np.mean(np.linalg.norm(velocity_changes, axis=1))
        metrics['trajectory_smoothness'] = trajectory_smoothness
    else:
        metrics['trajectory_smoothness'] = 0.0
    
    # 궤적 효율성 (직선 거리 대비 실제 이동 거리)
    positions = states[:, :3]
    if len(positions) > 1:
        total_path_length = calculate_path_length(positions)
        direct_distance = np.linalg.norm(positions[-1] - positions[0])
        
        if direct_distance > 1e-6:
            path_efficiency = direct_distance / total_path_length
        else:
            path_efficiency = 1.0
        
        metrics['path_efficiency'] = path_efficiency
        metrics['total_path_length'] = total_path_length
    
    # 궤적 복잡성 (방향 변화 빈도)
    trajectory_complexity = calculate_trajectory_complexity(positions)
    metrics['trajectory_complexity'] = trajectory_complexity
    
    # 액션-결과 일관성
    action_effectiveness = calculate_action_effectiveness(actions, positions)
    metrics['action_effectiveness'] = action_effectiveness
    
    return metrics


def calculate_path_length(positions: np.ndarray) -> float:
    """경로 길이 계산"""
    if len(positions) < 2:
        return 0.0
    
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.sum(distances)


def calculate_trajectory_complexity(positions: np.ndarray) -> float:
    """궤적 복잡성 계산 (곡률 기반)"""
    if len(positions) < 3:
        return 0.0
    
    # 곡률 계산을 위한 3점 기반 방법
    curvatures = []
    for i in range(1, len(positions) - 1):
        p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
        
        # 벡터 계산
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 곡률 계산 (외적의 크기 / 속도의 세제곱)
        cross_product = np.cross(v1, v2)
        if isinstance(cross_product, np.ndarray):
            cross_magnitude = np.linalg.norm(cross_product)
        else:
            cross_magnitude = abs(cross_product)
        
        v1_norm = np.linalg.norm(v1)
        if v1_norm > 1e-10:
            curvature = cross_magnitude / (v1_norm ** 3)
            curvatures.append(curvature)
    
    return np.mean(curvatures) if curvatures else 0.0


def calculate_action_effectiveness(actions: np.ndarray, positions: np.ndarray) -> float:
    """액션 효과성 계산"""
    if len(actions) < 2 or len(positions) < 2:
        return 0.0
    
    # 액션 크기와 위치 변화의 상관관계
    action_magnitudes = np.linalg.norm(actions[:-1], axis=1)  # 마지막 액션 제외
    position_changes = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # 0이 아닌 액션들에 대해서만 계산
    nonzero_indices = action_magnitudes > 1e-6
    
    if np.sum(nonzero_indices) < 2:
        return 0.0
    
    action_subset = action_magnitudes[nonzero_indices]
    position_subset = position_changes[nonzero_indices]
    
    # 상관계수 계산
    if len(action_subset) > 1:
        correlation = np.corrcoef(action_subset, position_subset)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    else:
        return 0.0


def count_threshold_crossings(values: np.ndarray, threshold: float) -> int:
    """임계값 교차 횟수 계산"""
    if len(values) < 2:
        return 0
    
    crossings = 0
    above_threshold = values[0] > threshold
    
    for i in range(1, len(values)):
        current_above = values[i] > threshold
        if current_above != above_threshold:
            crossings += 1
            above_threshold = current_above
    
    return crossings


def calculate_frequency_domain_metrics(signal_data: np.ndarray, dt: float = 30.0) -> Dict[str, float]:
    """주파수 도메인 메트릭 계산"""
    if len(signal_data) < 4:  # FFT를 위한 최소 데이터 포인트
        return {'dominant_frequency': 0.0, 'spectral_entropy': 0.0}
    
    # FFT 계산
    freqs, psd = signal.periodogram(signal_data, fs=1/dt)
    
    # 주요 주파수 찾기
    dominant_freq_idx = np.argmax(psd[1:]) + 1  # DC 성분 제외
    dominant_frequency = freqs[dominant_freq_idx]
    
    # 스펙트럼 엔트로피 (주파수 분포의 복잡성)
    psd_normalized = psd / np.sum(psd)
    psd_normalized = psd_normalized[psd_normalized > 0]  # 0인 값 제거
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized))
    
    return {
        'dominant_frequency': dominant_frequency,
        'spectral_entropy': spectral_entropy
    }


def calculate_statistical_summary(data: np.ndarray) -> Dict[str, float]:
    """통계적 요약 계산"""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'skewness': calculate_skewness(data),
        'kurtosis': calculate_kurtosis(data)
    }


def calculate_skewness(data: np.ndarray) -> float:
    """왜도 계산"""
    if len(data) < 2:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std < 1e-10:
        return 0.0
    
    skewness = np.mean(((data - mean) / std) ** 3)
    return skewness


def calculate_kurtosis(data: np.ndarray) -> float:
    """첨도 계산"""
    if len(data) < 2:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std < 1e-10:
        return 0.0
    
    kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # 정규분포는 0
    return kurtosis
