"""
물리 상수 및 환경 설정 상수 정의
"""

import numpy as np
import multiprocessing

# 물리 상수
MU_EARTH = 3.986e14  # 지구 중력 상수 (m^3/s^2)
R_EARTH = 6378e3  # 지구 반지름 (m)
J2_EARTH = 1.08263e-3  # 지구 J2 계수

# 기본 궤도 파라미터
DEFAULT_ORBIT_PARAMS = {
    "a": 7500e3,  # 반장축 (m)
    "e": 0.1,  # 이심률
    "i": 0.1,  # 경사각 (rad)
    "RAAN": 0.0,  # 승교점 적경 (rad)
    "omega": 0.0,  # 근점 편각 (rad)
    "M0": 0.0,  # 초기 평균 근점이각 (rad)
}

# 기준 궤도 무작위화 범위
ORBIT_BOUNDS = {
    "perigee_altitude_min": 400e3,  # 근지점 고도 최소값 (m)
    "perigee_altitude_max": 500e3,  # 근지점 고도 최대값 (m)
    "apogee_altitude_max": 2500e3,  # 원지점 고도 최대값 (m)
    "inclination_min": np.deg2rad(45.0),  # 경사각 최소값 (rad)
    "inclination_max": np.deg2rad(60.0),  # 경사각 최대값 (rad)
}

# 환경 파라미터
ENV_PARAMS = {
    "dt": 30.0,  # 시간 간격 (s)
    "k": 2,  # 추격자 행동 주기
    "delta_v_emax": 0.15,  # 회피자 최대 delta-v (m/s)
    "delta_v_pmax": 0.25,  # 추격자 최대 delta-v (m/s)
    "sigma_noise": 0.05,  # 추격자 노이즈 (m/s)
    "sensor_noise_sigma": 100,  # 센서 노이즈 (m, m/s)
    "sensor_range": 5e3,  # 센서 최대 범위 (m)
    "capture_distance": 1000.0,  # 포착 거리 (m)
    "evasion_distance": 10e3,  # 회피 거리 (m)
    "c": 0.01,  # 제어 비용 계수
    "max_steps": 1000,  # 최대 스텝 수
    "max_delta_v_budget": 250.0,  # 최대 추진제 예산 (m/s)
    "max_initial_separation": 3e3,  # 최대 초기 분리 거리 (m)
    "use_rk4": True,
    "use_gastm": True,
    # === Reward / LQ Zero-Sum options ===
    "reward_mode": "original",  # "original" | "lq_zero_sum" | "lq_zero_sum_shaped"
    "lqr_RE_diag": [1e-2, 1e-2, 1e-2],
    "reward_gamma": 1.0,
    "shape_alphas": [0.01, 0.005],
    "pursuer_policy": "heuristic",  # "heuristic" or "tvlqr"
    "lqr_horizon": 10,
    "lqr_Q_diag": [1.0, 1.0, 1.0, 0.05, 0.05, 0.05],
    "lqr_QN_diag": [5.0, 5.0, 5.0, 0.1, 0.1, 0.1],
    "lqr_R_diag": [1e-2, 1e-2, 1e-2],
}

# 버퍼 시간 설정
BUFFER_PARAMS = {
    "capture_buffer_steps": 3,  # 포획 상태가 유지되어야 하는 스텝 수
    "evasion_buffer_steps": 5,  # 회피 상태가 유지되어야 하는 스텝 수
    "safety_buffer_steps": 10,  # 안전도 점수가 유지되어야 하는 스텝 수
}

# 학습 파라미터
TRAINING_PARAMS = {
    "total_timesteps": 100000,
    "nash_total_timesteps": 50000,
    "learning_rate": 0.0001,
    "buffer_size": 100000,
    "batch_size": 512,
    "tau": 0.005,
    "gamma": 0.98,
    "net_arch": [512, 512, 512],
    "save_freq": 10000,
    "n_envs": min(multiprocessing.cpu_count(), 14),
}

# 시각화 설정
PLOT_PARAMS = {
    "figure_size_3d": (15, 10),
    "figure_size_2d": (12, 6),
    "dpi": 100,
    "colors": {
        "evader": "green",
        "pursuer": "red",
        "trajectory": "blue",
        "success": "g",
        "failure": "r",
    },
}

# 안전도 평가 임계값
SAFETY_THRESHOLDS = {
    "permanent_evasion": 0.7,  # 영구 회피 임계값
    "conditional_evasion": 0.4,  # 조건부 회피 임계값
    "temporary_evasion": 0.0,  # 임시 회피 임계값
}

# 결과 분석 설정
ANALYSIS_PARAMS = {
    "test_scenarios": 10,
    "demo_scenarios": 3,
    "window_size": 100,  # 이동 평균 윈도우 크기
    "eval_frequency": 10,  # Nash 평형 평가 주기
}

# 수치 안정성 파라미터
NUMERICAL_STABILITY = {
    "min_value": 1e-10,  # 최소값 (0으로 나누기 방지)
    "rtol": 1e-6,  # 상대 허용 오차 (ODE 솔버)
    "atol": 1e-6,  # 절대 허용 오차 (ODE 솔버)
}

# 파일 경로 설정
PATHS = {
    "logs": "./logs",
    "models": "./models",
    "results": "./results",
    "tensorboard": "./tensorboard_logs",
    "checkpoints": "./model_checkpoints",
    "plots": "./training_plots",
    "tests": "./test_results",
}

# 각도 변환
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
