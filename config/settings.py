"""
프로젝트 설정 관리 모듈
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from utils.constants import (
    ENV_PARAMS,
    BUFFER_PARAMS,
    TRAINING_PARAMS,
    PLOT_PARAMS,
    PATHS,
    DEFAULT_ORBIT_PARAMS,
)


@dataclass
class OrbitConfig:
    """궤도 설정"""

    a: float = DEFAULT_ORBIT_PARAMS["a"]
    e: float = DEFAULT_ORBIT_PARAMS["e"]
    i: float = DEFAULT_ORBIT_PARAMS["i"]
    RAAN: float = DEFAULT_ORBIT_PARAMS["RAAN"]
    omega: float = DEFAULT_ORBIT_PARAMS["omega"]
    M0: float = DEFAULT_ORBIT_PARAMS["M0"]


@dataclass
class EnvironmentConfig:
    """환경 설정"""

    dt: float = ENV_PARAMS["dt"]
    k: int = ENV_PARAMS["k"]
    delta_v_emax: float = ENV_PARAMS["delta_v_emax"]
    delta_v_pmax: float = ENV_PARAMS["delta_v_pmax"]
    sigma_noise: float = ENV_PARAMS["sigma_noise"]
    sensor_noise_sigma: float = ENV_PARAMS["sensor_noise_sigma"]
    sensor_range: float = ENV_PARAMS["sensor_range"]
    capture_distance: float = ENV_PARAMS["capture_distance"]
    evasion_distance: float = ENV_PARAMS["evasion_distance"]
    c: float = ENV_PARAMS["c"]
    max_steps: int = ENV_PARAMS["max_steps"]
    max_delta_v_budget: float = ENV_PARAMS["max_delta_v_budget"]
    max_initial_separation: float = ENV_PARAMS["max_initial_separation"]
    use_rk4: bool = ENV_PARAMS["use_rk4"]
    use_gastm: bool = ENV_PARAMS["use_gastm"]
    # 궤도 모드 주기 사용 여부
    use_orbit_cycles: bool = ENV_PARAMS["use_orbit_cycles"]

    # 버퍼 설정
    capture_buffer_steps: int = BUFFER_PARAMS["capture_buffer_steps"]
    evasion_buffer_steps: int = BUFFER_PARAMS["evasion_buffer_steps"]
    safety_buffer_steps: int = BUFFER_PARAMS["safety_buffer_steps"]


@dataclass
class TrainingConfig:
    """학습 설정"""

    total_timesteps: int = TRAINING_PARAMS["total_timesteps"]
    nash_total_timesteps: int = TRAINING_PARAMS["nash_total_timesteps"]
    learning_rate: float = TRAINING_PARAMS["learning_rate"]
    buffer_size: int = TRAINING_PARAMS["buffer_size"]
    batch_size: int = TRAINING_PARAMS["batch_size"]
    tau: float = TRAINING_PARAMS["tau"]
    gamma: float = TRAINING_PARAMS["gamma"]
    net_arch: list = field(default_factory=lambda: TRAINING_PARAMS["net_arch"])
    save_freq: int = TRAINING_PARAMS["save_freq"]
    n_envs: int = TRAINING_PARAMS["n_envs"]

    # 장치 설정
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_gpu: bool = field(default_factory=lambda: torch.cuda.is_available())

    # 로깅 설정
    log_interval: int = 100
    eval_frequency: int = 10
    verbose: int = 1


@dataclass
class VisualizationConfig:
    """시각화 설정"""

    figure_size_3d: tuple = PLOT_PARAMS["figure_size_3d"]
    figure_size_2d: tuple = PLOT_PARAMS["figure_size_2d"]
    dpi: int = PLOT_PARAMS["dpi"]
    colors: Dict[str, str] = field(default_factory=lambda: PLOT_PARAMS["colors"])
    save_plots: bool = True
    show_plots: bool = False


@dataclass
class PathConfig:
    """경로 설정"""

    logs: str = PATHS["logs"]
    models: str = PATHS["models"]
    results: str = PATHS["results"]
    tensorboard: str = PATHS["tensorboard"]
    checkpoints: str = PATHS["checkpoints"]
    plots: str = PATHS["plots"]
    tests: str = PATHS["tests"]

    def create_directories(self):
        """필요한 디렉토리 생성"""
        for path in [
            self.logs,
            self.models,
            self.results,
            self.tensorboard,
            self.checkpoints,
            self.plots,
            self.tests,
        ]:
            os.makedirs(path, exist_ok=True)


@dataclass
class ProjectConfig:
    """프로젝트 전체 설정"""

    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # 실험 설정
    experiment_name: str = "default_experiment"
    random_seed: Optional[int] = None
    debug_mode: bool = False

    def __post_init__(self):
        """초기화 후 처리"""
        # 디렉토리 생성
        self.paths.create_directories()

        # GPU 설정 확인
        if self.training.use_gpu and not torch.cuda.is_available():
            print("Warning: GPU가 요청되었지만 사용할 수 없습니다. CPU를 사용합니다.")
            self.training.device = "cpu"
            self.training.use_gpu = False

        # 디버그 모드에서는 스텝 수 감소
        if self.debug_mode:
            self.training.total_timesteps = min(1000, self.training.total_timesteps)
            self.training.nash_total_timesteps = min(
                5000, self.training.nash_total_timesteps
            )
            self.environment.max_steps = min(100, self.environment.max_steps)
            self.training.n_envs = 1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProjectConfig":
        """딕셔너리로부터 설정 생성"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "orbit": self.orbit.__dict__,
            "environment": self.environment.__dict__,
            "training": self.training.__dict__,
            "visualization": self.visualization.__dict__,
            "paths": self.paths.__dict__,
            "experiment_name": self.experiment_name,
            "random_seed": self.random_seed,
            "debug_mode": self.debug_mode,
        }

    def save_to_file(self, filepath: str):
        """설정을 파일로 저장"""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ProjectConfig":
        """파일로부터 설정 로드"""
        import json

        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# 기본 설정 인스턴스
default_config = ProjectConfig()


def get_config(
    experiment_name: Optional[str] = None,
    debug_mode: bool = False,
    custom_config: Optional[Dict[str, Any]] = None,
) -> ProjectConfig:
    """설정 인스턴스 생성 및 반환"""

    config = ProjectConfig()

    if experiment_name:
        config.experiment_name = experiment_name

    if debug_mode:
        config.debug_mode = True

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return config


def setup_environment_variables():
    """환경 변수 설정"""
    # CUDA 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # NumPy 설정
    import numpy as np

    np.seterr(divide="ignore", invalid="ignore")

    # Matplotlib 설정
    import matplotlib

    matplotlib.use("Agg")  # GUI 없는 환경에서 사용


# 초기화 시 환경 변수 설정
setup_environment_variables()
