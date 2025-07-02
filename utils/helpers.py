"""
공통 유틸리티 함수들
"""

import os
import json
import pickle
import datetime
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import sys


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        log_level: 로그 레벨 ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: 로그 파일 경로 (None이면 콘솔만)
        
    Returns:
        설정된 로거
    """
    logger = logging.getLogger("satellite_game")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    JSON 파일 저장
    
    Args:
        data: 저장할 데이터
        filepath: 파일 경로
        indent: 들여쓰기 레벨
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    JSON 파일 로드
    
    Args:
        filepath: 파일 경로
        
    Returns:
        로드된 데이터
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """
    피클 파일 저장
    
    Args:
        data: 저장할 데이터
        filepath: 파일 경로
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    피클 파일 로드
    
    Args:
        filepath: 파일 경로
        
    Returns:
        로드된 데이터
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_dir(dirpath: str):
    """
    디렉토리 생성 (존재하지 않는 경우)
    
    Args:
        dirpath: 디렉토리 경로
    """
    os.makedirs(dirpath, exist_ok=True)


def get_timestamp(format_str: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    현재 시간 타임스탬프 반환
    
    Args:
        format_str: 시간 포맷 문자열
        
    Returns:
        포맷된 타임스탬프
    """
    return datetime.datetime.now().strftime(format_str)


def set_random_seed(seed: int):
    """
    모든 랜덤 시드 설정
    
    Args:
        seed: 시드 값
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_gpu_availability() -> Dict[str, Union[bool, str, int]]:
    """
    GPU 사용 가능성 확인
    
    Returns:
        GPU 정보 딕셔너리
    """
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'memory_allocated': torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0,
        'memory_reserved': torch.cuda.memory_reserved(0) if torch.cuda.is_available() else 0
    }
    
    return gpu_info


def print_gpu_info():
    """GPU 정보 출력"""
    info = check_gpu_availability()
    
    print("=== GPU 정보 ===")
    print(f"GPU 사용 가능: {info['available']}")
    
    if info['available']:
        print(f"GPU 개수: {info['device_count']}")
        print(f"현재 GPU: {info['current_device']}")
        print(f"GPU 이름: {info['device_name']}")
        print(f"할당된 메모리: {info['memory_allocated'] / 1024**2:.1f} MB")
        print(f"예약된 메모리: {info['memory_reserved'] / 1024**2:.1f} MB")
    else:
        print("CUDA를 사용할 수 없습니다.")


def normalize_array(arr: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """
    배열 정규화
    
    Args:
        arr: 입력 배열
        min_val: 최소값
        max_val: 최대값
        
    Returns:
        정규화된 배열
    """
    arr_min, arr_max = np.min(arr), np.max(arr)
    
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (max_val - min_val) + min_val


def denormalize_array(arr: np.ndarray, original_min: float, original_max: float,
                     norm_min: float = -1.0, norm_max: float = 1.0) -> np.ndarray:
    """
    정규화 해제
    
    Args:
        arr: 정규화된 배열
        original_min: 원래 최소값
        original_max: 원래 최대값
        norm_min: 정규화 최소값
        norm_max: 정규화 최대값
        
    Returns:
        원래 스케일로 복원된 배열
    """
    normalized = (arr - norm_min) / (norm_max - norm_min)
    return normalized * (original_max - original_min) + original_min


def moving_average(data: Union[List[float], np.ndarray], window_size: int) -> np.ndarray:
    """
    이동 평균 계산
    
    Args:
        data: 입력 데이터
        window_size: 윈도우 크기
        
    Returns:
        이동 평균 배열
    """
    if len(data) < window_size:
        return np.array(data)
    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def exponential_moving_average(data: Union[List[float], np.ndarray], alpha: float = 0.1) -> np.ndarray:
    """
    지수 이동 평균 계산
    
    Args:
        data: 입력 데이터
        alpha: 스무딩 팩터 (0 < alpha <= 1)
        
    Returns:
        지수 이동 평균 배열
    """
    data = np.array(data)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema


def angle_difference(angle1: float, angle2: float) -> float:
    """
    두 각도 간의 최소 차이 계산 (라디안)
    
    Args:
        angle1: 첫 번째 각도
        angle2: 두 번째 각도
        
    Returns:
        최소 각도 차이 (-π ~ π)
    """
    diff = angle2 - angle1
    return ((diff + np.pi) % (2 * np.pi)) - np.pi


def wrap_angle(angle: float) -> float:
    """
    각도를 [-π, π] 범위로 래핑
    
    Args:
        angle: 입력 각도 (라디안)
        
    Returns:
        래핑된 각도
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def rotation_matrix_2d(angle: float) -> np.ndarray:
    """
    2D 회전 행렬 생성
    
    Args:
        angle: 회전 각도 (라디안)
        
    Returns:
        2x2 회전 행렬
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def rotation_matrix_3d(axis: str, angle: float) -> np.ndarray:
    """
    3D 회전 행렬 생성
    
    Args:
        axis: 회전축 ('x', 'y', 'z')
        angle: 회전 각도 (라디안)
        
    Returns:
        3x3 회전 행렬
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', or 'z'.")


def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray],
               default: float = 0.0) -> Union[float, np.ndarray]:
    """
    안전한 나눗셈 (0으로 나누기 방지)
    
    Args:
        numerator: 분자
        denominator: 분모
        default: 0으로 나눌 때 반환할 기본값
        
    Returns:
        나눗셈 결과
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(np.abs(denominator) > 1e-12, numerator / denominator, default)
    else:
        result = numerator / denominator if abs(denominator) > 1e-12 else default
    
    return result


def clamp(value: Union[float, np.ndarray], min_val: float, max_val: float) -> Union[float, np.ndarray]:
    """
    값을 지정된 범위로 제한
    
    Args:
        value: 입력 값
        min_val: 최소값
        max_val: 최대값
        
    Returns:
        제한된 값
    """
    return np.clip(value, min_val, max_val)


def format_time(seconds: float) -> str:
    """
    초를 읽기 쉬운 시간 형식으로 변환
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        포맷된 시간 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def format_bytes(size_bytes: int) -> str:
    """
    바이트를 읽기 쉬운 크기로 변환
    
    Args:
        size_bytes: 바이트 크기
        
    Returns:
        포맷된 크기 문자열
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def calculate_statistics(data: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    기본 통계 계산
    
    Args:
        data: 입력 데이터
        
    Returns:
        통계 딕셔너리
    """
    data = np.array(data)
    
    return {
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'range': np.max(data) - np.min(data)
    }


def print_statistics(data: Union[List[float], np.ndarray], name: str = "Data"):
    """
    통계 정보 출력
    
    Args:
        data: 입력 데이터
        name: 데이터 이름
    """
    stats = calculate_statistics(data)
    
    print(f"\n=== {name} 통계 ===")
    print(f"개수: {stats['count']}")
    print(f"평균: {stats['mean']:.4f}")
    print(f"표준편차: {stats['std']:.4f}")
    print(f"최솟값: {stats['min']:.4f}")
    print(f"최댓값: {stats['max']:.4f}")
    print(f"중앙값: {stats['median']:.4f}")
    print(f"25% 분위수: {stats['q25']:.4f}")
    print(f"75% 분위수: {stats['q75']:.4f}")
    print(f"범위: {stats['range']:.4f}")


def create_grid(x_range: Tuple[float, float], y_range: Tuple[float, float],
               x_points: int = 100, y_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D 그리드 생성
    
    Args:
        x_range: x 범위 (min, max)
        y_range: y 범위 (min, max)
        x_points: x 축 포인트 수
        y_points: y 축 포인트 수
        
    Returns:
        X, Y 메시그리드
    """
    x = np.linspace(x_range[0], x_range[1], x_points)
    y = np.linspace(y_range[0], y_range[1], y_points)
    return np.meshgrid(x, y)


def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    1D 선형 보간
    
    Args:
        x: 원래 x 좌표
        y: 원래 y 값
        x_new: 새로운 x 좌표
        
    Returns:
        보간된 y 값
    """
    return np.interp(x_new, x, y)


class Timer:
    """간단한 타이머 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """타이머 시작"""
        self.start_time = datetime.datetime.now()
        return self
    
    def stop(self):
        """타이머 중지"""
        self.end_time = datetime.datetime.now()
        return self
    
    def elapsed(self) -> float:
        """경과 시간 반환 (초)"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or datetime.datetime.now()
        return (end - self.start_time).total_seconds()
    
    def __enter__(self):
        """컨텍스트 매니저 시작"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop()


def benchmark_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    함수 실행 시간 측정
    
    Args:
        func: 측정할 함수
        *args: 함수 인자
        **kwargs: 함수 키워드 인자
        
    Returns:
        (함수 결과, 실행 시간)
    """
    with Timer() as timer:
        result = func(*args, **kwargs)
    
    return result, timer.elapsed()


def progress_bar(current: int, total: int, width: int = 50, prefix: str = "Progress") -> str:
    """
    간단한 프로그레스 바 생성
    
    Args:
        current: 현재 진행도
        total: 전체 크기
        width: 바 너비
        prefix: 접두사
        
    Returns:
        프로그레스 바 문자열
    """
    percent = current / max(total, 1) * 100
    filled = int(width * current / max(total, 1))
    bar = '█' * filled + '░' * (width - filled)
    
    return f"\r{prefix}: |{bar}| {percent:6.2f}% ({current}/{total})"