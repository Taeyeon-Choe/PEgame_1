# utils/optimization.py
"""
성능 최적화 유틸리티
"""

import torch
import numpy as np
from typing import Optional, Union, List
import gc
import warnings


class GPUOptimizer:
    """GPU 메모리 및 성능 최적화"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = False
        
    def enable_mixed_precision(self):
        """Mixed precision 학습 활성화"""
        if self.device == 'cuda':
            try:
                from torch.cuda.amp import autocast
                self.mixed_precision = True
                print("Mixed precision 학습 활성화")
            except ImportError:
                warnings.warn("Mixed precision을 사용할 수 없습니다.")
                
    def clear_cache(self):
        """GPU 캐시 정리"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            
    def optimize_memory(self):
        """메모리 최적화 설정"""
        if self.device == 'cuda':
            # 메모리 단편화 감소
            torch.cuda.set_per_process_memory_fraction(0.9)
            # 동적 그래프 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
    def get_memory_stats(self) -> dict:
        """GPU 메모리 통계"""
        if self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'free': (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1024**3      # GB
            }
        return {}


class BatchProcessor:
    """배치 처리 최적화"""
    
    @staticmethod
    def process_parallel_envs(observations: List[np.ndarray], 
                            batch_size: Optional[int] = None) -> torch.Tensor:
        """병렬 환경 관측값 배치 처리"""
        # 리스트를 numpy 배열로 효율적 변환
        if isinstance(observations, list):
            observations = np.stack(observations, axis=0)
        
        # PyTorch 텐서로 변환 (GPU 가능시 자동)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = torch.from_numpy(observations).float().to(device)
        
        # 배치 크기 조정
        if batch_size and len(tensor) > batch_size:
            # 큰 배치를 작은 배치로 분할
            batches = torch.split(tensor, batch_size)
            return batches
        
        return tensor
    
    @staticmethod
    def vectorized_step(envs, actions: Union[np.ndarray, torch.Tensor]) -> tuple:
        """벡터화된 환경 스텝"""
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # 병렬 스텝 실행
        observations, rewards, dones, infos = envs.step(actions)
        
        return observations, rewards, dones, infos


class MemoryPool:
    """메모리 재사용 풀"""
    
    def __init__(self, shape: tuple, dtype: np.dtype = np.float32, pool_size: int = 10):
        self.shape = shape
        self.dtype = dtype
        self.pool = []
        self.pool_size = pool_size
        
        # 초기 버퍼 생성
        for _ in range(pool_size):
            self.pool.append(np.empty(shape, dtype=dtype))
            
    def get_buffer(self) -> np.ndarray:
        """재사용 가능한 버퍼 반환"""
        if self.pool:
            return self.pool.pop()
        else:
            return np.empty(self.shape, dtype=self.dtype)
            
    def return_buffer(self, buffer: np.ndarray):
        """버퍼 반환"""
        if len(self.pool) < self.pool_size:
            self.pool.append(buffer)


class InPlaceOperations:
    """In-place 연산 최적화"""
    
    @staticmethod
    def normalize_inplace(arr: np.ndarray, min_val: float = -1.0, 
                         max_val: float = 1.0) -> np.ndarray:
        """In-place 정규화"""
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        
        if arr_max - arr_min > 1e-10:
            # In-place 연산
            arr -= arr_min
            arr /= (arr_max - arr_min)
            arr *= (max_val - min_val)
            arr += min_val
        else:
            arr.fill(0.0)
            
        return arr
    
    @staticmethod
    def clip_inplace(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """In-place 클리핑"""
        np.clip(arr, min_val, max_val, out=arr)
        return arr
    
    @staticmethod
    def add_noise_inplace(arr: np.ndarray, noise_scale: float) -> np.ndarray:
        """In-place 노이즈 추가"""
        noise = np.random.normal(0, noise_scale, arr.shape)
        arr += noise
        return arr


def optimize_environment(env_class):
    """환경 클래스 최적화 데코레이터"""
    class OptimizedEnv(env_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # 메모리 풀 초기화
            self.obs_pool = MemoryPool((9,), np.float32)
            self.state_pool = MemoryPool((6,), np.float64)
            
            # GPU 최적화
            self.gpu_optimizer = GPUOptimizer()
            if self.config.training.use_gpu:
                self.gpu_optimizer.optimize_memory()
        
        def reset(self):
            # 메모리 재사용
            if hasattr(self, 'state'):
                self.state_pool.return_buffer(self.state)
                
            obs = super().reset()
            return obs
        
        def step(self, action):
            # 배치 처리 준비
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
            obs, reward, done, info = super().step(action)
            
            # 주기적 메모리 정리
            if self.step_count % 1000 == 0 and self.config.training.use_gpu:
                self.gpu_optimizer.clear_cache()
                
            return obs, reward, done, info
    
    return OptimizedEnv


# 전역 최적화 설정
def setup_global_optimizations():
    """전역 최적화 설정"""
    # NumPy 스레드 설정
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # PyTorch 최적화
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    # Numba 설정
    from numba import config
    config.THREADING_LAYER = 'omp'
    
    # 경고 비활성화 (프로덕션)
    if not os.environ.get('DEBUG'):
        warnings.filterwarnings('ignore')
        np.seterr(all='ignore')


# 사용 예제
if __name__ == "__main__":
    # 전역 최적화 적용
    setup_global_optimizations()
    
    # GPU 최적화
    gpu_opt = GPUOptimizer()
    gpu_opt.enable_mixed_precision()
    gpu_opt.optimize_memory()
    
    # 메모리 통계 출력
    stats = gpu_opt.get_memory_stats()
    if stats:
        print(f"GPU 메모리 - 할당: {stats['allocated']:.1f}GB, "
              f"예약: {stats['reserved']:.1f}GB, "
              f"여유: {stats['free']:.1f}GB")
