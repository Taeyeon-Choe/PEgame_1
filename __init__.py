 # 루트 __init__.py
"""
위성 추격-회피 게임 패키지

실제 궤도 역학을 기반으로 한 위성 추격-회피 게임의 강화학습 프레임워크
"""

__version__ = "1.0.0"
__author__ = "Satellite Game Theory Research Team"
__email__ = "contact@example.com"

# 주요 모듈 import
from . import config
from . import orbital_mechanics  
from . import environment
from . import training
from . import analysis
from . import utils

# 편의를 위한 주요 클래스 직접 import
from .environment import PursuitEvasionEnv
from .training import SACTrainer, NashEquilibriumTrainer
from .analysis import ModelEvaluator
from .config import get_config, ProjectConfig

__all__ = [
    'config', 'orbital_mechanics', 'environment', 
    'training', 'analysis', 'utils',
    'PursuitEvasionEnv', 'SACTrainer', 'NashEquilibriumTrainer',
    'ModelEvaluator', 'get_config', 'ProjectConfig'
]
