# 위성 추격-회피 게임 강화학습 프레임워크

실제 궤도 역학을 기반으로 한 위성 추격-회피 게임의 강화학습 프레임워크입니다. SAC(Soft Actor-Critic) 알고리즘과 Nash Equilibrium 학습을 지원합니다.

## 주요 특징

### 물리 시뮬레이션
- **실제 궤도 역학**: Keplerian 궤도 요소 기반 정확한 위성 운동
- **J2 섭동 효과**: 지구 편평도에 의한 궤도 섭동 모델링
- **비선형 상대 동역학**: 고정밀 상대 운동 방정식

### 강화학습
- **SAC 알고리즘**: 연속 액션 공간에서의 효율적 학습
- **Nash Equilibrium**: Zero-Sum 게임에서의 균형점 탐색
- **멀티프로세싱**: 병렬 환경 실행으로 학습 가속화

### 최적화 기법
- **Numba JIT 컴파일**: 핵심 동역학 함수 가속화
- **RK4 적분기**: 고정밀 궤도 전파
- **상태 캐싱**: 반복 계산 최소화
- **GPU 지원**: CUDA 가속 학습

## 시스템 요구사항

### 필수 요구사항
- Python 3.8 이상 (3.10 또는 3.11 권장)
- 8GB 이상 RAM
- 10GB 이상 디스크 공간

### 권장 사양
- Python 3.10 ~ 3.11 (최신 안정 버전)
- 16GB 이상 RAM
- NVIDIA GPU (CUDA 11.0+)
- Ubuntu 20.04 / Windows 10 / macOS 11+

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/Taeyeon-Choe/PEgame_1.git
cd PEgame_1
```

### 2. 가상환경 생성 (권장)
```bash
# conda 사용시 (Python 3.10 권장)
conda create -n satellite-game python=3.10
conda activate satellite-game

# venv 사용시
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

### 3. 의존성 설치

#### 기본 설치
```bash
pip install -r requirements.txt
```

#### 개발 환경 설치
```bash
#pip install -e .[dev]
```

#### GPU 지원 설치 (CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. 설치 확인
```bash
python -c "import orbital_mechanics, environment, training; print('설치 성공!')"
```

## 빠른 시작

### 1. 기본 학습 실행
```bash
# 가장 간단한 실행
python main.py --mode train_standard --timesteps 10000

# 또는 Python 스크립트로
python examples/quick_start.py
```

### 2. 학습된 모델 평가
```bash
# 모델 평가
python main.py --mode evaluate --model-path models/sac_final.zip --n-tests 20

# 데모 실행
python main.py --mode demo --model-path models/sac_final.zip
```

### 3. Jupyter Notebook으로 시작
```bash
jupyter lab
# notebooks/getting_started.ipynb 열기
```

## 상세 사용법

### 1단계: 환경 이해하기

#### 환경 생성 및 테스트
```python
from config.settings import get_config
from environment.pursuit_evasion_env import PursuitEvasionEnv

# 설정 로드
config = get_config(experiment_name="my_experiment")

# 환경 생성
env = PursuitEvasionEnv(config)

# 환경 정보 확인
print(f"액션 공간: {env.action_space}")
print(f"관측 공간: {env.observation_space}")

# 랜덤 에피소드 실행
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print(f"에피소드 종료: {info['outcome']}")
        break
```

초기화 시 회피자의 궤도는 매 에피소드마다 반장축을 **7000~8500 km** 사이,
이심률을 **0~0.5** 사이에서 균일분포로 샘플링합니다. 단, 근지점 고도가 최소
**200 km** 이상이 되도록 조건을 두어 지구 내부에서 시작하지 않도록 합니다.
추격자는 이 궤도를 기준으로 5 km 이내 위치가 되도록 궤도 요소를 선택하므로,
매번 다른 초기 조건으로 학습을 진행할 수 있습니다.

#### Gim-Alfriend STM 환경 사용

기본 환경 대신 Gim-Alfriend STM(State Transition Matrix)을 활용하여
상대 궤도를 빠르게 전파할 수 있습니다.

```python
from environment.pursuit_evasion_env_ga_stm import PursuitEvasionEnvGASTM

env = PursuitEvasionEnvGASTM(config, use_gastm=True)
```

`use_gastm=False`로 설정하면 기존 비선형 수치 적분 방식으로 동작합니다.

### 2단계: 기본 SAC 학습

#### 단일 환경 학습 (테스트용)
```python
from training.trainer import create_trainer

# 간단한 학습 설정
config = get_config(debug_mode=True)
config.training.total_timesteps = 5000
config.training.n_envs = 1  # 단일 환경

env = PursuitEvasionEnv(config)
trainer = create_trainer(env, config)
trainer.setup_model()

# 학습 실행
trainer.train()

# 모델 저장
trainer.save_model("models/test_model.zip")
```

#### 병렬 환경 학습 (실제 학습)
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 실제 학습 설정
config = get_config(experiment_name="full_training")
config.training.total_timesteps = 100000
config.training.n_envs = 8  # 8개 병렬 환경

# 병렬 환경 생성
def make_env():
    return PursuitEvasionEnv(config)

env = SubprocVecEnv([make_env for _ in range(config.training.n_envs)])

# 트레이너 생성 및 학습
trainer = create_trainer(env, config)
trainer.setup_model()
trainer.train()

# 결과 저장
trainer.save_model("models/trained_model.zip")
```

### 3단계: Nash Equilibrium 학습

```python
from training.nash_equilibrium import train_nash_equilibrium_model

# Nash 학습 설정
config = get_config(experiment_name="nash_training")
config.training.nash_total_timesteps = 150000

env = PursuitEvasionEnv(config)

# Nash Equilibrium 학습
nash_trainer = train_nash_equilibrium_model(env, config)

# 결과 확인
stats = nash_trainer.get_nash_training_stats()
print(f"최종 Nash 메트릭: {stats['final_nash_metric']:.4f}")
```

### 4단계: 모델 평가 및 분석

```python
from stable_baselines3 import SAC
from analysis.evaluator import create_evaluator

# 모델 로드
model = SAC.load("models/trained_model.zip", env=env)

# 평가자 생성
evaluator = create_evaluator(model, env, config)

# 다중 시나리오 평가
results = evaluator.evaluate_multiple_scenarios(n_tests=50)

# 결과 출력
print(f"성공률: {results['summary']['success_rate']:.1f}%")
print(f"평균 최종 거리: {results['summary']['avg_final_distance']:.0f}m")
print(f"평균 Delta-V: {results['summary']['avg_evader_delta_v']:.1f}m/s")

# 데모 실행 (시각화 포함)
demo_result = evaluator.run_demonstration()
```

### 5단계: 커스텀 설정

```python
# 사용자 정의 환경 파라미터
config = get_config()
config.environment.capture_distance = 500.0  # 포획 거리 감소
config.environment.delta_v_emax = 12.0      # 회피자 성능 증가
config.environment.max_steps = 20000        # 더 긴 에피소드

# 사용자 정의 학습 파라미터
config.training.learning_rate = 0.0005
config.training.batch_size = 1024
config.training.buffer_size = 200000

# 설정 저장
config.save_to_file("configs/custom_config.json")

# 나중에 로드
loaded_config = ProjectConfig.load_from_file("configs/custom_config.json")
```

## 프로젝트 구조

```
satellite-pursuit-evasion/
├── config/                 # 설정 관리
│   ├── __init__.py
│   └── settings.py        # 프로젝트 설정
├── environment/           # 강화학습 환경
│   ├── __init__.py
│   └── pursuit_evasion_env.py
├── orbital_mechanics/     # 궤도 역학
│   ├── __init__.py
│   ├── orbit.py          # 궤도 클래스
│   ├── dynamics.py       # 동역학 함수
│   └── coordinate_transforms.py
├── training/             # 학습 모듈
│   ├── __init__.py
│   ├── trainer.py        # SAC 트레이너
│   ├── nash_equilibrium.py
│   └── callbacks.py
├── analysis/             # 분석 도구
│   ├── __init__.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── visualization.py
├── utils/                # 유틸리티
│   ├── __init__.py
│   ├── constants.py
│   └── helpers.py
├── examples/             # 예제 스크립트
├── tests/               # 테스트
├── notebooks/           # Jupyter 노트북
└── main.py             # 메인 실행 파일
```

## 문제 해결

### 일반적인 오류 및 해결방법

#### 1. ImportError: No module named 'xxx'
```bash
# 패키지 재설치
pip install -e .
```

#### 2. CUDA out of memory
```python
# 배치 크기 감소
config.training.batch_size = 256
```

#### 3. 멀티프로세싱 오류 (Windows)
```python
# 단일 환경 사용
config.training.n_envs = 1
```

#### 4. Numba 관련 오류
```bash
# Numba 캐시 초기화
python -c "import numba; numba.config.DISABLE_JIT = True"
```

### 디버깅 팁

#### 상세 로그 활성화
```python
config = get_config(debug_mode=True)
config.training.verbose = 2
```

#### 환경 동작 확인
```python
# 단계별 상태 출력
env = PursuitEvasionEnv(config)
obs, _ = env.reset()
print(f"초기 상태: {env.state}")
print(f"초기 거리: {np.linalg.norm(env.state[:3]):.0f}m")
```

## 성능 최적화

### 학습 속도 향상
1. **GPU 사용**: CUDA 가능시 자동 활성화
2. **병렬 환경**: CPU 코어수만큼 환경 실행
3. **배치 크기 조정**: GPU 메모리에 맞게 설정

### 메모리 사용량 감소
1. **리플레이 버퍼 크기 조정**
```python
config.training.buffer_size = 50000  # 기본값: 100000
```

2. **모델 체크포인트 주기 조정**
```python
config.training.save_freq = 20000  # 기본값: 10000
```

### 수치 안정성 향상
1. **RK4 적분기 사용**
```python
config.environment.use_rk4 = True
```

2. **시간 간격 조정**
```python
config.environment.dt = 15.0  # 더 작은 시간 간격
```

3. **GA STM 전파기 사용**
```python
config.environment.use_gastm = True
```

## Docker 사용법

### 빌드 및 실행
```bash
# CPU 버전
docker-compose up satellite-game

# GPU 버전
docker-compose up satellite-game-gpu

# Jupyter 서버
docker-compose up jupyter
```

## 기여 방법

1. Fork 후 새 브랜치 생성
2. 변경사항 구현 및 테스트
3. 코드 스타일 확인 (`black`, `flake8`)
4. Pull Request 제출

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 문의

- 이슈: GitHub Issues
- 이메일: contact@example.com
- 문서: [프로젝트 위키](https://github.com/yourusername/satellite-pursuit-evasion/wiki)
