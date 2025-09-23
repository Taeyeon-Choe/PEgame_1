"""
SAC 모델 트레이너 클래스
"""

import os
import csv
import json
import pickle
import datetime
from pathlib import Path

import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.pursuit_evasion_env import PursuitEvasionEnv
from environment.pursuit_evasion_env_ga_stm import PursuitEvasionEnvGASTM
from stable_baselines3.common.logger import configure
from typing import Optional, List, Dict, Any

from config.settings import ProjectConfig
from training.callbacks import (
    EvasionTrackingCallback,
    PerformanceCallback,
    EarlyStoppingCallback,
    EphemerisLoggerCallback,
    DetailedAnalysisCallback,
)
from utils.system_info import patch_sb3_system_info
from utils.matlab_templates import render_matlab_script

# 시스템 정보에 추가 패키지 버전 기록
patch_sb3_system_info()


class SACTrainer:
    """SAC 알고리즘 트레이너"""

    def __init__(self, env, config: Optional[ProjectConfig] = None, experiment_name=None,
                 log_dir=None, resume: bool = False):
        """
        트레이너 초기화

        Args:
            env: 학습 환경
            config: 프로젝트 설정
            experiment_name: 실험 이름
            log_dir: 로그 디렉토리 경로
            resume: 기존 로그 및 분석 데이터를 이어서 사용할지 여부
        """
        if config is None:
            from config.settings import default_config
            config = default_config

        self.env = env
        self.config = config
        self.experiment_name = experiment_name or config.experiment_name
        self.training_config = config.training
        self.paths_config = config.paths
        self.resume_mode = resume

        # 모델 및 로깅 설정
        self.model = None
        self.logger = None
        self.callbacks = []
        self._eval_env_ref = None

        # 학습 기록
        self.training_history = {}

        # 로그 디렉토리 설정
        if log_dir:
            self.log_dir = log_dir
            self._ensure_log_structure(self.log_dir)
        else:
            self.log_dir = self._setup_log_directory()
        
        # 설정 저장
        self.config.save_to_file(f"{self.log_dir}/config.json")
        print(f"설정 저장: {self.log_dir}/config.json")

        # 이어 학습을 위한 기존 데이터 로드
        self.resume_state = self._load_resume_state() if self.resume_mode else {}

    def _setup_log_directory(self) -> str:
        """로그 디렉토리 설정 및 생성"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"{self.paths_config.logs}/{self.experiment_name}_{timestamp}"
        
        self._ensure_log_structure(log_dir)
        
        self._write_matlab_training_script(log_dir)

        print(f"로그 디렉토리 생성 완료: {log_dir}")
        return log_dir

    def _ensure_log_structure(self, log_dir: str) -> None:
        """학습 로그 폴더 기본 구조 보장"""
        os.makedirs(log_dir, exist_ok=True)

        subdirs = ["plots", "models", "tensorboard", "data", "eval"]
        for subdir in subdirs:
            os.makedirs(f"{log_dir}/{subdir}", exist_ok=True)

    def _load_resume_state(self) -> Dict[str, Any]:
        """기존 학습 로그에서 이어학습에 필요한 데이터를 불러옴"""
        state: Dict[str, Any] = {}
        plots_dir = Path(self.log_dir) / "plots"

        shared_state: Dict[str, Any] = {}
        if plots_dir.exists():
            resume_file = plots_dir / "resume_state.json"
            shared_state = self._safe_json_load(resume_file) or {}
            shared_state = self._populate_shared_state_from_files(shared_state, plots_dir)
            if shared_state:
                state["shared"] = shared_state

        analysis_state = self._load_analysis_state()
        if analysis_state:
            state["analysis"] = analysis_state

        return state

    def _populate_shared_state_from_files(self, current: Dict[str, Any], plots_dir: Path) -> Dict[str, Any]:
        """기존 플롯/데이터 파일을 기반으로 상태 정보를 채움"""
        state = dict(current) if current else {}

        # 성공률 시퀀스
        if not state.get("success_rates"):
            success_rates = self._load_csv_series(plots_dir / "success_rate_data.csv", "success_rate")
            if success_rates:
                state["success_rates"] = success_rates
                state.setdefault("episode_count", len(success_rates))

        # 보상 데이터
        if not state.get("evader_rewards") or not state.get("pursuer_rewards"):
            reward_series = self._load_csv_multi(
                plots_dir / "rewards_data.csv",
                {
                    "evader_reward": float,
                    "pursuer_reward": float,
                },
            )
            evader_rewards = reward_series.get("evader_reward", [])
            pursuer_rewards = reward_series.get("pursuer_reward", [])
            if evader_rewards:
                state.setdefault("evader_rewards", evader_rewards)
            if pursuer_rewards:
                state.setdefault("pursuer_rewards", pursuer_rewards)

        # Nash 메트릭
        if not state.get("nash_metrics"):
            nash_metrics = self._load_csv_series(plots_dir / "nash_metrics.csv", "nash_metric")
            if nash_metrics:
                state["nash_metrics"] = nash_metrics

        # 결과 분포
        if not state.get("outcome_counts"):
            outcome_distribution = self._safe_json_load(plots_dir / "outcome_distribution.json")
            if isinstance(outcome_distribution, dict):
                normalized = {}
                for key, value in outcome_distribution.items():
                    try:
                        normalized[self._normalize_outcome_key(key)] = int(value)
                    except (TypeError, ValueError):
                        continue
                if normalized:
                    state["outcome_counts"] = normalized

        # 세부 통계 (회피 기록)
        stats = self._safe_json_load(plots_dir / "evasion_stats.json")
        if isinstance(stats, dict):
            state.setdefault("episode_count", stats.get("total_episodes", state.get("episode_count")))
            state.setdefault("captures", stats.get("captures"))
            state.setdefault("evasions", stats.get("evasions"))
            state.setdefault("permanent_evasions", stats.get("permanent_evasions"))
            state.setdefault("conditional_evasions", stats.get("conditional_evasions"))
            state.setdefault("temporary_evasions", stats.get("temporary_evasions"))
            state.setdefault("fuel_depleted", stats.get("fuel_depleted"))
            state.setdefault("max_steps", stats.get("max_steps"))
            if not state.get("evader_delta_vs"):
                state["evader_delta_vs"] = stats.get("evader_delta_v", [])
            if not state.get("episodes_info"):
                state["episodes_info"] = stats.get("episodes_info", [])

        # 에피소드 정보 기반 추가 데이터 계산
        episodes_info = state.get("episodes_info", [])
        if episodes_info and not state.get("buffer_times"):
            buffer_times = [
                float(info.get("buffer_time", 0))
                for info in episodes_info
                if info.get("buffer_time")
            ]
            if buffer_times:
                state["buffer_times"] = buffer_times

        if episodes_info and not state.get("recent_successes"):
            recent_successes = [1 if info.get("success") else 0 for info in episodes_info][-100:]
            if recent_successes:
                state["recent_successes"] = recent_successes

        if state.get("evader_delta_vs") and not state.get("recent_delta_v"):
            recent_delta = state["evader_delta_vs"][-100:]
            if recent_delta:
                state["recent_delta_v"] = recent_delta

        # outcome_counts가 없다면 episodes_info로 추정
        if not state.get("outcome_counts") and episodes_info:
            counts = {
                "captured": 0,
                "permanent_evasion": 0,
                "conditional_evasion": 0,
                "fuel_depleted": 0,
                "max_steps": 0,
            }
            for info in episodes_info:
                outcome = self._normalize_outcome_key(info.get("outcome", "unknown"))
                if outcome in counts:
                    counts[outcome] += 1
            state["outcome_counts"] = counts

        # 총 에피소드 수 보완
        if not state.get("episode_count") and state.get("success_rates"):
            state["episode_count"] = len(state["success_rates"])

        if not state.get("episode_count") and episodes_info:
            state["episode_count"] = episodes_info[-1].get("episode")

        return state

    def _load_csv_series(self, path: Path, column: str) -> List[float]:
        values: List[float] = []
        if not path.exists():
            return values
        try:
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    value = row.get(column)
                    if value in (None, ''):
                        continue
                    try:
                        values.append(float(value))
                    except ValueError:
                        try:
                            values.append(int(value))
                        except ValueError:
                            continue
        except Exception as exc:
            print(f"CSV 로드 실패: {path} ({exc})")
        return values

    def _load_csv_multi(self, path: Path, columns: Dict[str, Any]) -> Dict[str, List[float]]:
        data: Dict[str, List[float]] = {key: [] for key in columns.keys()}
        if not path.exists():
            return data
        try:
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for key, caster in columns.items():
                        value = row.get(key)
                        if value in (None, ''):
                            continue
                        try:
                            data[key].append(caster(value))
                        except (TypeError, ValueError):
                            continue
        except Exception as exc:
            print(f"CSV 로드 실패: {path} ({exc})")
        return data

    def _safe_json_load(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"JSON 로드 실패: {path} ({exc})")
            return None
        except Exception as exc:
            print(f"JSON 로드 실패: {path} ({exc})")
            return None

    @staticmethod
    def _normalize_outcome_key(key: str) -> str:
        normalized = key.lower().replace('-', ' ').replace('_', ' ').strip()
        mapping = {
            'captured': 'captured',
            'capture': 'captured',
            'permanent evasion': 'permanent_evasion',
            'permanent_evasion': 'permanent_evasion',
            'conditional evasion': 'conditional_evasion',
            'conditional_evasion': 'conditional_evasion',
            'temporary evasion': 'temporary_evasion',
            'temporary_evasion': 'temporary_evasion',
            'fuel depleted': 'fuel_depleted',
            'fuel_depleted': 'fuel_depleted',
            'max steps': 'max_steps',
            'max_steps': 'max_steps',
        }
        return mapping.get(normalized, normalized.replace(' ', '_'))

    def _load_analysis_state(self) -> Dict[str, Any]:
        analysis_dir = Path(self.log_dir) / "analysis"
        if not analysis_dir.exists():
            return {}

        pkl_path = analysis_dir / "analysis_data.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"분석 데이터 로드 실패: {pkl_path} ({exc})")

        json_path = analysis_dir / "analysis_data.json"
        data = self._safe_json_load(json_path)
        return data or {}

    def _write_matlab_training_script(self, log_dir: str) -> None:
        """학습 로그 폴더에 MATLAB 분석 스크립트를 생성."""
        run_name = Path(log_dir).name
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        destination = Path(log_dir) / "Analysis_training.m"

        try:
            render_matlab_script(
                "analysis_training_template.m",
                destination,
                {
                    "RUN_NAME": run_name,
                    "GENERATED_AT": generated_at,
                },
            )
        except FileNotFoundError as exc:
            print(f"[경고] MATLAB 학습 분석 스크립트 생성 실패: {exc}")

    def _make_eval_env(self) -> DummyVecEnv:
        """평가용 DummyVecEnv 생성"""

        def _init_env():
            if getattr(self.config.environment, "use_gastm", False):
                return PursuitEvasionEnvGASTM(self.config)
            return PursuitEvasionEnv(self.config)

        return DummyVecEnv([_init_env])

    def setup_model(self, model_params: Optional[Dict[str, Any]] = None, policy_kwargs=None):
        """SAC 모델 설정"""
        if model_params is None:
            model_params = {}
        
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": self.training_config.net_arch,
                "activation_fn": torch.nn.ReLU,
                "use_sde": False,
                "log_std_init": -3,
                "normalize_images": False,
            }

        # 기본 파라미터 설정
        default_params = {
            "learning_rate": self.training_config.learning_rate,
            "buffer_size": self.training_config.buffer_size,
            "learning_starts": 1000,
            "batch_size": self.training_config.batch_size,
            "tau": self.training_config.tau,
            "gamma": self.training_config.gamma,
            "train_freq": 1,
            "gradient_steps": 1,
            "action_noise": None,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto",
            "use_sde": False,
            "use_sde_at_warmup": False,
            "tensorboard_log": f"{self.log_dir}/tensorboard",
            "policy_kwargs": policy_kwargs,
            "verbose": self.training_config.verbose,
            "seed": self.config.random_seed,
            "device": self.training_config.device,
        }

        # 사용자 파라미터로 덮어쓰기
        default_params.update(model_params)

        print(f"\nSAC 모델 초기화")
        print(f"  - Device: {self.training_config.device}")
        print(f"  - 정책 네트워크: {policy_kwargs['net_arch']}")
        print(f"  - 학습률: {self.training_config.learning_rate}")
        print(f"  - 배치 크기: {self.training_config.batch_size}")
        print(f"  - 버퍼 크기: {self.training_config.buffer_size}")
        print(f"  - Verbose 레벨: {self.training_config.verbose}")
        
        self.model = SAC("MlpPolicy", self.env, **default_params)

        # 로거 설정
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(self.logger)

        return self.model

    def setup_callbacks(self, custom_callbacks: Optional[List] = None):
        """콜백 설정"""
        self.callbacks = []

        # 1. 성능 추적 콜백 (plots 저장 포함)
        shared_resume = self.resume_state.get("shared") if self.resume_mode else None

        performance_callback = PerformanceCallback(
            log_dir=self.log_dir,
            plot_freq=100,  # 100 에피소드마다 플롯 저장
            verbose=1,
            resume=self.resume_mode,
            resume_data=shared_resume,
        )
        self.callbacks.append(performance_callback)

        # 2. 회피 성공률 추적
        evasion_callback = EvasionTrackingCallback(
            verbose=1, 
            log_dir=f"{self.log_dir}/plots",
            resume=self.resume_mode,
            resume_data=shared_resume,
        )
        self.callbacks.append(evasion_callback)

        # 3. 최종 에피소드 궤적 기록
        ephemeris_callback = EphemerisLoggerCallback(log_dir=self.log_dir)
        self.callbacks.append(ephemeris_callback)

        # 4. 상세 분석 자동화 (MATLAB 등 후처리용 데이터 포함)
        detailed_callback = DetailedAnalysisCallback(
            plot_freq=5000,
            episode_plot_freq=200,
            save_dir=f"{self.log_dir}/analysis",
            verbose=1,
            resume_data=self.resume_state.get("analysis") if self.resume_mode else None,
        )
        self.callbacks.append(detailed_callback)

        # 5. 평가 콜백 (best 모델 저장)
        eval_env = self._make_eval_env()
        eval_freq = max(1, getattr(self.training_config, "save_freq", 10000))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.log_dir}/models",
            log_path=f"{self.log_dir}/eval",
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
        )
        self.callbacks.append(eval_callback)
        self._eval_env_ref = eval_env

        # 6. 조기 종료 (선택적)
        if hasattr(self.training_config, "early_stopping") and self.training_config.early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                target_success_rate=0.8, 
                patience=10000, 
                verbose=1
            )
            self.callbacks.append(early_stopping_callback)

        # 사용자 정의 콜백 추가
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)

        return CallbackList(self.callbacks)

    def train(self, total_timesteps: Optional[int] = None, callback=None, 
              reset_num_timesteps: bool = True, tb_log_name: str = "sac_run"):
        """
        모델 학습 실행

        Args:
            total_timesteps: 총 학습 스텝 수
            callback: 콜백 리스트
            reset_num_timesteps: 타임스텝 카운터 리셋 여부
            tb_log_name: Tensorboard 로그 이름

        Returns:
            self (메서드 체이닝용)
        """
        if self.model is None:
            raise ValueError("모델이 설정되지 않았습니다. setup_model()을 먼저 호출하세요.")

        if total_timesteps is None:
            total_timesteps = self.training_config.total_timesteps

        # 콜백 설정
        if callback is None:
            callback = self.setup_callbacks()

        print(f"\n{'='*50}")
        print(f"SAC 모델 학습 시작")
        print(f"{'='*50}")
        print(f"총 타임스텝: {total_timesteps:,}")
        print(f"저장 주기: {self.training_config.save_freq:,}")
        print(f"로그 디렉토리: {self.log_dir}")
        print(f"{'='*50}\n")

        # 학습 실행
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=self.training_config.log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True  # 진행 표시줄 활성화
        )

        # 평가용 환경 정리
        if self._eval_env_ref is not None:
            try:
                self._eval_env_ref.close()
            finally:
                self._eval_env_ref = None

        print(f"\n{'='*50}")
        print(f"학습 완료!")
        print(f"{'='*50}")

        # 최종 모델 저장
        final_model_path = f"{self.log_dir}/models/sac_final.zip"
        self.save_model(final_model_path)
        
        # 학습 통계 저장
        self._save_training_history()
        
        # 콜백에서 최종 통계 저장
        for cb in self.callbacks:
            if isinstance(cb, PerformanceCallback):
                cb.save_final_stats()
                break

        return self

    def save_model(self, save_path: str, save_replay_buffer: bool = True):
        """모델 저장"""
        if self.model is None:
            print("저장할 모델이 없습니다.")
            return

        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 모델 저장
        self.model.save(save_path)
        print(f"모델 저장됨: {save_path}")

        # 리플레이 버퍼 저장 (선택적)
        if save_replay_buffer and hasattr(self.model, "replay_buffer"):
            buffer_path = save_path.replace(".zip", "_replay_buffer.pkl")
            self.model.save_replay_buffer(buffer_path)
            print(f"리플레이 버퍼 저장됨: {buffer_path}")

    def load_model(self, load_path: str, load_replay_buffer: bool = True):
        """모델 로드"""
        try:
            self.model = SAC.load(load_path, env=self.env, device=self.training_config.device)
            print(f"모델 로드됨: {load_path}")

            # 새 로그 디렉토리에 맞춰 로거와 텐서보드 경로 재설정
            self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
            self.model.set_logger(self.logger)
            if hasattr(self.model, "tensorboard_log"):
                self.model.tensorboard_log = f"{self.log_dir}/tensorboard"

            # 리플레이 버퍼 로드 (선택적)
            if load_replay_buffer:
                buffer_path = load_path.replace(".zip", "_replay_buffer.pkl")
                if os.path.exists(buffer_path):
                    self.model.load_replay_buffer(buffer_path)
                    print(f"리플레이 버퍼 로드됨: {buffer_path}")

            return self.model

        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """모델 평가"""
        if self.model is None:
            print("평가할 모델이 없습니다.")
            return {}

        print(f"\n모델 평가 중... ({n_episodes} 에피소드)")

        results = {
            "episode_rewards": [], 
            "episode_lengths": [], 
            "success_count": 0
        }

        # 평가용 환경
        eval_env = self.env
        if hasattr(self.env, "num_envs") and self.env.num_envs > 1:
            eval_env = PursuitEvasionEnv(self.config)

        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            results["episode_rewards"].append(episode_reward)
            results["episode_lengths"].append(episode_length)

            # 성공 여부 판단
            if "outcome" in info:
                success = info["outcome"] in [
                    "permanent_evasion",
                    "conditional_evasion",
                    "max_steps_reached",
                ]
                if success:
                    results["success_count"] += 1

            print(f"에피소드 {episode+1}: 보상={episode_reward:.2f}, 길이={episode_length}")

        # 평가 결과 요약
        results["mean_reward"] = np.mean(results["episode_rewards"])
        results["std_reward"] = np.std(results["episode_rewards"])
        results["mean_length"] = np.mean(results["episode_lengths"])
        results["success_rate"] = results["success_count"] / n_episodes

        print(f"\n평가 결과:")
        print(f"  평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  평균 길이: {results['mean_length']:.1f}")
        print(f"  성공률: {results['success_rate']:.1%}")

        if eval_env is not self.env:
            eval_env.close()

        return results

    def get_training_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {
            "log_dir": self.log_dir,
            "experiment_name": self.experiment_name,
            "total_timesteps": self.model.num_timesteps if self.model else 0,
            "training_config": self.training_config.__dict__,
        }

        # 콜백에서 통계 수집
        for callback in self.callbacks:
            if isinstance(callback, EvasionTrackingCallback):
                stats.update({
                    "episodes_completed": callback.episode_count,
                    "success_rates": callback.success_rates,
                    "outcomes": callback.outcomes,
                    "nash_metrics": callback.nash_equilibrium_metrics,
                })
                break

        return stats

    def _save_training_history(self):
        """학습 히스토리 저장 - NumPy/Tensor 직렬화 처리"""
        import json
        
        def convert_to_serializable(obj):
            """객체를 JSON 직렬화 가능한 형태로 변환"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return obj
        
        history = self.get_training_stats()
        serializable_history = convert_to_serializable(history)
        
        # 파일 저장
        history_path = f"{self.log_dir}/training_history.json"
        try:
            with open(history_path, "w") as f:
                json.dump(serializable_history, f, indent=2, default=str)
            print(f"학습 히스토리 저장됨: {history_path}")
        except Exception as e:
            print(f"학습 히스토리 저장 중 오류: {e}")

    def continue_training(self, additional_timesteps: int, tb_log_name: str = "sac_continue") -> "SACTrainer":
        """학습 계속하기"""
        if self.model is None:
            print("계속할 모델이 없습니다.")
            return self

        print(f"\n학습 계속... (추가 {additional_timesteps:,} 스텝)")

        return self.train(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=False,
            tb_log_name=tb_log_name,
        )

    def fine_tune(self, new_env, fine_tune_timesteps: int, learning_rate_multiplier: float = 0.1) -> "SACTrainer":
        """파인 튜닝"""
        if self.model is None:
            print("파인 튜닝할 모델이 없습니다.")
            return self

        print(f"\n파인 튜닝 시작... ({fine_tune_timesteps:,} 스텝)")

        # 학습률 조정
        original_lr = self.model.learning_rate
        self.model.learning_rate = original_lr * learning_rate_multiplier

        # 새 환경 설정
        old_env = self.env
        self.env = new_env
        self.model.set_env(new_env)

        # 파인 튜닝 실행
        self.train(
            total_timesteps=fine_tune_timesteps,
            reset_num_timesteps=False,
            tb_log_name="sac_fine_tune",
        )

        # 설정 복원
        self.model.learning_rate = original_lr
        self.env = old_env

        print("파인 튜닝 완료!")
        return self

    def cleanup(self):
        """리소스 정리"""
        if self.model and hasattr(self.model, "replay_buffer"):
            del self.model.replay_buffer

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("리소스 정리 완료")


def create_trainer(
    env,
    config: Optional[ProjectConfig] = None,
    experiment_name: Optional[str] = None,
    log_dir: Optional[str] = None,
    resume: bool = False,
) -> SACTrainer:
    """트레이너 생성 헬퍼 함수"""
    if config is None:
        from config.settings import get_config
        config = get_config(experiment_name=experiment_name)

    trainer = SACTrainer(env, config, experiment_name=experiment_name, log_dir=log_dir, resume=resume)
    return trainer
