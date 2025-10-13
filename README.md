# Satellite Pursuit-Evasion Reinforcement Learning Framework

## Overview
This repository provides a reinforcement learning research framework for orbital pursuit-evasion scenarios.  
The environment simulates two satellites in Earth orbit with high-fidelity dynamics (Keplerian motion, J2 perturbation, and non-linear relative motion).  
Training is built on top of Stable-Baselines3 and targets continuous-action agents such as Soft Actor-Critic, TD3, and DDPG.  
On top of standard RL training you can explore minimax/Nash equilibrium strategies, automated evaluation pipelines, and rich post-processing that exports plots, CSV summaries, and MATLAB scripts.

## Key Features
- **Physics-accurate environment**: `environment/PursuitEvasionEnv` numerically propagates the evader’s chief orbit, enforces delta-V budgets, sensor ranges, stochastic noise, and multi-step capture/evasion buffers. The GA-STM variant (`PursuitEvasionEnvGASTM`) supports fast Gim-Alfriend state transition matrix propagation with TVLQR pursuer guidance.
- **Flexible RL stack**: `training/trainer.py` wraps Stable-Baselines3 (SAC/TD3/DDPG), multi-process vector environments, gSDE stabilisation, replay-buffer persistence, and periodic checkpointing. A minimax workflow in `training/nash_equilibrium.py` alternates evader training with pursuer strategy optimisation. 
- **Comprehensive analysis**: `analysis/evaluator.py` batch-evaluates trained policies, logs orbital trajectories, reward breakdowns, delta-V usage, and success/failure distributions. `analysis/visualization.py` renders 2D/3D trajectories, moving-average learning curves, SAC diagnostics, and produces CSV/Matlab artefacts for further study.
- **Configurable by design**: `config/settings.py` exposes typed dataclasses for orbit geometry, environment physics, training hyperparameters, plotting style, and output paths. Config objects auto-create directory structures and respect debug/GPU flags.
- **Tooling & reproducibility**: Extensive pytest coverage under `tests/`, reproducible experiment folders under `logs/`, Docker and Docker Compose files for CPU/GPU setups, and example notebooks/scripts for quick experiments.

## Repository Layout
| Path | Description |
| --- | --- |
| `main.py` | Command-line entry point with interactive menu, training/evaluation modes, and runtime overrides. |
| `config/` | Dataclasses and helpers for environment/training configuration and path management. |
| `environment/` | Pursuit-evasion Gymnasium environments (full dynamics and GA-STM accelerated variant). |
| `training/` | Trainer classes, callbacks, Nash equilibrium routines, and helper utilities. |
| `analysis/` | Evaluation scripts, visualisation helpers, metrics, and MATLAB template renderers. |
| `orbital_mechanics/` | Orbital dynamics primitives (chief orbit propagation, coordinate transforms, STM propagator). |
| `controllers/` | Pursuer guidance implementations including TVLQR control. |
| `examples/` | Minimal scripts (e.g., `examples/quick_start.py`) for rapid experimentation. |
| `tests/` | Pytest-based regression tests covering dynamics, training, and analysis modules. |
| `docker-compose.yml`, `Dockerfile*` | Container definitions for CPU/GPU training, TensorBoard, and Jupyter lab. |

## Getting Started

### Prerequisites
- Python 3.10 or 3.11 (3.8+ supported but newer versions recommended).
- Optional: NVIDIA GPU with CUDA 12.4 wheels (see `requirements.txt` for pinned PyTorch build).

### Installation
```bash
git clone https://github.com/Taeyeon-Choe/PEgame_1.git
cd PEgame_1

# (Optional) Create an isolated environment
python -m venv .venv          # or: conda create -n satellite-game python=3.11
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies (GPU users may prefer the bundled CUDA wheels)
pip install -r requirements.txt
```

### Quick Diagnostic
```bash
python -c "import orbital_mechanics, environment, training; print('Environment ready')"
```
The command instantiates core packages and confirms that dependencies are correctly installed.

## Training Workflows

### Interactive launcher
Running `python main.py` without arguments opens an interactive prompt that guides you through standard training, Nash equilibrium experiments, evaluations, and demos.

### Standard RL training
```bash
# SAC with default physics, automatic multi-env count, and log directory under ./logs/
python main.py --mode train_standard --experiment-name baseline_sac --timesteps 200000

# TD3 on GPU with GA-STM acceleration and custom save frequency
python main.py --mode train_standard --algorithm td3 --gpu --use-gastm --save-freq 20000

# Lower-fidelity debug run (short horizon, single env)
python examples/quick_start.py
```
Key runtime flags:
- `--n-envs`: override automatic vectorised environment count.
- `--cpu` / `--gpu`: force device selection.
- `--delta-v-emax`, `--delta-v-pmax`, `--max-delta-v-budget`, `--max-steps`: adjust physical constraints.
- `--reward-mode`: switch between `original`, `lq_zero_sum`, and shaped variants.

Training artefacts are stored at `logs/<experiment>_<timestamp>/` and include JSON configs, SB3 checkpoints (`models/`), TensorBoard summaries, CSV histories, and rendered plots.

### Nash equilibrium (minimax) training
```bash
python main.py --mode train_nash --experiment-name minimax_sac --timesteps 100000
```
`training/nash_equilibrium.py` alternates evader SAC updates with pursuer optimisation episodes, records Nash metrics, and saves intermediate models (`models/nash_iter*.zip`). The evaluator can later be reused to inspect convergence statistics.

## Evaluation & Visualisation
```bash
# Batch evaluation across randomised scenarios
python main.py --mode evaluate --model-path logs/<run>/models/sac_final.zip --n-tests 20

# Deterministic playback / visualisation
python main.py --mode demo --model-path logs/<run>/models/sac_final.zip
```
The evaluator exports:
- Scenario-level metrics (`test_results/<timestamp>/metrics.json`),
- Episode trajectories in LVLH/ECI coordinates,
- Delta-V consumption, safety buffers, and reward breakdowns,
- Aggregated plots (success rates, outcome histograms, SAC losses),
- Optional MATLAB scripts for customised post-processing (`analysis/templates`).

TensorBoard logs are under `tensorboard_logs/`; launch TensorBoard via docker-compose service or `tensorboard --logdir tensorboard_logs`.

## Configuration Tips
```python
from config.settings import get_config

config = get_config(
    experiment_name="custom_run",
    debug_mode=False,
)
config.environment.use_gastm = True
config.environment.reward_mode = "lq_zero_sum_shaped"
config.training.total_timesteps = 500_000
config.training.n_envs = 8
```
Pass custom configs directly to `create_trainer` or inject overrides from the CLI (`main.py` automatically maps flag values onto the dataclasses). `ProjectConfig` lazily creates required folders and ensures GPU flags are consistent with availability.

## Jupyter & Notebooks
- Launch notebooks with `jupyter lab` (see examples under `notebooks/`, e.g., `getting_started.ipynb`).
- `docker-compose up jupyter` starts a pre-configured Lab server mapped to the local `notebooks/` folder.

## Testing & Quality Checks
```bash
pytest
```
The suite covers environment dynamics, orbital mechanics utilities, trainer lifecycle, and analysis outputs. Running tests after modifying environment or training logic is recommended.

## Docker Support
- `docker-compose up satellite-game` provides a CPU environment with all dependencies installed.
- `docker-compose up satellite-game-gpu` enables CUDA-capable training (requires `nvidia-container-toolkit`).
- Additional services: `jupyter` (port 8888) and `tensorboard` (port 6006).

## Troubleshooting
- Enable debug mode via `--debug` or `get_config(debug_mode=True)` to shorten episodes and timesteps.
- If GA-STM propagation fails, the environment automatically falls back to numerical integration while logging the issue (see console output in debug mode).
- GPU memory issues: reduce `--n-envs`, lower `config.training.batch_size`, or switch to CPU mode.

## License
Distributed under the terms of the [MIT License](LICENSE).
