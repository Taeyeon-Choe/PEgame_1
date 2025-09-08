"""SB3 system information patch utilities."""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

from stable_baselines3.common import save_util as sb3_save_util

# Mapping from display name to importable module name
_EXTRAS: Dict[str, str] = {
    "torch": "torch",  # already recorded as PyTorch but kept for completeness
    "shimmy": "shimmy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "numba": "numba",
    "plotly": "plotly",
    "pandas": "pandas",
    "rich": "rich",
    "tqdm": "tqdm",
    "tensorboard": "tensorboard",
    "tensorboardX": "tensorboardX",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
    "black": "black",
    "flake8": "flake8",
    "mypy": "mypy",
    "pre-commit": "pre_commit",
}


def _get_version(module_name: str) -> str:
    """Return the version of a module or ``"Not Installed"``."""
    try:
        module = import_module(module_name)
        return getattr(module, "__version__", "Unknown")
    except Exception:
        return "Not Installed"


# Preserve the original function so we can call it
_ORIG_GET_SYSTEM_INFO = sb3_save_util.get_system_info


def _patched_get_system_info(print_info: bool = True) -> Tuple[Dict[str, str], str]:
    """Patched version of ``get_system_info`` with extra packages."""
    env_info, env_info_str = _ORIG_GET_SYSTEM_INFO(print_info=False)
    for display_name, module_name in _EXTRAS.items():
        version = _get_version(module_name)
        env_info[display_name] = version
        env_info_str += f"- {display_name}: {version}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str


def patch_sb3_system_info() -> None:
    """Patch Stable-Baselines3 to record extra package versions."""
    sb3_save_util.get_system_info = _patched_get_system_info
