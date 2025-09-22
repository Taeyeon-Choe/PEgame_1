"""Utility helpers for rendering MATLAB analysis scripts from templates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "analysis" / "templates"


def render_matlab_script(template_name: str, destination: Path, replacements: Dict[str, str]) -> Path:
    """Render a MATLAB script from the stored template.

    Args:
        template_name: Template file name located under ``analysis/templates``.
        destination: Full path (including file name) where the rendered script should be written.
        replacements: Mapping of placeholder keys to concrete values. Each key replaces
            occurrences of ``{{KEY}}`` within the template.

    Returns:
        Path to the generated script.

    Raises:
        FileNotFoundError: If the template is missing.
    """
    template_path = _TEMPLATE_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        placeholder = f"{{{{{key}}}}}"
        content = content.replace(placeholder, value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
