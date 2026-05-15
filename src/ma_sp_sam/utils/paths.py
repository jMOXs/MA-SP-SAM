from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_path(project_root: str | Path, path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else Path(project_root) / path
