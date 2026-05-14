"""Small file helpers used by training and inference."""

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read JSON content from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    """Write formatted JSON content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
