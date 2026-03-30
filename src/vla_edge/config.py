"""YAML recipe loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Recipe:
    """A deployment recipe - tested configuration for a model + hardware combo."""

    name: str
    model: str
    hardware: str
    optimization: dict[str, Any] = field(default_factory=dict)
    expected_performance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> Recipe:
        """Load a recipe from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            name=data.get("name", Path(path).stem),
            model=data["model"],
            hardware=data["hardware"],
            optimization=data.get("optimization", {}),
            expected_performance=data.get("expected_performance", {}),
            notes=data.get("notes", ""),
        )
