"""Tests for YAML recipe loading."""

from pathlib import Path

import pytest
import yaml

from vla_edge.config import Recipe


@pytest.fixture
def valid_recipe_yaml(tmp_path):
    """Create a valid recipe YAML file."""
    data = {
        "name": "test-recipe",
        "model": "smolvla",
        "hardware": "jetson-orin-nano-super",
        "optimization": {"dtype": "fp16", "quantize": None},
        "expected_performance": {"latency_ms": 150, "fps": 6.6},
        "notes": "Test recipe for unit testing.",
    }
    path = tmp_path / "test-recipe.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture
def minimal_recipe_yaml(tmp_path):
    """Recipe with only required fields."""
    data = {"model": "smolvla", "hardware": "cpu"}
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_valid_recipe(valid_recipe_yaml):
    """Load a complete recipe YAML."""
    recipe = Recipe.from_yaml(valid_recipe_yaml)
    assert recipe.name == "test-recipe"
    assert recipe.model == "smolvla"
    assert recipe.hardware == "jetson-orin-nano-super"
    assert recipe.optimization["dtype"] == "fp16"
    assert recipe.expected_performance["fps"] == 6.6
    assert "Test recipe" in recipe.notes


def test_load_minimal_recipe(minimal_recipe_yaml):
    """Load a recipe with only required fields."""
    recipe = Recipe.from_yaml(minimal_recipe_yaml)
    assert recipe.model == "smolvla"
    assert recipe.hardware == "cpu"
    assert recipe.name == "minimal"  # Defaults to filename stem
    assert recipe.optimization == {}
    assert recipe.notes == ""


def test_missing_model_raises(tmp_path):
    """Missing 'model' field should raise KeyError."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"hardware": "cpu"}))
    with pytest.raises(KeyError):
        Recipe.from_yaml(path)


def test_missing_hardware_raises(tmp_path):
    """Missing 'hardware' field should raise KeyError."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"model": "smolvla"}))
    with pytest.raises(KeyError):
        Recipe.from_yaml(path)


def test_load_existing_recipe():
    """Load the actual smolvla-jetson recipe from the repo."""
    recipe_path = Path("recipes/smolvla-jetson-orin-nano.yaml")
    if recipe_path.exists():
        recipe = Recipe.from_yaml(recipe_path)
        assert recipe.model == "smolvla"
        assert "jetson" in recipe.hardware
