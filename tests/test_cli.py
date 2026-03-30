"""Tests for the CLI."""

from typer.testing import CliRunner

from vla_edge.cli import app

runner = CliRunner()


def test_help():
    """CLI should show help text."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vla-edge" in result.stdout


def test_check_command():
    """check command should run without error."""
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0
    assert "Backend" in result.stdout


def test_version_command():
    """version command should show a version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "vla-edge" in result.stdout


def test_models_command():
    """models command should run without error."""
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0


def test_profile_help():
    """profile --help should show options."""
    result = runner.invoke(app, ["profile", "--help"])
    assert result.exit_code == 0
    assert "iterations" in result.stdout
    assert "hardware" in result.stdout
