# Changelog

All notable changes to vla-edge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffold with modular architecture (backends + models registries)
- CPU, CUDA, and Jetson hardware backends
- Latency profiler with warmup, percentiles, and FPS calculation
- Action safety validation (bounds, velocity, acceleration, workspace)
- Typer CLI with check, profile, models, and version commands
- pytest suite with hardware-aware markers (cpu, gpu, jetson)
- Pre-commit hooks (ruff, mypy, trailing whitespace)
- Deployment recipe system (YAML configs)
