---
description: Guide through adding a new hardware backend
argument-hint: Backend name (e.g., mps, hailo, qualcomm)
---

# Add Hardware Backend

Guide the user through adding a new hardware backend to vla-edge.

Backend name: $ARGUMENTS

## Steps

1. **Read the base class**: Read `src/vla_edge/backends/base.py` for HardwareBackend ABC
2. **Create the backend file**: `src/vla_edge/backends/{name}.py`
3. **Implement 4 methods**:
   - `is_available()` - detect if this hardware is present
   - `get_capabilities()` - return HardwareCapabilities dataclass
   - `load_model(model_path, dtype)` - load a model for this hardware
   - `infer(model, observation)` - run inference, return InferenceResult
4. **Register**: Decorate with `@register_backend("{name}")`
5. **Add lazy import**: Add to `_ensure_backends_loaded()` in registry.py
6. **Write tests**: `tests/test_backends/test_{name}.py` (mock hardware if needed)
7. **Run tests**: `pytest -v`

## Key: Keep it minimal. 4 methods. One file. No core changes.
