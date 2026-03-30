---
description: Guide through adding a new VLA model adapter
argument-hint: Model name (e.g., openvla, minivla)
---

# Add VLA Model Adapter

Guide the user through adding a new VLA model adapter to vla-edge.

Model name: $ARGUMENTS

## Steps

1. **Read the base class**: Read `src/vla_edge/models/base.py` to understand VLAModel ABC
2. **Create the adapter file**: `src/vla_edge/models/{name}.py`
3. **Implement the class**:
   - Subclass VLAModel
   - Decorate with `@register_model("{name}")`
   - Implement `predict(image, instruction, state)` returning np.ndarray
   - Implement `info` property returning ModelInfo
   - Override `preprocess_image()` if the model needs custom preprocessing
4. **Add lazy import**: Add to `_ensure_models_loaded()` in `src/vla_edge/registry.py`
5. **Write tests**: `tests/test_models/test_{name}.py` with dummy data
6. **Update recipe**: Create `recipes/{name}-{hardware}.yaml` if hardware-tested
7. **Run tests**: `pytest -v`

## Template

```python
from vla_edge.models.base import ModelInfo, VLAModel
from vla_edge.registry import register_model

@register_model("{name}")
class {Name}Model(VLAModel):
    def __init__(self):
        self._model = None

    def predict(self, image, instruction, state=None):
        if self._model is None:
            self._load()
        # ... inference logic
        return actions

    @property
    def info(self):
        return ModelInfo(name="{name}", param_count=..., action_dim=7)

    def _load(self):
        # Load from HuggingFace or local path
        pass
```
