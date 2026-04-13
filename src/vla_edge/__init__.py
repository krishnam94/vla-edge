"""vla-edge: Safety monitoring for learned robot policies.

Quick start:
    from vla_edge import SafetyGuard

    guard = SafetyGuard.from_demos(demo_actions)

    @guard.wrap
    def predict(obs):
        return model(obs)
"""

try:
    from vla_edge._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from vla_edge.validate.contract import safety_contract
from vla_edge.validate.guard import SafetyGuard

__all__ = ["__version__", "SafetyGuard", "safety_contract"]
