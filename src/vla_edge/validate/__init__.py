"""Validation and monitoring tools for VLA deployment.

Core:
    safety_contract     - One-line decorator for action-space enforcement
    ActionHealthMonitor - Continuous behavioral health signals

Conformal:
    ConformalActionMonitor  - Per-action conformal p-values
    AdaptiveConformalMonitor - Online calibration under shift (ACI)
    CUSUMDetector           - Optimal sequential shift detection
    MultiScaleCUSUM         - Step/chunk/episode temporal hierarchy

Violations:
    StallDetector              - Detect policy freeze
    JerkMonitor                - Detect denoising artifacts, chunk boundaries
    GripperOscillationDetector - Detect rapid gripper cycling

STL:
    STLActionMonitor - Temporal logic monitoring with robustness scores

Usage:
    from vla_edge.validate import safety_contract, StallDetector, ConformalActionMonitor
"""

from vla_edge.validate.conformal import (
    AdaptiveConformalMonitor,
    ConformalActionMonitor,
    CUSUMDetector,
    MultiScaleCUSUM,
)
from vla_edge.validate.contract import safety_contract
from vla_edge.validate.guard import SafetyGuard
from vla_edge.validate.monitor import ActionHealthMonitor, ActionHealthReport
from vla_edge.validate.signals import (
    ActionEnergyMonitor,
    CorrelationDivergenceMonitor,
    LatencyMonitor,
    LinearPredictabilityMonitor,
    MomentumCoherenceMonitor,
    SpectralEnergyMonitor,
)
from vla_edge.validate.stl_monitor import STLActionMonitor
from vla_edge.validate.violations import (
    GripperOscillationDetector,
    JerkMonitor,
    StallDetector,
)

__all__ = [
    "ActionEnergyMonitor",
    "ActionHealthMonitor",
    "ActionHealthReport",
    "AdaptiveConformalMonitor",
    "CUSUMDetector",
    "ConformalActionMonitor",
    "CorrelationDivergenceMonitor",
    "GripperOscillationDetector",
    "JerkMonitor",
    "LatencyMonitor",
    "LinearPredictabilityMonitor",
    "MomentumCoherenceMonitor",
    "MultiScaleCUSUM",
    "STLActionMonitor",
    "SafetyGuard",
    "SpectralEnergyMonitor",
    "StallDetector",
    "safety_contract",
]
