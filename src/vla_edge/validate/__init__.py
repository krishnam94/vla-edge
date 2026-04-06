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

from vla_edge.validate.contract import safety_contract
from vla_edge.validate.guard import SafetyGuard
from vla_edge.validate.monitor import ActionHealthMonitor, ActionHealthReport
from vla_edge.validate.conformal import (
    ConformalActionMonitor,
    AdaptiveConformalMonitor,
    CUSUMDetector,
    MultiScaleCUSUM,
)
from vla_edge.validate.violations import (
    StallDetector,
    JerkMonitor,
    GripperOscillationDetector,
)
from vla_edge.validate.stl_monitor import STLActionMonitor

__all__ = [
    "SafetyGuard",
    "safety_contract",
    "ActionHealthMonitor",
    "ActionHealthReport",
    "ConformalActionMonitor",
    "AdaptiveConformalMonitor",
    "CUSUMDetector",
    "MultiScaleCUSUM",
    "StallDetector",
    "JerkMonitor",
    "GripperOscillationDetector",
    "STLActionMonitor",
]
