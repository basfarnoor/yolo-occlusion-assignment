"""LiDAR-supported, CAM_FRONT-only offline evaluation for Relational OATM.

The package keeps online tracking and privileged evaluation in separate
stages.  Nothing in this package supplies nuScenes annotations, LiDAR point
counts, calibration, or ego pose to a tracker.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
