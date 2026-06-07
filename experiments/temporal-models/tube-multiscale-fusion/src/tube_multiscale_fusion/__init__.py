"""Tube multiscale fusion: global DINOv2 context + local tube transformer."""

from .classifier import TubeMultiscaleClassifier
from .fusion import FusionModule
from .global_branch import GlobalBranch, TimmBackbone
from .lit_module import LitTubeMultiscaleClassifier
from .local_branch import LocalBranch, LocalTubeEncoder, extract_tubes
from .model import TubeMultiscaleFusionModel

__all__ = [
    "FusionModule",
    "GlobalBranch",
    "LitTubeMultiscaleClassifier",
    "LocalBranch",
    "LocalTubeEncoder",
    "TimmBackbone",
    "TubeMultiscaleClassifier",
    "TubeMultiscaleFusionModel",
    "extract_tubes",
]
