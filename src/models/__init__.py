"""Model interfaces and implementations."""

from .base import BaseVisualCoTModel


def __getattr__(name: str):
    if name == "DeepEyesV2Model":
        from .deepeyes_v2 import DeepEyesV2Model
        return DeepEyesV2Model
    if name == "DirectVLMModel":
        from .direct_vlm import DirectVLMModel
        return DirectVLMModel
    if name == "GRITModel":
        from .grit import GRITModel
        return GRITModel
    if name == "VisualCoTModel":
        from .viscot import VisualCoTModel
        return VisualCoTModel
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")


__all__ = ["BaseVisualCoTModel", "DeepEyesV2Model", "DirectVLMModel", "GRITModel", "VisualCoTModel"]
