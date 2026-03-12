"""Test-Time Scaling (TTS) inference strategies."""

from .sampling import run_tts_sampling
from .scaling import run_tts_scaling

__all__ = ["run_tts_sampling", "run_tts_scaling"]
