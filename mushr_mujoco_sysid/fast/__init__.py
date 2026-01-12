"""
Fast inference path for mushr_mujoco_sysid models.

This module provides GPU-optimized inference sessions with:
- Preallocated tensors for batch=1 fixed-shape inference
- Single-compile boundary (torch.compile)
- Optional TF32 for extra speed
- Optional CUDA Graph replay for minimal latency/variance
"""

from .inference_session import FastInferenceSession

__all__ = ["FastInferenceSession"]
