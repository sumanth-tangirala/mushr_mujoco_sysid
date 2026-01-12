"""
Fast inference session for GPU-optimized real-time dynamics prediction.

Provides a compile-once boundary with preallocated tensors, optional TF32,
and optional CUDA Graph replay for minimal latency/variance.
"""

import json
import os
import warnings

import numpy as np
import torch

from ..config_utils import populate_config_defaults
from ..model_factory import build_model
from ..utils import load_standardizers_json
from ..models.system_models import StructuredDynamicsModel, DirectDynamicsModel


class FastInferenceSession:
    """
    GPU-first fast inference session for mushr_mujoco_sysid models.

    Features:
    - Single compile boundary (no redundant torch.compile calls)
    - Preallocated CUDA tensors for fixed-shape batch=1 inference
    - Optional TF32 for extra throughput
    - Optional CUDA Graph replay for minimal latency/variance
    """

    def __init__(
        self,
        exp_dir: str,
        dt: float,
        device: str = "cuda",
        dtype: str = "float32",
        use_compile: bool = False,
        use_tf32: bool = False,
        use_cudagraph: bool = False,
        warmup_iters: int = 3,
    ):
        """
        Initialize fast inference session.

        Args:
            exp_dir: Path to experiment directory (config.json, best.pt, standardizers.json)
            dt: Fixed timestep for dynamics
            device: 'cuda' or 'cpu' (cuda recommended)
            dtype: 'float32' or 'float64' (float32 recommended)
            use_compile: Enable torch.compile (recommended on GPU)
            use_tf32: Enable TF32 mode for matmul/convs (Ampere+ GPUs)
            use_cudagraph: Enable CUDA Graph replay (requires CUDA, fixed shapes)
            warmup_iters: Number of warmup iterations before optional graph capture
        """
        self.exp_dir = exp_dir
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.use_compile = use_compile
        self.use_tf32 = use_tf32
        self.use_cudagraph = use_cudagraph
        self.warmup_iters = warmup_iters

        # Validate device for CUDA Graph
        if self.use_cudagraph and not self.device.type == "cuda":
            raise ValueError("CUDA Graph replay requires device='cuda'")

        # Load config
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.config = populate_config_defaults(self.config)

        # Load standardizers
        std_path = os.path.join(exp_dir, "standardizers.json")
        if not os.path.exists(std_path):
            raise FileNotFoundError(f"Standardizers not found: {std_path}")

        self.input_std, self.target_std = load_standardizers_json(std_path)

        # Build model
        self.model = build_model(self.config, self.device)
        self.model.eval()

        # Load checkpoint
        ckpt_name = self.config.get("training", {}).get("ckpt_name", "best.pt")
        ckpt_path = os.path.join(exp_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

        # Enable TF32 if requested (Ampere+ GPUs)
        if self.use_tf32:
            if self.device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[FastInferenceSession] TF32 enabled")
            else:
                warnings.warn("TF32 requested but device is not CUDA; ignoring")

        # Preallocate fixed-shape tensors for batch=1
        self._xd0_buf = torch.zeros((1, 3), device=self.device, dtype=self.dtype)
        self._ut_buf = torch.zeros((1, 2), device=self.device, dtype=self.dtype)
        self._dt_tensor = torch.tensor([self.dt], device=self.device, dtype=self.dtype)

        # Cache standardizer tensors
        self._input_mean = torch.tensor(
            self.input_std.mean, device=self.device, dtype=self.dtype
        )
        self._input_std_val = torch.tensor(
            self.input_std.std, device=self.device, dtype=self.dtype
        )
        self._target_mean = torch.tensor(
            self.target_std.mean, device=self.device, dtype=self.dtype
        )
        self._target_std_val = torch.tensor(
            self.target_std.std, device=self.device, dtype=self.dtype
        )

        # Optional: compile the forward pass (single compile boundary)
        self._forward_fn = self._build_forward_fn()
        if self.use_compile:
            print("[FastInferenceSession] Compiling forward pass with torch.compile...")
            self._forward_fn = torch.compile(self._forward_fn)

        # Warmup
        print(f"[FastInferenceSession] Warming up ({self.warmup_iters} iterations)...")
        self._warmup()

        # Optional: capture CUDA Graph
        self._cuda_graph = None
        self._graph_xd0_input = None
        self._graph_ut_input = None
        self._graph_output = None

        if self.use_cudagraph:
            print("[FastInferenceSession] Capturing CUDA Graph...")
            self._capture_cuda_graph()

        print("[FastInferenceSession] Ready!")

    def _build_forward_fn(self):
        """Build the forward function that will optionally be compiled."""

        def forward_fn(xd0_norm, ut_norm):
            # Call model forward
            if isinstance(self.model, StructuredDynamicsModel):
                xd_next_norm = self.model(xd0_norm, ut_norm, self._dt_tensor[0])
            elif isinstance(self.model, DirectDynamicsModel):
                xd_next_norm = self.model(xd0_norm, ut_norm, self._dt_tensor)
            else:
                xd_next_norm = self.model(xd0_norm, ut_norm, self._dt_tensor[0])

            # Denormalize
            xd_next = xd_next_norm * self._target_std_val + self._target_mean
            return xd_next

        return forward_fn

    def _warmup(self):
        """Warmup the model to trigger any one-time compilation/kernel selection."""
        dummy_xd0 = torch.zeros((1, 3), device=self.device, dtype=self.dtype)
        dummy_ut = torch.zeros((1, 2), device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                xu_raw = torch.cat([dummy_xd0, dummy_ut], dim=1)
                xu_norm = (xu_raw - self._input_mean) / self._input_std_val
                xd0_norm = xu_norm[:, :3]
                ut_norm = xu_norm[:, 3:5]
                _ = self._forward_fn(xd0_norm, ut_norm)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _capture_cuda_graph(self):
        """Capture a CUDA Graph for fixed-shape replay."""
        if not self.device.type == "cuda":
            raise RuntimeError("CUDA Graph capture requires CUDA device")

        # Allocate static input/output buffers for graph
        self._graph_xd0_input = torch.zeros(
            (1, 3), device=self.device, dtype=self.dtype
        )
        self._graph_ut_input = torch.zeros((1, 2), device=self.device, dtype=self.dtype)

        # Warmup before capture
        with torch.inference_mode():
            for _ in range(3):
                xu_raw = torch.cat([self._graph_xd0_input, self._graph_ut_input], dim=1)
                xu_norm = (xu_raw - self._input_mean) / self._input_std_val
                xd0_norm = xu_norm[:, :3]
                ut_norm = xu_norm[:, 3:5]
                _ = self._forward_fn(xd0_norm, ut_norm)

        torch.cuda.synchronize()

        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()

        with torch.inference_mode():
            with torch.cuda.graph(self._cuda_graph):
                xu_raw = torch.cat([self._graph_xd0_input, self._graph_ut_input], dim=1)
                xu_norm = (xu_raw - self._input_mean) / self._input_std_val
                xd0_norm = xu_norm[:, :3]
                ut_norm = xu_norm[:, 3:5]
                self._graph_output = self._forward_fn(xd0_norm, ut_norm)

        torch.cuda.synchronize()
        print("[FastInferenceSession] CUDA Graph captured successfully")

    def predict_one(self, xd0: torch.Tensor, ut: torch.Tensor) -> torch.Tensor:
        """
        Fast value-only inference (torch tensors in/out).

        Args:
            xd0: Current velocity state [vx, vy, w], shape (3,) or (1,3), torch.Tensor
            ut: Control input [vel_cmd, steer_cmd], shape (2,) or (1,2), torch.Tensor

        Returns:
            xd_next: Predicted next velocity state, shape (1,3), torch.Tensor
        """
        # Ensure batch dimension
        if xd0.dim() == 1:
            xd0 = xd0.unsqueeze(0)
        if ut.dim() == 1:
            ut = ut.unsqueeze(0)

        with torch.inference_mode():
            if self.use_cudagraph and self._cuda_graph is not None:
                # CUDA Graph replay path
                self._graph_xd0_input.copy_(xd0)
                self._graph_ut_input.copy_(ut)
                self._cuda_graph.replay()
                return self._graph_output.clone()
            else:
                # Standard inference path
                xu_raw = torch.cat([xd0, ut], dim=1)
                xu_norm = (xu_raw - self._input_mean) / self._input_std_val
                xd0_norm = xu_norm[:, :3]
                ut_norm = xu_norm[:, 3:5]
                xd_next = self._forward_fn(xd0_norm, ut_norm)
                return xd_next

    def predict_one_numpy(self, xd0: np.ndarray, ut: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for numpy inputs/outputs.

        Args:
            xd0: Current velocity state [vx, vy, w], shape (3,), np.ndarray
            ut: Control input [vel_cmd, steer_cmd], shape (2,), np.ndarray

        Returns:
            xd_next: Predicted next velocity state, shape (3,), np.ndarray
        """
        xd0_t = torch.from_numpy(xd0).to(self.device, dtype=self.dtype)
        ut_t = torch.from_numpy(ut).to(self.device, dtype=self.dtype)

        xd_next_t = self.predict_one(xd0_t, ut_t)

        return xd_next_t[0].detach().cpu().numpy()

    def get_info(self) -> dict:
        """Return information about the session configuration."""
        return {
            "exp_dir": self.exp_dir,
            "model_type": self.config.get("model", {}).get("type", "unknown"),
            "dt": self.dt,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "use_compile": self.use_compile,
            "use_tf32": self.use_tf32,
            "use_cudagraph": self.use_cudagraph,
            "control_adapter_enabled": self.config.get("model", {})
            .get("control_adapter", {})
            .get("enabled", False),
            "learn_friction": self.config.get("model", {}).get("learn_friction", False),
            "learn_residual": self.config.get("model", {}).get("learn_residual", False),
        }
