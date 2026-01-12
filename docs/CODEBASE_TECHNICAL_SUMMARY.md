# MuSHR MuJoCo System Identification - Technical Summary

This document provides an in-depth technical summary of the codebase for reference when exploring C++ integration approaches.

---

## 1. Primary Use Case

This is a **system identification research codebase** for learning dynamics models of the MuSHR autonomous racing platform. The goal is to predict vehicle dynamics from state and control inputs, enabling accurate model-predictive control (MPC) or trajectory planning.

### Core Prediction Task

| Aspect | Details |
|--------|---------|
| **Input State** | Body-frame velocities: `[vx, vy, ω]` (m/s, m/s, rad/s) |
| **Control Input** | `[vel_cmd, steer_cmd]` (desired velocity, steering angle) |
| **Timestep** | `dt` (typically 0.05s) |
| **Output** | Predicted next velocity state: `[vx', vy', ω']` |

The models predict velocity states in the body frame. Pose (x, y, θ) is obtained by integrating velocity predictions over time using SE(2) exponential map integration.

---

## 2. Framework & Dependencies

### Core Stack

| Component | Library | Version | Notes |
|-----------|---------|---------|-------|
| Neural Networks | **PyTorch** | 2.5.1 | Primary ML framework |
| GPU Acceleration | pytorch-cuda | 12.4 | CUDA backend |
| Numerical Arrays | NumPy | 2.2.6 | Data processing |
| Visualization | Matplotlib | 3.10.8 | Plotting |
| Configuration | PyYAML | 6.0.3 | Config parsing |
| ROS Integration | rospy | 1.16.0 | Current ROS interface (pip) |
| ROS Utilities | rospkg | 1.6.1 | ROS package tools |

### Python Version
- Python 3.10 (specified in `environment.yaml`)

### Key Observation
The codebase is **pure PyTorch** with no ONNX, TensorFlow, or other inference frameworks currently implemented. All models are `torch.nn.Module` subclasses.

---

## 3. Project Structure

```
mushr_mujoco_sysid/
├── mushr_mujoco_sysid/          # Main Python package
│   ├── __init__.py
│   ├── plant.py                  # Physics-based kinematic bicycle model
│   ├── poly_sysid.py             # Polynomial baseline model
│   ├── data.py                   # Dataset classes and sampling
│   ├── dataloader.py             # Trajectory file loading
│   ├── model_factory.py          # Model construction from config
│   ├── evaluation.py             # Trajectory rollout evaluation
│   ├── config_utils.py           # Configuration handling
│   ├── utils.py                  # Standardization, helpers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py                # MLP building block
│   │   └── system_models.py      # StructuredDynamicsModel, DirectDynamicsModel
│   └── fast/
│       ├── __init__.py
│       ├── models_fast.py        # GPU-optimized model variants
│       └── inference_session.py  # Fast inference wrapper with CUDA Graphs
│
├── scripts/
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   ├── eval_poly_model.py        # Polynomial baseline evaluation
│   ├── compare_poly_vs_nn.py     # Model comparison
│   └── bench_inference.py        # Inference benchmarking
│
├── configs/                      # Model configurations (JSON)
│   └── v3_allstars_controls_vary/  # Current best configurations
│
├── data/
│   └── sysid_trajs_v3/           # Training data (3000 trajectories)
│
├── experiments/                  # Training outputs (checkpoints, logs)
├── docs/                         # Documentation
├── environment.yaml              # Conda environment specification
└── setup.py                      # Package installation
```

---

## 4. Model Architectures

### 4.1 Structured Dynamics Model (`StructuredDynamicsModel`)

This is the **best-performing model** (v3B achieves 0.006 MSE vs 8.34 for polynomial baseline).

#### Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │         StructuredDynamicsModel             │
                    └─────────────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────┼──────────────────────────────────┐
     │                                  │                                  │
     ▼                                  ▼                                  ▼
┌─────────────┐                  ┌─────────────┐                   ┌─────────────┐
│  Control    │                  │   Physics   │                   │  Residual   │
│  Adapter    │ ───────────────► │   Model     │ ◄───────────────  │  Network    │
│  (optional) │                  │ (MushrPlant)│                   │  (optional) │
└─────────────┘                  └─────────────┘                   └─────────────┘
       │                                │                                  │
       │                                ▼                                  │
       │                    ┌───────────────────────┐                      │
       │                    │   Friction Network    │                      │
       │                    │      (optional)       │                      │
       │                    └───────────────────────┘                      │
       │                                │                                  │
       └────────────────────────────────┼──────────────────────────────────┘
                                        │
                                        ▼
                              Output: [vx', vy', ω']
```

#### Component Details

**1. Control Adapter (`ControlAdapter`)**
- Purpose: Learn nonlinear actuator mappings (steering servo response, throttle dynamics)
- Two separate MLPs:
  - **Steering Network**: Maps raw steering command to effective steering angle
    - Input: `[δ_raw, vel_raw, vx, vy, ω]` + optional `[speed_mag, speed_sign]` (5-7 dims)
    - Hidden: `[64, 64]` with tanh activation
    - Output: `δ_eff` (1 dim), optionally scaled by `tanh() * steer_output_scale`
  - **Acceleration Network**: Maps raw velocity command to effective acceleration
    - Input: `[vel_raw, δ_raw, vx]` + optional `[speed_mag, speed_sign, vy, ω]` (3-7 dims)
    - Hidden: `[64, 64]` with tanh activation
    - Output: `acc_eff` (1 dim), optionally scaled by `tanh() * acc_output_scale`

**2. Physics Model (`MushrPlant`)**
- Kinematic bicycle model with fixed parameters:
  - Wheelbase `L = 0.31` meters
  - Mass `mass = 3.5` kg (not actively used in current model)
- Key operations:
  - Slip angle: `β = atan(0.5 * tan(δ))`
  - Angular velocity: `ω = 2 * sin(β) / L`
  - SE(2) exponential map integration for pose updates
  - Euler integration for velocity updates

**3. Friction Network (optional)**
- Purpose: Learn state-dependent friction coefficient
- Input: `[ω_prev]` + optional `[δ, vy, dt]` (1-4 dims)
- Hidden: `[32, 32]` with tanh activation
- Output: friction coefficient `k` via parameterization:
  - `softplus_offset_1`: `k = 1 + softplus(h)` (k ≥ 1)
  - `sigmoid_range`: `k = k_min + (k_max - k_min) * sigmoid(h)`
  - `exp`: `k = clamp(exp(h), k_min, k_max)`

**4. Residual Network (optional)**
- Purpose: Learn corrections to physics model output
- Input: `[vx, vy, ω, vel_eff, δ_eff]` (5 dims)
- Hidden: `[32, 32]` with tanh activation
- Output: `[Δvx, Δvy, Δω]` (3 dims) added to physics output

#### Forward Pass Logic

```python
def forward(xd0, ut_raw, dt):
    # 1. Control adaptation
    ut_eff = control_adapter(xd0, ut_raw) if enabled else ut_raw
    
    # 2. Compute friction (if learned)
    friction_k = friction_net(ω_prev, δ_eff, vy, dt) if enabled else 1.0
    
    # 3. Compute residual (if learned)
    residual = residual_net(xd0, ut_eff) if enabled else 0.0
    
    # 4. Physics model forward pass
    xd1 = plant.xdot(xd0, ut_eff, dt, friction=friction_k) + residual
    
    return xd1
```

### 4.2 Direct Dynamics Model (`DirectDynamicsModel`)

Pure end-to-end MLP, faster but slightly less accurate.

#### Architecture

```
Input [6]                    Hidden Layers                    Output [3]
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ vx, vy  │     │   256   │     │   256   │     │   128   │     │ vx'     │
│ ω       │ ──► │  tanh   │ ──► │  tanh   │ ──► │  tanh   │ ──► │ vy'     │
│ vel_cmd │     │         │     │         │     │         │     │ ω'      │
│ steer   │     └─────────┘     └─────────┘     └─────────┘     └─────────┘
│ dt      │
└─────────┘
```

- Optional control adapter as preprocessing step
- Typical hidden dims: `[256, 256, 128]` (~100k parameters)
- Uses tanh activation throughout

### 4.3 MLP Building Block

All neural networks use a common `MLP` class:

```python
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],      # e.g., [64, 64] or [256, 256, 128]
        activation: str = 'tanh',     # 'relu', 'tanh', 'elu', 'gelu', 'leaky_relu'
        dropout: float = 0.0,
        use_batch_norm: bool = False
    )
```

The network is constructed as:
```
Linear(input_dim, hidden[0]) → [BatchNorm] → Activation → [Dropout] →
Linear(hidden[0], hidden[1]) → [BatchNorm] → Activation → [Dropout] →
...
Linear(hidden[-1], output_dim)  # No activation on output
```

Weight initialization: Kaiming normal (fan_in mode)

---

## 5. Physics Model Details (`MushrPlant`)

The physics core implements a kinematic bicycle model with SE(2) Lie group integration.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `L` | 0.31 m | Wheelbase |
| `mass` | 3.5 kg | Vehicle mass |
| `velocity_idx` | 0 | Index of velocity command in control vector |
| `steering_idx` | 1 | Index of steering command in control vector |

### Core Mathematical Operations

**1. Slip Angle Computation**
```python
β = atan(0.5 * tan(δ))  # Rear axle slip angle approximation
```

**2. Angular Velocity from Slip Angle**
```python
ω = 2.0 * sin(β) / L
```

**3. SE(2) Representation**
```python
def SE2(x, y, θ):
    # Returns 3x3 homogeneous transformation matrix
    [[cos(θ), -sin(θ), x],
     [sin(θ),  cos(θ), y],
     [0,       0,      1]]
```

**4. SE(2) Exponential Map**
```python
def SE2_expmap(xd):  # xd = [vx, vy, ω]
    # Maps velocity twist to SE(2) transformation
    # Handles near-zero ω case separately
```

**5. Adjoint Map**
```python
def AdjointMap(pose):
    # For velocity frame transformation
    [[R, [y, -x]^T],
     [0,  1      ]]
```

**6. Forward Dynamics (`xdot` method)**

The main forward pass computes next velocity state:
1. Extract effective steering and acceleration (from control adapter or raw)
2. Compute current speed magnitude with sign preservation
3. Calculate slip angles (current and commanded)
4. Build SE(2) transformations for frame changes
5. Apply Euler integration for velocity update
6. Apply friction scaling to angular velocity change
7. Transform back to body frame using adjoint
8. Add residual correction if enabled

---

## 6. Data Normalization (Standardization)

### Critical Requirement
**All inputs and outputs are z-score standardized** using statistics computed from training data.

### Standardizer Class

```python
@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse(self, data):
        return data * self.std + self.mean
```

### Dimensions

| Standardizer | Dimensions | Components |
|--------------|------------|------------|
| Input | 5 | `[vx, vy, ω, vel_cmd, steer_cmd]` |
| Target | 3 | `[vx', vy', ω']` |

### Storage Format (`standardizers.json`)

```json
{
  "input": {
    "mean": [0.123, 0.045, 0.089, 0.567, 0.012],
    "std": [0.234, 0.056, 0.178, 0.345, 0.123]
  },
  "target": {
    "mean": [0.125, 0.044, 0.091],
    "std": [0.236, 0.057, 0.179]
  }
}
```

### Inference Pipeline

```python
# 1. Concatenate inputs
xu_raw = [vx, vy, ω, vel_cmd, steer_cmd]  # shape: (5,)

# 2. Normalize
xu_norm = (xu_raw - input_mean) / input_std

# 3. Split for model
xd0_norm = xu_norm[:3]   # [vx, vy, ω] normalized
ut_norm = xu_norm[3:5]   # [vel_cmd, steer_cmd] normalized

# 4. Model forward pass
xd_next_norm = model(xd0_norm, ut_norm, dt)

# 5. Denormalize output
xd_next = xd_next_norm * target_std + target_mean
```

---

## 7. Model Checkpoints

### File Format

Checkpoints are saved using `torch.save()`:
- **Filename**: `best.pt` (configurable via `training.ckpt_name`)
- **Format**: Python dictionary or raw state dict

### Checkpoint Structure

```python
# Option 1: Raw state dict
checkpoint = OrderedDict({
    'control_adapter.steer_net.network.0.weight': tensor(...),
    'control_adapter.steer_net.network.0.bias': tensor(...),
    'control_adapter.steer_net.network.2.weight': tensor(...),
    ...
    'residual_net.network.0.weight': tensor(...),
    ...
})

# Option 2: Wrapped dict
checkpoint = {
    'model_state_dict': OrderedDict({...}),
    # or 'model_state': OrderedDict({...})
}
```

### Weight Key Naming Convention

For `StructuredDynamicsModel`:
```
control_adapter.steer_net.network.{layer_idx}.weight
control_adapter.steer_net.network.{layer_idx}.bias
control_adapter.acc_net.network.{layer_idx}.weight
control_adapter.acc_net.network.{layer_idx}.bias
friction_net.network.{layer_idx}.weight  # if learn_friction=True
friction_net.network.{layer_idx}.bias
residual_net.network.{layer_idx}.weight  # if learn_residual=True
residual_net.network.{layer_idx}.bias
```

Layer indices follow `nn.Sequential` structure:
- Index 0: First Linear layer
- Index 1: Activation (no weights)
- Index 2: Second Linear layer
- etc.

For `DirectDynamicsModel`:
```
net.network.{layer_idx}.weight
net.network.{layer_idx}.bias
```

---

## 8. Configuration System

### Configuration File Format (JSON)

```json
{
  "seed": 4,
  "data": {
    "data_dir": "data/sysid_trajs_v3",
    "val_ratio": 0.1,
    "val_split_mode": "trajectory"
  },
  "model": {
    "type": "structured",           // or "direct"
    "learn_friction": false,
    "learn_residual": true,
    "control_adapter": {
      "enabled": true,
      "include_speed_mag_steer": true,
      "include_speed_sign_steer": true,
      "include_speed_mag_acc": true,
      "include_speed_sign_acc": true,
      "include_vy_acc": true,
      "include_w_acc": true,
      "steer_output_scale": 0.6,
      "acc_output_scale": 1.0
    }
  },
  "training": {
    "batch_size": 256,
    "epochs": 1000,
    "lr": 0.001,
    "device": "cuda",
    "run_root": "experiments/...",
    "run_name": "experiment_name",
    "loss": {
      "one_step_mse": {"enabled": true, "weight": 1.0},
      "rollout_mse": {"enabled": true, "horizon": 10, "weight": 0.5},
      "pose_mse": {"enabled": true, "weight": 0.05}
    }
  }
}
```

### Model Factory

Models are constructed from config via `build_model()`:

```python
def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    # 1. Create MushrPlant (physics model)
    # 2. Optionally create ControlAdapter from config
    # 3. Create StructuredDynamicsModel or DirectDynamicsModel
    # 4. Move to device and return
```

---

## 9. Fast Inference System

### `FastInferenceSession`

A production-ready inference wrapper with GPU optimizations:

```python
class FastInferenceSession:
    def __init__(
        self,
        exp_dir: str,           # Path to experiment (config.json, best.pt, standardizers.json)
        dt: float,              # Fixed timestep
        device: str = "cuda",
        dtype: str = "float32",
        use_compile: bool = False,    # torch.compile optimization
        use_tf32: bool = False,       # TensorFloat-32 for Ampere+ GPUs
        use_cudagraph: bool = False,  # CUDA Graph replay
        warmup_iters: int = 3
    )
```

#### Features
1. **Single compile boundary**: torch.compile applied once
2. **Preallocated tensors**: Fixed batch=1 buffers on GPU
3. **Cached standardizers**: Mean/std as GPU tensors
4. **CUDA Graph capture**: Optional graph replay for minimal latency

#### API

```python
# Tensor interface
def predict_one(xd0: Tensor, ut: Tensor) -> Tensor:
    # xd0: [vx, vy, ω], shape (3,) or (1, 3)
    # ut: [vel_cmd, steer_cmd], shape (2,) or (1, 2)
    # Returns: [vx', vy', ω'], shape (1, 3)

# NumPy interface
def predict_one_numpy(xd0: ndarray, ut: ndarray) -> ndarray:
    # Convenience wrapper with CPU ↔ GPU transfers
```

### Fast Model Variants

Optimized versions in `fast/models_fast.py`:
- `ControlAdapterFast`
- `StructuredDynamicsModelFast`
- `DirectDynamicsModelFast`

Assumptions for fast path:
- Batch size = 1
- float32 dtype
- GPU tensors
- No auxiliary outputs (value-only inference)

---

## 10. Model Performance Summary

From comprehensive v3 evaluation:

| Model | Type | Config | Trajectory MSE | Relative |
|-------|------|--------|----------------|----------|
| v3B | Structured | h=10, TF=0.0, seed=4 | 0.006 | Best |
| v3G | Direct | h=20, TF=0.0 | 0.008 | 4x faster |
| v3D | Structured | h=20, seed=2 | 0.029 | - |
| Polynomial | Physics-only | - | 8.34 | Baseline |

**Key findings**:
- Neural networks achieve **~1400x better accuracy** than physics baseline
- Structured models slightly outperform direct MLPs
- Rollout horizon of 10 with no teacher forcing works best
- All models are small enough for real-time inference

---

## 11. Approximate Model Sizes

| Component | Parameters | Notes |
|-----------|------------|-------|
| Control Adapter (steering) | ~8,500 | 5→64→64→1 |
| Control Adapter (accel) | ~4,700 | 3→64→64→1 |
| Friction Network | ~1,200 | 4→32→32→1 |
| Residual Network | ~1,300 | 5→32→32→3 |
| **Structured Total** | ~15,000 | With adapter + residual |
| Direct Model | ~100,000 | 6→256→256→128→3 |

These are small models suitable for real-time CPU inference if needed.

---

## 12. Data Format (v3 Trajectories)

### Trajectory File Format

Each trajectory is a text file with space-separated values:
```
x y theta xDot yDot thetaDot steering_angle velocity_desired
```

Columns:
1. `x`: Global X position (m)
2. `y`: Global Y position (m)
3. `theta`: Heading angle (rad)
4. `xDot`: Global X velocity (m/s)
5. `yDot`: Global Y velocity (m/s)
6. `thetaDot`: Angular velocity (rad/s)
7. `steering_angle`: Steering command (rad)
8. `velocity_desired`: Velocity command (m/s)

### Dataset Statistics
- 3,000 trajectories total
- 1,000 constant control trajectories
- 2,000 varying control trajectories
- Train/val split by trajectory (not timestep) to prevent data leakage

---

## 13. ROS Integration (Current State)

The current ROS integration is via Python `rospy`:
- Listed in `environment.yaml` as pip dependencies
- Uses ROS message passing paradigm
- Runs inference in Python process

This is the part that may be replaced with C++ approaches.

---

## 14. Key Files for C++ Integration Reference

| File | Lines | Purpose |
|------|-------|---------|
| `models/system_models.py` | 342 | Model architectures (must understand) |
| `models/mlp.py` | 120 | MLP implementation (simple to port) |
| `plant.py` | 173 | Physics model (pure math, portable) |
| `utils.py` | 95 | Standardizer (trivial to port) |
| `fast/inference_session.py` | 296 | Inference pattern (reference) |
| `model_factory.py` | 65 | Config → Model construction |

---

## 15. Summary of Technical Characteristics

1. **Framework**: Pure PyTorch 2.5.1, no other ML frameworks
2. **Model sizes**: Small (15k-100k parameters), suitable for real-time
3. **Activations**: Primarily `tanh`, some `softplus`/`sigmoid` for outputs
4. **Normalization**: Z-score standardization on inputs and outputs (required)
5. **Physics**: SE(2) Lie group kinematics, bicycle model
6. **Checkpoints**: PyTorch state dicts, weight matrices as tensors
7. **Inference**: Batch=1 optimized, CUDA Graph support available
8. **Precision**: float32 default, TF32 optional on Ampere+ GPUs
