# Plant and Model Structure

## Plant (deterministic backbone)
- File: `plant.py`, class `MushrPlant`.
- States in body frame: `xd = [vx, vy, w]`.
- Control indices: `velocity_idx=0`, `steering_idx=1`.
- Kinematics:
  - Bicycle-style slip angle `beta(delta) = atan(0.5 * tan(delta))`.
  - Turn rates `omega = 2 * sin(beta) / L`.
  - Adjoint/SE(2) utilities preserved from legacy code; integration via SE2 exponential map.
- `xdot(...)` inputs:
  - `xd0`: current twist.
  - `ut`: control `[acc_cmd, steer_cmd]`.
  - `dt`: step size.
  - Optional overrides:
    - `delta_override`, `acc_override` (bypass raw controls).
    - `friction` (scalar multiplier on yaw delta).
    - `residual` (3-vector additive twist residual).
- Output: next twist `xd1` in body frame.

## Learned components (optional)

### ControlAdapter (optional)
- File: `models/system_models.py`, class `ControlAdapter`.
- Purpose: adapt raw controls into effective `delta_eff` and `acc_eff`.
- Inputs per step: `[delta_raw, vel_raw, vx, vy, w]` plus optional features per net:
  - Steering net optional: `|v|`, `sign(vx)`.
  - Acc net optional: `|v|`, `sign(vx)`, `vy`, `w`.
- Outputs: `ut_eff = [acc_eff, delta_eff]`, with optional output scaling/clipping via tanh and user-specified scales.
- Typical use: capture actuator limits/slip-dependent effectiveness without altering plant structure.

### StructuredDynamicsModel (plant + learnables, optional fric/residual)
- File: `models/system_models.py`, class `StructuredDynamicsModel`.
- Uses `MushrPlant` for physics; optionally composes a `ControlAdapter`.
- Optional learnables:
  - `friction_net`: inputs `[omega_prev]` plus optional `delta`, `vy`, `dt`; output passed through `1 + softplus(...)` to stay >1. Purpose: yaw damping/gain adaptation.
  - `residual_net`: inputs `[xd0, ut_eff]`; output 3-d residual added to plant twist to capture unmodeled effects (kept small via loss/regularization externally).
- Forward:
  - Adapt controls (if adapter enabled) → compute `delta_eff`.
  - Compute `omega_prev` from current `xd0`.
  - Predict optional `friction_val`, optional `residual`.
  - Call `plant.xdot(..., friction=friction_val, residual=residual, delta_override=delta_eff, acc_override=acc_eff)`.

### DirectDynamicsModel (pure learned, optional adapter)
- File: `models/system_models.py`, class `DirectDynamicsModel`.
- Inputs: concatenated `[xd0 (3), ut_raw (2), dt (1)]`; adapter can pre-process controls if enabled.
- Architecture: MLP with configurable hidden dims.
- Output: predicted next twist `xd1`.
- Use case: fully learned dynamics when structured plant is insufficient.

## System harness
- File: `system.py` now only wires `MushrPlant` and imports model classes; legacy polynomial demo kept for reference/testing only. Use `train.py` pipeline for training/eval.
