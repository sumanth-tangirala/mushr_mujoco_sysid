# System Overview, Config Variants, and Empirical Results

This document mirrors `system_and_configs.md` but additionally annotates each configuration with **empirical performance** based on the single run of each setup under `experiments/`.

- **Per-setup metrics**: for each config we report
  - best validation loss (minimum `val_loss` over epochs), and
  - best held-out trajectory MSEs (`eval_vel_mse` and `eval_pos_mse`, when available).
- **Lower is better** for all these metrics.

## States, controls, and time step

- **State**: the dynamical state used throughout the code is the body-frame twist
  $\dot{x} = [v_x, v_y, \omega]^T$, where:
  - $v_x$: longitudinal velocity in the car body frame.
  - $v_y$: lateral velocity in the car body frame.
  - $\omega$: yaw-rate-like angular velocity (see curvature definition below).
- **Control**: the input used by both the plant and the learned models is
  $u_t = [a^{cmd}_t, \delta^{cmd}_t]^T$, where:
  - $a^{cmd}_t$: commanded longitudinal acceleration (or velocity command scaled to an acceleration-like quantity).
  - $\delta^{cmd}_t$: front steering command.
- **Time step**:
  - Each line in the sysid logs carries its own time step `dt` column (elapsed time since the previous sample).
  - The training/evaluation pipeline always uses this per-sample `dt` from the dataset; there is no separate `dt` parameter in the configs.
  - All models approximate the discrete-time mapping
    $$\dot{x}_{t+1} = f(\dot{x}_t, u_t, \Delta t),$$
    where $\Delta t$ is taken from the dataset for each step in these experiments.

## Base plant dynamics (deterministic backbone)

The class `MushrPlant` in `plant.py` encodes a bicycle-style kinematic model with a slip-angle parameterization and an SE(2) integration scheme.
This plant is **not learned**; it is fixed physics.
Learned components only ever modify its inputs or add small corrections to its outputs.

### Slip angle and curvature

- **Slip angle**: from steering $\delta$, the plant computes a slip angle
  $$\beta(\delta) = \tan^{-1}\big(0.5 \tan(\delta)\big),$$
  which roughly corresponds to the angle between the body-frame longitudinal axis and the velocity direction at the rear axle.
- **Body-frame slip angle from state**: from the current twist, the plant recovers an effective slip-like angle
  $$\beta_{prev} = \arctan2(v_y, v_x).$$
- **Curvature / yaw-rate factor**: given wheelbase $L$, the plant uses
  $$\omega(\beta) = \frac{2\sin(\beta)}{L}$$
  and evaluates it at both $\beta(\delta)$ and $\beta_{prev}$:
  - $\omega_t = \omega\big(\beta(\delta_t)\big)$ (current curvature from commanded steering),
  - $\omega_{t-1} = \omega(\beta_{prev})$ (curvature implied by the previous velocity direction).

### Translational update (in a slip-aligned frame)

Conceptually, `MushrPlant.xdot` does the following for translation:

1. **Transform to a slip-aligned frame** using an SE(2) transform built from $\beta_{prev}$.
2. **Apply commanded acceleration** $a$ along the local longitudinal axis for one time step using an Euler update:
   $$v_{x,t+1} \approx v_{x,t} + a_t \Delta t, \quad v_{y,t+1} \approx v_{y,t},$$
   in that slip-aligned frame.
3. **Transform back** to the original body frame to obtain a candidate updated twist $\dot{x}_{t+1}^{(0)}$ without yaw memory or friction.

In code this happens through:

- an adjoint map into the slip frame,
- construction of an acceleration vector $[a_t, 0, 0]^T$,
- Euler integration in that frame,
- and an adjoint map back to the body frame.

### Yaw update and "friction" hook

The plant then updates a yaw-related component using the curvature terms and a scalar `friction` factor:

1. **Speed magnitudes** (with a sign tied to the commanded acceleration):
   $$V_{prev} = \operatorname{sign}(a_t)\,\lVert[v_x, v_y]\rVert, \quad
     V_{curr} = \operatorname{sign}(a_t)\,\lVert[v_{x,t+1}^{(0)}, v_{y,t+1}^{(0)}]\rVert.$$
2. **Yaw-like increments** before friction:
   $$\theta'_{prev} = \omega_{t-1} V_{prev}, \quad
     \theta'_{curr} = \omega_t V_{curr}.$$
3. **Friction-scaled yaw increment**:
   - The plant constructs a vector
     $$w_{new} = [0,\; 0,\; (\theta'_{curr} - \theta'_{prev}) \cdot k_f]^T,$$
     where **$k_f$ is the `friction` input to the plant**.
   - If no friction is provided, $k_f = 1$ and the yaw change is purely determined by the geometry and speeds.

Finally, the updated twist is
$$\dot{x}_{t+1} = \dot{x}_{t+1}^{(0)} + w_{new} + r_t,$$
where $r_t$ is an **optional residual vector** discussed below.

### Optional hooks in the plant (where learning can intervene)

`MushrPlant.xdot` exposes four key hooks that higher-level models can choose to learn:

- **`delta_override` (optional, learned)**:
  - If provided, the plant ignores `ut[..., steering_idx]` and uses the override as the effective steering $\delta_t$.
  - This is how the learned `ControlAdapter` can replace the raw steering command with a state- and command-dependent effective steering.
- **`acc_override` (optional, learned)**:
  - Similarly replaces the raw acceleration / velocity command with a learned effective longitudinal command.
- **`friction` (optional, learned)**:
  - A scalar (per sample) factor $k_f$ that multiplies the yaw increment $\theta'_{curr} - \theta'_{prev}$.
  - When learned, this acts as a state- and input-dependent yaw gain/friction term.
- **`residual` (optional, learned)**:
  - A 3D residual vector $r_t$ added directly to the final twist $\dot{x}_{t+1}$.
  - Used to model unmodeled effects (e.g., contact nonlinearities, higher-order dynamics) that are not captured by the kinematic structure.

All learnable components in the current system are built around producing these optional overrides, scalars, or residuals.

## Learned components built on top of the plant

All learnable models live in `models/system_models.py` and are used by the training pipeline in `train.py`.

### ControlAdapter (optional learned control preprocessing)

**Role**: learn an effective mapping
$$u^{eff}_t = f_{CA}(\dot{x}_t, u_t)$$
that replaces raw commands with commands that better reflect how the hardware actually responds.

- **Inputs** per sample:
  - Raw controls: $a^{cmd}_t, \delta^{cmd}_t$.
  - State: $v_x, v_y, \omega$.
  - Optional extras (enabled by config flags):
    - speed magnitude $|v|$,
    - speed sign $\operatorname{sign}(v_x)$,
    - optionally $v_y$ and $\omega$ for the acceleration net.
- **Outputs**:
  - Learned effective steering $\delta^{eff}_t$.
  - Learned effective longitudinal command $a^{eff}_t$.
  - These can be optionally squashed with `tanh` and scaled by user-provided factors, e.g. `steer_output_scale`, `acc_output_scale`.
- **Where it plugs in**:
  - `StructuredDynamicsModel` and `DirectDynamicsModel` can both accept an optional `ControlAdapter`.
  - In the structured case, $a^{eff}_t$ and $\delta^{eff}_t$ are passed as `acc_override` and `delta_override` to the plant.

This is an **optional learned component**: configs can enable or disable it.

### Learned friction scaler (optional)

In the structured model, the plant’s `friction` hook is driven by a small neural network when `learn_friction` is enabled.

- **Model**: a network $f_{fric}$ with inputs
  $$z_t = [\omega_{t-1}, \; (\delta^{eff}_t)?, \; (v_y)?, \; (\Delta t)?],$$
  where the last three entries are included or skipped based on `friction_use_delta`, `friction_use_vy`, and `friction_use_dt`.
- **Output**: a scalar per sample
  $$k_f = 1 + \operatorname{softplus}(f_{fric}(z_t)) \;\ge 1.$$
- **Where it plugs in**: this $k_f$ is passed as the `friction` argument to `MushrPlant.xdot`, scaling the yaw increment.

This component is **optional and controlled by `learn_friction` and feature flags** in the config.

### Residual dynamics (optional)

The structured model can also learn a residual twist $r_t$ added on top of the plant’s prediction.

- **Model**: a network $f_{res}$ with inputs
  $$[\dot{x}_t, u^{eff}_t] \in \mathbb{R}^5$$
  and output
  $$r_t = f_{res}(\dot{x}_t, u^{eff}_t) \in \mathbb{R}^3.$$
- **Where it plugs in**: this is passed as the `residual` argument to `MushrPlant.xdot` and added directly to the final twist.

Again, this component is **optional and controlled by `learn_residual` in the config**.

### StructuredDynamicsModel (plant + optional learnables)

`StructuredDynamicsModel` composes all of the above pieces around the fixed plant:

1. Optionally adapt controls with `ControlAdapter` to get $u^{eff}_t$.
2. Compute geometry-based quantities from state (e.g., $\beta_{prev}$, $\omega_{t-1}$).
3. Optionally compute a friction scaler $k_f$ with the friction net.
4. Optionally compute a residual $r_t$ with the residual net.
5. Call `MushrPlant.xdot` with:
   - `delta_override` = $\delta^{eff}_t$ (if adapter present),
   - `acc_override` = $a^{eff}_t$ (if adapter present),
   - `friction` = $k_f$ (if friction net enabled),
   - `residual` = $r_t$ (if residual net enabled).

This yields a dynamics model that preserves the geometric plant structure, but can **optionally learn**:

- control effectiveness (via the adapter),
- yaw gain/friction (via the friction net),
- small unmodeled residual effects (via the residual net).

### DirectDynamicsModel (fully learned, optional adapter)

`DirectDynamicsModel` is a pure MLP that predicts the next twist without using `MushrPlant` internally.

- **Inputs**:
  - Current twist $\dot{x}_t$.
  - Either raw commands $u_t$ or adapted commands $u^{eff}_t$ if a `ControlAdapter` is attached.
  - Time step $\Delta t$.
- **Output**: a 3D vector $\dot{x}_{t+1}$ directly from the network.

The optional adapter plays the same role as in the structured model (learning how the hardware maps commands to effective actions), but the rest of the dynamics is entirely learned.

## Config-by-config behavior, learning, and results

All configs live in `configs/` and are consumed by `scripts/train.py`.
Below we summarize, for each config, which parts are **fixed** and which are **learned**, how they interact with the plant, and **how well they perform** in the provided experiment runs.

### Summary table (structure only)

The structural summary is identical to `system_and_configs.md`:

| Config file                           | Model type   | Control adapter | Learn friction | Friction features                 | Learn residual | High-level behavior |
|---------------------------------------|--------------|-----------------|----------------|-----------------------------------|----------------|----------------------|
| `config_structured_minimal.json`      | structured   | disabled        | no             | N/A                               | yes            | Plant + small residual only |
| `config_structured_adapter_only.json` | structured   | enabled         | no             | N/A                               | no             | Plant + learned control effectiveness |
| `config_structured_friction_only.json`| structured   | enabled         | yes            | `dt`, `delta`, `vy`               | no             | Plant + control adapter + learned yaw gain/friction |
| `config_structured_residual_only.json`| structured   | enabled         | no             | N/A                               | yes            | Plant + control adapter + residual corrections |
| `config_structured.json`              | structured   | enabled         | yes            | `dt`, `delta`, `vy`               | yes            | Full structured model: adapter + friction + residual |
| `config_control_adapter.json`         | structured   | enabled         | no             | N/A                               | no             | Control-adapter-only structured model |
| `config_direct_no_adapter.json`       | direct       | disabled        | N/A            | N/A                               | N/A            | Pure MLP dynamics on raw commands |
| `config_direct.json`                  | direct       | enabled         | N/A            | N/A                               | N/A            | Pure MLP dynamics with learned control adapter |

### Summary table (with empirical metrics)

From the runs in `experiments/` (each with a single `losses.csv`), we extract:

- **`best_val_loss`**: minimum validation loss over all epochs.
- **`best_eval_vel_mse`** / **`best_eval_pos_mse`**: minimum non-NaN held-out trajectory MSE for velocity / position.

| Config / experiment dir                      | Metrics (best\_val\_loss, best\_eval\_vel\_mse, best\_eval\_pos\_mse)                          | Qualitative summary                                                   |
|----------------------------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| **structured** (`experiments/structured/...`)             | `≈ (7.8e-03, 3.8e-04, 6.7e-04)`                                                                 | Competes closely with direct models; very low error overall.          |
| **structured_friction_only**                            | `≈ (5.18, 4.7e-01, 4.0e-01)`                                                                    | Much worse than other setups; friction learning alone is insufficient.|
| **control_adapter** (`structured_adapter_only`)         | `≈ (2.70, 2.0e-01, 3.7e-01)`                                                                    | Better than friction-only, but still far from best-performing configs.|
| **structured_residual_only**                            | `≈ (8.6e-03, 4.1e-04, 7.2e-04)`                                                                | Very similar performance to full structured model.                    |
| **structured_adapter_only**                             | same metrics as `control_adapter` above                                                         | Same behavior as `control_adapter` structurally, trained on the same logged time steps. |
| **direct**                                              | `≈ (7.2e-03, 3.9e-04, 8.5e-04)`                                                                | Among the very best models; slightly edges structured variants in val loss. |
| **structured_minimal**                                  | `≈ (6.1e-02, 7.2e-03, 7.4e-02)`                                                                | Significantly worse than richer structured models, but better than friction-only. |
| **direct_no_adapter**                                   | `≈ (7.4e-03, 3.9e-04, 9.0e-04)`                                                                | On par with `direct`, slightly behind in position MSE.                |

Below, we restate the qualitative config descriptions and append a short **Results** paragraph for each.

### `config_structured_minimal.json`

- **Model type**: `structured`.
- **Learned pieces**:
  - `learn_residual: true` → residual net $f_{res}$ is enabled.
  - `learn_friction: false` → friction scaler is fixed to 1.
  - `control_adapter.enabled: false` → raw commands are fed directly to the plant.
- **What it learns**:
  - A residual twist $r_t = f_{res}(\dot{x}_t, u_t)$ that corrects the plant’s prediction.
- **What it does in practice**:
  - Uses the exact `MushrPlant` equations for kinematics and yaw, with no learned adjustment of steering, acceleration, or yaw gain.
  - Only learns small additive corrections to match data.
  - This is the most conservative structured model: good for testing how far the bare plant plus a minimal residual can go.
- **Results (`experiments/structured_minimal/...`)**:
  - $\text{best\_val\_loss} \approx 6.1 \times 10^{-2}$.
  - $\text{best\_eval\_vel\_mse} \approx 7.2 \times 10^{-3}$, $\text{best\_eval\_pos\_mse} \approx 7.4 \times 10^{-2}$.
  - Clearly worse than richer structured and direct models, but still captures basic trends; acts as a lower bound on what the bare plant + residual can achieve.

### `config_structured_adapter_only.json`

- **Model type**: `structured`.
- **Learned pieces**:
  - `control_adapter.enabled: true` with all feature flags on and nontrivial output scales.
  - `learn_friction: false`, `learn_residual: false`.
- **What it learns**:
  - A mapping from $(\dot{x}_t, a^{cmd}_t, \delta^{cmd}_t)$ and additional speed-based features to effective commands $u^{eff}_t = [a^{eff}_t, \delta^{eff}_t]$.
- **What it does in practice**:
  - The plant still provides all geometric structure and yaw behavior.
  - The adapter is solely responsible for modeling actuator nonlinearities and state-dependent effectiveness (e.g., limited steering at high speed, asymmetry between forward and reverse, etc.).
  - No additional yaw gain or residual corrections are learned.
- **Results (`experiments/structured_adapter_only/...`)**:
  - Same run as `control_adapter` structurally: $\text{best\_val\_loss} \approx 2.7$, $\text{best\_eval\_vel\_mse} \approx 2.0 \times 10^{-1}$, $\text{best\_eval\_pos\_mse} \approx 3.7 \times 10^{-1}$.
  - Performs substantially worse than the best structured and direct models, indicating that learning only control effectiveness (without residuals or friction scaling) is not sufficient on this dataset.

### `config_structured_friction_only.json`

- **Model type**: `structured`.
- **Learned pieces**:
  - `control_adapter.enabled: true` (as in `config_structured_adapter_only.json`).
  - `learn_friction: true` with `friction_use_dt`, `friction_use_delta`, `friction_use_vy` all true.
  - `learn_residual: false`.
- **What it learns**:
  - An effective control map $u^{eff}_t$ via the adapter.
  - A friction scaler $k_f = 1 + \operatorname{softplus}(f_{fric}(\omega_{t-1}, \delta^{eff}_t, v_y, \Delta t))$ that multiplies the yaw increment.
- **What it does in practice**:
  - Uses plant geometry, but adjusts both how commands are interpreted and how aggressive yaw updates are as a function of speed, steering, and $\Delta t$.
  - No residual term is present, so all mismatch must be explained by command effectiveness and yaw gain alone.
- **Results (`experiments/structured_friction_only/...`)**:
  - $\text{best\_val\_loss} \approx 5.2$.
  - $\text{best\_eval\_vel\_mse} \approx 4.7 \times 10^{-1}$, $\text{best\_eval\_pos\_mse} \approx 4.0 \times 10^{-1}$.
  - This is one of the worst-performing setups; learning friction alone (even with an adapter) does not capture the dynamics well for these trajectories.

### `config_structured_residual_only.json`

- **Model type**: `structured`.
- **Learned pieces**:
  - `control_adapter.enabled: true`.
  - `learn_residual: true`.
  - `learn_friction: false`.
- **What it learns**:
  - A control adapter $u^{eff}_t$ as before.
  - A residual twist $r_t = f_{res}(\dot{x}_t, u^{eff}_t)$.
- **What it does in practice**:
  - Keeps fixed plant yaw gain (no friction learning).
  - Uses the adapter to model actuator behavior, and the residual net to capture any remaining modeling errors (e.g., lateral slip effects, higher-order dynamics).
  - Compared to `structured_friction_only`, this shifts modeling power from yaw scaling into a more general residual term.
- **Results (`experiments/structured_residual_only/...`)**:
  - $\text{best\_val\_loss} \approx 8.6 \times 10^{-3}$.
  - $\text{best\_eval\_vel\_mse} \approx 4.1 \times 10^{-4}$, $\text{best\_eval\_pos\_mse} \approx 7.2 \times 10^{-4}$.
  - Very close in performance to the full structured model and the direct models; residual learning appears far more valuable than friction learning alone.

### `config_structured.json`

- **Model type**: `structured`.
- **Learned pieces**:
  - `control_adapter.enabled: true`.
  - `learn_friction: true` with `friction_use_dt`, `friction_use_delta`, `friction_use_vy` all true.
  - `learn_residual: true`.
- **What it learns**:
  - Control adapter $u^{eff}_t$.
  - Friction scaler $k_f$ as a function of $\omega_{t-1}, \delta^{eff}_t, v_y, \Delta t$.
  - Residual twist $r_t$.
- **What it does in practice**:
  - This is the **full structured model**, using every optional learnable hook in the plant.
  - It can simultaneously:
    - correct actuator commands,
    - adjust yaw gain/friction in a state- and command-dependent way,
    - and model small remaining discrepancies with a residual.
  - In the sysid experiments, it is trained and evaluated using the logged per-sample `dt` from the dataset, just like the other configs.
- **Results (`experiments/structured/...`)**:
  - $\text{best\_val\_loss} \approx 7.8 \times 10^{-3}$.
  - $\text{best\_eval\_vel\_mse} \approx 3.8 \times 10^{-4}$, $\text{best\_eval\_pos\_mse} \approx 6.7 \times 10^{-4}$.
  - Performance is essentially on par with the best direct models; adding both friction and residuals on top of the adapter yields a highly accurate yet structured model.

### `config_control_adapter.json`

- **Model type**: `structured`.
- **Learned pieces**:
  - `control_adapter.enabled: true` with the same feature choices as `config_structured.json`.
  - `learn_friction: false`, `learn_residual: false`.
- **What it learns**:
  - Only the control adapter $u^{eff}_t$, with no learned friction or residual.
- **What it does in practice**:
  - Structurally similar to `structured_adapter_only`, but using the same dataset pipeline as the other configs.
  - It focuses on learning actuator behavior while keeping the plant’s yaw dynamics fully fixed.
  - Useful as a clean baseline when you only want to see the benefit of learned control effectiveness at a fine time scale.
- **Results (`experiments/control_adapter/...`)**:
  - Identical metrics to `structured_adapter_only` in this experiment: $\text{best\_val\_loss} \approx 2.7$, $\text{best\_eval\_vel\_mse} \approx 2.0 \times 10^{-1}$, $\text{best\_eval\_pos\_mse} \approx 3.7 \times 10^{-1}$.
  - As with the 0.1-s version, learning only the adapter without residuals or friction is not enough to reach low error on these trajectories.

### `config_direct_no_adapter.json`

- **Model type**: `direct`.
- **Learned pieces**:
  - A pure MLP dynamics model; `control_adapter.enabled: false`.
  - No friction or residual hooks, because the plant is not used.
- **What it learns**:
  - A mapping
    $$\dot{x}_{t+1} = f_\theta(\dot{x}_t, u_t, \Delta t)$$
    directly from data.
- **What it does in practice**:
  - Ignores the hand-designed plant entirely.
  - Learns the entire dynamics (including any effective control nonlinearity) inside a single network.
  - Serves as a fully data-driven baseline using the same logged per-sample time steps as the structured models, with no hard kinematic constraints.
- **Results (`experiments/direct_no_adapter/...`)**:
  - $\text{best\_val\_loss} \approx 7.4 \times 10^{-3}$.
  - $\text{best\_eval\_vel\_mse} \approx 3.9 \times 10^{-4}$, $\text{best\_eval\_pos\_mse} \approx 9.0 \times 10^{-4}$.
  - Very strong performance: close to the best structured model, despite lacking any hard kinematic structure or control adapter.

### `config_direct.json`

- **Model type**: `direct`.
- **Learned pieces**:
  - Same pure MLP dynamics as `config_direct_no_adapter.json`.
  - `control_adapter.enabled: true` with the same feature set as the structured configs.
- **What it learns**:
  - A control adapter $u^{eff}_t = f_{CA}(\dot{x}_t, u_t)$.
  - A dynamics network $f_\theta$ that maps $(\dot{x}_t, u^{eff}_t, \Delta t)$ to $\dot{x}_{t+1}$.
- **What it does in practice**:
  - Still ignores the hand-designed plant; everything is learned.
  - The adapter and dynamics net can distribute complexity between control interpretation and state evolution.
  - In the sysid experiments, it is likewise trained and evaluated using the logged per-sample `dt` from the dataset.
- **Results (`experiments/direct/...`)**:
  - $\text{best\_val\_loss} \approx 7.2 \times 10^{-3}$ (the lowest among all runs).
  - $\text{best\_eval\_vel\_mse} \approx 3.9 \times 10^{-4}$, $\text{best\_eval\_pos\_mse} \approx 8.5 \times 10^{-4}$.
  - This is marginally the best-performing setup overall in these experiments, slightly edging out the full structured model, but at the cost of giving up the explicit plant structure.
