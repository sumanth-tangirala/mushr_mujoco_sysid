# Clarifications on Methodology and Metrics

## Question 1: What Does "Worst-Case" Mean?

### Understanding the Different "Worst" Metrics

We report **two different types of "worst" metrics**:

#### 1. `worst_pos_error_mean_m` (Worst-Case Average Over Trajectories)

**Definition**: For each trajectory, find the maximum position error at any single timestep. Then average these maxima across all 100 trajectories.

**Formula**:
```
For trajectory i:
    worst_error_i = max(error[t] for all timesteps t in trajectory i)

worst_pos_error_mean = mean(worst_error_i for all trajectories i)
```

**Physical Meaning**: "On average, what's the worst position error you'll see at any point during a trajectory?"

**Example**:
- Trajectory 1: worst timestep has 0.15m error
- Trajectory 2: worst timestep has 0.20m error
- Trajectory 3: worst timestep has 0.12m error
- → Mean = (0.15 + 0.20 + 0.12) / 3 = 0.157m

**Results**:
- **Polynomial**: 4.05m (on average, drifts up to 4m at some point)
- **Neural Network**: 0.16m (on average, worst error is only 16cm)

---

#### 2. `worst_pos_error_max_m` (Absolute Worst Across Everything)

**Definition**: The single worst position error across ALL trajectories and ALL timesteps.

**Formula**:
```
worst_pos_error_max = max(error[t] for all timesteps t in all trajectories)
```

**Physical Meaning**: "What's the absolute worst prediction error we ever saw?"

**Example**:
- If one trajectory had a catastrophic 16m error at one timestep (while all others were fine), this would be reported as 16m.

**Results**:
- **Polynomial**: 16.07m (one trajectory diverged catastrophically)
- **Neural Network**: 0.60m (even worst case is only 60cm)

---

### Summary of "Worst" Metrics

| Metric | What It Measures | Polynomial | Neural Net |
|--------|------------------|-----------|-----------|
| `avg_pos_error_mean` | Average error over full trajectory | 1.58m | 0.08m |
| `worst_pos_error_mean` | Average of worst single-timestep errors | 4.05m | 0.16m |
| `worst_pos_error_max` | Absolute worst single timestep ever | 16.07m | 0.60m |

**Key Insight**: The neural network's **absolute worst case** (0.60m) is better than the polynomial's **typical worst** (4.05m mean).

---

## Question 2: What Is the Neural Network Predicting and How Was It Trained?

### What the Neural Network Predicts

The neural network is a **dynamics model** that predicts the next velocity state given the current velocity state and control input.

#### Input (at timestep t):
```
state:    [vx, vy, w]        # Body-frame velocities (m/s, m/s, rad/s)
control:  [vel_cmd, steer]   # Commanded velocity and steering angle
dt:       scalar              # Timestep duration (typically 0.1s)
```

#### Output (prediction for timestep t+1):
```
next_state: [vx', vy', w']   # Predicted next velocity state
```

#### Full Trajectory Rollout:
1. Start with ground truth initial state: `pose[0] = (x, y, θ)`, `vel[0] = (vx, vy, w)`
2. For each timestep `t`:
   - Predict next velocity: `vel[t+1] = NN(vel[t], control[t], dt[t])`
   - Integrate to get next pose: `pose[t+1] = integrate(pose[t], vel[t+1], dt[t])`
3. Continue until end of trajectory (no ground truth used after initialization)

**Key Point**: This is **open-loop prediction**. After the initial state, the model predicts the entire trajectory without any ground truth corrections.

---

### Neural Network Architecture

#### Type: Structured Dynamics Model (v3B_struct_h10_tf0_seed4)

**Architecture Components**:

1. **Physics-Based Structure**:
   - Kinematic bicycle model equations embedded in forward pass
   - Provides inductive bias about vehicle dynamics

2. **Control Adapter**:
   - Small neural network that learns transformations of control inputs
   - Captures nonlinear actuator response, saturation, delays
   - Input features: velocity magnitude/sign, steering, lateral velocity, angular velocity
   - Output: Adapted steering and acceleration commands

3. **Residual Network**:
   - MLP that predicts corrections to physics model
   - Captures unmodeled effects: tire slip, friction variations, model errors
   - Added to physics predictions to get final output

**Formula**:
```
vel_next = physics_model(vel, adapted_control, dt) + residual_network(vel, control, dt)
```

---

### Training Procedure

#### Dataset: v3 System Identification Trajectories
- **Source**: MuSHR vehicle in MuJoCo simulator
- **Size**: ~3000 trajectories with varying control sequences
- **Format**: (x, y, θ, vx, vy, w, steering_cmd, velocity_cmd) at 10Hz
- **Split**: 90% train, 10% validation (trajectory-level split)

#### Training Configuration:
```
Model:     v3B_struct_h10_tf0_seed4
Epochs:    180 (early stopping from max 1000)
Batch:     256 samples
Optimizer: Adam with lr=0.001
Scheduler: Cosine annealing (eta_min=1e-6)
Seed:      4
```

#### Loss Function: Multi-Component

1. **One-Step MSE** (weight=1.0):
   - Predict next state from current state
   - Standard supervised learning
   - `Loss = ||vel_pred - vel_gt||^2`

2. **Rollout MSE** (weight=0.5, horizon=10):
   - Unroll predictions for 10 steps
   - Model predicts, then uses its own predictions (no teacher forcing)
   - Penalizes accumulation of errors over time
   - `Loss = Σ ||vel_pred[t] - vel_gt[t]||^2` for t=1..10

3. **Pose MSE** (weight=0.05):
   - Integrate predicted velocities to get pose
   - Penalize position (x, y) errors
   - Ensures physically consistent predictions
   - `Loss = ||pose_pred - pose_gt||^2`

4. **Component Weighting**:
   - vx, vy: weight=1.0
   - w (angular velocity): weight=2.0 (more important for stability)

**Total Loss**:
```
Loss = 1.0 * one_step_mse + 0.5 * rollout_mse + 0.05 * pose_mse
```

---

### Key Training Insights

#### Why This Configuration Works:

1. **No Teacher Forcing (TF=0.0)**:
   - During rollout loss, model uses its own predictions
   - Forces model to be robust to its own errors
   - Better generalization at test time

2. **Rollout Horizon=10**:
   - Balances short-term accuracy with long-term stability
   - h=20 caused training instability (v3C/v3F experiments)
   - h=10 is the sweet spot for this dataset

3. **Structured Architecture**:
   - Physics inductive bias helps with data efficiency
   - Residuals allow model to correct physics errors
   - Control adapter handles actuator nonlinearities

4. **Multi-Step Rollout Loss**:
   - Critical for preventing error accumulation
   - Without this, model only learns one-step accuracy
   - Rollout loss teaches model to be stable over time

---

### Evaluation Methodology

#### How We Test:

1. **Select 100 random test trajectories** (seed=42, not seen during training)

2. **Initialize** with ground truth at t=0:
   ```
   pose[0] = (x_gt[0], y_gt[0], θ_gt[0])
   vel[0] = (vx_gt[0], vy_gt[0], w_gt[0])
   ```

3. **Rollout** full trajectory using only predicted states:
   ```python
   for t in range(trajectory_length):
       vel[t+1] = model.predict(vel[t], control[t], dt[t])
       pose[t+1] = integrate(pose[t], vel[t+1], dt[t])
   ```

4. **Compare** predictions to ground truth at every timestep

5. **Compute errors**:
   - Position: `sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)`
   - Velocity: `sqrt((vx_pred - vx_gt)^2 + (vy_pred - vy_gt)^2 + (w_pred - w_gt)^2)`

6. **Aggregate** over trajectories to get final metrics

---

### Why Neural Network Beats Polynomial Model

| Aspect | Polynomial Model | Neural Network |
|--------|-----------------|----------------|
| **Parameters** | Fixed/hardcoded | Learned from data |
| **Friction** | Constant (1.0) | State-dependent (learned) |
| **Control** | Linear/polynomial mapping | Nonlinear adapter (learned) |
| **Residuals** | None | Learned corrections |
| **Adaptation** | Cannot adapt | Generalizes from training |
| **Error Handling** | Accumulates errors | Rollout loss prevents accumulation |

**Bottom Line**: The neural network learns the actual dynamics from data, including all the messy nonlinear effects that are hard to model with fixed equations.

---

## File Locations

- **Summary CSV**: `summary_comparison.csv` (2 rows, 10 columns)
- **Detailed CSV**: `detailed_comparison.csv` (100 rows, per-trajectory)
- **Full Results**: `comparison_results.json`
- **Visualizations**: `all_trajectories_comparison.png` + 100 individual plots
- **This Document**: `CLARIFICATIONS.md`
