# LaserFind

**AI-powered gimbal tracking system using reinforcement learning and computer vision**

LaserFind trains an intelligent pan-tilt gimbal controller to automatically track moving objects in 3D space. The system uses NVIDIA Isaac Lab to train policies via massively parallel reinforcement learning, then deploys them for real-time visual tracking with multi-sensor fusion.

**Current Focus:** Isaac Lab RL training pipeline (Phase 1)

---

## Core Concept

Train a neural network policy to control a 2-DOF gimbal (pan/tilt servos) that can:
1. Track a moving object smoothly and accurately
2. Handle multiple targets with intelligent prioritization
3. Predict motion and lead the target
4. Recover gracefully from tracking loss

**Why RL instead of classical control?**
- Learns optimal policies from experience, not hand-tuned PIDs
- Handles non-linear dynamics and complex behaviors
- Generalizes to unseen scenarios through simulation diversity
- More technically impressive and novel

---

## Technical Architecture

### Phase 1: Isaac Lab Training (CURRENT PRIORITY)
```
Isaac Lab Environment (4096 parallel)
    ↓
Drone Tracking Simulation
    ↓
PPO Reinforcement Learning
    ↓
Trained Policy (ONNX)
```

### Phase 2: Deployment (Future)
```
Camera Feed → YOLOv8 Detection → Position Estimation
    ↓
Multi-Camera Fusion (Kalman Filter)
    ↓
RL Policy Inference (ONNX)
    ↓
Servo Commands → Gimbal Movement
```

---

## Phase 1: Isaac Lab Training System

**This is what we're building RIGHT NOW.**

### Environment Specifications

**File:** `isaac_lab_training/environments/drone_tracking_env.py`

Create an Isaac Lab `DirectRLEnv` with:

#### Scene Setup
- **Gimbal**: 2-DOF articulated body (pan joint + tilt joint)
  - Pan range: [-180°, +180°]
  - Tilt range: [-90°, +90°]
  - Max angular velocity: 180°/s (3.14 rad/s)
  - Mounted at origin (0, 0, 1m above ground)

- **Target Object**: Moving rigid body (sphere/cone, 0.3m diameter)
  - Spawns randomly 3-10m away from gimbal
  - Follows programmed behavior patterns (see below)
  - Physics-based movement (forces/velocities)

- **Environment**: Ground plane, basic lighting, no obstacles (start simple)

#### Observation Space (14 dimensions)
```python
[
    target_x, target_y, target_z,           # Target position (world frame) [3]
    target_vx, target_vy, target_vz,        # Target velocity [3]
    gimbal_pan, gimbal_tilt,                 # Current gimbal angles [2]
    gimbal_pan_vel, gimbal_tilt_vel,        # Current gimbal velocities [2]
    angle_error_pan, angle_error_tilt,      # Tracking error (aim vs target) [2]
    distance_to_target,                      # Euclidean distance [1]
    tracking_error_magnitude                 # sqrt(error_pan² + error_tilt²) [1]
]
```

#### Action Space (2 dimensions, continuous)
```python
[
    pan_angular_velocity,    # Normalized [-1, 1] → maps to [-3.14, +3.14] rad/s
    tilt_angular_velocity    # Normalized [-1, 1] → maps to [-3.14, +3.14] rad/s
]
```

#### Reward Function
```python
reward = (
    -10.0 * tracking_error_magnitude          # Primary: minimize angular error
    - 0.1 * action_magnitude                  # Smoothness: penalize large actions
    + 1.0 * stability_bonus                   # Bonus if error < 0.1 rad (5.7°)
    - 0.05 * velocity_penalty                 # Penalize excessive gimbal speed
)

# Stability bonus only given when tracking_error < 0.1 rad
stability_bonus = 1.0 if tracking_error < 0.1 else 0.0

# Velocity penalty quadratic to encourage smooth movements
velocity_penalty = (gimbal_pan_vel² + gimbal_tilt_vel²)
```

#### Episode Configuration
- **Number of parallel environments**: 4096
- **Episode length**: 30 seconds (3000 steps at 100Hz physics)
- **Reset conditions**:
  - Timeout (30 seconds elapsed)
  - Target out of bounds (>20m from origin)
  - Gimbal limits exceeded (rare, should be prevented by action clipping)

#### Episode Reset Behavior
On reset, randomize:
1. **Target starting position**: Random point on sphere, radius 3-10m
2. **Target starting velocity**: Random direction, 0-2 m/s
3. **Target behavior type**: Select from behavior library (see below)
4. **Gimbal starting angles**: Near-zero with small random offset (±10°)

---

### Target Behavior Library

**File:** `isaac_lab_training/environments/target_behaviors.py`

Implement 7 behavior patterns with increasing difficulty:

#### 1. Stationary (Easiest)
- Target hovers in place
- Small random drift (wind simulation): ±0.1 m/s
- **Use case**: Initial training, baseline validation

#### 2. Constant Velocity
- Straight line movement
- Random direction, constant speed 1-3 m/s
- **Use case**: Basic tracking, smooth pursuit

#### 3. Circular Orbit
- Orbit around a point at fixed radius (3-7m)
- Angular velocity: 0.3-0.7 rad/s
- **Use case**: Predictable but continuous movement

#### 4. Figure-8 (Lissajous Curve)
```python
x(t) = A * sin(ωt)
y(t) = B * sin(2ωt)
z(t) = C + D * sin(ωt)
```
- Smooth acceleration/deceleration
- **Use case**: Complex but predictable trajectory

#### 5. Random Jinking
- Base velocity + random acceleration changes every 0.2s
- Max acceleration: 5 m/s²
- **Use case**: Unpredictable movement, test robustness

#### 6. Approach Pattern
- Move toward a target point (origin) at constant speed
- Speed: 2-4 m/s
- **Use case**: Simulates incoming object

#### 7. Spiral Approach
- Combine radial inward movement + tangential orbit
- Decreasing radius spiral toward origin
- **Use case**: Most complex, combines multiple motion types

---

### Curriculum Learning Strategy

**File:** `isaac_lab_training/environments/curriculum_scheduler.py`

Progress through difficulty stages during training:

| Training Steps | Behavior Distribution | Complexity |
|----------------|----------------------|------------|
| 0 - 1M | 100% Stationary | 0.0 |
| 1M - 3M | 70% Stationary, 30% Constant Velocity | 0.2 |
| 3M - 5M | 50% Constant Velocity, 30% Circular, 20% Stationary | 0.4 |
| 5M - 7M | 40% Circular, 30% Figure-8, 30% Constant Velocity | 0.6 |
| 7M - 9M | 30% Figure-8, 30% Jinking, 40% Mixed | 0.8 |
| 9M - 10M | All behaviors equally (14% each) | 1.0 |

**Implementation**: At each episode reset, sample behavior type based on current training step.

---

### PPO Training Configuration

**File:** `isaac_lab_training/training/train_gimbal_controller.py`

#### Hyperparameters
```python
algorithm: PPO
learning_rate: 3e-4
n_steps: 2048                    # Steps per env before update
batch_size: 2048                 # Minibatch size for optimization
n_epochs: 10                     # Optimization epochs per update
gamma: 0.99                      # Discount factor
gae_lambda: 0.95                 # GAE parameter
clip_range: 0.2                  # PPO clipping parameter
ent_coef: 0.0                    # No entropy bonus (deterministic preferred)
vf_coef: 0.5                     # Value function coefficient
max_grad_norm: 0.5               # Gradient clipping
```

#### Training Setup
```python
total_timesteps: 10_000_000      # 10M steps total
num_envs: 4096                   # Parallel environments
steps_per_update: 2048 * 4096    # ~8.4M environment steps per update
```

#### Neural Network Architecture
```python
policy_network:
  type: MLP
  hidden_layers: [256, 256, 128]
  activation: ReLU
  output_activation: Tanh (for continuous actions)

value_network:
  type: MLP
  hidden_layers: [256, 256, 128]
  activation: ReLU
  output_activation: Linear
```

#### Observation Normalization
**CRITICAL**: Use `VecNormalize` wrapper
```python
normalize_observations: true
normalize_rewards: true
clip_observations: 10.0
clip_rewards: 10.0
```

Save normalization statistics (mean/std) for deployment!

#### Training Loop
1. Collect experience from all 4096 environments
2. Every 2048 steps per env → optimization update
3. Checkpoint every 100k steps
4. Log metrics to TensorBoard: mean reward, episode length, value loss, policy loss
5. Total training time estimate: **2-4 hours on RTX 3090**

---

### Policy Export

**File:** `isaac_lab_training/export/export_policy.py`

After training completes:

1. **Load trained model**: Load final PPO checkpoint
2. **Extract policy network**: Separate actor from critic
3. **Export to ONNX**:
```python
torch.onnx.export(
    policy_network,
    dummy_observation,
    "gimbal_policy.onnx",
    input_names=["observation"],
    output_names=["action"],
    dynamic_axes={"observation": {0: "batch_size"}},
    opset_version=14
)
```
4. **Save normalization parameters**:
```json
{
    "obs_mean": [14-element array],
    "obs_std": [14-element array],
    "clip_obs": 10.0
}
```
5. **Validate exported model**:
   - Load ONNX with onnxruntime
   - Compare outputs: PyTorch vs ONNX
   - Measure inference latency (target: <10ms)

---

## Project Structure (Phase 1 Focus)

```
LaserFind/
├── isaac_lab_training/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── drone_tracking_env.py          # Main Isaac Lab environment
│   │   ├── target_behaviors.py            # 7 behavior patterns
│   │   ├── curriculum_scheduler.py        # Difficulty progression
│   │   └── reward_functions.py            # Modular reward components
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_gimbal_controller.py     # PPO training script
│   │   └── config.yaml                     # Hyperparameters
│   ├── export/
│   │   ├── __init__.py
│   │   ├── export_policy.py               # ONNX export
│   │   └── validate_export.py             # Test exported model
│   ├── tests/
│   │   ├── test_environment.py            # Unit tests for env
│   │   ├── test_behaviors.py              # Validate behaviors
│   │   └── test_reward.py                 # Reward function tests
│   └── requirements.txt
├── README.md
└── .gitignore
```

---

## Dependencies (Phase 1)

```txt
# Core
torch>=2.0.0
numpy>=1.24.0

# Isaac Lab (install separately)
# Follow: https://isaac-sim.github.io/IsaacLab/source/setup/installation.html

# RL Training
stable-baselines3>=2.0.0
tensorboard>=2.13.0
gymnasium>=0.29.0

# Export
onnx>=1.14.0
onnxruntime-gpu>=1.15.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## Implementation Priority Order

### Step 1: Basic Environment (Week 1, Days 1-2)
- [ ] Create `DroneTrackingEnv` class skeleton
- [ ] Implement gimbal articulation (2 revolute joints)
- [ ] Add target rigid body (simple sphere)
- [ ] Define observation space computation
- [ ] Define action space mapping
- [ ] Implement basic reward function
- [ ] Test environment loads and runs

### Step 2: Target Behaviors (Week 1, Days 3-4)
- [ ] Implement all 7 behavior patterns
- [ ] Create `TargetBehaviorLibrary` class
- [ ] Test each behavior in isolation
- [ ] Verify physics looks realistic

### Step 3: Curriculum Scheduler (Week 1, Day 5)
- [ ] Implement `CurriculumScheduler` class
- [ ] Behavior distribution sampling
- [ ] Progress tracking through training

### Step 4: Training Script (Week 1, Days 6-7)
- [ ] PPO training loop with Stable-Baselines3
- [ ] VecNormalize wrapper integration
- [ ] TensorBoard logging
- [ ] Checkpoint saving
- [ ] Training monitoring

### Step 5: First Training Run (Week 2, Days 1-2)
- [ ] Start training with 4096 envs
- [ ] Monitor training curves
- [ ] Validate learning is happening
- [ ] Identify issues, iterate

### Step 6: Policy Export (Week 2, Day 3)
- [ ] ONNX export script
- [ ] Save normalization params
- [ ] Validation tests
- [ ] Inference speed benchmarks

### Step 7: Visualization & Analysis (Week 2, Days 4-5)
- [ ] Plot training curves
- [ ] Evaluate policy in simulation
- [ ] Record demo videos
- [ ] Measure tracking accuracy

---

## Key Metrics to Track

### Training Metrics (TensorBoard)
- Episode reward (mean, std, min, max)
- Episode length
- Tracking error (mean, p50, p95)
- Gimbal action magnitude
- Policy loss, value loss
- Explained variance

### Evaluation Metrics (Post-Training)
- **Tracking accuracy**: Mean angular error across all behaviors
- **Stability**: Percentage of time with error < 5°
- **Smoothness**: Mean absolute angular acceleration
- **Recovery time**: Time to reacquire target after loss
- **Success rate**: Percentage of episodes with mean error < 10°

### Target Performance (After 10M Steps)
| Metric | Target |
|--------|--------|
| Mean tracking error | < 3° |
| Stable tracking (error < 5°) | > 80% of time |
| Policy inference time | < 10ms |
| Training time | 2-4 hours |

---

## Advanced Features (Optional, After Basic System Works)

### Multi-Target Tracking
- Extend environment to 3-5 simultaneous targets
- Add target selection to observation space
- Reward based on tracking highest-priority target
- Learn attention/prioritization behavior

### Attention Mechanism
- Add attention weights to observation
- Visualize which sensors/targets policy focuses on
- Train end-to-end multi-modal fusion

### Domain Randomization
- Randomize lighting conditions
- Add sensor noise to observations
- Vary target sizes and shapes
- Test robustness to distribution shift

### Predictive Tracking
- Add target acceleration to observation space
- Reward leading the target (aim where it will be)
- Handle systematic latency in simulation

---

## Debugging Tips

### Environment Not Learning
- **Check reward scale**: Should be roughly [-50, +10] per step
- **Verify observations are normalized**: Use VecNormalize
- **Plot tracking error over time**: Should decrease
- **Reduce environment complexity**: Start with stationary targets only

### Training Unstable
- **Lower learning rate**: Try 1e-4 instead of 3e-4
- **Increase batch size**: More stable gradients
- **Check for NaN**: Add gradient clipping, check reward computation
- **Verify environment resets**: Target should respawn properly

### Policy Doesn't Transfer Well
- **Domain randomization**: Add noise to observations
- **Curriculum too fast**: Slow down difficulty progression
- **Overfitting**: Train longer with more diverse scenarios

---

## Isaac Lab Installation (Quick Start)

```bash
# 1. Install Isaac Sim (via Omniverse Launcher or standalone)
# Download from: https://developer.nvidia.com/isaac-sim

# 2. Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 3. Create conda environment
./isaaclab.sh --conda

# 4. Install Isaac Lab
./isaaclab.sh --install

# 5. Verify installation
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless
```

---

## Running Training

```bash
# Navigate to project root
cd LaserFind/isaac_lab_training

# Start training (headless mode for speed)
python training/train_gimbal_controller.py \
    --num-envs 4096 \
    --timesteps 10000000 \
    --headless \
    --save-dir ./trained_models \
    --log-dir ./logs

# Monitor with TensorBoard
tensorboard --logdir ./logs
```

---

## Expected Training Results

After 10M timesteps (~2-4 hours on RTX 3090):

### Reward Curve
- Initial: -50 (random policy)
- After 1M steps: -20 (learns basic tracking)
- After 5M steps: -5 (smooth tracking)
- Final: +2 to +5 (optimal policy)

### Behavior Progression
- **0-1M**: Learns to point gimbal toward target
- **1-3M**: Learns smooth pursuit of slow targets
- **3-5M**: Handles circular and figure-8 patterns
- **5-7M**: Begins handling evasive maneuvers
- **7-10M**: Robust tracking across all behaviors

### Visual Indicators of Success
- Gimbal smoothly follows target
- Quick reacquisition after target changes direction
- Minimal oscillation or overshooting
- Stable tracking even during evasive maneuvers

---

## Phase 2 Preview (Not Current Priority)

Once RL training is complete and validated, we'll build:

### Detection System
- YOLOv8 for object detection (30+ FPS)
- Multi-camera position estimation
- Kalman filter for smooth tracking

### Hardware Integration
- Servo controller (Dynamixel or PWM)
- Camera capture and synchronization
- Real-time control loop (30 Hz)

### Deployment
- Load ONNX policy with onnxruntime
- Apply observation normalization
- Convert policy outputs to servo commands
- Visualize tracking in real-time

**But don't worry about Phase 2 yet.** First, we build and validate the RL training system.

---

## Success Criteria (Phase 1)

✅ **Environment loads without errors in Isaac Lab**
✅ **4096 parallel environments run at >100 FPS**
✅ **Training completes 10M steps in 2-4 hours**
✅ **TensorBoard shows clear learning (reward increases)**
✅ **Exported ONNX model inference < 10ms**
✅ **Policy tracks targets with <5° mean error**
✅ **Handles all 7 behavior patterns robustly**
✅ **Gimbal movements are smooth (no jitter)**

---

## Why This Approach Works

1. **Massive parallelization**: 4096 environments = 4096x faster data collection
2. **Curriculum learning**: Start easy, increase difficulty → faster convergence
3. **Rich reward shaping**: Multiple terms guide learning effectively
4. **Proven algorithm**: PPO is stable and well-understood
5. **Domain diversity**: 7 behavior patterns ensure generalization
6. **Export to ONNX**: Portable, fast inference for deployment

This is a **complete, end-to-end RL training system** that produces a deployable gimbal tracking policy.

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions about implementation details or Isaac Lab setup, open an issue in the repository.
