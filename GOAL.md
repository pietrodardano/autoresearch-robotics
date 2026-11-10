---
goal: "Predict chaotic Mackey-Glass time series dynamics with minimum error and minimal parameter complexity."
track: reservoir
task: toy_mackey_glass
learning_rule: closed_form
seeds: [42, 43, 44]
metric: nmse
metric_direction: min
min_gain: 0.01
noise_floor: 0.0001
hard_timeout: 120
frozen:
  - reservoir_weights
  - input_weights
  - bias
constraints:
  max_params: 500000
  max_state_dim: 2000
  max_readout_features: 2000
  max_flops_or_sops: 10000000.0
  max_memory_mb: 4096.0
  max_wall_time_s: 60.0
---

# Project Goal: Autonomous Robotics & Dynamical Systems Research

## Objective
Train and optimize a reservoir computing architecture for continuous state prediction on chaotic dynamics (Mackey-Glass benchmark).

## Paradigm & Track
- **Track**: `reservoir`
- **Frozen**: Reservoir recurrence, input projection, reservoir biases.
- **Trainable**: Readout weights only.
- **Learning Signal**: Closed-form ridge regression (convex/analytic solution only).

## Hard Constraints
- Max Parameters: 500,000
- Max Reservoir State Dimension: 2,000
- Max Readout Features: 2,000
- Max FLOPs/SOPs per inference step: 10,000,000
- Max GPU/Process Memory: 4,096 MB
- Max Wall-clock Time: 60.0 seconds per seed
- Hard Timeout: 120 seconds

