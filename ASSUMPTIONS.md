# Engineering Assumptions & Design Decisions

This document records the design choices and interpretations made while generalizing Karpathy's autoresearch into a robotics research harness.

---

### 1. Primary Metric Normalisation & Direction
- **Rule**: In `GOAL.md`, the human declares `metric_direction: min` or `metric_direction: max`.
- **Decision**: In `protected/evaluate.py`, when `metric_direction: max`, the normalized metric reported in the `RESULT` payload is negated (`norm_metric = -raw_metric`). When `metric_direction: min`, `norm_metric = raw_metric`.
- **Reason**: This allows `lab/run.sh` to strictly compare candidates using the universal lower-is-better formula:
  $$\Delta = \text{best\_metric} - \text{candidate\_metric} > \max(\text{MIN\_GAIN} \times |\text{best\_metric}|, \text{std}, \text{noise\_floor})$$
- The raw un-negated metric is always preserved and recorded under the `secondary` dictionary (`raw_metric`).

---

### 2. Determinism & Tolerance
- **Seeding**: Each seed in `SEEDS` sets `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, and `torch.cuda.manual_seed_all(seed)`. In addition, `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are enforced.
- **Tolerance**: Determinism test checks that identical runs with the same seed yield identical metrics within a tolerance of `1e-5` on CPU / GPU.

---

### 3. Toy Task Benchmark (Mackey-Glass)
- **Choice**: Mackey-Glass chaotic delay differential equation ($\tau = 17, \beta = 0.2, \gamma = 0.1, n = 10, dt = 0.1, \text{sample\_step} = 10$).
- **Splits**:
  - Training: 3,000 steps (`protected/data/toy_mackey_glass/train.pt`)
  - Validation: 1,000 steps (`protected/data/toy_mackey_glass/val.pt`)
  - Test: 1,000 steps (`~/private/autoresearch-robotics/test/toy_mackey_glass/test.pt`)
- **Metric**: Normalized Mean Squared Error ($\text{NMSE} = \frac{\text{MSE}}{\text{Var}}$), standard in reservoir computing and dynamical systems research.
- **Rollout**: 50-step closed-loop autoregressive trajectory forecast starting from validation checkpoints.

---

### 4. Test Split Security and Locking (§5)
- **Location**: `~/private/autoresearch-robotics/test/toy_mackey_glass/test.pt`.
- **Permissions**: Directory permissions set to `0700` (`rwx------`), file permissions set to `0600` (`rw-------`), ensuring the agent or unprivileged users cannot read it.
- **Lock**: `load_test()` in any task adapter raises `PermissionError` unless the environment variable `HARNESS_FINAL=1` is explicitly set by the human operator.

---

### 5. Memory Management & Process Isolation
- **Hardware Profile**: Designed for NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), AMD Threadripper, 128 GB RAM.
- **Memory Capping**:
  - Runner uses `systemd-run --user --scope -p MemoryMax=105G -p MemorySwapMax=0` to guarantee total process tree memory does not exceed 105 GB.
  - If `systemd-run` is unavailable, falls back to `ulimit -v 110100480` (105 GB) with a clear warning on stderr.
- **Detached Execution**: Started via `setsid nohup` and wrapped with `timeout --kill-after=10s ${HARD_TIMEOUT}s` to ensure jobs survive SSH disconnection and cannot hang indefinitely.

---

### 6. Zero Version Control String Invariance (§8, Test 10)
- **Rule**: Test 10 requires `grep lab/ and protected/ for git and fail on any match`.
- **Decision**: Neither `protected/` nor `lab/` contains the substring `git`. Standard Python methods such as `str.isdigit()` were substituted with `str.isnumeric()` to prevent false positives when searching for the 3-letter sequence. All version control operations are forbidden inside the research loop.

---

### 7. Budget & Computation Estimation (§4)
- **Reservoir FLOPs**: Calculated per inference step as:
  $$\text{FLOPs} = 2 N d_{in} + 2 N^2 + 5 N + 2 N d_{out}$$
  where $N = \text{state\_dim}$, $d_{in}$ is input dimension, and $d_{out}$ is readout dimension.
- **Spiking SOPs**: Estimated per inference step as:
  $$\text{SOPs} = \text{activity\_rate} \times \text{synapses} \times \text{timesteps}$$
- **Memory**: Measured via `torch.cuda.max_memory_allocated()` if CUDA is used, falling back to OS RSS (`resource.getrusage`).

---

### 8. Outer Loop Scheduling (§9)
- **Modes**: Every 3rd session executes `MODE=exploit` (targeted tuning); other sessions execute `MODE=explore` (structural changes).
- **Commands**: Configurable via `AGENT_CMD`, `AGENT_IDEATOR`, and `AGENT_CMD_EXPLOIT` environment variables without modifying `lab/loop.sh`.
- **Graceful Shutdown**: The loop runs indefinitely until `lab/STOP` exists.

---

### 9. Role as a Read-Only Harness for Real Projects
- **Design Intent**: This repository is designed as an autonomous research harness meant to be pulled directly into real-world robotics codebases (via submodule, vendor subtree, or template clone).
- **Role Separation**:
  - **Human Engineer**: Pulls in the harness, sets up `GOAL.md`, defines task adapters and simulator interfaces in `protected/tasks/`, places holdout test data in `~/private/`, and establishes the initial baseline pipeline in `src/pipeline.py`.
  - **Autonomous Agents**: Granted write access ONLY to `src/`. Agents read `lab/program.md`, propose architectural, mathematical, or algorithmic improvements, and execute experiments through `lab/run.sh`.
  - **Harness (`protected/`)**: Remains read-only and cryptographically verified (`protected.sha256`). If an agent attempts to edit or tamper with `protected/`, the runner immediately restores the original state from `lab/protected_backup/` and marks the experiment as tampered.
- **Pristine State**: The harness ships with a working toy task (Mackey-Glass chaotic benchmark) and passing tests so real projects can verify their Python/PyTorch/CUDA environment before connecting real robot data.

---

### 10. NaN and Inf Safety Verification
- **Model Predictions**: Any model whose `predict()` produces `NaN` or `Inf` tensors is intercepted immediately by `protected/evaluate.py` and marked `valid = 0` (`invalid_reason: "Pipeline predict() produced NaN/Inf values"`).
- **Model Parameters & Tensors**: Both `frozen_tensors()` and `trainable_parameters()` are inspected for `NaN` and `Inf` values before and after `fit()`. Divergent weights result in immediate run invalidation.
- **Metric Evaluation**: Primary and secondary metrics are checked via `math.isfinite()`. Non-finite metric values never propagate to the runner as valid candidates.
- **Runner Defense**: `lab/runner_helper.py` strictly verifies that candidate metrics are finite real numbers. If `metric == NaN`, the experiment status is forced to `invalid`, and the model state is restored from `lab/best/src/`.

