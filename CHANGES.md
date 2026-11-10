# Changes From Upstream Autoresearch

This repository transforms Karpathy's text-pretraining `autoresearch` into a general research harness for robotics, dynamical systems, and machine learning.

### 1. Removal of Language-Model Specifics
- **Upstream**: Hardcoded for language model pretraining using `tiktoken`, BPE tokenizers, FineWeb text shards, and validation bits-per-byte (`val_bpb`).
- **Fork**: Completely removed all language model dependencies and scripts (`train.py`, `prepare.py`, tokenizer tooling). Replaced with task adapters and tensor datasets suitable for robotics state estimation, trajectory prediction, and control.

### 2. Paradigm & Track Structure (§2)
- **Upstream**: Single paradigm (unconstrained gradient descent on neural network weights).
- **Fork**: Introduced strict tracks declared in `GOAL.md` (`reservoir`, `dl_supervised`, `world_model`, `probabilistic`, `spiking`, `rl_policy`). Each track locks what can be trained, what must remain frozen, and what learning signals are permitted.

### 3. Pipeline Contract (`src/pipeline.py`) (§3)
- **Upstream**: Single monolithic script `train.py`.
- **Fork**: Formal pipeline interface (`build`, `fit`, `predict`, `rollout`, `frozen_tensors`, `trainable_parameters`, `stats`). All preprocessing and normalization is encapsulated within `src/` and fit strictly on training data.

### 4. Protected Harness & Guardrails (`protected/`) (§4)
- **Upstream**: Evaluation lived inside editable or semi-editable user files.
- **Fork**: Evaluation is isolated in `protected/`:
  - `evaluate.py`: The only entry point. Enforces multi-seed evaluation, budgeting, and constraint checking.
  - `guards.py`: Verifies immutability of frozen tensors via cryptographic hashing, checks track rules (e.g. rejecting gradient encoders on reservoir tracks), and prevents validation/test data leakage.
  - `budget.py`: Automatically tracks parameters, state dimensions, FLOPs/SOPs, GPU/RAM memory, and wall-clock execution time.
  - `tasks/`: Task adapters owning data splits and metrics (including closed-loop simulation rollouts).

### 5. Version Control-Free Experimentation (§6, §6b)
- **Upstream**: Relied directly on `git commit`, `git branch`, and `git reset` inside the research loop.
- **Fork**: Completely detached from version control. The loop never invokes git. Experiment records live entirely in `lab/ledger.tsv` (append-only TSV) and `lab/exp/<id>/` (`result.json`, `log_tail.txt`, `diff.patch`, `src.tgz`, `job.log`).

### 6. Detached Runner & Outer Loop Supervisor (`lab/`) (§6, §9)
- **Upstream**: Interactive foreground execution.
- **Fork**:
  - `lab/run.sh`: Detached execution using `setsid nohup`, protected by `systemd-run` cgroup memory caps (105 GB) and hard timeouts. Implements cryptographic tamper detection and auto-restore from `lab/protected_backup/`.
  - `lab/loop.sh`: Outer loop running continuous fresh agent sessions with alternating explore/exploit modes, logging to `lab/loop.log`, and terminating cleanly via `lab/STOP`.

### 7. Private Test Split Security (§5)
- **Upstream**: No separate inaccessible test evaluation.
- **Fork**: Test split is stored outside the repository (`~/private/`) with restricted permissions (`0600`/`0700`). It is strictly locked against agent access and only accessible when `HARNESS_FINAL=1` is provided by the human operator.

