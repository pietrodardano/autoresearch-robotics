# Autoresearch Robotics

An autonomous, self-contained research harness for robotics, dynamical systems, and machine learning.
Designed to be **pulled into real robotics projects as a read-only evaluation and experimentation scaffold**.

This harness adapts Karpathy's `autoresearch` methodology for physical systems and neural architectures across 6 research paradigms:
- **Reservoir Computing** (Echo State Networks, liquid state machines, convex closed-form readouts)
- **Deep Learning** (Supervised end-effector tracking, self-supervised representation, policy learning)
- **World Models** (Latent dynamics, multi-step transition prediction, planning rollouts)
- **Probabilistic Learning** (Bayesian models, deep ensembles, variational inference, uncertainty calibration)
- **Neuromorphic / Spiking Networks** (Surrogate gradients or local STDP learning rules)
- **Reinforcement Learning** (Continuous control, policy/value networks)

---

## Harness Design & Project Integration

This repository is designed as a **read-only autonomous harness** that downstream robotics projects pull in. 

```
Real Project Root /
├── GOAL.md                  Declared track, metric, and hardware constraints (human-locked)
├── protected/               The evaluation harness (read-only for agents; SHA-256 verified)
│   ├── evaluate.py          Only entry point: runs multi-seed evaluation, emits RESULT line
│   ├── guards.py            Enforces frozen components, track compliance, and prevents data leakage
│   ├── budget.py            Accounts for parameters, state dim, FLOPs/SOPs, memory, and wall time
│   ├── tasks/               Task adapters (simulators, robot dataset splits, closed-loop rollout)
│   └── data/                Fixed train and validation splits (test split held separately)
├── src/                     Active codebase that autonomous coding agents may modify
│   ├── pipeline.py          Mandatory interface contract (fit, predict, rollout, stats)
│   └── ...                  Encoders, models, losses, and learning rules
├── lab/                     Supervisory scripts and persistent memory (no version control)
│   ├── run.sh               Detached runner (start, wait, status, brief, show)
│   ├── loop.sh              Outer loop supervisor managing fresh agent sessions
│   ├── program.md           Agent instructions (short prompt for coding LLMs)
│   ├── ledger.tsv           Append-only experiment record
│   ├── best/                Current best src/ snapshot and locked metric
│   ├── exp/<id>/            Per-experiment logs, diff patches, src snapshot, and result JSON
│   ├── protected.sha256     Integrity manifest of protected/
│   └── protected_backup/    Trusted backup for automatic tamper recovery
└── tests/                   Validation suite ensuring harness invariants hold
```

### How Real Projects Pull in this Harness

1. **Pull as Subproject or Template**:
   Clone or vendor this harness into your robotics repository.
2. **Define the Project Objective (`GOAL.md`)**:
   Select your track (`reservoir`, `dl_supervised`, `world_model`, `probabilistic`, `spiking`, or `rl_policy`) using `GOAL.md.template` as a reference. Lock your evaluation metric (e.g., trajectory tracking RMSE, slip detection calibration, contact force error).
3. **Connect Your Robot Task (`protected/tasks/`)**:
   Implement a task adapter in `protected/tasks/<your_robot_task>.py` that interfaces with your robot dataset or physics simulator (Isaac Gym, MuJoCo, Brax, etc.).
4. **Hold Out the Test Split**:
   Keep the final test split stored outside the repository (e.g. `~/private/<project>/test/`) with restricted permissions (`chmod 600`). It remains invisible to agents until human verification with `HARNESS_FINAL=1`.
5. **Establish the Baseline in `src/`**:
   Provide your initial baseline model in `src/pipeline.py` implementing the pipeline contract.
6. **Deploy the Agent**:
   Point your coding agent (Claude Code, Gemini, OpenAI, etc.) to `lab/program.md` or launch the unattended outer loop via `./lab/loop.sh`. Agents will autonomously iterate, test hypotheses, and log results without ever interacting with version control.

---

## Quickstart

### 1. Requirements
Ensure Python 3.10+ and PyTorch (with CUDA 12.8+ if running on GPU) are installed:
```bash
pip install -r requirements.txt
```

### 2. Run a Single Experiment
Start an experiment in the background (detached under `setsid nohup`):
```bash
./lab/run.sh start baseline "Initial reservoir baseline"
```

Check runner status:
```bash
./lab/run.sh status
```

Wait for the run to complete and apply keep/discard decisions:
```bash
./lab/run.sh wait
```

View the experiment summary brief (bounded strictly to 1,500 characters):
```bash
./lab/run.sh brief
```

Inspect a specific experiment and its code diff:
```bash
./lab/run.sh show 000
```

---

## Autonomous Outer Loop

### How to Run
The outer loop runs continuous, stateless agent sessions. Every third session runs in `exploit` mode (targeted parameter/code tuning); other sessions run in `explore` mode (architectural/structural hypothesis testing).

1. Set your agent invocation command:
```bash
export AGENT_CMD="my-agent-cli --program lab/program.md"
# Optionally differentiate ideator and exploit models:
export AGENT_IDEATOR="claude-3-7-sonnet --program lab/program.md"
export AGENT_CMD_EXPLOIT="claude-3-5-haiku --program lab/program.md"
```

2. Launch in a persistent terminal session (`tmux` or `nohup`):
```bash
tmux new -s research
./lab/loop.sh
```

### How to Stop
To terminate the research loop gracefully after the current session completes:
```bash
touch lab/STOP
```
The loop will detect `lab/STOP`, log the exit, and cleanly terminate.

---

## Adding a New Task

To introduce a new benchmark task (e.g. `robot_arm_reach`):

1. **Create Adapter**:
   Add `protected/tasks/robot_arm_reach.py` implementing:
   - `load_train() -> TaskData`
   - `load_val() -> TaskData`
   - `load_test() -> TaskData` (enforcing `HARNESS_FINAL=1` check)
   - `validation_inputs(val_data)` and `validation_targets(val_data)`
   - `metric_fns: Dict[str, Callable]`
   - `rollout_eval(pipeline)` (if task includes closed-loop simulation)
   - `leakage_check(train_data, val_data, pipeline)`

2. **Add Data**:
   - Store train/val data under `protected/data/robot_arm_reach/`.
   - Store private test split under `~/private/<project>/test/robot_arm_reach/test.pt` (`chmod 600`).

3. **Update Hash Manifest & Backup**:
   Whenever `protected/` is modified by the human operator, refresh the manifest and backup:
   ```bash
   find protected/ -type f ! -path '*/__pycache__*' | sort | xargs sha256sum > lab/protected.sha256
   rsync -a --delete --exclude='__pycache__' protected/ lab/protected_backup/
   ```

4. **Update `GOAL.md`**:
   Set `task: robot_arm_reach` and define the appropriate metric and caps.

---

## Adding a New Track

To define a new paradigm (e.g. `quantum_reservoir`):

1. Edit `protected/guards.py` in `check_track_compliance(pipeline, track, learning_rule)` to add the verification rules for the new track.
2. Define which parameters must remain frozen and which learning rules are permitted.
3. Update `lab/protected.sha256` and `lab/protected_backup/` as shown above.
4. Add an example configuration to `GOAL.md.template`.

---

## Running Final Test Evaluation by Hand

The test split is locked against the agent process. Only the human operator can evaluate the best model on the holdout test split:

```bash
HARNESS_FINAL=1 python -c "
from protected.tasks.toy_mackey_glass import load_test, validation_inputs, validation_targets, metric_fns
import src.pipeline as p

pipe = p.build({'seed': 42})
test_data = load_test()
inputs = validation_inputs(test_data)
targets = validation_targets(test_data)
preds = pipe.predict(inputs)
nmse = metric_fns['nmse'](targets, preds)
print(f'Final Test Split NMSE: {nmse:.6f}')
"
```

---

## Running the Harness Test Suite

Run all 12 invariant tests:
```bash
pytest -v tests/
```
