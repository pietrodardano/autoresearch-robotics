"""
Toy Task Adapter: Mackey-Glass Chaotic Time-Series Prediction.
Benchmark for reservoir computing, dynamical systems, and robotics predictive state modeling.
"""

from __future__ import annotations
import os
import stat
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
import torch

from protected import guards


@dataclass
class TaskData:
    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "toy_mackey_glass")
PRIVATE_TEST_DIR = os.path.expanduser("~/private/autoresearch-robotics/test/toy_mackey_glass")


def load_train() -> TaskData:
    """Loads fixed training split."""
    path = os.path.join(DATA_DIR, "train.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")
    data = torch.load(path, weights_only=True)
    return TaskData(inputs=data["inputs"], targets=data["targets"], metadata={"name": "toy_mackey_glass_train"})


def load_val() -> TaskData:
    """Loads fixed validation split. Locked during training to prevent leakage."""
    if guards.is_in_fit_context():
        guards.mark_leakage("load_val() invoked during pipeline.fit()")
        raise guards.LeakageViolation("Data leakage: Validation data access is prohibited during pipeline.fit()")
    path = os.path.join(DATA_DIR, "val.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Validation data not found at {path}")
    data = torch.load(path, weights_only=True)
    return TaskData(inputs=data["inputs"], targets=data["targets"], metadata={"name": "toy_mackey_glass_val"})


def load_test() -> TaskData:
    """
    Loads test split from private storage outside repository.
    Raises PermissionError unless HARNESS_FINAL=1 environment variable is set.
    """
    if os.environ.get("HARNESS_FINAL") != "1":
        raise PermissionError(
            "Access to the test split is locked for agent processes. "
            "Only human operators may run final evaluation with HARNESS_FINAL=1."
        )
    
    test_file = os.path.join(PRIVATE_TEST_DIR, "test.pt")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test split not found at {test_file}")
    
    # Verify strict file permissions (no group or others read permission)
    st = os.stat(test_file)
    mode = stat.S_IMODE(st.st_mode)
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        raise PermissionError(f"Insecure test split permissions: {oct(mode)}. Group/other read not allowed.")
    
    data = torch.load(test_file, weights_only=True)
    return TaskData(inputs=data["inputs"], targets=data["targets"], metadata={"name": "toy_mackey_glass_test"})


def validation_inputs(val_data: TaskData) -> torch.Tensor:
    """Returns validation inputs for predict()."""
    return val_data.inputs


def validation_targets(val_data: TaskData) -> torch.Tensor:
    """Returns validation targets for scoring. Locked during fit()."""
    if guards.is_in_fit_context():
        guards.mark_leakage("validation_targets accessed during pipeline.fit()")
        raise guards.LeakageViolation("Data leakage: Validation targets cannot be read during pipeline.fit()")
    return val_data.targets


def calc_nmse(targets: torch.Tensor, preds: torch.Tensor) -> float:
    """Normalized Mean Squared Error: MSE(targets, preds) / Var(targets)."""
    t = targets.detach().cpu().float()
    p = preds.detach().cpu().float()
    mse = torch.mean((t - p) ** 2)
    var = torch.var(t, unbiased=False)
    if var == 0:
        return float(mse.item())
    return float((mse / var).item())


def calc_mse(targets: torch.Tensor, preds: torch.Tensor) -> float:
    t = targets.detach().cpu().float()
    p = preds.detach().cpu().float()
    return float(torch.mean((t - p) ** 2).item())


def calc_mae(targets: torch.Tensor, preds: torch.Tensor) -> float:
    t = targets.detach().cpu().float()
    p = preds.detach().cpu().float()
    return float(torch.mean(torch.abs(t - p)).item())


metric_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "nmse": calc_nmse,
    "mse": calc_mse,
    "mae": calc_mae,
}


def rollout_eval(pipeline: Any, horizon: int = 50, num_seeds: int = 3) -> Dict[str, float]:
    """
    Evaluates multi-step closed-loop autoregressive rollout on validation data.
    """
    val_data = load_val()
    val_in = val_data.inputs
    val_tgt = val_data.targets
    
    # Starting points evenly spaced across validation set
    start_indices = [0, len(val_in) // 3, 2 * len(val_in) // 3]
    errors = []
    
    for start_idx in start_indices:
        if start_idx + horizon > len(val_tgt):
            continue
        init_state = val_in[start_idx : start_idx + 1]
        ground_truth = val_tgt[start_idx : start_idx + horizon]
        
        # Rollout through pipeline interface
        pred_rollout = pipeline.rollout(initial=init_state, actions=None, horizon=horizon)
        if isinstance(pred_rollout, torch.Tensor):
            pred_rollout = pred_rollout.reshape(-1, 1)[:horizon]
            err = calc_nmse(ground_truth[:len(pred_rollout)], pred_rollout)
            errors.append(err)
            
    avg_rollout_nmse = float(sum(errors) / len(errors)) if errors else 0.0
    return {"rollout_nmse": avg_rollout_nmse}


def leakage_check(train_data: TaskData, val_data: TaskData, pipeline: Any = None) -> None:
    """Verifies that no validation sample or state leaked into training."""
    # 1. Check if leakage flag was raised during fit
    guards.check_leakage_status()
    # 2. Check for sample collision / hash overlap
    guards.check_data_leakage(train_data.inputs, val_data.inputs)
