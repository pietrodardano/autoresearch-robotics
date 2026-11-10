"""
Guards module for the autoresearch robotics harness.
Enforces frozen-component immutability, track compliance, budget constraints,
and prevents validation/test data leakage.
"""

from __future__ import annotations
import contextlib
import hashlib
from typing import Any, Dict, List, Tuple
import torch


class GuardViolation(Exception):
    """Raised when an architectural, track, or frozen constraint is violated."""
    pass


class LeakageViolation(Exception):
    """Raised when validation or test data leaks into pipeline fitting."""
    pass


_IN_FIT_CONTEXT: bool = False
_LEAKAGE_DETECTED: bool = False
_LEAKAGE_REASON: str = ""


@contextlib.contextmanager
def training_context():
    """Context manager active during pipeline.fit(). Traps any calls to validation loaders."""
    global _IN_FIT_CONTEXT, _LEAKAGE_DETECTED, _LEAKAGE_REASON
    prev_state = _IN_FIT_CONTEXT
    _IN_FIT_CONTEXT = True
    _LEAKAGE_DETECTED = False
    _LEAKAGE_REASON = ""
    try:
        yield
    finally:
        _IN_FIT_CONTEXT = prev_state


def is_in_fit_context() -> bool:
    """Returns True if pipeline.fit() is currently executing."""
    return _IN_FIT_CONTEXT


def mark_leakage(reason: str):
    """Mark that data leakage occurred during training."""
    global _LEAKAGE_DETECTED, _LEAKAGE_REASON
    _LEAKAGE_DETECTED = True
    _LEAKAGE_REASON = reason


def check_leakage_status():
    """Raises LeakageViolation if leakage was detected during training."""
    global _LEAKAGE_DETECTED, _LEAKAGE_REASON
    if _LEAKAGE_DETECTED:
        raise LeakageViolation(f"Data leakage detected during fit: {_LEAKAGE_REASON}")


def check_finite_tensors(tensors: List[torch.Tensor], context: str = "tensors") -> None:
    """Verifies that none of the provided tensors contain NaN or Inf values."""
    for idx, t in enumerate(tensors):
        if isinstance(t, torch.Tensor):
            if torch.isnan(t).any():
                raise GuardViolation(f"NaN detected in {context} at index {idx}")
            if torch.isinf(t).any():
                raise GuardViolation(f"Inf detected in {context} at index {idx}")


def hash_tensors(tensors: List[torch.Tensor]) -> str:
    """Computes a deterministic SHA-256 hash across a list of tensors."""
    hasher = hashlib.sha256()
    for idx, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise GuardViolation(f"Item {idx} in frozen_tensors is not a torch.Tensor (type: {type(t)})")
        hasher.update(str(t.dtype).encode("utf-8"))
        hasher.update(str(tuple(t.shape)).encode("utf-8"))
        # Ensure tensor is contiguous on CPU before reading bytes
        t_cpu = t.detach().contiguous().cpu()
        hasher.update(t_cpu.numpy().tobytes())
    return hasher.hexdigest()


def check_frozen_before(pipeline: Any) -> str:
    """
    Checks that frozen tensors have requires_grad=False and returns their hash.
    Fails if any frozen tensor requires gradients.
    """
    if not hasattr(pipeline, "frozen_tensors"):
        raise GuardViolation("Pipeline must implement frozen_tensors() method.")
    
    frozen = pipeline.frozen_tensors()
    if not isinstance(frozen, list):
        raise GuardViolation("frozen_tensors() must return a list of torch.Tensors.")
    
    for idx, t in enumerate(frozen):
        if not isinstance(t, torch.Tensor):
            raise GuardViolation(f"Item {idx} in frozen_tensors is not a torch.Tensor")
        if t.requires_grad:
            raise GuardViolation(f"Frozen tensor at index {idx} has requires_grad=True")
    
    check_finite_tensors(frozen, "frozen_tensors")
    return hash_tensors(frozen)


def check_frozen_after(pipeline: Any, initial_hash: str) -> None:
    """Re-hashes frozen tensors and verifies they did not change during fit()."""
    frozen = pipeline.frozen_tensors()
    check_finite_tensors(frozen, "frozen_tensors")
    current_hash = hash_tensors(frozen)
    if current_hash != initial_hash:
        raise GuardViolation(
            f"Frozen tensors were modified during fit()! "
            f"(Initial hash: {initial_hash[:12]}..., Current hash: {current_hash[:12]}...)"
        )


def check_track_compliance(pipeline: Any, track: str, learning_rule: str) -> None:
    """
    Verifies that the pipeline strictly adheres to the track rules declared in GOAL.md.
    """
    if not hasattr(pipeline, "trainable_parameters"):
        raise GuardViolation("Pipeline must implement trainable_parameters() method.")
    
    trainable_params = pipeline.trainable_parameters()
    if not isinstance(trainable_params, list):
        raise GuardViolation("trainable_parameters() must return a list of torch.nn.Parameter.")
    
    check_finite_tensors(trainable_params, "trainable_parameters")
    
    for p in trainable_params:
        if not isinstance(p, torch.nn.Parameter) and not isinstance(p, torch.Tensor):
            raise GuardViolation(f"Trainable item {type(p)} is not a Parameter or Tensor.")
    
    stats = pipeline.stats() if hasattr(pipeline, "stats") else {}
    
    if track == "reservoir":
        # Rule: reservoir weights must be frozen, only readout can be trainable
        frozen = pipeline.frozen_tensors()
        if len(frozen) == 0:
            raise GuardViolation("Reservoir track requires frozen reservoir tensors in frozen_tensors().")
        
        # Check that no non-readout parameter is trainable
        # For instance, check if any encoder/backbone has requires_grad=True
        # In a reservoir pipeline, the only trainable parameters are the readout
        for idx, p in enumerate(trainable_params):
            # Check attribute tags if present
            is_readout = getattr(p, "_is_readout", None)
            param_name = getattr(p, "_name", "")
            if is_readout is False:
                raise GuardViolation(
                    f"Trainable parameter {param_name or idx} is not a readout parameter on reservoir track."
                )
            # If pipeline has an encoder with trainable parameters
            if "encoder" in param_name.lower():
                raise GuardViolation(
                    f"Encoder parameter {param_name} is marked trainable. "
                    f"Gradient-trained encoders are not allowed on the reservoir track."
                )
        
        # Check learning rule
        if learning_rule not in ("closed_form", "ridge", "convex"):
            raise GuardViolation(
                f"Invalid learning rule '{learning_rule}' for reservoir track. "
                f"Must be closed_form or convex."
            )

    elif track == "spiking":
        if learning_rule not in ("surrogate_gradient", "local_rule"):
            raise GuardViolation(
                f"Spiking track must declare either 'surrogate_gradient' or 'local_rule', got '{learning_rule}'."
            )

    elif track == "dl_supervised":
        if learning_rule not in ("gradient", "backprop", "adam", "adamw", "sgd"):
            # Gradient-based only
            pass

    elif track in ("world_model", "probabilistic", "rl_policy"):
        pass

    else:
        raise GuardViolation(f"Unrecognized track: '{track}'")


def check_data_leakage(train_inputs: torch.Tensor, val_inputs: torch.Tensor) -> None:
    """Verifies that no exact sample in the validation set overlaps with the training set."""
    # Flatten sample representations to check overlap
    t_flat = train_inputs.detach().cpu().reshape(train_inputs.shape[0], -1)
    v_flat = val_inputs.detach().cpu().reshape(val_inputs.shape[0], -1)
    
    # Fast hash-set comparison
    train_hashes = {hashlib.sha256(row.numpy().tobytes()).digest() for row in t_flat}
    for idx, v_row in enumerate(v_flat):
        v_h = hashlib.sha256(v_row.numpy().tobytes()).digest()
        if v_h in train_hashes:
            raise LeakageViolation(
                f"Validation sample {idx} is identical to a training sample! Data leakage detected."
            )


def check_constraints(stats: Dict[str, Any], constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Checks all measured budget values against the hard caps in GOAL.md.
    Returns (True, []) if all pass, or (False, [violated_cap_names]).
    """
    violated = []
    
    cap_map = {
        "max_params": ("params", lambda m, c: m <= c),
        "max_state_dim": ("state_dim", lambda m, c: m <= c),
        "max_readout_features": ("readout_features", lambda m, c: m <= c),
        "max_flops_or_sops": ("flops_or_sops", lambda m, c: m <= c),
        "max_memory_mb": ("peak_mem_mb", lambda m, c: m <= c),
        "max_wall_time_s": ("wall_time_s", lambda m, c: m <= c),
    }
    
    for cap_name, (stat_key, check_fn) in cap_map.items():
        if cap_name in constraints:
            cap_val = constraints[cap_name]
            stat_val = stats.get(stat_key)
            if stat_val is not None:
                if not check_fn(stat_val, cap_val):
                    violated.append(f"{cap_name} (measured {stat_val} > cap {cap_val})")
            else:
                # If constraint specified but stat missing
                pass
    
    return len(violated) == 0, violated
