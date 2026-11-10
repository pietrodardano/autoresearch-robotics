"""
Budget accounting module for the autoresearch robotics harness.
Measures parameter counts, state dimensions, readout features,
estimated FLOPs or SOPs per inference, peak memory, and wall-clock execution time.
"""

from __future__ import annotations
import os
import resource
import time
from typing import Any, Dict
import torch


def count_parameters(pipeline: Any) -> int:
    """Computes total parameter count in the pipeline (frozen and trainable)."""
    total = 0
    # Check if pipeline has stats
    if hasattr(pipeline, "stats"):
        st = pipeline.stats()
        if "params" in st and isinstance(st["params"], int):
            total = max(total, st["params"])
            
    # Count from torch.nn.Module if applicable
    if isinstance(pipeline, torch.nn.Module):
        total = sum(p.numel() for p in pipeline.parameters()) + sum(b.numel() for b in pipeline.buffers())
        return total
    
    # Or sum frozen_tensors and trainable_parameters
    tensor_count = 0
    if hasattr(pipeline, "frozen_tensors"):
        for t in pipeline.frozen_tensors():
            if isinstance(t, torch.Tensor):
                tensor_count += t.numel()
    if hasattr(pipeline, "trainable_parameters"):
        for p in pipeline.trainable_parameters():
            if isinstance(p, torch.Tensor):
                tensor_count += p.numel()
                
    return max(total, tensor_count)


def estimate_flops_or_sops(pipeline: Any, sample_inputs: torch.Tensor, track: str) -> float:
    """
    Estimates FLOPs (floating point operations) or SOPs (synaptic operations) per inference step.
    """
    stats = pipeline.stats() if hasattr(pipeline, "stats") else {}
    state_dim = stats.get("state_dim", 100)
    readout_features = stats.get("readout_features", state_dim)
    
    in_dim = sample_inputs.shape[-1] if len(sample_inputs.shape) >= 2 else 1
    
    if track == "spiking":
        # Estimate Synaptic Operations (SOPs): spikes * synaptic connections
        # Typically estimated as activity_rate * connections * timesteps
        activity_rate = stats.get("mean_spike_rate", 0.05)
        synapses = state_dim * in_dim + state_dim * state_dim + readout_features
        timesteps = stats.get("timesteps", 1)
        sops = activity_rate * synapses * timesteps
        return float(sops)
    
    elif track == "reservoir":
        # Standard Echo State / Reservoir Network forward step FLOPs:
        # 1. W_in @ u: 2 * N * d_in
        # 2. W_res @ x: 2 * N^2 (or sparse non-zeros)
        # 3. State update: leak, tanh, bias: ~5 * N
        # 4. Readout: 2 * readout_features * d_out
        out_dim = stats.get("out_dim", in_dim)
        flops_step = 2.0 * state_dim * in_dim + 2.0 * (state_dim ** 2) + 5.0 * state_dim + 2.0 * readout_features * out_dim
        return float(flops_step)
    
    else:
        # Default estimation based on parameter operations
        total_p = count_parameters(pipeline)
        # 2 FLOPs per parameter per inference step as lower bound
        return float(2.0 * total_p)


def get_peak_memory_mb() -> float:
    """Returns peak GPU memory if available; otherwise falls back to system RSS memory."""
    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    # Fallback to OS getrusage (RSS in KB on Linux)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss / 1024.0)


def measure_budget(pipeline: Any, sample_inputs: torch.Tensor, track: str, wall_time_s: float) -> Dict[str, Any]:
    """Collects all budget measurements for a pipeline run."""
    stats = pipeline.stats() if hasattr(pipeline, "stats") else {}
    params = count_parameters(pipeline)
    state_dim = stats.get("state_dim", 0)
    readout_features = stats.get("readout_features", 0)
    flops_or_sops = estimate_flops_or_sops(pipeline, sample_inputs, track)
    peak_mem_mb = get_peak_memory_mb()
    
    return {
        "params": int(params),
        "state_dim": int(state_dim),
        "readout_features": int(readout_features),
        "flops_or_sops": float(flops_or_sops),
        "peak_mem_mb": float(round(peak_mem_mb, 2)),
        "wall_time_s": float(round(wall_time_s, 4)),
    }
