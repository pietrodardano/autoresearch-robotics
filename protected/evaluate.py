"""
Protected evaluation entry point for the autoresearch robotics harness.
Loads task, builds pipeline across seeds, applies guards, evaluates validation metrics,
measures budget, checks constraints, and prints the locked RESULT line.
"""

from __future__ import annotations
import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import yaml

from protected import budget, guards


def load_goal_config(goal_path: str = "GOAL.md") -> Dict[str, Any]:
    """Parses project configuration and constraints from GOAL.md frontmatter."""
    if not os.path.exists(goal_path):
        raise FileNotFoundError(f"GOAL.md not found at {goal_path}")
    
    with open(goal_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse YAML frontmatter between --- and ---
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            parsed = yaml.safe_load(parts[1])
            if isinstance(parsed, dict):
                return parsed
            
    # Fallback: parse entire file as YAML
    parsed = yaml.safe_load(content)
    if isinstance(parsed, dict):
        return parsed
    
    raise ValueError("Failed to parse YAML frontmatter from GOAL.md")


def run_evaluation(goal_path: str = "GOAL.md") -> Dict[str, Any]:
    """Main evaluation execution logic."""
    total_start_time = time.perf_counter()
    cfg = load_goal_config(goal_path)

    track = cfg.get("track", "reservoir")
    task_name = cfg.get("task", "toy_mackey_glass")
    learning_rule = cfg.get("learning_rule", "closed_form")
    seeds: List[int] = cfg.get("seeds", [42, 43, 44])
    metric_name = cfg.get("metric", "nmse")
    metric_direction = cfg.get("metric_direction", "min").lower()
    constraints = cfg.get("constraints", {})

    # Import task adapter
    task_module_name = f"protected.tasks.{task_name}"
    try:
        task = importlib.import_module(task_module_name)
    except ModuleNotFoundError as e:
        return {
            "metric": float("nan"),
            "metric_name": metric_name,
            "std": 0.0,
            "seeds": 0,
            "valid": 0,
            "invalid_reason": f"Could not load task adapter '{task_module_name}': {e}",
            "constraints": {"status": "error", "violated": []},
            "secondary": {},
            "time_s": round(time.perf_counter() - total_start_time, 4),
        }

    # Import pipeline module under test
    try:
        import src.pipeline as pipeline_module
    except Exception as e:
        return {
            "metric": float("nan"),
            "metric_name": metric_name,
            "std": 0.0,
            "seeds": 0,
            "valid": 0,
            "invalid_reason": f"Failed to import src.pipeline: {e}",
            "constraints": {"status": "error", "violated": []},
            "secondary": {},
            "time_s": round(time.perf_counter() - total_start_time, 4),
        }

    # Load datasets
    try:
        train_data = task.load_train()
        val_data = task.load_val()
    except Exception as e:
        return {
            "metric": float("nan"),
            "metric_name": metric_name,
            "std": 0.0,
            "seeds": 0,
            "valid": 0,
            "invalid_reason": f"Failed loading dataset splits: {e}",
            "constraints": {"status": "error", "violated": []},
            "secondary": {},
            "time_s": round(time.perf_counter() - total_start_time, 4),
        }

    primary_metrics: List[float] = []
    secondary_metrics_acc: Dict[str, List[float]] = {}
    budget_records: List[Dict[str, Any]] = []

    valid = 1
    invalid_reason = ""

    for seed in seeds:
        # Seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.reset_peak_memory_stats()

        seed_start = time.perf_counter()

        try:
            # Build pipeline
            pipeline = pipeline_module.build({
                "seed": seed,
                "track": track,
                "task": task_name,
                "learning_rule": learning_rule,
                "constraints": constraints,
            })

            # Guard 1: verify frozen tensors require_grad=False and get initial hash
            initial_frozen_hash = guards.check_frozen_before(pipeline)

            # Fit on training data ONLY within the guarded context
            with guards.training_context():
                pipeline.fit(train_data)

            # Guard 2: verify frozen tensors were not modified during fit
            guards.check_frozen_after(pipeline, initial_frozen_hash)

            # Guard 3: check track compliance
            guards.check_track_compliance(pipeline, track, learning_rule)

            # Guard 4: verify no validation leakage occurred
            task.leakage_check(train_data, val_data, pipeline)

            # Evaluate on validation inputs ONLY
            val_in = task.validation_inputs(val_data)
            preds = pipeline.predict(val_in)

            # Check predictions for NaN / Inf
            if isinstance(preds, torch.Tensor):
                if torch.isnan(preds).any():
                    raise guards.GuardViolation("Pipeline predict() produced NaN values")
                if torch.isinf(preds).any():
                    raise guards.GuardViolation("Pipeline predict() produced Inf values")

            val_targets = task.validation_targets(val_data)

            # Compute metric
            if metric_name not in task.metric_fns:
                raise ValueError(f"Task adapter does not define metric '{metric_name}'")
            score = task.metric_fns[metric_name](val_targets, preds)
            
            # Check metric score for NaN / Inf
            if math.isnan(score) or math.isinf(score):
                raise guards.GuardViolation(f"Metric '{metric_name}' computed non-finite score: {score}")

            primary_metrics.append(float(score))

            # Additional secondary task metrics
            for m_name, m_fn in task.metric_fns.items():
                if m_name != metric_name:
                    m_val = float(m_fn(val_targets, preds))
                    if not math.isnan(m_val) and not math.isinf(m_val):
                        secondary_metrics_acc.setdefault(m_name, []).append(m_val)

            # Measure budget
            seed_wall_s = time.perf_counter() - seed_start
            b_stats = budget.measure_budget(pipeline, val_in, track, seed_wall_s)
            budget_records.append(b_stats)

        except (guards.GuardViolation, guards.LeakageViolation) as e:
            valid = 0
            invalid_reason = f"Guard violation: {str(e)}"
            break
        except Exception as e:
            valid = 0
            invalid_reason = f"Execution error during evaluation: {str(e)}"
            break

    # Aggregate budget stats
    if budget_records:
        avg_budget = {
            "params": int(np.mean([b["params"] for b in budget_records])),
            "state_dim": int(np.mean([b["state_dim"] for b in budget_records])),
            "readout_features": int(np.mean([b["readout_features"] for b in budget_records])),
            "flops_or_sops": float(np.mean([b["flops_or_sops"] for b in budget_records])),
            "peak_mem_mb": float(np.max([b["peak_mem_mb"] for b in budget_records])),
            "wall_time_s": float(np.mean([b["wall_time_s"] for b in budget_records])),
        }
    else:
        avg_budget = {
            "params": 0, "state_dim": 0, "readout_features": 0,
            "flops_or_sops": 0.0, "peak_mem_mb": 0.0, "wall_time_s": 0.0
        }

    # Hard constraints check
    constraints_ok, violated_caps = guards.check_constraints(avg_budget, constraints)
    if not constraints_ok:
        valid = 0
        if not invalid_reason:
            invalid_reason = f"Constraint violated: {', '.join(violated_caps)}"

    # Aggregate metric
    if primary_metrics and valid == 1:
        raw_mean = float(np.mean(primary_metrics))
        metric_std = float(np.std(primary_metrics)) if len(primary_metrics) > 1 else 0.0
        if math.isnan(raw_mean) or math.isinf(raw_mean):
            valid = 0
            invalid_reason = "Non-finite aggregate metric (NaN/Inf)"
            norm_metric = float("nan")
        else:
            # Lower-is-better normalisation for runner
            norm_metric = raw_mean if metric_direction == "min" else -raw_mean
    else:
        norm_metric = float("nan")
        raw_mean = float("nan")
        metric_std = 0.0

    # Build secondary metrics map
    secondary_map = {
        "params": avg_budget["params"],
        "state_dim": avg_budget["state_dim"],
        "readout_features": avg_budget["readout_features"],
        "flops_sops": avg_budget["flops_or_sops"],
        "mem_mb": avg_budget["peak_mem_mb"],
        "wall_s": avg_budget["wall_time_s"],
        "raw_metric": raw_mean,
    }
    for m_name, vals in secondary_metrics_acc.items():
        secondary_map[m_name] = float(np.mean(vals))

    total_time_s = float(round(time.perf_counter() - total_start_time, 4))

    result = {
        "metric": norm_metric,
        "metric_name": metric_name,
        "std": round(metric_std, 6),
        "seeds": len(primary_metrics),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "constraints": {
            "status": "ok" if constraints_ok else "violated",
            "violated": violated_caps,
        },
        "secondary": secondary_map,
        "time_s": total_time_s,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate harness entry point")
    parser.add_argument("--goal", default="GOAL.md", help="Path to GOAL.md configuration")
    args = parser.parse_args()

    result = run_evaluation(args.goal)
    print(f"RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
