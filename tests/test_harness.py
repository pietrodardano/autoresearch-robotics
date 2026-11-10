"""
Harness Test Suite.
Validates the 12 required invariants specified in section 8:
1. Smoke test
2. Frozen guard catches violations (including gradient encoder on reservoir track)
3. Determinism
4. Leakage detection
5. Constraint enforcement
6. Tamper detection
7. Runner keep/discard decisions
8. Detached survival across parent exit
9. Test-split lock
10. No version control interactions
11. Append-only ledger integrity
12. Brief output size limit
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
import time
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from protected import budget, evaluate, guards
from protected.tasks import toy_mackey_glass as toy_task
from lab import runner_helper


def test_01_smoke_test():
    """1. Smoke test: Run full pipeline on toy task in under two minutes."""
    start_t = time.perf_counter()
    result = evaluate.run_evaluation("GOAL.md")
    elapsed = time.perf_counter() - start_t

    assert elapsed < 120.0, f"Evaluation exceeded 2 minutes: {elapsed:.2f}s"
    assert result["valid"] == 1, f"Run was invalid: {result.get('invalid_reason')}"
    assert result["metric_name"] == "nmse"
    assert result["metric"] < 0.05, f"Expected NMSE < 0.05, got {result['metric']}"
    assert result["seeds"] >= 1


def test_02_frozen_guard_catches_violations(monkeypatch):
    """2. Frozen guard catches violations: trainable reservoir weights and gradient encoders."""
    # Part A: Reservoir weight marked trainable
    class LeakyGradPipeline:
        def __init__(self, cfg):
            self.W_in = torch.randn(10, 1, requires_grad=False)
            # Violation: frozen reservoir weight requires grad!
            self.W_res = torch.nn.Parameter(torch.randn(10, 10), requires_grad=True)
            self.b = torch.zeros(10, requires_grad=False)
            self.W_out = torch.nn.Parameter(torch.zeros(1, 11), requires_grad=True)
            self.W_out._is_readout = True
        def frozen_tensors(self):
            return [self.W_in, self.W_res, self.b]
        def trainable_parameters(self):
            return [self.W_out]
        def stats(self):
            return {"params": 100, "state_dim": 10, "readout_features": 11}
        def fit(self, train): pass
        def predict(self, inputs): return torch.zeros_like(inputs)
        def rollout(self, init, act, h): return torch.zeros(h, 1)

    monkeypatch.setattr("src.pipeline.build", lambda cfg: LeakyGradPipeline(cfg))
    res_grad = evaluate.run_evaluation("GOAL.md")
    assert res_grad["valid"] == 0
    assert "Frozen tensor" in res_grad["invalid_reason"] or "requires_grad=True" in res_grad["invalid_reason"]

    # Part B: Gradient-trained encoder on reservoir track
    class EncoderTrainedPipeline:
        def __init__(self, cfg):
            self.W_in = torch.randn(10, 1, requires_grad=False)
            self.W_res = torch.randn(10, 10, requires_grad=False)
            self.b = torch.zeros(10, requires_grad=False)
            # Encoder weight marked trainable
            enc_p = torch.nn.Parameter(torch.randn(5, 1), requires_grad=True)
            enc_p._name = "encoder.weight"
            enc_p._is_readout = False
            self.enc_param = enc_p
            self.W_out = torch.nn.Parameter(torch.zeros(1, 11), requires_grad=True)
            self.W_out._is_readout = True
        def frozen_tensors(self):
            return [self.W_in, self.W_res, self.b]
        def trainable_parameters(self):
            return [self.enc_param, self.W_out]
        def stats(self):
            return {"params": 110, "state_dim": 10, "readout_features": 11}
        def fit(self, train): pass
        def predict(self, inputs): return torch.zeros_like(inputs)
        def rollout(self, init, act, h): return torch.zeros(h, 1)

    monkeypatch.setattr("src.pipeline.build", lambda cfg: EncoderTrainedPipeline(cfg))
    res_enc = evaluate.run_evaluation("GOAL.md")
    assert res_enc["valid"] == 0
    assert "encoder" in res_enc["invalid_reason"].lower() or "readout" in res_enc["invalid_reason"].lower()


def test_03_determinism():
    """3. Determinism: Same seed produces the same metric within declared tolerance."""
    goal_single_seed = tempfile.NamedTemporaryFile("w", delete=False, suffix=".md")
    try:
        goal_single_seed.write("""---
track: reservoir
task: toy_mackey_glass
learning_rule: closed_form
seeds: [123]
metric: nmse
metric_direction: min
min_gain: 0.01
noise_floor: 0.0001
hard_timeout: 120
frozen: ["reservoir"]
constraints:
  max_params: 500000
---
""")
        goal_single_seed.flush()
        goal_single_seed.close()

        res1 = evaluate.run_evaluation(goal_single_seed.name)
        res2 = evaluate.run_evaluation(goal_single_seed.name)

        assert res1["valid"] == 1 and res2["valid"] == 1
        tolerance = 1e-5
        diff = abs(res1["metric"] - res2["metric"])
        assert diff <= tolerance, f"Non-deterministic metrics: {res1['metric']} vs {res2['metric']} (diff {diff})"
    finally:
        if os.path.exists(goal_single_seed.name):
            os.remove(goal_single_seed.name)


def test_04_leakage_detection(monkeypatch):
    """4. Leakage detection: A pipeline that reads validation targets in fit must fail."""
    class LeakyFitPipeline:
        def __init__(self, cfg):
            self.W_in = torch.randn(10, 1, requires_grad=False)
            self.W_res = torch.randn(10, 10, requires_grad=False)
            self.b = torch.zeros(10, requires_grad=False)
            self.W_out = torch.nn.Parameter(torch.zeros(1, 11), requires_grad=True)
            self.W_out._is_readout = True
        def frozen_tensors(self): return [self.W_in, self.W_res, self.b]
        def trainable_parameters(self): return [self.W_out]
        def stats(self): return {"params": 100, "state_dim": 10, "readout_features": 11}
        def fit(self, train):
            # Attempt to illegally access validation data during fit()
            _ = toy_task.load_val()
        def predict(self, inputs): return torch.zeros_like(inputs)
        def rollout(self, init, act, h): return torch.zeros(h, 1)

    monkeypatch.setattr("src.pipeline.build", lambda cfg: LeakyFitPipeline(cfg))
    res = evaluate.run_evaluation("GOAL.md")
    assert res["valid"] == 0
    assert "leakage" in res["invalid_reason"].lower()


def test_05_constraint_enforcement(monkeypatch):
    """5. Constraint enforcement: A pipeline exceeding parameter or state cap produces valid=0."""
    class BloatedPipeline:
        def __init__(self, cfg):
            self.W_in = torch.randn(10, 1, requires_grad=False)
            self.W_res = torch.randn(10, 10, requires_grad=False)
            self.b = torch.zeros(10, requires_grad=False)
            self.W_out = torch.nn.Parameter(torch.zeros(1, 11), requires_grad=True)
            self.W_out._is_readout = True
        def frozen_tensors(self): return [self.W_in, self.W_res, self.b]
        def trainable_parameters(self): return [self.W_out]
        def stats(self):
            # Exceed max_state_dim cap (cap is 2000 in GOAL.md)
            return {"params": 100, "state_dim": 5000, "readout_features": 11}
        def fit(self, train): pass
        def predict(self, inputs): return torch.zeros_like(inputs)
        def rollout(self, init, act, h): return torch.zeros(h, 1)

    monkeypatch.setattr("src.pipeline.build", lambda cfg: BloatedPipeline(cfg))
    res = evaluate.run_evaluation("GOAL.md")
    assert res["valid"] == 0
    assert "max_state_dim" in res["invalid_reason"]


def test_06_tamper_detection():
    """6. Tamper detection: Editing a file in protected/ is detected and restored by runner."""
    guards_file = os.path.join(REPO_ROOT, "protected", "guards.py")
    with open(guards_file, "r") as f:
        original_code = f.read()

    try:
        # Tamper with file
        with open(guards_file, "a") as f:
            f.write("\n# TAMPER_TEST_INJECTION\n")

        # Verify sha256 check fails
        sha_proc = subprocess.run(
            ["sha256sum", "--status", "-c", "lab/protected.sha256"],
            cwd=REPO_ROOT,
            capture_output=True
        )
        assert sha_proc.returncode != 0, "Tampering was not detected by sha256sum manifest!"

        # Restore from backup
        subprocess.run(
            ["rsync", "-a", "--delete", "lab/protected_backup/", "protected/"],
            cwd=REPO_ROOT,
            check=True
        )

        with open(guards_file, "r") as f:
            restored_code = f.read()
        assert restored_code == original_code, "File was not restored byte-identically!"
    finally:
        # Guarantee protected/ matches backup after test
        subprocess.run(
            ["rsync", "-a", "--delete", "lab/protected_backup/", "protected/"],
            cwd=REPO_ROOT,
            check=True
        )


def test_07_runner_keep_discard(tmp_path):
    """7. Runner keep/discard: Simulated RESULT lines produce expected keep/discard decisions."""
    best_file = str(tmp_path / "metric.json")
    res_file = str(tmp_path / "result.json")

    # Baseline establish
    with open(res_file, "w") as f:
        json_data = {
            "metric": 0.100000, "metric_name": "nmse", "std": 0.001,
            "seeds": 3, "valid": 1, "constraints": {"status": "ok", "violated": []},
            "secondary": {"params": 1000}
        }
        import json
        json.dump(json_data, f)

    dec_base = runner_helper.evaluate_decision(res_file, best_file)
    assert dec_base["status"] == "keep"
    assert dec_base.get("is_baseline") is True

    # Record best
    with open(best_file, "w") as f:
        json.dump({"id": "000", "metric": 0.100000, "metric_name": "nmse", "std": 0.001}, f)

    # Significant improvement (0.100 -> 0.050, gain 0.050 > max(0.01*0.1, 0.001, 0.0001))
    with open(res_file, "w") as f:
        json_data["metric"] = 0.050000
        json.dump(json_data, f)
    dec_keep = runner_helper.evaluate_decision(res_file, best_file)
    assert dec_keep["status"] == "keep"

    # Tiny improvement below threshold (0.100 -> 0.09995, gain 0.00005 < noise_floor 0.0001)
    with open(res_file, "w") as f:
        json_data["metric"] = 0.099950
        json.dump(json_data, f)
    dec_discard_noise = runner_helper.evaluate_decision(res_file, best_file)
    assert dec_discard_noise["status"] == "discard"

    # Worse metric (0.100 -> 0.120)
    with open(res_file, "w") as f:
        json_data["metric"] = 0.120000
        json.dump(json_data, f)
    dec_discard_worse = runner_helper.evaluate_decision(res_file, best_file)
    assert dec_discard_worse["status"] == "discard"

    # Invalid run
    with open(res_file, "w") as f:
        json_data["valid"] = 0
        json.dump(json_data, f)
    dec_inv = runner_helper.evaluate_decision(res_file, best_file)
    assert dec_inv["status"] == "invalid"


def test_08_detached_survival(tmp_path):
    """8. Detached survival: A started job keeps running after its parent shell exits."""
    out_file = str(tmp_path / "survival.txt")
    cmd = f"setsid nohup bash -c 'sleep 0.8; echo \"survived\" > \"{out_file}\"' >/dev/null 2>&1 &"
    
    # Run in short-lived subshell that terminates immediately
    subprocess.run(["bash", "-c", cmd], check=True)
    
    # Wait for detached job to complete
    time.sleep(1.2)
    assert os.path.exists(out_file), "Detached job did not survive parent shell exit!"
    with open(out_file, "r") as f:
        content = f.read().strip()
    assert content == "survived"


def test_09_test_split_lock(monkeypatch):
    """9. Test-split lock: Reading the test split without HARNESS_FINAL=1 must fail."""
    monkeypatch.delenv("HARNESS_FINAL", raising=False)
    with pytest.raises(PermissionError) as exc_info:
        toy_task.load_test()
    assert "HARNESS_FINAL=1" in str(exc_info.value)

    # With HARNESS_FINAL=1, it should successfully load
    monkeypatch.setenv("HARNESS_FINAL", "1")
    test_data = toy_task.load_test()
    assert test_data.inputs is not None
    assert test_data.targets is not None
    assert len(test_data.inputs) == 1000


def test_10_no_version_control():
    """10. No version control: Grep lab/ and protected/ for git fails on any match. .git unchanged."""
    # Grep lab/ and protected/ for git
    proc = subprocess.run(
        ["grep", "-rn", "-i", "git", "lab/", "protected/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    assert proc.returncode != 0, f"Found version control 'git' occurrences:\n{proc.stdout}"

    # Verify .git HEAD and object counts
    git_dir = os.path.join(REPO_ROOT, ".git")
    if os.path.exists(git_dir):
        head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).strip()
        objs_before = subprocess.check_output(["git", "count-objects", "-v"], cwd=REPO_ROOT).strip()

        # Run an evaluation
        _ = evaluate.run_evaluation("GOAL.md")

        head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).strip()
        objs_after = subprocess.check_output(["git", "count-objects", "-v"], cwd=REPO_ROOT).strip()

        assert head_before == head_after, "HEAD changed during evaluation!"
        assert objs_before == objs_after, "Git objects created during evaluation!"


def test_11_ledger_is_append_only(tmp_path):
    """11. Ledger is append-only: First row and header remain byte-identical after two experiments."""
    test_ledger = str(tmp_path / "test_ledger.tsv")
    header = "id\tts(YYYY-MM-DDTHH:MM)\ttag\tstatus\tmetric\tmetric_name\tstd\tvalid\tsecondary\tparent\tdiff\tconstraints\tdesc\n"
    with open(test_ledger, "w") as f:
        f.write(header)

    # Append first row
    runner_helper.append_ledger_row(
        test_ledger, "000", "baseline", "keep", "0.010000", "nmse", "0.001000",
        1, "params=90k", "000", "f1+100-0", "ok", "First baseline experiment"
    )

    with open(test_ledger, "rb") as f:
        bytes_after_first = f.read()

    # Append second row
    runner_helper.append_ledger_row(
        test_ledger, "001", "architecture", "discard", "0.012000", "nmse", "0.001000",
        1, "params=95k", "000", "f1+10-2", "ok", "Second experiment"
    )

    with open(test_ledger, "rb") as f:
        bytes_after_second = f.read()

    # Check prefix byte equality
    assert bytes_after_second.startswith(bytes_after_first), "First row or header was modified!"


def test_12_brief_size(tmp_path):
    """12. Brief size: After 200 simulated ledger rows, brief output stays under 1500 characters."""
    sim_ledger = str(tmp_path / "sim_ledger.tsv")
    header = "id\tts(YYYY-MM-DDTHH:MM)\ttag\tstatus\tmetric\tmetric_name\tstd\tvalid\tsecondary\tparent\tdiff\tconstraints\tdesc\n"
    with open(sim_ledger, "w") as f:
        f.write(header)

    sim_best = str(tmp_path / "best.json")
    import json
    with open(sim_best, "w") as f:
        json.dump({"id": "190", "metric": 0.005123, "metric_name": "nmse", "std": 0.00012}, f)

    for i in range(200):
        exp_id = f"{i:03d}"
        status = "keep" if i % 10 == 0 else "discard"
        metric = f"{1.0 / (i + 1):.6f}"
        runner_helper.append_ledger_row(
            sim_ledger, exp_id, "architecture", status, metric, "nmse", "0.0001",
            1, "params=90k;flops=1.8e+05;mem_mb=526.4;wall_s=0.11", "000",
            "f1+20-5", "ok", f"Simulated experiment description {i} testing brevity constraint."
        )

    brief_text = runner_helper.generate_brief(sim_ledger, sim_best)
    char_count = len(brief_text)
    assert char_count <= 1500, f"Brief exceeded 1500 characters: {char_count} chars"


def test_13_nan_and_inf_safety(monkeypatch):
    """13. NaN and Inf safety: NaN or Inf in predictions, weights, or metrics must produce valid=0."""
    class NanPredictPipeline:
        def __init__(self, cfg):
            self.W_in = torch.randn(10, 1, requires_grad=False)
            self.W_res = torch.randn(10, 10, requires_grad=False)
            self.b = torch.zeros(10, requires_grad=False)
            self.W_out = torch.nn.Parameter(torch.zeros(1, 11), requires_grad=True)
            self.W_out._is_readout = True
        def frozen_tensors(self): return [self.W_in, self.W_res, self.b]
        def trainable_parameters(self): return [self.W_out]
        def stats(self): return {"params": 100, "state_dim": 10, "readout_features": 11}
        def fit(self, train): pass
        def predict(self, inputs):
            # Returns tensor containing NaN
            out = torch.zeros_like(inputs)
            out[0, 0] = float("nan")
            return out
        def rollout(self, init, act, h): return torch.zeros(h, 1)

    monkeypatch.setattr("src.pipeline.build", lambda cfg: NanPredictPipeline(cfg))
    res = evaluate.run_evaluation("GOAL.md")
    assert res["valid"] == 0, "NaN in predict() was not marked invalid!"
    assert "nan" in res["invalid_reason"].lower()

    # Also test NaN handling in decision evaluation
    import tempfile, json
    with tempfile.NamedTemporaryFile("w", delete=False) as tf:
        json.dump({"metric": float("nan"), "metric_name": "nmse", "valid": 1, "constraints": {"status": "ok"}}, tf)
        tmp_name = tf.name
    try:
        dec = runner_helper.evaluate_decision(tmp_name, "lab/best/metric.json")
        assert dec["status"] == "invalid", "NaN metric was not rejected by runner decision logic!"
        assert "nan" in dec["reason"].lower()
    finally:
        os.remove(tmp_name)

