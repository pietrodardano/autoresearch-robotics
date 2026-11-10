"""
Runner helper script for the autoresearch harness.
Handles configuration parsing, metric comparison, ledger management,
and brief generation without any version control interactions.
"""

from __future__ import annotations
import argparse
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional
import yaml


def load_goal_config(goal_path: str = "GOAL.md") -> Dict[str, Any]:
    """Loads and parses GOAL.md configuration."""
    if not os.path.exists(goal_path):
        return {}
    with open(goal_path, "r", encoding="utf-8") as f:
        content = f.read()
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            parsed = yaml.safe_load(parts[1])
            if isinstance(parsed, dict):
                return parsed
    parsed = yaml.safe_load(content)
    return parsed if isinstance(parsed, dict) else {}


def next_exp_id(exp_dir: str = "lab/exp") -> str:
    """Finds next sequential 3-character numeric experiment ID (000, 001, ...)."""
    if not os.path.exists(exp_dir):
        return "000"
    existing = []
    for item in os.listdir(exp_dir):
        if len(item) == 3 and item.isnumeric():
            existing.append(int(item))
    if not existing:
        return "000"
    return f"{max(existing) + 1:03d}"


def get_hard_timeout(goal_path: str = "GOAL.md") -> int:
    """Returns hard timeout in seconds from GOAL.md."""
    cfg = load_goal_config(goal_path)
    return int(cfg.get("hard_timeout", 120))


def format_secondary(secondary_dict: Dict[str, Any]) -> str:
    """Formats secondary metrics into compact k=v pairs joined by ';'."""
    items = []
    if "params" in secondary_dict:
        p = int(secondary_dict["params"])
        items.append(f"params={p//1000}k" if p >= 1000 else f"params={p}")
    if "sop" in secondary_dict:
        items.append(f"sop={secondary_dict['sop']:.1e}")
    elif "flops_sops" in secondary_dict:
        items.append(f"flops={secondary_dict['flops_sops']:.1e}")
    if "mem_mb" in secondary_dict:
        mb = float(secondary_dict["mem_mb"])
        items.append(f"mem_mb={mb:.1f}")
    if "wall_s" in secondary_dict:
        ws = float(secondary_dict["wall_s"])
        items.append(f"wall_s={ws:.2f}")
    for k, v in secondary_dict.items():
        if k not in ("params", "sop", "flops_sops", "mem_mb", "wall_s", "raw_metric"):
            if isinstance(v, float):
                items.append(f"{k}={v:.4f}")
            else:
                items.append(f"{k}={v}")
    return ";".join(items)


def evaluate_decision(
    result_json_path: str,
    best_metric_path: str,
    goal_path: str = "GOAL.md"
) -> Dict[str, Any]:
    """
    Evaluates keep/discard decision based on RESULT line and GOAL.md rules.
    Returns status, formatted values, and ledger details.
    """
    cfg = load_goal_config(goal_path)
    min_gain = float(cfg.get("min_gain", 0.01))
    noise_floor = float(cfg.get("noise_floor", 0.0001))

    # Read result line
    if not os.path.exists(result_json_path):
        return {
            "status": "crash",
            "metric": "nan",
            "metric_name": "unknown",
            "std": "0.0",
            "valid": 0,
            "secondary": "",
            "constraints": "crash",
            "reason": "Missing result.json",
        }

    with open(result_json_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    valid = int(res.get("valid", 0))
    inv_reason = res.get("invalid_reason", "")
    constraints_info = res.get("constraints", {})
    constraints_status = constraints_info.get("status", "ok")
    violated = constraints_info.get("violated", [])

    metric_name = res.get("metric_name", "metric")
    metric_val = res.get("metric", float("nan"))
    std_val = float(res.get("std", 0.0))
    secondary_str = format_secondary(res.get("secondary", {}))

    constraints_col = "ok" if constraints_status == "ok" and not violated else (
        ",".join(violated) if violated else "violated"
    )

    if valid == 0 or constraints_status != "ok" or violated:
        return {
            "status": "invalid",
            "metric": f"{metric_val:.6f}" if isinstance(metric_val, (int, float)) and not str(metric_val).lower().startswith("nan") else "nan",
            "metric_name": metric_name,
            "std": f"{std_val:.6f}",
            "valid": 0,
            "secondary": secondary_str,
            "constraints": constraints_col,
            "reason": inv_reason or constraints_col,
        }

    # Strict NaN and Inf verification
    try:
        m_num = float(metric_val)
        if math.isnan(m_num) or math.isinf(m_num):
            return {
                "status": "invalid",
                "metric": "nan",
                "metric_name": metric_name,
                "std": f"{std_val:.6f}",
                "valid": 0,
                "secondary": secondary_str,
                "constraints": "nan_detected",
                "reason": "Non-finite metric value (NaN or Inf)",
            }
    except (ValueError, TypeError):
        return {
            "status": "invalid",
            "metric": "nan",
            "metric_name": metric_name,
            "std": f"{std_val:.6f}",
            "valid": 0,
            "secondary": secondary_str,
            "constraints": "invalid_metric",
            "reason": f"Unparseable metric value: {metric_val}",
        }

    # Valid run: check against previous best
    if not os.path.exists(best_metric_path):
        # Baseline run: first valid result is kept
        return {
            "status": "keep",
            "metric": f"{metric_val:.6f}",
            "metric_name": metric_name,
            "std": f"{std_val:.6f}",
            "valid": 1,
            "secondary": secondary_str,
            "constraints": constraints_col,
            "is_baseline": True,
            "raw_metric": metric_val,
            "raw_std": std_val,
        }

    with open(best_metric_path, "r", encoding="utf-8") as f:
        best_data = json.load(f)

    best_metric = float(best_data["metric"])
    gain = best_metric - float(metric_val)
    threshold = max(min_gain * abs(best_metric), std_val, noise_floor)

    if gain > threshold:
        status = "keep"
        reason = f"Metric improved by {gain:.6f} > threshold {threshold:.6f}"
    else:
        status = "discard"
        reason = f"Metric improvement {gain:.6f} <= threshold {threshold:.6f}"

    return {
        "status": status,
        "metric": f"{metric_val:.6f}",
        "metric_name": metric_name,
        "std": f"{std_val:.6f}",
        "valid": 1,
        "secondary": secondary_str,
        "constraints": constraints_col,
        "best_metric": best_metric,
        "gain": gain,
        "threshold": threshold,
        "reason": reason,
        "raw_metric": metric_val,
        "raw_std": std_val,
    }


def append_ledger_row(
    ledger_path: str,
    exp_id: str,
    tag: str,
    status: str,
    metric: str,
    metric_name: str,
    std: str,
    valid: int,
    secondary: str,
    parent: str,
    diff_summary: str,
    constraints: str,
    desc: str
) -> None:
    """Appends a single tab-separated row to the ledger."""
    now_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    clean_desc = desc.replace("\t", " ").replace("\n", " ")[:120]
    
    row = f"{exp_id}\t{now_ts}\t{tag}\t{status}\t{metric}\t{metric_name}\t{std}\t{valid}\t{secondary}\t{parent}\t{diff_summary}\t{constraints}\t{clean_desc}\n"
    
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(row)


def generate_brief(ledger_path: str = "lab/ledger.tsv", best_metric_path: str = "lab/best/metric.json") -> str:
    """Generates the brief summary (strictly <= 1500 characters)."""
    best_str = "none"
    if os.path.exists(best_metric_path):
        try:
            with open(best_metric_path, "r", encoding="utf-8") as f:
                b = json.load(f)
            best_str = f"ID={b.get('id', '?')} {b.get('metric_name', 'metric')}={float(b.get('metric', 0.0)):.6f} (std={float(b.get('std', 0.0)):.6f})"
        except Exception:
            pass

    status_counts = {"keep": 0, "discard": 0, "invalid": 0, "crash": 0, "tampered": 0}
    rows = []
    keeps = []

    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\r\n") for l in f]
        
        # Skip header
        data_lines = lines[1:] if len(lines) > 1 else []
        for line in data_lines:
            parts = line.split("\t")
            if len(parts) >= 13:
                r_id, r_ts, r_tag, r_stat, r_met, r_mname, r_std, r_val, r_sec, r_par, r_diff, r_con, r_desc = parts[:13]
                entry = {
                    "id": r_id, "tag": r_tag, "status": r_stat, "metric": r_met,
                    "metric_name": r_mname, "std": r_std, "secondary": r_sec,
                    "diff": r_diff, "desc": r_desc
                }
                rows.append(entry)
                if r_stat in status_counts:
                    status_counts[r_stat] += 1
                if r_stat == "keep":
                    try:
                        m_val = float(r_met)
                        keeps.append((m_val, entry))
                    except ValueError:
                        pass

    counts_str = " ".join(f"{k}={v}" for k, v in status_counts.items())
    
    lines_out = []
    lines_out.append(f"Best: {best_str}")
    lines_out.append(f"Counts: {counts_str}")
    
    lines_out.append("Top 5 keeps:")
    keeps.sort(key=lambda x: x[0])
    for m, r in keeps[:5]:
        lines_out.append(f"  {r['id']}: {m:.6f} tag={r['tag']} {r['secondary']}")
    if not keeps:
        lines_out.append("  (none)")

    lines_out.append("Last 8 experiments:")
    for r in rows[-8:]:
        desc_trunc = r["desc"][:80]
        lines_out.append(f"  {r['id']} {r['status']:<7} {r['metric']:<10} {r['tag']:<12} {desc_trunc}")
    if not rows:
        lines_out.append("  (none)")

    output = "\n".join(lines_out)
    if len(output) > 1500:
        output = output[:1497] + "..."
    return output


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    subparsers.add_parser("next_id")
    subparsers.add_parser("timeout")
    
    dec_parser = subparsers.add_parser("decision")
    dec_parser.add_argument("--result", required=True)
    dec_parser.add_argument("--best", default="lab/best/metric.json")
    dec_parser.add_argument("--goal", default="GOAL.md")

    ledger_parser = subparsers.add_parser("append_ledger")
    ledger_parser.add_argument("--ledger", default="lab/ledger.tsv")
    ledger_parser.add_argument("--id", required=True)
    ledger_parser.add_argument("--tag", required=True)
    ledger_parser.add_argument("--status", required=True)
    ledger_parser.add_argument("--metric", required=True)
    ledger_parser.add_argument("--metric-name", required=True)
    ledger_parser.add_argument("--std", required=True)
    ledger_parser.add_argument("--valid", type=int, required=True)
    ledger_parser.add_argument("--secondary", required=True)
    ledger_parser.add_argument("--parent", required=True)
    ledger_parser.add_argument("--diff", required=True)
    ledger_parser.add_argument("--constraints", required=True)
    ledger_parser.add_argument("--desc", required=True)

    brief_parser = subparsers.add_parser("brief")
    brief_parser.add_argument("--ledger", default="lab/ledger.tsv")
    brief_parser.add_argument("--best", default="lab/best/metric.json")

    args = parser.parse_args()

    if args.cmd == "next_id":
        print(next_exp_id())
    elif args.cmd == "timeout":
        print(get_hard_timeout())
    elif args.cmd == "decision":
        res = evaluate_decision(args.result, args.best, args.goal)
        print(json.dumps(res))
    elif args.cmd == "append_ledger":
        append_ledger_row(
            args.ledger, args.id, args.tag, args.status, args.metric,
            args.metric_name, args.std, args.valid, args.secondary,
            args.parent, args.diff, args.constraints, args.desc
        )
    elif args.cmd == "brief":
        print(generate_brief(args.ledger, args.best))


if __name__ == "__main__":
    main()
