#!/usr/bin/env python3
"""Audit two verified MTP reports using raw captures, excluding warmups from timing.

Usage: python3 scripts/audit_mtp_reports.py baseline.json candidate.json > audit.json
Exit 2 rejects malformed or incomparable inputs; a slower candidate still exits 0.
This checks recorded evidence, not model quality or unrecorded machine conditions.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


CONFIG = (
    "device", "model_path", "mtp_path", "context_capacity", "block_size",
    "max_tokens", "measured_runs_per_prompt_per_mode", "sampling", "workload",
    "kernel_mode", "norm_mode", "metadata_mode", "attention_mode",
    "comparison_enabled",
)
COUNTS = ("completion_tokens", "accepted_drafts", "proposed_drafts", "rounds")
PHASES = (
    "target_seconds", "draft_seconds", "verification_seconds", "rollback_seconds",
    "prefill_target_seconds", "prefill_draft_seconds",
)
MODES = ("mtp", "sequential_target")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")
    return value


def number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: invalid number")
    require(value > 0 if positive else value >= 0, f"{label}: invalid duration")
    return value


def capture_key(capture):
    return capture["prompt_name"], capture["run"], capture["mode"]


def nominal(capture):
    return all(capture[side]["thermal_state"] == "nominal"
               for side in ("conditions_before", "conditions_after"))


def read_report(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    require(isinstance(report, dict), "report must be an object")
    require(report["kind"] == "verified_mtp_chat_benchmark", "wrong benchmark kind")
    require(isinstance(report["engine_version"], str) and report["engine_version"],
            "missing engine_version")
    for field in CONFIG:
        require(field in report, f"missing configuration: {field}")
    require(report["comparison_enabled"] is True, "paired comparison must be enabled")
    runs = integer(report["measured_runs_per_prompt_per_mode"], "measured runs", 1)
    block = integer(report["block_size"], "block_size", 1)
    integer(report["context_capacity"], "context_capacity", 1)
    integer(report["max_tokens"], "max_tokens", 2)
    require(isinstance(report["workload"], list) and report["workload"], "empty workload")
    names = set()
    prompt_lengths = {}
    for prompt in report["workload"]:
        name = prompt["name"]
        require(isinstance(name, str) and name and name not in names, "invalid/duplicate prompt")
        names.add(name)
        prompt_lengths[name] = integer(prompt["prompt_tokens"], "prompt_tokens", 1)
        require(isinstance(prompt["prompt_token_ids"], list), "invalid prompt token IDs")
        require(len(prompt["prompt_token_ids"]) == prompt_lengths[name], "prompt length mismatch")
        for token in prompt["prompt_token_ids"]:
            integer(token, "prompt token ID")
    require(isinstance(report["captures"], list) and report["captures"], "empty captures")
    keyed = {}
    for index, capture in enumerate(report["captures"]):
        label = f"capture[{index}]"
        key = capture_key(capture)
        require(key not in keyed, f"{label}: duplicate key {key}")
        require(capture["prompt_name"] in names, f"{label}: unknown prompt")
        require(capture["mode"] in MODES, f"{label}: unknown mode")
        run = integer(capture["run"], f"{label}.run")
        require(run <= runs, f"{label}: unexpected run")
        require(type(capture["warmup"]) is bool and capture["warmup"] == (run == 0),
                f"{label}: warmup/run disagreement")
        require(capture["prompt_tokens"] == prompt_lengths[capture["prompt_name"]],
                f"{label}: prompt length disagreement")
        stats = capture["stats"]
        for field in COUNTS:
            integer(stats[field], f"{label}.{field}")
        ids = stats["token_ids"]
        require(isinstance(ids, list) and len(ids) >= 2, f"{label}: insufficient output IDs")
        for token in ids:
            integer(token, f"{label}: output ID")
        require(len(ids) == capture["completion_tokens"] == stats["completion_tokens"],
                f"{label}: completion count disagrees with IDs")
        require(len(ids) <= report["max_tokens"], f"{label}: excessive completion length")
        require(capture["sustained_decode_tokens"] == len(ids) - 1,
                f"{label}: sustained count must exclude initial prefill token")
        require(isinstance(capture["output_text"], str), f"{label}: invalid output text")
        require(capture["finish_reason"] in ("length", "stop"), f"{label}: invalid finish reason")
        seconds = number(capture["decode_seconds"], f"{label}.decode_seconds", True)
        require(stats["decode_seconds"] == seconds, f"{label}: decode duration disagreement")
        number(capture["prefill_seconds"], f"{label}.prefill_seconds")
        number(capture["generation_wall_seconds"], f"{label}.generation_wall_seconds", True)
        require(math.isclose(capture["sustained_decode_tokens_per_second"],
                             (len(ids) - 1) / seconds, rel_tol=1e-9), f"{label}: rate disagreement")
        for field in PHASES:
            number(stats[field], f"{label}.{field}")
        require(stats["accepted_drafts"] <= stats["proposed_drafts"], f"{label}: invalid acceptance")
        widths = stats["round_widths"]
        require(isinstance(widths, list) and len(widths) == stats["rounds"],
                f"{label}: round count disagreement")
        require(all(type(width) is int and 1 <= width <= block for width in widths),
                f"{label}: invalid round width")
        if capture["mode"] == "sequential_target":
            require(all(stats[field] == 0 for field in COUNTS[1:]), f"{label}: sequential draft counts")
        for side in ("conditions_before", "conditions_after"):
            require(type(capture[side]["low_power_mode"]) is bool, f"{label}: invalid power status")
            require(capture[side]["thermal_state"] in ("nominal", "fair", "serious", "critical"),
                    f"{label}: invalid thermal status")
        keyed[key] = capture
    expected = {(name, run, mode) for name in names for run in range(runs + 1) for mode in MODES}
    require(set(keyed) == expected, "missing or unexpected prompt/run/mode captures")
    report["_keyed"] = keyed
    report["_source"] = {"path": str(path), "engine_version": report["engine_version"],
                         "sha256": hashlib.sha256(raw).hexdigest()}
    return report


def summarize(captures):
    if not captures:
        return None
    seconds = sum(c["decode_seconds"] for c in captures)
    tokens = sum(len(c["stats"]["token_ids"]) - 1 for c in captures)
    return {
        "measured_captures": len(captures), "sustained_decode_tokens": tokens,
        "decode_seconds": seconds, "weighted_sustained_tps": tokens / seconds,
        "median_sustained_tps": statistics.median(
            (len(c["stats"]["token_ids"]) - 1) / c["decode_seconds"] for c in captures),
        "counts": {field: sum(c["stats"][field] for c in captures) for field in COUNTS},
        "phase_seconds": {field: sum(c["stats"][field] for c in captures) for field in PHASES},
    }


def compare_subset(baseline, candidate, keys):
    result = {}
    for mode in MODES:
        selected = sorted(key for key in keys if key[2] == mode and key[1] > 0)
        before, after = [summarize([r["_keyed"][key] for key in selected]) for r in (baseline, candidate)]
        result[mode] = {"baseline": before, "candidate": after}
        if before and after:
            result[mode].update({
                "weighted_tps_change_percent": (after["weighted_sustained_tps"] / before["weighted_sustained_tps"] - 1) * 100,
                "decode_seconds_delta": after["decode_seconds"] - before["decode_seconds"],
                "phase_seconds_delta": {field: after["phase_seconds"][field] - before["phase_seconds"][field] for field in PHASES},
            })
    return result


def initial_nominal_keys(report):
    keys = set()
    for capture in report["captures"]:
        if not nominal(capture):
            break
        keys.add(capture_key(capture))
    return keys


def conditions(report):
    first = next(((i, c) for i, c in enumerate(report["captures"]) if not nominal(c)), None)
    return {
        "low_power_values": sorted({c[s]["low_power_mode"] for c in report["captures"]
                                    for s in ("conditions_before", "conditions_after")}),
        "thermal_states": sorted({c[s]["thermal_state"] for c in report["captures"]
                                   for s in ("conditions_before", "conditions_after")}),
        "first_non_nominal_capture": None if first is None else {
            "capture_index": first[0], "key": capture_key(first[1]),
            "before": first[1]["conditions_before"], "after": first[1]["conditions_after"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        reports = []
        for role, path in (("baseline", args.baseline), ("candidate", args.candidate)):
            try:
                reports.append(read_report(path))
            except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
                raise ValueError(f"{role} {path}: {exc}") from exc
        baseline, candidate = reports
        versions = f"baseline {baseline['engine_version']} -> candidate {candidate['engine_version']}"
        for field in CONFIG:
            require(baseline[field] == candidate[field], f"{versions}: configuration mismatch: {field}")
        for key, before in baseline["_keyed"].items():
            after = candidate["_keyed"][key]
            for field in ("warmup", "output_text", "finish_reason", "completion_tokens", "sustained_decode_tokens"):
                require(before[field] == after[field], f"{versions}: capture {key} mismatch: {field}")
            for field in (*COUNTS, "round_widths", "token_ids"):
                require(before["stats"][field] == after["stats"][field], f"{versions}: capture {key} stats mismatch: {field}")
        keys = set(baseline["_keyed"])
        audit = {
            "kind": "paired_mtp_report_audit", "baseline": baseline["_source"],
            "candidate": candidate["_source"], "matching_capture_count": len(keys),
            "all_output_traces_and_counts_equal": True,
            "conditions": {"baseline": conditions(baseline), "candidate": conditions(candidate)},
            "all_measured": compare_subset(baseline, candidate, keys),
            "matching_initial_nominal_measured": compare_subset(
                baseline, candidate, initial_nominal_keys(baseline) & initial_nominal_keys(candidate)),
            "per_prompt": {prompt["name"]: compare_subset(
                baseline, candidate, {key for key in keys if key[0] == prompt["name"]})
                for prompt in baseline["workload"]},
            "notes": "Rates and phase sums are recomputed from raw measured captures. Warmups are checked for trace equality but excluded from timing. Initial-nominal comparison intersects keys before either report's first non-nominal snapshot. Matching snapshots cannot exclude unrecorded differences in machine conditions.",
        }
        print(json.dumps(audit, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(f"audit rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
