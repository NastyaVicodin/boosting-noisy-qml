#!/usr/bin/env python3
"""Stage one fresh selected-task rerun in a new directory without executing it.

The staged run uses the saved paper-relevant scientific source, seeds, arrays,
targets, limits and shot allocations. Its one-run manifest and source-path
freeze are newly generated for portability; this is not a bytewise replay of
the historical campaign or a continuation from an inherited checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PATH_KEYS = ("teacher_path", "train_split", "validation_split", "initial_path",
             "orders_path", "margin_target_registry", "classification_target_registry",
             "target_registry")
PUBLICATION_KEYS = ("prescribed_campaign", "prescribed_manifest_sha256",
                    "configuration", "paper_teacher", "selected_attempt_kind",
                    "selected_result_sha256", "selected_start_sha256",
                    "source_start_sha256", "source_campaign",
                    "paired_n25_initial_and_order")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_for(root: Path, row: dict, pec: bool) -> Path:
    if not pec:
        name = row["prescribed_campaign"]
        aliases = {
            "strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z": "base",
            "strict_v5_lowshot_depth_noise_3x3_ref_fix_v2_20261020T104700Z": "reference",
            "strict_v5_fixed_noise_p01_p02_dual_v1_20261020T132536Z": "higher_noise",
        }
        return root / "experiments" / aliases[name] / "source"
    source = row["source_campaign"]
    if source.endswith("/pilot10"):
        return root / "experiments/pec/pilot10/source"
    if source.endswith("/explore21"):
        return root / "experiments/pec/explore21/source"
    name = Path(source).name
    return root / "experiments/pec/historical" / name / "source"


def materialize(root: Path, output_root: Path, row: dict, pec: bool) -> tuple[Path, str]:
    root = root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root == root or root in output_root.parents:
        raise ValueError("Rerun output must be outside the publication snapshot")
    identity = row["run_id"]
    out = output_root / "artifacts" / ("rerun_" + identity)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    source = source_for(root, row, pec)
    if not source.is_dir() or not (source / "runner.py").is_file():
        raise FileNotFoundError("Executed runner source unavailable")
    out.mkdir(parents=True)
    shutil.copytree(source, out / "source")
    shutil.copytree(root / "src", out / "frozen_src")
    manifest = out / "manifest"
    manifest.mkdir()
    scientific = {k: v for k, v in row.items() if k not in PUBLICATION_KEYS}
    scientific["index"] = 0
    for key in PATH_KEYS:
        if key in scientific:
            path = (root / scientific[key]).resolve(strict=True)
            if root not in path.parents:
                raise ValueError(f"Path leaves publication snapshot: {key}")
            scientific[key] = str(path)
    runs = manifest / "runs.jsonl"
    runs.write_text(json.dumps(scientific, sort_keys=True, allow_nan=False) + "\n")
    protocol = manifest / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "boosting_noisy_qml.portable_one_run.v1",
        "source_kind": "explicit_pec" if pec else "n25_reference_or_boosting",
        "historical_run_id": identity,
        "historical_result_sha256": row.get("selected_result_sha256"),
        "historical_manifest_sha256": row.get("prescribed_manifest_sha256"),
        "historical_source_campaign": row.get("source_campaign") or row.get("prescribed_campaign"),
        "single_run_index_rebased_to_zero": True,
        "reused_frozen_task_inputs_and_targets": True,
        "paper_scientific_source_unchanged": True,
        "unrelated_fashion_mnist_cache_fallback_removed": True,
        "historical_continuation_state_recreated": False,
        "campaign_specific_launch_preflight_recreated": False,
        "full_historical_environment_not_certified": True,
    }, indent=2, sort_keys=True) + "\n")
    paths = sorted((out / "source").glob("*.py"))
    paths += sorted((out / "frozen_src").rglob("*.py"))
    paths += [protocol, runs]
    paths += [Path(scientific[key]) for key in PATH_KEYS if key in scientific]
    freeze = manifest / "frozen_execution.json"
    freeze.write_text(json.dumps({
        "schema": "boosting_noisy_qml.portable_freeze.v1",
        "files": [{"path": str(path), "sha256": digest(path)} for path in paths],
    }, indent=2, sort_keys=True) + "\n")
    return out, digest(protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id")
    group.add_argument("--pec-slot", nargs=3, metavar=("CONFIG", "TEACHER", "REPETITION"))
    args = parser.parse_args()
    filename = ("explicit_pec_runs.jsonl" if args.pec_slot else "n25_runs.jsonl")
    rows = [json.loads(line) for line in
            (args.snapshot / "run_specs" / filename).read_text().splitlines() if line]
    matches = ([r for r in rows if r["run_id"] == args.run_id] if args.run_id else
               [r for r in rows if (r["configuration"], str(r["paper_teacher"]),
                                    str(r["repetition"])) == tuple(args.pec_slot)])
    if len(matches) != 1:
        raise ValueError("Exactly one saved run must match the requested identity")
    out, protocol_hash = materialize(args.snapshot, args.output_root, matches[0],
                                     pec=bool(args.pec_slot))
    print(json.dumps({"staged": str(out), "training_started": False,
                      "manual_training_command":
                      f"python {out / 'source/runner.py'} --index 0 --protocol-sha256 {protocol_hash}"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
