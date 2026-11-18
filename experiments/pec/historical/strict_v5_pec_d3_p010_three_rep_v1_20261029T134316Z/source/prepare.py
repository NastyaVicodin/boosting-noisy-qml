"""Prepare nine precommitted d3/p=.01 PEC third repetitions. No training or draws."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OLD = ROOT / "artifacts/strict_v5_pec240_d3_p010_v1_20261022T202903Z"
REF = ROOT / "artifacts/strict_v5_lowshot_depth_noise_3x3_dual_v1_20261020T1030Z"
PLAN = ROOT / "artifacts/strict_v5_pec_2v3_runtime_estimate_20261029T132235Z"
PLAN_FILE = PLAN / "PROPOSED_NEW_SLOTS.csv"
SOURCE_FILES = (
    "common.py", "audit.py", "endpoints.py", "lineage.py", "patience.py",
    "reference_kernel.py", "train_runner.py", "pec_measurements.py",
    "pec_kernel.py", "environment.sh", "runner.py",
)
MODULES = (
    "boosting_qml.pqc.density_matrix", "boosting_qml.pqc.measurement",
    "boosting_qml.pec.explicit_gatewise", "boosting_qml.training.convergence",
    "boosting_qml.weak_learners.pqc_binary", "numpy_compat",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def source_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> None:
    assert not (OUT / "manifest/runs.jsonl").exists()
    assert not (OUT / "manifest/frozen_execution.json").exists()
    assert not any((OUT / "runs").iterdir())
    assert (OUT / "source/launch_preflight.py").exists()
    assert (OUT / "source/submit.py").exists()

    plan = list(csv.DictReader(PLAN_FILE.open(newline="")))
    selected = [p for p in plan if int(p["depth"]) == 3
                and float(p["noise"]) == .01
                and int(p["pec_shots"]) in (180, 210, 240)
                and int(p["repetition"]) == 2]
    assert len(selected) == 9
    selected.sort(key=lambda p: (int(p["pec_shots"]), int(p["paper_teacher"])))
    assert {(int(p["pec_shots"]), int(p["paper_teacher"])) for p in selected} == {
        (n, t) for n in (180, 210, 240) for t in range(3)
    }

    originals = read_jsonl(OLD / "manifest/runs.jsonl")
    old_by_teacher = {int(r["teacher"]): r for r in originals
                      if int(r["teacher"]) in (2, 3, 4) and int(r["repetition"]) == 0}
    assert set(old_by_teacher) == {2, 3, 4}
    references = read_jsonl(REF / "manifest/runs.jsonl")
    ref_by_teacher = {int(r["teacher"]): r for r in references
                      if r["method"] == "reference" and int(r["depth"]) == 3
                      and int(r["teacher"]) in (2, 3, 4) and int(r["repetition"]) == 2}
    assert set(ref_by_teacher) == {2, 3, 4}

    rows = []
    for index, p in enumerate(selected):
        n = int(p["pec_shots"])
        paper_teacher = int(p["paper_teacher"])
        teacher = int(p["source_teacher"])
        assert teacher == (2, 3, 4)[paper_teacher]
        original = old_by_teacher[teacher]
        reference = ref_by_teacher[teacher]
        assert original["depth"] == reference["depth"] == 3
        assert original["noise_strength_1q"] == original["noise_strength_2q"] == .01
        assert reference["noise_strength_1q"] == reference["noise_strength_2q"] == 0
        assert original["train_split_sha256"] == reference["train_split_sha256"] == p["train_split_sha256"]
        assert original["validation_split_sha256"] == reference["validation_split_sha256"] == p["validation_split_sha256"]
        assert original["epsilon_mar"] == reference["epsilon_mar"] == float(p["margin_target"])
        assert original["minimum_correct"] == reference["minimum_correct"] == int(p["classification_minimum_correct"])
        assert original["a_star"] == reference["a_star"]
        assert original["target_id"] == reference["margin_target_id"]
        assert original["classification_target_id"] == reference["classification_target_id"]
        assert original["target_registry_sha256"] == reference["classification_target_registry_sha256"]
        assert original["margin_target_registry_sha256"] == reference["margin_target_registry_sha256"]
        assert original["teacher_sha256"] == reference["teacher_sha256"]
        assert original["patience_epochs"] == reference["patience_epochs"] == 100
        assert original["max_epochs"] == reference["max_epochs"] == 500
        assert original["max_updates"] == reference["max_updates"] == 12000
        assert original["online_budget_raw_shots"] == reference["online_budget_raw_shots"] == 549144980
        assert reference["pair_id"] == p["comparison_pair_id"]
        assert reference["initial_path"] == p["initial_path"]
        assert reference["orders_path"] == p["orders_path"]
        assert reference["initial_array_sha256"] == p["initial_array_sha256"]
        assert reference["orders_array_sha256"] == p["orders_array_sha256"]
        assert math.ceil(original["a_star"] * 384) == original["minimum_correct"]

        row = copy.deepcopy(original)
        row.update(index=index, repetition=2, paper_teacher=paper_teacher,
                   comparison_pair_id=p["comparison_pair_id"],
                   pair_id=p["proposed_pec_pair_id"],
                   run_id=f"d3_t{teacher:02d}_r02_pec{n}_patience100_dual_expand",
                   original_pec_template_run_id=original["run_id"],
                   reference_run_id=reference["run_id"],
                   initialization_seed=reference["initialization_seed"],
                   reference_stream_seeds=reference["reference_stream_seeds"])
        for field in ("initial_path", "initial_file_sha256", "initial_array_sha256",
                      "orders_path", "orders_sha256", "orders_array_sha256"):
            row[field] = reference[field]
        for field in ("N_fit", "N_monitor", "N_confirm", "N_diagnostic"):
            row[field] = n
        # Historical PEC's boost_seed is not used by this frozen PEC kernel.
        row.pop("boost_seed", None)
        for key in ("train_split", "validation_split", "teacher_path", "initial_path", "orders_path"):
            assert "/test/" not in row[key]
            assert Path(row[key]).exists() and sha(Path(row[key])) == row.get(key + "_sha256", sha(Path(row[key])))
        assert sha(Path(row["initial_path"])) == row["initial_file_sha256"]
        assert sha(Path(row["orders_path"])) == row["orders_sha256"]
        rows.append(row)

    assert len({r["pair_id"] for r in rows}) == len(rows) == 9
    for teacher in (2, 3, 4):
        members = [r for r in rows if r["teacher"] == teacher]
        assert len(members) == 3
        assert len({r["initial_array_sha256"] for r in members}) == 1
        assert len({r["orders_array_sha256"] for r in members}) == 1
        assert len({r["pair_id"] for r in members}) == 3

    # All estimator/training source bytes are inherited from the executed PEC240 package.
    copied = []
    for name in SOURCE_FILES:
        source = OLD / "source" / name
        target = OUT / "source" / name
        text = source.read_text()
        if name == "runner.py":
            assert text.count("choices=range(10)") == 1
            text = text.replace("choices=range(10)", "choices=range(9)")
        with target.open("x") as stream:
            stream.write(text)
        copied.append({"source": str(source), "destination": str(target),
                       "source_sha256": sha(source), "destination_sha256": sha(target)})
    for source in sorted((OLD / "frozen_src").rglob("*.py")):
        target = OUT / "frozen_src" / source.relative_to(OLD / "frozen_src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        copied.append({"source": str(source), "destination": str(target),
                       "source_sha256": sha(source), "destination_sha256": sha(target)})
    assert (OUT / "source/pec_kernel.py").read_text().count("assert n in (120,180,210,240)") == 1

    with (OUT / "manifest/runs.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    save_json(OUT / "manifest/lineage.json", {
        "utc": utc(), "source_campaign": str(OLD),
        "source_protocol_sha256": sha(OLD / "manifest/protocol.json"),
        "source_frozen_sha256": sha(OLD / "manifest/frozen_execution.json"),
        "reference_campaign": str(REF),
        "reference_manifest_sha256": sha(REF / "manifest/runs.jsonl"),
        "proposal": str(PLAN_FILE), "proposal_sha256": sha(PLAN_FILE),
        "copied_files": copied,
        "wrapper_delta": "Only runner index range 10 to 9; new preparation, preflight and submission wrappers. Training, estimator, noise, optimizer and stopping source unchanged.",
        "third_repetition_pairing": "N25 reference r02 initial and order arrays shared across N180/210/240; allocation-specific PEC pair_id seeds independent new streams.",
        "historical_pairing_caveat": "PEC r00/r01 at this depth/noise share arrays across allocations but differ from current N25 reference r00/r01 arrays. Historical PEC allocation streams also share tape prefixes.",
        "new_measurement_samples": 0, "historical_results_modified": False,
    })

    old_runtime = json.loads((OLD / "manifest/runtime.json").read_text())
    runtime_modules = {}
    for module in MODULES:
        relative = Path(*module.split("."))
        path = OUT / "frozen_src" / relative.with_suffix(".py")
        assert path.exists()
        assert sha(path) == old_runtime["modules"][module]["sha256"]
        runtime_modules[module] = source_record(path)
    save_json(OUT / "manifest/runtime.json", {
        "utc": utc(), "executable": old_runtime["executable"],
        "versions": old_runtime["versions"], "modules": runtime_modules,
    })

    protocol = {
        "schema": "strict_v5.pec_d3_p010_three_rep.v1", "utc": utc(),
        "user_authorized": True,
        "authorization": "User 2026-10-29: okay, lets submit this option then; context specifies d3,p=.01, PEC180/210/240, three repetitions per selected teacher.",
        "run_count": 9, "selected_source_teachers": [2, 3, 4],
        "repetition": 2, "shot_allocations": [180, 210, 240],
        "noise_strength_1q": .01, "noise_strength_2q": .01,
        "max_epochs": 500, "max_updates": 12000,
        "online_budget_raw_shots": 549144980,
        "patience": {"epochs": 100, "min_delta": 0,
                     "restore_best": True,
                     "metric": "Arithmetic mean of 24 sampled smooth-hinge minibatch fitting losses after each completed epoch; no warmup"},
        "stopping": "First independent confirmation of both targets, patience100, 500 epochs, or existing online-shot budget; restored best checkpoint checked independently for unresolved endpoints.",
        "target_policy": "Use unchanged unrounded training-margin threshold and integer validation-correct cutoff for each source teacher.",
        "gradients": "Frozen simulator-supplied exact ideal-mean VJP at sampled signed PEC forward score; no measured gradient shots.",
        "accounting": "Algorithm=fitting plus once-only edge plus measured gradient shots. Monitoring, confirmation and diagnostics separate; all acquired online shots count toward existing cap. No gamma-squared remultiplication.",
        "pairing": "All three allocations share each teacher's existing N25 reference r02 initial and order arrays. New allocation-specific PEC pair IDs separate measurement streams. Historical r00/r01 arrays and results remain unchanged.",
        "source_campaign": str(OLD), "reference_campaign": str(REF),
        "proposal_file": str(PLAN_FILE), "proposal_sha256": sha(PLAN_FILE),
        "scheduler": {"array": "0-8%9", "cpu_per_task": 1, "memory_gb_per_task": 8,
                      "wall_hours": 96, "walltime_is_queue_limit_not_scientific_stop": True,
                      "max_concurrent_training_tasks_across_current_campaigns": 20},
        "safety": {"test_access": False, "automatic_retry": False,
                   "historical_files_edited": False,
                   "existing_pec120_stays_held": True},
    }
    save_json(OUT / "manifest/protocol.json", protocol)
    protocol_hash = sha(OUT / "manifest/protocol.json")

    launch = OUT / "launch_pec_r02.sbatch"
    with launch.open("x") as stream:
        stream.write(f"""#!/bin/bash
#SBATCH --job-name=v5_pec_d3p010_r02
#SBATCH --partition=cpu-skylake
#SBATCH --array=0-8%9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=96:00:00
#SBATCH --output={OUT}/logs/pec_r02_%A_%a.out
#SBATCH --error={OUT}/logs/pec_r02_%A_%a.err
set -euo pipefail
cd {ROOT}
source {OUT}/source/environment.sh {OUT}
{ROOT}/.venv/bin/python -B {OUT}/source/launch_preflight.py
srun --ntasks=1 --cpu-bind=threads {ROOT}/.venv/bin/python -B {OUT}/source/runner.py --index \"$SLURM_ARRAY_TASK_ID\" --protocol-sha256 {protocol_hash}
""")

    frozen_paths = set((OUT / "source").glob("*.py")) | set((OUT / "source").glob("*.sh"))
    frozen_paths |= set((OUT / "frozen_src").rglob("*.py"))
    frozen_paths |= set((OUT / "manifest").glob("*.json"))
    frozen_paths |= {OUT / "manifest/runs.jsonl", launch, PLAN_FILE,
                     OLD / "manifest/protocol.json", OLD / "manifest/frozen_execution.json",
                     REF / "manifest/runs.jsonl"}
    for row in rows:
        for key in ("train_split", "validation_split", "initial_path", "orders_path",
                    "margin_target_registry", "target_registry", "teacher_path"):
            frozen_paths.add(Path(row[key]))
    assert all("/datasets/test/" not in str(path) and "test_prediction" not in path.name
               for path in frozen_paths)
    save_json(OUT / "manifest/frozen_execution.json", {
        "utc": utc(), "protocol_sha256": protocol_hash,
        "files": [source_record(path) for path in sorted(frozen_paths)],
        "policy": "Immutable after submission; only new receipts and acquired run records may be appended.",
    })
    print(json.dumps({"campaign": str(OUT), "run_count": len(rows),
                      "protocol_sha256": protocol_hash,
                      "frozen_sha256": sha(OUT / "manifest/frozen_execution.json"),
                      "frozen_file_count": len(frozen_paths),
                      "measurement_samples_drawn": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
