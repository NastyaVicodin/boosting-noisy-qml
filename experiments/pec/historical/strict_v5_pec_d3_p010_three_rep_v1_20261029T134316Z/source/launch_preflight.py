"""Source, input and manifest checks only; never evaluates a circuit."""
from common import *
from reference_kernel import array_hash
import ast
import importlib
import importlib.metadata
import math
import numpy as np


def preflight():
    frozen = check_frozen()
    protocol = load_json(OUT / "manifest/protocol.json")
    rows = jsonl(OUT / "manifest/runs.jsonl")
    assert frozen["protocol_sha256"] == digest(OUT / "manifest/protocol.json")
    assert protocol["user_authorized"] and not protocol["safety"]["test_access"]
    assert protocol["run_count"] == len(rows) == 9
    assert [row["index"] for row in rows] == list(range(9))
    assert {(row["N_fit"], row["teacher"], row["repetition"]) for row in rows} == {
        (n, teacher, 2) for n in (180, 210, 240) for teacher in (2, 3, 4)
    }
    assert len({row["run_id"] for row in rows}) == len({row["pair_id"] for row in rows}) == 9
    assert protocol["scheduler"]["wall_hours"] == 96
    assert protocol["scheduler"]["max_concurrent_training_tasks_across_current_campaigns"] == 20
    assert protocol["online_budget_raw_shots"] == 549144980

    proposal = list(csv.DictReader(Path(protocol["proposal_file"]).open(newline="")))
    selected = {(int(p["pec_shots"]), int(p["source_teacher"])): p for p in proposal
                if int(p["depth"]) == 3 and float(p["noise"]) == .01
                and int(p["pec_shots"]) in (180, 210, 240) and int(p["repetition"]) == 2}
    assert len(selected) == 9
    reference = jsonl(Path(protocol["reference_campaign"]) / "manifest/runs.jsonl")
    reference = {r["teacher"]: r for r in reference if r["method"] == "reference"
                 and r["depth"] == 3 and r["teacher"] in (2, 3, 4)
                 and r["repetition"] == 2}
    assert set(reference) == {2, 3, 4}
    original = jsonl(Path(protocol["source_campaign"]) / "manifest/runs.jsonl")
    original = {r["teacher"]: r for r in original if r["teacher"] in (2, 3, 4)
                and r["repetition"] == 0}
    assert set(original) == {2, 3, 4}
    assert digest(Path(protocol["proposal_file"])) == protocol["proposal_sha256"]

    for row in rows:
        n, teacher = row["N_fit"], row["teacher"]
        p, ref, old = selected[n, teacher], reference[teacher], original[teacher]
        assert row["kind"] == "train" and row["method"] == "explicit_sampled_pec"
        assert row["N_fit"] == row["N_monitor"] == row["N_confirm"] == row["N_diagnostic"] == n
        assert row["N_edge"] == row["N_gradient"] == 0
        assert row["depth"] == 3 and row["noise_strength_1q"] == row["noise_strength_2q"] == .01
        assert row["mu"] == .1 and row["batch_size"] == 64
        assert row["max_updates"] == 12000 and row["max_epochs"] == 500
        assert row["patience_epochs"] == 100 and row["patience_min_delta"] == 0 and row["restore_best"]
        assert row["online_budget_raw_shots"] == 549144980
        assert row["gradient_convention"] == "exact_ideal_mean_vjp_at_signed_PEC_score"
        assert row["minimum_correct"] == int(p["classification_minimum_correct"])
        assert row["minimum_correct"] == math.ceil(row["a_star"] * 384)
        assert row["epsilon_mar"] == float(p["margin_target"])
        assert row["pair_id"] == p["proposed_pec_pair_id"]
        assert row["comparison_pair_id"] == p["comparison_pair_id"] == ref["pair_id"]
        assert row["paper_teacher"] == int(p["paper_teacher"])
        assert row["original_pec_template_run_id"] == old["run_id"]
        assert row["reference_run_id"] == ref["run_id"]
        for key in ("train_split", "validation_split", "teacher_path",
                    "margin_target_registry", "target_registry"):
            assert row[key] == old[key]
        for key in ("train_split_sha256", "validation_split_sha256", "teacher_sha256",
                    "epsilon_mar", "minimum_correct", "a_star"):
            assert row[key] == old[key]
        for key in ("initial_path", "initial_file_sha256", "initial_array_sha256",
                    "orders_path", "orders_sha256", "orders_array_sha256",
                    "initialization_seed", "reference_stream_seeds"):
            assert row[key] == ref[key]
        assert row["initial_array_sha256"] == p["initial_array_sha256"]
        assert row["orders_array_sha256"] == p["orders_array_sha256"]
        for key in ("initial_path", "orders_path"):
            assert "/test/" not in row[key] and "test_prediction" not in row[key]
        assert digest(row["initial_path"]) == row["initial_file_sha256"]
        assert digest(row["orders_path"]) == row["orders_sha256"]
        initial = np.load(row["initial_path"], allow_pickle=False)
        orders = np.load(row["orders_path"], allow_pickle=False)
        assert array_hash(initial) == row["initial_array_sha256"]
        assert array_hash(orders) == row["orders_array_sha256"]
        assert orders.shape == (500, 1536) and initial.shape == (15, 4)
    for teacher in (2, 3, 4):
        group = [r for r in rows if r["teacher"] == teacher]
        assert len({r["initial_array_sha256"] for r in group}) == 1
        assert len({r["orders_array_sha256"] for r in group}) == 1
        assert len({r["pair_id"] for r in group}) == 3

    old_dir = Path(protocol["source_campaign"])
    for name in ("common.py", "audit.py", "endpoints.py", "lineage.py", "patience.py",
                 "reference_kernel.py", "train_runner.py", "pec_measurements.py", "pec_kernel.py"):
        assert digest(OUT / "source" / name) == digest(old_dir / "source" / name)
    old_runner = (old_dir / "source/runner.py").read_text()
    assert (OUT / "source/runner.py").read_text() == old_runner.replace("choices=range(10)", "choices=range(9)")
    kernel = (OUT / "source/pec_kernel.py").read_text()
    assert kernel.count("assert n in (120,180,210,240)") == 1
    assert "n_qubits=4,n_layers=3,noise_strength_1q=.01,noise_strength_2q=.01" in kernel
    for path in list((OUT / "source").glob("*.py")) + list(FROZEN.rglob("*.py")):
        ast.parse(path.read_text())

    runtime = load_json(OUT / "manifest/runtime.json")
    assert sys.executable == runtime["executable"]
    for name, version in runtime["versions"].items():
        assert importlib.metadata.version(name) == version
    for name, record in runtime["modules"].items():
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == Path(record["path"]).resolve()
        assert digest(module.__file__) == record["sha256"]
    return {"utc": now(), "status": "PASS", "run_count": 9,
            "protocol_sha256": digest(OUT / "manifest/protocol.json"),
            "frozen_sha256": digest(OUT / "manifest/frozen_execution.json"),
            "files_verified": len(frozen["files"]),
            "sample_draws": 0, "circuit_evaluations": 0, "test_access": False}


if __name__ == "__main__":
    print(json.dumps(preflight(), sort_keys=True))
