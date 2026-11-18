#!/usr/bin/env python3
"""Verify the publication snapshot using saved records and replay diagnostics.

No circuit, measurement, optimizer, or raw test array is evaluated here.
The report is written to a new output directory, never into the archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PAPER_TEACHERS = {1: (1, 2, 4), 2: (2, 3, 4), 3: (2, 3, 4),
                  4: (0, 1, 4), 5: (0, 1, 2)}
CLAIMS = {
    "resource_available_comparisons": 56,
    "resource_boosting_favoured": 39,
    "resource_cells_two_or_three_teachers": 36,
    "resource_cells_one_teacher": 20,
    "test_classification_cells": 22,
    "test_margin_cells": 14,
    "test_classification_median_gain_pp_rounded_2dp": 2.15,
    "test_margin_median_gain_pp_rounded_2dp": 3.84,
    "real_margin_hits": 110,
    "adar_margin_hits": 111,
    "real_theory_figure_histories": 57,
    "real_theory_figure_observed_median_rounds": 3,
    "real_theory_figure_calculated_median_rounds": 120,
    "bound_replay_models": 221,
    "bound_largest_margin_to_exponential": 206,
    "bound_above_margin_majority_models": 221,
    "pec_trajectories": 72,
    "pec_margin_hits": 41,
    "pec_classification_hits": 57,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def close(a: str | float, b: str | float, rel: float = 1e-11,
          absolute: float = 1e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=absolute)


def snapshot_entries(root: Path):
    """Iterate publication files, excluding only the root Git control entry."""
    for child in root.iterdir():
        if child.name == ".git" and not child.is_symlink():
            continue
        yield child
        if child.is_dir() and not child.is_symlink():
            yield from child.rglob("*")


def verify(root: Path) -> dict:
    errors: list[str] = []
    root = root.resolve(strict=True)
    receipt = json.loads((root / "PUBLICATION_RECEIPT.json").read_text())
    receipt_paths = [entry["public_path"] for entry in receipt["files"]]
    check(len(receipt_paths) == len(set(receipt_paths)),
          "Duplicate path in publication receipt", errors)
    entries = list(snapshot_entries(root))
    actual_paths = {path.relative_to(root).as_posix()
                    for path in entries if path.is_file()}
    unexpected = actual_paths - set(receipt_paths) - {"PUBLICATION_RECEIPT.json"}
    check(not unexpected,
          f"Files absent from publication receipt: {sorted(unexpected)}", errors)
    for entry in receipt["files"]:
        rel = Path(entry["public_path"])
        check(not rel.is_absolute() and ".." not in rel.parts,
              "Invalid receipt path", errors)
        path = root / rel
        check(path.is_file() and not path.is_symlink(),
              f"Missing or linked file: {rel}", errors)
        if path.is_file():
            check(digest(path) == entry["public_sha256"],
                  f"Hash mismatch: {rel}", errors)
    check(not any(path.is_symlink() for path in entries),
          "Snapshot contains symlinks", errors)

    tables = {p.stem: read_csv(p) for p in (root / "data").glob("*.csv")}
    figure_receipt = json.loads((root / "FIGURE_REBUILD_RECEIPT.json").read_text())
    check(len(figure_receipt["numeric_figures"]) == 7 and
          figure_receipt["circuit_evaluations"] == 0 and
          figure_receipt["measurements_drawn"] == 0,
          "Generated-figure receipt scope changed", errors)
    for name, expected_hash in figure_receipt["input_sha256"].items():
        check(digest(root / "data" / name) == expected_hash,
              f"Generated-figure input hash mismatch: {name}", errors)
    for name, expected_hash in figure_receipt["output_sha256"].items():
        check(digest(root / name) == expected_hash,
              f"Generated-figure output hash mismatch: {name}", errors)
    manuscript_tables = json.loads(
        (root / "tables/MANUSCRIPT_TABLE_PROVENANCE.json").read_text())
    check(len(manuscript_tables["tables"]) == 4,
          "Author-written manuscript table coverage changed", errors)
    for entry in manuscript_tables["tables"]:
        table_path = root / entry["file"]
        check(table_path.is_file() and digest(table_path) == entry["sha256"]
              and (r"\label{" + entry["label"] + "}") in table_path.read_text(),
              f"Author-written manuscript table mismatch: {entry['file']}", errors)
    check(all((root / "figures" / name).is_file() for name in
              ("pqc_architecture.pdf", "circuit.png", "class_balance.pdf")),
          "Static manuscript illustration missing", errors)
    teachers = tables["teachers"]
    targets = tables["targets"]
    check(len(teachers) == len(targets) == 15, "Expected 15 selected tasks", errors)
    target_by_task = {(int(r["depth"]), int(r["paper_teacher"])): r
                      for r in targets}
    check(len(target_by_task) == 15, "Duplicate task target", errors)
    for row in teachers:
        d, t = int(row["depth"]), int(row["paper_teacher"])
        check(int(row["source_teacher"]) == PAPER_TEACHERS[d][t],
              f"Teacher mapping mismatch d{d} t{t}", errors)
        for kind in ("teacher", "train", "validation", "test"):
            path = root / row[kind + "_file"]
            check(path.is_file() and digest(path) == row[kind + "_sha256"],
                  f"Task file mismatch d{d} t{t} {kind}", errors)
        check((int(row["train_n"]), int(row["validation_n"]),
               int(row["test_n"])) == (1536, 384, 1024),
              f"Task size mismatch d{d} t{t}", errors)
    for (d, t), row in target_by_task.items():
        minimum = math.ceil(float(row["classification_accuracy_target_stored"]) * 384)
        check(minimum == int(row["minimum_correct_executed"]),
              f"Integer cutoff mismatch d{d} t{t}", errors)
        check(0 < float(row["margin_threshold_original"]) < 1,
              f"Margin target out of range d{d} t{t}", errors)
    check(target_by_task[1, 0]["minimum_correct_executed"] == "380",
          "d1/t0 executed cutoff differs from 380", errors)
    check(target_by_task[2, 0]["minimum_correct_executed"] == "374",
          "d2/t0 executed cutoff differs from 374", errors)

    # Recompute both targets from their five constituent historical scalars.
    margin_rows = (tables["margin_calibration_d2_d3"] +
                   tables["margin_calibration_d1_d4_d5"])
    margin_by_task: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in margin_rows:
        margin_by_task[int(row["depth"]), int(row["teacher_task"])].append(row)
    class_by_task = {(int(r["depth"]), int(r["teacher_task"])): r
                     for r in tables["classification_calibration"]}
    check(len(margin_rows) == 125 and len(class_by_task) == 25,
          "Calibration scalar coverage changed", errors)
    for (d, t), target in target_by_task.items():
        source_teacher = PAPER_TEACHERS[d][t]
        margin = margin_by_task[d, source_teacher]
        check(len(margin) == 5 and
              {int(r["run_index"]) for r in margin} == set(range(5)),
              f"Margin calibration repetition coverage d{d} t{t}", errors)
        if len(margin) == 5:
            median_margin = statistics.median(
                float(r["train_ramp_loss_mu_0p1"]) for r in margin)
            check(median_margin == float(target["margin_threshold_original"]),
                  f"Unrounded margin target median mismatch d{d} t{t}", errors)
        class_row = class_by_task.get((d, source_teacher))
        check(class_row is not None,
              f"Classification calibration missing d{d} t{t}", errors)
        if class_row is not None:
            median_accuracy = statistics.median(
                float(class_row[f"run_{rep}_validation_accuracy"])
                for rep in range(5))
            check(median_accuracy == float(class_row["a_star_corrected"]) ==
                  float(target["classification_accuracy_target_stored"]),
                  f"Unrounded classification target median mismatch d{d} t{t}", errors)

    screening = json.loads((root / "data/accepted_teacher_registry.json").read_text())
    check(len(screening["depths"]) == 5 and all(
          len(screening["depths"][str(d)]["assignments"]) == 5
          for d in range(1, 6)),
          "Accepted teacher-bank coverage differs from 5 × 5", errors)
    check(len(tables["teacher_candidate_screening"]) == 300 and
          len(tables["teacher_screening_depth_summary"]) == 5,
          "Teacher candidate-screening coverage changed", errors)

    run_specs = [json.loads(line) for line in
                 (root / "run_specs/n25_runs.jsonl").read_text().splitlines() if line]
    check(len(run_specs) == 315 and len({r["run_id"] for r in run_specs}) == 315,
          "Expected 315 unique N=25 specifications", errors)
    for row in run_specs:
        d = int(row["depth"])
        t = PAPER_TEACHERS[d].index(int(row["teacher"]))
        target = target_by_task[d, t]
        check(close(row["epsilon_mar"], target["margin_threshold_original"]),
              f"Margin target mismatch: {row['run_id']}", errors)
        check(int(row["minimum_correct"]) == int(target["minimum_correct_executed"]),
              f"Classification target mismatch: {row['run_id']}", errors)
        for key, hash_key in (("teacher_path", "teacher_sha256"),
                              ("train_split", "train_split_sha256"),
                              ("validation_split", "validation_split_sha256"),
                              ("initial_path", "initial_file_sha256"),
                              ("orders_path", "orders_sha256"),
                              ("margin_target_registry", "margin_target_registry_sha256"),
                              ("classification_target_registry", "classification_target_registry_sha256")):
            check(digest(root / row[key]) == row[hash_key],
                  f"Run input hash mismatch: {row['run_id']} {key}", errors)
        check(int(row["N_fit"]) == 25 and int(row["N_monitor"]) == 25
              and int(row["N_confirm"]) == 25,
              f"Non-25-shot main run: {row['run_id']}", errors)
        if row["method"] != "reference":
            check(int(row["N_edge"]) == 25,
                  f"Non-25-shot edge: {row['run_id']}", errors)
    check(Counter(r["method"] for r in run_specs) ==
          {"reference": 45, "adaboost_r": 135, "real_adaboost": 135},
          "N=25 run grid differs from 45 + 135 + 135", errors)

    endpoints = tables["resource_endpoints"]
    check(len(endpoints) == 630, "Expected 630 endpoint records", errors)
    check(len({(r["run_id"], r["endpoint"]) for r in endpoints}) == 630,
          "Duplicate endpoint records", errors)
    known_ids = {r["run_id"] for r in run_specs}
    check({r["run_id"] for r in endpoints} == known_ids,
          "Endpoint/run-spec identities differ", errors)
    claim_values: dict[str, int | float | dict] = {}
    for method, key in (("reference", "reference_margin_hits"),
                        ("adaboost_r", "adar_margin_hits"),
                        ("real_adaboost", "real_margin_hits")):
        subset = [r for r in endpoints if r["method"] == method
                  and r["endpoint"] == "margin"]
        claim_values[key] = sum(r["confirmed_hit"] == "1" for r in subset)
    claim_values["reference_classification_hits"] = sum(
        r["confirmed_hit"] == "1" for r in endpoints
        if r["method"] == "reference" and r["endpoint"] == "classification")

    # Recompute the two-of-three gate and successful-run median directly from
    # the selected endpoint rows, rather than trusting the pre-aggregated table.
    teacher_gate = tables["resource_teacher_gate"]
    for gate in teacher_gate:
        method = gate["method"]
        matching = [r for r in endpoints
                    if r["endpoint"] == gate["endpoint"] and
                    r["method"] == method and
                    r["depth"] == gate["depth"] and
                    r["teacher"] == gate["teacher"] and
                    (method == "reference" or
                     close(r["noise_strength_1q"], gate["noise"]))]
        check(len(matching) == 3 and
              {int(r["repetition"]) for r in matching} == set(range(3)),
              f"Teacher repetition coverage mismatch {gate['endpoint']} "
              f"{gate['depth']} {gate['teacher']} {method} {gate['noise']}", errors)
        hits = [r for r in matching if r["confirmed_hit"] == "1"]
        check(len(hits) == int(gate["hits"]) and
              (len(hits) >= 2) == (gate["teacher_gate_pass"] == "1"),
              f"Teacher success gate mismatch {gate['endpoint']} "
              f"{gate['depth']} {gate['teacher']} {method} {gate['noise']}", errors)
        if hits and gate["teacher_gate_pass"] == "1":
            computed = statistics.median(
                int(r["first_hit_algorithm_raw_shots"]) for r in hits)
            check(close(computed,
                        gate["teacher_cost_median_confirmed_first_hit_shots"]),
                  f"Teacher successful-cost median mismatch {gate['endpoint']} "
                  f"{gate['depth']} {gate['teacher']} {method} {gate['noise']}", errors)
        else:
            check(not gate["teacher_cost_median_confirmed_first_hit_shots"],
                  f"Cost reported without teacher gate {gate['endpoint']} "
                  f"{gate['depth']} {gate['teacher']} {method} {gate['noise']}", errors)

    paired = tables["resource_teacher_ratios"]
    by_cell: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in paired:
        key = (row["endpoint"], row["noise"], row["depth"], row["boosting_method"])
        value = float(row["projected_pec_to_boosting_ratio"])
        check(close(value, float(row["projected_pec_shots"]) /
                    float(row["boosting_first_hit_shots"])),
              "Teacher PEC/boosting ratio mismatch", errors)
        by_cell[key].append(value)
    all_compared = []
    for row in tables["resource_cell_ratios"]:
        key = (row["endpoint"], row["noise"], row["depth"], row["boosting_method"])
        values = by_cell.get(key, [])
        check(len(values) == int(row["jointly_qualifying_teachers"]),
              f"Teacher count mismatch {key}", errors)
        if len(values) >= 2:
            check(close(statistics.median(values),
                        row["projected_pec_to_boosting_teacher_ratio_median"]),
                  f"Cell ratio mismatch {key}", errors)
        if values:
            all_compared.append((row, statistics.median(values)))
    claim_values["resource_available_comparisons"] = len(all_compared)
    claim_values["resource_boosting_favoured"] = sum(
        ratio > 1 for _, ratio in all_compared)
    claim_values["resource_cells_two_or_three_teachers"] = sum(
        int(row["jointly_qualifying_teachers"]) >= 2 for row, _ in all_compared)
    claim_values["resource_cells_one_teacher"] = sum(
        int(row["jointly_qualifying_teachers"]) == 1 for row, _ in all_compared)

    for endpoint, key in (("classification", "test_classification"),
                          ("margin", "test_margin")):
        cells = [r for r in tables["test_cell_summary"]
                 if r["endpoint"] == endpoint and
                 r["resource_cell_reportable"] == "1" and
                 r["full_cell_coverage"] == "1"]
        values = [float(r["sampled25_cell_median_delta_pp"]) for r in cells]
        claim_values[key + "_cells"] = len(cells)
        claim_values[key + "_median_gain_pp_rounded_2dp"] = round(
            statistics.median(values), 2)
        claim_values[key + "_positive_cells"] = sum(v > 0 for v in values)
        check(len(values) == claim_values[key + "_positive_cells"],
              f"Nonpositive {endpoint} paired test cell", errors)

    # Trace test-scalar medians back to each selected first-hit run. These
    # stored accuracies were measured historically; no test array is parsed
    # or used in a numerical calculation here.
    by_test: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    endpoint_index = {(r["run_id"], r["endpoint"]): r for r in endpoints}
    task_index = {(r["depth"], r["paper_teacher"]): r for r in teachers}
    p005_names = {"Reference25": "reference", "Ada25": "adaboost_r",
                  "Real25": "real_adaboost"}
    for row in tables["test_run_scalars_p005"]:
        if row["target_hit"] != "1":
            continue
        method = p005_names[row["method"]]
        depth = int(row["depth"])
        paper_teacher = PAPER_TEACHERS[depth].index(int(row["teacher"]))
        check(endpoint_index[row["run_id"], row["endpoint"]]["confirmed_hit"] == "1",
              f"p=.005 test scalar without confirmed hit: {row['run_id']}", errors)
        check(row["test_split_sha256"] ==
              task_index[row["depth"], str(paper_teacher)]["test_sha256"],
              f"p=.005 test split identity mismatch: {row['run_id']}", errors)
        by_test["0.005", row["depth"], str(paper_teacher), row["endpoint"], method].append(row)
    for row in tables["test_run_scalars_p01_p02"]:
        check(endpoint_index[row["run_id"], row["endpoint"]]["confirmed_hit"] == "1",
              f"p=.01/.02 test scalar without confirmed hit: {row['run_id']}", errors)
        check(row["test_split_sha256"] ==
              task_index[row["depth"], row["paper_teacher"]]["test_sha256"],
              f"p=.01/.02 test split identity mismatch: {row['run_id']}", errors)
        by_test[row["comparison_p"], row["depth"], row["paper_teacher"],
                row["endpoint"], row["method"]].append(row)
    check(len(tables["test_run_scalars_p005"]) == 270 and
          len(tables["test_run_scalars_p01_p02"]) == 411,
          "Per-run test-scalar coverage changed", errors)
    for row in tables["test_teacher_differences"]:
        key = (row["comparison_p"], row["depth"], row["paper_teacher"],
               row["endpoint"])
        reference = by_test[key + ("reference",)]
        boost = by_test[key + (row["method"],)]
        check({r["run_id"] for r in reference} == set(json.loads(row["reference_run_ids"]))
              and {r["run_id"] for r in boost} == set(json.loads(row["boost_run_ids"])),
              f"Test run-ID pairing mismatch: {key} {row['method']}", errors)
        check(len(reference) == int(row["reference_hits"]) and
              len(boost) == int(row["boost_hits"]),
              f"Test first-hit coverage mismatch: {key} {row['method']}", errors)
        if not reference or not boost:
            continue
        ref_median = statistics.median(float(r["test_accuracy_sampled25_mean"])
                                       for r in reference)
        boost_median = statistics.median(float(r["test_accuracy_sampled25_mean"])
                                         for r in boost)
        delta = 100 * (boost_median - ref_median)
        check(close(ref_median, row["reference_median_sampled25_mean"],
                    absolute=1e-12) and
              close(boost_median, row["boost_median_sampled25_mean"],
                    absolute=1e-12) and
              close(delta, row["delta_sampled25_pp"], absolute=1e-12),
              f"Test teacher-paired median mismatch: {key} {row['method']}", errors)

    stages = tables["bound_stages"]
    claim_values["bound_replay_models"] = len(stages)
    dominant_steps: Counter[str] = Counter()
    for row in stages:
        check(row["validation_status"] == "validated_replay",
              "Unvalidated bound replay", errors)
        check(all(row[k] == "0" for k in (
            "inequality_ramp_to_margin_violation",
            "inequality_margin_to_exp_violation",
            "inequality_exp_to_method_violation")),
            "Bound inequality violation", errors)
        vals = [float(row[k]) for k in
                ("R_ramp", "R_margin", "B_exp", "B_method")]
        epsilon = float(row["epsilon_mar"])
        check(epsilon > 0 and all(math.isfinite(v) for v in vals),
              f"Invalid bound values: {row['run_id']}", errors)
        check(all(a <= b + 1e-7 for a, b in zip(vals, vals[1:])),
              "Bound stage ordering mismatch", errors)
        if epsilon > 0:
            stage_fields = ("R_ramp_over_epsilon", "R_margin_over_epsilon",
                            "B_exp_over_epsilon", "B_method_over_epsilon")
            for value, field in zip(vals, stage_fields):
                check(close(value / epsilon, row[field], absolute=1e-9),
                      f"Bound normalized stage mismatch: {row['run_id']} {field}",
                      errors)
            increments = [(b - a) / epsilon for a, b in zip(vals, vals[1:])]
            increment_fields = ("margin_minus_ramp_over_epsilon",
                                "exp_minus_margin_over_epsilon",
                                "method_minus_exp_over_epsilon")
            for value, field in zip(increments, increment_fields):
                check(close(value, row[field], absolute=1e-9),
                      f"Bound adjacent increase mismatch: {row['run_id']} {field}",
                      errors)
            labels = ("ramp_to_margin", "margin_to_exponential",
                      "exponential_to_method")
            dominant = labels[max(range(3), key=lambda i: increments[i])]
            dominant_steps[dominant] += 1
            check(dominant == row["largest_increase_step"],
                  f"Bound dominant-step mismatch: {row['run_id']}", errors)
    claim_values["bound_largest_margin_to_exponential"] = dominant_steps[
        "margin_to_exponential"]
    margin_index = tables["reconstructed_training_margin_index"]
    decomposition = tables["bound_exponential_decomposition"]
    stage_by_run = {row["run_id"]: row for row in stages}
    margin_by_run = {row["run_id"]: row for row in margin_index}
    decomposition_by_run = {row["run_id"]: row for row in decomposition}
    check(len(stage_by_run) == len(margin_by_run) ==
          len(decomposition_by_run) == 221 and
          set(stage_by_run) == set(margin_by_run) == set(decomposition_by_run),
          "Reconstructed-margin run coverage mismatch", errors)
    above_majority = 0
    with zipfile.ZipFile(root / "records/reconstructed_training_margins.zip") as stream:
        check(set(stream.namelist()) ==
              {row["archive_name"] for row in margin_index},
              "Reconstructed-margin archive inventory mismatch", errors)
        for run_id, entry in margin_by_run.items():
            stage = stage_by_run.get(run_id)
            row = decomposition_by_run.get(run_id)
            if stage is None or row is None:
                continue
            check(all(entry[key] == stage[key] == row[key] for key in
                      ("method", "depth", "noise", "paper_teacher", "repetition")),
                  f"Reconstructed-margin identity mismatch: {run_id}", errors)
            check(entry["archive_name"] == f"{run_id}/u_float64.npy" and
                  entry["dtype"] == "float64" and
                  entry["origin"] == "saved_model_replay_diagnostic" and
                  entry["u_float64_sha256"] == row["u_float64_sha256"] ==
                  stage["reconstructed_u_float64_sha256"],
                  f"Reconstructed-margin provenance mismatch: {run_id}", errors)
            payload = stream.read(entry["archive_name"])
            check(hashlib.sha256(payload).hexdigest() ==
                  entry["u_float64_sha256"],
                  f"Reconstructed-margin hash mismatch: {run_id}", errors)
            vector = np.load(io.BytesIO(payload), allow_pickle=False)
            count = int(stage["n_train"])
            check(vector.dtype == np.dtype("float64") and vector.shape == (count,)
                  and np.all(np.isfinite(vector)) and
                  int(entry["n_train"]) == int(row["n_train"]) == count,
                  f"Reconstructed-margin array mismatch: {run_id}", errors)
            if vector.shape != (count,) or not np.all(np.isfinite(vector)):
                continue
            mu = float(stage["mu"])
            coefficient_sum = float(stage["A_executed_float32"])
            check(close(mu, row["mu"], absolute=1e-12) and
                  close(coefficient_sum, row["A_executed_float32"],
                        absolute=1e-12),
                  f"Reconstructed-margin constants mismatch: {run_id}", errors)
            above = vector > mu
            exponent = coefficient_sum * (mu - vector)
            maximum = float(np.max(exponent))
            log_b_exp = maximum + math.log(float(np.mean(
                np.exp(exponent - maximum), dtype=np.float64)))
            b_exp = math.exp(log_b_exp)
            r_margin = int(np.count_nonzero(~above)) / count
            above_contribution = float(np.sum(np.exp(exponent[above]),
                                              dtype=np.float64)) / count
            below_contribution = float(np.sum(np.expm1(exponent[~above]),
                                              dtype=np.float64)) / count
            increase = above_contribution + below_contribution
            share = above_contribution / increase
            computed = {
                "above_margin_n": int(np.count_nonzero(above)),
                "at_or_below_margin_n": int(np.count_nonzero(~above)),
                "above_margin_contribution": above_contribution,
                "at_or_below_margin_contribution": below_contribution,
                "total_increase": increase,
                "above_margin_share": share,
                "R_margin_recomputed": r_margin,
                "B_exp_recomputed": b_exp,
                "log_B_exp_recomputed": log_b_exp,
            }
            for field, value in computed.items():
                equal = (int(row[field]) == value if field.endswith("_n")
                         else math.isclose(float(row[field]), value,
                                           rel_tol=1e-10, abs_tol=1e-10))
                check(equal, f"Exponential decomposition mismatch: {run_id} {field}",
                      errors)
            check(math.isclose(r_margin, float(stage["R_margin"]),
                               rel_tol=1e-10, abs_tol=1e-10) and
                  math.isclose(b_exp, float(stage["B_exp"]),
                               rel_tol=1e-10, abs_tol=1e-10) and
                  math.isclose(log_b_exp, float(stage["log_B_exp"]),
                               rel_tol=1e-10, abs_tol=1e-10) and
                  math.isclose(increase, b_exp - r_margin,
                               rel_tol=1e-10, abs_tol=1e-10),
                  f"Reconstructed-margin stage mismatch: {run_id}", errors)
            above_majority += share > 0.5
    claim_values["bound_above_margin_majority_models"] = above_majority
    round_by_run = {r["run_id"]: r for r in tables["round_bound_endpoints"]}
    members = [round_by_run[r["run_id"]] for r in stages
               if r["current_round_count_figure_57_member"] == "1"]
    claim_values["real_theory_figure_histories"] = len(members)
    claim_values["real_theory_figure_observed_median_rounds"] = statistics.median(
        int(r["first_margin_hit_round"]) for r in members)
    claim_values["real_theory_figure_calculated_median_rounds"] = statistics.median(
        int(r["real_T_plugin"]) for r in members)
    check(all(r["method"] == "real_adaboost" and
              r["real_plugin_status"] == "valid_retrospective" for r in members),
          "Round figure inclusion mismatch", errors)

    pec = tables["pec_runs"]
    pec_specs = [json.loads(line) for line in
                 (root / "run_specs/explicit_pec_runs.jsonl").read_text().splitlines() if line]
    check(len(pec_specs) == 72, "Expected 72 portable PEC specifications", errors)
    pec_by_slot = {(r["configuration"], str(r["paper_teacher"]),
                    str(r["repetition"])): r for r in pec_specs}
    check(len(pec_by_slot) == 72, "Duplicate portable PEC slot", errors)
    claim_values["pec_n25_input_order_mismatch_slots"] = sum(
        not r["paired_n25_initial_and_order"] for r in pec_specs)
    for row in pec_specs:
        for key, hash_key in (("teacher_path", "teacher_sha256"),
                              ("train_split", "train_split_sha256"),
                              ("validation_split", "validation_split_sha256"),
                              ("initial_path", "initial_file_sha256"),
                              ("orders_path", "orders_sha256"),
                              ("margin_target_registry", "margin_target_registry_sha256")):
            check(digest(root / row[key]) == row[hash_key],
                  f"PEC input hash mismatch: {row['configuration']} {key}", errors)
        if "classification_target_registry" in row:
            check(digest(root / row["classification_target_registry"]) ==
                  row["classification_target_registry_sha256"],
                  "PEC classification registry hash mismatch", errors)
        check(int(row["N_fit"]) == int(row["N_monitor"]) ==
              int(row["N_confirm"]) == int(row["configuration"].split("_n")[-1]),
              "PEC shot allocation mismatch", errors)
    claim_values["pec_trajectories"] = len(pec)
    claim_values["pec_margin_hits"] = sum(
        r["margin_confirmed_hit"] == "1" for r in pec)
    claim_values["pec_classification_hits"] = sum(
        r["classification_confirmed_hit"] == "1" for r in pec)
    for config in {r["configuration"] for r in pec}:
        subset = [r for r in pec if r["configuration"] == config]
        check(len(subset) == 9 and
              {(r["paper_teacher"], r["repetition"]) for r in subset} ==
              {(str(t), str(rep)) for t in range(3) for rep in range(3)},
              f"Incomplete PEC configuration {config}", errors)
        check(all(r["status"] == "completed" for r in subset),
              f"Nonterminal PEC trajectory {config}", errors)
    for row in pec:
        slot = (row["configuration"], row["paper_teacher"], row["repetition"])
        spec = pec_by_slot.get(slot)
        check(spec is not None and
              spec["selected_result_sha256"] == row["result_sha256"],
              f"PEC selected-result lineage mismatch: {slot}", errors)
    all_methods = tables["pec_all_methods"]
    check(len(all_methods) == 720, "Expected 720 PEC figure scalar rows", errors)
    index = {(r["configuration"], r["target"], r["method"],
              r["paper_teacher"], r["repetition"]): r for r in all_methods}
    check(len(index) == len(all_methods), "Duplicate PEC figure rows", errors)
    for row in all_methods:
        if row["method"] != "projected_PEC":
            continue
        identity = (row["configuration"], row["target"], "reference_N25",
                    row["paper_teacher"], row["repetition"])
        reference = index[identity]
        if reference["confirmed_hit"] == "1":
            check(close(row["first_hit_algorithm_shots"],
                        float(reference["first_hit_algorithm_shots"]) *
                        float(row["gamma_squared"])),
                  "Projected PEC cost mismatch", errors)
        else:
            check(not row["first_hit_algorithm_shots"],
                  "Projected PEC cost on reference nonhit", errors)
        check(not row["confirmed_hit"], "Projected PEC has an observed hit", errors)

    # The compact tables are accompanied by the selected run histories. The
    # archive keeps all scalar JSON/JSONL events, including unsuccessful fits,
    # with private absolute paths redacted. Their original hashes are indexed.
    log_index = tables["selected_run_record_index"]
    by_log_archive: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in log_index:
        by_log_archive[row["cohort"], row["identity"]].append(row)
    cohorts = Counter(cohort for cohort, _ in by_log_archive)
    check(cohorts == {"n25": 315, "pec": 72, "superseded": 46,
                      "pec_precontinuation": 5},
          "Selected and inherited run-history coverage changed", errors)
    n25_result_hashes = {r["run_id"]: r["terminal_result_sha256"] for r in
                         endpoints if r["endpoint"] == "margin"}
    pec_result_hashes = {
        f"{r['configuration']}_t{r['paper_teacher']}_r{r['repetition']}":
        r["result_sha256"] for r in pec}
    for (cohort, identity), records in by_log_archive.items():
        archive = root / "records" / cohort / (identity + ".zip")
        with zipfile.ZipFile(archive) as stream:
            names = set(stream.namelist())
            check(names == {r["filename"] for r in records},
                  f"Selected history inventory mismatch: {identity}", errors)
            for row in records:
                payload = stream.read(row["filename"])
                check(hashlib.sha256(payload).hexdigest() == row["sanitized_sha256"],
                      f"Selected history entry hash mismatch: {identity}", errors)
                check(not any(marker in payload for marker in
                              (b"/zfsstore/", b"/home/", b"file://")),
                      f"Private path in selected history: {identity}", errors)
            result = [r for r in records if r["filename"] == "result.json"]
            if cohort in {"n25", "pec"}:
                expected = (n25_result_hashes if cohort == "n25"
                            else pec_result_hashes).get(identity)
                check(len(result) == 1 and result[0]["original_sha256"] == expected,
                      f"Selected history/result provenance mismatch: {identity}", errors)

    proof_index = tables["first_hit_proof_index"]
    check(len(proof_index) == 535 + 98,
          "Confirmed first-hit proof coverage changed", errors)
    with zipfile.ZipFile(root / "records/first_hit_proofs.zip") as stream:
        check(set(stream.namelist()) == {r["archive_name"] for r in proof_index},
              "First-hit proof inventory mismatch", errors)
        for row in proof_index:
            check(hashlib.sha256(stream.read(row["archive_name"])).hexdigest() ==
                  row["sanitized_sha256"],
                  f"First-hit proof payload hash mismatch: {row['archive_name']}", errors)
            if row["cohort"] == "n25":
                endpoint = endpoint_index[row["identity"], row["endpoint"]]
                expected = endpoint["first_hit_sha256"]
            else:
                spec = next((r for r in pec if
                             f"{r['configuration']}_t{r['paper_teacher']}_r{r['repetition']}"
                             == row["identity"]), None)
                expected = (spec[row["endpoint"] + "_first_hit_proof_sha256"]
                            if spec else None)
            check(row["original_sha256"] == expected,
                  f"First-hit proof/source lineage mismatch: {row['archive_name']}", errors)

    # Replayed vectors are diagnostic measurement draws, separate from the
    # historical algorithm-shot totals. They support the 221-model stage audit.
    vector_index = tables["bound_replay_vector_index"]
    check(len(vector_index) == 1748, "Expected 1,748 bound-replay vectors", errors)
    vector_counts = Counter(r["run_id"] for r in vector_index)
    check(all(vector_counts[r["run_id"]] == int(r["vectors_expected"])
              for r in stages), "Bound replay vector coverage mismatch", errors)
    with zipfile.ZipFile(root / "records" / "bound_replay_vectors.zip") as stream:
        check(set(stream.namelist()) == {r["archive_name"] for r in vector_index},
              "Bound replay vector inventory mismatch", errors)
        for row in vector_index:
            check(hashlib.sha256(stream.read(row["archive_name"])).hexdigest() ==
                  row["vector_sha256"],
                  f"Bound replay vector hash mismatch: {row['archive_name']}", errors)

    checkpoint_index = tables["first_hit_models"]
    with zipfile.ZipFile(root / "models" / "first_hit_checkpoints.zip") as stream:
        check({r["archive_name"] for r in checkpoint_index}.issubset(
                  set(stream.namelist())), "First-hit checkpoint missing", errors)
        for row in checkpoint_index:
            check(hashlib.sha256(stream.read(row["archive_name"])).hexdigest() ==
                  row["checkpoint_sha256"],
                  f"First-hit checkpoint hash mismatch: {row['archive_name']}", errors)

    pec_models = tables["pec_first_hit_models"]
    check(len(pec_models) == 98, "Explicit PEC first-hit model coverage changed", errors)
    with zipfile.ZipFile(root / "models" / "pec_first_hit_checkpoints.zip") as stream:
        check(len(stream.namelist()) == 96 and
              set(stream.namelist()) == {r["archive_name"] for r in pec_models},
              "Explicit PEC first-hit checkpoint inventory mismatch", errors)
        for row in pec_models:
            check(hashlib.sha256(stream.read(row["archive_name"])).hexdigest() ==
                  row["checkpoint_sha256"],
                  f"PEC checkpoint hash mismatch: {row['archive_name']}", errors)
            slot = (row["configuration"], row["paper_teacher"], row["repetition"])
            scalar = next((r for r in pec if
                           (r["configuration"], r["paper_teacher"], r["repetition"]) == slot), None)
            check(scalar is not None and
                  scalar[row["endpoint"] + "_confirmed_hit"] == "1" and
                  scalar[row["endpoint"] + "_first_hit_proof_sha256"] ==
                  row["first_hit_proof_sha256"] and
                  scalar["result_sha256"] == row["selected_result_sha256"],
                  f"PEC checkpoint lineage mismatch: {slot}", errors)

    claim_disagreements = {k: {"manuscript": expected, "records": claim_values.get(k)}
                           for k, expected in CLAIMS.items()
                           if claim_values.get(k) != expected}
    return {"scope": "archived records and saved replay diagnostics; no training or measurements",
            "files_verified": len(receipt["files"]), "task_count": len(teachers),
            "run_specs": len(run_specs), "endpoint_rows": len(endpoints),
            "claims_from_records": claim_values,
            "manuscript_claim_disagreements": claim_disagreements,
            "structural_errors": errors,
            "passed": not errors and not claim_disagreements}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"],
                      "claim_disagreements": len(result["manuscript_claim_disagreements"]),
                      "structural_errors": len(result["structural_errors"])}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
