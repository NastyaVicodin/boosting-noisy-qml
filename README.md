# Beyond Error Mitigation: Using Boosting for Noisy Quantum Machine Learning

Quantum error mitigation estimates what an ideal circuit would have produced, but reconstructing that output can require many measurements. This study asks whether binary classification can instead use information that survives in noisy quantum circuits. It combines noisy parameterized quantum classifiers with AdaBoostR and Real AdaBoost, then compares the circuit executions and predictive results with probabilistic error cancellation (PEC) at matched learning targets. This repository contains the code, fixed tasks, targets, and saved results behind that comparison.

The main comparison uses four-qubit circuits at depths 1–5, three selected teacher tasks per depth, and physical noise strengths 0.005, 0.01, and 0.02. The reference and two boosting methods use 25 **forward measurement samples per circuit–input evaluation**. Circuit derivatives and the boosting outer-patience statistic come from the simulator; these experiments are not a complete hardware-shot-cost comparison. The separate explicit PEC experiments use their recorded raw-shot allocations of 60, 100, 180, 210, or 240 as specified by each run, and their costs are counted directly.

## Usage and attribution

If you use, modify, or redistribute this code, retain its attribution. Publications or reports presenting results obtained using this code or a modified version must cite the accompanying paper with its complete author list:

> Anastasiia Lukianenko, Yash J. Patel, Nurbolat Kenbayev, and Vedran Dunjko. “Beyond Error Mitigation: Using Boosting for Noisy Quantum Machine Learning.”

The binding terms are in the custom [LICENSE](LICENSE). The author-specified wording is also in [USAGE_AND_ATTRIBUTION.md](USAGE_AND_ATTRIBUTION.md), and [CITATION.cff](CITATION.cff) provides machine-readable citation details. This license is not BSD-3-Clause or an OSI-approved license.

## Install and check the archive

The publication tooling was checked with Python 3.11.13, CPU JAX/JAXlib 0.4.35, and the versions in [requirements.txt](requirements.txt). The historical execution environment used Python 3.9; its complete binary image and host microarchitecture were not preserved. Use a fresh Linux environment with enough disk space for the snapshot and generated figures:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
python scripts/verify_archive.py --snapshot . --output ../publication_verification/archive_check.json
```

`verify_archive.py` checks the included file hashes, selected task and run identities, five-model target medians, endpoint records, reporting calculations, selected-model bound stages, and explicit PEC scalar coverage. It hashes the packaged test files for integrity but does not load them for analysis; it does not train, evaluate circuits, or draw measurements. Write the report to a new path outside this snapshot.

## Rebuild figures and tables without training

```bash
python scripts/rebuild_figures.py --snapshot . --output ../publication_verification/paper_figures
```

The script reads the saved result tables and writes new plots and tables under the requested output directory. It does not evaluate circuits or draw measurements. [PAPER_TO_CODE.md](PAPER_TO_CODE.md) identifies the data used for each result. The three illustrative figures are already included in `figures/`; this command rebuilds the numerical plots and tables. The numerical plots use the same method colors and page-sized layouts as the paper figures.

## Small smoke test and frozen-task reruns

```bash
python scripts/smoke_test.py --output ../publication_verification/smoke
```

The smoke test is a tiny synthetic training/analysis case. It checks the runnable pipeline, not the published numerical findings or bitwise replay of a historical trajectory. To stage one historical task with its saved teacher, splits, initialization, minibatch order, targets, shot policy, and unchanged scientific source, run:

```bash
python scripts/materialize_run.py --snapshot . --output-root ../publication_reruns \
  --run-id d1_t01_r00_p0p005_adaboost_r
```

For an explicit PEC example, replace `--run-id ...` by `--pec-slot d1_p0.020_n210 0 0`. Staging **does not launch training**; it prints the command for a deliberate manual run. Use a new output root outside this snapshot. The staged command is a fresh selected-task rerun, including for five historically continued PEC slots; it does not restore their inherited checkpoint state or historical spend. A new run draws new measurements and is not an archived result. The portable one-run manifest rebases its index to zero and bypasses campaign-specific launch preflight, so the original scheduler environment is not reproduced byte for byte.

The full selected-task grid contains 45 reference, 135 AdaBoostR, 135 Real AdaBoost, and 72 explicit PEC prescribed trajectories. Rerunning that grid can be expensive. Estimate compute time, storage, and allocation before starting it. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) distinguishes the supported workflows, saved seeds and RNG states, and what is still needed to regenerate the original teacher bank and calibration stages independently.

## Change code or train new models

To train one **unchanged** selected task, run `materialize_run.py` as shown above, inspect the printed run directory and `manual_training_command`, then run that command in the installed environment. It starts a new fit and draws new measurements; its result is separate from the saved paper result.

For a **new model or protocol**, create a new Git branch and work in a separate copy. `scripts/smoke_test.py` is a small training example; the PQC weak learner is in `src/boosting_qml/weak_learners/pqc_binary.py`, the boosting rules are in `src/boosting_qml/boosting/`, and explicit PEC sampling is in `src/boosting_qml/pec/explicit_gatewise.py`. Use a new run ID and output directory, save the changed code, settings, seeds, software versions, and shot counts, and report the new results separately. The archived run specifications and `materialize_run.py` describe the fixed paper protocols; they do not set up an arbitrary changed experiment. Keep the test split out of fitting and model selection. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the step-by-step workflow.

## Scope and provenance

`data/` contains teacher-screening and calibration results, per-run test summaries, selected-run outcomes (including nonhits), and reporting exclusions; `datasets/` contains the 15 selected synthetic tasks; `inputs/` holds initialization and minibatch-order arrays; `targets/` holds full-precision targets; `run_specs/` identifies all prescribed trajectories; and `experiments/` and `src/` contain the code needed for the selected runs and checks. `figures/` and `tables/` contain the rebuilt numerical plots, three illustrative figures, and LaTeX tables. The current tree presents one set of reported result figures. The older source snapshots in `experiments/pec/historical/` correspond to selected PEC runs, and `records/superseded/` and `records/pec_precontinuation/` document interrupted or continued attempts. They are retained to explain the final results, not counted as extra repetitions. `models/` holds first-hit checkpoints; `records/` also holds confirmed-hit proofs and separately regenerated bound-analysis predictions. Those regenerated predictions are not original measurement arrays. [PUBLICATION_RECEIPT.json](PUBLICATION_RECEIPT.json) lists the included files and their SHA-256 hashes. The preserved test splits are for independent evaluation only; they were not used for fitting, target construction, or model selection.

See [CREDITS.md](CREDITS.md) for contributor and dependency attribution.
