# Verification report for the publication snapshot

The development baseline was commit `f9e6105a3fbec54e7b2d7c217c375645b45f7670`. The original development worktree and its research artifacts were left intact; this snapshot was assembled from a separate cleanup worktree and an explicit frozen-record allowlist. The baseline's targeted scientific tests passed (18 tests). No historical source, target, outcome, accounting ledger, or manuscript file was edited.

## Environment and checks

The publication checks used Linux, Python 3.11.13, CPU JAX/JAXlib 0.4.35, and the pinned package versions in `requirements.txt`. The local CPU environment had JAX 64-bit mode disabled. Historical training used Python 3.9; its exact binary image, machine architecture, and scheduler environment were not recovered. These differences rule out a claim of bitwise historical rerun.

From the root of this snapshot, the archived-record checks are:

```bash
python scripts/verify_archive.py --snapshot . --output ../publication_verification/archive_check.json
python scripts/rebuild_figures.py --snapshot . --output ../publication_verification/paper_figures
```

The archive check validates receipt and source hashes, 15 selected task identities and split hashes, the 25-teacher accepted registry and screening coverage, five-model calibration medians, all 315 N=25 and 72 PEC run specifications, 630 N=25 endpoints, teacher-level two-of-three gates and successful-run cost medians, per-run test-scalar pairing, 221 selected-model bound-stage rows and reconstructed training-margin arrays, 1,748 separately regenerated replay vectors, selected/superseded scalar histories, all confirmed first-hit proofs and N=25/PEC checkpoint archives. Hashing a packaged test file checks its integrity; the verifier does not parse it into examples or use it for model selection or numerical analysis.

The figure builder reads saved scalar CSV tables only. It rebuilds seven numerical PDFs and PNG previews, the four-stage bound table, and the full-precision target table in a new directory. It does not open datasets, run a circuit, draw measurements, or train a model. The earlier Section 7.4 and PEC plot builders were checked during preparation; their underlying resource and hit scalar rows matched the rebuilt rows. The current source tree uses `scripts/rebuild_figures.py`; the original development history remains in the separate private development repository. PDF binary hashes need not match because rendering metadata can differ.

The focused publication tests are run with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_publication_protocol.py
python scripts/smoke_test.py --output ../publication_verification/smoke
```

The five focused tests check the archive scan, executed patience and epoch-mean convention, algorithm-shot accounting and first-hit immutability, and integer classification cutoffs. They passed. The smoke test passed on a new eight-example, one-epoch, four-qubit N=25 toy fit; it is a behaviour check, not a paper-data rerun. Static staging and frozen-file/import checks passed for representative reference, both boosting, and eight explicit-PEC source variants. Staging did not launch training. The one-run command is a fresh selected-task scaffold and does not recreate an inherited continuation or historical campaign preflight.

## Findings from archived records

The saved N=25 endpoints give 23/45 reference margin hits and 33/45 reference classification hits, 111/135 AdaBoostR margin hits, and 110/135 Real AdaBoost margin hits. Of 56 teacher-paired cost comparisons with an available ratio, 39 favor boosting; 36 have at least two qualifying teachers and 20 have one. All 36 reportable paired test-accuracy cell medians are positive: the classification subset has 22 cells and a +2.15 percentage-point median, and the margin subset has 14 cells and a +3.84 percentage-point median. These are saved test-scalar summaries, not new evaluations of held-out examples.

The selected-model bound replay covers 221 first-confirmed-margin-hit models; 206 have their largest adjacent bound increase between positive-margin error and the exponential expression. On the retained reconstructed training margins, examples above the threshold contribute a majority of that increase in all 221 models; the per-run contributions are preserved and checked separately. The separately defined Real AdaBoost theoretical-round figure has 57 retrospective histories, with medians of 3 observed and 120 calculated rounds. The explicit Sampled-PEC figure uses 72 completed trajectories across eight allocations, with 41 confirmed margin hits and 57 confirmed classification hits. Its recorded raw-shot costs are never multiplied by `γ²` again. Projected PEC has no observed hit count: its cost is a proxy derived from matched successful reference first-hit costs.

## Limits and manuscript item to check

Archived-record reproduction, a small toy smoke test, and a fresh selected-task rerun are distinct. The 25-teacher screening registry, scalar calibration results, selected arrays, and frozen targets are present, but an exact consumed low-level margin-calibration source revision and the full historical binary environment are unavailable. Full teacher-generation and calibration reproduction has therefore **not** been demonstrated. Saved-model bound vectors were regenerated in a separate diagnostic; scalar agreement corroborates them but does not bitwise certify historical measurement vectors. The printed selected-task rerun command begins a new measurement history. The two personal Fashion-MNIST cache fallbacks in an unrelated loader were removed only in the publication copy; original and exported hashes are both in the receipt, and paper-relevant numerical paths remain unchanged.

The current manuscript's teacher-selection paragraph says that three candidates passing the checks were kept at each depth and that this choice preceded the methods comparison runs. The archived screening records instead show a bank of **five** passing teachers per depth, followed by manual selection of three for the paper. The available records do not establish the selection's timing relative to inspection of the corresponding comparison outcomes. The manuscript should describe the five-teacher bank and manual three-teacher selection without claiming a prospectively fixed chronology.

The numerical figures have been restyled to match the manuscript's method colors and page-sized layouts. In the all-method PEC comparison, costs are on the horizontal axis: horizontal segments show the minimum–maximum teacher range and vertical ticks show the median. The saved costs, hit counts, and historical results are unchanged.

The cropped Fig. 1 architecture matches the executed ansatz's per-layer RX–RY–RZ gates, CZ ring, scaled data RX, final local block, and parity readout. Appendix Fig. 9 instead sketches the separate Pauli-noise complexity argument with a different gate layout; it should not be presented as the numerical circuit.

Preparing the publication tree removed 14 unused older orchestration and comparison files from `src/`, two older plot builders, and duplicate or superseded figure exports. The frozen source trees corresponding to selected paper runs, first-hit records, superseded attempts, and PEC continuation records remain. The removed files remain in the separate private development repository; they are not needed by the documented training, verification, or figure-building commands. The original development checkout was not changed.

Section 7.3 defines the plotted algorithm-shot cost as fitting plus boosting-edge acquisitions through the first confirmed target, but it should also state the exclusions explicitly: monitoring, confirmation, post-fit diagnostics, calibration, final evaluation, simulator-supplied gradients, and exact-mean outer-patience calculations are outside that plotted cost. The separate historical auxiliary acquisitions remain real expenditure. This is a manuscript clarification, not a reason to change the archived cost values.

No full experiment grid was rerun for this preparation. The snapshot includes a custom project license with a mandatory paper-citation condition. The repository remains private; approval to make it public remains pending.
