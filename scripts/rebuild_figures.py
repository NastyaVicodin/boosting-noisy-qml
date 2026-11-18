#!/usr/bin/env python3
"""Rebuild the manuscript's numerical figures from the publication scalar archive.

Only the CSV tables in ``data/`` are read. In particular, this program never
loads a dataset, model checkpoint, prediction vector, or held-out test array;
it performs no circuit evaluation, fitting, or measurement sampling. Outputs
are written to a new directory, leaving the archived evidence unchanged.

Usage::

    python scripts/rebuild_figures.py --snapshot . --output ../paper-figures

Static circuit and data-construction illustrations are outside this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


NOISES = (0.005, 0.01, 0.02)
TARGETS = ("margin", "classification")
BOOST = ("adaboost_r", "real_adaboost")
METHODS = ("reference", *BOOST)
STYLES = {
    "reference": ("Reference", "#68737D", "o"),
    "projected_pec": ("Projected PEC", "#262C30", "^"),
    "adaboost_r": ("AdaBoost $\\mathbb{R}$", "#1B9E77", "o"),
    "real_adaboost": ("Real AdaBoost", "#7B3294", "s"),
    "explicit_PEC": ("Sampled PEC", "#D95F02", "P"),
}
# Match the method colors and compact page width of manuscript Figs. 2--5.
# This visual layer is separate from the scalar calculations below.
PAPER_WIDTH_IN = 155.9 / 25.4
PAPER_COLORS = {
    "reference": "#68737D",
    "projected_pec": "#30363A",
    "adaboost_r": "#1B9E77",
    "real_adaboost": "#7B3294",
}
PAPER_LINESTYLES = {
    "reference": (0, (2, 2)),
    "projected_pec": (0, (5, 2)),
    "adaboost_r": "-",
    "real_adaboost": (0, (4, 2)),
}
PAPER_LABELS = {
    "reference": "Reference",
    "projected_pec": "Projected PEC",
    "adaboost_r": "AdaBoost $\\mathbb{R}$",
    "real_adaboost": "Real AdaBoost",
}
PEC_TO_METHOD = {
    "reference_N25": "reference",
    "projected_PEC": "projected_pec",
    "AdaBoostR_N25": "adaboost_r",
    "RealAdaBoost_N25": "real_adaboost",
    "explicit_PEC": "explicit_PEC",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def assert_unique(rows: list[dict], fields: tuple[str, ...], name: str) -> None:
    keys = [tuple(row[f] for f in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"Duplicate {name} identity: {fields}")


def index(rows: list[dict], fields: tuple[str, ...], name: str) -> dict:
    assert_unique(rows, fields, name)
    return {tuple(row[f] for f in fields): row for row in rows}


def fnum(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"Non-finite scalar {value}")
    return result


def same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=2e-11, abs_tol=1e-7)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def style_handle(method: str) -> Line2D:
    label, color, marker = STYLES[method]
    return Line2D([0], [0], color=color, marker=marker, lw=1.3,
                  markersize=5, label=label)


def plot_summary(ax, x: float, values: list[float], method: str,
                 *, units: float = 1.0, size: float = 35) -> None:
    """Show a median and teacher range; a cross identifies one teacher."""
    if not values:
        return
    vals = [v / units for v in values]
    _, color, marker = STYLES[method]
    if len(vals) == 1:
        ax.scatter([x], vals, color=color, marker="x", s=size + 9,
                   linewidths=1.5, zorder=5)
        return
    center = median(vals)
    lo, hi = min(vals), max(vals)
    ax.plot([x, x], [lo, hi], color=color, lw=1.15, zorder=3)
    ax.plot([x - 0.055, x + 0.055], [center, center], color=color,
            lw=2.2, zorder=4)
    ax.scatter([x], [center], marker=marker, color=color, s=size,
               edgecolor="white", linewidth=0.4, zorder=5)


def paper_summary(ax, x: float, values: list[float], method: str,
                  *, units: float = 1.0) -> None:
    """Manuscript mark: teacher range, median tick, or one-teacher cross."""
    if not values:
        return
    vals = [v / units for v in values]
    color = PAPER_COLORS[method]
    if len(vals) == 1:
        ax.plot(x, vals[0], marker="x", color=color, markersize=4.8,
                markeredgewidth=1.0, linestyle="None", zorder=5)
        return
    center = median(vals)
    ax.plot([x, x], [min(vals), max(vals)], color=color, lw=0.8, zorder=3)
    for bound in (min(vals), max(vals)):
        ax.plot([x - 0.035, x + 0.035], [bound, bound], color=color,
                lw=0.75, zorder=4)
    ax.plot([x - 0.065, x + 0.065], [center, center], color=color,
            lw=1.7, zorder=5)


def paper_legend_handle(method: str) -> Line2D:
    return Line2D([0], [0], color=PAPER_COLORS[method],
                  ls=PAPER_LINESTYLES[method], lw=1.3,
                  label=PAPER_LABELS[method])


def paper_axes(ax, *, logarithmic: bool = False) -> None:
    if logarithmic:
        ax.set_yscale("log")
        ax.minorticks_off()
    ax.grid(axis="y", which="major", color="#DEE2E5", lw=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7.6, width=0.7, length=2.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def load(snapshot: Path) -> dict[str, list[dict[str, str]]]:
    names = (
        "targets", "resource_teacher_gate", "resource_teacher_ratios",
        "resource_cell_ratios", "resource_success_counts", "test_cell_summary",
        "test_teacher_differences", "pec_all_methods", "pec_plot_summary",
        "bound_stages", "round_bound_endpoints", "resource_cell_gate",
    )
    data = {name: read_csv(snapshot / "data" / f"{name}.csv") for name in names}
    expected = {
        "targets": 15, "resource_teacher_gate": 270,
        "resource_teacher_ratios": 120, "resource_cell_ratios": 60,
        "resource_success_counts": 15, "test_cell_summary": 60,
        "test_teacher_differences": 120, "pec_all_methods": 720,
        "pec_plot_summary": 80, "bound_stages": 221,
        "round_bound_endpoints": 270, "resource_cell_gate": 90,
    }
    for name, count in expected.items():
        if len(data[name]) != count:
            raise AssertionError(f"{name}: {len(data[name])} rows, expected {count}")
    return data


def validate_resource(data: dict) -> tuple[dict, dict, dict]:
    teachers = index(data["resource_teacher_gate"],
        ("endpoint", "noise", "depth", "method", "paper_teacher"), "resource teacher")
    pairs = defaultdict(list)
    for row in data["resource_teacher_ratios"]:
        key = (row["endpoint"], row["noise"], row["depth"], row["boosting_method"])
        paper = row["paper_teacher"]
        ref = teachers[key[0], key[1], key[2], "reference", paper]
        boost = teachers[key[0], key[1], key[2], key[3], paper]
        if ref["teacher_gate_pass"] != "1" or boost["teacher_gate_pass"] != "1":
            raise AssertionError(f"Paired ratio includes unqualified teacher {key, paper}")
        gamma = fnum(row["gamma_squared"])
        actual = gamma * fnum(ref["teacher_cost_median_confirmed_first_hit_shots"]) / fnum(
            boost["teacher_cost_median_confirmed_first_hit_shots"])
        if not same(actual, fnum(row["projected_pec_to_boosting_ratio"])):
            raise AssertionError(f"Paired resource ratio mismatch {key, paper}")
        pairs[key].append(row)
    pair_index = index(data["resource_cell_ratios"],
        ("endpoint", "noise", "depth", "boosting_method"), "resource ratio cell")
    for key, row in pair_index.items():
        group = pairs.get(key, [])
        if len(group) != int(row["jointly_qualifying_teachers"]):
            raise AssertionError(f"Resource comparison coverage mismatch {key}")
        if len(group) >= 2:
            val = median(fnum(x["projected_pec_to_boosting_ratio"]) for x in group)
            if not same(val, fnum(row["projected_pec_to_boosting_teacher_ratio_median"])):
                raise AssertionError(f"Reported cell ratio mismatch {key}")
        elif row["projected_pec_to_boosting_teacher_ratio_median"]:
            raise AssertionError(f"Single-teacher cell reported as multi-teacher {key}")
    if len([g for g in pairs.values() if g]) != 56:
        raise AssertionError("Expected 56 available resource comparisons")
    return teachers, pairs, pair_index


def ratio_figure(out: Path, pairs: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(PAPER_WIDTH_IN, 5.95), sharex=True)
    fig.subplots_adjust(top=0.905, bottom=0.095, left=0.165, right=0.985,
                        hspace=0.25)
    for panel_idx, (ax, target) in enumerate(zip(axes, TARGETS)):
        for pidx, p in enumerate(NOISES):
            for method, offset in (("adaboost_r", -0.15), ("real_adaboost", 0.15)):
                xs, ys = [], []
                for depth in range(1, 6):
                    rows = pairs.get((target, str(p), str(depth), method), [])
                    vals = [fnum(r["projected_pec_to_boosting_ratio"]) for r in rows]
                    x = pidx * 5.6 + depth + offset
                    paper_summary(ax, x, vals, method)
                    xs.append(x)
                    ys.append(median(vals) if vals else float("nan"))
                ax.plot(xs, ys, color=PAPER_COLORS[method],
                        ls=PAPER_LINESTYLES[method], lw=1.1, zorder=2)
        paper_axes(ax, logarithmic=True)
        ax.axhline(1, color="#606970", lw=0.75, ls=(0, (3, 2)))
        limits = ax.get_ylim()
        ticks = [v for v in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
                 if limits[0] <= v <= limits[1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:g}" for v in ticks])
        for divider in (5.8, 11.4):
            ax.axvline(divider, color="#D8DEE2", lw=0.65)
        ax.set_ylabel("Projected PEC / boosting\nshot ratio", fontsize=8.5)
        ax.text(0.012, 0.97,
                "(a) Margin target" if panel_idx == 0 else "(b) Classification target",
                transform=ax.transAxes, va="top", fontsize=8.5)
        for pidx, p in enumerate(NOISES):
            ax.text(pidx * 5.6 + 3, 1.02, f"$p={p:g}$", ha="center",
                    transform=ax.get_xaxis_transform(), fontsize=8)
    axes[-1].set_xlim(0.55, 16.65)
    axes[-1].set_xticks([pidx * 5.6 + d for pidx in range(3) for d in range(1, 6)])
    axes[-1].set_xticklabels([str(d) for _ in NOISES for d in range(1, 6)])
    axes[-1].set_xlabel("Circuit depth $d$", fontsize=8.5)
    handles = [paper_legend_handle(m) for m in BOOST]
    handles.append(Line2D([0], [0], marker="x", ls="None", color="#444444",
                          markersize=5, label="One qualifying teacher"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.55, 0.99))
    save(fig, out / "figures/n25_resource_ratios.pdf")


def resource_cost_and_hits(out: Path, data: dict, teachers: dict, target: str) -> None:
    """Paper layout: three noise panels above their complete teacher hit grids."""
    success = index(data["resource_success_counts"], ("noise", "depth"), "success counts")
    cells = index(data["resource_cell_gate"],
        ("endpoint", "noise", "depth", "method"), "resource cell")
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 7.15))
    gs = fig.add_gridspec(2, 3, height_ratios=(0.43, 0.57),
                          left=0.145, right=0.985, top=0.88, bottom=0.045,
                          hspace=0.27, wspace=0.15)
    method_offsets = {"reference": -0.18, "projected_pec": -0.06,
                      "adaboost_r": 0.06, "real_adaboost": 0.18}
    shared_plot = None
    for col_idx, p in enumerate(NOISES):
        ax = fig.add_subplot(gs[0, col_idx], sharey=shared_plot)
        if shared_plot is None:
            shared_plot = ax
        tab = fig.add_subplot(gs[1, col_idx])
        paper_axes(ax, logarithmic=True)
        ax.set_title(f"$p={p:g}$", fontsize=8.5, pad=4)
        for method in ("reference", "projected_pec", *BOOST):
            line_x, line_y = [], []
            source = "reference" if method == "projected_pec" else method
            for depth in range(1, 6):
                group = [teachers[target, str(p), str(depth), source, str(t)]
                         for t in range(3)]
                vals = [fnum(row["teacher_cost_median_confirmed_first_hit_shots"])
                        for row in group if row["teacher_gate_pass"] == "1"]
                if method == "projected_pec":
                    gamma = fnum(cells[target, str(p), str(depth), "reference"]["gamma_squared"])
                    vals = [gamma * v for v in vals]
                x = depth + method_offsets[method]
                paper_summary(ax, x, vals, method, units=1e6)
                line_x.append(x)
                line_y.append(median(vals) / 1e6 if vals else float("nan"))
            ax.plot(line_x, line_y, color=PAPER_COLORS[method],
                    ls=PAPER_LINESTYLES[method], lw=1.0, zorder=2)
        ax.set_ylim(0.5, 2e4)
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.65, 5.35)
        ax.set_xlabel("Depth $d$", fontsize=8.2, labelpad=2)
        if col_idx == 0:
            ax.set_ylabel("Algorithm shots\n(millions)", fontsize=8.2)
        else:
            ax.tick_params(labelleft=False)
        tab.axis("off")
        headings = ["$d$", "$j$", "Ref.", "Ada$\\mathbb{R}$", "Real"]
        matrix = []
        for depth in range(1, 6):
            row = success[str(p), str(depth)]
            for teacher in range(3):
                matrix.append([str(depth) if teacher == 0 else "", str(teacher)] +
                    [f"{row[f'{method}_{target}_paper_teacher_{teacher}_hits']}/3"
                     for method in METHODS])
            matrix.append(["", "Total"] +
                [f"{sum(int(row[f'{method}_{target}_paper_teacher_{teacher}_hits']) for teacher in range(3))}/9"
                 for method in METHODS])
        table = tab.table(cellText=matrix, colLabels=headings, loc="center",
                          cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1],
                          colWidths=[0.105, 0.175, 0.24, 0.25, 0.23])
        table.auto_set_font_size(False)
        table.set_fontsize(6.25)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#E1E5E7")
            cell.set_linewidth(0.22)
            if r == 0:
                cell.set_facecolor("#E7EDF1")
                cell.set_text_props(weight="bold")
            elif (r % 4) == 0:
                cell.set_facecolor("#E3E9ED")
                cell.set_text_props(weight="bold")
            elif c in (0, 1):
                cell.set_facecolor("#F2F5F6")
            else:
                n = int(matrix[r - 1][c].split("/")[0])
                cell.set_facecolor("#DFEEE5" if n >= 2 else
                                   "#F7E9CE" if n == 1 else "#F8E1DF")
    fig.text(0.565, 0.475, "Confirmed hits", ha="center", fontsize=8.3)
    fig.legend(handles=[paper_legend_handle(m) for m in
                        ("reference", "projected_pec", *BOOST)],
               loc="upper center", bbox_to_anchor=(0.51, 0.975), ncol=2,
               frameon=False, fontsize=8.1)
    save(fig, out / f"figures/n25_{target}_costs_hits.pdf")


def test_figure(out: Path, data: dict, pairs: dict) -> None:
    cells = index(data["test_cell_summary"],
        ("endpoint", "comparison_p", "depth", "method"), "test cell")
    teachers = defaultdict(list)
    for row in data["test_teacher_differences"]:
        key = (row["endpoint"], row["comparison_p"], row["depth"], row["method"])
        teachers[key].append(row)
    if set(teachers) != {k for k, rows in pairs.items() if rows}:
        raise AssertionError("Test and resource comparison teacher cells differ")
    fig, axes = plt.subplots(2, 1, figsize=(PAPER_WIDTH_IN, 5.55), sharex=True)
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.16, right=0.985, hspace=0.26)
    for panel_idx, (ax, target) in enumerate(zip(axes, TARGETS)):
        for pidx, p in enumerate(NOISES):
            for method, offset in (("adaboost_r", -0.14), ("real_adaboost", 0.14)):
                for depth in range(1, 6):
                    key = (target, str(p), str(depth), method)
                    group = teachers.get(key, [])
                    matched = {r["paper_teacher"] for r in pairs.get(key, [])}
                    if {r["paper_teacher"] for r in group} != matched:
                        raise AssertionError(f"Test teacher subset mismatch {key}")
                    vals = [fnum(row["delta_sampled25_pp"]) for row in group]
                    x = pidx * 5.6 + depth + offset
                    if len(vals) == 1:
                        ax.scatter(x, vals[0], color=PAPER_COLORS[method], marker="x",
                                   s=25, linewidth=1.0, zorder=4)
                    elif vals:
                        for i, val in enumerate(vals):
                            ax.scatter(x + 0.035 * (i - 1), val,
                                       color=PAPER_COLORS[method],
                                       marker="o" if method == "adaboost_r" else "s",
                                       alpha=0.35, s=8, linewidths=0, zorder=3)
                        mid = median(vals)
                        ax.scatter(x, mid, color=PAPER_COLORS[method],
                                   marker="o" if method == "adaboost_r" else "s",
                                   edgecolor="white", linewidth=0.35, s=18, zorder=5)
                    reported = cells.get(key)
                    if reported is None:
                        raise AssertionError(f"Missing test cell {key}")
                    if len(vals) >= 2:
                        if not same(median(vals), fnum(reported["sampled25_cell_median_delta_pp"])):
                            raise AssertionError(f"Test median mismatch {key}")
                    elif reported["sampled25_cell_median_delta_pp"]:
                        raise AssertionError(f"Unreportable test cell has reported median {key}")
        paper_axes(ax)
        ax.axhline(0, color="#687178", lw=0.8, ls="--")
        for divider in (5.8, 11.4):
            ax.axvline(divider, color="#D8DEE2", lw=0.7)
        ax.set_ylabel("Boosting − reference\n(percentage points)", fontsize=8.3)
        ax.text(0.01, 0.97,
                "(a) Margin target" if panel_idx == 0 else "(b) Classification target",
                transform=ax.transAxes, va="top", fontsize=8.4)
        for pidx, p in enumerate(NOISES):
            ax.text(pidx * 5.6 + 3, 1.02, f"$p={p:g}$", ha="center",
                    transform=ax.get_xaxis_transform(), fontsize=8.0)
    axes[-1].set_xlim(0.55, 16.65)
    axes[-1].set_xticks([pidx * 5.6 + d for pidx in range(3) for d in range(1, 6)])
    axes[-1].set_xticklabels([str(d) for _ in NOISES for d in range(1, 6)])
    axes[-1].set_xlabel("Circuit depth $d$", fontsize=8.5)
    handles = [Line2D([0], [0], color=PAPER_COLORS[m], ls="None",
                      marker="o" if m == "adaboost_r" else "s",
                      markersize=4, label=PAPER_LABELS[m]) for m in BOOST]
    handles.append(Line2D([0], [0], marker="x", ls="None", color="#444444",
                          markersize=5, label="One qualifying teacher"))
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.54, 0.995), fontsize=8, frameon=False)
    save(fig, out / "figures/n25_full_grid_test_accuracy_differences.pdf")


def pec_teacher_costs(data: dict) -> tuple[dict, dict]:
    """Summarize all 3 repetitions before applying the same 2/3 teacher gate."""
    groups = defaultdict(list)
    for row in data["pec_all_methods"]:
        key = (row["configuration"], row["target"], row["method"], row["paper_teacher"])
        groups[key].append(row)
    if len(groups) != 8 * 2 * 5 * 3:
        raise AssertionError("Incomplete PEC teacher cells")
    summary = {}
    for key, rows in groups.items():
        if len(rows) != 3 or {r["repetition"] for r in rows} != {"0", "1", "2"}:
            raise AssertionError(f"Incomplete PEC repetitions {key}")
        if key[2] == "projected_PEC":
            assert all(r["row_kind"] == "derived_cost_proxy" for r in rows)
            continue
        if any(r["trajectory_status"] != "completed" or r["unresolved"] != "0"
               for r in rows):
            raise AssertionError(f"Unfinished observed PEC trajectory {key}")
        hit = [r for r in rows if r["confirmed_hit"] == "1"]
        miss = [r for r in rows if r["terminal_nonhit"] == "1"]
        if len(hit) + len(miss) != 3:
            raise AssertionError(f"Unresolved PEC endpoint {key}")
        costs = [fnum(r["first_hit_algorithm_shots"]) for r in hit]
        summary[key] = {"hits": len(hit), "nonhits": len(miss),
                        "qualifies": len(hit) >= 2,
                        "cost": median(costs) if len(hit) >= 2 else None,
                        "gamma": fnum(rows[0]["gamma_squared"]),
                        "depth": int(rows[0]["depth"]),
                        "noise": fnum(rows[0]["noise"]),
                        "shots": int(rows[0]["pec_shots"])}
    saved = index(data["pec_plot_summary"],
        ("configuration", "target", "method"), "PEC plot summary")
    for key, row in saved.items():
        if key[2] == "projected_PEC":
            continue
        source = [summary[key + (str(t),)] for t in range(3)]
        if sum(s["hits"] for s in source) != int(row["confirmed_hits"]):
            raise AssertionError(f"PEC hit summary mismatch {key}")
        if sum(s["nonhits"] for s in source) != int(row["terminal_nonhits"]):
            raise AssertionError(f"PEC nonhit summary mismatch {key}")
    configs = {}
    for key, row in summary.items():
        configs[key[0]] = (row["depth"], row["noise"], row["shots"], row["gamma"])
    if len(configs) != 8:
        raise AssertionError("Expected eight PEC configurations")
    return summary, configs


def pec_ratio_figure(out: Path, summary: dict, configs: dict) -> list[dict]:
    """Paper Fig. 6: eight configurations, two adjacent ratio panels and hit counts."""
    order = sorted(configs, key=lambda c: configs[c][:3])
    ratios = []
    orange = "#D95F02"
    fig = plt.figure(figsize=(6.14, 4.05))
    gs = fig.add_gridspec(1, 5, width_ratios=(1.50, 0.95, 0.60, 0.95, 0.60),
                          left=0.02, right=0.99, top=0.82, bottom=0.21,
                          wspace=0.05)
    labels = fig.add_subplot(gs[0, 0])
    ratio_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]
    hit_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])]
    labels.set_xlim(0, 1)
    labels.set_ylim(len(order) - 0.5, -0.5)
    labels.axis("off")
    labels.text(0.00, 1.04, "$d$   $p$", transform=labels.transAxes,
                fontsize=7.1, fontweight="bold")
    labels.text(0.32, 1.04, "PEC / $25\\gamma^2$ shots", transform=labels.transAxes,
                fontsize=7.1, fontweight="bold")
    for i, config in enumerate(order):
        depth, noise, shots, g = configs[config]
        projected = 25 * g
        labels.text(0.00, i, f"{depth}  {noise:.3f}", va="center", fontsize=6.7)
        labels.text(0.32, i, f"{shots} / {projected:.1f} ({100 * shots / projected:.1f}%)",
                    va="center", fontsize=6.2)
    for target_idx, target in enumerate(TARGETS):
        ax, hits_ax = ratio_axes[target_idx], hit_axes[target_idx]
        ax.set_xscale("log")
        ax.set_xlim(0.22, 4.35)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(FixedLocator((0.25, 0.5, 1, 2, 4)))
        ax.xaxis.set_major_formatter(FixedFormatter(("0.25", "0.5", "1", "2", "4")))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axvline(1, color="#B7791F", ls="--", lw=1.0, zorder=1)
        ax.grid(axis="x", color="#E1E5E7", lw=0.55)
        ax.set_title("(a) Margin target" if target == "margin" else
                     "(b) Classification target", loc="left", fontweight="bold",
                     fontsize=7.6, pad=20)
        ax.tick_params(axis="x", labelsize=6.4, pad=2)
        ax.set_xlabel("PEC / Projected PEC", fontsize=7.2)
        hits_ax.set_xlim(0, 1)
        hits_ax.set_ylim(len(order) - 0.5, -0.5)
        hits_ax.axis("off")
        hits_ax.text(0.5, 1.115, "Hits / 9", ha="center", fontsize=7.1,
                     transform=hits_ax.transAxes)
        hits_ax.text(0.25, 1.04, "PEC", color=orange, ha="center", fontsize=6.8,
                     transform=hits_ax.transAxes)
        hits_ax.text(0.75, 1.04, "Ref.", color="#69777D", ha="center", fontsize=6.8,
                     transform=hits_ax.transAxes)
        for i, config in enumerate(order):
            g = configs[config][3]
            values = []
            pec_hits = 0
            ref_hits = 0
            for teacher in range(3):
                j = str(teacher)
                p = summary[config, target, "explicit_PEC", j]
                r = summary[config, target, "reference_N25", j]
                pec_hits += p["hits"]
                ref_hits += r["hits"]
                if p["qualifies"] and r["qualifies"]:
                    values.append(p["cost"] / (g * r["cost"]))
            ratios.append({"configuration": config, "target": target,
                           "reference_hits": ref_hits, "pec_hits": pec_hits,
                           "jointly_qualifying_teachers": len(values),
                           "teacher_ratios": "|".join(repr(v) for v in values),
                           "median_ratio": repr(median(values)) if values else ""})
            if len(values) == 1:
                ax.scatter(values[0], i, marker="x", color=orange,
                           s=48, linewidths=1.6, zorder=5)
            elif values:
                lo, hi = min(values), max(values)
                ax.plot([lo, hi], [i, i], color=orange,
                        lw=1.65, zorder=3)
                for endpoint in (lo, hi):
                    ax.plot([endpoint, endpoint], [i - 0.075, i + 0.075],
                            color=orange, lw=0.95, zorder=4)
                ax.plot([median(values), median(values)], [i - 0.15, i + 0.15],
                        color=orange, lw=2.6, zorder=4)
            for x, count, fill in ((0.0, pec_hits, "#FAE8D8"),
                                   (0.5, ref_hits, "#E8ECEE")):
                hits_ax.add_patch(Rectangle((x + 0.01, i - 0.34), 0.47, 0.68,
                                            facecolor=fill, edgecolor="white", lw=0.5))
                hits_ax.text(x + 0.245, i, f"{count}/9", ha="center", va="center",
                             fontsize=6.9)
    for i in (0.5, 1.5, 2.5, 4.5):
        for ax in (labels, *ratio_axes, *hit_axes):
            ax.axhline(i, color="#C8D0D5", lw=0.65, zorder=0)
    fig.text(0.5, 0.035,
             "Bars: teacher minimum–maximum and median   ×  One qualifying teacher"
             "   - -  Equal cost", ha="center", fontsize=6.8)
    save(fig, out / "figures/n25_explicit_pec_resources_hits.pdf")
    write_csv(out / "data/pec_teacher_paired_ratios.csv", ratios,
              ["configuration", "target", "reference_hits", "pec_hits",
               "jointly_qualifying_teachers", "teacher_ratios", "median_ratio"])
    return ratios


def all_methods_pec_figure(out: Path, summary: dict, configs: dict, data: dict) -> None:
    order = ["reference_N25", "AdaBoostR_N25", "RealAdaBoost_N25",
             "projected_PEC", "pec180", "pec210", "pec240"]
    configs_by_shots = {configs[c][2]: c for c in configs
                        if configs[c][0] == 3 and same(configs[c][1], 0.01)}
    if set(configs_by_shots) != {180, 210, 240}:
        raise AssertionError("Missing d3 p=.01 PEC allocations")
    # The N=25 rows are deliberately reused across allocation panels. Verify
    # identity before using the 180-shot row as their common representative.
    pec_rows = index(data["pec_all_methods"],
        ("configuration", "target", "method", "paper_teacher", "repetition"),
        "PEC all-method repetitions")
    for target in TARGETS:
        for teacher in range(3):
            for method in ("reference_N25", "AdaBoostR_N25", "RealAdaBoost_N25"):
                entries = [summary[configs_by_shots[n], target, method, str(teacher)]
                           for n in (180, 210, 240)]
                signatures = {(e["hits"], e["nonhits"], e["cost"]) for e in entries}
                if len(signatures) != 1:
                    raise AssertionError(f"Reused N25 result changed across PEC allocations: "
                                         f"{target, teacher, method}")
                for rep in range(3):
                    original = [pec_rows[configs_by_shots[n], target, method,
                                         str(teacher), str(rep)] for n in (180, 210, 240)]
                    identity = {(r["source_result_sha256"], r["first_hit_proof_sha256"],
                                 r["confirmed_hit"], r["first_hit_algorithm_shots"])
                                for r in original}
                    if len(identity) != 1:
                        raise AssertionError(f"N25 result/proof identity changed across PEC allocations: "
                                             f"{target, teacher, rep, method}")
    if len({configs[configs_by_shots[n]][3] for n in (180, 210, 240)}) != 1:
        raise AssertionError("Inconsistent d3 p=.01 projected PEC factor")
    projected_shots = 25 * configs[configs_by_shots[180]][3]
    paper_colors = {
        "reference_N25": "#69777D",
        "AdaBoostR_N25": "#1B9E77",
        "RealAdaBoost_N25": "#7B3294",
        "projected_PEC": "#A6761D",
        "explicit_PEC": "#D95F02",
    }
    # Paper Fig. 7 places method costs on horizontal log axes beside the
    # matching teacher hit-count tables. Retain the checked scalar cohort.
    fig = plt.figure(figsize=(6.14, 6.75))
    gs = fig.add_gridspec(2, 2, left=0.27, right=0.99, top=0.90,
                          bottom=0.09, hspace=0.28, wspace=0.07,
                          width_ratios=(0.53, 0.47))
    for target_idx, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[target_idx, 0])
        tab = fig.add_subplot(gs[target_idx, 1])
        ax.set_xscale("log")
        ax.set_title("(a) Margin target" if target == "margin" else
                     "(b) Classification target", loc="left", fontweight="bold",
                     fontsize=8.0)
        full = []
        for y, name in enumerate(order):
            c = configs_by_shots[int(name[3:])] if name.startswith("pec") else configs_by_shots[180]
            method = "explicit_PEC" if name.startswith("pec") else name
            vals = []
            for teacher in range(3):
                j = str(teacher)
                if method == "projected_PEC":
                    ref = summary[c, target, "reference_N25", j]
                    if ref["qualifies"]:
                        vals.append(ref["cost"] * configs[c][3])
                else:
                    val = summary[c, target, method, j]
                    if val["qualifies"]:
                        vals.append(val["cost"] / 1e6)
            if method == "projected_PEC":
                vals = [v / 1e6 for v in vals]
            full.extend(vals)
            color = paper_colors[method]
            if len(vals) == 1:
                ax.scatter(vals[0], y, marker="x", color=color,
                           s=46, linewidths=1.6, zorder=5)
            elif vals:
                lo, hi = min(vals), max(vals)
                ax.plot([lo, hi], [y, y], color=color, lw=1.65,
                        zorder=3)
                for endpoint in (lo, hi):
                    ax.plot([endpoint, endpoint], [y - 0.075, y + 0.075],
                            color=color, lw=0.95, zorder=4)
                ax.plot([median(vals), median(vals)], [y - 0.14, y + 0.14],
                        color=color, lw=2.7, zorder=4)
        ax.set_xlim(0.75, max(full) * 1.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([
            "Reference", "AdaBoost $\\mathbb{R}$", "Real AdaBoost", "Projected PEC",
            *[f"PEC {n} ({100 * n / projected_shots:.1f}%)" for n in (180, 210, 240)],
        ])
        for label, name in zip(ax.get_yticklabels(), order):
            method = "explicit_PEC" if name.startswith("pec") else name
            label.set_color(paper_colors[method])
        ax.tick_params(axis="y", labelsize=7.2, pad=3)
        ax.set_xticks((1, 3, 10, 30, 100))
        ax.set_xticklabels(("1", "3", "10", "30", "100"))
        ax.tick_params(axis="x", labelsize=7.2)
        ax.grid(axis="x", which="major", color="#E1E5E7", lw=0.6)
        ax.axhline(3.5, color="#BEC8CD", lw=0.9)
        ax.set_xlabel("Algorithm shots (millions)", fontsize=7.4)
        tab.axis("off")
        cells = []
        for name in order:
            if name == "projected_PEC":
                cells.append(["–"] * 4)
                continue
            c = configs_by_shots[int(name[3:])] if name.startswith("pec") else configs_by_shots[180]
            method = "explicit_PEC" if name.startswith("pec") else name
            hits = [summary[c, target, method, str(t)]["hits"] for t in range(3)]
            cells.append([f"{h}/3" for h in hits] + [f"{sum(hits)}/9"])
        table = tab.table(cellText=cells, loc="center", cellLoc="center",
                          bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        for (r, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(1.2)
            if r == 3:
                cell.set_facecolor("#F4F0E7")
            else:
                hits = int(cells[r][col].split("/")[0])
                threshold = 6 if col == 3 else 2
                cell.set_facecolor("#DDF0E3" if hits >= threshold else
                                   "#EEE7F4" if hits else "#FAECCA")
                if col == 3:
                    cell.set_text_props(weight="bold")
        for col, heading in enumerate(("0", "1", "2", "All")):
            tab.text((col + 0.5) / 4, 1.025, heading, ha="center", va="bottom",
                     transform=tab.transAxes, fontsize=7.2, fontweight="bold")
    fig.suptitle(f"$d=3$, $p=0.010$; projected allocation "
                 f"$25\\gamma^2\\simeq {projected_shots:.1f}$ shots",
                 y=0.975, fontsize=8.2)
    save(fig, out / "figures/n25_all_methods_pec_comparison.pdf")


def rounds_figure(out: Path, data: dict) -> None:
    rows = [r for r in data["round_bound_endpoints"] if r["method"] == "real_adaboost"]
    if len(rows) != 135:
        raise AssertionError("Expected 135 Real AdaBoost trajectories")
    groups = defaultdict(list)
    for r in rows:
        groups[r["depth"], r["noise"], r["paper_teacher"]].append(r)
    observed = defaultdict(list)
    calculated = defaultdict(list)
    observed_teacher_count = Counter()
    calculated_teacher_count = Counter()
    theoretical_ids = set()
    for (depth, noise, teacher), group in groups.items():
        hits = [r for r in group if r["confirmed_margin_hit"] == "1"]
        eligible = [r for r in hits if r["real_T_plugin"]]
        key = (depth, noise)
        if len(hits) >= 2:
            observed[key].extend(float(r["first_margin_hit_round"]) for r in hits)
            observed_teacher_count[key] += 1
        if len(eligible) >= 2:
            calculated[key].extend(float(r["real_T_plugin"]) for r in eligible)
            calculated_teacher_count[key] += 1
            theoretical_ids.update(r["run_id"] for r in eligible)
    replay_ids = {r["run_id"] for r in data["bound_stages"]
                  if r["current_round_count_figure_57_member"] == "1"}
    if len(theoretical_ids) != 57 or theoretical_ids != replay_ids:
        raise AssertionError("Current 57-run theoretical-round cohort changed")
    gamma = index(data["resource_cell_gate"],
        ("endpoint", "noise", "depth", "method"), "gamma cells")
    hit_counts = index(data["resource_success_counts"],
        ("noise", "depth"), "margin hit counts")
    purple, ochre = "#7B3294", "#B7791F"
    fig, axes = plt.subplots(1, 3, figsize=(5.88, 4.85), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.81, bottom=0.34, wspace=0.13)
    fig.text(0.5, 0.96, "Real AdaBoost", color=purple,
             ha="center", va="top", fontsize=8.5, fontweight="bold")
    for ax, p in zip(axes, NOISES):
        pstr = str(p)
        observed_line_x, observed_line_y = [], []
        calculated_line_x, calculated_line_y = [], []
        gamma_line_x, gamma_line_y = [], []
        for depth in range(1, 6):
            key = (str(depth), pstr)
            for values, teacher_count, x, color, xvalues, yvalues in (
                (observed.get(key, []), observed_teacher_count[key], depth - 0.10,
                 purple, observed_line_x, observed_line_y),
                (calculated.get(key, []), calculated_teacher_count[key], depth + 0.10,
                 purple, calculated_line_x, calculated_line_y),
            ):
                if not values:
                    xvalues.append(x)
                    yvalues.append(math.nan)
                    continue
                center = median(values)
                if teacher_count == 1:
                    ax.scatter(x, center, color=color, marker="x", s=43,
                               linewidth=1.5, zorder=5)
                else:
                    lo, hi = min(values), max(values)
                    ax.plot([x, x], [lo, hi], color=color,
                            lw=1.0, zorder=3)
                    for endpoint in (lo, hi):
                        ax.plot([x - 0.025, x + 0.025], [endpoint, endpoint],
                                color=color, lw=0.9, zorder=3)
                    ax.plot([x - 0.05, x + 0.05], [center, center], color=color,
                            lw=2.1, zorder=4)
                xvalues.append(x)
                yvalues.append(center)
            g = fnum(gamma["margin", pstr, str(depth), "reference"]["gamma_squared"])
            ax.scatter(depth, g, color=ochre, marker="o", s=15, zorder=3)
            gamma_line_x.append(depth)
            gamma_line_y.append(g)
            c = hit_counts[pstr, str(depth)]
            ax.text(depth, -0.20, f'{sum(int(c[f"real_adaboost_margin_paper_teacher_{t}_hits"]) for t in range(3))}/9',
                    ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6.2)
            ax.text(depth, -0.30, f'{observed_teacher_count[key]}/3',
                    ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6.2)
            ax.text(depth, -0.40, f'{calculated_teacher_count[key]}/3',
                    ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6.2)
        ax.plot(observed_line_x, observed_line_y, color=purple, lw=1.45)
        ax.plot(calculated_line_x, calculated_line_y, color=purple,
                lw=1.1, ls="--")
        ax.plot(gamma_line_x, gamma_line_y, color=ochre, lw=1.25, ls=":")
        ax.set_title(f"$p={p:.3f}$", fontsize=7.8)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 2e4)
        ax.set_xlim(0.55, 5.45)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Depth", fontsize=7.3)
        ax.tick_params(axis="both", labelsize=6.8)
        ax.grid(axis="y", which="major", color="#E3E7E9", lw=0.55)
    axes[0].set_ylabel("Round count or PEC overhead $\\gamma^2$", fontsize=7.2)
    for y, label in ((-0.20, "Margin hits"), (-0.30, "Obs. teachers"),
                     (-0.40, "Theory teachers")):
        axes[0].text(-0.12, y, label, ha="right", va="top",
                     transform=axes[0].transAxes, fontsize=6.2)
    handles = [Line2D([0], [0], color=purple, lw=1.45,
                      label="Observed rounds"),
               Line2D([0], [0], color=purple, ls="--", lw=1.45,
                      label="Theoretical rounds"),
               Line2D([0], [0], color=ochre, ls=":", lw=1.45,
                      label="PEC overhead $\\gamma^2$"),
               Line2D([0], [0], color=purple, marker="x", lw=0,
                      label="One qualifying teacher")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=4, frameon=False, fontsize=6.5)
    save(fig, out / "figures/n25_boosting_rounds.pdf")


def tables(out: Path, data: dict) -> None:
    bound = data["bound_stages"]
    if Counter(r["method"] for r in bound) != {"adaboost_r": 111, "real_adaboost": 110}:
        raise AssertionError("Bound replay cohort changed")
    stage_fields = ("R_ramp_over_epsilon", "R_margin_over_epsilon",
                    "B_exp_over_epsilon", "B_method_over_epsilon")
    increment_fields = ("margin_minus_ramp_over_epsilon",
                        "exp_minus_margin_over_epsilon",
                        "method_minus_exp_over_epsilon")
    labels = ("ramp_to_margin", "margin_to_exponential",
              "exponential_to_method")
    calculated: dict[str, tuple[float, ...]] = {}
    dominant: Counter[str] = Counter()
    for row in bound:
        epsilon = float(row["epsilon_mar"])
        if epsilon <= 0 or row["run_id"] in calculated:
            raise AssertionError("Invalid or duplicate bound run")
        values = tuple(float(row[field]) for field in
                       ("R_ramp", "R_margin", "B_exp", "B_method"))
        normalized = tuple(value / epsilon for value in values)
        increases = tuple((b - a) / epsilon for a, b in zip(values, values[1:]))
        for observed, field in zip(normalized, stage_fields):
            if not math.isclose(observed, float(row[field]),
                                rel_tol=1e-11, abs_tol=1e-9):
                raise AssertionError(f"Bound stage ratio changed: {row['run_id']} {field}")
        for observed, field in zip(increases, increment_fields):
            if not math.isclose(observed, float(row[field]),
                                rel_tol=1e-11, abs_tol=1e-9):
                raise AssertionError(f"Bound adjacent increase changed: {row['run_id']} {field}")
        step = labels[max(range(3), key=lambda i: increases[i])]
        if step != row["largest_increase_step"]:
            raise AssertionError(f"Bound dominant step changed: {row['run_id']}")
        calculated[row["run_id"]] = normalized
        dominant[step] += 1
    lines = [
        "% Replayed selected first-confirmed-margin-hit N=25 models only.",
        "% Normalize each run by its own unrounded epsilon_mar before the median.",
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & $n$ & $R_{\rm ramp}/\varepsilon_{\rm mar}$ & $R_{\rm margin}/\varepsilon_{\rm mar}$ & $B_{\exp}/\varepsilon_{\rm mar}$ & $B_{\rm method}/\varepsilon_{\rm mar}$ \\",
        r"\midrule",
    ]
    for method, label in (("adaboost_r", r"AdaBoost$\mathbb R$"),
                          ("real_adaboost", "Real AdaBoost"), (None, "Both")):
        group = bound if method is None else [r for r in bound if r["method"] == method]
        vals = [median(calculated[r["run_id"]][i] for r in group)
                for i in range(len(stage_fields))]
        lines.append(f"{label} & {len(group)} & " + " & ".join(f"{v:.4f}" for v in vals) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}%", r"}",
                  r"\caption{Four-stage boosting-bound comparison for the first independently confirmed margin-hit models. Each stage is divided by that task's unrounded margin target before the median over models is taken. The stages use the same reconstructed training margins for each selected-model replay.}",
                  r"\label{tab:n25_bound_stages}", r"\end{table}"))
    output = "\n".join(lines) + "\n"
    (out / "tables/n25_bound_stages.tex").write_text(output, encoding="utf-8")
    if dominant["margin_to_exponential"] != 206:
        raise AssertionError("Bound-dominance count changed")
    targets = data["targets"]
    assert_unique(targets, ("depth", "paper_teacher"), "task targets")
    target_idx = index(targets, ("depth", "paper_teacher"), "target cutoffs")
    if (target_idx["1", "0"]["minimum_correct_executed"],
        target_idx["2", "0"]["minimum_correct_executed"]) != ("380", "374"):
        raise AssertionError("Executed integer classification cutoffs changed")
    tex = [r"\begin{tabular}{rrrrr}", r"\toprule",
           r"Depth & Teacher & $\varepsilon_{\rm mar}$ & Correct / 384 & Cutoff (\%) \\",
           r"\midrule"]
    for r in sorted(targets, key=lambda x: (int(x["depth"]), int(x["paper_teacher"]))):
        exact_percent = Fraction(r["cutoff_percent_exact_fraction"])
        if exact_percent != Fraction(100 * int(r["minimum_correct_executed"]), 384):
            raise AssertionError(f"Target cutoff percentage mismatch: {r['depth'], r['paper_teacher']}")
        tex.append(f'{r["depth"]} & {r["paper_teacher"]} & '
                   f'{fnum(r["margin_threshold_original"]):.17g} & '
                   f'{r["minimum_correct_executed"]} & '
                   f'{float(exact_percent):.4f}' + r" \\")
    tex.extend((r"\bottomrule", r"\end{tabular}"))
    (out / "tables/n25_targets.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Root of the verified publication snapshot")
    parser.add_argument("--output", type=Path, required=True,
                        help="New empty directory for generated figures/tables")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    out = args.output.resolve()
    if out == snapshot or snapshot in out.parents or out in snapshot.parents:
        raise SystemExit("Output must be separate from the immutable snapshot")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "tables", "data"):
        (out / name).mkdir()
    data = load(snapshot)
    teachers, pairs, _ = validate_resource(data)
    ratio_figure(out, pairs)
    for target in TARGETS:
        resource_cost_and_hits(out, data, teachers, target)
    test_figure(out, data, pairs)
    pec_summary, pec_configs = pec_teacher_costs(data)
    pec_ratios = pec_ratio_figure(out, pec_summary, pec_configs)
    all_methods_pec_figure(out, pec_summary, pec_configs, data)
    rounds_figure(out, data)
    tables(out, data)
    comparison_medians = [median(fnum(r["projected_pec_to_boosting_ratio"]) for r in group)
                          for group in pairs.values() if group]
    if sum(v > 1 for v in comparison_medians) != 39:
        raise AssertionError("Expected 39/56 boosting-favourable resource comparisons")
    reportable_test = [r for r in data["test_cell_summary"]
                       if r["resource_cell_reportable"] == "1"]
    if len(reportable_test) != 36 or not all(fnum(r["sampled25_cell_median_delta_pp"]) > 0
                                            for r in reportable_test):
        raise AssertionError("Expected 36 positive reportable test comparisons")
    receipt = {
        "input": "Saved scalar CSVs in the verified publication snapshot",
        "resource_comparisons_available": 56,
        "resource_comparisons_favouring_boosting": 39,
        "resource_cells_with_at_least_two_teachers": 36,
        "test_cells_with_positive_paired_median": 36,
        "bound_first_hit_models": 221,
        "real_round_figure_theoretical_runs": 57,
        "pec_completed_trajectories": 72,
        "pec_teacher_paired_ratio_rows": len(pec_ratios),
        "numeric_figures": sorted(p.name for p in (out / "figures").glob("*.pdf")),
        "circuit_evaluations": 0,
        "measurements_drawn": 0,
        "raw_test_arrays_read": False,
        "static_artwork_generated": False,
        "input_sha256": {
            path.name: sha256(path)
            for path in sorted((snapshot / "data").glob("*.csv"))
            if path.name != "pec_teacher_paired_ratios.csv"  # generated above
        },
        "output_sha256": {
            str(path.relative_to(out)): sha256(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        },
    }
    (out / "FIGURE_REBUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
