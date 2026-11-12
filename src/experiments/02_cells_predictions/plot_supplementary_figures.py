"""Paper-only rendering of frozen benchmark summaries; no collection/report edits.

Run with --help for paths. Raw RNA, full-data PCA and scUNVEIL are the only
displayed methods. Axis bounds follow their mean +/- sample SD; reference
lines never widen the axes. ReSisTrace panels retain a shared data-derived
scale. The final snapshot manifest pins the exact 100-repetition reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from pca_comparison import sha256, verify
from pca_report import COLORS, FIGURE_NAMES, LABELS

METHODS = ("raw_log1p", "pca_full", "scunveil")
ORDER = ("TumorReactiveTCells", "MitochondrialPotential", "SCOPEseq2", "ReSisTrace")
PAPER_NAMES = {**FIGURE_NAMES, "SCOPEseq2": "scope_morphology_r2"}
TASKS = {"ReSisTrace": ("Carboplatin", "Olaparib", "NK", "Control"),
         "SCOPEseq2": ("c2_Area", "c2_Circ.")}
TITLES = {"Control": "Untreated control", "c2_Area": "Cell area",
          "c2_Circ.": "Cell circularity", "TumorReactiveTCells": "Tumor-reactive T cells",
          "MitochondrialPotential": "Mitochondrial potential"}
Y_LABELS = {"ReSisTrace": "Test lineage AUROC", "TumorReactiveTCells": "Within-patient AUROC",
            "SCOPEseq2": "Held-out region R²", "MitochondrialPotential": "Within-visit AUROC"}


def limits(values):
    low = float((values["mean"] - values.sd).min())
    high = float((values["mean"] + values.sd).max())
    span = high - low
    padding = .08 * span if span > 0 else max(abs(low) * .05, 1e-6)
    return low - padding, high + padding


def render(summary, dataset, n, output):
    tasks = TASKS.get(dataset, (dataset,))
    shown = summary[summary.method.isin(METHODS)].copy()
    width = 160 if len(tasks) > 1 else 90
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": .65, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(tasks), figsize=(width / 25.4, 80 / 25.4),
        squeeze=False, sharey=dataset == "ReSisTrace")
    metadata = {}
    for index, (ax, task) in enumerate(zip(axes.ravel(), tasks)):
        part = shown[shown.task == task]
        for method in METHODS:
            values = part[part.method == method].sort_values("retention_rate")
            if list(values.retention_rate) != [.01, .1, 1.] or not np.isfinite(values[["mean", "sd"]]).all().all():
                raise ValueError(f"Missing/nonfinite three-point summary: {dataset}/{task}/{method}")
            ax.errorbar(values.retention_rate, values["mean"], yerr=values.sd,
                color=COLORS[method], marker="o", ms=3.3, lw=1.3, capsize=2,
                elinewidth=.8, capthick=.8, zorder=2 + METHODS.index(method))
        low, high = limits(shown if dataset == "ReSisTrace" else part)
        reference = 0. if dataset == "SCOPEseq2" else .5
        visible = low <= reference <= high
        if visible:
            ax.axhline(reference, color="#888888", lw=.65, ls=":", zorder=0)
        # Set bounds after plotting: no baseline or metric boundary affects them.
        ax.set_ylim(low, high)
        ax.set_xscale("log")
        ax.set_xlim(.01 / 1.25, 1.25)
        ax.set_xticks([.01, .1, 1.], ["1%", "10%", "100%"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color="#E0E0E0", lw=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(TITLES.get(task, task), pad=7, weight="bold")
        ax.tick_params(length=2.5, width=.65, pad=3)
        if dataset != "ReSisTrace":
            ax.set_xlabel("UMI retention")
        if dataset != "ReSisTrace" or index == 0:
            ax.set_ylabel(Y_LABELS[dataset])
        metadata[task] = {"y_limits": [low, high], "reference_visible": bool(visible)}
    if dataset == "ReSisTrace":
        fig.supxlabel("UMI retention", y=.20, fontsize=8)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ms=3.3, lw=1.3,
                      label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .065),
        ncol=2 if width == 90 else 3, frameon=False, fontsize=7, columnspacing=1.2)
    fig.text(.52, .02, f"Mean ± SD across {n} paired repetitions", ha="center",
        fontsize=6.5, color="#555555")
    fig.subplots_adjust(left=.18 if width == 90 else .095, right=.98, top=.88,
        bottom=.34, wspace=.35)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{PAPER_NAMES[dataset]}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
        default=Path("/workspace/results/analysis/benchmark_suites/paper_snapshot-final-100-repetitions.json"))
    parser.add_argument("--output", type=Path,
        default=Path("/workspace/results/analysis/benchmark_suites/paper_figures-final-100"))
    parser.add_argument("--paper-figures", type=Path,
        default=Path("/workspace/subprojects/04_scunveil-paper/src/figures"))
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    provenance = {"snapshot": str(args.snapshot.resolve()), "snapshot_sha256": sha256(args.snapshot),
        "script_sha256": sha256(Path(__file__)), "methods": METHODS,
        "axis_policy": "means +/- sample SD with 8% padding; no forced reference or metric bounds",
        "datasets": {}}
    for dataset in ORDER:
        item = snapshot["datasets"][dataset]
        report = Path(item["report"])
        verify(report)
        meta = json.loads((report / "report_metadata.json").read_text())
        if meta["kind"] != "unscaled_benchmark_v2" or meta["n_repetitions"] != 100 or meta["synthetic"]:
            raise ValueError(f"Expected final corrected 100-repetition report: {dataset}")
        summary = pd.read_csv(report / "summary.tsv", sep="\t")
        if not summary.n_repetitions.eq(100).all():
            raise ValueError(f"Mixed repetition counts: {dataset}")
        info = render(summary, dataset, 100, args.output / dataset)
        source = args.output / dataset / f"{PAPER_NAMES[dataset]}.pdf"
        args.paper_figures.mkdir(parents=True, exist_ok=True)
        destination = args.paper_figures / source.name
        shutil.copyfile(source, destination)
        provenance["datasets"][dataset] = {"report": str(report),
            "summary_sha256": sha256(report / "summary.tsv"), "panels": info,
            "paper_pdf": str(destination), "figure_sha256": sha256(destination)}
        print(dataset, json.dumps(info), flush=True)
    (args.output / "plot_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
