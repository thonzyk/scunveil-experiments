"""Plot a paper figure from an existing immutable report; no fitting or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator, FormatStrFormatter
import pandas as pd

ROOT = Path("/workspace/results/analysis/ReSisTrace")
TASKS = ("Carboplatin", "Olaparib", "NK", "Control")
RATES = (.01, .1, 1.)
# Intended final placement: full width in a two-column paper, without resizing.
FIGURE_WIDTH_MM = 180
FIGURE_HEIGHT_MM = 75
STYLES = {
    "scunveil": ("scUNVEIL", "#007C91", "o"),
    "raw_log1p": ("Raw log1p RNA", "#D87532", "o"),
    "depth_qc": ("RNA depth features", "#A1A5AC", "o"),
}


def plot_metric(data, metric, output, n):
    summary = data.groupby(["condition", "retention_rate", "method"])[metric].agg(["mean", "std"])
    prevalence = data[data.method == "raw_log1p"].groupby(["condition", "retention_rate"]).prevalence.mean()
    fig, axes = plt.subplots(1, 4, figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
                             sharex=True, sharey=True)
    for ax, task in zip(axes, TASKS):
        ax.set_title("Untreated control" if task == "Control" else task, pad=7, weight="bold")
        for method in ("depth_qc", "raw_log1p", "scunveil"):
            _, color, marker = STYLES[method]
            part = summary.xs((task, method), level=("condition", "method")).reindex(RATES)
            ax.errorbar(RATES, part["mean"], yerr=part["std"], color=color, marker=marker,
                        ms=3.3, lw=1.3 if method != "depth_qc" else 1, capsize=2,
                        elinewidth=.8, capthick=.8, zorder=3 if method == "scunveil" else 2)
        if metric == "average_precision":
            ax.plot(RATES, prevalence.loc[task].reindex(RATES), ls=":", color="#737373", lw=1.1)
            ax.set_ylim(0, float((summary["mean"] + summary["std"]).max()) * 1.09)
        else:
            ax.axhline(.5, ls=":", color="#737373", lw=1, zorder=2)
            low = min(.5, float((summary["mean"] - summary["std"]).min()))
            high = max(.5, float((summary["mean"] + summary["std"]).max()))
            ax.set_ylim(max(0, low - .025), min(1, high + .035))
        ax.set_xscale("log")
        ax.set_xlim(.007, 1.4)
        ax.set_xticks(RATES, ["1%", "10%", "100%"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MultipleLocator(.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(visible=True, axis="y", which="major", color="#CDD2D7", lw=.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.65, pad=3)
    axes[0].set_ylabel("Test lineage average precision" if metric == "average_precision" else "Test lineage AUROC")
    handles = [Line2D([], [], color=color, marker=marker, ms=3.3, lw=1.3, label=label)
               for label, color, marker in STYLES.values()]
    reference = "Positive-class prevalence" if metric == "average_precision" else "Chance AUROC (0.5)"
    handles.append(Line2D([], [], color="#737373", ls=":", label=reference))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .995), ncol=4,
               frameon=False, handlelength=1.8, handletextpad=.5, columnspacing=1.3, fontsize=7)
    fig.supxlabel("UMI retention", y=.105, fontsize=8)
    fig.text(.53, .025, f"Mean ± SD across {n} paired repetitions",
             ha="center", fontsize=7, color="#555555")
    fig.subplots_adjust(left=.095, right=.985, bottom=.255, top=.78, wspace=.20)
    name = "resistrace_depth_robustness" if metric == "average_precision" else "resistrace_depth_robustness_auroc"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    summary.rename(columns={"std": "sd"}).to_csv(output / ("ap_summary.tsv" if metric == "average_precision" else "auroc_summary.tsv"), sep="\t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Report directory or REPORT.md; defaults to latest collection's latest report")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to report/paper_figure")
    args = parser.parse_args()
    if args.report:
        report = args.report.resolve()
        if report.is_file():
            report = report.parent
    else:
        manifests = list((ROOT / "collections").glob("*/collection.json"))
        if not manifests:
            parser.error("No collections found. Build a report first.")
        collection = max(manifests, key=lambda p: p.stat().st_mtime).parent
        report = Path(json.loads((collection / "LATEST_REPORT.json").read_text())["report"]).parent
    output = args.output or report / "paper_figure"
    metadata_path = report / "report_metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()).get("kind") == "unscaled_benchmark_v2":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmark_report import render_existing_report
        render_existing_report(report, output)
        return
    if metadata_path.exists() and json.loads(metadata_path.read_text()).get("kind") == "full_data_pca_comparison":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pca_report import render_existing_report
        render_existing_report(report, output)
        return
    output.mkdir(parents=True, exist_ok=True)
    source = report / "metrics.tsv"
    frame = pd.read_csv(source, sep="\t", dtype={"biological_replicate": str})
    data = frame[(frame.split == "test") & (frame.biological_replicate == "pooled")].copy()
    keys = ["repetition", "condition", "retention_rate", "method"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate test metric rows")
    n = data.repetition.nunique()
    if n < 2:
        raise ValueError("At least two repetitions required for SD")
    expected = pd.MultiIndex.from_product(
        [sorted(data.repetition.unique()), TASKS, RATES, STYLES], names=keys)
    if not data.set_index(keys).index.sort_values().equals(expected.sort_values()):
        raise ValueError("Expected complete paired coverage of all tasks, rates and methods")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.linewidth": .65,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    for metric in ("auroc",):
        plot_metric(data, metric, output, n)
    script = Path(__file__).resolve()
    provenance = {"report": str(report), "n_repetitions": n,
                  "metrics_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "script": str(script), "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                  "figure_width_mm": FIGURE_WIDTH_MM, "figure_height_mm": FIGURE_HEIGHT_MM,
                  "raster_dpi": 600}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    protocol = (
        f"Points and error bars show means and sample SD across {n} repetitions, each using "
        f"fresh grouped train/dev/test partitions and fresh binomial UMI thinning at the indicated "
        f"retention rate in all training, development and test cells, paired across "
        f"methods. Splits are shared across retention rates. Encoder weights and calibration-selected "
        f"regularization remain fixed. Both biological replicates contribute to training and testing; "
        f"repetition SD describes within-dataset resampling variability, not uncertainty across "
        f"independent biological experiments.\n")
    representations = (
        "for frozen scUNVEIL embeddings (all 2,048 features), raw log1p RNA and RNA depth features "
        "(log1p total UMIs and detected genes), each evaluated using separately calibrated logistic regression. ")
    (output / "CAPTION_AUROC.md").write_text(
        "**Lineage-recovery ranking performance under reduced UMI sampling.** "
        "Test lineage area under the receiver operating characteristic curve (AUROC) " + representations +
        "AUROC summarizes true-positive versus false-positive rates across thresholds. "
        "Dotted lines mark chance AUROC (0.5). " + protocol)
    (output / "CAPTION.md").write_text((output / "CAPTION_AUROC.md").read_text())
    print(output / "resistrace_depth_robustness_auroc.png")


if __name__ == "__main__":
    main()
