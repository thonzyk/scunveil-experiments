"""One-column AUROC figure, drawn only from a completed report snapshot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, NullLocator
import pandas as pd

from tmre_common import LABELS, read_json

FIGURE_METHODS = ("scunveil", "raw_log1p")
COLORS = {"scunveil": "#007C91", "raw_log1p": "#D87532"}


def build_figures(report, output=None):
    report = Path(report)
    metadata = read_json(report / "report_metadata.json")
    if metadata.get("kind") == "unscaled_benchmark_v2":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmark_report import render_existing_report
        return render_existing_report(report, Path(output) if output is not None else report / "paper_figure")
    if metadata.get("kind") == "full_data_pca_comparison":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pca_report import render_existing_report
        return render_existing_report(report, Path(output) if output is not None else report / "paper_figure")
    summary = pd.read_csv(report / "summary.tsv", sep="\t")
    summary = summary[(summary.split == "test") & (summary.scope == "macro_patient") &
                      summary.method.isin(FIGURE_METHODS)]
    rates = sorted(summary.retention_rate.unique())
    metadata = read_json(report / "report_metadata.json")
    output = Path(output) if output is not None else report / "paper_figure"
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(90 / 25.4, 75 / 25.4))
    for method in reversed(FIGURE_METHODS):
        part = summary[summary.method == method].set_index("retention_rate").reindex(rates)
        ax.errorbar(rates, part.mean_auroc, yerr=part.sd_auroc.fillna(0), color=COLORS[method],
                    marker="o", ms=3.3, lw=1.3, capsize=2, elinewidth=.8, capthick=.8)
    ax.axhline(.5, ls=":", color="#737373", lw=1)
    low = min(.5, float((summary.mean_auroc - summary.sd_auroc.fillna(0)).min()))
    high = max(.5, float((summary.mean_auroc + summary.sd_auroc.fillna(0)).max()))
    ax.set_ylim(max(0, low - .035), min(1, high + .035))
    ax.set_xscale("log")
    ax.set_xlim(min(rates) * .7, max(rates) * 1.4)
    ax.set_xticks(rates, [f"{rate:.0%}" for rate in rates])
    ax.xaxis.set_minor_locator(NullLocator())
    # Preserve readable 7.5 pt ticks at one-column size even with a wide SD range.
    ax.yaxis.set_major_locator(MultipleLocator(.1 if high - low > .55 else .05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(visible=True, axis="y", color="#CDD2D7", lw=.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("UMI retention")
    ax.set_ylabel("Within-visit AUROC")
    ax.set_title("Mitochondrial potential", pad=7, weight="bold")
    ax.tick_params(length=2.5, width=.65, pad=3)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ms=3.3, lw=1.3, label=LABELS[m]) for m in FIGURE_METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1), ncol=2,
               frameon=False, handlelength=1.8, columnspacing=1, fontsize=7)
    n = metadata["n_repetitions"]
    subtitle = f"Mean ± SD across {n} paired repetitions" if n >= 2 else "One paired repetition; SD unavailable"
    if metadata["synthetic"]:
        subtitle = "SYNTHETIC CPU FIXTURE\n" + subtitle
    fig.text(.54, .025, subtitle, ha="center", fontsize=6.5, color="#555555")
    fig.subplots_adjust(left=.19, right=.97, bottom=.23, top=.71)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"mitochondrial_potential_auroc.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    (output / "CAPTION.md").write_text(
        "**Mitochondrial-potential prediction under molecular undersampling.** "
        "The target is FACS-measured TMRE-high versus TMRE-low in human peripheral-blood T cells. "
        "AUROC is calculated within each held-out subject/visit and averaged equally across visits. "
        "Repetitions redraw subject roles, balanced cells and binomial molecule counts, then fit fresh "
        "training-only scalers and sklearn logistic regressions. Both methods receive the same cells, "
        "roles and counts. Cell selections and roles are shared across retention rates. "
        f"Points and error bars show mean and sample SD over the first {n} completed paired repetitions, "
        "conditional on one fixed calibration subject, selected C values and frozen encoder weights. "
        "All native scUNVEIL features are used. The dotted line is chance AUROC. "
        "Repeated sampling of four evaluation subjects does not supply independent biological replication. "
        "High/low gates use separate libraries; subset composition and systematic processing effects "
        "remain possible contributors. TMRE measures polarization, not directly ATP production or flux.\n")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("/workspace/results/analysis/MitochondrialPotential/report"))
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the checksummed report")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.report.resolve()):
        parser.error("Choose a new output outside the committed report")
    build_figures(args.report, args.output)
