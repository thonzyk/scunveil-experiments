"""Two continuous morphology outcomes; only scUNVEIL and raw RNA in the figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator
import pandas as pd

from scope_common import LABELS, read_json

COLORS = {"scunveil": "#007C91", "raw_log1p": "#D87532"}
TARGETS = {"c2_Area": "Cell area", "c2_Circ.": "Circularity"}


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
    summary = summary[(summary.split == "test") & (summary.scope == "macro_region") & summary.method.isin(COLORS)]
    metadata = read_json(report / "report_metadata.json")
    output = Path(output) if output is not None else report / "paper_figure"
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 75 / 25.4))
    for letter, ax, (target, label) in zip("ab", axes, TARGETS.items()):
        part = summary[summary.target == target]
        rates = sorted(part.retention_rate.unique())
        for method in reversed(COLORS):
            values = part[part.method == method].set_index("retention_rate").reindex(rates)
            ax.errorbar(rates, values.mean_r2, yerr=values.sd_r2.fillna(0), color=COLORS[method],
                marker="o", ms=3.3, lw=1.3, capsize=2, elinewidth=.8, capthick=.8)
        ax.axhline(0, ls=":", color="#737373", lw=1)
        low = min(0, float((part.mean_r2 - part.sd_r2.fillna(0)).min()))
        high = max(0, float((part.mean_r2 + part.sd_r2.fillna(0)).max()))
        margin = max(.035, (high - low) * .07)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xscale("log")
        ax.set_xlim(min(rates) * .7, max(rates) * 1.4)
        ax.set_xticks(rates, [f"{r:.0%}" for r in rates])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(visible=True, axis="y", color="#CDD2D7", lw=.55)
        ax.set_axisbelow(True)
        ax.set_xlabel("UMI retention")
        ax.set_ylabel("Held-out region R²")
        ax.set_title(f"{letter}  {label}", loc="left", pad=7, weight="bold")
        ax.tick_params(length=2.5, width=.65, pad=3)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ms=3.3, lw=1.3, label=LABELS[m]) for m in COLORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1), ncol=2,
               frameon=False, handlelength=1.8, columnspacing=1.5, fontsize=8)
    n = metadata["n_repetitions"]
    subtitle = f"Mean ± SD across {n} paired repetitions" if n > 1 else "One paired repetition; SD unavailable"
    if metadata["synthetic"]:
        subtitle = "SYNTHETIC CPU FIXTURE — " + subtitle
    fig.text(.52, .025, subtitle, ha="center", fontsize=6.5, color="#555555")
    fig.subplots_adjust(left=.09, right=.98, bottom=.23, top=.76, wspace=.35)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"morphology_r2.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    (output / "CAPTION.md").write_text(
        "**Prediction of independently imaged cell morphology from sparsely measured RNA.** "
        "Human glioblastoma cells from one patient were profiled with SCOPE-seq2. Targets are cell area "
        "and circularity from the published source-standardized imaging table, linked by exact cell barcode. "
        f"{metadata['calibration_regions']} fixed technical device regions support one-time ridge regularization selection. Each repetition "
        f"randomly assigns the other {metadata['pool_regions']} regions to "
        f"{metadata['pool_regions'] - metadata['dev_regions'] - metadata['test_regions']} train, "
        f"{metadata['dev_regions']} dev and {metadata['test_regions']} test regions, draws fresh "
        "binomial molecule samples in all roles, and fits fresh training-only scalers and ridge heads. "
        "Assignments and sampled count matrices are identical across representations; assignments are "
        f"also shared across retention rates. scUNVEIL uses all {metadata['embedding_dim']} native frozen "
        "features. Raw RNA uses log1p counts without library normalization or PCA. R² is computed within "
        "each test region and averaged equally per repetition. Points show the first "
        f"{n} completed repetitions, with sample SD when available; error bars are not confidence intervals "
        "or biological replication. The RNA-summary diagnostic is in report tables. This evaluates technical "
        "region-held-out prediction within one patient, not generalization to new donors.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("/workspace/results/analysis/SCOPEseq2/report"))
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the checksummed report")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.report.resolve()):
        parser.error("Choose a new output outside the committed report")
    build_figures(args.report, args.output)
