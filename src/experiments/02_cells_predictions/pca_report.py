"""Atomic reports and paper-sized curves for paired full-data PCA comparisons."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd

from pca_comparison import completed, lock, read_json, seal, sha256, verify, write_json

LABELS = {"scunveil": "scUNVEIL", "raw_log1p": "Raw log1p RNA", "pca_full": "PCA, full-data fit",
          "depth_qc": "RNA depth features", "depth": "RNA summary", "clone_size": "Clonotype size",
          "depth_mito": "RNA summary + mitochondrial fraction"}
COLORS = {"scunveil": "#007C91", "raw_log1p": "#D87532", "pca_full": "#8460A8", "depth_qc": "#A1A5AC"}
MAIN = ("raw_log1p", "scunveil", "pca_full")
FIGURE_NAMES = {"ReSisTrace": "resistrace_depth_robustness_auroc",
                "TumorReactiveTCells": "tumor_reactivity_auroc", "SCOPEseq2": "morphology_r2",
                "MitochondrialPotential": "mitochondrial_potential_auroc"}
CONTEXT = {
    "ReSisTrace": "Prospective lineage-linked drug-response labels; lineage-disjoint train/dev/test within conditions. "
        "Scores are pooled test-lineage AUROC, after averaging cell predictions per lineage. This measures "
        "lineage-associated response, not direct proof of each cell's eventual fate.",
    "TumorReactiveTCells": "Experimentally tested tumor-reactivity labels; patient-disjoint splits with one randomly "
        "selected cell per clonotype per repetition. Primary AUROC is averaged over scorable held-out patients. "
        "This is recognition of tested clonotypes, with a small number of independent patients.",
    "SCOPEseq2": "Matched microscopy cell area and circularity; technical-region-disjoint splits. Primary R² is "
        "averaged equally over held-out regions. Regions are technical partitions, not independent donors. "
        "Negative R² values remain visible and are not clipped.",
    "MitochondrialPotential": "Independently measured TMRE-high versus TMRE-low FACS gates; subject-disjoint splits. "
        "Primary AUROC averages visits within each held-out subject, then subjects. Balanced extreme gates "
        "are a classification task, not continuous membrane-potential estimation. RNA-summary and mitochondrial "
        "fraction controls remain in the diagnostic tables. High/low gates use separate libraries, so "
        "composition and systematic processing effects may contribute. TMRE measures polarization, not ATP production or flux.",
}


def primary_scores(metrics, dataset):
    data = metrics[metrics.split == "test"].copy()
    metric = "r2" if dataset == "SCOPEseq2" else "auroc"
    if dataset == "ReSisTrace":
        data = data[data.biological_replicate.astype(str) == "pooled"].copy()
        data["task"] = data.condition
    else:
        scope = "macro_region" if dataset == "SCOPEseq2" else "macro_patient"
        data = data[data.scope == scope].copy()
        data["task"] = data.target if dataset == "SCOPEseq2" else dataset
    return data[["repetition", "retention_rate", "task", "method", metric]].rename(columns={metric: "score"}), metric


def summarize(primary):
    keys = ["retention_rate", "task", "method"]
    return primary.groupby(keys, sort=True, observed=True).score.agg(
        mean="mean", sd="std", n_repetitions="size").reset_index()


def validate_primary(primary, settings, count, methods):
    tasks = settings["targets"] if "targets" in settings else (
        ["Carboplatin", "Olaparib", "NK", "Control"] if "condition" in settings.get("_fields", [])
        else primary.task.unique().tolist())
    keys = ["repetition", "retention_rate", "task", "method"]
    expected = pd.MultiIndex.from_product([range(count), settings["retention_rates"], tasks, methods], names=keys)
    if primary.duplicated(keys).any() or not primary.set_index(keys).index.sort_values().equals(expected.sort_values()):
        raise ValueError("Incomplete paired method/rate/task coverage; refusing an unpaired summary")
    if not np.isfinite(primary.score).all():
        raise ValueError("Nonfinite primary metric")


def figure(summary, dataset, n, output, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    tasks = (["Carboplatin", "Olaparib", "NK", "Control"] if dataset == "ReSisTrace"
             else ["c2_Area", "c2_Circ."] if dataset == "SCOPEseq2" else [dataset])
    titles = {"Control": "Untreated control", "c2_Area": "Cell area", "c2_Circ.": "Cell circularity",
              "TumorReactiveTCells": "Tumor-reactive T cells", "MitochondrialPotential": "Mitochondrial potential"}
    methods = (*MAIN, "depth_qc") if dataset == "ReSisTrace" else MAIN
    rates = sorted(summary.retention_rate.unique())
    width = 180 if len(tasks) > 1 else 90
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(tasks), figsize=(width / 25.4, 80 / 25.4), squeeze=False,
                             sharey=dataset == "ReSisTrace")
    for ax, task in zip(axes.ravel(), tasks):
        part = summary[(summary.task == task) & summary.method.isin(methods)]
        for method in reversed(methods):
            values = part[part.method == method].set_index("retention_rate").loc[rates]
            ax.errorbar(rates, values["mean"], yerr=values.sd.fillna(0), color=COLORS[method],
                        marker="o", ms=3.3, lw=1.3, capsize=2, elinewidth=.8, capthick=.8)
        ax.axhline(0 if dataset == "SCOPEseq2" else .5, color="#888888", lw=.65, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(min(rates) / 1.25, max(rates) * 1.25)
        ax.set_xticks(rates, [f"{r:.0%}" for r in rates])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color="#E0E0E0", lw=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(titles.get(task, task), pad=7, weight="bold")
        if dataset != "ReSisTrace":
            ax.set_xlabel("UMI retention")
        ax.tick_params(length=2.5, width=.65, pad=3)
        # Include means ± SD and the reference line, with all negative R² visible.
        low = min(float((part["mean"] - part.sd.fillna(0)).min()), 0 if dataset == "SCOPEseq2" else .5)
        high = max(float((part["mean"] + part.sd.fillna(0)).max()), 0 if dataset == "SCOPEseq2" else .5)
        margin = max(.025, (high - low) * .08)
        if dataset != "ReSisTrace":
            ax.set_ylim((low - margin, high + margin) if dataset == "SCOPEseq2" else
                        (max(0, low - margin), min(1, high + margin)))
        if dataset != "ReSisTrace" or ax == axes.ravel()[0]:
            ax.set_ylabel({"ReSisTrace": "Test lineage AUROC", "TumorReactiveTCells": "Within-patient AUROC",
                "SCOPEseq2": "Held-out region R²", "MitochondrialPotential": "Within-visit AUROC"}[dataset])
    if dataset == "ReSisTrace":
        shown = summary[summary.method.isin(methods)]
        low = min(float((shown["mean"] - shown.sd.fillna(0)).min()), .5)
        high = float((shown["mean"] + shown.sd.fillna(0)).max())
        axes[0, 0].set_ylim(max(0, low - .025), min(1, high + .035))
        fig.supxlabel("UMI retention", y=.20, fontsize=8)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ms=3.3, lw=1.3, label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .065),
               ncol=2 if width == 90 else len(methods), frameon=False, fontsize=7, columnspacing=1.2)
    subtitle = f"Mean ± SD across {n} paired repetitions" if n > 1 else "One paired repetition; SD unavailable"
    if synthetic:
        subtitle = "SYNTHETIC CPU FIXTURE — " + subtitle
    fig.text(.52, .02, subtitle, ha="center", fontsize=6.5, color="#555555")
    fig.subplots_adjust(left=.18 if width == 90 else .07, right=.98, top=.88, bottom=.34, wspace=.35)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{FIGURE_NAMES[dataset]}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def caption(metadata):
    pca = metadata["pca"]
    n = metadata["n_repetitions"]
    return (f"**{metadata['dataset']}: prediction under molecular undersampling, with full-data PCA.** "
        + CONTEXT[metadata["dataset"]] + " "
        "All methods use the same cells, grouped train/dev/test assignments and binomial count draws in each "
        "repetition. Cell selection and group roles are shared across retention rates. Counts in all roles "
        "are thinned at the indicated retention. The original separate calibration groups select fixed "
        "regularization values once; training-only scalers and sklearn heads are fitted afresh each repetition. "
        f"scUNVEIL uses all {metadata['embedding_dim']:,} frozen features. Raw RNA uses sparse log1p counts. "
        f"PCA uses {pca['n_components']:,} centered, unwhitened log1p components fitted once on all "
        f"{pca['n_fit_cells']:,} full-depth eligible cells, deliberately including held-out cells without labels. "
        "Its frozen loadings and mean are applied to the current thinned counts at every depth. "
        f"Points and error bars are mean and sample SD across the first {n} completed paired repetitions; "
        "SD is unavailable with one repetition. These are overlapping repeated splits, not independent biological "
        "cohorts or confidence intervals. The dotted reference is chance AUROC (0.5) or zero R².\n")


def render_existing_report(report, output):
    """Used by the original per-dataset plot scripts for a PCA report snapshot."""
    report, output = Path(report), Path(output)
    metadata = read_json(report / "report_metadata.json")
    if output.exists():
        raise FileExistsError(f"Choose a new figure directory: {output}")
    figure(pd.read_csv(report / "summary.tsv", sep="\t"), metadata["dataset"],
           metadata["n_repetitions"], output, metadata["synthetic"])
    (output / "CAPTION.md").write_text(caption(metadata))


def build_report(root, max_repetitions=100):
    root = Path(root).resolve()
    manifest = read_json(root / "collection.json")
    if manifest.get("kind") != "full_data_pca_comparison":
        raise ValueError("Expected a PCA comparison collection")
    dataset = manifest["dataset"]
    with lock(root / "report.lock"):
        # Snapshot published paths now. A collector can publish more while we read.
        paths = completed(root)
        if max_repetitions:
            paths = paths[:max_repetitions]
        if not paths:
            raise ValueError("No complete paired PCA repetitions yet")
        basis = verify(root / "pca_basis")
        verify(root / "calibration")
        pieces, receipts = [], []
        for path in paths:
            info = verify(path)
            source = Path(info["source_repetition"])
            if not info.get("pairing_verified") or sha256(source / "complete.json") != info["source_receipt_sha256"]:
                raise ValueError("Original repetition receipt changed or pairing was not verified")
            verify(source, list(info["source_files"]))
            pieces.append(pd.read_csv(path / "metrics.tsv", sep="\t", dtype={"biological_replicate": str}))
            receipts.append({"directory": str(path), "receipt_sha256": sha256(path / "complete.json"),
                             "source_directory": str(source), "source_receipt_sha256": info["source_receipt_sha256"]})
        metrics = pd.concat(pieces, ignore_index=True)
        primary, metric = primary_scores(metrics, dataset)
        settings = {**manifest["settings"], "_fields": list(metrics.columns)}
        methods = sorted(metrics.method.unique())
        validate_primary(primary, settings, len(paths), methods)
        summary = summarize(primary)
        pivot = primary.pivot(index=["repetition", "retention_rate", "task"], columns="method", values="score")
        differences = pd.concat([(pivot.scunveil - pivot[method]).rename("difference").reset_index().assign(baseline=method)
                                  for method in methods if method != "scunveil"], ignore_index=True)
        paired = differences.groupby(["retention_rate", "task", "baseline"]).difference.agg(
            mean="mean", sd="std", n_repetitions="size", fraction_positive=lambda x: float((x > 0).mean())).reset_index()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"{timestamp}_{len(paths)}_repetitions"
        pending = root / "reports" / (".pending_" + name)
        pending.mkdir(parents=True)
        metrics.to_csv(pending / "metrics.tsv", sep="\t", index=False)
        primary.to_csv(pending / "primary_scores.tsv", sep="\t", index=False)
        summary.to_csv(pending / "summary.tsv", sep="\t", index=False)
        differences.to_csv(pending / "paired_differences.tsv", sep="\t", index=False)
        paired.to_csv(pending / "paired_summary.tsv", sep="\t", index=False)
        # Preserve original diagnostic tables in the native metric format. New
        # predictions/models are PCA-only, and refer back to the unchanged source.
        metadata = {"kind": "full_data_pca_comparison", "dataset": dataset, "collection": str(root),
            "source_collection": manifest["source_collection"],
            "n_repetitions": len(paths), "replicate_ids": list(range(len(paths))), "primary_metric": metric,
            "embedding_dim": read_json(Path(manifest["source_collection"]) / "model.json")["embedding_dim"],
            "pca": basis["pca"], "pca_basis_receipt_sha256": sha256(root / "pca_basis/complete.json"),
            "calibration_receipt_sha256": sha256(root / "calibration/complete.json"), "repetitions": receipts,
            "report_code": sha256(Path(__file__)), "synthetic": manifest["settings"]["model_version"] == "synthetic",
            "figure_methods": list(MAIN) + (["depth_qc"] if dataset == "ReSisTrace" else [])}
        write_json(pending / "report_metadata.json", metadata)
        synthetic = manifest["settings"]["model_version"] == "synthetic"
        figure(summary, dataset, len(paths), pending / "paper_figure", synthetic)
        pca = basis["pca"]
        text = [f"# {dataset}: full-data PCA comparison\n"]
        if synthetic:
            text.append("**SYNTHETIC CPU FIXTURE — not a biological result.**\n")
        text.extend([f"First **{len(paths)} complete paired repetitions**, in collection order. Primary metric: **{metric.upper()}**.\n",
            CONTEXT[dataset] + "\n",
            f"PCA was fitted once on **all {pca['n_fit_cells']:,} eligible full-depth cells** and {pca['n_genes']:,} genes, "
            f"including calibration/train/dev/test. It uses **{pca['n_components']:,} components** "
            f"(requested {pca['settings']['requested_components']:,}; limited by min(n_cells − 1, n_genes)). "
            f"scUNVEIL uses all **{metadata['embedding_dim']:,}** frozen features.\n",
            "The PCA mean and loadings are frozen. At every retention level, inputs are independently binomially "
            "thinned integer counts and then log1p transformed. Full-depth coordinates are never substituted for "
            "shallow measurements. No library normalization, HVG filtering or PCA whitening is applied. "
            "The downstream scaler is fitted on training cells only.\n",
            "PCA has intentionally privileged, transductive access to full-depth held-out RNA, without labels. "
            "This is not an inductive PCA comparison. scUNVEIL additionally has its own external pretraining history. "
            "This fixed high-dimensional PCA configuration is not a search for the optimal number of PCs.\n",
            "Original raw-RNA, scUNVEIL and shortcut-control results are reused unchanged. The original scientific "
            "code is checked, and splits, selected cells, dilution seeds and measurement audits are replayed and "
            "matched before a PCA repetition is published. Fresh sklearn probes are fitted in every repetition. "
            "PCA regularization is selected once per retention (and ReSisTrace condition), using exactly the "
            "original separate calibration groups, folds, criterion and parameter grid; it is then reused.\n",
            "Error bars are sample SD (ddof=1) across paired repetitions, not standard errors or confidence intervals. "
            "Overlapping repetitions reuse biological subjects/cells; 100 repetitions are not 100 independent cohorts.\n",
            f"![Paired comparison](paper_figure/{FIGURE_NAMES[dataset]}.png)\n",
            "## Test performance: mean ± SD\n",
            f"| Task | UMI retention | {LABELS['raw_log1p']} | scUNVEIL | {LABELS['pca_full']} |",
            "| --- | --- | --- | --- | --- |"])
        for (task, rate), group in summary.groupby(["task", "retention_rate"], sort=True):
            values = group.set_index("method")
            cells = [f"{values.loc[m, 'mean']:.3f} ± {values.loc[m, 'sd']:.3f}" if len(paths) > 1
                     else f"{values.loc[m, 'mean']:.3f} (SD unavailable)" for m in MAIN]
            text.append(f"| {task} | {rate:.0%} | " + " | ".join(cells) + " |")
        text.extend(["\n## Paired scUNVEIL − PCA differences\n",
                     "Positive values favor scUNVEIL; negative values favor PCA.\n",
                     "| Task | UMI retention | Mean Δ ± SD | Fraction Δ > 0 |",
                     "| --- | --- | --- | --- |"])
        for row in paired[paired.baseline == "pca_full"].itertuples():
            sd = f"{row.sd:.3f}" if len(paths) > 1 else "unavailable"
            text.append(f"| {row.task} | {row.retention_rate:.0%} | {row.mean:+.3f} ± {sd} | {row.fraction_positive:.1%} |")
        text.extend(["\n`metrics.tsv` retains every original diagnostic scope, method and metric, plus PCA; "
            "`primary_scores.tsv`, `summary.tsv`, `paired_differences.tsv` and `paired_summary.tsv` contain the "
            "reported test summaries. All original shortcut controls remain available in these tables. "
            "Prediction files in the PCA collection contain PCA only; original predictions are in the source collection.\n",
            f"Source collection: `{manifest['source_collection']}`.\n",
            f"PCA collection: `{root}`.\n",
            f"PCA solver: {pca['solver']}. Explained-variance fraction: {pca['explained_variance_ratio_sum']:.4f}.\n",
            "PCA interpretation: a gain over this baseline supports the pretrained representation under this "
            "particular full-data PCA comparison. A tie or loss is retained and should narrow the claim; it does "
            "not invalidate the labels or justify removing the baseline. Existing paper assets are unchanged.\n"])
        (pending / "REPORT.md").write_text("\n".join(text))
        (pending / "paper_figure/CAPTION.md").write_text(caption(metadata))
        seal(pending, {"n_repetitions": len(paths), "replicate_ids": list(range(len(paths))),
                       "collection": str(root), "source_collection": manifest["source_collection"]})
        destination = root / "reports" / name
        pending.rename(destination)
        stable = root.parent.parent / "report"
        if stable.exists() and not stable.is_symlink():
            raise ValueError(f"Stable report path is a real directory; refusing to replace: {stable}")
        temporary = stable.with_name(f".report_pca_{os.getpid()}")
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(destination, target_is_directory=True)
        temporary.replace(stable)
        write_json(root / "LATEST_REPORT.json", {"report": str(destination / "REPORT.md")})
        return stable
