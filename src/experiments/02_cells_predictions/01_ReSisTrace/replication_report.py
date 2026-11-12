"""Summarize immutable completed experiments, never unfinished scratch work."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from common import CONDITIONS, read_json, sha256_file, write_json
from replication_common import completed_replicates, exclusive_lock, seal, verify, write_frame

METRICS = ("average_precision", "auroc", "balanced_accuracy", "brier")
LABELS = {"scunveil": "scUNVEIL (all features)", "raw_log1p": "Raw log1p RNA", "depth_qc": "RNA depth features"}
COLORS = {"scunveil": "#126782", "raw_log1p": "#d16b35", "depth_qc": "#777777"}


def summarize(frame, by):
    return frame.groupby(by, as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n_repetitions=("repetition", "nunique"),
        repetition_q025=("value", lambda x: x.quantile(.025)),
        repetition_q975=("value", lambda x: x.quantile(.975)))


def summary_frames(metrics):
    ids = ["repetition", "retention_rate", "condition", "split", "biological_replicate", "method"]
    long = metrics.melt(id_vars=ids, value_vars=list(METRICS), var_name="metric", value_name="value")
    summary = summarize(long, [k for k in ids if k not in ("repetition",)] + ["metric"])
    keys = [k for k in ids if k != "method"] + ["metric"]
    wide = long.pivot(index=keys, columns="method", values="value")
    differences = pd.concat([(wide.scunveil - wide[baseline]).rename("value").reset_index().assign(baseline=baseline)
                             for baseline in ("raw_log1p", "depth_qc")], ignore_index=True)
    delta_summary = summarize(differences, [k for k in keys if k != "repetition"] + ["baseline"])
    return long, summary, differences, delta_summary


def plots(output, summary, differences, settings, synthetic):
    order = sorted(settings["retention_rates"])
    for metric, ylabel in (("average_precision", "Lineage average precision"), ("auroc", "Lineage AUROC")):
        table = summary[(summary.split == "test") & (summary.biological_replicate == "pooled") & (summary.metric == metric)]
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
        for axis, condition in zip(axes, CONDITIONS):
            for method, color in COLORS.items():
                part = table[(table.condition == condition) & (table.method == method)].sort_values("retention_rate")
                axis.errorbar(part.retention_rate, part["mean"], yerr=part.sd.fillna(0), marker="o", capsize=4,
                              color=color, label=LABELS[method])
            axis.set_xscale("log")
            axis.set_xticks(order, [f"{p:.0%}" for p in order])
            axis.set(title=condition, xlabel="UMIs retained in train / dev / test", ylabel=ylabel)
            axis.legend(fontsize=8)
        fig.suptitle(("SYNTHETIC CPU FIXTURE\n" if synthetic else "") +
                     "ReSisTrace: fresh grouped splits, dilution and training\nError bars = repetition SD; C fixed from separate calibration")
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(output / f"{metric}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    table = differences[(differences.split == "test") & (differences.biological_replicate == "pooled") &
                        (differences.metric == "average_precision")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    for axis, condition in zip(axes, CONDITIONS):
        for baseline in ("raw_log1p", "depth_qc"):
            part = table[(table.condition == condition) & (table.baseline == baseline)].sort_values("retention_rate")
            axis.errorbar(part.retention_rate, part["mean"], yerr=part.sd.fillna(0), capsize=4, marker="o",
                          color=COLORS[baseline], label=f"minus {LABELS[baseline]}")
        axis.axhline(0, linestyle=":", color="grey")
        axis.set_xscale("log")
        axis.set_xticks(order, [f"{p:.0%}" for p in order])
        axis.set(title=condition, xlabel="UMIs retained", ylabel="Paired scUNVEIL minus control AP")
        axis.legend(fontsize=8)
    fig.suptitle(("SYNTHETIC CPU FIXTURE\n" if synthetic else "") + "Paired gains: mean ± repetition SD")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(output / f"paired_gains.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(root, max_repetitions=100):
    if max_repetitions is not None and (type(max_repetitions) is not int or max_repetitions < 1):
        raise ValueError("max_repetitions must be a positive integer or None")
    root = Path(root).resolve()
    manifest = read_json(root / "collection.json")
    settings = manifest["settings"]
    with exclusive_lock(root / "report.lock"):
        # Snapshot the committed directory list once. New commits during this
        # report belong to the next snapshot; committed files never change.
        repetitions = completed_replicates(root)
        available = len(repetitions)
        if max_repetitions is not None:
            repetitions = repetitions[:max_repetitions]
        if not repetitions:
            raise ValueError(f"No complete repetitions in {root}. Calibration/pending work is excluded.")
        verify(root / "calibration")
        model = read_json(root / "model.json")
        widths = {info["embedding_dim"] for _, info in repetitions}
        if widths != {model["embedding_dim"]}:
            raise ValueError("Mixed or truncated scUNVEIL widths in collection")
        n = len(repetitions)
        metrics = pd.concat([pd.read_csv(p / "metrics.tsv", sep="\t", dtype={"biological_replicate": str})
                             for p, _ in repetitions], ignore_index=True)
        required = len(settings["retention_rates"]) * len(CONDITIONS) * 2 * 3 * 3
        if any(len(metrics[metrics.repetition == info["replicate"]]) != required for _, info in repetitions):
            raise ValueError("Completed repetition has an unexpected metric layout")
        long, summary, differences, delta_summary = summary_frames(metrics)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        parent = root / "reports"
        parent.mkdir(exist_ok=True)
        output = parent / f".pending-{timestamp}"
        output.mkdir()
        for name, frame in (("metrics", metrics), ("long_metrics", long), ("summary", summary),
                            ("paired_differences", differences), ("paired_summary", delta_summary)):
            write_frame(output / f"{name}.tsv", frame)
        for filename in ("measurement_audit.tsv", "selected_c.tsv", "fit_times.tsv"):
            write_frame(output / filename, pd.concat([
                pd.read_csv(p / filename, sep="\t").assign(repetition=info["replicate"])
                for p, info in repetitions], ignore_index=True))
        write_frame(output / "completed_repetitions.tsv", pd.DataFrame([
            {"repetition": info["replicate"], "elapsed_seconds": info["elapsed_seconds"],
             "train_cells": info["n_train_cells"], "dev_cells": info["n_dev_cells"],
             "test_cells": info["n_test_cells"], "receipt_sha256": sha256_file(p / "complete.json")}
            for p, info in repetitions]))
        synthetic = model["model_version"] == "synthetic"
        plots(output, summary, delta_summary, settings, synthetic)
        primary = summary[(summary.split == "test") & (summary.biological_replicate == "pooled") &
                          (summary.metric == "auroc")]
        delta = delta_summary[(delta_summary.split == "test") & (delta_summary.biological_replicate == "pooled") &
                              (delta_summary.metric == "auroc")]
        lines = ["# ReSisTrace: repeated grouped train/dev/test experiments", "",
                 f"Collection `{root.name}`; snapshot of **{n} complete repetitions**; "
                 f"scUNVEIL model `{model['model_version']}`, **all {model['embedding_dim']} features**.", ""]
        lines += [f"Reporting the first {n} completed repetitions in collection order "
                  f"({available} available at build time). Selection uses repetition index, not performance; "
                  "all saved repetitions are preserved.", ""]
        if synthetic:
            lines += ["**SYNTHETIC CPU FIXTURE — software validation only, not biological results.**", ""]
        lines += [
            "Each repetition freshly partitions whole lineage groups, stratified by condition, biological replicate and outcome; "
            "freshly dilutes train/dev/test RNA; fits fresh training-only scalers and sklearn logistic classifiers. "
            "Splits and counts are identical across methods, and splits are shared across retention rates.", "",
            f"A fixed {settings['calibration_fraction']:.0%} of source lineage groups is reserved for calibration and excluded "
            f"from every repetition's train/dev/test sets. Remaining groups split approximately "
            f"{settings['train_fraction']:.0%}/{settings['dev_fraction']:.0%}/{settings['test_fraction']:.0%}. "
            "C is selected separately per condition, rate and representation by grouped CV on calibration lineages only, then frozen. "
            "Final probes train on the training partition only; dev cells are scored separately and never folded into training.", "",
            "This tests held-out lineages WITHIN the available biological experiments: both biological replicates are represented "
            "in training and testing. It is a different protocol from the historical replicate-held-out benchmark; their scores "
            "must not be pooled or interpreted as directly equivalent.", "",
            "## Test lineage AUROC: mean ± sample SD", "",
            "![Test lineage AUROC](paper_figure/resistrace_depth_robustness_auroc.png)", "",
            "AUROC summarizes ranking across classification thresholds; 0.5 represents chance and 1.0 perfect ranking.", "",
            "| Retention | Task | Representation | Mean AUROC | Repetition SD | Repetitions |",
            "|---|---|---|---:|---:|---:|"]
        for r in primary.itertuples():
            sd = f"{r.sd:.4f}" if pd.notna(r.sd) else "n/a"
            lines.append(f"| {r.retention_rate:.0%} | {r.condition} | {LABELS[r.method]} | {r.mean:.4f} | {sd} | {r.n_repetitions} |")
        lines += ["", "## Paired AUROC gains", "", "Positive values favor scUNVEIL. Pairing precedes summarization.", "",
                  "| Retention | Task | Control | Mean gain | Repetition SD |", "|---|---|---|---:|---:|"]
        for r in delta.itertuples():
            sd = f"{r.sd:.4f}" if pd.notna(r.sd) else "n/a"
            lines.append(f"| {r.retention_rate:.0%} | {r.condition} | {LABELS[r.baseline]} | {r.mean:+.4f} | {sd} |")
        calibration = pd.read_csv(root / "calibration/tuning.tsv", sep="\t")
        mean_tuning = calibration.groupby(["retention_rate", "condition", "method", "C"], as_index=False).lineage_ap.mean()
        selected = mean_tuning.sort_values(["lineage_ap", "C"], ascending=[False, True]).drop_duplicates(
            ["retention_rate", "condition", "method"])
        boundaries = selected[selected.C.isin([min(settings["c_values"]), max(settings["c_values"])])]
        write_frame(output / "calibration_grid_audit.tsv", selected.assign(at_boundary=selected.index.isin(boundaries.index)))
        lines += ["", "## Interpretation and uncertainty", "",
                  "- SD includes split/training-composition and molecular-draw variability. Full-depth results also have SD "
                  "once at least two repetitions complete, because their training/test groups change.",
                  "- C, calibration data and pretrained encoder weights stay fixed. The encoder is never fine-tuned. "
                  "SD therefore excludes hyperparameter-selection and pretraining uncertainty.",
                  "- Repeated partitions reuse one finite source dataset and have overlapping cells/lineages across repetitions. "
                  "Repetition SD and empirical quantiles describe this resampling procedure; they are not confidence intervals "
                  "for new donors/experiments. Do not treat the repetitions as independent biological replicates.",
                  "- Primary AP/AUROC give equal weight to test lineage groups after averaging their cell predictions. "
                  "AP depends on prevalence. Per-biological-replicate scores and dev scores are retained in the TSV files.",
                  "- Raw RNA uses log1p counts without library normalization or PCA. RNA depth features use log1p total UMIs and "
                  "log1p detected genes, computed from the same diluted matrices. Empty diluted cells remain included.",
                  "- Outcomes concern observed lineage recovery, not proven resistance of an individual sequenced cell. "
                  "Untreated recovery measures growth/procedure effects. Source identifiers are sample-local, so cross-sample "
                  "barcode disjointness cannot be established. Pretraining overlap, CNV and cell-cycle confounding remain unaudited.",
                  f"- {len(boundaries)}/{len(selected)} calibration selections touch a C-grid boundary; inspect calibration_grid_audit.tsv. "
                  "No test outcome is used to retune C.",
                  "- This report includes only atomically committed, checksummed repetitions. Pending/interrupted repetitions "
                  "are excluded in their entirety, keeping every comparison paired.", "",
                  "## Files", "", "`average_precision.svg/png`, `auroc.svg/png`, `paired_gains.svg/png`; "
                  "`summary.tsv` and `paired_summary.tsv` hold mean/SD; `metrics.tsv` and `paired_differences.tsv` retain "
                  "each complete repetition. Source splits, predictions and optional fitted models remain in the collection's "
                  "`replicates/` directory; fixed calibration groups, CV splits and tuning are in `calibration/`.", ""]
        (output / "REPORT.md").write_text("\n".join(lines))
        seal(output, {"n_complete_repetitions": n, "replicate_ids": [info["replicate"] for _, info in repetitions],
                      "collection": str(root), "embedding_dim": model["embedding_dim"]})
        destination = parent / f"{timestamp}_{n}_repetitions"
        output.rename(destination)
        if not synthetic and n >= 2:
            subprocess.run([sys.executable, str(Path(__file__).with_name("plot_paper_figure.py")),
                            "--report", str(destination)], check=True)
            # Include automatically generated figure files in the report receipt.
            seal(destination, {"n_complete_repetitions": n,
                               "replicate_ids": [info["replicate"] for _, info in repetitions],
                               "collection": str(root), "embedding_dim": model["embedding_dim"]})
        # Atomic pointer replacement exposes a complete report, never a half-written directory.
        stable = root.parent.parent / "report"
        temporary = stable.with_name(f".report-{timestamp}")
        temporary.symlink_to(destination, target_is_directory=True)
        os.replace(temporary, stable)
        write_json(root / "LATEST_REPORT.json", {"report": str(destination / "REPORT.md"), "n_repetitions": n})
        return destination
