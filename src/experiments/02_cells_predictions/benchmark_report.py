"""Atomic, collection-local reports for the corrected unscaled protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pca_comparison import completed, digest, lock, read_json, seal, sha256, verify, write_json
from pca_report import CONTEXT, FIGURE_NAMES, LABELS, MAIN, figure, primary_scores, summarize, validate_primary

KIND = "unscaled_benchmark_v2"


def caption(meta):
    pca = meta["pca"]
    return (f"**{meta['dataset']}: independently measured outcomes under molecular undersampling.** "
        + CONTEXT[meta["dataset"]] + " "
        f"All methods receive identical selected cells, grouped roles and binomially sampled counts at 100%, 10% and 1% "
        "retention, in training, dev and test. Raw RNA is sparse log1p counts; scUNVEIL uses all "
        f"{meta['embedding_dim']:,} frozen library features; PCA uses {pca['n_components']:,} components fitted once "
        f"on all {pca['n_fit_cells']:,} eligible full-depth cells including held-out cells without labels. "
        "The frozen PCA mean/loadings project the current diluted counts. No downstream standardization, "
        "variance scaling, whitening, library normalization or extra centering is applied to any representation. "
        "PCA's defining full-depth gene centering and the estimators' intercepts remain. "
        "Regularization is recalibrated once per method/retention/task using reserved calibration groups and "
        "the reported AUROC or within-region R². Area and circularity have independent ridge penalties. "
        "Class/sample weights follow the declared original task protocol. ReSisTrace averages direct "
        "cell probabilities within lineages, without class-prior correction. "
        f"Points/error bars are mean/sample SD (ddof=1) across the first {meta['n_repetitions']} completed paired "
        "repetitions. These overlapping repetitions are not independent biological cohorts or confidence intervals. "
        "The reference is AUROC=0.5 or R²=0.\n")


def render_existing_report(report, output):
    report, output = Path(report), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new figure directory: {output}")
    meta = read_json(report / "report_metadata.json")
    figure(pd.read_csv(report / "summary.tsv", sep="\t"), meta["dataset"], meta["n_repetitions"], output, meta["synthetic"])
    (output / "CAPTION.md").write_text(caption(meta))


def build_report(root, max_repetitions=100):
    root = Path(root).resolve()
    manifest = read_json(root / "collection.json")
    if manifest["kind"] != KIND:
        raise ValueError("Expected the corrected unscaled benchmark protocol")
    with lock(root / "report.lock"):
        paths = completed(root)
        if max_repetitions:
            paths = paths[:max_repetitions]
        if not paths:
            raise ValueError("No completed repetitions in this collection yet")
        verify(root / "calibration")
        calibration_hash = sha256(root / "calibration/complete.json")
        frames, receipts = [], []
        for index, path in enumerate(paths):
            info = verify(path)
            if (info["protocol"] != KIND or not info["no_scaler"] or
                info["collection_identity"] != digest(manifest["identity"]) or
                info["calibration_sha256"] != calibration_hash or
                info.get("repetition", info.get("replicate")) != index):
                raise ValueError("Mixed protocols, repetition IDs or calibrations; refusing report")
            frames.append(pd.read_csv(path / "metrics.tsv", sep="\t", dtype={"biological_replicate": str}))
            receipts.append({"directory": str(path), "receipt_sha256": sha256(path / "complete.json")})
        dataset = manifest["dataset"]
        metrics = pd.concat(frames, ignore_index=True)
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
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{len(paths)}_repetitions"
        pending = root / "reports" / (".pending_" + name)
        pending.mkdir(parents=True)
        for filename, frame in (("metrics", metrics), ("primary_scores", primary), ("summary", summary),
                                ("paired_differences", differences), ("paired_summary", paired)):
            frame.to_csv(pending / (filename + ".tsv"), sep="\t", index=False)
        meta = {"kind": KIND, "dataset": dataset, "collection": str(root), "source_collection": manifest["source_collection"],
            "n_repetitions": len(paths), "replicate_ids": list(range(len(paths))), "primary_metric": metric,
            "preprocessing": "no downstream scaler or normalization in ANY representation",
            "embedding_dim": read_json(root / "model.json")["embedding_dim"], "pca": read_json(root / "pca.json"),
            "calibration_sha256": calibration_hash, "repetitions": receipts,
            "synthetic": manifest["settings"]["model_version"] == "synthetic", "report_code": sha256(Path(__file__))}
        write_json(pending / "report_metadata.json", meta)
        figure(summary, dataset, len(paths), pending / "paper_figure", meta["synthetic"])
        (pending / "paper_figure/CAPTION.md").write_text(caption(meta))
        text = [f"# {dataset}: corrected unscaled benchmark\n"]
        if meta["synthetic"]:
            text.append("**SYNTHETIC CPU FIXTURE — not a biological result.**\n")
        text += [f"**{len(paths)} complete paired repetitions** from this collection only. Primary metric: **{metric.upper()}**.\n",
                 caption(meta), "This collection recalibrates and refits every method. No old C/alpha, fitted probes, "
                 "predictions or metric rows were reused. Only verified, computation-identical cached features/PCA bases "
                 "may be shared. Fresh collections preserve historical collections; resume fixes calibration within this collection.\n",
                 "Full-data PCA is intentionally transductive, using full-depth held-out RNA without labels. "
                 "scUNVEIL has its external pretraining history. The PCA dimensionality is fixed, not selected to optimize test scores.\n",
                 f"![Paired comparison](paper_figure/{FIGURE_NAMES[dataset]}.png)\n",
                 "## Test performance: mean ± sample SD\n",
                 "| Task | UMI retention | Raw log1p RNA | PCA, unscaled PCs | scUNVEIL |",
                 "| --- | --- | --- | --- | --- |"]
        for (task, rate), part in summary.groupby(["task", "retention_rate"], sort=True):
            keyed = part.set_index("method")
            values = [f"{keyed.loc[m, 'mean']:.3f} ± {keyed.loc[m, 'sd']:.3f}" if len(paths) > 1
                      else f"{keyed.loc[m, 'mean']:.3f} (SD unavailable)" for m in ("raw_log1p", "pca_full", "scunveil")]
            text.append(f"| {task} | {rate:.0%} | " + " | ".join(values) + " |")
        text += ["\n## Calibration and interpretation\n",
            "`calibration/tuning.tsv`, `selected.json`, `grid_audit.tsv` and `folds.tsv.gz` in the collection "
            "record the objective, fold assignments, parameter search and boundary checks. Strictly improving "
            "grid endpoints expand using calibration data only; an unresolved improving boundary stops evaluation. "
            "Boundary gains no larger than 0.00001 objective units are recorded as plateaus rather than "
            "requiring an infinite-regularization limit. "
            "Exact ties favor stronger regularization. Edge plateaus are recorded without endless expansion.\n",
            "For ReSisTrace, AUROC uses pooled lineage-averaged direct probabilities per condition. T-cell AUROC "
            "averages scorable patients. Mitochondrial calibration holds out visits within one reserved subject; "
            "evaluation holds out subjects and averages visits within subjects. SCOPE-seq2 uses region R² with "
            "independent target penalties. Dev/test outcomes never choose hyperparameters.\n",
            "`metrics.tsv` includes native diagnostic scopes, shortcut controls and additional metrics. "
            "`summary.tsv` summarizes primary test scores. `paired_differences.tsv` and `paired_summary.tsv` "
            "report scUNVEIL minus each comparator. Positive differences favor scUNVEIL. SD across overlapping "
            "repetitions is not uncertainty over independent cohorts; one-time calibration and frozen weights remain fixed.\n",
            f"Collection: `{root}`. Original data/split provenance: `{manifest['source_collection']}`.\n",
            "Paper LaTeX and copied paper figures are not automatically updated.\n"]
        (pending / "REPORT.md").write_text("\n".join(text))
        seal(pending, {"kind": KIND, "collection": str(root), "n_repetitions": len(paths)})
        destination = root / "reports" / name
        pending.rename(destination)
        stable = root.parent.parent / "report"
        if stable.exists() and not stable.is_symlink():
            raise ValueError("Refusing to replace a real report directory")
        temporary = stable.with_name(".unscaled_report_" + str(os.getpid()))
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(destination, target_is_directory=True)
        temporary.replace(stable)
        write_json(root / "LATEST_REPORT.json", {"report": str(destination / "REPORT.md")})
        return stable
