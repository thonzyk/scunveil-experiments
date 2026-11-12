"""Validate committed paired repetitions and atomically replace the stable report link."""
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plot_paper_figure import build_figures
from scope_common import (HERE, LABELS, METHODS, completed, lock, read_json, seal, sha256,
                          verify, write_frame, write_json)

TARGET_LABELS = {"c2_Area": "Cell area", "c2_Circ.": "Circularity"}


def validate_metrics(frame, info, settings):
    keys = ["retention_rate", "split", "method", "target", "scope", "region"]
    if frame[keys].duplicated().any() or set(frame.repetition) != {info["repetition"]}:
        raise ValueError("Duplicate or mismatched repetition metrics")
    expected = {(rate, role, method, target, scope, region)
        for rate in settings["retention_rates"] for role in ("dev", "test") for method in METHODS
        for target in settings["targets"]
        for scope, regions in (("region", info[f"{role}_regions"]), ("macro_region", ["ALL"]), ("pooled", ["ALL"]))
        for region in regions}
    if set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("Incomplete paired repetition")
    if not np.isfinite(frame[["r2", "mae"]].to_numpy()).all():
        raise ValueError("Non-finite primary metrics")
    for _, group in frame.groupby(["retention_rate", "split", "method", "target"]):
        if not np.isclose(group.loc[group.scope == "macro_region", "r2"].item(),
                          group.loc[group.scope == "region", "r2"].mean()):
            raise ValueError("Macro R2 differs from equal-region mean")


def value(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{mean:.3f} (SD unavailable)"


def build_report(root, max_repetitions=100):
    if max_repetitions is not None and (type(max_repetitions) is not int or max_repetitions < 1):
        raise ValueError("max_repetitions must be positive or None")
    root = Path(root).resolve()
    # One report writer per dataset; the collector has a separate lock.
    with lock(root.parent.parent / "report.lock"):
        manifest = read_json(root / "collection.json")
        settings = manifest["settings"]
        repetitions = completed(root)
        available = len(repetitions)
        if max_repetitions is not None:
            repetitions = repetitions[:max_repetitions]
        if not repetitions:
            raise ValueError("No completed repetitions yet; wait for caching/calibration and the first repetition")
        n = len(repetitions)
        prepared = Path(read_json(root / "prepared.json")["directory"])
        verify(prepared)
        verify(root / "calibration")
        verify(root / "full_depth_cache")
        model = read_json(root / "model.json")
        if {info["embedding_dim"] for _, info in repetitions} != {model["embedding_dim"]}:
            raise ValueError("Mixed embedding widths")
        frames = []
        for path, info in repetitions:
            part = pd.read_csv(path / "metrics.tsv", sep="\t", dtype={"region": str})
            validate_metrics(part, info, settings)
            frames.append(part)
        metrics = pd.concat(frames, ignore_index=True)
        aggregate = metrics[metrics.scope.isin(["macro_region", "pooled"])].copy()
        summary = aggregate.groupby(["target", "retention_rate", "split", "scope", "method"], as_index=False).agg(
            mean_r2=("r2", "mean"), sd_r2=("r2", "std"), mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"), mean_pearson=("pearson", "mean"), sd_pearson=("pearson", "std"),
            mean_mae=("mae", "mean"), sd_mae=("mae", "std"), n_repetitions=("repetition", "nunique"))
        wide = aggregate.pivot(index=["repetition", "target", "retention_rate", "split", "scope"],
                               columns="method", values="r2")
        paired = pd.concat([(wide.scunveil - wide[m]).rename("gain").reset_index().assign(baseline=m)
                            for m in METHODS if m != "scunveil"], ignore_index=True)
        paired_summary = paired.groupby(["target", "retention_rate", "split", "scope", "baseline"], as_index=False).agg(
            mean_gain=("gain", "mean"), sd_gain=("gain", "std"), wins=("gain", lambda x: int((x > 0).sum())),
            n_repetitions=("repetition", "nunique"))
        per_region = metrics[(metrics.scope == "region") & (metrics.split == "test")].groupby(
            ["region", "target", "retention_rate", "method"], as_index=False).agg(
            mean_r2=("r2", "mean"), sd_r2=("r2", "std"), n_test_appearances=("repetition", "nunique"))
        balanced = per_region.groupby(["target", "retention_rate", "method"], as_index=False).agg(
            equal_region_mean_r2=("mean_r2", "mean"), n_test_regions=("region", "nunique"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        parent = root / "reports"
        parent.mkdir(exist_ok=True)
        output = parent / f".pending-{timestamp}"
        output.mkdir()
        regions = pd.read_csv(root / "region_partitions.tsv", sep="\t", dtype={"region": str})
        metadata = {"collection": str(root), "n_repetitions": n, "available_repetitions": available,
            "calibration_regions": int((regions.partition == "calibration").sum()),
            "pool_regions": int((regions.partition == "pool").sum()),
            "test_regions": settings["test_regions"], "dev_regions": settings["dev_regions"],
            "embedding_dim": model["embedding_dim"], "synthetic": model.get("model_version") == "synthetic",
            "selection": "first completed repetitions by index; never selected by prediction performance",
            "report_code": {name: sha256(HERE / name) for name in ("report.py", "plot_paper_figure.py", "build_report.py")}}
        write_json(output / "report_metadata.json", metadata)
        for name, frame in (("metrics", metrics), ("summary", summary), ("paired_gains", paired),
            ("paired_summary", paired_summary), ("per_region", per_region), ("region_balanced_summary", balanced)):
            write_frame(output / f"{name}.tsv", frame)
        for name in ("measurement_audit", "fit_times"):
            write_frame(output / f"{name}.tsv", pd.concat([
                pd.read_csv(path / f"{name}.tsv", sep="\t", dtype={"region": str}).assign(repetition=info["repetition"])
                for path, info in repetitions], ignore_index=True))
        write_frame(output / "region_partitions.tsv", regions)
        write_json(output / "source.json", read_json(prepared / "source.json"))
        write_json(output / "model.json", model)
        write_json(output / "gene_mapping.json", read_json(root / "full_depth_cache/complete.json")["gene_mapping"])
        write_json(output / "selected_alpha.json", read_json(root / "calibration/selected_alpha.json"))
        tuning = pd.read_csv(root / "calibration/tuning.tsv", sep="\t", dtype={"held_out_region": str})
        write_frame(output / "calibration_tuning.tsv", tuning)
        selected = read_json(root / "calibration/selected_alpha.json")
        write_frame(output / "calibration_grid_audit.tsv", pd.DataFrame([
            {"retention_rate": r, "method": m, "alpha": selected[f"keep_{float(r):g}"][m],
             "at_grid_boundary": selected[f"keep_{float(r):g}"][m] in
                (min(settings["alpha_values"]), max(settings["alpha_values"]))}
            for r in settings["retention_rates"] for m in METHODS]))
        write_frame(output / "completed_repetitions.tsv", pd.DataFrame([
            {"repetition": info["repetition"], "elapsed_seconds": info["elapsed_seconds"],
             "receipt_sha256": sha256(path / "complete.json"),
             **{f"{role}_regions": ",".join(info[f"{role}_regions"]) for role in ("train", "dev", "test")}}
            for path, info in repetitions]))
        build_figures(output)
        lines = ["# SCOPE-seq2: cell morphology from sparse RNA", "",
            f"Collection `{root.name}`. **{n} paired repetitions reported** ({available} available). "
            "First completed repetitions by index; no performance-based selection.", ""]
        if metadata["synthetic"]:
            lines += ["**SYNTHETIC CPU FIXTURE — software validation only; not biological results.**", ""]
        calibration = regions.loc[regions.partition == "calibration", "region"].tolist()
        n_cal = int(regions.loc[regions.partition == "calibration", "n_cells"].sum())
        lines += ["## What is being predicted?", "",
            "Two independently imaged properties of the same cell: **cell area** (size) and **circularity** "
            "(shape). These are continuous targets, so the primary metric is **R²**, not AUROC. R²=1 is "
            "perfect; 0 is the score of predicting the evaluated region's mean; negative values indicate "
            "worse squared error than that reference. Spearman/Pearson correlations and MAE are secondary.", "",
            "Real-data release: 956 cells with an exact region_S_Q barcode join between human GBM "
            "UMI counts (GSM4567140/PJ069) and SourceDataForFigure4C. All 60,725 measured gene rows "
            "are retained. Imaging targets were globally z-scored by the original authors; R² and "
            "correlations are invariant to this fixed affine transformation. MAE is in published standardized "
            "units, not square micrometres. Other image columns are retained for provenance, not predictors.", "",
            "## Protocol", "",
            f"Fixed calibration regions: **{', '.join(calibration)}** ({n_cal} cells). "
            f"Whole-region combinations are chosen solely by proximity to a {settings['calibration_target_fraction']:.0%} cell-count budget, with "
            "seeded tie-breaking. These regions are excluded from every main repetition. One-time "
            "leave-one-calibration-region-out tuning uses fixed molecule draws. Ridge alpha is chosen "
            "separately per representation/rate, by mean R² with equal weights for both targets and calibration "
            "regions. Larger alpha breaks exact ties. Calibration scores are tuning diagnostics, not test evidence.", "",
            f"Every repetition randomly assigns {settings['test_regions']} pool regions to test and "
            f"{settings['dev_regions']} to dev; the other pool regions train. "
            "All cells in a region remain together. These are technical regions of one device "
            "from **one patient**, not donors or independent biological replicates. Dev predictions are "
            "diagnostic: they never choose alpha, refit the model, select repetitions or gate collection.", "",
            "Cell eligibility and microscopy targets are fixed before thinning. Each rate/repetition has "
            "a fresh binomial molecule draw shared across all representations, applied to train, dev and "
            "test. The same region assignment is used at all rates. Empty diluted cells remain included. "
            "At 100% the count draw is an identity, so variability comes from region splits and refitting. "
            "Full-depth frozen embeddings are cached; shallow embeddings, training scalers and regressors "
            "are freshly computed for each repetition.", "",
            f"scUNVEIL uses **all {model['embedding_dim']} native features** of the pinned frozen checkpoint. "
            "Raw RNA is sparse log1p integer counts without library normalization, HVG selection or PCA. "
            "Both use training-only variance scaling followed by sklearn ridge regression with an intercept. "
            "Sparse RNA is not centered, which the intercept accommodates. Ridge regularization is shared "
            "across the two independently fitted output heads. GPU accelerates the encoder only; regressors "
            "run on CPU. The RNA-summary control uses log1p total UMIs and log1p detected genes from "
            "the same diluted cell, to diagnose whether size/shape prediction is largely RNA-depth information.", "",
            "## Held-out test results", "",
            "Primary R² is calculated within each held-out technical region, then averaged equally per "
            "repetition. Pooled R² is secondary. `per_region.tsv` and `region_balanced_summary.tsv` show "
            "test coverage and guard against unequal region appearances in finite random repetitions.", "",
            "![Morphology prediction](paper_figure/morphology_r2.png)", "",
            "| Target | UMI retention | Raw log1p RNA, mean ± SD | scUNVEIL, mean ± SD | Paired R² gain ± SD |",
            "|---|---:|---:|---:|---:|"]
        primary = summary[(summary.split == "test") & (summary.scope == "macro_region")]
        for target in settings["targets"]:
            for rate in sorted(settings["retention_rates"], reverse=True):
                part = primary[(primary.target == target) & (primary.retention_rate == rate)].set_index("method")
                gain = paired_summary[(paired_summary.target == target) & (paired_summary.retention_rate == rate) &
                    (paired_summary.split == "test") & (paired_summary.scope == "macro_region") &
                    (paired_summary.baseline == "raw_log1p")].iloc[0]
                raw, scu = part.loc["raw_log1p"], part.loc["scunveil"]
                lines.append(f"| {TARGET_LABELS[target]} | {rate:.0%} | {value(raw.mean_r2, raw.sd_r2)} | "
                    f"{value(scu.mean_r2, scu.sd_r2)} | {value(gain.mean_gain, gain.sd_gain)} |")
        lines += ["", "## RNA-depth shortcut diagnostic", "",
            "| Target | UMI retention | Representation | Test R², mean ± SD |", "|---|---:|---|---:|"]
        for row in primary.itertuples():
            lines.append(f"| {TARGET_LABELS[row.target]} | {row.retention_rate:.0%} | {LABELS[row.method]} | "
                         f"{value(row.mean_r2, row.sd_r2)} |")
        lines += ["", "## Interpretation and limitations", "",
            "- First check whether full-depth RNA predicts the endpoint at all. Then inspect how raw RNA "
            "and scUNVEIL change at 10% and 1%, and the paired differences. Relative percentage gains in "
            "R² can mislead near zero or with negative scores, so the report uses absolute paired gains.",
            "- If RNA summary matches scUNVEIL, the task may mostly measure transcript abundance or size "
            "rather than information in a distributed cell-state program. Area is particularly exposed to "
            "this shortcut; circularity is a complementary endpoint, not a guarantee against confounding.",
            "- Cell type, cell cycle, viability, dissociation and imaging artifacts can explain associations. "
            "This benchmark does not establish prediction within a fixed cell type, a mechanism, or causality. "
            "No expression-derived cell-type or malignancy label is used to manufacture the imaging target.",
            "- Sample SD summarizes repeat technical-region assignments, molecule draws and fresh fitted "
            "heads, conditional on fixed calibration regions, alpha and encoder weights. Repetitions overlap; "
            "error bars are not confidence intervals or independent biological replication. No held-out-patient "
            "claim is possible from this one patient, regardless of repetition count.",
            "- Source-QC cells are selected before dilution and remain fixed. Original imaging linkage is "
            "imperfect; the published source table has fewer cells than the 1,110 initially linked images. "
            "Binomial thinning models molecule sampling, not all effects of a poor assay. Pretraining overlap "
            "and platform-specific capture biases remain unverified.", "",
            "## Provenance", "",
            "[Liu et al., SCOPE-seq2](https://doi.org/10.1038/s41598-020-76599-w); "
            "[GSM4567140](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4567140). "
            "Sources, count-to-image joins, checkpoint and inference code, calibration, committed repetitions "
            "and report snapshots are recorded with checksums. Pending repetitions are ignored, and "
            "reporting can be repeated while the collector runs. No experiment-wide test score gates collection.", ""]
        (output / "REPORT.md").write_text("\n".join(lines))
        seal(output, {"n_repetitions": n, "replicate_ids": [info["repetition"] for _, info in repetitions],
                      "collection": str(root)})
        destination = parent / f"{timestamp}_{n}_repetitions"
        output.rename(destination)
        stable = root.parent.parent / "report"
        temporary = stable.with_name(f".report-{timestamp}")
        temporary.symlink_to(destination, target_is_directory=True)
        os.replace(temporary, stable)
        write_json(root / "LATEST_REPORT.json", {"report": str(destination / "REPORT.md"), "n_repetitions": n})
        return stable
