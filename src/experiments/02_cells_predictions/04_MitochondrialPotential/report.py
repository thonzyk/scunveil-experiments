"""Summarize committed paired repetitions; atomically replace the stable report link."""
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plot_paper_figure import build_figures
from tmre_common import (HERE, LABELS, METHODS, completed, lock, read_json, seal, sha256,
                         verify, write_frame, write_json)


def validate_metrics(frame, info, settings):
    keys = ["retention_rate", "split", "method", "scope", "patient", "visit"]
    if frame[keys].duplicated().any() or set(frame.repetition) != {info["repetition"]}:
        raise ValueError("Duplicate or mismatched repetition scores")
    expected = set()
    for rate in settings["retention_rates"]:
        for role in ("dev", "test"):
            for method in METHODS:
                for patient in info[f"{role}_patients"]:
                    for visit in ("pre", "post"):
                        expected.add((rate, role, method, "visit", patient, visit))
                    expected.add((rate, role, method, "patient", patient, "ALL"))
                for scope in ("macro_patient", "pooled"):
                    expected.add((rate, role, method, scope, "ALL", "ALL"))
    if set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("Incomplete paired repetition")
    if not np.isfinite(frame[["auroc", "average_precision"]].to_numpy()).all():
        raise ValueError("Nonfinite outcome scores")
    for _, group in frame.groupby(["retention_rate", "split", "method"], observed=True):
        patient_rows = group[group.scope == "patient"]
        for row in patient_rows.itertuples():
            actual = group[(group.scope == "visit") & (group.patient == row.patient)].auroc.mean()
            if not np.isclose(actual, row.auroc):
                raise ValueError("Subject AUROC differs from equal-visit mean")
        actual = group.loc[group.scope == "macro_patient", "auroc"].item()
        if not np.isclose(actual, patient_rows.auroc.mean()):
            raise ValueError("Macro AUROC differs from equal-subject mean")


def format_value(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{mean:.3f} (SD unavailable)"


def build_report(root, max_repetitions=100):
    if max_repetitions is not None and (type(max_repetitions) is not int or max_repetitions < 1):
        raise ValueError("max_repetitions must be positive or None")
    root = Path(root).resolve()
    with lock(root / "report.lock"):
        manifest = read_json(root / "collection.json")
        settings = manifest["settings"]
        repetitions = completed(root)
        available = len(repetitions)
        if max_repetitions is not None:
            repetitions = repetitions[:max_repetitions]
        if not repetitions:
            raise ValueError("No completed repetitions yet; startup/calibration or the first repetition may still be running")
        n = len(repetitions)
        prepared = Path(read_json(root / "prepared.json")["directory"])
        source = verify(prepared)
        verify(root / "calibration")
        model = read_json(root / "model.json")
        if {info["embedding_dim"] for _, info in repetitions} != {model["embedding_dim"]}:
            raise ValueError("Mixed embedding widths")
        frames = []
        for path, info in repetitions:
            part = pd.read_csv(path / "metrics.tsv", sep="\t")
            validate_metrics(part, info, settings)
            frames.append(part)
        metrics = pd.concat(frames, ignore_index=True)
        aggregate = metrics[metrics.scope.isin(["macro_patient", "pooled"])].copy()
        summary = aggregate.groupby(["retention_rate", "split", "scope", "method"], as_index=False).agg(
            mean_auroc=("auroc", "mean"), sd_auroc=("auroc", "std"),
            mean_ap=("average_precision", "mean"), sd_ap=("average_precision", "std"),
            n_repetitions=("repetition", "nunique"))
        wide = aggregate.pivot(index=["repetition", "retention_rate", "split", "scope"],
                               columns="method", values="auroc")
        paired = pd.concat([(wide.scunveil - wide[m]).rename("gain").reset_index().assign(baseline=m)
                            for m in METHODS if m != "scunveil"], ignore_index=True)
        paired_summary = paired.groupby(["retention_rate", "split", "scope", "baseline"], as_index=False).agg(
            mean_gain=("gain", "mean"), sd_gain=("gain", "std"),
            wins=("gain", lambda x: int((x > 0).sum())), n_repetitions=("repetition", "nunique"))
        per_patient = metrics[(metrics.scope == "patient") & (metrics.split == "test")].groupby(
            ["patient", "retention_rate", "method"], as_index=False).agg(
                mean_auroc=("auroc", "mean"), sd_auroc=("auroc", "std"),
                n_test_appearances=("repetition", "nunique"))
        per_visit = metrics[(metrics.scope == "visit") & (metrics.split == "test")].groupby(
            ["patient", "visit", "retention_rate", "method"], as_index=False).agg(
                mean_auroc=("auroc", "mean"), sd_auroc=("auroc", "std"),
                n_test_appearances=("repetition", "nunique"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        parent = root / "reports"
        parent.mkdir(exist_ok=True)
        output = parent / f".pending-{timestamp}"
        output.mkdir()
        metadata = {"collection": str(root), "n_repetitions": n, "available_repetitions": available,
            "embedding_dim": model["embedding_dim"], "synthetic": model.get("model_version") == "synthetic",
            "selection": "first completed repetitions by index; never selected by performance",
            "primary_metric": "mean of within-visit AUROCs, first within subject then equally across test subjects",
            "report_code": {name: sha256(HERE / name) for name in ("report.py", "plot_paper_figure.py", "build_report.py")}}
        write_json(output / "report_metadata.json", metadata)
        for name, frame in (("metrics", metrics), ("summary", summary), ("paired_gains", paired),
                            ("paired_summary", paired_summary), ("per_patient", per_patient), ("per_visit", per_visit)):
            write_frame(output / f"{name}.tsv", frame)
        for name in ("measurement_audit", "fit_times"):
            write_frame(output / f"{name}.tsv", pd.concat([
                pd.read_csv(path / f"{name}.tsv", sep="\t").assign(repetition=info["repetition"])
                for path, info in repetitions], ignore_index=True))
        patients = pd.read_csv(root / "patient_partitions.tsv", sep="\t")
        write_frame(output / "patient_partitions.tsv", patients)
        write_frame(output / "library_audit.tsv", pd.read_csv(prepared / "library_audit.tsv", sep="\t"))
        write_frame(output / "completed_repetitions.tsv", pd.DataFrame([
            {"repetition": info["repetition"], "elapsed_seconds": info["elapsed_seconds"],
             "receipt_sha256": sha256(path / "complete.json"),
             **{f"{role}_patients": ",".join(info[f"{role}_patients"]) for role in ("train", "dev", "test")}}
            for path, info in repetitions]))
        choices = read_json(root / "calibration/selected_c.json")
        tuning = pd.read_csv(root / "calibration/tuning.tsv", sep="\t")
        grid = []
        for rate in settings["retention_rates"]:
            for method, c in choices[f"keep_{float(rate):g}"].items():
                part = tuning[(tuning.retention_rate == rate) & (tuning.method == method) & (tuning.C == c)]
                grid.append({"retention_rate": rate, "method": method, "C": c,
                    "calibration_auroc": part.auroc.mean(), "at_grid_boundary": c in
                    (min(settings["c_values"]), max(settings["c_values"]))})
        grid = pd.DataFrame(grid)
        write_frame(output / "calibration_grid_audit.tsv", grid)
        # Calibration-only feasibility diagnostic; never a stopping/selection rule based on test scores.
        cal_raw = grid[grid.method == "raw_log1p"].set_index("retention_rate").calibration_auroc
        diagnostic = pd.DataFrame([{"retention_rate": rate, "raw_calibration_auroc": score,
            "drop_from_full_depth": cal_raw.loc[1.0] - score} for rate, score in cal_raw.items()])
        write_frame(output / "calibration_screen.tsv", diagnostic)
        mapping_path = root / "gene_mapping_first_batch.json"
        if mapping_path.exists():
            write_json(output / mapping_path.name, read_json(mapping_path))
        build_figures(output)
        cal = patients.loc[patients.partition == "calibration", "patient"].item()
        lines = ["# Mitochondrial potential from sparse RNA", "",
            f"Collection `{root.name}`. **{n} paired repetitions reported** ({available} available). "
            "First completed repetitions in collection order; no result-based selection.", ""]
        if metadata["synthetic"]:
            lines += ["**SYNTHETIC CPU FIXTURE — software validation only; no biological result.**", ""]
        lines += ["## Protocol", "",
            "Target: independently FACS-measured TMRE-high (1) versus TMRE-low (0), in peripheral-blood "
            "CD3+ T cells from the five lung-cancer subjects in GSE315510. Twenty source libraries comprise "
            "both gates at both pre-/post-treatment visits. Kidney libraries are excluded because their "
            "donor-demultiplexing map was not verified. Subject keys use response category and the Patient "
            "suffix in GEO titles as cohort-local identifiers only; no response label or identity enters a probe.", "",
            f"Prepared cohort: {source['n_cells']:,} cells, {source['n_genes']:,} RNA features. "
            f"Original-count eligibility: {settings['min_umis']}–{settings['max_umis']} UMIs, "
            f"{settings['min_genes']}–{settings['max_genes']} detected genes and mitochondrial fraction "
            f"≤{settings['max_mito_fraction']:.0%}. Eligibility is fixed before thinning. These are original "
            "CellRanger counts: the source paper's CellBender, Scrublet, TCR filters and gene removal are "
            "not reproduced. No normalization, PCA or label-derived gene filtering is applied.", "",
            f"Permanently reserved calibration subject: **{cal}**. It is selected uniformly with a fixed "
            "seed, before observing model scores. One fixed balanced cell sample and fixed molecule draws "
            "support C selection by training on pre and validating on post, then reversing visits. "
            "Choose C separately per method/rate by mean held-out-visit AUROC; ties prefer smaller C. "
            "This is within-subject calibration, not held-out-subject validation. Its scores are optimistic "
            "tuning diagnostics and do not enter the main test evidence. Calibration cells are never in a main repetition.", "",
            "Each repetition assigns the other four subjects to two training, one development and one test "
            "subject. All visits, gates and libraries from a subject stay together. Development predictions "
            "are diagnostic; they never revise C or enter training. Each subject/visit contributes equal "
            f"numbers of high/low cells, up to {settings['cells_per_gate_visit']} per gate, freshly sampled "
            "without replacement. The smaller available gate sets the paired count. This is at most "
            f"{4 * 4 * settings['cells_per_gate_visit']:,} cells per main repetition. Training weights give "
            "each subject/visit/gate equal total weight.", "",
            "The same selected cells and subject roles are used across retention rates. Independent binomial "
            "draws at 100%, 10% and 1% retention are paired across all methods in all roles. At 100%, "
            "counts are unchanged; full-depth frozen embeddings are cached only as selected cells are used. "
            "Diluted embeddings are regenerated from fresh counts each repetition. Cells with no retained "
            "molecules remain included. Every repetition fits fresh training-only scalers and CPU sklearn "
            "L2 logistic regressions. GPU accelerates encoder inference only; encoder weights and C stay fixed.", "",
            f"scUNVEIL uses all **{model['embedding_dim']}** native features. The sole main RNA baseline is "
            "sparse raw log1p counts. RNA summary uses log1p total UMIs and detected genes; the additional "
            "control adds mitochondrial UMI fraction, defined as zero for empty cells. Every control uses "
            "the same sampled molecules. No original-depth measurement is passed into a diluted probe.", "",
            "## Held-out test results", "",
            "Primary AUROC is computed within each test subject/visit, then averaged equally over visits "
            "and subjects. This prevents treatment-time composition from driving a pooled score. Each visit "
            "has balanced sampled high/low gates; AP therefore describes this artificial 50% prevalence, "
            "not the natural population prevalence. Pooled AUROC/AP are retained as secondary diagnostics.", "",
            "![Mitochondrial-potential AUROC](paper_figure/mitochondrial_potential_auroc.png)", "",
            "| UMI retention | Raw log1p RNA, mean ± SD | scUNVEIL, mean ± SD | Paired AUROC gain ± SD |",
            "|---|---:|---:|---:|"]
        primary = summary[(summary.split == "test") & (summary.scope == "macro_patient")]
        for rate in sorted(primary.retention_rate.unique(), reverse=True):
            part = primary[primary.retention_rate == rate].set_index("method")
            gain = paired_summary[(paired_summary.retention_rate == rate) & (paired_summary.split == "test") &
                (paired_summary.scope == "macro_patient") & (paired_summary.baseline == "raw_log1p")].iloc[0]
            raw, scu = part.loc["raw_log1p"], part.loc["scunveil"]
            lines.append(f"| {rate:.0%} | {format_value(raw.mean_auroc, raw.sd_auroc)} | "
                         f"{format_value(scu.mean_auroc, scu.sd_auroc)} | {format_value(gain.mean_gain, gain.sd_gain)} |")
        lines += ["", "| UMI retention | Diagnostic control | AUROC, mean ± SD |", "|---|---|---:|"]
        for row in primary[primary.method.isin(["depth", "depth_mito"])].itertuples():
            lines.append(f"| {row.retention_rate:.0%} | {LABELS[row.method]} | {format_value(row.mean_auroc, row.sd_auroc)} |")
        lines += ["", "## How to interpret", "",
            "1. Establish that full-depth raw RNA predicts the external TMRE label and that dilution reduces "
            "its accuracy. A strong raw result with negligible deterioration may leave little room for transfer.",
            "2. Inspect paired gains over raw RNA and both shortcut controls, with per-subject and per-visit "
            "results. A mean advantage driven by one subject, one visit or RNA-depth differences is weak evidence.",
            "3. Read calibration_screen.tsv as a tuning diagnostic only. No calibration or test score gates "
            "whether a repetition is collected. Calibration-grid boundary choices warrant inspection.", "",
            "## Limits", "",
            "- TMRE labels mitochondrial polarization, not ATP output, metabolic flux, mitochondrial "
            "health or clinical response. High potential can accompany dysfunction.",
            "- Only four non-calibration subjects supply test evidence. SD reflects overlapping subject "
            "assignments, cell sampling, thinning and fresh fits, conditional on fixed calibration/C/weights; "
            "it is not a confidence interval over independent biological experiments. Test appearances vary. "
            "Consult per_patient.tsv and per_visit.tsv; repeated partitions naturally recur.",
            "- The target is a high/low sort category with sample-specific thresholds, not quantitative per-cell "
            "fluorescence. Each gate has a separate source library. Subject-disjoint evaluation reduces "
            "memorization of libraries, but systematic gate-associated processing differences remain possible.",
            "- RNA-defined T-cell subset composition and activation may explain prediction. No independent "
            "per-cell subset annotation was verified; this experiment does not establish prediction within "
            "a fixed T-cell state. Source sorting, fixed RNA QC, gate balancing and cell caps define the "
            "evaluated population. Clonotypes are unavailable; related cells within a subject are not "
            "independent evidence. Whole-subject separation prevents within-patient clonal leakage across roles.",
            "- Source-author filtering is not reproduced, and original versus published analyzed cell totals "
            "can differ. Binomial thinning models undersampling, not all assay defects. Pretraining overlap "
            "and superiority over other pretrained embeddings have not been audited.", "",
            "## Outputs and provenance", "",
            "Snapshots are built only from checksum-verified committed repetitions. The stable report link "
            "is replaced atomically and can be rebuilt while collection continues. Incomplete pending work "
            "is ignored. Each repetition stores selected cells/roles, predictions, measurement hashes, "
            "fit times, fixed C values, seeds and a checksum receipt. No paper integration is performed.", "",
            "Source: [GSE315510](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE315510); "
            "[Lindau et al., 2026](https://doi.org/10.1093/jimmun/vkag120).", ""]
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
