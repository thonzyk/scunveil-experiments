"""Read only committed paired repetitions and publish an atomic stable report."""
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plot_paper_figure import build_figures
from tcr_common import (HERE, LABELS, METHODS, completed, lock, read_json, seal, sha256,
                        verify, write_frame, write_json)


def validate_metrics(frame, info, settings):
    keys = ["retention_rate", "split", "method", "scope", "patient"]
    if frame[keys].duplicated().any() or set(frame.repetition) != {info["repetition"]}:
        raise ValueError("Duplicate or mismatched repetition scores")
    expected = {(rate, role, method, scope, patient)
        for rate in settings["retention_rates"] for role in ("dev", "test") for method in METHODS
        for scope, patients in (("patient", info[f"{role}_patients"]),
                               ("macro_patient", ["ALL"]), ("pooled", ["ALL"])) for patient in patients}
    if set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("Incomplete paired repetition")
    for _, group in frame.groupby(["retention_rate", "split", "method"]):
        actual = group.loc[group.scope == "macro_patient", "auroc"].item()
        source = group.loc[group.scope == "patient", "auroc"].mean()
        if not np.isfinite(actual) or not np.isclose(actual, source):
            raise ValueError("Macro AUROC differs from equal-patient mean")


def build_report(root, max_repetitions=100):
    if max_repetitions is not None and (type(max_repetitions) is not int or max_repetitions < 1):
        raise ValueError("max_repetitions must be positive or None")
    root = Path(root).resolve()
    with lock(root / "report.lock"):
        manifest = read_json(root / "collection.json")
        repetitions = completed(root)
        available = len(repetitions)
        if max_repetitions is not None:
            repetitions = repetitions[:max_repetitions]
        if not repetitions:
            raise ValueError("No completed repetitions yet; collection may be caching/calibrating. Rerun reporting later.")
        n = len(repetitions)
        prepared = Path(read_json(root / "prepared.json")["directory"])
        verify(prepared)
        verify(root / "calibration")
        model = read_json(root / "model.json")
        if {info["embedding_dim"] for _, info in repetitions} != {model["embedding_dim"]}:
            raise ValueError("Mixed embedding widths")
        frames = []
        for path, info in repetitions:
            part = pd.read_csv(path / "metrics.tsv", sep="\t")
            validate_metrics(part, info, manifest["settings"])
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
                n_test_appearances=("repetition", "nunique"), n_scorable=("auroc", "count"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        parent = root / "reports"
        parent.mkdir(exist_ok=True)
        output = parent / f".pending-{timestamp}"
        output.mkdir()
        metadata = {"collection": str(root), "n_repetitions": n, "available_repetitions": available,
            "embedding_dim": model["embedding_dim"], "synthetic": model.get("model_version") == "synthetic",
            "selection": "first completed repetitions by index; never selected by performance",
            "report_code": {name: sha256(HERE / name) for name in ("report.py", "plot_paper_figure.py", "build_report.py")}}
        write_json(output / "report_metadata.json", metadata)
        for name, frame in (("metrics", metrics), ("summary", summary), ("paired_gains", paired),
                            ("paired_summary", paired_summary), ("per_patient", per_patient)):
            write_frame(output / f"{name}.tsv", frame)
        for name in ("measurement_audit", "fit_times"):
            write_frame(output / f"{name}.tsv", pd.concat([
                pd.read_csv(path / f"{name}.tsv", sep="\t").assign(repetition=info["repetition"])
                for path, info in repetitions], ignore_index=True))
        patients = pd.read_csv(root / "patient_partitions.tsv", sep="\t")
        write_frame(output / "patient_partitions.tsv", patients)
        write_frame(output / "patient_audit.tsv", pd.read_csv(prepared / "patients.tsv", sep="\t"))
        write_frame(output / "completed_repetitions.tsv", pd.DataFrame([
            {"repetition": info["repetition"], "elapsed_seconds": info["elapsed_seconds"],
             "receipt_sha256": sha256(path / "complete.json"),
             **{f"{role}_patients": ",".join(info[f"{role}_patients"]) for role in ("train", "dev", "test")}}
            for path, info in repetitions]))
        choices = read_json(root / "calibration/selected_c.json")
        tuning = pd.read_csv(root / "calibration/tuning.tsv", sep="\t")
        grid = []
        for rate in manifest["settings"]["retention_rates"]:
            for method, c in choices[f"keep_{float(rate):g}"].items():
                part = tuning[(tuning.retention_rate == rate) & (tuning.method == method) & (tuning.C == c)]
                grid.append({"retention_rate": rate, "method": method, "C": c,
                    "calibration_auroc": part.auroc.mean(), "at_grid_boundary": c in
                    (min(manifest["settings"]["c_values"]), max(manifest["settings"]["c_values"]))})
        write_frame(output / "calibration_grid_audit.tsv", pd.DataFrame(grid))
        build_figures(output)
        lines = ["# Tumor-reactive T cells: prediction under molecular undersampling", "",
            f"Collection `{root.name}`. **{n} paired repetitions reported** ({available} available). "
            "Only the first completed repetitions in collection order are included.", ""]
        if metadata["synthetic"]:
            lines += ["**SYNTHETIC CPU FIXTURE — software validation only.**", ""]
        calibration = patients.loc[patients.partition == "calibration", "patient"].tolist()
        lines += ["## Protocol", "",
            "Source: GSE254250, strict `Reactivity (n=118)` TR/NTR annotations. Real-data release: "
            "118 tested clonotypes (63 TR, 55 NTR), 3990 cells, nine patients; seven patients contain both classes. "
            "Reactivity was assayed against autologous tumor cells. TCR sequences, assay readouts beyond the label, "
            "patient IDs and source cluster assignments are not classifier features.", "",
            "Fixed calibration patients: **" + ", ".join(calibration) + "**. "
            "Whole-patient combinations are selected by proximity to the configured calibration clonotype "
            "budget (20% by default), among patients with sufficient tested clones of each class; "
            "seeded draws break count-based ties. No model performance enters this selection. "
            "They are excluded from every repetition. One fixed random cell per calibration clonotype and "
            "fixed calibration molecule draws support leave-one-calibration-patient-out C selection, separately "
            "per method and rate, by mean within-patient AUROC. Ties favor smaller C. These selection scores "
            "are optimistic calibration diagnostics, not independent test evidence.", "",
            f"Each repetition draws {manifest['settings']['test_patients']} test patients and "
            f"{manifest['settings']['dev_patients']} dev patient(s) from the remaining pool; the rest train. "
            "Each evaluation role must contain at least one mixed-label patient; training must contain both classes. "
            "Assignments are drawn using these label-availability constraints, never prediction performance. "
            "Dev predictions are diagnostic and never used to refit or retune. Entire patients and clonotypes stay together.", "",
            "One new random cell is selected per clonotype in all roles. That cell selection and patient split "
            "are shared across rates. Every rate/repetition has an independent binomial molecule draw shared "
            "across methods. Empty diluted cells remain included; functional labels stay fixed. Full-depth "
            "frozen embeddings are cached, but selected cells, training-only scalers and probes are refreshed.", "",
            f"scUNVEIL uses **all {model['embedding_dim']} native features**. Raw RNA uses sparse log1p counts "
            "without library normalization or fitted PCA. RNA summary uses log1p total UMIs and detected genes "
            "from the same sampled cell. Clonotype size uses the original number of strict-label cells and "
            "therefore accesses clone-abundance metadata beyond the single selected cell. It is a shortcut diagnostic. "
            "Probes use sklearn class-balanced L2 logistic regression on CPU, one training example per clonotype, "
            "with training-only variance scaling. GPU accelerates only frozen encoder inference.", "",
            "## Held-out test results", "",
            "Primary AUROC is calculated within each mixed-label test patient, then averaged equally within "
            "each repetition. Single-label test patients are preserved in predictions and pooled metrics, but "
            "their within-patient AUROC is undefined. See `per_patient.tsv` for test coverage and individual results.", "",
            "![Within-patient AUROC](paper_figure/tumor_reactivity_auroc.png)", "",
            "| UMI retention | Raw log1p RNA, mean ± SD | scUNVEIL, mean ± SD | Paired AUROC gain ± SD |",
            "|---|---:|---:|---:|"]
        def value(mean, sd):
            return f"{mean:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{mean:.3f} (SD unavailable)"
        primary = summary[(summary.split == "test") & (summary.scope == "macro_patient")]
        for rate in sorted(primary.retention_rate.unique(), reverse=True):
            part = primary[primary.retention_rate == rate].set_index("method")
            gain = paired_summary[(paired_summary.retention_rate == rate) & (paired_summary.split == "test") &
                                  (paired_summary.scope == "macro_patient") & (paired_summary.baseline == "raw_log1p")].iloc[0]
            raw, scu = part.loc["raw_log1p"], part.loc["scunveil"]
            lines.append(f"| {rate:.0%} | {value(raw.mean_auroc, raw.sd_auroc)} | "
                         f"{value(scu.mean_auroc, scu.sd_auroc)} | {value(gain.mean_gain, gain.sd_gain)} |")
        lines += ["", "| UMI retention | Representation | Within-patient AUROC, mean ± SD |",
                  "|---|---|---:|"]
        for row in primary.itertuples():
            lines.append(f"| {row.retention_rate:.0%} | {LABELS[row.method]} | {value(row.mean_auroc, row.sd_auroc)} |")
        lines += ["", "## Interpretation and limitations", "",
            "- Check full-depth prediction, raw-RNA deterioration with dilution, and paired scUNVEIL gains. "
            "Inspect RNA-summary and clonotype-size baselines before attributing gains to a richer cell state.",
            "- Error bars are sample SD over fresh patient assignments, selected cells, molecular draws and fitted "
            "probes, conditional on fixed calibration patients/C and a frozen encoder. Overlapping repetitions "
            "are not independent biological experiments or confidence intervals for new patients.",
            "- Only nine source patients exist; calibration reservation further reduces the test pool. "
            "Patient-specific AUROC can be coarse with few tested clonotypes. Test appearances are unequal; "
            "inspect per_patient.tsv. More repetitions cannot supply missing biological replication.",
            "- The label belongs to the tested receptor/clonotype. This does not directly measure killing by "
            "the sequenced cell, nor does RNA alone determine TCR binding specificity. Only experimentally "
            "tested clonotypes are included, introducing selection bias.",
            "- Patients TIPC301, TIPC413 and TIPC416 share a source sequencing library. Patient-disjoint "
            "splits do not establish sequencing-library-disjoint generalization. Cell-state/composition, "
            "clonal expansion and pretraining overlap remain possible explanations to investigate.",
            "- Full-depth results also vary because cell selection and patient partitions change. At 100% "
            "retention molecule draws are identities. Repeated assignment combinations can recur naturally.",
            "- Binomial thinning models molecular undersampling, not all low-quality assay effects. "
            "This protocol differs from archived all-cell/512-feature results and published predicTCR training; "
            "those scores are not matched comparisons.", "",
            "## Provenance", "",
            "[GSE254250](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE254250); "
            "[Meng et al.](https://doi.org/10.1126/scitranslmed.adh9562). Prepared counts, complete paired "
            "repetitions, calibration and report snapshots have checksum receipts. Summary/AP/pooled results, "
            "patient coverage, actual measurement depths, fit times and C-grid diagnostics are retained as TSVs. "
            "No test score gates whether collection continues.", ""]
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
