#!/usr/bin/env python3
"""Unscaled, paired benchmarks with collection-independent feature caching."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter
import uuid
import warnings

from pca_comparison import (Adapter, DATASETS, compare_frames, completed, digest, lock,
                            read_json, seal, sha256, verify, write_json)
from checkpoint_validation import computation_code_hash

HERE = Path(__file__).resolve().parent
KIND = "unscaled_benchmark_v2"
DEFAULT_GRIDS = {"c_values": [10. ** i for i in range(-7, 4)],
                 "alpha_values": [10. ** i for i in range(-3, 8)], "max_grid_extensions": 8,
                 "boundary_score_tolerance": 1e-5}


def resolve_source(output, explicit=None):
    """Follow provenance links to the original prepared-data/split protocol."""
    output = Path(output)
    if explicit:
        root = Path(explicit).resolve()
    elif (output / "report").is_symlink():
        root = (output / "report").resolve().parent.parent
    else:
        candidates = [p.parent for p in (output / "collections").glob("*/collection.json")
                      if read_json(p).get("kind") is None]
        if len(candidates) != 1:
            raise ValueError(f"Need one original collection or a stable report in {output}")
        root = candidates[0]
    seen = set()
    while "source_collection" in read_json(root / "collection.json"):
        if str(root) in seen:
            raise ValueError("Cyclic source provenance")
        seen.add(str(root))
        root = Path(read_json(root / "collection.json")["source_collection"]).resolve()
    return root


def weights(dataset, obs):
    import numpy as np
    if dataset == "ReSisTrace":
        groups = obs.groupby("group", observed=True).outcome.agg(["first", "size"])
        counts = groups["first"].value_counts()
        return np.asarray([len(groups) / (2 * counts.loc[y] * groups.loc[g, "size"])
                           for g, y in zip(obs.group, obs.outcome)])
    if dataset == "MitochondrialPotential":
        sizes = obs.groupby(["patient", "visit", "outcome"], observed=True).outcome.transform("size").to_numpy()
        values = 1. / sizes
        return values * len(values) / values.sum()
    return None


def fit_probe(dataset, x, obs, parameter, settings):
    """One estimator only: no scaler, normalization, centering or extra transforms."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    if dataset == "SCOPEseq2":
        alpha = np.asarray(parameter, dtype=float) if isinstance(parameter, list) else float(parameter)
        estimator = Ridge(alpha=alpha, solver="lsqr", tol=settings["ridge_tol"], max_iter=settings["max_iter"])
        y = obs[settings["targets"]].to_numpy(float)
    else:
        estimator = LogisticRegression(C=float(parameter), solver="lbfgs" if dataset == "ReSisTrace" else "liblinear",
            class_weight="balanced" if dataset == "TumorReactiveTCells" else None,
            max_iter=settings["max_iter"], random_state=settings["seed"])
        y = obs.outcome.to_numpy()
    model = make_pipeline(estimator)
    sample_weights = weights(dataset, obs)
    kwargs = {"logisticregression__sample_weight": sample_weights} if sample_weights is not None else {}
    # L-BFGS can terminate its line search abnormally on float32 inputs at
    # small C, even with only two depth features. Promotion is exact for the
    # cached float32 values: it changes arithmetic precision, not feature scale.
    # Keep disk/GPU caches in float32 and promote only the current training view.
    fit_x = x.astype(np.float64, copy=False) if dataset == "ReSisTrace" else x
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fit_x, y, **kwargs)
    iterations = model[-1].n_iter_
    if iterations is not None and (np.asarray(iterations) >= settings["max_iter"]).any():
        raise RuntimeError("Unconverged predictor; no result published")
    return model


class TargetAlphas(list):
    """List of per-target ridge penalties, compatible with legacy progress formatting."""
    def __format__(self, _):
        return "/".join(f"{value:g}" for value in self)


@contextmanager
def protocol(adapter):
    """Reuse audited split/draw/score code, replacing all predictor construction."""
    module = adapter.experiment
    old = {key: getattr(module, key) for key in ("METHODS", "fit_probe")}
    module.METHODS = tuple(old["METHODS"]) + ("pca_full",)
    module.fit_probe = lambda x, obs, p, s: fit_probe(adapter.dataset, x, obs, p, s)
    if adapter.dataset == "ReSisTrace":
        old["predict_probe"] = module.predict_probe
        module.predict_probe = lambda model, x, _: model.predict_proba(x)[:, 1]
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(module, key, value)


def seed(adapter, settings, stream, repetition=None, rate=None):
    function = adapter.common.random_seed if adapter.dataset == "ReSisTrace" else adapter.common.seed
    return function(settings, stream, repetition, rate)


def calibration_design(adapter, data, settings, partitions):
    import numpy as np
    import pandas as pd
    dataset, module = adapter.dataset, adapter.experiment
    if dataset == "ReSisTrace":
        indices = np.flatnonzero(data.obs.group.isin(partitions[partitions.partition == "calibration"].group))
    elif dataset == "SCOPEseq2":
        regions = partitions.loc[partitions.partition == "calibration", "region"]
        indices = np.flatnonzero(data.obs.region.isin(regions))
    else:
        patients = partitions.loc[partitions.partition == "calibration", "patient"].tolist()
        selection_seed = seed(adapter, settings, "calibration cells")
        indices = (module.select_cells(data, patients, selection_seed, settings) if dataset == "MitochondrialPotential"
                   else module.select_cells(data, patients, selection_seed))
    obs = data.obs.iloc[indices].copy()
    tasks = list(module.CONDITIONS) if dataset == "ReSisTrace" else settings["targets"] if dataset == "SCOPEseq2" else [dataset]
    folds = {}
    for task in tasks:
        if dataset == "ReSisTrace":
            local = np.flatnonzero(obs.condition.eq(task))
            folds[task] = [(str(i), local[a], local[b]) for i, (a, b) in
                           enumerate(module.calibration_cv(obs.iloc[local], settings, task))]
        else:
            column = "region" if dataset == "SCOPEseq2" else "visit" if dataset == "MitochondrialPotential" else "patient"
            folds[task] = [(str(value), np.flatnonzero(obs[column].ne(value)), np.flatnonzero(obs[column].eq(value)))
                           for value in sorted(obs[column].unique())]
    rows = []
    for task, values in folds.items():
        for fold, train, dev in values:
            for role, ix in (("train", train), ("dev", dev)):
                frame = obs.iloc[ix].reset_index().assign(task=task, fold=fold, role=role)
                rows.append(frame)
    return indices, obs, tasks, folds, pd.concat(rows, ignore_index=True)


def score_calibration(dataset, obs, prediction, target=None):
    from sklearn.metrics import roc_auc_score, r2_score
    if dataset == "SCOPEseq2":
        return float(r2_score(obs[target], prediction))
    if dataset == "ReSisTrace":
        frame = obs[["group", "outcome"]].copy()
        frame["prediction"] = prediction
        frame = frame.groupby("group", observed=True).agg(outcome=("outcome", "first"), prediction=("prediction", "mean"))
        return float(roc_auc_score(frame.outcome, frame.prediction))
    return float(roc_auc_score(obs.outcome, prediction))


def search_grid(evaluate, initial, stronger_is_larger, extension_limit, tolerance=1e-5):
    """Expand only an improving edge; tied edge plateaus do not expand forever."""
    import numpy as np
    scores = {}
    values = sorted(set(float(v) for v in initial))
    if len(values) < 2 or values[0] <= 0:
        raise ValueError("Calibration needs at least two positive grid values")
    for step in range(extension_limit + 1):
        for value in values:
            if value not in scores:
                scores[value] = float(evaluate(value))
                if not np.isfinite(scores[value]):
                    raise ValueError("Nonfinite calibration objective")
        ordered = sorted(scores, reverse=stronger_is_larger)
        best = max(ordered, key=lambda value: scores[value])
        edges = sorted(scores)
        adjacent = edges[1] if best == edges[0] else edges[-2] if best == edges[-1] else None
        improving_edge = adjacent is not None and scores[best] > scores[adjacent] + tolerance
        if not improving_edge:
            return best, scores, "edge plateau" if adjacent is not None else "interior"
        if step == extension_limit:
            raise RuntimeError("Calibration optimum still improves at an expanded grid boundary; inspect calibration before evaluating")
        values.append(best / 10 if best == edges[0] else best * 10)
    raise AssertionError("unreachable")


def calibrate(adapter, data, encoder, settings, root, partitions, grids):
    import numpy as np
    import pandas as pd
    from full_data_pca import progress
    destination = root / "calibration"
    if destination.exists():
        verify(destination)
        return read_json(destination / "selected.json")
    pending = root / "pending/calibration"
    if pending.exists():
        import shutil
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    indices, obs, tasks, folds, assignments = calibration_design(adapter, data, settings, partitions)
    obs.reset_index().to_csv(pending / "cells.tsv", sep="\t", index=False)
    assignments.to_csv(pending / "folds.tsv.gz", sep="\t", index=False, compression="gzip")
    partitions.to_csv(pending / "partitions.tsv", sep="\t", index=False)
    selected, tuning, boundary = {}, [], []
    initial = grids["alpha_values"] if adapter.dataset == "SCOPEseq2" else grids["c_values"]
    total = sum(len(v) for v in folds.values()) * len(initial) * len(adapter.experiment.METHODS) * len(settings["retention_rates"])
    with progress(total, f"{adapter.dataset}: fresh calibration fits") as bar:
        for rate in settings["retention_rates"]:
            _, features = encoder.features(indices, rate, seed(adapter, settings, "calibration molecules", rate=rate))
            selected[adapter.common.rate_key(rate)] = {}
            for method in adapter.experiment.METHODS:
                params = {}
                for task in tasks:
                    def objective(parameter):
                        values = []
                        # Extended grids add actual fit counts to the same bar.
                        if parameter not in initial:
                            bar.total += len(folds[task])
                            bar.refresh()
                        for fold, train, dev in folds[task]:
                            bar.set_postfix_str(f"keep={rate:g} {method} {task} parameter={parameter:g}")
                            tick = perf_counter()
                            fit_settings = settings
                            if adapter.dataset == "SCOPEseq2":
                                fit_settings = {**settings, "targets": [task]}
                            model = fit_probe(adapter.dataset, features[method][train], obs.iloc[train], parameter, fit_settings)
                            prediction = (model.predict(features[method][dev]).ravel() if adapter.dataset == "SCOPEseq2"
                                          else model.predict_proba(features[method][dev])[:, 1])
                            score = score_calibration(adapter.dataset, obs.iloc[dev], prediction, task)
                            values.append(score)
                            tuning.append({"retention_rate": rate, "method": method, "task": task, "fold": fold,
                                           "parameter": parameter, "score": score, "seconds": perf_counter() - tick})
                            bar.update(1)
                        return np.mean(values)
                    best, _, location = search_grid(objective, initial, adapter.dataset == "SCOPEseq2",
                        grids["max_grid_extensions"], grids["boundary_score_tolerance"])
                    params[task] = best
                    boundary.append({"retention_rate": rate, "method": method, "task": task, "selected": best,
                                     "location": location})
                key = adapter.common.rate_key(rate)
                if adapter.dataset == "ReSisTrace":
                    for task in tasks:
                        selected[key].setdefault(task, {})[method] = params[task]
                elif adapter.dataset == "SCOPEseq2":
                    selected[key][method] = [params[t] for t in tasks]
                else:
                    selected[key][method] = params[tasks[0]]
            del features
    pd.DataFrame(tuning).to_csv(pending / "tuning.tsv", sep="\t", index=False)
    pd.DataFrame(boundary).to_csv(pending / "grid_audit.tsv", sep="\t", index=False)
    write_json(pending / "selected.json", selected)
    seal(pending, {"objective": "mean held-out-region R2 per target" if adapter.dataset == "SCOPEseq2" else
                   "mean held-out-fold AUROC with final evaluation aggregation", "no_scaler": True,
                   "probabilities": "direct estimator probabilities; no prior correction", "n_cells": len(obs)})
    pending.rename(destination)
    return selected


def scientific_identity(adapter, source, settings, grids, pca_settings):
    original = read_json(source / "collection.json")
    adapter.check_source_code(original)
    return {"protocol": KIND, "dataset": adapter.dataset, "source_identity": original["identity"],
        "settings": settings, "grids": grids, "pca_settings": pca_settings,
        "preprocessing": "none after raw log1p / native scUNVEIL / unwhitened full-data PCA / defined controls",
        "recalibration_metric": "R2 per target" if adapter.dataset == "SCOPEseq2" else "AUROC",
        "code": {name: computation_code_hash(HERE / name) for name in ("benchmark_runner.py", "benchmark_features.py", "full_data_pca.py", "pca_comparison.py")},
        "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "anndata", "scikit-learn", "tensorflow")}}


def run_dataset(dataset, source, root, *, target=100, cache_mode="reuse", allow_legacy=True,
                threads=8, device="gpu", factory=None, grids=None, pca_settings=None, adapter=None):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from benchmark_features import CachedEncoder, clear_cache
    from full_data_pca import PCA_SETTINGS, progress
    adapter = adapter or Adapter(dataset)
    root, source = Path(root), Path(source)
    source_manifest = read_json(source / "collection.json")
    settings = {**source_manifest["settings"], "threads": threads, "device": device}
    grids = {**DEFAULT_GRIDS, **(grids or {})}
    pca_settings = {**PCA_SETTINGS, **(pca_settings or {})}
    identity = scientific_identity(adapter, source, settings, grids, pca_settings)
    output = Path(settings["output_root"])
    with lock(output / "benchmark.lock"), threadpool_limits(threads):
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "collection.json"
        if manifest_path.exists():
            if read_json(manifest_path)["identity"] != identity:
                raise ValueError("Resume settings/code/environment changed; create fresh results instead")
        else:
            write_json(manifest_path, {"kind": KIND, "dataset": dataset, "identity": identity, "settings": settings,
                "grids": grids, "source_collection": str(source.resolve()), "created_utc": datetime.now(timezone.utc).isoformat()})
        write_json(root / "computation_code_audit.json", {
            "files": {name: {"actual_sha256": sha256(HERE / name), "computation_identity_sha256": computation_code_hash(HERE / name)}
                      for name in ("benchmark_runner.py", "benchmark_features.py")},
            "note": "Exact allowlisted checkpoint-validation-only maintenance patch; predictors, features and calibration unchanged."})
        if cache_mode == "rebuild":
            clear_cache(output)
            allow_legacy = False
        if not allow_legacy:
            # The rebuild policy is preserved on resume to keep legacy caches excluded.
            write_json(root / "cache_policy.json", {"allow_legacy_import": False})
        if (root / "cache_policy.json").exists():
            allow_legacy = allow_legacy and read_json(root / "cache_policy.json")["allow_legacy_import"]
        existing = completed(root)
        for path in existing:
            verify(path)
        if len(existing) >= target:
            return root
        data, prepared_path, counts_hash = adapter.load_data(source, source_manifest)
        write_json(root / "prepared.json", {"counts_path": str(prepared_path), "counts_sha256": counts_hash})
        partitions = adapter.partitions(data, settings)
        encoder = CachedEncoder(adapter, data, settings, root, source, counts_hash, allow_legacy, factory, pca_settings)
        workspace = root / "workspace"
        with protocol(adapter):
            write_json(root / "status.json", {"state": "fresh calibration", "completed_repetitions": len(existing)})
            selected = calibrate(adapter, data, encoder, settings, root, partitions, grids)
            if dataset == "SCOPEseq2":
                selected = {rate: {m: TargetAlphas(v) for m, v in methods.items()} for rate, methods in selected.items()}
            with progress(target - len(existing), f"{dataset}: repetitions remaining to {target}") as bar:
                for index in range(len(existing), target):
                    write_json(root / "status.json", {"state": "collecting", "completed_repetitions": index, "target": target})
                    unfinished = workspace / "replicates" / f"replicate_{index:06d}"
                    if unfinished.exists():
                        import shutil
                        shutil.rmtree(unfinished)
                    tick = perf_counter()
                    generated = adapter.experiment.collect_one(data, encoder, settings, workspace, partitions, selected, index)
                    info = verify(generated)
                    source_rep = source / "replicates" / generated.name
                    if source_rep.exists():
                        verify(source_rep, list(adapter.paired_files))
                        compare_frames(source_rep, generated, adapter.paired_files)
                    metadata = {k: v for k, v in info.items() if k != "files"}
                    metadata.update(protocol=KIND, pca_dimension=encoder.pca.dimension, no_scaler=True,
                        probabilities="direct; no prior correction", calibration_sha256=sha256(root / "calibration/complete.json"),
                        collection_identity=digest(identity), paired_source_verified=source_rep.exists(),
                        feature_cache={"hits": encoder.store.hits, "misses": encoder.store.misses})
                    seal(generated, metadata)
                    destination = root / "replicates" / generated.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    generated.rename(destination)
                    bar.update(1)
                    write_json(root / "status.json", {"state": "collecting", "completed_repetitions": index + 1,
                        "target": target, "last_repetition_seconds": perf_counter() - tick,
                        "feature_cache_hits": encoder.store.hits, "feature_cache_misses": encoder.store.misses})
        write_json(root / "status.json", {"state": "target reached", "completed_repetitions": len(completed(root))})
    return root


def suite_main(args):
    output = args.analysis_root
    controller = output / "benchmark_suites"
    controller.mkdir(parents=True, exist_ok=True)
    with lock(controller / "runner.lock"):
        if args.results == "fresh":
            suite_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            suite = {"kind": KIND, "suite_id": suite_id, "device": args.device, "threads": args.threads,
                     "datasets": {}, "allow_legacy_import": args.cache != "rebuild"}
            for dataset in DATASETS:
                source = resolve_source(output / dataset)
                root = output / dataset / "collections" / ("unscaled-" + suite_id)
                suite["datasets"][dataset] = {"source": str(source), "collection": str(root),
                    "cache_rebuild_pending": args.cache == "rebuild"}
            suite_path = controller / (suite_id + ".json")
            write_json(suite_path, suite)
            write_json(controller / "LATEST_UNSCALED.json", {"suite": str(suite_path)})
        else:
            suite_path = args.suite or Path(read_json(controller / "LATEST_UNSCALED.json")["suite"])
            suite = read_json(suite_path)
            if suite["kind"] != KIND:
                raise ValueError("Wrong suite protocol")
            if args.device != suite["device"] or args.threads != suite["threads"]:
                raise ValueError("Resume with the same --device and --threads as the original suite")
            if args.cache == "rebuild":
                suite["allow_legacy_import"] = False
                for value in suite["datasets"].values():
                    value["cache_rebuild_pending"] = True
                write_json(suite_path, suite)
        print(f"Suite: {suite_path}", flush=True)
        for dataset in DATASETS:
            entry = suite["datasets"][dataset]
            command = [sys.executable, "-u", str(HERE / "benchmark_runner.py"), "--dataset", dataset,
                "--source-collection", entry["source"], "--collection", entry["collection"],
                "--max-replicates", str(args.max_replicates), "--threads", str(args.threads), "--device", args.device]
            # Clear the cache under the same lock used by every dataset worker.
            # Persist before starting inference so interruption never clears it twice.
            if entry["cache_rebuild_pending"]:
                from benchmark_features import clear_cache
                with lock(output / dataset / "benchmark.lock"):
                    clear_cache(output / dataset)
                entry["cache_rebuild_pending"] = False
                write_json(suite_path, suite)
            if not suite["allow_legacy_import"]:
                command.append("--no-legacy-import")
            print(f"\n[{list(DATASETS).index(dataset) + 1}/4] {dataset}: total target {args.max_replicates}", flush=True)
            subprocess.run(command, check=True)
        print("All four reports refreshed successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", choices=("fresh", "resume"), default="resume")
    parser.add_argument("--cache", choices=("reuse", "rebuild"), default="reuse")
    parser.add_argument("--max-replicates", type=int, default=100, help="TOTAL completed target, not additional repetitions")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--analysis-root", type=Path, default=Path("/workspace/results/analysis"))
    parser.add_argument("--suite", type=Path, help="Pin a suite JSON when resuming")
    parser.add_argument("--dataset", choices=DATASETS, help="Worker / individual-dataset mode")
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--source-collection", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-legacy-import", action="store_true")
    args = parser.parse_args()
    if args.max_replicates < 1 or args.threads < 1:
        parser.error("--max-replicates and --threads must be positive")
    if args.report_only and args.collection is None:
        parser.error("--report-only needs --collection")
    if args.results == "fresh" and args.suite:
        parser.error("--suite is for resuming; fresh creates a new suite")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try:
        if args.report_only:
            from benchmark_report import build_report
            print(build_report(args.collection, args.max_replicates), flush=True)
        elif args.dataset:
            if args.collection is None or args.source_collection is None:
                parser.error("Individual mode needs --collection and --source-collection")
            root = run_dataset(args.dataset, args.source_collection, args.collection, target=args.max_replicates,
                cache_mode=args.cache, allow_legacy=not args.no_legacy_import, threads=args.threads, device=args.device)
            from benchmark_report import build_report
            print(f"Report: {build_report(root, args.max_replicates) / 'REPORT.md'}", flush=True)
        else:
            suite_main(args)
    except KeyboardInterrupt:
        print("Stopped safely. Resume with --results resume --cache reuse and the same total target.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
