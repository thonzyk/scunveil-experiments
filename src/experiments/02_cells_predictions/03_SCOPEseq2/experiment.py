"""Paired molecular thinning with fresh technical-region-held-out ridge fits."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import shutil
from time import perf_counter

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from features import Encoder, make_model, progress
from prepare import prepare
from scope_common import (METHODS, collection_directory, completed, digest, lock, rate_key,
                          read_json, seal, seed, verify, write_frame, write_json)


def region_partitions(obs, settings):
    table = obs.groupby("region", observed=True).size().rename("n_cells").to_frame()
    if len(table) < settings["calibration_regions"] + settings["test_regions"] + settings["dev_regions"] + 2:
        raise ValueError("Need calibration, dev, test and at least two training regions")
    choices = list(combinations(sorted(table.index), settings["calibration_regions"]))
    target = len(obs) * settings["calibration_target_fraction"]
    distances = np.array([abs(table.loc[list(choice), "n_cells"].sum() - target) for choice in choices])
    best = np.flatnonzero(np.isclose(distances, distances.min()))
    rng = np.random.default_rng(seed(settings, "calibration regions"))
    calibration = choices[int(rng.choice(best))]
    table["partition"] = np.where(table.index.isin(calibration), "calibration", "pool")
    return table.reset_index()


def repetition_split(regions, settings, repetition):
    pool = regions[regions.partition == "pool"].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed(settings, "region split", repetition))
    order = rng.permutation(len(pool))
    roles = np.full(len(pool), "train", dtype=object)
    n_test, n_dev = settings["test_regions"], settings["dev_regions"]
    roles[order[:n_test]] = "test"
    roles[order[n_test:n_test + n_dev]] = "dev"
    pool["role"] = roles
    return pool


def fit_probe(x, obs, alpha, settings):
    if not obs.index.is_unique or len(obs) < 2:
        raise ValueError("Unique training cells required")
    model = make_pipeline(StandardScaler(with_mean=not sparse.issparse(x)),
        Ridge(alpha=float(alpha), solver="lsqr", tol=settings["ridge_tol"], max_iter=settings["max_iter"]))
    model.fit(x, obs[settings["targets"]].to_numpy(dtype=float))
    iterations = model[-1].n_iter_
    if iterations is not None and (np.asarray(iterations) >= settings["max_iter"]).any():
        raise RuntimeError("Ridge reached max_iter; inspect convergence rather than reporting an unfinished fit")
    return model


def scores(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if len(observed) < 2 or not np.isfinite(predicted).all() or np.var(observed) == 0:
        raise ValueError("Need finite predictions and a nonconstant evaluation target")
    variable = np.var(predicted) > 0
    return {"r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic) if variable else np.nan,
        "pearson": float(np.corrcoef(observed, predicted)[0, 1]) if variable else np.nan}


def pending(root, name):
    directory = Path(root) / "pending" / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def calibrate(data, encoder, settings, root, regions):
    destination = root / "calibration"
    if destination.exists():
        verify(destination)
        return read_json(destination / "selected_alpha.json")
    directory = pending(root, "calibration")
    cal = regions.loc[regions.partition == "calibration", "region"].tolist()
    indices = np.flatnonzero(data.obs.region.isin(cal).to_numpy())
    obs = data.obs.iloc[indices]
    write_frame(directory / "cells.tsv", obs.reset_index())
    selected, rows = {}, []
    total = len(settings["retention_rates"]) * (1 + len(METHODS) * len(cal) * len(settings["alpha_values"]))
    with progress(total, "One-time ridge alpha calibration") as bar:
        for rate in settings["retention_rates"]:
            _, features = encoder.features(indices, rate, seed(settings, "calibration molecules", rate=rate))
            bar.update(1)
            selected[rate_key(rate)] = {}
            for method in METHODS:
                by_alpha = {float(a): [] for a in sorted(settings["alpha_values"], reverse=True)}
                for region in cal:
                    train = np.flatnonzero(obs.region.ne(region).to_numpy())
                    dev = np.flatnonzero(obs.region.eq(region).to_numpy())
                    for alpha in by_alpha:
                        bar.set_postfix_str(f"keep={rate:g} {method} region={region} alpha={alpha:g}")
                        started = perf_counter()
                        model = fit_probe(features[method][train], obs.iloc[train], alpha, settings)
                        prediction = model.predict(features[method][dev])
                        for column, target in enumerate(settings["targets"]):
                            metric = scores(obs.iloc[dev][target].to_numpy(), prediction[:, column])
                            by_alpha[alpha].append(metric["r2"])
                            rows.append({"retention_rate": rate, "method": method, "held_out_region": region,
                                "target": target, "alpha": alpha, "seconds": perf_counter() - started, **metric})
                        bar.update(1)
                # Same alpha for both output heads. Average R2 equally over targets and regions.
                # Descending insertion order favors stronger regularization on exact ties.
                selected[rate_key(rate)][method] = max(by_alpha, key=lambda a: np.mean(by_alpha[a]))
            del features
    write_frame(directory / "tuning.tsv", pd.DataFrame(rows))
    write_json(directory / "selected_alpha.json", selected)
    seal(directory, {"calibration_regions": cal, "n_cells": len(obs),
        "selection_metric": "equal mean held-out-region R2 over both targets; larger alpha breaks exact ties",
        "molecular_seeds": {rate_key(r): seed(settings, "calibration molecules", rate=r)
                            for r in settings["retention_rates"]}})
    directory.rename(destination)
    return selected


def evaluate_predictions(obs, predictions, rate, repetition, settings):
    rows = []
    for role, part in obs.groupby("role", observed=True, sort=True):
        for method in METHODS:
            for target in settings["targets"]:
                column = f"{method}__{target}"
                base = {"repetition": repetition, "retention_rate": rate, "split": role,
                        "method": method, "target": target}
                local = []
                for region, group in part.groupby("region", observed=True, sort=True):
                    metric = scores(group[target].to_numpy(), predictions.loc[group.index, column].to_numpy())
                    local.append({**base, "scope": "region", "region": region, "n_cells": len(group), **metric})
                rows.extend(local)
                rows.append({**base, "scope": "macro_region", "region": "ALL", "n_cells": len(part),
                    **{key: float(np.mean([r[key] for r in local])) for key in ("r2", "mae")},
                    **{key: float(pd.Series([r[key] for r in local]).mean()) for key in ("spearman", "pearson")}})
                rows.append({**base, "scope": "pooled", "region": "ALL", "n_cells": len(part),
                    **scores(part[target].to_numpy(), predictions.loc[part.index, column].to_numpy())})
    return rows


def collect_one(data, encoder, settings, root, regions, selected, repetition):
    directory = pending(root, f"replicate_{repetition:06d}")
    started = perf_counter()
    assignments = repetition_split(regions, settings, repetition)
    indices = np.flatnonzero(data.obs.region.isin(assignments.region).to_numpy())
    obs = data.obs.iloc[indices].copy()
    obs["role"] = obs.region.map(assignments.set_index("region").role)
    if obs.role.isna().any():
        raise ValueError("Missing region assignment")
    write_frame(directory / "region_assignments.tsv", assignments)
    write_frame(directory / "cells.tsv", obs.reset_index())
    train = np.flatnonzero(obs.role.eq("train").to_numpy())
    evaluation = np.flatnonzero(obs.role.ne("train").to_numpy())
    rows, audits, fits = [], [], []
    with progress(len(settings["retention_rates"]) * (1 + len(METHODS)) + 1,
                  f"Repetition {repetition + 1}") as bar:
        for rate in settings["retention_rates"]:
            draw = seed(settings, "repetition molecules", repetition, rate)
            bar.set_postfix_str(f"keep={rate:g}: paired counts / all native embeddings")
            counts, features = encoder.features(indices, rate, draw)
            bar.update(1)
            for (role, region), group in obs.groupby(["role", "region"], observed=True):
                local = obs.index.get_indexer(group.index)
                totals = np.asarray(counts[local].sum(axis=1)).ravel()
                audits.append({"retention_rate": rate, "split": role, "region": region,
                    "dilution_seed": draw, "n_cells": len(group), "total_umis": int(totals.sum()),
                    "median_umis": float(np.median(totals)), "empty_cells": int((totals == 0).sum()),
                    "counts_sha256": digest([counts[local].indptr.tolist(), counts[local].indices.tolist(),
                                             counts[local].data.tolist()])})
            eval_obs = obs.iloc[evaluation]
            predictions = pd.DataFrame(index=eval_obs.index)
            for method in METHODS:
                alpha = selected[rate_key(rate)][method]
                bar.set_postfix_str(f"keep={rate:g}: {method}, fresh ridge fit alpha={alpha:g}")
                tick = perf_counter()
                model = fit_probe(features[method][train], obs.iloc[train], alpha, settings)
                prediction = model.predict(features[method][evaluation])
                for column, target in enumerate(settings["targets"]):
                    predictions[f"{method}__{target}"] = prediction[:, column]
                fits.append({"retention_rate": rate, "method": method, "alpha": alpha,
                    "n_train_cells": len(train), "n_features": features[method].shape[1],
                    "max_solver_iterations": int(np.max(model[-1].n_iter_)), "seconds": perf_counter() - tick})
                if settings["save_models"]:
                    path = directory / "probes" / rate_key(rate) / f"{method}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, path)
                bar.update(1)
            rows.extend(evaluate_predictions(eval_obs, predictions, rate, repetition, settings))
            write_frame(directory / "predictions" / f"{rate_key(rate)}.tsv.gz", eval_obs.join(predictions).reset_index())
            del counts, features
        for name, values in (("metrics", rows), ("measurement_audit", audits), ("fit_times", fits)):
            write_frame(directory / f"{name}.tsv", pd.DataFrame(values))
        write_json(directory / "selected_alpha.json", selected)
        seal(directory, {"repetition": repetition, "elapsed_seconds": perf_counter() - started,
            "split_seed": seed(settings, "region split", repetition), "embedding_dim": encoder.dimension,
            "cells_sha256": digest(obs.index.tolist()),
            **{f"{role}_regions": sorted(assignments.loc[assignments.role == role, "region"])
               for role in ("train", "dev", "test")},
            "runtime_settings": {key: settings[key] for key in ("device", "batch_size", "threads", "save_models")}})
        destination = root / "replicates" / directory.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        directory.rename(destination)
        bar.update(1)
    return destination


def collect(settings, max_replicates=None, model_factory=make_model, prepared_input=None):
    if max_replicates is not None and (type(max_replicates) is not int or max_replicates < 1):
        raise ValueError("max_replicates must be a positive TOTAL target")
    prepared_path, prepared = prepared_input or prepare(settings)
    root = collection_directory(settings, prepared)
    tqdm.write(f"Collection: {root}")
    with lock(root / "collector.lock"):
        index = len(completed(root))
        if max_replicates is not None and index >= max_replicates:
            tqdm.write(f"Already have {index} repetitions; target reached")
            return root
        overall = None
        try:
            verify(prepared_path)
            data = ad.read_h5ad(prepared_path / "counts.h5ad")
            write_json(root / "prepared.json", {"directory": str(prepared_path), "receipt": prepared})
            regions = region_partitions(data.obs, settings)
            write_frame(root / "region_partitions.tsv", regions)
            write_json(root / "status.json", {"state": "full-depth cache", "completed_repetitions": index})
            encoder = Encoder(data, settings, root, model_factory)
            encoder.full_cache()
            write_json(root / "status.json", {"state": "calibrating ridge alpha", "completed_repetitions": index})
            selected = calibrate(data, encoder, settings, root, regions)
            if max_replicates is not None:
                overall = progress(max_replicates - index, "Remaining collection repetitions")
            while max_replicates is None or index < max_replicates:
                write_json(root / "status.json", {"state": "collecting", "completed_repetitions": index,
                                                  "current_repetition": index})
                directory = collect_one(data, encoder, settings, root, regions, selected, index)
                index += 1
                if overall is not None:
                    overall.update(1)
                seconds = read_json(directory / "complete.json")["elapsed_seconds"]
                tqdm.write(f"Saved repetition {index} ({seconds:.1f} sec): {directory}. Ctrl+C stops safely.")
            write_json(root / "status.json", {"state": "target reached", "completed_repetitions": index})
        except KeyboardInterrupt:
            count = len(completed(root, checksums=False))
            write_json(root / "status.json", {"state": "stopped", "completed_repetitions": count})
            tqdm.write(f"Stopped; {count} completed repetitions preserved. Report: python build_report.py")
        except Exception as exc:
            write_json(root / "status.json", {"state": "error", "completed_repetitions": len(completed(root, checksums=False)),
                                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            if overall is not None:
                overall.close()
    return root
