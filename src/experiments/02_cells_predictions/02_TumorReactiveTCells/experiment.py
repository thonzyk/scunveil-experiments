"""Fresh patient splits, one cell per clonotype, paired molecule draws and probe fits."""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
import shutil
from time import perf_counter
import warnings

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from features import Encoder, make_model, progress
from prepare import clone_table, prepare
from tcr_common import (METHODS, collection_directory, completed, digest, lock, rate_key,
                        read_json, seal, seed, verify, write_frame, write_json)


def patient_partitions(obs, settings):
    clones = clone_table(obs)
    table = clones.groupby("patient", observed=True).agg(n_clones=("outcome", "size"), reactive=("outcome", "sum"))
    table["nonreactive"] = table.n_clones - table.reactive
    eligible = table.index[(table[["reactive", "nonreactive"]] >=
                            settings["min_calibration_clones_per_class"]).all(axis=1)].to_numpy()
    if len(eligible) < settings["calibration_patients"]:
        raise ValueError("Too few patients with enough tested clonotypes of both classes for calibration")
    # Reserve whole patients near the ReSisTrace 20% calibration budget, using
    # sample counts only. Do not spend most labels on calibration by chance.
    choices = list(combinations(eligible, settings["calibration_patients"]))
    target = len(clones) * settings["calibration_target_fraction"]
    distances = np.array([abs(table.loc[list(choice), "n_clones"].sum() - target) for choice in choices])
    best = np.flatnonzero(np.isclose(distances, distances.min()))
    rng = np.random.default_rng(seed(settings, "calibration patients"))
    calibration = choices[int(rng.choice(best))]
    table["partition"] = np.where(table.index.isin(calibration), "calibration", "pool")
    table["mixed_labels"] = (table.reactive > 0) & (table.nonreactive > 0)
    pool = table[table.partition == "pool"]
    if len(pool) < settings["test_patients"] + settings["dev_patients"] + 2 or pool.mixed_labels.sum() < 2:
        raise ValueError("Need train patients and mixed-label dev/test patients after calibration reservation")
    return table.reset_index()


def repetition_split(patients, settings, repetition):
    pool = patients[patients.partition == "pool"].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed(settings, "patient split", repetition))
    # Rejection sampling is over patient assignments, never prediction performance.
    for _ in range(10000):
        order = rng.permutation(len(pool))
        roles = np.full(len(pool), "train", dtype=object)
        n_test, n_dev = settings["test_patients"], settings["dev_patients"]
        roles[order[:n_test]] = "test"
        roles[order[n_test:n_test + n_dev]] = "dev"
        pool["role"] = roles
        train = pool[pool.role == "train"]
        if (pool.loc[pool.role == "test", "mixed_labels"].any() and
                pool.loc[pool.role == "dev", "mixed_labels"].any() and
                train.reactive.sum() > 0 and train.nonreactive.sum() > 0):
            return pool
    raise ValueError("Cannot construct patient-disjoint train/dev/test assignments with scorable dev/test")


def select_cells(data, patients, selection_seed):
    """One random cell per clonotype; use this same selection at every depth/method."""
    rng = np.random.default_rng(selection_seed)
    indices = []
    eligible = data.obs.patient.isin(patients).to_numpy()
    for _, rows in data.obs.iloc[np.flatnonzero(eligible)].groupby("clonotype_id", sort=True, observed=True):
        cell_id = rng.choice(rows.index.to_numpy())
        indices.append(data.obs_names.get_loc(cell_id))
    indices = np.array(indices, dtype=int)
    if not len(indices) or data.obs.iloc[indices].clonotype_id.duplicated().any():
        raise ValueError("Expected one selected cell per clonotype")
    return indices


def fit_probe(x, obs, c, settings):
    if obs.clonotype_id.duplicated().any() or obs.outcome.nunique() != 2:
        raise ValueError("Training must use unique tested clonotypes with both outcomes")
    # Few clonotypes make the primal solver inexpensive; avoid dual convergence
    # problems for nearly constant depth summaries. Sparse RNA stays sparse.
    model = make_pipeline(StandardScaler(with_mean=not sparse.issparse(x)), LogisticRegression(
        C=float(c), solver="liblinear", dual=False, class_weight="balanced",
        max_iter=settings["max_iter"], random_state=settings["seed"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, obs.outcome.to_numpy())
    return model


def scores(outcomes, probabilities):
    if len(np.unique(outcomes)) < 2:
        return {"auroc": np.nan, "average_precision": np.nan}
    return {"auroc": float(roc_auc_score(outcomes, probabilities)),
            "average_precision": float(average_precision_score(outcomes, probabilities))}


def pending(root, name):
    directory = Path(root) / "pending" / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def calibrate(data, encoder, settings, root, patients):
    destination = root / "calibration"
    if destination.exists():
        verify(destination)
        return read_json(destination / "selected_c.json")
    directory = pending(root, "calibration")
    cal_patients = patients.loc[patients.partition == "calibration", "patient"].tolist()
    indices = select_cells(data, cal_patients, seed(settings, "calibration cells"))
    obs = data.obs.iloc[indices].copy()
    write_frame(directory / "selected_cells.tsv", obs.reset_index())
    selected, rows, predictions = {}, [], []
    total = len(settings["retention_rates"]) * (1 + len(METHODS) * len(cal_patients) * len(settings["c_values"]))
    with progress(total, "One-time C calibration") as bar:
        for rate in settings["retention_rates"]:
            _, features = encoder.features(indices, rate, seed(settings, "calibration molecules", rate=rate))
            selected[rate_key(rate)] = {}
            bar.update(1)
            for method in METHODS:
                by_c = {float(c): [] for c in sorted(settings["c_values"])}
                for patient in cal_patients:
                    train = np.flatnonzero(obs.patient.ne(patient).to_numpy())
                    dev = np.flatnonzero(obs.patient.eq(patient).to_numpy())
                    for c in by_c:
                        bar.set_postfix_str(f"keep={rate:g} {method} held-out={patient} C={c:g}")
                        started = perf_counter()
                        model = fit_probe(features[method][train], obs.iloc[train], c, settings)
                        probabilities = model.predict_proba(features[method][dev])[:, 1]
                        result = scores(obs.iloc[dev].outcome.to_numpy(), probabilities)
                        if not np.isfinite(result["auroc"]):
                            raise ValueError("Calibration folds require both classes")
                        by_c[c].append(result["auroc"])
                        rows.append({"retention_rate": rate, "method": method, "held_out_patient": patient,
                            "C": c, "seconds": perf_counter() - started, **result})
                        predictions.append(pd.DataFrame({"clonotype_id": obs.clonotype_id.iloc[dev].to_numpy(),
                            "patient": patient, "outcome": obs.outcome.iloc[dev].to_numpy(),
                            "probability": probabilities, "retention_rate": rate, "method": method, "C": c}))
                        bar.update(1)
                selected[rate_key(rate)][method] = max(by_c, key=lambda c: np.mean(by_c[c]))
            del features
    write_frame(directory / "tuning.tsv", pd.DataFrame(rows))
    write_frame(directory / "validation_predictions.tsv.gz", pd.concat(predictions, ignore_index=True))
    write_json(directory / "selected_c.json", selected)
    seal(directory, {"calibration_patients": cal_patients, "n_clones": len(obs),
        "selection_seed": seed(settings, "calibration cells"),
        "molecular_seeds": {rate_key(r): seed(settings, "calibration molecules", rate=r)
                            for r in settings["retention_rates"]},
        "selection_metric": "mean held-out calibration-patient AUROC; smaller C breaks ties"})
    directory.rename(destination)
    return selected


def evaluate_predictions(obs, predictions, rate, repetition):
    rows = []
    for role, part in obs.groupby("role", sort=True, observed=True):
        for method in METHODS:
            probabilities = predictions.loc[part.index, method].to_numpy()
            base = {"repetition": repetition, "retention_rate": rate, "split": role, "method": method}
            patient_rows = []
            for patient, group in part.groupby("patient", sort=True, observed=True):
                result = scores(group.outcome.to_numpy(), predictions.loc[group.index, method].to_numpy())
                patient_rows.append({**base, "scope": "patient", "patient": patient,
                    "n_clones": len(group), "n_reactive": int(group.outcome.sum()), **result})
            rows.extend(patient_rows)
            rows.append({**base, "scope": "pooled", "patient": "ALL", "n_clones": len(part),
                "n_reactive": int(part.outcome.sum()), **scores(part.outcome.to_numpy(), probabilities)})
            valid = [row for row in patient_rows if np.isfinite(row["auroc"])]
            if not valid:
                raise ValueError("At least one mixed-label patient is required in each evaluation role")
            rows.append({**base, "scope": "macro_patient", "patient": "ALL", "n_clones": len(part),
                "n_reactive": int(part.outcome.sum()), "n_scorable_patients": len(valid),
                "auroc": float(np.mean([row["auroc"] for row in valid])),
                "average_precision": float(np.mean([row["average_precision"] for row in valid]))})
    return rows


def collect_one(data, encoder, settings, root, patients, selected, repetition):
    directory = pending(root, f"replicate_{repetition:06d}")
    started = perf_counter()
    assignments = repetition_split(patients, settings, repetition)
    indices = select_cells(data, assignments.patient.tolist(), seed(settings, "selected cells", repetition))
    obs = data.obs.iloc[indices].copy()
    obs["role"] = obs.patient.map(assignments.set_index("patient").role)
    if obs.role.isna().any():
        raise ValueError("Missing patient role")
    write_frame(directory / "patient_assignments.tsv", assignments)
    write_frame(directory / "selected_cells.tsv", obs.reset_index())
    rows, audits, fits = [], [], []
    train = np.flatnonzero(obs.role.eq("train").to_numpy())
    evaluation = np.flatnonzero(obs.role.ne("train").to_numpy())
    with progress(len(settings["retention_rates"]) * (1 + len(METHODS)) + 1,
                  f"Repetition {repetition + 1}") as bar:
        for rate in settings["retention_rates"]:
            draw = seed(settings, "repetition molecules", repetition, rate)
            bar.set_postfix_str(f"keep={rate:g}: paired counts / all native embeddings")
            counts, features = encoder.features(indices, rate, draw)
            bar.update(1)
            for (role, patient), group in obs.groupby(["role", "patient"], observed=True):
                local = obs.index.get_indexer(group.index)
                totals = np.asarray(counts[local].sum(axis=1)).ravel()
                audits.append({"repetition": repetition, "retention_rate": rate, "split": role,
                    "patient": patient, "dilution_seed": draw, "n_cells": len(group),
                    "total_umis": int(totals.sum()), "median_umis": float(np.median(totals)),
                    "empty_cells": int((totals == 0).sum()),
                    "counts_sha256": digest([counts[local].indptr.tolist(), counts[local].indices.tolist(),
                                             counts[local].data.tolist()])})
            eval_obs = obs.iloc[evaluation]
            predictions = pd.DataFrame(index=eval_obs.index)
            fit_settings = {**settings, "seed": seed(settings, "classifier fit", repetition)}
            for method in METHODS:
                c = selected[rate_key(rate)][method]
                bar.set_postfix_str(f"keep={rate:g}: {method}, fresh sklearn fit C={c:g}")
                tick = perf_counter()
                model = fit_probe(features[method][train], obs.iloc[train], c, fit_settings)
                predictions[method] = model.predict_proba(features[method][evaluation])[:, 1]
                fits.append({"retention_rate": rate, "method": method, "C": c,
                    "fit_seed": fit_settings["seed"], "n_train_clones": len(train),
                    "n_features": features[method].shape[1], "seconds": perf_counter() - tick})
                if settings["save_models"]:
                    path = directory / "probes" / rate_key(rate) / f"{method}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, path)
                bar.update(1)
            rows.extend(evaluate_predictions(eval_obs, predictions, rate, repetition))
            write_frame(directory / "predictions" / f"{rate_key(rate)}.tsv.gz",
                        eval_obs.join(predictions).reset_index())
            del counts, features
        for name, values in (("metrics", rows), ("measurement_audit", audits), ("fit_times", fits)):
            write_frame(directory / f"{name}.tsv", pd.DataFrame(values))
        write_json(directory / "selected_c.json", selected)
        seal(directory, {"repetition": repetition, "elapsed_seconds": perf_counter() - started,
            "split_seed": seed(settings, "patient split", repetition),
            "selection_seed": seed(settings, "selected cells", repetition),
            "fit_seed": seed(settings, "classifier fit", repetition), "embedding_dim": encoder.dimension,
            "selected_cells_sha256": digest(obs.index.tolist()),
            **{f"{role}_patients": sorted(assignments.loc[assignments.role == role, "patient"])
               for role in ("train", "dev", "test")},
            "runtime_settings": {k: settings[k] for k in ("device", "batch_size", "threads", "save_models")}})
        destination = root / "replicates" / directory.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        directory.rename(destination)
        bar.update(1)
    return destination


def collect(settings, max_replicates=None, model_factory=make_model, prepared_input=None):
    if max_replicates is not None and (type(max_replicates) is not int or max_replicates < 1):
        raise ValueError("max_replicates must be a positive total target")
    prepared_path, prepared = prepared_input or prepare(settings)
    root = collection_directory(settings, prepared)
    tqdm.write(f"Collection: {root}")
    with lock(root / "collector.lock"):
        index = len(completed(root))
        if max_replicates is not None and index >= max_replicates:
            tqdm.write(f"Already have {index} completed repetitions; target reached")
            return root
        overall = None
        try:
            verify(prepared_path)
            data = ad.read_h5ad(prepared_path / "counts.h5ad")
            clones = clone_table(data.obs)
            data.obs["n_cells"] = data.obs.clonotype_id.map(clones.n_cells).astype(int)
            write_json(root / "prepared.json", {"directory": str(prepared_path), "receipt": prepared})
            patients = patient_partitions(data.obs, settings)
            write_frame(root / "patient_partitions.tsv", patients)
            write_json(root / "status.json", {"state": "full-depth cache", "completed_repetitions": index})
            encoder = Encoder(data, settings, root, model_factory)
            encoder.full_cache()
            write_json(root / "status.json", {"state": "calibrating C", "completed_repetitions": index})
            selected = calibrate(data, encoder, settings, root, patients)
            if max_replicates is not None:
                overall = progress(max_replicates - index, "Remaining collection repetitions")
            while max_replicates is None or index < max_replicates:
                write_json(root / "status.json", {"state": "collecting", "completed_repetitions": index,
                                                  "current_repetition": index})
                directory = collect_one(data, encoder, settings, root, patients, selected, index)
                index += 1
                if overall is not None:
                    overall.update(1)
                seconds = read_json(directory / "complete.json")["elapsed_seconds"]
                tqdm.write(f"Saved repetition {index} ({seconds / 60:.2f} min): {directory}. Ctrl+C stops safely.")
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
