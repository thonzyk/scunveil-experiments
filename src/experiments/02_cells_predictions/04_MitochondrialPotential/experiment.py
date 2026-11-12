"""Patient-disjoint TMRE prediction with paired fresh cells, molecules and CPU fits."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import shutil
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
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from features import Encoder, make_model, progress
from prepare import prepare
from tmre_common import (METHODS, collection_directory, completed, digest, lock, rate_key,
                         read_json, seal, seed, write_frame, write_json, verify)


def patient_partitions(obs, settings):
    subjects = sorted(obs.patient.unique())
    if len(subjects) != 5:
        raise ValueError("This protocol requires the five lung subjects")
    for patient in subjects:
        part = obs[obs.patient == patient]
        if set(zip(part.visit, part.outcome)) != {(v, y) for v in ("pre", "post") for y in (0, 1)}:
            raise ValueError("Each subject needs both TMRE gates at both visits")
    # Seeded uniform whole-subject reservation; never selected by outcome performance.
    calibration = np.random.default_rng(seed(settings, "calibration patient")).choice(subjects)
    table = obs.groupby("patient", observed=True).agg(n_cells=("outcome", "size"), high_cells=("outcome", "sum"))
    table["partition"] = np.where(table.index == calibration, "calibration", "pool")
    return table.reset_index()


def repetition_split(patients, settings, repetition):
    pool = patients[patients.partition == "pool"].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed(settings, "patient split", repetition))
    order = rng.permutation(len(pool))
    roles = np.full(len(pool), "train", object)
    roles[order[0]] = "test"
    roles[order[1]] = "dev"
    pool["role"] = roles
    if len(pool) != 4:
        raise ValueError("Expected four non-calibration subjects")
    return pool


def select_cells(data, patients, selection_seed, settings):
    """Equal gate counts per patient/visit, capped by the smaller gate and configured budget."""
    rng = np.random.default_rng(selection_seed)
    obs = data.obs[data.obs.patient.isin(patients)]
    indices = []
    for _, group in obs.groupby(["patient", "visit"], sort=True, observed=True):
        high, low = group[group.outcome == 1], group[group.outcome == 0]
        count = min(len(high), len(low), settings["cells_per_gate_visit"])
        if count < 2:
            raise ValueError("Fewer than two eligible cells in a paired TMRE gate")
        for gate in (low, high):
            selected = rng.choice(gate.index.to_numpy(), size=count, replace=False)
            indices.extend(data.obs_names.get_indexer(selected).tolist())
    indices = np.sort(np.asarray(indices, dtype=int))
    if len(indices) == 0 or len(set(indices)) != len(indices) or (indices < 0).any():
        raise ValueError("Invalid selected cell indices")
    return indices


def fit_probe(x, obs, c, settings):
    if obs.outcome.nunique() != 2:
        raise ValueError("Both TMRE classes required in training")
    # Each subject/visit/gate has equal total training weight, regardless of recovered cells.
    sizes = obs.groupby(["patient", "visit", "outcome"], observed=True).outcome.transform("size").to_numpy()
    weights = 1.0 / sizes
    weights *= len(weights) / weights.sum()
    model = make_pipeline(StandardScaler(with_mean=not sparse.issparse(x)), LogisticRegression(
        C=float(c), solver="liblinear", max_iter=settings["max_iter"], random_state=settings["seed"]))
    with warnings.catch_warnings(), threadpool_limits(limits=settings["threads"]):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, obs.outcome.to_numpy(), logisticregression__sample_weight=weights)
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
    indices = select_cells(data, cal_patients, seed(settings, "calibration cells"), settings)
    obs = data.obs.iloc[indices]
    write_frame(directory / "selected_cells.tsv", obs.reset_index())
    selected, rows = {}, []
    total = len(settings["retention_rates"]) * (1 + len(METHODS) * 2 * len(settings["c_values"]))
    with progress(total, "One-time C calibration, held-out visits") as bar:
        for rate in settings["retention_rates"]:
            _, features = encoder.features(indices, rate, seed(settings, "calibration molecules", rate=rate))
            selected[rate_key(rate)] = {}
            bar.update(1)
            for method in METHODS:
                by_c = {float(c): [] for c in sorted(settings["c_values"])}
                for visit in ("pre", "post"):
                    train = np.flatnonzero(obs.visit.ne(visit))
                    dev = np.flatnonzero(obs.visit.eq(visit))
                    for c in by_c:
                        bar.set_postfix_str(f"keep={rate:g} {method}, validate={visit}, C={c:g}")
                        start = perf_counter()
                        model = fit_probe(features[method][train], obs.iloc[train], c, settings)
                        result = scores(obs.outcome.iloc[dev], model.predict_proba(features[method][dev])[:, 1])
                        by_c[c].append(result["auroc"])
                        rows.append({"retention_rate": rate, "method": method, "held_out_visit": visit,
                                     "C": c, "seconds": perf_counter() - start, **result})
                        bar.update(1)
                selected[rate_key(rate)][method] = max(by_c, key=lambda c: np.mean(by_c[c]))
            del features
    write_frame(directory / "tuning.tsv", pd.DataFrame(rows))
    write_json(directory / "selected_c.json", selected)
    seal(directory, {"calibration_patients": cal_patients, "n_cells": len(obs),
                     "selection_metric": "mean held-out-visit AUROC in one permanently reserved subject; ties smaller C",
                     "selection_seed": seed(settings, "calibration cells")})
    directory.rename(destination)
    return selected


def evaluate_predictions(obs, predictions, rate, repetition):
    rows = []
    for role, part in obs.groupby("role", sort=True, observed=True):
        for method in METHODS:
            base = {"repetition": repetition, "retention_rate": rate, "split": role, "method": method}
            visits, patients = [], []
            for (patient, visit), group in part.groupby(["patient", "visit"], sort=True, observed=True):
                result = scores(group.outcome, predictions.loc[group.index, method])
                if not np.isfinite(result["auroc"]):
                    raise ValueError("Each held-out visit must contain both outcomes")
                row = {**base, "scope": "visit", "patient": patient, "visit": visit,
                       "n_cells": len(group), "n_high": int(group.outcome.sum()), **result}
                visits.append(row)
            rows.extend(visits)
            for patient, group in part.groupby("patient", observed=True):
                own = [row for row in visits if row["patient"] == patient]
                patients.append({**base, "scope": "patient", "patient": patient, "visit": "ALL",
                    "n_cells": len(group), "n_high": int(group.outcome.sum()),
                    **{key: float(np.mean([row[key] for row in own])) for key in ("auroc", "average_precision")}})
            rows.extend(patients)
            rows.append({**base, "scope": "macro_patient", "patient": "ALL", "visit": "ALL",
                "n_cells": len(part), "n_high": int(part.outcome.sum()),
                **{key: float(np.mean([row[key] for row in patients])) for key in ("auroc", "average_precision")}})
            rows.append({**base, "scope": "pooled", "patient": "ALL", "visit": "ALL",
                "n_cells": len(part), "n_high": int(part.outcome.sum()),
                **scores(part.outcome, predictions.loc[part.index, method])})
    return rows


def collect_one(data, encoder, settings, root, patients, selected, repetition):
    directory = pending(root, f"replicate_{repetition:06d}")
    started = perf_counter()
    assignments = repetition_split(patients, settings, repetition)
    indices = select_cells(data, assignments.patient, seed(settings, "selected cells", repetition), settings)
    obs = data.obs.iloc[indices].copy()
    obs["role"] = obs.patient.map(assignments.set_index("patient").role)
    if obs.role.isna().any():
        raise ValueError("Missing subject role")
    write_frame(directory / "patient_assignments.tsv", assignments)
    write_frame(directory / "selected_cells.tsv", obs.reset_index())
    train = np.flatnonzero(obs.role == "train")
    evaluation = np.flatnonzero(obs.role != "train")
    eval_obs = obs.iloc[evaluation]
    rows, audits, fits = [], [], []
    with progress(len(settings["retention_rates"]) * (1 + len(METHODS)) + 1,
                  f"Repetition {repetition + 1}") as bar:
        for rate in settings["retention_rates"]:
            draw = seed(settings, "repetition molecules", repetition, rate)
            counts, features = encoder.features(indices, rate, draw)
            bar.update(1)
            for (role, patient, visit, outcome), group in obs.groupby(["role", "patient", "visit", "outcome"], observed=True):
                local = obs.index.get_indexer(group.index)
                x = counts[local]
                totals = np.asarray(x.sum(axis=1)).ravel()
                audits.append({"repetition": repetition, "retention_rate": rate, "split": role,
                    "patient": patient, "visit": visit, "outcome": outcome, "dilution_seed": draw,
                    "n_cells": len(group), "total_umis": int(totals.sum()),
                    "median_umis": float(np.median(totals)), "empty_cells": int((totals == 0).sum()),
                    "counts_sha256": digest([x.indptr.tolist(), x.indices.tolist(), x.data.tolist()])})
            predictions = pd.DataFrame(index=eval_obs.index)
            fit_settings = {**settings, "seed": seed(settings, "classifier fit", repetition)}
            for method in METHODS:
                c = selected[rate_key(rate)][method]
                bar.set_postfix_str(f"keep={rate:g}: {method}, fresh sklearn fit C={c:g}")
                tick = perf_counter()
                model = fit_probe(features[method][train], obs.iloc[train], c, fit_settings)
                predictions[method] = model.predict_proba(features[method][evaluation])[:, 1]
                fits.append({"retention_rate": rate, "method": method, "C": c,
                    "fit_seed": fit_settings["seed"], "n_train_cells": len(train),
                    "n_features": features[method].shape[1], "seconds": perf_counter() - tick})
                if settings["save_models"]:
                    path = directory / "probes" / rate_key(rate) / f"{method}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, path)
                bar.update(1)
            rows.extend(evaluate_predictions(eval_obs, predictions, rate, repetition))
            write_frame(directory / "predictions" / f"{rate_key(rate)}.tsv.gz", eval_obs.join(predictions).reset_index())
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
            write_json(root / "prepared.json", {"directory": str(prepared_path), "receipt": prepared})
            patients = patient_partitions(data.obs, settings)
            write_frame(root / "patient_partitions.tsv", patients)
            encoder = Encoder(data, settings, root, model_factory)
            write_json(root / "status.json", {"state": "calibrating C", "completed_repetitions": index})
            selected = calibrate(data, encoder, settings, root, patients)
            # Resuming existing calibration does not initialize the encoder/model.
            encoder.ensure_model()
            milestone = max_replicates or max(settings["progress_target"],
                (index // settings["progress_target"] + 1) * settings["progress_target"])
            overall = progress(milestone - index, f"Repetitions remaining to {milestone}")
            session_start = perf_counter()
            session_initial = index
            while max_replicates is None or index < max_replicates:
                write_json(root / "status.json", {"state": "collecting", "completed_repetitions": index,
                    "current_repetition": index, "eta_target": milestone})
                directory = collect_one(data, encoder, settings, root, patients, selected, index)
                index += 1
                overall.update(1)
                mean = (perf_counter() - session_start) / (index - session_initial)
                remaining = max(0, milestone - index) * mean
                write_json(root / "status.json", {"state": "collecting", "completed_repetitions": index,
                    "eta_target": milestone, "session_seconds_per_repetition": mean,
                    "estimated_seconds_remaining": remaining})
                seconds = read_json(directory / "complete.json")["elapsed_seconds"]
                tqdm.write(f"Saved repetition {index} ({seconds / 60:.2f} min). "
                           f"Estimated {remaining / 60:.1f} min to {milestone}. Ctrl+C stops safely.")
                if max_replicates is None and index >= milestone:
                    overall.close()
                    milestone += settings["progress_target"]
                    overall = progress(settings["progress_target"], f"Repetitions remaining to {milestone}")
            write_json(root / "status.json", {"state": "target reached", "completed_repetitions": index})
        except KeyboardInterrupt:
            count = len(completed(root, checksums=False))
            write_json(root / "status.json", {"state": "stopped", "completed_repetitions": count})
            tqdm.write(f"Stopped; {count} completed repetitions preserved. Run python build_report.py")
        except Exception as exc:
            write_json(root / "status.json", {"state": "error", "completed_repetitions": len(completed(root, checksums=False)),
                                               "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            if overall is not None:
                overall.close()
    return root
