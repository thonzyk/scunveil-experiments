"""Fresh grouped splits, molecule draws and sklearn fits; fixed calibration C."""
from __future__ import annotations

from pathlib import Path
import shutil
from time import perf_counter

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from common import CONDITIONS, digest, read_json, write_json
from evaluate import METHODS, fit_probe, group_predictions, predict_probe, scores
from prepare import prepare
from representations import binomial_umi_view, depth_qc, log1p_counts, make_model
from replication_common import (collection_directory, completed_replicates, exclusive_lock,
                                random_seed, rate_key, seal, verify, write_array, write_frame)

BAR = "{desc} | LEFT {remaining} | elapsed {elapsed} | {n_fmt}/{total_fmt} {bar:10} | {postfix}"


def progress_bar(total, description):
    return tqdm(total=total, desc=description, smoothing=0, dynamic_ncols=False,
                bar_format=BAR, mininterval=1)


def group_table(obs):
    columns = ["condition", "replicate", "outcome"]
    if (obs.groupby("group", observed=True)[columns].nunique() != 1).any().any():
        raise ValueError("Each lineage must have one condition, replicate and outcome")
    groups = obs.groupby("group", observed=True, sort=True).agg(
        condition=("condition", "first"), replicate=("replicate", "first"),
        outcome=("outcome", "first"), n_cells=("outcome", "size"))
    groups["stratum"] = (groups.condition.astype(str) + ":" + groups.replicate.astype(str)
                         + ":" + groups.outcome.astype(str))
    return groups


def partition_calibration(obs, settings):
    groups = group_table(obs)
    pool, calibration = train_test_split(
        np.arange(len(groups)), test_size=settings["calibration_fraction"],
        stratify=groups.stratum, random_state=random_seed(settings, "calibration partition"))
    groups["partition"] = "pool"
    groups.iloc[calibration, groups.columns.get_loc("partition")] = "calibration"
    if set(groups.iloc[pool].index) & set(groups.iloc[calibration].index):
        raise RuntimeError("Calibration partition overlap")
    return groups.reset_index()


def repetition_split(groups, settings, replicate):
    pool = groups[groups.partition == "pool"].copy().reset_index(drop=True)
    fitting, test = train_test_split(
        np.arange(len(pool)), test_size=settings["test_fraction"], stratify=pool.stratum,
        random_state=random_seed(settings, "test split", replicate))
    train, dev = train_test_split(
        fitting, test_size=settings["dev_fraction"] / (1 - settings["test_fraction"]),
        stratify=pool.iloc[fitting].stratum,
        random_state=random_seed(settings, "train dev split", replicate))
    pool["role"] = ""
    for role, indices in (("train", train), ("dev", dev), ("test", test)):
        pool.iloc[indices, pool.columns.get_loc("role")] = role
    if not pool.role.ne("").all() or set(pool.group) & set(groups[groups.partition == "calibration"].group):
        raise RuntimeError("Incomplete or leaking repetition partition")
    # Every source stratum must occur in every role, for usable within-sample
    # diagnostics as well as the pooled score. Fail rather than silently retry.
    for role in ("train", "dev", "test"):
        if set(pool.loc[pool.role == role, "stratum"]) != set(pool.stratum):
            raise ValueError("Too few lineage groups for stratified train/dev/test splitting")
    return pool


def calibration_cv(obs, settings, condition):
    groups = group_table(obs)
    if groups.stratum.value_counts().min() < settings["calibration_folds"]:
        raise ValueError(f"Too few calibration lineages per replicate/outcome for {condition}; "
                         "increase calibration_fraction or reduce calibration_folds")
    splitter = StratifiedKFold(n_splits=settings["calibration_folds"], shuffle=True,
                               random_state=random_seed(settings, "calibration CV"))
    result = []
    for train, dev in splitter.split(np.zeros(len(groups)), groups.stratum):
        a = np.flatnonzero(obs.group.isin(groups.index[train]).to_numpy())
        b = np.flatnonzero(obs.group.isin(groups.index[dev]).to_numpy())
        if set(obs.iloc[a].group) & set(obs.iloc[b].group):
            raise RuntimeError("Lineage overlap in calibration CV")
        result.append((a, b))
    return result


class FrozenEncoder:
    def __init__(self, data, settings, root, factory):
        self.data, self.settings, self.root, self.factory = data, settings, root, factory
        self.model = None
        self.dimension = None
        self.full = None

    def ensure_model(self):
        if self.model is None:
            # The legacy factory checks a requested minimum. The actual output
            # width below always comes from the checkpoint, never this argument.
            model, info = self.factory({**self.settings, "embedding_dim": 2048})
            self.dimension = int(model.config.emb_dim)
            model.verbose = False
            path = self.root / "model.json"
            if path.exists():
                previous = read_json(path)
                if previous["snapshot_revision"] != info["snapshot_revision"] or previous["embedding_dim"] != self.dimension:
                    raise ValueError("Checkpoint snapshot/width changed during collection")
            write_json(path, {**info, "embedding_dim": self.dimension,
                              "embedding_policy": "all native features; no truncation"})
            self.model = model

    def embeddings(self, counts, obs):
        self.ensure_model()
        view = ad.AnnData(X=counts, obs=obs.copy(), var=self.data.var.copy())
        result = self.model.process_anndata(view, batch_size=self.settings["batch_size"])
        values = result.get_embeddings(n_features=self.dimension).astype(np.float32)
        if values.shape != (len(obs), self.dimension) or not np.isfinite(values).all():
            raise ValueError("Invalid full-width embeddings")
        return values, result.gene_mapping_summary

    def ensure_full_cache(self):
        directory = self.root / "full_depth_cache"
        if (directory / "complete.json").exists():
            info = verify(directory)
            if info["cell_ids_sha256"] != digest(self.data.obs_names.tolist()):
                raise ValueError("Full-depth embedding/count row mismatch")
            if read_json(self.root / "model.json")["embedding_dim"] != info["embedding_dim"]:
                raise ValueError("Full-depth cache/model width mismatch")
            self.dimension = info["embedding_dim"]
        else:
            directory.mkdir(parents=True, exist_ok=True)
            with progress_bar(1, "One-time full-depth embeddings") as progress:
                progress.set_postfix_str("all pretrained features; frozen encoder")
                values, mapping = self.embeddings(self.data.X, self.data.obs)
                write_array(directory / "scunveil.npy", values)
                seal(directory, {"embedding_dim": self.dimension,
                     "cell_ids_sha256": digest(self.data.obs_names.tolist()), "gene_mapping": mapping})
                progress.update(1)
        self.full = np.load(directory / "scunveil.npy", mmap_mode="r")
        if self.full.shape != (self.data.n_obs, self.dimension):
            raise ValueError("Invalid full-depth embedding cache")

    def features(self, indices, rate, draw_seed):
        obs = self.data.obs.iloc[indices]
        original = self.data.X[indices]
        counts = binomial_umi_view(original, obs.index, rate, draw_seed)
        if rate == 1:
            embeddings = self.full[indices]
        else:
            embeddings, _ = self.embeddings(counts, obs)
        values = {"scunveil": embeddings, "raw_log1p": log1p_counts(counts), "depth_qc": depth_qc(counts)}
        if any(x.shape[0] != len(obs) for x in values.values()):
            raise ValueError("Unpaired representation rows")
        return counts, values


def fresh_pending(root, name):
    directory = root / "pending" / name
    # Only unpublished scratch work is replaced. Completed repetitions are
    # immutable and live under a different directory.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def calibrate(data, encoder, settings, root):
    destination = root / "calibration"
    if destination.exists():
        verify(destination)
        return (pd.read_csv(destination / "lineage_partitions.tsv", sep="\t", dtype={"replicate": str}),
                read_json(destination / "selected_c.json"))
    directory = fresh_pending(root, "calibration")
    groups = partition_calibration(data.obs, settings)
    write_frame(directory / "lineage_partitions.tsv", groups)
    calibration_ids = set(groups.loc[groups.partition == "calibration", "group"])
    indices = np.flatnonzero(data.obs.group.isin(calibration_ids).to_numpy())
    obs = data.obs.iloc[indices]
    condition_splits, split_rows = {}, []
    for condition in CONDITIONS:
        local = np.flatnonzero(obs.condition.to_numpy() == condition)
        condition_splits[condition] = (local, calibration_cv(obs.iloc[local], settings, condition))
        for fold, (train, dev) in enumerate(condition_splits[condition][1]):
            for role, ix in (("train", train), ("dev", dev)):
                frame = obs.iloc[local[ix]][["condition", "replicate", "group", "outcome"]].reset_index()
                split_rows.append(frame.assign(calibration_fold=fold, role=role))
    write_frame(directory / "inner_splits.tsv.gz", pd.concat(split_rows, ignore_index=True))
    selected, tuning, measurement = {}, [], []
    n_fits = len(CONDITIONS) * len(METHODS) * settings["calibration_folds"] * len(settings["c_values"])
    started = perf_counter()
    with progress_bar(len(settings["retention_rates"]) * (n_fits + 1), "One-time C calibration") as progress:
        for rate in settings["retention_rates"]:
            draw_seed = random_seed(settings, "calibration molecules", rate=rate)
            progress.set_postfix_str(f"keep={rate:g}: shared calibration RNA / all features")
            counts, features = encoder.features(indices, rate, draw_seed)
            measurement.append({"retention_rate": rate, "dilution_seed": draw_seed,
                                "n_cells": len(obs), "input_umis": int(counts.sum())})
            progress.update(1)
            selected[rate_key(rate)] = {}
            for condition, (local, folds) in condition_splits.items():
                selected[rate_key(rate)][condition] = {}
                for method in METHODS:
                    by_c = {float(c): [] for c in settings["c_values"]}
                    for fold, (train, dev) in enumerate(folds):
                        a, b = local[train], local[dev]
                        for c in sorted(by_c):
                            progress.set_postfix_str(f"keep={rate:g} {condition} {method} fold={fold + 1} C={c:g}")
                            tick = perf_counter()
                            model = fit_probe(features[method][a], obs.iloc[a], c, settings)
                            p = predict_probe(model, features[method][b], obs.iloc[a])
                            grouped = group_predictions(obs.iloc[b], {method: p})
                            value = float(average_precision_score(grouped.outcome, grouped[method]))
                            by_c[c].append(value)
                            tuning.append({"retention_rate": rate, "condition": condition, "method": method,
                                           "fold": fold, "C": c, "lineage_ap": value,
                                           "seconds": perf_counter() - tick})
                            progress.update(1)
                    selected[rate_key(rate)][condition][method] = max(sorted(by_c), key=lambda c: np.mean(by_c[c]))
    write_frame(directory / "tuning.tsv", pd.DataFrame(tuning))
    write_frame(directory / "measurement_audit.tsv", pd.DataFrame(measurement))
    write_json(directory / "selected_c.json", selected)
    seal(directory, {"n_calibration_cells": len(obs), "n_calibration_groups": len(calibration_ids),
                     "elapsed_seconds": perf_counter() - started,
                     "scope": "Calibration lineages excluded from every repetition's train/dev/test sets"})
    directory.rename(destination)
    return groups, selected


def evaluate_predictions(obs, probabilities, condition, rate, replicate, role):
    grouped = group_predictions(obs, probabilities)
    source = obs.drop_duplicates("group").set_index("group")
    grouped["biological_replicate"] = grouped.group.map(source.replicate.astype(str))
    rows = []
    subsets = [("pooled", grouped)] + list(grouped.groupby("biological_replicate", sort=True, observed=True))
    for biological_replicate, subset in subsets:
        for method in METHODS:
            rows.append({"repetition": replicate, "retention_rate": rate, "condition": condition,
                         "split": role, "biological_replicate": str(biological_replicate), "method": method,
                         "n_cells": int(subset.n_cells.sum()), "n_lineages": len(subset),
                         "prevalence": float(subset.outcome.mean()), **scores(subset.outcome, subset[method])})
    return grouped, rows


def collect_one(data, encoder, settings, root, groups, selected, replicate):
    started = perf_counter()
    directory = fresh_pending(root, f"replicate_{replicate:06d}")
    assignments = repetition_split(groups, settings, replicate)
    indices = np.flatnonzero(data.obs.group.isin(assignments.group).to_numpy())
    obs = data.obs.iloc[indices].copy()
    obs["role"] = obs.group.map(assignments.set_index("group").role)
    write_frame(directory / "lineage_splits.tsv", assignments)
    write_frame(directory / "cell_splits.tsv.gz", obs[["condition", "replicate", "group", "outcome", "role"]].reset_index())
    fit_settings = {**settings, "seed": random_seed(settings, "probe fit", replicate)}
    rows, measurement, selected_rows, elapsed_fits = [], [], [], []
    n_steps = len(settings["retention_rates"]) * (1 + len(CONDITIONS) * len(METHODS)) + 1
    with progress_bar(n_steps, f"Repetition {replicate + 1}") as progress:
        for rate in settings["retention_rates"]:
            draw_seed = random_seed(settings, "repetition molecules", replicate, rate)
            progress.set_postfix_str(f"keep={rate:g}: fresh paired RNA / all scUNVEIL features")
            counts, features = encoder.features(indices, rate, draw_seed)
            progress.update(1)
            for role in ("train", "dev", "test"):
                ix = np.flatnonzero(obs.role.to_numpy() == role)
                depth = np.asarray(counts[ix].sum(axis=1)).ravel()
                measurement.append({"repetition": replicate, "retention_rate": rate, "split": role,
                                    "dilution_seed": draw_seed, "n_cells": len(ix), "input_umis": int(depth.sum()),
                                    "mean_input_umis": float(depth.mean()), "empty_cells": int((depth == 0).sum())})
            for condition in CONDITIONS:
                train = np.flatnonzero((obs.condition.to_numpy() == condition) & (obs.role.to_numpy() == "train"))
                evaluation = np.flatnonzero((obs.condition.to_numpy() == condition) & (obs.role.to_numpy() != "train"))
                train_obs = obs.iloc[train]
                predictions = {}
                for method in METHODS:
                    c = selected[rate_key(rate)][condition][method]
                    progress.set_postfix_str(f"keep={rate:g} {condition} {method} fresh fit C={c:g}")
                    tick = perf_counter()
                    model = fit_probe(features[method][train], train_obs, c, fit_settings)
                    predictions[method] = predict_probe(model, features[method][evaluation], train_obs)
                    elapsed_fits.append({"retention_rate": rate, "condition": condition, "method": method,
                                         "seconds": perf_counter() - tick})
                    if settings["save_models"]:
                        path = directory / "probes" / rate_key(rate) / condition / f"{method}.joblib"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(model, path)
                    selected_rows.append({"retention_rate": rate, "condition": condition, "method": method,
                                          "C": c, "n_features": features[method].shape[1]})
                    progress.update(1)
                evaluation_obs = obs.iloc[evaluation]
                frame = evaluation_obs[["condition", "replicate", "group", "outcome", "role"]].copy()
                for method, values in predictions.items():
                    frame[method] = values
                base = directory / "predictions" / rate_key(rate) / condition
                write_frame(base / "cells.tsv.gz", frame.reset_index())
                lineages = []
                for role in ("dev", "test"):
                    mask = evaluation_obs.role.to_numpy() == role
                    grouped, values = evaluate_predictions(evaluation_obs.iloc[np.flatnonzero(mask)],
                        {m: p[mask] for m, p in predictions.items()}, condition, rate, replicate, role)
                    rows.extend(values)
                    lineages.append(grouped.assign(split=role))
                write_frame(base / "lineages.tsv.gz", pd.concat(lineages, ignore_index=True))
            del features, counts
        write_frame(directory / "metrics.tsv", pd.DataFrame(rows))
        write_frame(directory / "measurement_audit.tsv", pd.DataFrame(measurement))
        write_frame(directory / "fit_times.tsv", pd.DataFrame(elapsed_fits))
        write_frame(directory / "selected_c.tsv", pd.DataFrame(selected_rows))
        seal(directory, {"replicate": replicate, "elapsed_seconds": perf_counter() - started,
                         "test_split_seed": random_seed(settings, "test split", replicate),
                         "train_dev_split_seed": random_seed(settings, "train dev split", replicate),
                         "fit_seed": fit_settings["seed"], "embedding_dim": encoder.dimension,
                         "runtime_settings": {k: settings[k] for k in ("threads", "device", "batch_size", "save_models")},
                         "n_pool_cells": len(obs), "n_train_cells": int((obs.role == "train").sum()),
                         "n_dev_cells": int((obs.role == "dev").sum()), "n_test_cells": int((obs.role == "test").sum()),
                         "selection_scope": "C fixed using separate calibration lineages; dev never used to fit final probes"})
        destination = root / "replicates" / directory.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace a completed repetition: {destination}")
        directory.rename(destination)
        progress.update(1)
    return destination


def collect(settings, max_replicates=None, model_factory=make_model):
    if max_replicates is not None and (type(max_replicates) is not int or max_replicates < 1):
        raise ValueError("max_replicates must be a positive total target")
    prepare(settings)
    root = collection_directory(settings)
    tqdm.write(f"Collection: {root}")
    with exclusive_lock(root / "collector.lock"):
        existing = completed_replicates(root)
        index = len(existing)
        if max_replicates is not None and index >= max_replicates:
            tqdm.write(f"Already have {index} completed repetitions; target reached.")
            return root
        data = ad.read_h5ad(Path(settings["output_root"]) / "prepared/counts.h5ad")
        data.obs["replicate"] = data.obs.replicate.astype(str)
        encoder = FrozenEncoder(data, settings, root, model_factory)
        try:
            write_json(root / "status.json", {"state": "calibrating", "completed_replicates": index})
            encoder.ensure_full_cache()
            groups, selected = calibrate(data, encoder, settings, root)
            while True:
                if max_replicates is not None and index >= max_replicates:
                    break
                write_json(root / "status.json", {"state": "collecting", "completed_replicates": index,
                                                  "current_replicate": index})
                path = collect_one(data, encoder, settings, root, groups, selected, index)
                index += 1
                seconds = read_json(path / "complete.json")["elapsed_seconds"]
                tqdm.write(f"Saved repetition {index}: {path} ({seconds / 60:.2f} min). "
                           "Ctrl+C stops; completed repetitions remain available for reporting.")
                write_json(root / "status.json", {"state": "between repetitions", "completed_replicates": index})
        except KeyboardInterrupt:
            # A signal may arrive immediately after the atomic commit, so count
            # committed directories instead of trusting the local counter.
            count = len(completed_replicates(root, verify_checksums=False))
            write_json(root / "status.json", {"state": "stopped", "completed_replicates": count})
            tqdm.write(f"Stopped. {count} completed repetitions preserved. Report with: "
                       f"python build_report.py --collection {root}")
            return root
        write_json(root / "status.json", {"state": "target reached", "completed_replicates": index})
    return root
