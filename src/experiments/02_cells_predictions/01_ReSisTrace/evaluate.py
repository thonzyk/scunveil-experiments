"""Nested, lineage-grouped tuning inside each replicate-held-out comparison."""
from __future__ import annotations

import warnings
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm.auto import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CONDITIONS, digest, read_json, run_directory, sha256_file, write_json, write_table
from representations import log1p_counts, verify_view

METHODS = ("scunveil", "raw_log1p", "depth_qc")
METRICS = ("average_precision", "auroc", "balanced_accuracy", "brier")


def group_predictions(obs, probabilities):
    frame = obs[["group", "outcome"]].copy()
    if frame.groupby("group", observed=True)["outcome"].nunique().max() > 1:
        raise ValueError("A lineage group has conflicting outcomes")
    for name, values in probabilities.items():
        frame[name] = np.asarray(values)
    return frame.groupby("group", sort=True, observed=True).agg(
        outcome=("outcome", "first"), n_cells=("outcome", "size"),
        **{name: (name, "mean") for name in probabilities},
    ).reset_index()


def training_weights(obs):
    """Each known lineage has equal weight; the two classes have equal totals."""
    groups = obs.groupby("group", observed=True)["outcome"].agg(["first", "nunique", "size"])
    if groups["nunique"].max() > 1 or groups["first"].nunique() != 2:
        raise ValueError("Training requires consistent lineage labels and both classes")
    class_counts = groups["first"].value_counts()
    return np.array([
        len(groups) / (2 * class_counts.loc[label] * groups.loc[group, "size"])
        for group, label in zip(obs["group"], obs["outcome"])
    ], dtype=np.float64)


def inner_splits(obs, config):
    if obs.groupby("group", observed=True)["outcome"].nunique().max() > 1:
        raise ValueError("Mixed labels within a lineage group")
    group_counts = obs.drop_duplicates("group")["outcome"].value_counts()
    if len(group_counts) != 2 or group_counts.min() < config["inner_folds"]:
        raise ValueError("Too few independent positive/negative groups for inner CV")
    splitter = StratifiedGroupKFold(
        n_splits=config["inner_folds"], shuffle=True, random_state=config["seed"],
    )
    splits = list(splitter.split(np.zeros(len(obs)), obs["outcome"], obs["group"]))
    for train, dev in splits:
        if set(obs.iloc[train]["group"]) & set(obs.iloc[dev]["group"]):
            raise RuntimeError("Lineage leakage in inner validation")
        if obs.iloc[train]["outcome"].nunique() != 2 or obs.iloc[dev]["outcome"].nunique() != 2:
            raise ValueError("An inner fold lacks a class; reduce inner_folds")
    return splits


def fit_probe(x, obs, c, config):
    model = make_pipeline(StandardScaler(with_mean=not issparse(x)), LogisticRegression(
        C=float(c), solver="lbfgs", max_iter=config["max_iter"], random_state=config["seed"],
    ))
    # An unconverged fit must not silently enter the comparison.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, obs["outcome"].to_numpy(), logisticregression__sample_weight=training_weights(obs))
    return model


def predict_probe(model, x, train_obs):
    """Undo the balanced-class training prior using training lineages only.

    This odds adjustment preserves AP/AUROC rankings. It does not guarantee
    calibration under replicate shift; Brier remains a descriptive diagnostic.
    """
    probability = model.predict_proba(x)[:, 1]
    prior = train_obs.drop_duplicates("group")["outcome"].mean()
    numerator = probability * prior
    return numerator / (numerator + (1 - probability) * (1 - prior))


def scores(labels, probabilities):
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are needed for evaluation")
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= 0.5)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def paired_bootstrap(grouped, repeats, seed, show_progress=False):
    """Paired stratified lineage bootstrap CONDITIONAL on the test replicate.

    Models are fixed; this is not an interval over future experiments, training
    sets, or the model-selection process. Keep that distinction in every report.
    """
    rng = np.random.default_rng(seed)
    y = grouped["outcome"].to_numpy()
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    arrays = {method: grouped[method].to_numpy() for method in METHODS}
    values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    for _ in tqdm(range(repeats), desc="Lineage bootstrap", unit="draw", position=2,
                  leave=False, disable=not show_progress or repeats == 0, smoothing=0, dynamic_ncols=True):
        indices = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in classes])
        for method in METHODS:
            for metric, value in scores(y[indices], arrays[method][indices]).items():
                values[method][metric].append(value)
    intervals, differences = [], []
    for method in METHODS:
        point = scores(y, arrays[method])
        for metric in METRICS:
            row = {"method": method, "metric": metric, "value": point[metric]}
            if repeats:
                row.update(zip(("ci_low", "ci_high"), np.quantile(values[method][metric], [0.025, 0.975]).tolist()))
            intervals.append(row)
    for baseline in ("raw_log1p", "depth_qc"):
        for metric in METRICS:
            row = {
                "baseline": baseline, "metric": metric,
                "difference": scores(y, arrays["scunveil"])[metric] - scores(y, arrays[baseline])[metric],
            }
            if repeats:
                distribution = np.asarray(values["scunveil"][metric]) - values[baseline][metric]
                row.update(zip(("ci_low", "ci_high"), np.quantile(distribution, [0.025, 0.975]).tolist()))
            differences.append(row)
    return pd.DataFrame(intervals), pd.DataFrame(differences)


def permute_groups(obs, seed):
    """Shuffle labels between released lineage groups within each replicate."""
    result = obs.copy()
    rng = np.random.default_rng(seed)
    for _, subset in obs.groupby(["condition", "replicate"], sort=True, observed=True):
        labels = subset.groupby("group", sort=True, observed=True)["outcome"].first()
        shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
        result.loc[subset.index, "outcome"] = subset["group"].map(shuffled).to_numpy()
    return result


def select_training_cells(obs, indices, fraction, seed):
    """Approximately fraction of training cells, stratified by lineage outcome.

    Sample whole known lineages in a seeded random order. Choose the prefix
    whose cell count is closest to the class-specific requested budget. Never
    consult the held-out replicate or use unavailable cells for inner tuning.
    """
    if fraction == 1:
        return indices.copy()
    selected = []
    subset = obs.iloc[indices]
    groups = subset.groupby("group", observed=True, sort=True)["outcome"].agg(["first", "size", "nunique"])
    if groups["nunique"].max() > 1:
        raise ValueError("Conflicting outcomes in a training lineage")
    for label in (0, 1):
        candidates = groups[groups["first"] == label]
        token = digest({"seed": seed, "label": label, "groups": candidates.index.tolist()})
        rng = np.random.default_rng(int(token[:16], 16))
        order = rng.permutation(len(candidates))
        cumulative = np.cumsum(candidates["size"].to_numpy()[order])
        if not len(cumulative):
            raise ValueError("Training replicate lacks an outcome class")
        target = fraction * int(candidates["size"].sum())
        count = int(np.argmin(np.abs(cumulative - target))) + 1
        selected.extend(candidates.index[order[:count]])
    return indices[subset["group"].isin(selected).to_numpy()]


def evaluate_fold(obs, x, fixed, condition, test_replicate, config, directory,
                  fit_progress=None, training_features=None):
    started = perf_counter()
    train = np.flatnonzero((obs["condition"] == condition) & (obs["replicate"] != str(test_replicate)))
    test = np.flatnonzero((obs["condition"] == condition) & (obs["replicate"] == str(test_replicate)))
    if not len(train) or not len(test):
        raise ValueError(f"Missing replicate for {condition}")
    all_train = train.copy()
    train = select_training_cells(obs, train, config["train_cell_fraction"], config["seed"])
    train_obs, test_obs = obs.iloc[train], obs.iloc[test]
    if set(train_obs["group"]) & set(test_obs["group"]):
        raise ValueError("Lineage groups overlap between outer training and test")
    if train_obs["outcome"].nunique() != 2 or test_obs["outcome"].nunique() != 2:
        raise ValueError(f"Both replicates of {condition} must contain both outcome classes")
    reused = config.get("fixed_regularization")
    cv = inner_splits(train_obs, config) if reused is None else []
    directory.mkdir(parents=True, exist_ok=True)
    write_table(directory / "training_cells.tsv", obs.iloc[all_train][["group", "outcome"]].assign(
        available_for_training=np.isin(all_train, train),
    ).reset_index())
    split_rows = []
    for fold, (a, b) in enumerate(cv):
        for role, indices in (("train", train[a]), ("dev", train[b])):
            split_rows.extend({"inner_fold": fold, "role": role, "cell_id": obs.index[i], "group": obs.iloc[i]["group"]} for i in indices)
    write_table(directory / "inner_splits.tsv", pd.DataFrame(split_rows, columns=["inner_fold", "role", "cell_id", "group"]))
    tuning, selected, final_fit_seconds = [], {}, {}
    features = {**fixed, "raw_log1p": x}
    train_features = features if training_features is None else training_features
    if reused is not None and (set(reused) != set(METHODS) or any(c not in config["c_values"] for c in reused.values())):
        raise ValueError("Cached regularization does not match methods/grid")
    for method in METHODS:
        if reused is not None:
            selected[method] = float(reused[method])
            continue
        by_c = {float(c): [] for c in config["c_values"]}
        for fold, (a, b) in enumerate(cv):
            a_x, b_x = train_features[method][train[a]], train_features[method][train[b]]
            for c in sorted(by_c):
                if fit_progress is not None:
                    fit_progress.set_postfix_str(f"{condition}/rep{test_replicate} {method} inner={fold + 1} C={c:g}")
                fit_started = perf_counter()
                probe = fit_probe(a_x, train_obs.iloc[a], c, config)
                probabilities = predict_probe(probe, b_x, train_obs.iloc[a])
                grouped = group_predictions(train_obs.iloc[b], {method: probabilities})
                score = float(average_precision_score(grouped["outcome"], grouped[method]))
                by_c[c].append(score)
                tuning.append({"method": method, "inner_fold": fold, "C": c, "lineage_average_precision": score,
                               "elapsed_seconds": perf_counter() - fit_started})
                if fit_progress is not None:
                    fit_progress.update(1)
        # Deterministic tie-break favors stronger regularization.
        selected[method] = max(sorted(by_c), key=lambda c: float(np.mean(by_c[c])))
    probabilities = {}
    for method in METHODS:
        a_x, b_x = train_features[method][train], features[method][test]
        if fit_progress is not None:
            fit_progress.set_postfix_str(f"{condition}/rep{test_replicate} {method} final C={selected[method]:g}")
        fit_started = perf_counter()
        probe = fit_probe(a_x, train_obs, selected[method], config)
        probabilities[method] = predict_probe(probe, b_x, train_obs)
        final_fit_seconds[method] = perf_counter() - fit_started
        if fit_progress is not None:
            fit_progress.update(1)
        # Pickles are generated locally, never downloaded.
        import joblib
        joblib.dump(probe, directory / f"probe_{method}.joblib")
    predictions = test_obs[["condition", "replicate", "group", "outcome"]].copy()
    for method, values in probabilities.items():
        predictions[method] = values
    write_table(directory / "cell_predictions.tsv", predictions.reset_index())
    grouped = group_predictions(test_obs, probabilities)
    write_table(directory / "lineage_predictions.tsv", grouped)
    write_table(directory / "tuning.tsv", pd.DataFrame(tuning, columns=["method", "inner_fold", "C", "lineage_average_precision", "elapsed_seconds"]))
    lineage_metrics, differences = paired_bootstrap(
        grouped, config["bootstrap_repeats"], config["seed"] + test_replicate,
        show_progress=fit_progress is not None,
    )
    write_table(directory / "lineage_metrics.tsv", lineage_metrics)
    write_table(directory / "paired_differences.tsv", differences)
    cell_metrics = pd.DataFrame([
        {"method": method, **scores(test_obs["outcome"].to_numpy(), probabilities[method])} for method in METHODS
    ])
    write_table(directory / "cell_metrics.tsv", cell_metrics)
    write_json(directory / "complete.json", {
        "condition": condition, "test_replicate": test_replicate,
        "train_cells": len(train), "test_cells": len(test),
        "all_train_cells": len(all_train),
        "train_cell_fraction_requested": config["train_cell_fraction"],
        "train_cell_fraction_actual": len(train) / len(all_train),
        "train_umi_keep_probability": config.get("retention_rate", config["train_umi_keep_probability"]),
        "test_umi_keep_probability": config.get("retention_rate", 1.0),
        "inner_validation_umi_keep_probability": config.get("retention_rate", config["train_umi_keep_probability"]),
        "train_groups": train_obs["group"].nunique(), "test_groups": len(grouped),
        "test_prevalence": float(grouped["outcome"].mean()),
        "selected_C": selected, "raw_feature_count": x.shape[1],
        "tuning_reused": reused is not None, "tuning_source": config.get("tuning_source"),
        "final_fit_seconds": final_fit_seconds, "elapsed_seconds": perf_counter() - started,
        "raw_transform": "log1p(raw UMI counts); no library-size normalization",
        "scaling": "training-only per-feature standard deviation; sparse inputs are not centered",
        "interval_scope": "stratified lineage resampling conditional on this replicate and fixed fitted models",
        "files": {name: sha256_file(directory / name) for name in (
            "cell_predictions.tsv", "lineage_predictions.tsv", "lineage_metrics.tsv",
            "paired_differences.tsv", "cell_metrics.tsv", "tuning.tsv", "inner_splits.tsv",
            "training_cells.tsv",
        )},
    })


def evaluate(config):
    run = run_directory(config)
    tqdm.write(f"Run directory: {run}")
    jobs = [(str(depth), "observed" if permutation == 0 else f"permutation_{permutation:04d}", condition, replicate)
            for depth in config["depths"] for permutation in range(config["permutation_repeats"] + 1)
            for condition in CONDITIONS for replicate in (1, 2)]
    pending = sum(not (run / "evaluation" / depth / name / condition / f"test_rep{replicate}" / "complete.json").exists()
                  for depth, name, condition, replicate in jobs)
    fits_per_fold = len(METHODS) * (config["inner_folds"] * len({float(c) for c in config["c_values"]}) + 1)
    started = perf_counter()
    with tqdm(total=len(jobs), desc="Evaluation folds", unit="fold", position=0, dynamic_ncols=True) as folds, \
         tqdm(total=pending * fits_per_fold, desc="Pending classifier fits", unit="fit", position=1, dynamic_ncols=True) as fits:
        for depth in config["depths"]:
            view_path = run / "views" / str(depth)
            manifest = verify_view(view_path)
            view = ad.read_h5ad(view_path / "counts.h5ad")
            view.obs["replicate"] = view.obs["replicate"].astype(str)
            if digest(view.obs_names.tolist()) != manifest["cell_ids_sha256"]:
                raise ValueError("Cached embedding/count observation order mismatch")
            fixed = {method: np.load(view_path / f"{method}.npy", mmap_mode="r") for method in ("scunveil", "depth_qc")}
            if any(values.shape[0] != view.n_obs for values in fixed.values()):
                raise ValueError("Representation row count differs from RNA")
            x = log1p_counts(view.X)
            training_features = None
            training_view = None
            if config["train_umi_keep_probability"] < 1:
                training_path = run / "training_views" / str(depth)
                training_manifest = verify_view(training_path)
                if training_manifest["cell_ids_sha256"] != manifest["cell_ids_sha256"]:
                    raise ValueError("Training/test-view cell order differs")
                if training_manifest["umi_keep_probability"] != config["train_umi_keep_probability"]:
                    raise ValueError("Training view has an unexpected UMI keep probability")
                training_view = ad.read_h5ad(training_path / "counts.h5ad")
                if digest(training_view.obs_names.tolist()) != manifest["cell_ids_sha256"]:
                    raise ValueError("Training count rows differ from cached embedding rows")
                training_features = {"raw_log1p": log1p_counts(training_view.X)}
                training_features.update({
                    method: np.load(training_path / f"{method}.npy", mmap_mode="r")
                    for method in ("scunveil", "depth_qc")
                })
                if any(values.shape[0] != view.n_obs for values in training_features.values()):
                    raise ValueError("Training representation row count differs from test view")
            for permutation in range(config["permutation_repeats"] + 1):
                name = "observed" if permutation == 0 else f"permutation_{permutation:04d}"
                obs = view.obs if permutation == 0 else permute_groups(view.obs, config["seed"] + permutation)
                fold_config = config if permutation == 0 else {**config, "bootstrap_repeats": 0}
                for condition in CONDITIONS:
                    for test_replicate in (1, 2):
                        label = f"{depth}/{name}/{condition}/test_rep{test_replicate}"
                        directory = run / "evaluation" / str(depth) / name / condition / f"test_rep{test_replicate}"
                        folds.set_postfix_str(label)
                        if (directory / "complete.json").exists():
                            record = read_json(directory / "complete.json")
                            for filename, expected in record["files"].items():
                                if sha256_file(directory / filename) != expected:
                                    raise ValueError(f"Corrupted completed fold: {directory}")
                            tqdm.write(f"Verified cached {label}")
                        else:
                            evaluate_fold(
                                obs, x, fixed, condition, test_replicate, fold_config, directory,
                                fit_progress=fits, training_features=training_features,
                            )
                        folds.update(1)
    write_json(run / "evaluation_timing.json", {
        "elapsed_seconds": perf_counter() - started, "folds": len(jobs),
        "newly_evaluated_folds": pending, "classifier_fits": pending * fits_per_fold,
        "note": "ETA is based on completed work; methods/folds have different costs. Bootstrap ETA is separate.",
    })
    return run
