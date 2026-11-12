"""Paired molecule-retention protocol shared by the two biological benchmarks.

Imports of common/representations resolve to the invoking experiment directory.
The biological adapters retain their original grouped outer/inner evaluations.
"""
from __future__ import annotations

import hashlib
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import digest, read_json, run_directory, sha256_file, write_json, write_table


def seed(config, repeat, stream, rate=None):
    token = digest({"seed": config["seed"], "repeat": repeat, "stream": stream, "rate": rate})
    return int(token[:8], 16)


def rate_name(rate):
    return f"keep_{float(rate)}"


def view_directory(run, rate, repeat):
    # Full-depth RNA is identical in every repeat, including repeated tuning.
    return run / "views" / rate_name(rate) / f"repeat_{0 if rate == 1 else repeat:03d}"


def view_jobs(config, repeats=None):
    return [(rate, repeat) for rate in config["retention_rates"]
            for repeat in range(1 if rate == 1 else config["repeats"] if repeats is None else repeats)]


def evaluation_jobs(config, repeats=None):
    return view_jobs(config, repeats)


def planned_repeats(config, run):
    path = run / "runtime_plan.json"
    return read_json(path)["effective_repeats"] if path.exists() else config["repeats"]


def thin(counts, cell_ids, probability, draw_seed):
    """Independent binomial molecule retention, stable under row reordering.

    Empty sampled cells remain in the benchmark. Rates and repetitions use
    independent random streams; all representations consume this exact matrix.
    """
    if not np.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    result = counts.tocsr(copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    if len(cell_ids) != result.shape[0]:
        raise ValueError("Count rows and cell IDs differ")
    if not np.isfinite(result.data).all() or np.any(result.data < 0) or np.any(result.data != np.floor(result.data)):
        raise ValueError("Thinning requires nonnegative integer UMI counts")
    if probability == 1:
        return result
    for i, cell in enumerate(cell_ids):
        a, b = result.indptr[i:i + 2]
        token = hashlib.sha256(f"{draw_seed}:{probability}:{cell}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
        result.data[a:b] = rng.binomial(result.data[a:b].astype(np.int64), probability)
    result.eliminate_zeros()
    return result


def verify(directory):
    info = read_json(directory / "complete.json")
    for filename, expected in info["files"].items():
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"Corrupted retention artifact: {directory / filename}")
    return info


def embed(config, load_data, model_factory, mask=None, jobs=None, overall=None):
    run = run_directory(config)
    tqdm.write(f"Run directory: {run}")
    data = model = None
    jobs = view_jobs(config, planned_repeats(config, run)) if jobs is None else jobs
    with tqdm(total=len(jobs), desc="RNA views / embeddings", unit="view", position=1 if overall else 0,
              smoothing=0, dynamic_ncols=True) as progress:
        for rate, repeat in jobs:
            output = view_directory(run, rate, repeat)
            progress.set_postfix_str(f"keep={rate:g}, repeat={repeat + 1}")
            if (output / "complete.json").exists():
                verify(output)
                progress.update(1)
                continue
            if overall:
                expected = (overall.plan["costs"][rate_name(rate)]["view_seconds"] * config["runtime_safety_factor"]
                            if overall.plan else 0)
                overall.check_budget(expected)
            started = perf_counter()
            if data is None:
                data = load_data(config)
                if not data.n_obs:
                    raise ValueError("Empty cohort")
                write_table(run / "cohort.tsv", data.obs.reset_index())
            draw_seed = seed(config, repeat, "molecules", rate)
            counts = thin(data.X, data.obs_names, rate, draw_seed)
            if mask is not None:
                counts = counts.multiply(~mask(data.var.feature_name.to_numpy())[None, :]).tocsr()
                counts.eliminate_zeros()
            view = ad.AnnData(X=counts, obs=data.obs.copy(), var=data.var.copy())
            output.mkdir(parents=True, exist_ok=True)
            tmp = output / "counts.part.h5ad"
            view.write_h5ad(tmp, compression="gzip")
            tmp.replace(output / "counts.h5ad")
            original = np.asarray(data.X.sum(axis=1)).ravel()
            retained = np.asarray(counts.sum(axis=1)).ravel()
            write_table(output / "measurement_audit.tsv", pd.DataFrame({
                "cell_id": data.obs_names, "source_umis": original, "input_umis": retained,
                "empty_input": retained == 0,
            }))
            if model is None:
                model, info = model_factory(config)
                # Keep inference's inner bar from obscuring the overall ETA.
                if overall:
                    model.verbose = False
                if (run / "model.json").exists() and read_json(run / "model.json")["snapshot_revision"] != info["snapshot_revision"]:
                    raise ValueError("Checkpoint snapshot changed during resume")
                write_json(run / "model.json", info)
            result = model.process_anndata(view, batch_size=config["batch_size"])
            features = result.get_embeddings(n_features=config["embedding_dim"]).astype(np.float32)
            if features.shape != (view.n_obs, config["embedding_dim"]) or not np.isfinite(features).all():
                raise ValueError("Invalid embedding shape or values")
            tmp = output / "scunveil.npy.part"
            with tmp.open("wb") as stream:
                np.save(stream, features, allow_pickle=False)
            tmp.replace(output / "scunveil.npy")
            write_json(output / "complete.json", {
                "retention_rate": rate, "repeat": repeat, "dilution_seed": draw_seed,
                "cell_ids_sha256": digest(view.obs_names.tolist()), "n_cells": view.n_obs,
                "n_genes": view.n_vars, "source_umis": int(original.sum()),
                "input_umis": int(retained.sum()), "empty_cells": int((retained == 0).sum()),
                "gene_mapping": result.gene_mapping_summary,
                "elapsed_seconds": perf_counter() - started,
                "files": {f: sha256_file(output / f) for f in
                          ("counts.h5ad", "scunveil.npy", "measurement_audit.tsv")},
            })
            if overall and overall.plan:
                overall.advance(overall.plan["costs"][rate_name(rate)]["view_seconds"] * config["runtime_safety_factor"],
                                f"embedding keep={rate:g}, draw={repeat + 1}")
            progress.update(1)
    return run


def fold_directory(run, rate, repeat, task, test_group):
    return run / "evaluation" / rate_name(rate) / f"repeat_{repeat:03d}" / task / f"test_{test_group}"


class FitProgress:
    """Forward fit counts and calibrated per-method costs to the overall bar."""
    def __init__(self, bar, overall, rate, method_names):
        self.bar, self.overall, self.rate = bar, overall, rate
        self.method_names = method_names
        self.method, self.final, self.label = None, False, ""

    def set_postfix_str(self, label):
        self.label = label
        self.method = next(m for m in self.method_names if m in label.split())
        self.final = " final " in label
        if self.overall:
            expected = 0
            if self.overall.plan:
                costs = self.overall.plan["costs"][rate_name(self.rate)]["methods"][self.method]
                expected = costs["final_seconds" if self.final else "tuning_fit_seconds"]
                expected *= self.overall.config["runtime_safety_factor"]
            self.overall.check_budget(expected)
            self.overall.bar.set_postfix_str(label)
        self.bar.set_postfix_str(label)

    def update(self, n):
        self.bar.update(n)
        if self.overall and self.overall.plan:
            costs = self.overall.plan["costs"][rate_name(self.rate)]["methods"][self.method]
            seconds = costs["final_seconds" if self.final else "tuning_fit_seconds"]
            self.overall.advance(n * seconds * self.overall.config["runtime_safety_factor"], self.label)


def evaluate(config, outer_folds, feature_builder, fit_fold, method_names,
             tuning_fits_per_method, contexts=None, overall=None):
    run = run_directory(config)
    contexts = evaluation_jobs(config, planned_repeats(config, run)) if contexts is None else contexts
    jobs = [(rate, repeat, task, test) for rate, repeat in contexts for task, test in outer_folds]
    pending_jobs = [job for job in jobs if not (fold_directory(run, *job) / "complete.json").exists()]
    pending_fits = sum(len(method_names) * (tuning_fits_per_method + 1 if repeat == 0 else 1)
                       for rate, repeat, task, test in pending_jobs)
    started = perf_counter()
    with tqdm(total=len(jobs), desc="Evaluation folds", unit="fold", position=1 if overall else 0,
              smoothing=0, dynamic_ncols=True) as folds, \
         tqdm(total=pending_fits, desc="Pending probe fits", unit="fit", position=2 if overall else 1,
              smoothing=0, dynamic_ncols=True) as fits:
        for rate, repeat in contexts:
            pending_context = any(not (fold_directory(run, rate, repeat, task, test) / "complete.json").exists()
                                  for task, test in outer_folds)
            directory = view_directory(run, rate, repeat)
            receipt = verify(directory)
            if pending_context:
                view = ad.read_h5ad(directory / "counts.h5ad")
                if digest(view.obs_names.tolist()) != receipt["cell_ids_sha256"]:
                    raise ValueError("Count/embedding observation order mismatch")
                features = feature_builder(view)
                features["scunveil"] = np.load(directory / "scunveil.npy", mmap_mode="r")
                if features["scunveil"].shape != (view.n_obs, config["embedding_dim"]):
                    raise ValueError("Cached embedding shape mismatch")
            for task, test in outer_folds:
                fold_config = {**config, "seed": seed(config, 0, "validation"),
                               "retention_rate": rate, "repeat": repeat, "bootstrap_repeats": 0}
                output = fold_directory(run, rate, repeat, task, test)
                folds.set_postfix_str(f"keep={rate:g} draw={repeat + 1} {task}/test={test}")
                if (output / "complete.json").exists():
                    verify(output)
                else:
                    if overall:
                        overall.check_budget()
                    if repeat > 0:
                        canonical = fold_directory(run, rate, 0, task, test)
                        tuned = verify(canonical)
                        fold_config["fixed_regularization"] = tuned.get("selected_C", tuned.get("selected_alpha"))
                        fold_config["tuning_source"] = str(canonical.relative_to(run))
                    progress = FitProgress(fits, overall, rate, method_names)
                    fit_fold(view, features, task, test, fold_config, output, progress)
                    record = read_json(output / "complete.json")
                    record["files"].update({p.name: sha256_file(p) for p in output.glob("*.joblib")})
                    write_json(output / "complete.json", {**record, "retention_rate": rate, "repeat": repeat,
                               "dilution_seed": receipt["dilution_seed"], "split_seed": fold_config["seed"],
                               "view_cell_ids_sha256": receipt["cell_ids_sha256"]})
                    if overall and overall.plan:
                        overhead = overall.plan["costs"][rate_name(rate)]["fold_overhead_seconds"]
                        overall.advance(overhead * config["runtime_safety_factor"], f"completed {task}/test={test}")
                folds.update(1)
    write_json(run / "evaluation_timing.json", {
        "seconds": perf_counter() - started, "folds": len(jobs), "new_folds": len(pending_jobs),
        "new_fits": pending_fits, "tuning_policy": "repeat 0 only; fixed training-selected regularization thereafter",
    })
    return run


class OverallProgress:
    """Calibrated work weights; tqdm's cumulative-average ETA (smoothing=0)."""
    def __init__(self, config):
        self.config, self.plan = config, None
        self.started = perf_counter()
        self.bar = tqdm(total=None, desc="TOTAL (calibrating)", position=0,
                        smoothing=0, mininterval=1, dynamic_ncols=False,
                        bar_format="{desc} | LEFT {remaining} | elapsed {elapsed} | {percentage:3.0f}% {bar:10} | {postfix}")

    @property
    def elapsed(self):
        return perf_counter() - self.started

    def check_budget(self, expected_seconds=0):
        limit = self.config["runtime_budget_minutes"] * 60 - self.config["runtime_reserve_seconds"]
        if self.elapsed + expected_seconds >= limit:
            raise RuntimeError("Runtime budget reached at a safe work boundary. Completed artifacts are preserved; "
                               "the benchmark is incomplete. Resume the same command to continue.")

    def calibrate(self, plan, pending_seconds):
        self.plan = plan
        self.bar.total = self.elapsed + pending_seconds + self.config["runtime_reserve_seconds"]
        self.bar.n = self.elapsed
        self.bar.set_description_str("TOTAL")
        self.bar.set_postfix_str(f"{plan['effective_repeats']} dilution draws; calibrated ETA")
        self.bar.refresh()

    def advance(self, seconds, label):
        self.bar.set_postfix_str(label, refresh=False)
        self.bar.update(max(0, seconds))

    def finish(self):
        if self.bar.total is not None:
            self.bar.update(max(0, self.bar.total - self.bar.n))
        self.bar.set_postfix_str("complete")

    def close(self):
        self.bar.close()


def remaining_work(config, run, repeats, costs, outer_folds, method_names, tuning_fits_per_method):
    """Seconds of outstanding work, weighted by the pilot's measured costs."""
    seconds = 0.0
    for rate, repeat in view_jobs(config, repeats):
        if not (view_directory(run, rate, repeat) / "complete.json").exists():
            seconds += costs[rate_name(rate)]["view_seconds"]
    for rate, repeat in evaluation_jobs(config, repeats):
        rate_costs = costs[rate_name(rate)]
        for task, test in outer_folds:
            if (fold_directory(run, rate, repeat, task, test) / "complete.json").exists():
                continue
            seconds += rate_costs["fold_overhead_seconds"]
            for method in method_names:
                method_costs = rate_costs["methods"][method]
                seconds += method_costs["final_seconds"]
                if repeat == 0:
                    seconds += tuning_fits_per_method * method_costs["tuning_fit_seconds"]
    return seconds * config["runtime_safety_factor"]


def choose_repeats(config, cost, elapsed):
    """Choose sample count from timings only; never consult prediction scores."""
    minimum = min(config["minimum_repeats"], config["repeats"])
    for repeats in range(config["repeats"], minimum - 1, -1):
        estimate = elapsed + cost(repeats) + config["runtime_reserve_seconds"]
        if estimate <= config["runtime_budget_minutes"] * 60:
            return repeats, estimate
    raise RuntimeError(
        f"Timing preflight cannot fit even {minimum} paired dilution draws within "
        f"{config['runtime_budget_minutes']:g} minutes. Pilot artifacts are cached; no full workload was launched. "
        "Use fewer repeats/minimum_repeats or a larger runtime_budget_minutes."
    )


def preflight(config, adapter, model_factory, overall=None):
    """Run one reusable representative fold per rate, then budget the workload."""
    started = perf_counter()
    run = run_directory(config)
    pilot_contexts = [(rate, 0) for rate in config["retention_rates"]]
    mask = adapter.immunoglobulin_mask if config.get("exclude_immunoglobulin") else None
    embed(config, adapter.load_data, model_factory, mask, jobs=pilot_contexts, overall=overall)
    source = view_directory(run, config["retention_rates"][0], 0) / "counts.h5ad"
    data = ad.read_h5ad(source, backed="r")
    try:
        pilot_fold = adapter.pilot_fold(data.obs)
    finally:
        data.file.close()
    method_names = adapter.method_names(config)
    tuning_count = adapter.tuning_fits_per_method(config)
    evaluate(config, [pilot_fold], adapter.features, adapter.fit_fold, method_names, tuning_count,
             contexts=pilot_contexts, overall=overall)
    costs = {}
    for rate, repeat in pilot_contexts:
        view_info = verify(view_directory(run, rate, repeat))
        path = fold_directory(run, rate, repeat, *pilot_fold)
        info = verify(path)
        table = pd.read_csv(path / "tuning.tsv", sep="\t")
        time_column = "elapsed_seconds" if "elapsed_seconds" in table else "seconds"
        method_costs = {}
        for method in method_names:
            timings = table[table.method == method][time_column]
            if len(timings) != tuning_count:
                raise ValueError("Pilot tuning table is incomplete")
            method_costs[method] = {"tuning_fit_seconds": float(timings.mean()),
                                    "final_seconds": info["final_fit_seconds"][method]}
        fitting = float(table[time_column].sum()) + sum(info["final_fit_seconds"].values())
        costs[rate_name(rate)] = {"view_seconds": view_info["elapsed_seconds"], "methods": method_costs,
                                 "fold_overhead_seconds": max(0, info["elapsed_seconds"] - fitting)}
    elapsed = overall.elapsed if overall else perf_counter() - started
    cost = lambda n: remaining_work(config, run, n, costs, adapter.OUTER_FOLDS, method_names, tuning_count)
    plan_path = run / "runtime_plan.json"
    if plan_path.exists():
        # A resume never changes the already chosen scientific sample count.
        plan = read_json(plan_path)
        repeats = plan["effective_repeats"]
        estimate = elapsed + cost(repeats) + config["runtime_reserve_seconds"]
        if estimate > config["runtime_budget_minutes"] * 60:
            raise RuntimeError("Cached plan's remaining work exceeds this invocation's runtime budget; "
                               "increase runtime_budget_minutes to resume this sample count.")
    else:
        repeats, estimate = choose_repeats(config, cost, elapsed)
        plan = {"requested_repeats": config["repeats"], "effective_repeats": repeats,
                "pilot_fold": list(pilot_fold), "pilot_selection": "largest outer-training cell count; no prediction scores",
                "calibration_seconds": elapsed, "estimated_total_seconds": estimate,
                "budget_seconds": config["runtime_budget_minutes"] * 60,
                "safety_factor": config["runtime_safety_factor"], "costs": costs,
                "tuning_policy": "repeat 0 per rate and biological outer fold; fixed thereafter"}
        write_json(plan_path, plan)
    tqdm.write(f"Timing plan: {repeats}/{config['repeats']} dilution draws; "
               f"estimated total {estimate / 60:.1f} min including calibration, margin and report reserve; "
               f"budget {config['runtime_budget_minutes']:g} min. {plan_path}")
    if overall:
        overall.calibrate(plan, cost(repeats))
    return plan


def run_all(config, adapter, model_factory):
    """One overall ETA spans preparation, calibration, embedding, fits and report."""
    overall = OverallProgress(config)
    model_and_info = None

    def cached_factory(settings):
        nonlocal model_and_info
        if model_and_info is None:
            model_and_info = model_factory(settings)
        return model_and_info

    try:
        adapter.prepare(config)
        preflight(config, adapter, cached_factory, overall)
        run = run_directory(config)
        mask = adapter.immunoglobulin_mask if config.get("exclude_immunoglobulin") else None
        embed(config, adapter.load_data, cached_factory, mask, overall=overall)
        evaluate(config, adapter.OUTER_FOLDS, adapter.features, adapter.fit_fold,
                 adapter.method_names(config), adapter.tuning_fits_per_method(config), overall=overall)
        overall.bar.set_postfix_str("writing report")
        output = adapter.report(config)
        overall.finish()
        write_json(run / "runtime_complete.json", {"elapsed_seconds": overall.elapsed,
                   "budget_seconds": config["runtime_budget_minutes"] * 60,
                   "effective_repeats": planned_repeats(config, run), "complete": True})
        return output
    finally:
        overall.close()


def summarize(frame, keys, value="value"):
    return frame.groupby(keys, as_index=False, dropna=False).agg(
        mean=(value, "mean"), sd=(value, "std"), n_repeats=("repeat", "nunique"),
        repeat_q025=(value, lambda x: x.quantile(.025)), repeat_q975=(value, lambda x: x.quantile(.975)),
    )


def report(config, title, outer_folds, read_metrics, primary_metric, notes):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    run = run_directory(config)
    output = run / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows, tuning, audit, fold_audit = [], [], [], []
    repeats = planned_repeats(config, run)
    for rate, repeat in view_jobs(config, repeats):
        info = verify(view_directory(run, rate, repeat))
        audit.append({"retention_rate": rate, "repeat": repeat,
                      **{k: info[k] for k in ("n_cells", "source_umis", "input_umis", "empty_cells")},
                      **info["gene_mapping"]})
    for rate, repeat in evaluation_jobs(config, repeats):
        for task, test in outer_folds:
            directory = fold_directory(run, rate, repeat, task, test)
            info = verify(directory)
            context = {"retention_rate": rate, "repeat": repeat, "task": task, "test_group": str(test)}
            fold_audit.append({**context, **{k: info[k] for k in (
                "train_cells", "test_cells", "train_groups", "test_groups", "test_prevalence", "n_train", "n_test",
                "dilution_seed", "split_seed", "view_cell_ids_sha256", "tuning_reused", "tuning_source",
            ) if k in info}})
            rows.append(read_metrics(directory).assign(**context))
            selected = info.get("selected_C", info.get("selected_alpha"))
            grid = config["c_values"] if "selected_C" in info else config["alphas"]
            for method, parameter in selected.items():
                tuning.append({**context, "method": method, "selected_regularization": parameter,
                               "at_grid_boundary": parameter in (min(grid), max(grid)),
                               "tuning_reused": info["tuning_reused"]})
    scores = pd.concat(rows, ignore_index=True)
    keys = ["retention_rate", "repeat", "task", "test_group", "subset", "metric"]
    wide = scores.pivot(index=keys, columns="method", values="value")
    differences = pd.concat([
        (wide["scunveil"] - wide[m]).rename("value").reset_index().assign(baseline=m)
        for m in wide.columns if m != "scunveil"
    ], ignore_index=True)
    group_keys = ["retention_rate", "task", "test_group", "subset", "metric"]
    group_summary = summarize(scores, group_keys + ["method"])
    group_differences = summarize(differences, group_keys + ["baseline"])
    # Average biological test groups WITHIN each repeat before calculating SD.
    macro_keys = ["retention_rate", "repeat", "task", "subset", "metric"]
    macro = scores.groupby(macro_keys + ["method"], as_index=False).value.mean()
    macro_diff = differences.groupby(macro_keys + ["baseline"], as_index=False).value.mean()
    macro_summary = summarize(macro, [k for k in macro_keys if k != "repeat"] + ["method"])
    macro_diff_summary = summarize(macro_diff, [k for k in macro_keys if k != "repeat"] + ["baseline"])
    for name, frame in (("metrics", scores), ("paired_differences", differences),
                        ("group_summary", group_summary), ("group_paired_summary", group_differences),
                        ("repeat_macro_metrics", macro), ("repeat_macro_differences", macro_diff),
                        ("macro_summary", macro_summary), ("macro_paired_summary", macro_diff_summary),
                        ("selected_regularization", pd.DataFrame(tuning)), ("measurement_audit", pd.DataFrame(audit)),
                        ("fold_audit", pd.DataFrame(fold_audit))):
        write_table(output / f"{name}.tsv", frame)
    labels = {"scunveil": "scUNVEIL", "raw_log1p": "Raw log1p RNA", "depth_qc": "RNA QC", "ighg_only": "IGHG-only"}
    colors = {"scunveil": "#126782", "raw_log1p": "#d16b35", "depth_qc": "#777777", "ighg_only": "#639d36"}
    metric_label = {"average_precision": "Lineage average precision", "spearman": "Spearman correlation"}[primary_metric]
    primary = macro_summary[(macro_summary.subset == "primary") & (macro_summary.metric == primary_metric)]
    delta = macro_diff_summary[(macro_diff_summary.subset == "primary") & (macro_diff_summary.metric == primary_metric)]
    synthetic = read_json(run / "model.json")["model_version"] == "synthetic"
    for name, frame, category in (("retention_curves", primary, "method"), ("paired_gains", delta, "baseline")):
        tasks = list(primary.task.unique())
        fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), squeeze=False)
        for axis, task in zip(axes.flat, tasks):
            table = frame[frame.task == task]
            for method in table[category].unique():
                part = table[table[category] == method].sort_values("retention_rate")
                axis.errorbar(part.retention_rate, part["mean"], yerr=part.sd.fillna(0),
                              marker="o", capsize=3, label=labels[method], color=colors[method])
            axis.set_xscale("log")
            ticks = sorted(config["retention_rates"])
            axis.set_xticks(ticks, [f"{r:.0%}" for r in ticks])
            axis.set(title=task, xlabel="UMIs retained in train / validation / test",
                     ylabel=metric_label if category == "method" else f"scUNVEIL minus control\n({metric_label})")
            if category == "baseline":
                axis.axhline(0, color="grey", linestyle=":")
            axis.legend(fontsize=8)
        fig.suptitle(("SYNTHETIC CPU FIXTURE\n" if synthetic else "") + title + "\nError bars = repeat SD")
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    # Group-level figures keep heterogeneous biological outcomes visible.
    group_primary = group_summary[(group_summary.subset == "primary") & (group_summary.metric == primary_metric)]
    panels = list(group_primary.groupby(["task", "test_group"], sort=True))
    fig, axes = plt.subplots((len(panels) + 3) // 4, min(4, len(panels)),
                             figsize=(5 * min(4, len(panels)), 4 * ((len(panels) + 3) // 4)), squeeze=False)
    for axis, ((task, test), table) in zip(axes.flat, panels):
        for method, part in table.groupby("method"):
            part = part.sort_values("retention_rate")
            axis.errorbar(part.retention_rate, part["mean"], yerr=part.sd.fillna(0), marker="o",
                          capsize=3, label=labels[method], color=colors[method])
        axis.set_xscale("log")
        ticks = sorted(config["retention_rates"])
        axis.set_xticks(ticks, [f"{r:.0%}" for r in ticks])
        axis.set(title=f"{task} / test {test}", xlabel="UMIs retained", ylabel=metric_label)
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    fig.suptitle(("SYNTHETIC CPU FIXTURE\n" if synthetic else "") + "Biological test groups; error bars = repeat SD")
    fig.tight_layout()
    fig.savefig(output / "group_retention_curves.svg", bbox_inches="tight")
    plt.close(fig)
    lines = [f"# {title}", "", f"Run `{run.name}`.", ""]
    if synthetic:
        lines += ["**SYNTHETIC CPU FIXTURE — software validation only.**", ""]
    lines += [
        "Matched binomial UMI retention in training, validation, and testing; all training cells remain available. "
        "Every representation uses the same count draw and split. Cohort membership and labels are fixed from the source data; "
        "empty thinned cells are retained. Raw RNA is sparse log1p(count), without library normalization or PCA.", "",
        f"Requested dilution draws: {config['repeats']}; executed: {repeats}. Full depth is evaluated once (repeat SD unavailable). "
        "Biological outer/inner assignments are fixed. Each representation tunes independently on draw 0 at each retention rate and biological outer fold, "
        "using training/validation labels only. Later draws refit the final model with those settings; no test labels enter tuning.", "",
        "Means average biological test groups equally within a repeat, then average repeats. Error bars are repeat SD; "
        "the empirical repeat quantiles in TSVs describe molecular-draw variability conditional on fixed tuning, not confidence intervals for new donors. "
        "Repeated molecular draws do not create independent biological experiments.", "",
        f"## Primary metric: {primary_metric}", "",
        "| Retention | Task | Representation | Mean | Repeat SD | Repeats |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in primary.itertuples():
        sd = f"{r.sd:.4f}" if pd.notna(r.sd) else "n/a"
        lines.append(f"| {r.retention_rate:.0%} | {r.task} | {labels[r.method]} | {r.mean:.4f} | {sd} | {r.n_repeats} |")
    lines += ["", "## Paired absolute gains", "", "Positive primary-metric differences favor scUNVEIL.", "",
              "| Retention | Task | Control | Mean difference | Repeat SD | Repeats |",
              "|---|---|---|---:|---:|---:|"]
    for r in delta.itertuples():
        sd = f"{r.sd:.4f}" if pd.notna(r.sd) else "n/a"
        lines.append(f"| {r.retention_rate:.0%} | {r.task} | {labels[r.baseline]} | {r.mean:+.4f} | {sd} | {r.n_repeats} |")
    boundary = pd.DataFrame(tuning).query("not tuning_reused").groupby("method").at_grid_boundary.agg(["sum", "count"])
    lines += ["", "## Regularization audit", "", "Boundary selections may require a broader grid; inspect tuning.tsv before final conclusions.", ""]
    for method, r in boundary.iterrows():
        lines.append(f"- {labels[method]}: {int(r['sum'])}/{int(r['count'])} selections at a grid boundary.")
    lines += ["", "## Interpretation and limits", ""] + [f"- {note}" for note in notes]
    lines += ["", "## Outputs", "", "`retention_curves.svg/png`, `paired_gains.svg/png`, and `group_retention_curves.svg` "
              "show absolute scores and paired gains. `metrics.tsv` and `paired_differences.tsv` retain every repeat and test group. "
              "`macro_summary.tsv` and `macro_paired_summary.tsv` summarize repeats after equal-weight group averaging; "
              "`group_summary.tsv` and `group_paired_summary.tsv` expose heterogeneity. "
              "`measurement_audit.tsv` records actual UMIs, empty cells, and gene mapping. "
              "`fold_audit.tsv` records train/test sizes, seeds and (for ReSisTrace) lineage prevalence. "
              "Per-fold predictions, fitted probes, tuning and splits remain under `evaluation/`.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    write_json(output / "complete.json", {"run": run.name, "config": config,
               "files": {p.name: sha256_file(p) for p in output.iterdir() if p.suffix in (".tsv", ".md", ".svg", ".png")}})
    print(f"Report: {output / 'REPORT.md'}", flush=True)
    return output
