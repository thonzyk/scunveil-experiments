"""Matched RNA views and cached frozen scUNVEIL embeddings (no fitted PCA)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from common import (
    MODEL_SOURCE, digest, local_model_hash, prepared_manifest, provenance,
    read_json, run_directory, sha256_file, write_json, write_table,
)


def log1p_counts(counts):
    """Sparse log1p of raw UMIs; no library-size normalization or PCA."""
    x = counts.astype(np.float32).tocsr(copy=True)
    np.log1p(x.data, out=x.data)
    return x


def exact_depth_view(counts, cell_ids, depth, seed):
    """Sample molecules without replacement; each retained row has exact depth.

    Seeds depend on the cell ID, so filtering/reordering unrelated cells does
    not change a given cell's sampled view. No original counts enter the probe.
    """
    counts = counts.tocsr()
    if depth == "full":
        return counts.copy()
    result = counts.copy()
    for i, cell_id in enumerate(cell_ids):
        start, end = counts.indptr[i:i + 2]
        values = counts.data[start:end].astype(np.int64)
        if values.sum() < depth:
            raise ValueError(f"Cell {cell_id} has fewer than {depth} UMIs")
        token = hashlib.sha256(f"{seed}:{depth}:{cell_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
        result.data[start:end] = rng.multivariate_hypergeometric(values, depth)
    result.eliminate_zeros()
    return result


def binomial_umi_view(counts, cell_ids, probability, seed):
    """Independent Binomial(count, p) thinning, stable under cell reordering.

    Unlike exact_depth_view, totals fluctuate around p times each cell's depth.
    Empty sampled cells are retained so the observation set stays identical.
    """
    if not 0 < probability <= 1:
        raise ValueError("UMI keep probability must be in (0, 1]")
    counts = counts.tocsr()
    result = counts.copy()
    if probability == 1:
        return result
    for i, cell_id in enumerate(cell_ids):
        start, end = counts.indptr[i:i + 2]
        token = hashlib.sha256(f"binomial:{seed}:{probability}:{cell_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
        result.data[start:end] = rng.binomial(counts.data[start:end].astype(np.int64), probability)
    result.eliminate_zeros()
    return result


def depth_qc(counts):
    """A deliberately small depth/detection shortcut control at the SAME depth."""
    x = counts.tocsr()
    return np.column_stack((
        np.log1p(np.asarray(x.sum(axis=1)).ravel()), np.log1p(np.diff(x.indptr)),
    )).astype(np.float32)


def make_model(config):
    # This experiment must test the local inference checkout, not stale PyPI code.
    sys.path.insert(0, str(MODEL_SOURCE))
    import tensorflow as tf
    import scunveil
    from scunveil import scUnveil

    source = Path(scunveil.__file__).resolve()
    if not source.is_relative_to(MODEL_SOURCE.resolve()):
        raise RuntimeError(f"Unexpected scunveil import path: {source}")
    gpus = tf.config.list_physical_devices("GPU")
    if config["device"] == "gpu" and not gpus:
        raise RuntimeError("GPU requested but TensorFlow sees no GPU; use device=cpu or auto")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = scUnveil(model_version=config["model_version"], verbose=True)
    if config["embedding_dim"] > model.config.emb_dim:
        raise ValueError("embedding_dim exceeds the pretrained embedding dimension")
    return model, {
        "import_path": str(source), "local_source_sha256": local_model_hash(),
        "model_version": model.model_version, "checkpoint_path": str(model.checkpoint_path),
        "snapshot_revision": model.checkpoint_path.name,
        "visible_gpus": [gpu.name for gpu in gpus], "provenance": provenance(),
    }


def save_array(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def verify_view(path):
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Corrupted cached view artifact: {path / name}")
    return manifest


def embed(config, model_factory=make_model):
    prepared = prepared_manifest(config)
    run = run_directory(config)
    print(f"Run directory: {run}", flush=True)
    numeric = [d for d in config["depths"] if d != "full"]
    threshold = max(numeric, default=0)
    model = None
    data = None
    jobs = [(depth, "views", 1.0) for depth in config["depths"]]
    if config["train_umi_keep_probability"] < 1:
        jobs += [(depth, "training_views", config["train_umi_keep_probability"]) for depth in config["depths"]]
    for depth, role, keep_probability in jobs:
        directory = run / role / str(depth)
        if (directory / "manifest.json").exists():
            verify_view(directory)
            print(f"Verified cached {role}, depth {depth}", flush=True)
            continue
        if data is None:
            data = ad.read_h5ad(Path(config["output_root"]) / "prepared" / "counts.h5ad")
            eligible = data.obs["panel_umis"].to_numpy() >= threshold
            write_table(run / "depth_eligibility.tsv", data.obs.assign(
                eligible=eligible, required_panel_umis=threshold,
            ).reset_index()[["cell_id", "condition", "replicate", "outcome", "panel_umis", "eligible", "required_panel_umis"]])
            # SAME eligible cells for all requested depths, including full.
            data = data[eligible].copy()
            if not data.n_obs:
                raise ValueError("No cells have enough UMIs for the requested depth ladder")
        directory.mkdir(parents=True, exist_ok=True)
        view = ad.AnnData(
            X=binomial_umi_view(
                exact_depth_view(data.X, data.obs_names, depth, config["seed"]),
                data.obs_names, keep_probability, config["seed"],
            ),
            obs=data.obs.copy(), var=data.var.copy(),
        )
        temporary = directory / "counts.part.h5ad"
        view.write_h5ad(temporary, compression="gzip")
        temporary.replace(directory / "counts.h5ad")
        write_table(directory / "cells.tsv", view.obs.reset_index())
        if model is None:
            model, model_info = model_factory(config)
            if (run / "model.json").exists():
                previous = read_json(run / "model.json")
                if previous["snapshot_revision"] != model_info["snapshot_revision"]:
                    raise ValueError("Model snapshot changed during resume; use a new run_name")
            write_json(run / "model.json", model_info)
        print(f"scUNVEIL {role}, depth {depth}, UMI keep={keep_probability:g}: {view.shape}", flush=True)
        result = model.process_anndata(view, batch_size=config["batch_size"])
        embeddings = result.get_embeddings(n_features=config["embedding_dim"]).astype(np.float32)
        if embeddings.shape != (view.n_obs, config["embedding_dim"]) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid scUNVEIL embedding shape or values")
        save_array(directory / "scunveil.npy", embeddings)
        save_array(directory / "depth_qc.npy", depth_qc(view.X))
        files = {name: sha256_file(directory / name) for name in (
            "counts.h5ad", "cells.tsv", "scunveil.npy", "depth_qc.npy",
        )}
        write_json(directory / "manifest.json", {
            "depth": depth, "seed": config["seed"], "n_cells": view.n_obs,
            "role": role, "umi_keep_probability": keep_probability,
            "total_umis": int(view.X.sum()),
            "n_genes": view.n_vars, "embedding_dim": config["embedding_dim"],
            "cell_ids_sha256": digest(view.obs_names.tolist()),
            "prepared_counts_sha256": prepared["counts_sha256"], "files": files,
            "gene_mapping": result.gene_mapping_summary,
            "depth_rule": "exact depth view followed by independent binomial thinning at umi_keep_probability",
        })
    return run
