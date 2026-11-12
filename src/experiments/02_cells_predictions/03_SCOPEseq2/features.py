"""Identical molecule draws, full-width frozen embeddings and shortcut controls."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import anndata as ad
import numpy as np
from tqdm.auto import tqdm

from scope_common import (MODEL_SOURCE, digest, read_json, seal, sha256, verify,
                        write_array, write_json)


def progress(total, desc):
    return tqdm(total=total, desc=desc, smoothing=0, mininterval=1, dynamic_ncols=False,
                bar_format="{desc} | LEFT {remaining} | elapsed {elapsed} | {n_fmt}/{total_fmt} {bar:10} | {postfix}")


def thin(counts, cell_ids, probability, draw_seed):
    if not 0 < probability <= 1 or len(cell_ids) != counts.shape[0]:
        raise ValueError("Invalid thinning probability/rows")
    result = counts.copy().tocsr()
    if probability != 1:
        for row, cell_id in enumerate(cell_ids):
            start, end = result.indptr[row:row + 2]
            token = hashlib.sha256(f"{draw_seed}:{probability}:{cell_id}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
            result.data[start:end] = rng.binomial(result.data[start:end].astype(np.int64), probability)
        result.eliminate_zeros()
    return result


def make_model(settings):
    sys.path.insert(0, str(MODEL_SOURCE))
    import tensorflow as tf
    import scunveil
    from scunveil import scUnveil
    if not Path(scunveil.__file__).resolve().is_relative_to(MODEL_SOURCE.resolve()):
        raise RuntimeError("Expected local inference checkout")
    devices = tf.config.list_physical_devices("GPU")
    if settings["device"] == "gpu" and not devices:
        raise RuntimeError("GPU requested but TensorFlow sees no GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    model = scUnveil(model_version=settings["model_version"], verbose=False)
    return model, {"model_version": model.model_version,
                   "checkpoint_path": str(model.checkpoint_path),
                   "snapshot_revision": model.checkpoint_path.name,
                   "checkpoint_files": {str(p.relative_to(model.checkpoint_path)): sha256(p)
                                        for p in model.checkpoint_path.rglob("*") if p.is_file()},
                   "import_path": str(scunveil.__file__)}


def log1p(counts):
    values = counts.astype(np.float32).tocsr(copy=True)
    np.log1p(values.data, out=values.data)
    return values


class Encoder:
    def __init__(self, data, settings, root, factory=make_model):
        self.data, self.settings, self.root, self.factory = data, settings, Path(root), factory
        self.model, self.dimension, self.full = None, None, None

    def ensure_model(self):
        if self.model is not None:
            return
        self.model, info = self.factory(self.settings)
        self.dimension = int(self.model.config.emb_dim)
        if self.settings["model_version"] == "2026-10-02_11-54-21_w64" and self.dimension != 2048:
            raise ValueError("The pinned checkpoint must expose all 2048 features")
        info.update(embedding_dim=self.dimension, embedding_policy="all native features; no truncation")
        path = self.root / "model.json"
        if path.exists():
            previous = read_json(path)
            for key in ("model_version", "snapshot_revision", "embedding_dim", "checkpoint_files"):
                if previous.get(key) != info.get(key):
                    raise ValueError(f"Checkpoint changed on resume: {key}")
        write_json(path, info)

    def embeddings(self, counts, obs, description):
        self.ensure_model()
        pieces, mappings = [], []
        batch = self.settings["batch_size"]
        with progress((len(obs) + batch - 1) // batch, description) as bar:
            for start in range(0, len(obs), batch):
                stop = min(start + batch, len(obs))
                view = ad.AnnData(counts[start:stop], obs=obs.iloc[start:stop].copy(), var=self.data.var.copy())
                result = self.model.process_anndata(view, batch_size=batch)
                values = result.get_embeddings(n_features=self.dimension).astype(np.float32)
                if values.shape != (stop - start, self.dimension) or not np.isfinite(values).all():
                    raise ValueError("Invalid full-width embeddings")
                pieces.append(values)
                mappings.append(result.gene_mapping_summary)
                bar.update(1)
        return np.concatenate(pieces), mappings[0]

    def full_cache(self):
        self.ensure_model()
        cache = self.root / "full_depth_cache"
        if (cache / "complete.json").exists():
            info = verify(cache)
            if info["cells_sha256"] != digest(self.data.obs_names.tolist()) or info["embedding_dim"] != self.dimension:
                raise ValueError("Full-depth cache row/width mismatch")
        else:
            cache.mkdir(parents=True, exist_ok=True)
            values, mapping = self.embeddings(self.data.X, self.data.obs, "Cache full-depth cells")
            write_array(cache / "scunveil.npy", values)
            seal(cache, {"cells_sha256": digest(self.data.obs_names.tolist()),
                         "embedding_dim": self.dimension, "gene_mapping": mapping})
        self.full = np.load(cache / "scunveil.npy", mmap_mode="r")
        if self.full.shape != (self.data.n_obs, self.dimension):
            raise ValueError("Invalid full-depth cache shape")

    def features(self, indices, rate, draw_seed):
        obs = self.data.obs.iloc[indices]
        counts = thin(self.data.X[indices], obs.index, rate, draw_seed)
        embeddings = self.full[indices] if rate == 1 else self.embeddings(
            counts, obs, f"Embedding {len(obs)} cells, keep={rate:g}")[0]
        total = np.asarray(counts.sum(axis=1)).ravel()
        values = {"scunveil": embeddings, "raw_log1p": log1p(counts),
            "depth": np.column_stack([np.log1p(total), np.log1p(np.diff(counts.indptr))]).astype(np.float32)}
        if any(x.shape[0] != len(obs) for x in values.values()):
            raise ValueError("Unpaired feature rows")
        return counts, values


