"""Identical molecule draws, full-width frozen embeddings and shortcut controls."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
import sys

import anndata as ad
import numpy as np
from tqdm.auto import tqdm

from tmre_common import MODEL_SOURCE, read_json, sha256, write_json


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
                mapping_path = self.root / "gene_mapping_first_batch.json"
                if not mapping_path.exists():
                    write_json(mapping_path, {"description": "First inference batch only; UMI fractions vary by batch/draw",
                                              "gene_mapping": result.gene_mapping_summary,
                                              "n_cells": stop - start})
                bar.update(1)
        return np.concatenate(pieces), mappings[0]

    def full_embeddings(self, indices):
        """Cache only selected original-count cells; SQLite transactions survive interruption."""
        self.ensure_model()
        path = self.root / "full_depth_cache.sqlite"
        obs = self.data.obs.iloc[indices]
        result = np.empty((len(indices), self.dimension), dtype=np.float32)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (cell TEXT PRIMARY KEY, vector BLOB NOT NULL, sha256 TEXT NOT NULL)")
            missing = []
            for position, cell in enumerate(obs.index):
                row = db.execute("SELECT vector, sha256 FROM embeddings WHERE cell=?", (str(cell),)).fetchone()
                if row is None:
                    missing.append(position)
                else:
                    blob, checksum = row
                    if hashlib.sha256(blob).hexdigest() != checksum:
                        raise ValueError("Corrupted full-depth embedding cache")
                    vector = np.frombuffer(blob, dtype=np.float32)
                    if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
                        raise ValueError("Invalid full-depth cache width or values")
                    result[position] = vector
            if missing:
                local = np.asarray(missing, dtype=int)
                values, _ = self.embeddings(self.data.X[indices[local]], obs.iloc[local],
                                             f"Cache {len(local)} new full-depth cells")
                entries = []
                for position, value in zip(local, values):
                    blob = np.asarray(value, dtype=np.float32).tobytes()
                    entries.append((str(obs.index[position]), blob, hashlib.sha256(blob).hexdigest()))
                    result[position] = value
                db.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", entries)
                db.commit()
        return result

    def features(self, indices, rate, draw_seed):
        indices = np.asarray(indices, dtype=int)
        obs = self.data.obs.iloc[indices]
        counts = thin(self.data.X[indices], obs.index, rate, draw_seed)
        embeddings = self.full_embeddings(indices) if rate == 1 else self.embeddings(
            counts, obs, f"Embedding {len(obs)} cells, keep={rate:g}")[0]
        total = np.asarray(counts.sum(axis=1)).ravel()
        depth = np.column_stack([np.log1p(total), np.log1p(np.diff(counts.indptr))]).astype(np.float32)
        mitochondrial = self.data.var.gene_symbol.str.upper().str.startswith("MT-").to_numpy()
        mt_counts = np.asarray(counts[:, mitochondrial].sum(axis=1)).ravel()
        fraction = np.divide(mt_counts, total, out=np.zeros(len(total), float), where=total > 0)
        values = {"scunveil": embeddings, "raw_log1p": log1p(counts), "depth": depth,
                  "depth_mito": np.column_stack([depth, fraction]).astype(np.float32)}
        if any(x.shape[0] != len(obs) for x in values.values()):
            raise ValueError("Unpaired feature rows")
        return counts, values
