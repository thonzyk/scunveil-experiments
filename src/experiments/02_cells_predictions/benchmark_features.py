"""Versioned, checksummed feature caches; no labels, fitted probes or calibration.

Deleting this module's managed cache never deletes source data or result collections.
Callers hold the dataset's benchmark.lock throughout reads, writes and cache rebuilds.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import shutil
import sqlite3
import subprocess
import uuid

import numpy as np

from full_data_pca import FullDataPCA, PCA_SETTINGS, log_counts
from pca_comparison import digest, read_json, seal, sha256, verify, write_json
from checkpoint_validation import computation_code_hash, selected_checkpoint_files, validate_selected_checkpoint

HERE = Path(__file__).resolve().parent


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        pending = self.root / "pending"
        if pending.exists():
            shutil.rmtree(pending)
        self.verified = set()
        self.hits, self.misses = 0, 0

    def path(self, identity):
        return self.root / "objects" / identity["kind"] / digest(identity)

    def find(self, identity):
        path = self.path(identity)
        if not path.exists():
            self.misses += 1
            return None
        if path not in self.verified:
            verify(path)
            if read_json(path / "identity.json") != identity:
                raise ValueError(f"Feature cache identity mismatch: {path}")
            self.verified.add(path)
        self.hits += 1
        return path

    def publish(self, identity, producer):
        destination = self.path(identity)
        if destination.exists():
            return self.find(identity)
        temporary = self.root / "pending" / uuid.uuid4().hex
        temporary.mkdir(parents=True)
        producer(temporary)
        write_json(temporary / "identity.json", identity)
        seal(temporary, {"identity_sha256": digest(identity)})
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        self.verified.add(destination)
        return destination

    def array(self, identity, producer):
        path = self.find(identity)
        if path is None:
            def save(directory):
                values = np.asarray(producer(), dtype=np.float32)
                if values.ndim != 2 or not np.isfinite(values).all():
                    raise ValueError("Invalid cached feature matrix")
                np.save(directory / "values.npy", values, allow_pickle=False)
            path = self.publish(identity, save)
        return np.load(path / "values.npy", mmap_mode="r", allow_pickle=False)


def clear_cache(output):
    """Only this exact managed subtree, under a caller-held dataset lock."""
    root = Path(output) / "feature_cache" / "unscaled_v2"
    if root.is_symlink() or root.parent.is_symlink():
        raise ValueError("Refusing cache deletion through a symlink")
    if root.exists():
        shutil.rmtree(root)


class CachedEncoder:
    def __init__(self, adapter, data, settings, collection, source, counts_hash,
                 allow_legacy=True, factory=None, pca_settings=None):
        self.adapter, self.data, self.settings = adapter, data, settings
        self.root, self.source = Path(collection), Path(source)
        self.allow_legacy = allow_legacy
        self.factory = factory
        self.backend = None
        self.store = FeatureStore(Path(settings["output_root"]) / "feature_cache/unscaled_v2")
        self.model_info = read_json(self.source / "model.json")
        self.dimension = self.model_info["embedding_dim"]
        if settings["model_version"] != "synthetic" and self.dimension != 2048:
            raise ValueError("All 2048 scUNVEIL features are required")
        self.source_manifest = read_json(self.source / "collection.json")
        self.input = {"counts_sha256": counts_hash,
                      "cells_sha256": digest(data.obs_names.tolist()),
                      "genes_sha256": digest(data.var_names.tolist())}
        self.environment = {p: version(p) for p in ("numpy", "scipy", "anndata", "tensorflow")}
        scientific_file = "representations.py" if adapter.dataset == "ReSisTrace" else "features.py"
        if adapter.dataset == "ReSisTrace":
            import common
            model_source = common.MODEL_SOURCE
        else:
            model_source = adapter.common.MODEL_SOURCE
        inference = model_source / "scunveil"
        checkpoint_hashes = validate_selected_checkpoint(self.model_info)
        self.embedding_identity = {"input": self.input, "checkpoint": {
            k: self.model_info.get(k) for k in ("model_version", "snapshot_revision")},
            "checkpoint_files": checkpoint_hashes,
            "dimension": self.dimension, "environment": self.environment,
            "extraction_code": sha256(adapter.directory / scientific_file),
            "inference_code": {p.name: sha256(p) for p in sorted(inference.glob("*.py"))},
            "batch_size": settings["batch_size"], "device": settings["device"],
            "threads": settings["threads"], "cache_implementation": computation_code_hash(Path(__file__))}
        hardware = {"machine": platform.machine(), "processor": platform.processor(),
                    "environment": {k: os.environ.get(k) for k in
                        ("CUDA_VISIBLE_DEVICES", "TF_DETERMINISTIC_OPS", "TF_ENABLE_ONEDNN_OPTS")}}
        if settings["device"] != "cpu":
            try:
                hardware["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader"], text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                hardware["gpu"] = "unavailable"
        self.embedding_identity["hardware"] = hardware
        self.legacy_matrix = None
        self.legacy_checked = False
        self.pca = self.prepare_pca({**PCA_SETTINGS, **(pca_settings or {})})
        feature_identity = {"embedding": self.embedding_identity, "pca_basis_content": self.pca_hash}
        pin = self.root / "feature_identity.json"
        if pin.exists() and read_json(pin) != feature_identity:
            raise ValueError("Frozen feature computation changed on resume; use fresh results")
        if not pin.exists():
            write_json(pin, feature_identity)
        write_json(self.root / "model.json", self.model_info)
        write_json(self.root / "pca.json", {**self.pca.info, "basis_sha256": self.pca_hash})

    def prepare_pca(self, settings):
        identity = {"kind": "pca_basis", "input": self.input, "settings": settings,
                    "code": sha256(HERE / "full_data_pca.py"), "threads": self.settings["threads"],
                    "environment": {p: self.environment[p] for p in ("numpy", "scipy")}}
        path = self.store.find(identity)
        if path is None:
            def produce(directory):
                # Legacy results are immutable. Import only a verified basis
                # with the same input, solver code, parameters and environment.
                if self.allow_legacy:
                    for manifest_path in sorted(self.source.parent.glob("pca_comparison-*/collection.json")):
                        manifest = read_json(manifest_path)
                        basis = manifest_path.parent / "pca_basis"
                        legacy = manifest.get("identity", {})
                        if (not (basis / "complete.json").exists() or
                            manifest.get("source_collection") != str(self.source) or
                            legacy.get("code", {}).get("full_data_pca.py") != identity["code"] or
                            manifest.get("pca_settings") != settings or
                            manifest.get("threads") != self.settings["threads"] or
                            any(legacy.get("packages", {}).get(p) != self.environment[p] for p in ("numpy", "scipy"))):
                            continue
                        receipt = verify(basis)
                        if (receipt["counts_sha256"] != self.input["counts_sha256"] or
                            receipt["cell_ids_sha256"] != self.input["cells_sha256"] or
                            receipt["gene_ids_sha256"] != self.input["genes_sha256"]):
                            continue
                        shutil.copyfile(basis / "basis.npz", directory / "basis.npz")
                        write_json(directory / "pca.json", receipt["pca"])
                        write_json(directory / "origin.json", {"imported_from": str(basis)})
                        print(f"Imported verified PCA basis: {basis}", flush=True)
                        return
                pca = FullDataPCA.fit(self.data.X, settings)
                pca.save(directory / "basis.npz")
                write_json(directory / "pca.json", pca.info)
            path = self.store.publish(identity, produce)
        pca = FullDataPCA.load(path / "basis.npz", read_json(path / "pca.json"))
        content = hashlib.sha256()
        for array in (pca.mean, pca.components, pca.singular_values):
            content.update(str(array.shape).encode()); content.update(array.dtype.str.encode())
            content.update(np.ascontiguousarray(array).data)
        self.pca_hash = content.hexdigest()
        return pca

    def legacy_compatible(self):
        original = self.source_manifest
        env = original["identity"].get("environment", {}).get("packages",
            original.get("provenance", {}).get("packages", {}))
        return self.allow_legacy and bool(self.model_info.get("checkpoint_files")) and all(env.get(p) == v for p, v in self.environment.items()) and all(
            original["settings"].get(k) == self.settings[k] for k in ("batch_size", "device", "threads"))

    def inference(self, counts, obs):
        if self.backend is None:
            module = self.adapter.experiment
            factory = self.factory or module.make_model
            # The original encoder is used only for its fixed library call.
            self.backend = (module.FrozenEncoder(self.data, self.settings, self.root / "encoder", factory)
                            if self.adapter.dataset == "ReSisTrace" else
                            module.Encoder(self.data, self.settings, self.root / "encoder", factory))
        (self.root / "encoder").mkdir(parents=True, exist_ok=True)
        if self.adapter.dataset == "ReSisTrace":
            values, _ = self.backend.embeddings(counts, obs)
        else:
            values, _ = self.backend.embeddings(counts, obs, f"scUNVEIL: {len(obs)} cells")
        actual = read_json(self.root / "encoder/model.json")
        for key in ("snapshot_revision", "embedding_dim"):
            if self.model_info.get(key) != actual.get(key):
                raise ValueError(f"Inference checkpoint changed: {key}")
        if selected_checkpoint_files(self.model_info) != selected_checkpoint_files(actual):
            raise ValueError("Selected inference checkpoint files changed")
        if values.shape != (len(obs), self.dimension):
            raise ValueError("Embedding row/width mismatch")
        return values

    def full_embeddings(self, indices):
        identity = {"kind": "full_scunveil", **self.embedding_identity}
        db_path = self.store.root / ("full_scunveil_" + digest(identity) + ".sqlite")
        obs = self.data.obs.iloc[indices]
        result = np.empty((len(indices), self.dimension), np.float32)
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS features (cell TEXT PRIMARY KEY, vector BLOB, sha256 TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            expected = json.dumps(identity, sort_keys=True)
            previous = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if previous is None:
                if db.execute("SELECT COUNT(*) FROM features").fetchone()[0]:
                    raise ValueError("Full-depth cache is missing its computation identity")
                db.execute("INSERT INTO metadata VALUES ('identity',?)", (expected,))
                db.commit()
            elif previous[0] != expected:
                raise ValueError("Full-depth cache computation identity mismatch")
            missing = []
            for j, cell in enumerate(obs.index):
                row = db.execute("SELECT vector,sha256 FROM features WHERE cell=?", (str(cell),)).fetchone()
                if row is None:
                    missing.append(j)
                else:
                    blob, checksum = row
                    if hashlib.sha256(blob).hexdigest() != checksum:
                        raise ValueError("Corrupt full-depth feature cache")
                    value = np.frombuffer(blob, dtype=np.float32)
                    if value.shape != (self.dimension,) or not np.isfinite(value).all():
                        raise ValueError("Invalid cached embedding")
                    result[j] = value
            if missing:
                need_inference = list(missing)
                if self.legacy_compatible():
                    if not self.legacy_checked:
                        directory = self.source / "full_depth_cache"
                        if (directory / "complete.json").exists():
                            receipt = verify(directory)
                            row_hash = receipt.get("cells_sha256", receipt.get("cell_ids_sha256"))
                            if row_hash != self.input["cells_sha256"] or receipt["embedding_dim"] != self.dimension:
                                raise ValueError("Legacy embedding cache row/width mismatch")
                            self.legacy_matrix = np.load(directory / "scunveil.npy", mmap_mode="r")
                        self.legacy_checked = True
                    if self.legacy_matrix is not None:
                        if self.legacy_matrix.shape != (self.data.n_obs, self.dimension):
                            raise ValueError("Legacy full-depth cache shape mismatch")
                        result[missing] = self.legacy_matrix[indices[missing]]
                        need_inference = []
                    elif (self.source / "full_depth_cache.sqlite").exists():
                        with sqlite3.connect(f"file:{self.source / 'full_depth_cache.sqlite'}?mode=ro", uri=True) as old:
                            for j in missing:
                                row = old.execute("SELECT vector,sha256 FROM embeddings WHERE cell=?", (str(obs.index[j]),)).fetchone()
                                if row:
                                    blob, checksum = row
                                    if hashlib.sha256(blob).hexdigest() != checksum:
                                        raise ValueError("Corrupt legacy full-depth embedding")
                                    value = np.frombuffer(blob, dtype=np.float32)
                                    if value.shape != (self.dimension,) or not np.isfinite(value).all():
                                        raise ValueError("Invalid legacy embedding")
                                    result[j] = value
                                    need_inference.remove(j)
                if need_inference:
                    local = np.asarray(need_inference)
                    result[local] = self.inference(self.data.X[indices[local]], obs.iloc[local])
                entries = []
                for j in missing:
                    blob = np.asarray(result[j], np.float32).tobytes()
                    entries.append((str(obs.index[j]), blob, hashlib.sha256(blob).hexdigest()))
                db.executemany("INSERT INTO features VALUES (?,?,?)", entries)
                db.commit()
        return result

    def features(self, indices, rate, draw_seed):
        indices = np.asarray(indices, dtype=int)
        obs = self.data.obs.iloc[indices]
        counts = self.adapter.thin(self.data.X[indices], obs.index, rate, draw_seed)
        # Fingerprint the realized input, not just a seed or a file name.
        h = hashlib.sha256()
        for a in (counts.indptr, counts.indices, counts.data):
            h.update(a.dtype.str.encode()); h.update(a.tobytes())
        view = {"rows": digest(obs.index.tolist()), "rate": float(rate),
                "draw_seed": int(draw_seed), "realized_counts": h.hexdigest()}
        if rate == 1:
            scu = self.full_embeddings(indices)
            full_key = {"kind": "full_pca_scores", "basis": self.pca_hash, "input": self.input,
                        "implementation": computation_code_hash(Path(__file__)), "threads": self.settings["threads"]}
            pcs = self.store.array(full_key, lambda: self.pca.transform(self.data.X))[indices]
        else:
            scu_key = {"kind": "scunveil_scores", "encoder": self.embedding_identity, "view": view}
            scu = self.store.array(scu_key, lambda: self.inference(counts, obs))
            pca_key = {"kind": "pca_scores", "basis": self.pca_hash, "input": self.input, "view": view,
                       "implementation": computation_code_hash(Path(__file__)), "threads": self.settings["threads"]}
            pcs = self.store.array(pca_key, lambda: self.pca.transform(counts))
        total = np.asarray(counts.sum(axis=1)).ravel()
        depth = np.column_stack([np.log1p(total), np.log1p(np.diff(counts.indptr))]).astype(np.float32)
        values = {"scunveil": scu, "pca_full": pcs, "raw_log1p": log_counts(counts), "depth": depth}
        if self.adapter.dataset == "ReSisTrace":
            # Preserve the explicitly defined shortcut features, without a scaler.
            import representations
            values["depth_qc"] = representations.depth_qc(counts)
        if self.adapter.dataset == "TumorReactiveTCells":
            values["clone_size"] = np.log1p(obs.n_cells.to_numpy()[:, None]).astype(np.float32)
        if self.adapter.dataset == "MitochondrialPotential":
            mt = self.data.var.gene_symbol.str.upper().str.startswith("MT-").to_numpy()
            mt_count = np.asarray(counts[:, mt].sum(axis=1)).ravel()
            fraction = np.divide(mt_count, total, out=np.zeros(len(total), float), where=total > 0)
            values["depth_mito"] = np.column_stack([depth, fraction]).astype(np.float32)
        values = {m: values[m] for m in (*self.adapter.experiment.METHODS,)}
        for name, array in values.items():
            if (array.shape[0] != len(indices) or (name == "scunveil" and array.shape[1] != self.dimension)
                    or (name == "pca_full" and array.shape[1] != self.pca.dimension)):
                raise ValueError("Unpaired feature rows or truncated embedding")
        return counts, values
