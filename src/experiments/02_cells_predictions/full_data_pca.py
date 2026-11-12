"""Centered PCA of sparse log1p counts, fitted once without materializing X centered.

The deliberate full-depth/all-cell fit is transductive. Transform must always receive
the current (possibly thinned) counts, rather than cached full-depth coordinates.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from tqdm.auto import tqdm

PCA_METHOD = "pca_full"
PCA_LABEL = "PCA, full-data fit"
PCA_SETTINGS = {"requested_components": 2048, "seed": 8675309,
                "oversamples": 32, "power_iterations": 4, "gram_max_cells": 4096}


def progress(total, desc):
    return tqdm(total=total, desc=desc, smoothing=0, mininterval=1,
                bar_format="{desc} | LEFT {remaining} | elapsed {elapsed} | {n_fmt}/{total_fmt} {bar:10} | {postfix}")


def log_counts(counts):
    result = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError("PCA requires finite nonnegative counts")
    np.log1p(result.data, out=result.data)
    return result


class FullDataPCA:
    def __init__(self, mean, components, singular_values, info):
        self.mean = np.asarray(mean)
        self.components = np.asarray(components)
        self.singular_values = np.asarray(singular_values)
        self.info = info
        self.dimension = len(self.components)
        self.offset = self.components @ self.mean
        if self.components.shape[1] != len(self.mean) or not all(
                np.isfinite(a).all() for a in (self.mean, self.components, self.singular_values)):
            raise ValueError("Invalid PCA basis")

    @classmethod
    def fit(cls, counts, settings=None):
        settings = {**PCA_SETTINGS, **(settings or {})}
        x = log_counts(counts)
        n, m = x.shape
        k = min(settings["requested_components"], n - 1, m)
        if k < 1:
            raise ValueError("PCA needs at least two cells and one gene")
        # Some scipy sparse versions accumulate in float32 even when mean's
        # dtype=float64 is requested. Explicit small float64 blocks avoid that
        # drift (and avoid making a second full float64 sparse matrix here).
        mean = np.zeros(m, dtype=np.float64)
        total_sq = 0.0
        for start in range(0, n, 2048):
            block = x[start:start + 2048].astype(np.float64)
            mean += np.asarray(block.sum(axis=0)).ravel()
            total_sq += float(block.data @ block.data)
        mean /= n
        # Small cohorts: exact sample Gram eigendecomposition, including the
        # n_cells-1 case. The large gene-by-gene covariance is never constructed.
        if n <= settings["gram_max_cells"]:
            with progress(3, f"Full-data PCA ({n} cells, {k} PCs): exact Gram") as bar:
                xd = x.astype(np.float64)
                gram = (xd @ xd.T).toarray()
                xm = np.asarray(xd @ mean).ravel()
                gram -= xm[:, None]
                gram -= xm[None, :]
                gram += mean @ mean
                bar.update(1)
                eigenvalues, u = linalg.eigh(gram, subset_by_index=(n - k, n - 1), check_finite=False)
                eigenvalues, u = eigenvalues[::-1], u[:, ::-1]
                singular = np.sqrt(np.maximum(eigenvalues, 0))
                bar.update(1)
                vt = np.asarray(xd.T @ u).T - u.sum(axis=0)[:, None] * mean
                valid = eigenvalues > np.finfo(float).eps * max(total_sq, 1) * max(n, m)
                vt = np.divide(vt, singular[:, None], out=np.zeros_like(vt), where=valid[:, None])
                singular[~valid] = 0
                bar.update(1)
            solver = "exact centered sample-Gram eigendecomposition (float64)"
        else:
            # Halko range finder: sparse products + explicit rank-one centering.
            # Only n_cells x (k+32) and n_genes x (k+32) dense work arrays.
            width = min(k + settings["oversamples"], n - 1, m)
            mu = mean.astype(np.float32)

            def forward(a):
                return np.asarray(x @ a) - mu @ a

            def transpose(a):
                return np.asarray(x.T @ a) - mu[:, None] * a.sum(axis=0)

            def orth(a):
                return linalg.qr(a, mode="economic", overwrite_a=True, check_finite=False)[0]

            with progress(3 + 2 * settings["power_iterations"],
                          f"Full-data PCA ({n} cells, {k} PCs): randomized SVD") as bar:
                rng = np.random.default_rng(settings["seed"])
                omega = rng.standard_normal((m, width), dtype=np.float32)
                q = orth(forward(omega))
                del omega
                bar.update(1)
                for _ in range(settings["power_iterations"]):
                    z = orth(transpose(q))
                    bar.update(1)
                    q = orth(forward(z))
                    del z
                    bar.update(1)
                b = transpose(q).T.copy()
                del q
                bar.update(1)
                _, singular, vt = linalg.svd(b, full_matrices=False, check_finite=False, overwrite_a=True)
                singular, vt = singular[:k], vt[:k]
                bar.update(1)
            solver = "centered randomized SVD (float32 products; QR-normalized power iterations)"
        # Fix arbitrary signs for a deterministic saved projection.
        signs = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])
        signs[signs == 0] = 1
        vt *= signs[:, None]
        total_ss = total_sq - n * float(mean @ mean)
        info = {"n_fit_cells": n, "n_genes": m, "n_components": k, "settings": settings,
                "solver": solver, "explained_variance_ratio_sum":
                    float(np.square(singular.astype(float)).sum() / total_ss) if total_ss > 0 else 0,
                "fit_scope": "ALL prepared eligible cells, ALL original UMIs, including calibration/train/dev/test",
                "preprocessing": "gene-centered log1p integer counts; no library normalization, HVG selection or whitening",
                "transform": "(log1p(current diluted counts) - frozen full-depth mean) @ frozen components.T"}
        return cls(mean, vt.astype(np.float32), singular, info)

    def transform(self, counts):
        if counts.shape[1] != len(self.mean):
            raise ValueError("PCA gene dimension changed")
        result = np.asarray(log_counts(counts) @ self.components.T) - self.offset
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite PCA coordinates")
        return result.astype(np.float32)

    def save(self, path):
        with open(path, "wb") as stream:
            np.savez(stream, mean=self.mean, components=self.components, singular_values=self.singular_values)

    @classmethod
    def load(cls, path, info):
        with np.load(path, allow_pickle=False) as saved:
            return cls(saved["mean"], saved["components"], saved["singular_values"], info)
