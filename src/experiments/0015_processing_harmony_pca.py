"""Generate Harmony-corrected PCA embeddings for downstream datasets.

Environment setup:
    conda create -n harmony python=3.10 -y
    conda activate harmony
    pip install scanpy harmonypy
"""

import time

import anndata as ad
import harmonypy as hm
import numpy as np
import scanpy as sc

from constants import DATA_ROOT, EMB_NAME, TIMING_PATH


N_COMPONENTS = (10, 50, 512)
N_HIGHLY_VARIABLE_GENES = 2_000
BATCH_KEY_CANDIDATES = ("donor_id", "donor", "batch")
LIMIT_N_CELLS = None  # Set to an integer for a smoke test.


def batch_key_for(adata: ad.AnnData) -> str:
    for key in BATCH_KEY_CANDIDATES:
        if key in adata.obs:
            return key
    raise ValueError(
        "Harmony requires a batch column. Tried: "
        f"{', '.join(BATCH_KEY_CANDIDATES)}"
    )


def append_timing(operation: str, seconds: float) -> None:
    if not TIMING_PATH.exists():
        TIMING_PATH.write_text("operation\tseconds\n", encoding="utf-8")
    with TIMING_PATH.open("a", encoding="utf-8") as stream:
        print(f"{operation}\t{seconds}", file=stream)


collections_root = DATA_ROOT / "downstream_datasets"

for collection_path in sorted(collections_root.iterdir()):
    input_path = collection_path / f"{collection_path.name}.h5ad"
    if not input_path.exists():
        continue

    output_dir = collection_path / EMB_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(input_path)
    if source.raw is None:
        raise ValueError(f"AnnData.raw is missing: {input_path}")

    adata = source.raw.to_adata()
    del source

    if LIMIT_N_CELLS is not None:
        if not 0 < LIMIT_N_CELLS <= adata.n_obs:
            raise ValueError(
                f"LIMIT_N_CELLS must be between 1 and {adata.n_obs}, "
                f"got {LIMIT_N_CELLS}"
            )
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(adata.n_obs, LIMIT_N_CELLS, replace=False))
        adata = adata[indices].copy()

    batch_key = batch_key_for(adata)

    start_time = time.time()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(N_HIGHLY_VARIABLE_GENES, adata.n_vars),
    )
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    preprocessing_seconds = time.time() - start_time

    for n_components in N_COMPONENTS:
        if n_components >= min(adata.n_obs, adata.n_vars):
            raise ValueError(
                f"Cannot compute {n_components} PCs for {input_path} "
                f"with shape {adata.shape}"
            )

        component_start = time.time()
        sc.tl.pca(adata, n_comps=n_components)
        harmony = hm.run_harmony(
            adata.obsm["X_pca"],
            adata.obs,
            batch_key,
        )
        embeddings = np.asarray(harmony.Z_corr)

        # harmonypy versions have returned both (PCs, cells) and
        # (cells, PCs); store every representation as (cells, features).
        if embeddings.shape == (n_components, adata.n_obs):
            embeddings = embeddings.T
        if embeddings.shape != (adata.n_obs, n_components):
            raise ValueError(
                f"Unexpected Harmony shape {embeddings.shape}; expected "
                f"({adata.n_obs}, {n_components})"
            )

        if LIMIT_N_CELLS is None:
            np.save(
                output_dir / f"harmony_pca_{n_components}.npy",
                embeddings,
            )

        if collection_path.name == "Hao_PBMC" and LIMIT_N_CELLS is None:
            append_timing(
                f"harmony_pca_{n_components}",
                preprocessing_seconds + time.time() - component_start,
            )

        if LIMIT_N_CELLS is not None:
            print(
                f"[done ] Harmony {n_components}-PC smoke test passed "
                f"with {adata.n_obs} cells"
            )

