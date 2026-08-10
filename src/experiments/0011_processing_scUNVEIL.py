import os
# Set to True to force CPU-only execution for CPU-runtime measurements.
# Set to False to allow to use available GPUs.
FORCE_CPU_ONLY = False
if FORCE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

import anndata as ad
import numpy as np
import time

from constants import DATA_ROOT, PBMC_50_MARKERS, EMB_NAME, IMPUTE_NAME, SC_UNVEIL_VERSION, TIMING_PATH

from scunveil import scUnveil

BATCH_SIZE = 1_000
LIMIT_N_CELLS = None  # Set to an integer for a smoke test.

scUNVEIL = scUnveil(model_version=SC_UNVEIL_VERSION)

collections_root = DATA_ROOT / "downstream_datasets"

for collection_path in sorted(collections_root.iterdir()):

    collection_h5ad = collection_path / f"{collection_path.name}.h5ad"
    collection_adata = ad.read_h5ad(
        collection_h5ad,
        backed=None if collection_path.name == 'Hao_PBMC' else 'r'
    )

    if LIMIT_N_CELLS is not None:
        if not 0 < LIMIT_N_CELLS <= collection_adata.n_obs:
            raise ValueError(
                f"LIMIT_N_CELLS must be between 1 and {collection_adata.n_obs}, "
                f"got {LIMIT_N_CELLS}"
            )
        rng = np.random.default_rng(0)
        indices = np.sort(
            rng.choice(collection_adata.n_obs, LIMIT_N_CELLS, replace=False)
        )
        collection_adata = collection_adata[indices].to_memory()
    if collection_adata.raw is None:
        raise ValueError("AnnData.raw is missing.")
    if collection_adata.shape != collection_adata.raw.shape:
        raise ValueError(
            f"Shape mismatch: adata={collection_adata.shape}, raw={collection_adata.raw.shape}"
        )

    time_start = time.time()
    scUNVEIL.set_input_anndata(collection_adata, batch_size=BATCH_SIZE)

    embeddings = scUNVEIL.get_embeddings(n_features=None)
    time_emb = time.time() - time_start

    marker_imputation = scUNVEIL.get_specific_genes_imputation(PBMC_50_MARKERS, batch_size=BATCH_SIZE)
    time_impute = time.time() - time_start

    if collection_path.name == 'Hao_PBMC' and FORCE_CPU_ONLY and LIMIT_N_CELLS is None:
        # speed test only for PBMC dataset
        if not TIMING_PATH.exists():
            TIMING_PATH.write_text(
                "operation\tseconds\n",
                encoding="utf-8",
            )

        with open(TIMING_PATH, 'a', encoding='utf-8') as fwa:
            print(f'scUNVEIL_emb\t{time_emb}', file=fwa)
            print(f'scUNVEIL_impute\t{time_impute}', file=fwa)


    if LIMIT_N_CELLS is None:
        np.save(collection_path / EMB_NAME / f'scunveil_{embeddings.shape[1]}.npy', embeddings)

    if collection_path.name == 'Hao_PBMC' and LIMIT_N_CELLS is None:
        np.save(collection_path / IMPUTE_NAME / f'scunveil_{marker_imputation.shape[1]}_markers.npy', marker_imputation.X)

    if LIMIT_N_CELLS is not None:
        print(f"[done ] smoke test passed with {collection_adata.n_obs} cells")


