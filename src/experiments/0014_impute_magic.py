import scanpy as sc
import magic
import numpy as np
import time

from constants import DATA_ROOT, TIMING_PATH, PBMC_50_MARKERS

LIMIT_N_CELLS = None  # Set to an integer for a smoke test.

adata = sc.read_h5ad(DATA_ROOT / "downstream_datasets/Hao_PBMC/Hao_PBMC.h5ad").raw.to_adata()

if LIMIT_N_CELLS is not None:
    if not 0 < LIMIT_N_CELLS <= adata.n_obs:
        raise ValueError(
            f"LIMIT_N_CELLS must be between 1 and {adata.n_obs}, "
            f"got {LIMIT_N_CELLS}"
        )
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(adata.n_obs, LIMIT_N_CELLS, replace=False))
    adata = adata[indices].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)



magic_op = magic.MAGIC(
    solver='approximate',   # landmark-based, scales to 100k+ cells
    n_jobs=-1,              # use all cores
    verbose=1,              # so you can see progress
)

start_time = time.time()

adata_magic = magic_op.fit_transform(adata)

elaps_time = time.time() - start_time

X_magic = adata_magic.X
if hasattr(X_magic, 'toarray'):
    X_magic = X_magic.toarray()
X_magic = X_magic.astype(np.float16)

# log what MAGIC actually chose
print(f"MAGIC params used: t={magic_op.t}, knn={magic_op.knn}, decay={magic_op.decay}")


# speed test only for the full PBMC dataset
if LIMIT_N_CELLS is None and not TIMING_PATH.exists():
    TIMING_PATH.write_text(
        "operation\tseconds\n",
        encoding="utf-8",
    )

if LIMIT_N_CELLS is None:
    with open(TIMING_PATH, 'a', encoding='utf-8') as fwa:
        print(f'MAGIC\t{elaps_time}', file=fwa)


if LIMIT_N_CELLS is None:
    np.save(DATA_ROOT / 'downstream_datasets/Hao_PBMC/imputation/magic.npy', X_magic)

list_gene_names = list(adata.var['feature_name'])
gene_index_lookup = {gene_name: gene_index for gene_index, gene_name in enumerate(list_gene_names)}

marker_indices = np.array([gene_index_lookup[gene_name] for gene_name in PBMC_50_MARKERS])

magic_50_marker = np.array(X_magic[:, marker_indices])
if LIMIT_N_CELLS is None:
    np.save(DATA_ROOT / "downstream_datasets/Hao_PBMC/imputation/magic_50_marker.npy", magic_50_marker)
else:
    print(f"[done ] smoke test passed with {adata.n_obs} cells")

