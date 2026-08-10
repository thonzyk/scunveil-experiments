import numpy as np
import os
import gc

import anndata as ad
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
import tempfile

from constants import DATA_ROOT, PBMC_50_MARKERS, EMB_NAME, IMPUTE_NAME, MEASUREMENT_NAME


BATCH_SIZE = 10_000
LIMIT_N_CELLS = None  # Set to an integer for a smoke test.

def low_memory_vstack(blocks):
    if not blocks:
        raise ValueError("Cannot stack an empty list")

    blocks[:] = [x.tocsr(copy=False) for x in blocks]

    n_rows = sum(x.shape[0] for x in blocks)
    n_cols = blocks[0].shape[1]
    nnz = sum(x.nnz for x in blocks)
    dtype = np.result_type(*(x.dtype for x in blocks))

    if nnz == 0:
        blocks.clear()
        return sp.csr_matrix((n_rows, n_cols), dtype=dtype)

    index_dtype = (
        np.int32
        if max(n_rows, n_cols, nnz) <= np.iinfo(np.int32).max
        else np.int64
    )

    tmp = tempfile.TemporaryDirectory(prefix="csr_vstack_")

    data = np.memmap(
        os.path.join(tmp.name, "data"),
        mode="w+",
        dtype=dtype,
        shape=nnz,
    )
    indices = np.memmap(
        os.path.join(tmp.name, "indices"),
        mode="w+",
        dtype=index_dtype,
        shape=nnz,
    )
    indptr = np.memmap(
        os.path.join(tmp.name, "indptr"),
        mode="w+",
        dtype=index_dtype,
        shape=n_rows + 1,
    )

    indptr[0] = 0
    row_offset = 0
    nnz_offset = 0

    for i, block in enumerate(blocks):
        next_row = row_offset + block.shape[0]
        next_nnz = nnz_offset + block.nnz

        data[nnz_offset:next_nnz] = block.data
        indices[nnz_offset:next_nnz] = block.indices
        indptr[row_offset + 1:next_row + 1] = (
            block.indptr[1:].astype(index_dtype, copy=False)
            + nnz_offset
        )

        # Release each original chunk after copying it.
        blocks[i] = None
        row_offset = next_row
        nnz_offset = next_nnz

    result = sp.csr_matrix(
        (data, indices, indptr),
        shape=(n_rows, n_cols),
        copy=False,
    )

    # Keeps the temporary files alive until result is deleted.
    result._vstack_tmpdir = tmp

    return result


datasets_root = DATA_ROOT / "downstream_datasets"

for dataset_path in sorted(datasets_root.iterdir()):

    dataset_h5ad = dataset_path / f"{dataset_path.name}.h5ad"
    dataset_adata = ad.read_h5ad(dataset_h5ad, backed='r')

    if LIMIT_N_CELLS is not None:
        if not 0 < LIMIT_N_CELLS <= dataset_adata.n_obs:
            raise ValueError(
                f"LIMIT_N_CELLS must be between 1 and {dataset_adata.n_obs}, "
                f"got {LIMIT_N_CELLS}"
            )
        rng = np.random.default_rng(0)
        indices = np.sort(
            rng.choice(dataset_adata.n_obs, LIMIT_N_CELLS, replace=False)
        )
        dataset_adata = dataset_adata[indices].to_memory()

    os.makedirs(dataset_path / EMB_NAME, exist_ok=True)
    os.makedirs(dataset_path / IMPUTE_NAME, exist_ok=True)
    os.makedirs(dataset_path / MEASUREMENT_NAME, exist_ok=True)


    rna_log1p = []

    for b_i in tqdm(range(0, dataset_adata.n_obs, BATCH_SIZE)):
        adata_slice = dataset_adata[b_i:b_i+BATCH_SIZE].to_memory()

        splice_x = adata_slice.raw.X

        rna_log1p.append(np.log1p(splice_x))
        del adata_slice, splice_x

    rna_log1p = low_memory_vstack(rna_log1p)
    if LIMIT_N_CELLS is None:
        sp.save_npz(dataset_path / MEASUREMENT_NAME / 'rna_log1p.npz', rna_log1p, compressed=False)

    if dataset_path.name == 'Hao_PBMC':
        # marker gene test only for PBMC dataset
        list_gene_names = list(dataset_adata.var['feature_name'])
        gene_index_lookup = {gene_name: gene_index for gene_index, gene_name in enumerate(list_gene_names)}
        marker_indices = np.array([gene_index_lookup[gene_name] for gene_name in PBMC_50_MARKERS])

        rna_marker_log1p = np.array(rna_log1p[:, marker_indices].copy().todense())

        if LIMIT_N_CELLS is None:
            np.save(dataset_path / MEASUREMENT_NAME / 'rna_marker_log1p.npy', rna_marker_log1p)

    if LIMIT_N_CELLS is not None:
        print(f"[done ] smoke test passed with {dataset_adata.n_obs} cells")

    del rna_log1p
    gc.collect()










