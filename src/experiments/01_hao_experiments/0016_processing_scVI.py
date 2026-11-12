"""Generate scVI embeddings and imputed expression for downstream datasets.

Environment setup:
    conda create -n scvi python=3.10 -y
    conda activate scvi
    pip install "scvi-tools[tutorials]"

The three embedding models preserve the legacy benchmark dimensions. A
separate default scVI model produces log10 CPM imputation in row batches so
the full matrix does not need to be materialized in RAM. Marker imputation is
only written for Hao PBMC, where the corresponding benchmark is defined.
"""

import os

FORCE_CPU_ONLY = False
if FORCE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from pathlib import Path

import anndata as ad
import numpy as np
import scvi
from numpy.lib.format import open_memmap

from constants import DATA_ROOT, EMB_NAME, IMPUTE_NAME, PBMC_50_MARKERS, TIMING_PATH


LATENT_DIMS = (10, 50, 512)
IMPUTATION_BATCH_SIZE = 1_024
BATCH_KEY_CANDIDATES = ("donor_id", "donor", "batch")
GENE_SYMBOL_COLUMNS = ("feature_name", "gene_name", "gene_symbols", "symbol")
LIMIT_N_CELLS = None  # Set to an integer for a smoke test.


def batch_key_for(adata: ad.AnnData) -> str:
    for key in BATCH_KEY_CANDIDATES:
        if key in adata.obs:
            return key
    raise ValueError(
        "scVI requires a batch column. Tried: "
        f"{', '.join(BATCH_KEY_CANDIDATES)}"
    )


def gene_symbols(adata: ad.AnnData) -> np.ndarray:
    for column in GENE_SYMBOL_COLUMNS:
        if column in adata.var:
            return adata.var[column].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def marker_indices_for(adata: ad.AnnData) -> np.ndarray:
    lookup = {symbol: i for i, symbol in enumerate(gene_symbols(adata))}
    missing = [marker for marker in PBMC_50_MARKERS if marker not in lookup]
    if missing:
        raise ValueError(f"Missing PBMC marker genes: {missing}")
    return np.asarray([lookup[marker] for marker in PBMC_50_MARKERS])


def append_timing(operation: str, seconds: float) -> None:
    if not TIMING_PATH.exists():
        TIMING_PATH.write_text("operation\tseconds\n", encoding="utf-8")
    with TIMING_PATH.open("a", encoding="utf-8") as stream:
        print(f"{operation}\t{seconds}", file=stream)


def load_counts(path: Path) -> ad.AnnData:
    source = ad.read_h5ad(path)
    if source.raw is None:
        raise ValueError(f"AnnData.raw is missing: {path}")
    counts = source.raw.to_adata()
    del source

    if LIMIT_N_CELLS is not None:
        if not 0 < LIMIT_N_CELLS <= counts.n_obs:
            raise ValueError(
                f"LIMIT_N_CELLS must be between 1 and {counts.n_obs}, "
                f"got {LIMIT_N_CELLS}"
            )
        rng = np.random.default_rng(0)
        indices = np.sort(
            rng.choice(counts.n_obs, LIMIT_N_CELLS, replace=False)
        )
        counts = counts[indices].copy()

    return counts


def train_model(adata: ad.AnnData, n_latent: int | None = None):
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key_for(adata))
    kwargs = {} if n_latent is None else {"n_latent": n_latent}
    model = scvi.model.SCVI(adata, **kwargs)
    model.train(early_stopping=True)
    return model


def write_imputation(
    model,
    adata: ad.AnnData,
    output_path: Path,
    marker_output_path: Path | None,
) -> None:
    output = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=adata.shape,
    )
    markers = marker_indices_for(adata) if marker_output_path else None
    marker_output = (
        np.empty((adata.n_obs, len(markers)), dtype=np.float16)
        if markers is not None
        else None
    )

    for start in range(0, adata.n_obs, IMPUTATION_BATCH_SIZE):
        stop = min(start + IMPUTATION_BATCH_SIZE, adata.n_obs)
        values = model.get_normalized_expression(
            indices=np.arange(start, stop),
            library_size=1e6,
            return_numpy=True,
        )
        values = np.log10(np.asarray(values) + 1e-9).astype(np.float16)
        output[start:stop] = values
        if marker_output is not None:
            marker_output[start:stop] = values[:, markers]

    output.flush()
    del output
    if marker_output_path is not None:
        np.save(marker_output_path, marker_output)


collections_root = DATA_ROOT / "downstream_datasets"

for collection_path in sorted(collections_root.iterdir()):
    input_path = collection_path / f"{collection_path.name}.h5ad"
    if not input_path.exists():
        continue

    embedding_dir = collection_path / EMB_NAME
    imputation_dir = collection_path / IMPUTE_NAME
    embedding_dir.mkdir(parents=True, exist_ok=True)
    imputation_dir.mkdir(parents=True, exist_ok=True)

    for n_latent in LATENT_DIMS:
        adata = load_counts(input_path)
        start_time = time.time()
        model = train_model(adata, n_latent=n_latent)
        embeddings = model.get_latent_representation()
        if LIMIT_N_CELLS is None:
            np.save(embedding_dir / f"scvi_{n_latent}.npy", embeddings)
        else:
            print(
                f"[done ] scVI {n_latent}-latent smoke test passed "
                f"with {adata.n_obs} cells"
            )

        if collection_path.name == "Hao_PBMC" and FORCE_CPU_ONLY and LIMIT_N_CELLS is None:
            append_timing(f"scVI_{n_latent}", time.time() - start_time)

        del embeddings, model, adata

    adata = load_counts(input_path)
    start_time = time.time()
    model = train_model(adata)
    marker_path = (
        imputation_dir / "scvi_50_markers.npy"
        if collection_path.name == "Hao_PBMC" and LIMIT_N_CELLS is None
        else None
    )
    imputation_path = (
        imputation_dir / "scvi.npy"
        if LIMIT_N_CELLS is None
        else imputation_dir / f"_scvi_smoke_{LIMIT_N_CELLS}.npy"
    )
    write_imputation(model, adata, imputation_path, marker_path)

    if LIMIT_N_CELLS is not None:
        print(
            f"[done ] scVI imputation smoke test passed "
            f"with {adata.n_obs} cells"
        )

    if collection_path.name == "Hao_PBMC" and FORCE_CPU_ONLY and LIMIT_N_CELLS is None:
        append_timing("scVI_imputation", time.time() - start_time)

    del model, adata

