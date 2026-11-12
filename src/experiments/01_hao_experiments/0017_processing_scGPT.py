"""Generate scGPT zero-shot cell embeddings for downstream datasets.

Install scGPT in its own environment and place the human pretrained model at
``src/experiments/scGPT_weights_human``. The directory should contain the
model, vocabulary, and argument files expected by ``scgpt.tasks.embed_data``.
"""

import os

FORCE_CPU_ONLY = False
if FORCE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from pathlib import Path

import anndata as ad
import numpy as np
import scgpt as scg

from constants import DATA_ROOT, EMB_NAME, TIMING_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "scGPT_weights_human"
BATCH_SIZE = 64
GENE_SYMBOL_COLUMNS = ("feature_name", "gene_name", "gene_symbols", "symbol")
LIMIT_N_CELLS = None  # Set to an integer for a smoke test.


def append_timing(operation: str, seconds: float) -> None:
    if not TIMING_PATH.exists():
        TIMING_PATH.write_text("operation\tseconds\n", encoding="utf-8")
    with TIMING_PATH.open("a", encoding="utf-8") as stream:
        print(f"{operation}\t{seconds}", file=stream)


def load_counts_with_symbols(path: Path) -> ad.AnnData:
    source = ad.read_h5ad(path)
    if source.raw is None:
        raise ValueError(f"AnnData.raw is missing: {path}")
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

    for column in GENE_SYMBOL_COLUMNS:
        if column in adata.var:
            adata.var["gene_name"] = adata.var[column].astype(str).to_numpy()
            return adata
    adata.var["gene_name"] = adata.var_names.astype(str)
    return adata


if not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"scGPT model directory not found: {MODEL_DIR}. "
        "Download the pretrained human model there before running."
    )

collections_root = DATA_ROOT / "downstream_datasets"

for collection_path in sorted(collections_root.iterdir()):
    input_path = collection_path / f"{collection_path.name}.h5ad"
    if not input_path.exists():
        continue

    output_dir = collection_path / EMB_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = load_counts_with_symbols(input_path)

    start_time = time.time()
    embedded = scg.tasks.embed_data(
        adata,
        model_dir=str(MODEL_DIR),
        gene_col="gene_name",
        batch_size=BATCH_SIZE,
        return_new_adata=True,
    )
    embeddings = np.asarray(embedded.X)
    if embeddings.shape[0] != adata.n_obs:
        raise ValueError(
            f"Unexpected scGPT shape {embeddings.shape}; expected "
            f"{adata.n_obs} rows"
        )
    if LIMIT_N_CELLS is None:
        np.save(output_dir / "scGPT.npy", embeddings)
    else:
        print(f"[done ] smoke test passed with {adata.n_obs} cells")

    if collection_path.name == "Hao_PBMC" and FORCE_CPU_ONLY and LIMIT_N_CELLS is None:
        append_timing("scGPT", time.time() - start_time)

    del embeddings, embedded, adata

