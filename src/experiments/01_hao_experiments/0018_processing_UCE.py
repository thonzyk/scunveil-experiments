"""Generate 4- and 33-layer UCE embeddings for downstream datasets.

Clone UCE into ``src/experiments/UCE`` and download its model files into the
repository's ``model_files`` directory. Run this script from the UCE conda
environment. UCE consumes raw counts with HGNC symbols as ``var_names``.
"""

import os

FORCE_CPU_ONLY = False
if FORCE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

from constants import DATA_ROOT, EMB_NAME, TIMING_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
UCE_REPO = SCRIPT_DIR / "UCE"
MODEL_DIR = UCE_REPO / "model_files"
EVAL_SCRIPT = UCE_REPO / "eval_single_anndata.py"
SPECIES = "human"
EMBEDDING_DIM = 1_280
LIMIT_N_CELLS = None  # Set to an integer for a deterministic smoke test.
GENE_SYMBOL_COLUMNS = ("feature_name", "gene_name", "gene_symbols", "symbol")

MODEL_CONFIGS = (
    (4, MODEL_DIR / "4layer_model.torch", 25),
    (33, MODEL_DIR / "33l_8ep_1024t_1280.torch", 2),
)


def append_timing(operation: str, seconds: float) -> None:
    if not TIMING_PATH.exists():
        TIMING_PATH.write_text("operation\tseconds\n", encoding="utf-8")
    with TIMING_PATH.open("a", encoding="utf-8") as stream:
        print(f"{operation}\t{seconds}", file=stream)


def read_n_obs(path: Path) -> int:
    adata = ad.read_h5ad(path, backed="r")
    n_obs = adata.n_obs
    adata.file.close()
    return n_obs


def prepare_counts(input_path: Path, prepared_path: Path) -> None:
    if prepared_path.exists():
        print(f"[prep ] reusing {prepared_path}")
        return

    source = ad.read_h5ad(input_path)
    if source.raw is None:
        raise ValueError(f"AnnData.raw is missing: {input_path}")
    adata = source.raw.to_adata()
    del source

    for column in GENE_SYMBOL_COLUMNS:
        if column in adata.var:
            adata.var_names = adata.var[column].astype(str).to_numpy()
            break
    else:
        if all(str(name).startswith("ENSG") for name in adata.var_names[:5]):
            raise ValueError(
                f"No gene-symbol column found in {input_path}; available "
                f"columns: {adata.var.columns.tolist()}"
            )

    adata.var_names_make_unique()
    adata.write_h5ad(prepared_path, compression="gzip")


def input_for_run(prepared_path: Path, work_dir: Path) -> tuple[Path, int]:
    n_obs = read_n_obs(prepared_path)
    if LIMIT_N_CELLS is None:
        return prepared_path, n_obs
    if not 0 < LIMIT_N_CELLS <= n_obs:
        raise ValueError(
            f"LIMIT_N_CELLS must be between 1 and {n_obs}, "
            f"got {LIMIT_N_CELLS}"
        )

    subset_path = work_dir / f"_uce_subset_{LIMIT_N_CELLS}.h5ad"
    if not subset_path.exists():
        adata = ad.read_h5ad(prepared_path)
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(n_obs, LIMIT_N_CELLS, replace=False))
        adata[indices].copy().write_h5ad(subset_path, compression="gzip")
    return subset_path, LIMIT_N_CELLS


def newest_uce_output(run_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in run_dir.rglob("*.h5ad")
            if "uce" in path.name.lower()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No UCE output h5ad found under {run_dir}")
    return candidates[-1]


if not EVAL_SCRIPT.is_file():
    raise FileNotFoundError(
        f"UCE evaluation script not found: {EVAL_SCRIPT}. "
        "Clone the UCE repository before running."
    )

for _, weights, _ in MODEL_CONFIGS:
    if not weights.is_file():
        raise FileNotFoundError(f"UCE weights not found: {weights}")

collections_root = DATA_ROOT / "downstream_datasets"

for collection_path in sorted(collections_root.iterdir()):
    input_path = collection_path / f"{collection_path.name}.h5ad"
    if not input_path.exists():
        continue

    embedding_dir = collection_path / EMB_NAME
    embedding_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = embedding_dir / "_uce_prepared_counts.h5ad"
    prepare_counts(input_path, prepared_path)
    data_path, expected_n_obs = input_for_run(prepared_path, embedding_dir)

    for n_layers, weights, batch_size in MODEL_CONFIGS:
        run_dir = embedding_dir / f"_uce_{n_layers}l_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--adata_path",
            str(data_path),
            "--dir",
            f"{run_dir}/",
            "--species",
            SPECIES,
            "--model_loc",
            str(weights),
            "--nlayers",
            str(n_layers),
            "--batch_size",
            str(batch_size),
        ]
        print(f"[run  ] {' '.join(command)}")
        subprocess.run(command, cwd=UCE_REPO, check=True)

        result = ad.read_h5ad(newest_uce_output(run_dir))
        if "X_uce" not in result.obsm:
            raise KeyError(f"UCE result is missing obsm['X_uce']")
        embeddings = np.asarray(result.obsm["X_uce"])
        expected_shape = (expected_n_obs, EMBEDDING_DIM)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Unexpected UCE shape {embeddings.shape}; expected "
                f"{expected_shape}"
            )

        if LIMIT_N_CELLS is None:
            np.save(
                embedding_dir / f"UCE_{n_layers}L_{EMBEDDING_DIM}.npy",
                embeddings,
            )
        else:
            print(
                f"[done ] {n_layers}L smoke test passed with "
                f"{LIMIT_N_CELLS} cells"
            )

        if collection_path.name == "Hao_PBMC" and FORCE_CPU_ONLY and LIMIT_N_CELLS is None:
            append_timing(f"UCE_{n_layers}L", time.time() - start_time)

        del embeddings, result

