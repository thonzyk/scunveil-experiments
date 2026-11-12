"""
Run scFoundation zero-shot cell embedding generation for all downstream datasets.

================================================================
PREREQUISITES — run these once in a shell before running this script.
================================================================

# 1. Create the conda environment
# conda create -n scfoundation python=3.9 pip -y && \
# conda activate scfoundation && \
# pip install --upgrade pip setuptools wheel && \
# pip install --index-url https://download.pytorch.org/whl/cu117 \
#     torch==1.13.1+cu117 && \
# pip install \
#     numpy==1.23.5 \
#     pandas==1.5.3 \
#     scipy==1.11.4 \
#     scikit-learn==1.3.2 \
#     anndata==0.9.2 \
#     scanpy==1.9.6 \
#     einops==0.7.0 \
#     local-attention==1.8.6 \
#     tqdm

# 2. Clone the scFoundation repository
# git clone https://github.com/biomap-research/scFoundation.git

# 3. Download the 0.1B-parameter checkpoint https://huggingface.co/genbio-ai/scFoundation/resolve/main/models.ckpt
#    Place the checkpoint at:
#      scFoundation/model/models/models.ckpt
#    so that:
#      mkdir -p scFoundation/model/models
#      # put 01B-resolution.ckpt (or equivalent 0.1B checkpoint) in place
================================================================

Expected layout
---------------
<script directory>/
├── run_scfoundation.py
├── constants.py
└── scFoundation/
    ├── OS_scRNA_gene_index.19264.tsv
    └── model/
        ├── get_embedding.py
        └── models/
            └── models.ckpt

Dataset layout
--------------
DATA_ROOT/
└── downstream_datasets/
    ├── Hao_PBMC/
    │   ├── Hao_PBMC.h5ad
    │   └── <EMB_NAME>/
    └── Other_Collection/
        ├── Other_Collection.h5ad
        └── <EMB_NAME>/
"""

import os

# Set to True only for CPU-runtime measurements.
FORCE_CPU_ONLY = False
if FORCE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from constants import DATA_ROOT, EMB_NAME, TIMING_PATH


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCF_REPO = SCRIPT_DIR / "scFoundation"
SCF_MODEL_DIR = SCF_REPO / "model"
SCF_SCRIPT = SCF_MODEL_DIR / "get_embedding.py"
SCF_GENE_FILE = SCF_REPO / "OS_scRNA_gene_index.19264.tsv"
SCF_CHECKPOINT = SCF_MODEL_DIR / "models" / "models.ckpt"

COLLECTIONS_ROOT = DATA_ROOT / "downstream_datasets"

N_GENES = 19_264
EMBEDDING_DIM = 3_072
RAW_GENE_COLUMN = "feature_name"
LIMIT_N_CELLS = None  # Set to an integer for a smoke test.


def read_n_obs(path: Path) -> int:
    adata = ad.read_h5ad(path, backed="r")
    n_obs = adata.n_obs
    adata.file.close()
    return n_obs


def align_genes(
    input_path: Path,
    output_path: Path,
    target_genes: list[str],
) -> None:
    """Align AnnData.raw to scFoundation's ordered gene vocabulary."""
    if output_path.exists():
        cached = ad.read_h5ad(output_path, backed="r")
        valid = (
            cached.n_vars == N_GENES
            and cached.uns.get("scfoundation_gene_column")
            == RAW_GENE_COLUMN
        )
        cached.file.close()

        if valid:
            print(f"[align] reusing {output_path}")
            return

        print(f"[align] removing invalid cached file {output_path}")
        output_path.unlink()

    print(f"[align] loading {input_path}")
    adata = ad.read_h5ad(input_path)

    if adata.raw is None:
        raise ValueError(f"AnnData.raw is missing: {input_path}")

    if adata.n_obs != adata.raw.n_obs:
        raise ValueError(
            f"Cell-count mismatch: "
            f"adata={adata.n_obs}, raw={adata.raw.n_obs}"
        )

    if RAW_GENE_COLUMN not in adata.raw.var.columns:
        raise ValueError(
            f"Missing adata.raw.var[{RAW_GENE_COLUMN!r}]. "
            f"Available columns: {adata.raw.var.columns.tolist()}"
        )

    source_genes = pd.Index(
        adata.raw.var[RAW_GENE_COLUMN].astype(str).str.strip()
    )

    target_lookup = {
        gene: target_i
        for target_i, gene in enumerate(target_genes)
    }

    source_columns = []
    target_columns = []

    for source_i, gene in enumerate(source_genes):
        target_i = target_lookup.get(gene)

        if target_i is not None:
            source_columns.append(source_i)
            target_columns.append(target_i)

    if not source_columns:
        raise ValueError(
            f"No genes matched the scFoundation vocabulary: {input_path}"
        )

    source_columns = np.asarray(source_columns, dtype=np.int64)
    target_columns = np.asarray(target_columns, dtype=np.int64)

    matched_genes = len(np.unique(target_columns))
    duplicate_mappings = len(source_columns) - matched_genes

    print(
        f"[align] matched {matched_genes:,} / "
        f"{N_GENES:,} scFoundation genes"
    )

    if duplicate_mappings:
        print(
            f"[align] summing {duplicate_mappings:,} duplicate "
            f"source-gene mappings"
        )

    if matched_genes < 10_000:
        print(
            "[warn ] low gene match; check Ensembl IDs "
            "versus gene symbols"
        )

    X = sparse.csr_matrix(
        adata.raw.X[:, source_columns],
        dtype=np.float32,
    )

    mapping = sparse.csr_matrix(
        (
            np.ones(len(source_columns), dtype=np.float32),
            (
                np.arange(len(source_columns)),
                target_columns,
            ),
        ),
        shape=(len(source_columns), N_GENES),
        dtype=np.float32,
    )

    aligned_X = sparse.csr_matrix(
        X @ mapping,
        dtype=np.float32,
    )

    aligned = ad.AnnData(
        X=aligned_X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(
            index=pd.Index(target_genes, name="gene_name")
        ),
    )
    aligned.var["gene_name"] = target_genes
    aligned.uns["scfoundation_matched_genes"] = matched_genes
    aligned.uns["scfoundation_source_columns"] = len(source_columns)
    aligned.uns["scfoundation_duplicate_mappings"] = duplicate_mappings
    aligned.uns["scfoundation_gene_column"] = RAW_GENE_COLUMN

    print(f"[align] writing {output_path}")
    aligned.write_h5ad(output_path, compression="gzip")


def prepare_input(
    aligned_path: Path,
    embedding_dir: Path,
    collection_name: str,
) -> tuple[Path, str, int]:
    """Return the full aligned input or a deterministic smoke-test subset."""
    n_cells = read_n_obs(aligned_path)

    if LIMIT_N_CELLS is None:
        return aligned_path, f"{collection_name}_scfoundation", n_cells

    if not 0 < LIMIT_N_CELLS <= n_cells:
        raise ValueError(
            f"LIMIT_N_CELLS must be between 1 and {n_cells}, "
            f"got {LIMIT_N_CELLS}"
        )

    subset_path = embedding_dir / f"_scfoundation_subset_{LIMIT_N_CELLS}.h5ad"

    if not subset_path.exists():
        adata = ad.read_h5ad(aligned_path)
        rng = np.random.default_rng(0)
        indices = np.sort(
            rng.choice(adata.n_obs, LIMIT_N_CELLS, replace=False)
        )
        adata[indices].copy().write_h5ad(
            subset_path,
            compression="gzip",
        )

    task_name = f"{collection_name}_scfoundation_smoke_{LIMIT_N_CELLS}"
    return subset_path, task_name, LIMIT_N_CELLS


def run_scfoundation(
    data_path: Path,
    output_dir: Path,
    task_name: str,
) -> Path:
    """Run get_embedding.py and return the newly created embedding file."""
    command = [
        sys.executable,
        str(SCF_SCRIPT),
        "--task_name", task_name,
        "--input_type", "singlecell",
        "--output_type", "cell",
        "--pool_type", "all",
        "--tgthighres", "f1",
        "--pre_normalized", "F",
        "--version", "rde",
        "--data_path", str(data_path),
        "--save_path", str(output_dir),
    ]

    started = time.time()
    print(f"[run  ] {' '.join(command)}")

    subprocess.run(
        command,
        cwd=SCF_MODEL_DIR,
        check=True,
        env=os.environ.copy(),
    )

    candidates = [
        path
        for path in output_dir.rglob("*.npy")
        if task_name in path.name and path.stat().st_mtime >= started - 2
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No new embedding file found for task '{task_name}'"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def save_timing(seconds: float) -> None:
    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TIMING_PATH.exists():
        TIMING_PATH.write_text(
            "operation\tseconds\n",
            encoding="utf-8",
        )

    with TIMING_PATH.open("a", encoding="utf-8") as file:
        print(f"scFoundation_emb\t{seconds}", file=file)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

for required_path in (
    SCF_SCRIPT,
    SCF_GENE_FILE,
    SCF_CHECKPOINT,
    COLLECTIONS_ROOT,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

scf_genes = (
    pd.read_csv(SCF_GENE_FILE, sep="\t")["gene_name"]
    .astype(str)
    .str.strip()
    .tolist()
)

if len(scf_genes) != N_GENES or len(set(scf_genes)) != N_GENES:
    raise ValueError("Invalid scFoundation gene vocabulary.")

for collection_path in sorted(COLLECTIONS_ROOT.iterdir()):
    if not collection_path.is_dir():
        continue

    collection_name = collection_path.name
    collection_h5ad = collection_path / f"{collection_name}.h5ad"
    embedding_dir = collection_path / EMB_NAME
    aligned_h5ad = collection_path / "_scfoundation_aligned_19264.h5ad"

    if not collection_h5ad.exists():
        raise FileNotFoundError(collection_h5ad)

    print(f"\n{'=' * 70}\n[data ] {collection_name}\n{'=' * 70}")

    embedding_dir.mkdir(parents=True, exist_ok=True)

    final_path = embedding_dir / "scFoundation.npy"

    if LIMIT_N_CELLS is None and final_path.exists():
        print(f"[skip ] output already exists: {final_path}")
        continue

    align_genes(
        collection_h5ad,
        aligned_h5ad,
        scf_genes,
    )

    data_path, task_name, expected_cells = prepare_input(
        aligned_h5ad,
        embedding_dir,
        collection_name,
    )

    started = time.time()

    generated_path = run_scfoundation(
        data_path,
        embedding_dir,
        task_name,
    )

    runtime = time.time() - started

    embeddings = np.load(generated_path, mmap_mode="r")
    expected_shape = (expected_cells, EMBEDDING_DIM)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{collection_name}: got {embeddings.shape}, "
            f"expected {expected_shape}"
        )

    if not np.issubdtype(embeddings.dtype, np.number):
        raise TypeError(f"Invalid embedding dtype: {embeddings.dtype}")

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"Non-finite embeddings found in {generated_path}"
        )

    print(f"[done ] shape: {embeddings.shape}")
    print(f"[done ] time:  {runtime:.3f} seconds")

    if LIMIT_N_CELLS is None:
        final_path = embedding_dir / f"scFoundation.npy"

        if generated_path.resolve() != final_path.resolve():
            generated_path.replace(final_path)

        print(f"[save ] {final_path}")

        if collection_name == "Hao_PBMC" and FORCE_CPU_ONLY:
            save_timing(runtime)
    else:
        print("[done ] smoke test passed")