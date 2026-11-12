"""Strict functionally assayed GSE254250 cells; original integer RNA counts only."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from tcr_common import HERE, digest, lock, seal, sha256, verify, write_frame, write_json

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE254nnn/GSE254250/suppl/"
SAMPLES = ("249", "262", "282", "309", "418", "432", "Pool")
METADATA = "GSE254250_PDAC_9sample_CD8_RNA_metadata_matrix.csv.gz"
LEGACY = HERE.parents[1] / "_archive/04_experimental_validation_width/02_tumor_reactive_tcr/this_data"
SOURCE_FILES = (METADATA,) + tuple(f"GSE254250_{s}_GEX_filtered_feature_bc_matrix.h5" for s in SAMPLES)


def obtain(filename, output_root):
    """Reuse existing source downloads read-only; put any new downloads in results."""
    existing = LEGACY / filename
    if existing.is_file():
        return existing
    path = Path(output_root) / "downloads" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock(path.with_suffix(path.suffix + ".lock")):
        if not path.is_file():
            temporary = path.with_name(path.name + ".part")
            print(f"Downloading {filename}", flush=True)
            try:
                with urlopen(GEO + filename, timeout=120) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return path


def validate_counts(data):
    data.X = sparse.csr_matrix(data.X)
    data.X.sum_duplicates()
    data.X.eliminate_zeros()
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique")
    values = data.X.data
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Expected nonnegative integer UMI counts")
    data.X = data.X.astype(np.int64)
    required = {"patient", "clonotype_id", "outcome", "library"}
    if not required.issubset(data.obs) or data.obs[list(required)].isna().any().any():
        raise ValueError("Missing required labels/identifiers")
    if set(data.obs.outcome) != {0, 1}:
        raise ValueError("Both assayed outcomes required")
    groups = data.obs.groupby("clonotype_id", observed=True)
    if groups.patient.nunique().max() != 1 or groups.outcome.nunique().max() != 1:
        raise ValueError("A clonotype must have exactly one patient and assay label")
    return data


def clone_table(obs):
    return obs.groupby("clonotype_id", observed=True, sort=True).agg(
        patient=("patient", "first"), outcome=("outcome", "first"), n_cells=("patient", "size"))


def load_counts(paths):
    meta = pd.read_csv(paths[METADATA], usecols=range(9), low_memory=False)
    meta = meta[meta["Reactivity (n=118)"].isin(["TR", "NTR"])].copy()
    if not meta["Unnamed: 0"].is_unique:
        raise ValueError("Duplicate metadata cell identifiers")
    meta["barcode"] = meta["Unnamed: 0"].str.split("_", n=1).str[1]
    matrices, observations, gene_ids, gene_symbols = [], [], None, None
    for sample in SAMPLES:
        patients = ("TIPC301", "TIPC413", "TIPC416") if sample == "Pool" else ("TIPC" + sample,)
        subset = meta[meta["orig.ident"].isin(patients)].copy()
        with h5py.File(paths[f"GSE254250_{sample}_GEX_filtered_feature_bc_matrix.h5"]) as handle:
            group = handle["matrix"]
            selected = np.flatnonzero(group["features/feature_type"].asstr()[:] == "Gene Expression")
            ids = group["features/id"].asstr()[:][selected]
            symbols = group["features/name"].asstr()[:][selected]
            if gene_ids is None:
                gene_ids, gene_symbols = ids, symbols
            elif not np.array_equal(gene_ids, ids) or not np.array_equal(gene_symbols, symbols):
                raise ValueError(f"Inconsistent RNA gene panel: {sample}")
            barcodes = pd.Index(group["barcodes"].asstr()[:])
            if not barcodes.is_unique:
                raise ValueError(f"Duplicate source barcodes: {sample}")
            columns = barcodes.get_indexer(subset.barcode)
            if (columns < 0).any() or len(set(columns)) != len(columns):
                raise ValueError(f"Missing or duplicated selected cells: {sample}")
            matrix = sparse.csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]),
                                       shape=tuple(group["shape"][:]))
            matrices.append(matrix[selected, :][:, columns].T.tocsr())
        observations.append(pd.DataFrame({"patient": subset["orig.ident"].to_numpy(),
            "clonotype_id": subset.clonotype_id.astype(str).to_numpy(),
            "outcome": subset["Reactivity (n=118)"].map({"TR": 1, "NTR": 0}).to_numpy(),
            "library": sample}, index=pd.Index(subset["Unnamed: 0"], name="cell_id")))
    obs = pd.concat(observations)
    if len(obs) != len(meta):
        raise ValueError("Not all strict-label cells matched to counts")
    data = validate_counts(ad.AnnData(sparse.vstack(matrices, format="csr"), obs=obs,
        var=pd.DataFrame({"feature_id": gene_ids, "gene_symbol": gene_symbols}, index=gene_ids)))
    if (data.n_obs, data.obs.clonotype_id.nunique(), data.obs.patient.nunique()) != (3990, 118, 9):
        raise ValueError("Source release differs from the expected 3990 cells / 118 clonotypes / 9 patients")
    return data


def prepare(settings):
    paths = {name: obtain(name, settings["output_root"]) for name in SOURCE_FILES}
    identity = {"sources": {name: sha256(path) for name, path in paths.items()},
                "loader_sha256": sha256(Path(__file__)), "labels": "Reactivity (n=118): TR/NTR"}
    directory = Path(settings["output_root"]) / "prepared" / digest(identity)[:12]
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory.parent / f"{directory.name}.lock"):
        if (directory / "complete.json").is_file():
            return directory, verify(directory)
        data = load_counts(paths)
        clones = clone_table(data.obs)
        data.write_h5ad(directory / "counts.h5ad", compression="gzip")
        write_frame(directory / "clonotypes.tsv", clones.reset_index())
        audit = clones.groupby("patient").agg(n_clones=("outcome", "size"), reactive=("outcome", "sum"))
        audit["nonreactive"] = audit.n_clones - audit.reactive
        audit["n_cells"] = data.obs.groupby("patient", observed=True).size()
        audit["library"] = data.obs.groupby("patient", observed=True).library.first()
        write_frame(directory / "patients.tsv", audit.reset_index())
        write_json(directory / "source.json", {"identity": identity, "paths": {k: str(v) for k, v in paths.items()},
            "study": "https://doi.org/10.1126/scitranslmed.adh9562", "accession": "GSE254250",
            "counts_policy": "Only Gene Expression features; no RNA normalization or label-derived gene selection"})
        seal(directory, {"identity": identity, "n_cells": data.n_obs, "n_clones": len(clones),
                         "n_patients": data.obs.patient.nunique()})
    return directory, verify(directory)
