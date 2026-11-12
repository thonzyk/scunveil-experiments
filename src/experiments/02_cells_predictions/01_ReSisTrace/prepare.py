"""Download only pretreatment GEO tables and build a sparse, audited dataset."""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from urllib.request import urlopen

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from common import (
    SAMPLES, SCHEMA_VERSION, UPSTREAM_REVISION, digest, prepared_manifest,
    provenance, read_json, sha256_file, write_json, write_table,
)


def sample_name(sample):
    return f'{sample["condition"]}_{sample["replicate"]}'


def source_files(sample):
    prefix = f'{sample["gsm"]}_ReSisTrace_{sample_name(sample)}_pre_treatment'
    base = f'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6938nnn/{sample["gsm"]}/suppl/'
    return {kind: (prefix + suffix, base + prefix + suffix) for kind, suffix in (
        ("counts", "_UMIcounts.tsv.gz"), ("metadata", "_cell_info.tsv.gz"),
    )}


def obtain(path, url):
    """Download atomically. Existing files are checked against their receipt."""
    path = Path(path)
    receipt = path.with_name(path.name + ".json")
    if path.exists() and receipt.exists():
        record = read_json(receipt)
        if record["url"] != url or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Source file changed or was corrupted: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    print(f"Downloading {url}", flush=True)
    with urlopen(url, timeout=60) as response, temporary.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    temporary.replace(path)
    write_json(receipt, {"url": url, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return path


def lineage_groups(metadata, namespace):
    """Connected components of BOTH released sister and pre/post family IDs.

    IDs are sample-local integers, not barcode sequences. Missing family IDs
    denote singletons for this grouping; they are never one shared 'NA' group.
    """
    n = len(metadata)
    parents = list(range(n))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for column in ("sisters", "prer_r_group"):
        seen = {}
        for i, value in enumerate(metadata[column]):
            if pd.isna(value) or str(value).strip() in ("", "NA", "nan"):
                continue
            token = str(value).strip()
            if re.fullmatch(r"\d+\.0+", token):
                token = token.split(".")[0]
            if token in seen:
                parents[find(i)] = find(seen[token])
            else:
                seen[token] = i
    return np.array([f"{namespace}:family:{find(i)}" for i in range(n)])


def read_metadata(path, sample):
    metadata = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    metadata.columns = metadata.columns.str.strip()
    required = {"orig.ident", "nCount_RNA", "nFeature_RNA", "percent.mt", "drugSens", "sisters", "prer_r_group"}
    if required - set(metadata):
        raise ValueError(f"Missing GEO metadata fields: {required - set(metadata)}")
    metadata.index = metadata.index.astype(str).str.strip()
    if not metadata.index.is_unique or not metadata.index.str.startswith("BT_").all():
        raise ValueError("Expected unique BT_ pretreatment cell IDs")
    if not metadata["orig.ident"].str.strip().eq("BeforeTreatment").all():
        raise ValueError("Post-treatment observations must never enter this experiment")
    metadata["drugSens"] = metadata["drugSens"].fillna("").str.strip()
    unknown = set(metadata["drugSens"]) - {"", "pre-resistant", "pre-sensitive"}
    if unknown:
        raise ValueError(f"Unexpected outcome labels: {unknown}")
    metadata["group"] = lineage_groups(metadata, sample_name(sample))
    for column in ("nCount_RNA", "nFeature_RNA", "percent.mt"):
        metadata[column] = pd.to_numeric(metadata[column], errors="raise")
        if not np.isfinite(metadata[column]).all():
            raise ValueError(f"Non-finite source QC column: {column}")
    return metadata


def read_count_table(path, metadata, chunk_rows=128):
    """Stream genes x cells; never allocate the full dense expression table."""
    chunks, genes = [], []
    with gzip.open(path, "rt") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")[1:]
    header = pd.Index([s.strip().strip('"') for s in header])
    if not header.is_unique or set(header) != set(metadata.index):
        raise ValueError("Expression and metadata cell IDs are not an exact one-to-one match")
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=chunk_rows):
        if list(chunk.columns) != list(header):
            raise ValueError("Expression header changed while parsing")
        values = chunk.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("Expression must contain finite nonnegative integer UMI counts")
        if np.max(values, initial=0) > np.iinfo(np.int32).max:
            raise ValueError("Count exceeds int32 capacity")
        chunks.append(sparse.csr_matrix(values.astype(np.int32)))
        genes.extend(str(s).strip() for s in chunk.index)
    if not chunks or len(set(genes)) != len(genes):
        raise ValueError("Empty count table or duplicated original gene IDs")
    # Transpose only the sparse matrix and explicitly align metadata ordering.
    counts = sparse.vstack(chunks, format="csr").T.tocsr()[header.get_indexer(metadata.index)]
    totals = np.asarray(counts.sum(axis=1)).ravel()
    detected = np.diff(counts.indptr)
    if not np.allclose(totals, metadata["nCount_RNA"], rtol=1e-6, atol=0.01):
        raise ValueError("Count totals disagree with the released nCount_RNA metadata")
    if not np.array_equal(detected, metadata["nFeature_RNA"].to_numpy()):
        raise ValueError("Detected gene counts disagree with nFeature_RNA metadata")
    # Permit only genuine human Ensembl genes: excludes lineage/GFP constructs.
    human = np.array([bool(re.fullmatch(r"ENSG\d+(?:\.\d+)?", g)) for g in genes])
    removed = [g for g, keep in zip(genes, human) if not keep]
    human_genes = [g.split(".")[0] for g, keep in zip(genes, human) if keep]
    if not human_genes:
        raise ValueError("No human Ensembl genes in the released expression table")
    # If several versioned features resolve to one stable ID, sum their counts.
    unique = pd.Index(pd.unique(np.asarray(human_genes)))
    projection = sparse.csr_matrix(
        (np.ones(len(human_genes), dtype=np.int32), (np.arange(len(human_genes)), unique.get_indexer(human_genes))),
        shape=(len(human_genes), len(unique)),
    )
    return (counts[:, human] @ projection).tocsr(), unique, removed


def prepare(config):
    root = Path(config["output_root"])
    destination = root / "prepared"
    if (destination / "manifest.json").exists():
        existing = prepared_manifest(config)
        if existing.get("preparation_code_sha256") != sha256_file(__file__):
            raise ValueError("Preparation code changed; use a new output_root to rebuild the dataset")
        print(f"Verified existing prepared dataset: {destination}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    samples, audits, receipts = [], [], []
    for sample in SAMPLES:
        paths = {}
        for kind, (name, url) in source_files(sample).items():
            paths[kind] = obtain(root / "sources" / name, url)
            receipts.append(read_json(paths[kind].with_name(name + ".json")))
        metadata = read_metadata(paths["metadata"], sample)
        counts, genes, removed = read_count_table(paths["counts"], metadata)
        known = metadata["drugSens"].ne("").to_numpy()
        obs = metadata.loc[known].copy()
        obs["source_cell_id"] = obs.index
        obs.index = pd.Index([f"{sample_name(sample)}:{s}" for s in obs.index], name="cell_id")
        obs["condition"] = sample["condition"]
        obs["replicate"] = str(sample["replicate"])
        obs["outcome"] = obs["drugSens"].map({"pre-resistant": 1, "pre-sensitive": 0}).astype(int)
        if obs.groupby("group")["outcome"].nunique().max() > 1:
            raise ValueError("Conflicting outcome labels within a released lineage group")
        for column in ("sisters", "prer_r_group"):
            obs[column] = obs[column].fillna("").astype(str)
        sample_data = ad.AnnData(
            X=counts[known], obs=obs, var=pd.DataFrame({"feature_id": genes}, index=genes),
        )
        samples.append(sample_data)
        audits.append({
            "sample": sample_name(sample), "gsm": sample["gsm"],
            "all_qc_cells": len(metadata), "labeled_cells": len(obs),
            "unlabeled_excluded": int((~known).sum()), "recovered": int(obs["outcome"].sum()),
            "groups": obs["group"].nunique(), "human_genes": len(genes),
            "removed_construct_features": ";".join(removed),
        })
        print(f'{sample_name(sample)}: {len(obs):,} labeled cells, {len(genes):,} human genes', flush=True)
    # Harmonize the released assay panels, using feature presence only, not
    # expression variance or outcome labels. Source filtering differs by sample.
    common_genes = sorted(set.intersection(*(set(a.var_names) for a in samples)))
    if not common_genes:
        raise ValueError("Samples share no human genes")
    data = ad.concat([a[:, common_genes] for a in samples], join="inner", merge="same")
    data.X = data.X.tocsr()
    depth = np.asarray(data.X.sum(axis=1)).ravel()
    if np.any(depth <= 0):
        raise ValueError("A labeled cell has zero counts after panel harmonization")
    data.obs["panel_umis"] = depth
    data.obs["panel_genes"] = np.diff(data.X.indptr)
    for audit, sample_data in zip(audits, samples):
        panel_depth = np.asarray(sample_data[:, common_genes].X.sum(axis=1)).ravel()
        audit["common_genes"] = len(common_genes)
        audit["median_panel_umis"] = float(np.median(panel_depth))
        audit["panel_umi_fraction"] = float(panel_depth.sum() / sample_data.obs["nCount_RNA"].sum())
    temporary = destination / "counts.part.h5ad"
    data.write_h5ad(temporary, compression="gzip")
    temporary.replace(destination / "counts.h5ad")
    write_table(destination / "sample_audit.tsv", pd.DataFrame(audits))
    write_table(destination / "cells.tsv", data.obs.reset_index())
    manifest = {
        "schema_version": SCHEMA_VERSION, "counts_sha256": sha256_file(destination / "counts.h5ad"),
        "preparation_code_sha256": sha256_file(__file__),
        "n_cells": data.n_obs, "n_genes": data.n_vars,
        "cell_ids_sha256": digest(data.obs_names.tolist()),
        "gene_ids_sha256": digest(data.var_names.tolist()),
        "sources": receipts, "provenance": provenance(),
        "panel_rule": "intersection of released human Ensembl features across eight pretreatment samples",
        "group_rule": "sample-local connected components of sisters and prer_r_group; missing IDs are singletons",
        "pretraining_overlap": "not yet audited; do not claim an unseen-pretraining-study benchmark",
        "upstream_export_code": f"https://github.com/TangSoftwareLab/ReSisTrace/blob/{UPSTREAM_REVISION}/codes/data_for_GEO_submission.Rmd",
        "limitations": [
            "Only exact family IDs released in metadata are grouped; raw cross-sample barcode sequences are not available here.",
            "No recovered barcode is an uncertain negative, not proof of treatment sensitivity.",
            "Released QC/gene filtering is inherited; this is not raw FASTQ reprocessing.",
            "CNV and cell-cycle covariates are not supplied in these GEO metadata tables.",
        ],
    }
    write_json(destination / "manifest.json", manifest)
    print(f"Prepared {data.shape}: {destination / 'counts.h5ad'}", flush=True)
