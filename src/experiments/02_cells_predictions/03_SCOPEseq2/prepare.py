"""Pinned human UMI counts joined exactly to independently measured imaging features."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
import sys
from urllib.request import urlopen

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scope_common import digest, lock, seal, sha256, verify, write_frame, write_json

COUNT_NAME = "GSM4567140_PJ069.matrix.txt.gz"
IMAGE_NAME = "41598_2020_76599_MOESM3_ESM.xls"
READER_NAME = "xlrd-2.0.2-py2.py3-none-any.whl"
SOURCES = {
    COUNT_NAME: ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4567nnn/GSM4567140/suppl/" + COUNT_NAME,
        "9ab6c3ca00a7fb062f8988a8019afeb069aeefd8de57267a0af4dc408e693de5"),
    IMAGE_NAME: ("https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-020-76599-w/MediaObjects/" + IMAGE_NAME,
        "df61c4b0fa897056cdb15c5a3dfa829a39fbe26eb28cb2831179536ec375679b"),
    READER_NAME: ("https://files.pythonhosted.org/packages/1a/62/c8d562e7766786ba6587d09c5a8ba9f718ed3fa8af7f4553e8f91c36f302/" + READER_NAME,
        "ea762c3d29f4cca48d82df517b6d89fbce4db3107f9d78713e48cd321d5c9aa9")}
IMAGE_COLUMNS = ["cell_id", "c2_Area", "c2_Perim.", "c2_Width", "c2_Height", "c2_Major", "c2_Minor",
    "c2_Circ.", "c2_Round", "c2_Solidity", "c2_Feret", "c2_MinFeret", "c2_Mean", "c2_StdDev",
    "c2_Min", "c2_Max", "c2_Median"]


def obtain(name, output_root):
    url, expected = SOURCES[name]
    path = Path(output_root) / "downloads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock(path.with_name(path.name + ".lock")):
        if not path.is_file():
            temporary = path.with_name(path.name + ".part")
            print(f"Downloading {name}", flush=True)
            try:
                with urlopen(url, timeout=45) as source, temporary.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                if sha256(temporary) != expected:
                    raise ValueError(f"Source checksum mismatch: {name}; source release may have changed")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(path) != expected:
            raise ValueError(f"Cached source checksum mismatch: {path}")
    return path


def imaging_table(path, reader):
    # xlrd is pure Python. Read the pinned wheel in place; do not install or change
    # any package in the user's environment. The wheel is checked like source data.
    sys.path.insert(0, str(reader))
    import xlrd
    if xlrd.__version__ != "2.0.2":
        raise ValueError("Expected pinned xlrd 2.0.2")
    sheet = xlrd.open_workbook(str(path)).sheet_by_name("SourceDataForFigure4C")
    headers = sheet.row_values(1)
    if headers != ["Cell_barcode (region_S_Q)"] + IMAGE_COLUMNS[1:]:
        raise ValueError("Unexpected imaging source-table schema")
    frame = pd.DataFrame([sheet.row_values(i) for i in range(2, sheet.nrows)], columns=IMAGE_COLUMNS)
    if len(frame) != 956 or not frame.cell_id.is_unique or not frame.cell_id.str.fullmatch(r"[0-9]+_[0-9]+_[0-9]+").all():
        raise ValueError("Expected the 956 uniquely linked source-QC cells")
    if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite source imaging measurements")
    frame["region"] = frame.cell_id.str.split("_").str[0]
    if set(frame.region) != set(map(str, range(10))):
        raise ValueError("Expected ten technical device regions")
    return frame.set_index("cell_id").sort_index()


def validate_counts(data, targets):
    data.X = sparse.csr_matrix(data.X)
    data.X.sum_duplicates()
    data.X.eliminate_zeros()
    values = data.X.data
    if not data.n_obs or not data.n_vars or not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Nonempty counts with unique cell/gene identifiers required")
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any() or (values >= 2**63).any():
        raise ValueError("Expected nonnegative integer UMI counts")
    data.X = data.X.astype(np.int64)
    if not {"region", *targets}.issubset(data.obs) or data.obs.region.isna().any():
        raise ValueError("Missing technical regions or imaging targets")
    if not np.isfinite(data.obs[targets].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite imaging targets")
    if not {"feature_id", "gene_symbol"}.issubset(data.var):
        raise ValueError("Need matching Ensembl IDs and gene symbols")
    for region, part in data.obs.groupby("region", observed=True):
        if len(part) < 4 or (part[targets].var() <= 0).any():
            raise ValueError(f"Region {region} has too few cells or constant targets")
    return data


def load_counts(count_path, obs, targets):
    with gzip.open(count_path, "rt") as stream:
        header = next(csv.reader(stream, delimiter="\t"))
    # GEO has one trailing tab in the header but not in the data rows.
    if header[-1] == "":
        header = header[:-1]
    if header[:2] != ["gid", "gene"] or len(set(header[2:])) != len(header[2:]):
        raise ValueError("Unexpected or duplicate RNA cell barcodes")
    if len(obs.index.difference(header[2:])):
        raise ValueError("Imaging barcodes missing from RNA; never join by row order")
    matrices, genes = [], []
    # Keep every measured gene. Slice cells before converting to a numerical matrix.
    for chunk in pd.read_csv(count_path, sep="\t", header=0, usecols=range(len(header)), chunksize=512):
        counts = chunk.loc[:, obs.index].to_numpy(dtype=np.float64)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
            raise ValueError("Source contains non-integer UMI counts")
        matrices.append(sparse.csr_matrix(counts, dtype=np.int64))
        genes.append(chunk[["gid", "gene"]])
    var = pd.concat(genes, ignore_index=True).rename(columns={"gid": "feature_id", "gene": "gene_symbol"})
    var.index = pd.Index(var.feature_id, name="gene_id")
    data = validate_counts(ad.AnnData(sparse.vstack(matrices, format="csr").T.tocsr(),
                                      obs=obs.copy(), var=var), targets)
    if data.n_obs != 956 or len(header) - 2 != 1954:
        raise ValueError("Source RNA/imaging cohort size changed")
    return data


def prepare(settings):
    paths = {name: obtain(name, settings["output_root"]) for name in SOURCES}
    identity = {"sources": {name: sha256(path) for name, path in paths.items()},
        "loader_sha256": sha256(Path(__file__)), "targets": settings["targets"],
        "join": "exact region_S_Q barcode; all 956 source-table cells; no RNA-based relabelling"}
    directory = Path(settings["output_root"]) / "prepared" / digest(identity)[:12]
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory.parent / f"{directory.name}.lock"):
        if (directory / "complete.json").exists():
            return directory, verify(directory)
        obs = imaging_table(paths[IMAGE_NAME], paths[READER_NAME])
        data = load_counts(paths[COUNT_NAME], obs, settings["targets"])
        data.write_h5ad(directory / "counts.h5ad", compression="gzip")
        write_frame(directory / "imaging.tsv", obs.reset_index())
        regions = obs.groupby("region", observed=True).size().rename("n_cells").reset_index()
        write_frame(directory / "regions.tsv", regions)
        write_json(directory / "source.json", {"identity": identity,
            "study": "https://doi.org/10.1038/s41598-020-76599-w", "accession": "GSE151137 / GSM4567140 (PJ069)",
            "urls": {name: source[0] for name, source in SOURCES.items()},
            "source_rna_cells": 1954, "source_imaging_cells": 956, "matched_cells": data.n_obs,
            "target_units": "Published globally z-scored imaging features; physical areas unavailable. R2 and correlations are invariant to this fixed affine scaling.",
            "eligibility": "All and only cells in SourceDataForFigure4C, following original imaging/multiplet/viability QC; fixed before dilution.",
            "grouping": "Ten technical device regions within one anonymous human GBM patient; not biological replicates."})
        seal(directory, {"n_cells": data.n_obs, "n_genes": data.n_vars, "n_regions": 10, "identity": identity})
    return directory, verify(directory)


if __name__ == "__main__":
    from scope_common import load_settings
    directory, receipt = prepare(load_settings())
    print(f"Prepared {receipt['n_cells']} cells x {receipt['n_genes']} genes: {directory}")
