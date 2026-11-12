"""Public original CellRanger UMIs; audited lung subject/visit/TMRE library mapping."""
from __future__ import annotations

from pathlib import Path
import gzip
import re
import time

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.io import mmread
from tqdm.auto import tqdm

from tmre_common import HERE, digest, lock, seal, sha256, verify, write_frame, write_json

SOFT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE315nnn/GSE315510/soft/GSE315510_family.soft.gz"
LIST_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE315nnn/GSE315510/suppl/filelist.txt"


def obtain(url, output_root, expected_size=None):
    path = Path(output_root) / "downloads" / url.rsplit("/", 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock(path.with_name(path.name + ".lock")):
        receipt = path.with_name(path.name + ".sha256")
        if path.is_file():
            if expected_size is not None and path.stat().st_size != expected_size:
                raise ValueError(f"Cached source size mismatch: {path}")
            if receipt.exists() and receipt.read_text().strip() != sha256(path):
                raise ValueError(f"Cached source checksum mismatch: {path}")
            return path
        partial = path.with_name(path.name + ".part")
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size is not None and offset == expected_size:
                partial.replace(path)
                receipt.write_text(sha256(path) + "\n")
                return path
            try:
                with requests.get(url, headers={"Range": f"bytes={offset}-"}, stream=True,
                                  timeout=(20, 45)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        header = response.headers.get("Content-Range", "")
                        if not header.startswith(f"bytes {offset}-"):
                            raise ValueError("Unexpected source range response")
                    elif offset:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as target, tqdm(
                            total=expected_size, initial=offset, unit="B", unit_scale=True,
                            desc=path.name, smoothing=0, mininterval=1) as bar:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                target.write(block)
                                bar.update(len(block))
                if expected_size is not None and partial.stat().st_size != expected_size:
                    raise ValueError(f"Downloaded source size mismatch: {path}")
                partial.replace(path)
                receipt.write_text(sha256(path) + "\n")
                return path
            except (requests.RequestException, OSError):
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
    raise RuntimeError("Download failed")


def validate_manifest(manifest, soft_text):
    """Reject source drift; subject IDs use title distinctions, never adjacency alone."""
    records = {}
    for block in soft_text.split("^SAMPLE = ")[1:]:
        lines = block.splitlines()
        records[lines[0].strip()] = lines
    if len(manifest) != 20 or manifest.accession.duplicated().any():
        raise ValueError("Expected 20 unique lung libraries")
    for row in manifest.itertuples():
        lines = records.get(row.accession, [])
        titles = [s.split(" = ", 1)[1] for s in lines if s.startswith("!Sample_title = ")]
        if titles != [row.title] or not row.title.startswith("Lung"):
            raise ValueError(f"GEO sample title changed: {row.accession}")
        title = row.title.lower()
        if ("tmre high" in title) != bool(row.outcome) or ("post-treatment" in title) != (row.visit == "post"):
            raise ValueError("Incorrect phenotype/visit mapping")
        response = "PR" if "partial disease" in title else "SD" if "stable disease" in title else "PD"
        match = re.search(r"patient (\d+)", title)
        if not match and response != "PD":
            raise ValueError("Missing disambiguating subject suffix")
        expected = f"LUNG_{response}{match.group(1) if match else '1'}"
        if row.subject != expected:
            raise ValueError("Subject identity disagrees with title")
        urls = {s.split(" = ", 1)[1].replace("ftp://", "https://") for s in lines
                if s.startswith("!Sample_supplementary_file_")}
        if {row.matrix, row.features, row.barcodes} != urls:
            raise ValueError("Processed source links changed")
    groups = manifest.groupby("subject", observed=True)
    expected = {(v, y) for v in ("pre", "post") for y in (0, 1)}
    if len(groups) != 5 or any(set(zip(g.visit, g.outcome)) != expected or len(g) != 4 for _, g in groups):
        raise ValueError("Each of five subjects needs both gates at both visits")


def validate_counts(data):
    data.X = sparse.csr_matrix(data.X)
    data.X.sum_duplicates()
    data.X.eliminate_zeros()
    x = data.X.data
    if not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError("Expected nonnegative integer UMI counts")
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Unique cells and gene features required")
    required = ["patient", "visit", "library", "outcome"]
    if not set(required).issubset(data.obs) or data.obs[required].isna().any().any():
        raise ValueError("Missing subject/visit/library/outcome metadata")
    if set(data.obs.outcome) != {0, 1} or set(data.obs.visit) != {"pre", "post"}:
        raise ValueError("Both TMRE outcomes and visits required")
    for _, g in data.obs.groupby("library", observed=True):
        if any(g[k].nunique() != 1 for k in ("patient", "visit", "outcome")):
            raise ValueError("Source library has inconsistent assignment")
    for _, g in data.obs.groupby(["patient", "visit"], observed=True):
        if set(g.outcome) != {0, 1}:
            raise ValueError("Each subject/visit must retain both measured gates")
    if "gene_symbol" not in data.var or not data.var.gene_symbol.str.upper().str.startswith("MT-").any():
        raise ValueError("Gene symbols with mitochondrial features required")
    data.X = data.X.astype(np.int64)
    return data


def load_counts(manifest, paths, settings):
    matrices, observations, audits = [], [], []
    reference = None
    for row in manifest.itertuples():
        genes = pd.read_csv(paths[row.features], sep="\t", header=None, compression="gzip", dtype=str)
        barcodes = pd.read_csv(paths[row.barcodes], sep="\t", header=None, compression="gzip", dtype=str)[0]
        if genes.shape[1] != 3 or set(genes[2]) != {"Gene Expression"}:
            raise ValueError("Expected only RNA Gene Expression features")
        if reference is None:
            reference = genes
        elif not genes.equals(reference):
            raise ValueError("Source libraries use different gene panels")
        if not genes[0].is_unique or not barcodes.is_unique:
            raise ValueError("Duplicate source gene or barcode identifiers")
        with gzip.open(paths[row.matrix], "rb") as stream:
            x = mmread(stream).T.tocsr()
        if x.shape != (len(barcodes), len(genes)):
            raise ValueError("MTX/barcode/feature dimensions disagree")
        if not np.isfinite(x.data).all() or (x.data < 0).any() or (x.data != np.floor(x.data)).any():
            raise ValueError("Expected original nonnegative integer UMI counts")
        x.sum_duplicates(); x.eliminate_zeros()
        total = np.asarray(x.sum(axis=1)).ravel()
        detected = np.diff(x.indptr)
        mt = genes[1].str.upper().str.startswith("MT-").to_numpy()
        fraction = np.divide(np.asarray(x[:, mt].sum(axis=1)).ravel(), total,
                             out=np.zeros(len(total), float), where=total > 0)
        keep = ((total >= settings["min_umis"]) & (total <= settings["max_umis"]) &
                (detected >= settings["min_genes"]) & (detected <= settings["max_genes"]) &
                (fraction <= settings["max_mito_fraction"]))
        audits.append({"accession": row.accession, "patient": row.subject, "visit": row.visit,
            "outcome": row.outcome, "n_source_cells": len(total), "n_eligible_cells": int(keep.sum()),
            "median_original_umis": float(np.median(total[keep])) if keep.any() else np.nan})
        if not keep.any():
            raise ValueError(f"No QC-eligible cells in {row.accession}; cannot silently drop a gate")
        matrices.append(x[keep])
        observations.append(pd.DataFrame({"patient": row.subject, "visit": row.visit,
            "library": row.accession, "outcome": row.outcome, "source_barcode": barcodes[keep].to_numpy(),
            "original_umis": total[keep], "original_detected_genes": detected[keep],
            "original_mito_fraction": fraction[keep]},
            index=pd.Index([f"{row.accession}:{b}" for b in barcodes[keep]], name="cell_id")))
    var = pd.DataFrame({"feature_id": reference[0].to_numpy(), "gene_symbol": reference[1].to_numpy()},
                       index=pd.Index(reference[0], name="feature_id_index"))
    data = validate_counts(ad.AnnData(sparse.vstack(matrices, format="csr"),
                                     obs=pd.concat(observations), var=var))
    return data, pd.DataFrame(audits)


def prepare(settings):
    manifest = pd.read_csv(HERE / "source_manifest.tsv", sep="\t")
    soft = obtain(SOFT_URL, settings["output_root"])
    listing = obtain(LIST_URL, settings["output_root"])
    with gzip.open(soft, "rt") as stream:
        validate_manifest(manifest, stream.read())
    sizes = pd.read_csv(listing, sep="\t").set_index("Name")["Size"].to_dict()
    paths = {}
    for row in manifest.itertuples():
        for url in (row.features, row.barcodes, row.matrix):
            name = url.rsplit("/", 1)[1]
            if name not in sizes:
                raise ValueError(f"Processed source absent from GEO listing: {name}")
            paths[url] = obtain(url, settings["output_root"], int(sizes[name]))
    identity = {"source_checksums": {u: sha256(p) for u, p in paths.items()},
                "soft_sha256": sha256(soft), "filelist_sha256": sha256(listing),
                "manifest_sha256": sha256(HERE / "source_manifest.tsv"),
                "loader_sha256": sha256(Path(__file__)),
                "qc": {k: settings[k] for k in ("min_umis", "max_umis", "min_genes", "max_genes", "max_mito_fraction")}}
    directory = Path(settings["output_root"]) / "prepared" / digest(identity)[:12]
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory.parent / f"{directory.name}.lock"):
        if (directory / "complete.json").exists():
            return directory, verify(directory)
        data, audit = load_counts(manifest, paths, settings)
        data.write_h5ad(directory / "counts.h5ad", compression="gzip")
        write_frame(directory / "library_audit.tsv", audit)
        write_frame(directory / "source_manifest.tsv", manifest)
        write_json(directory / "source.json", {"identity": identity, "accession": "GSE315510",
            "study": "https://doi.org/10.1093/jimmun/vkag120",
            "subject_ids": "Cohort-local keys from response category + Patient suffix in GEO sample titles",
            "selection": "20 original lung libraries; fixed RNA QC before dilution; all RNA genes retained",
            "not_reproduced": "No CellBender, Scrublet, TCR filtering, source author annotations or Harmony"})
        seal(directory, {"identity": identity, "n_cells": data.n_obs, "n_genes": data.n_vars,
                         "n_patients": data.obs.patient.nunique(), "n_libraries": data.obs.library.nunique()})
    return directory, verify(directory)


if __name__ == "__main__":
    from tmre_common import load_settings
    directory, _ = prepare(load_settings())
    print(f"Prepared: {directory}")
