"""One tiny synthetic original collection per subprocess; never read real dataset counts."""
import os
from pathlib import Path
import sys
import subprocess
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from full_data_pca import log_counts
from pca_comparison import (Adapter, completed, lock, read_json, run_comparison,
                            seal, sha256, verify, write_json)
from pca_report import build_report, primary_scores


def fixture(dataset, output, adapter):
    settings = {**adapter.common.load_settings(), "output_root": str(output), "device": "cpu",
                "model_version": "synthetic", "threads": 1, "save_models": False, "max_iter": 3000}
    settings.update({"alpha_values": [1, 10]} if dataset == "SCOPEseq2" else {"c_values": [.01, .1]})
    rng = np.random.default_rng(92)
    rows = []
    if dataset == "ReSisTrace":
        settings["calibration_folds"] = 2
        for condition in adapter.experiment.CONDITIONS:
            for bio in ("1", "2"):
                for outcome in (0, 1):
                    for clone in range(12):
                        group = f"{condition}:{bio}:{outcome}:{clone}"
                        for _ in range(2):
                            rows.append({"condition": condition, "replicate": bio, "outcome": outcome, "group": group})
    elif dataset == "TumorReactiveTCells":
        for patient in range(9):
            for clone in range(6):
                for _ in range(2 + clone % 3):
                    rows.append({"patient": f"P{patient}", "clonotype_id": f"P{patient}C{clone}",
                                 "outcome": int(clone >= 3), "library": f"L{patient}", "n_cells": 2 + clone % 3})
    elif dataset == "MitochondrialPotential":
        settings["cells_per_gate_visit"] = 4
        for patient in range(5):
            for visit in ("pre", "post"):
                for outcome in (0, 1):
                    for _ in range(8):
                        rows.append({"patient": f"P{patient}", "visit": visit, "outcome": outcome,
                                     "library": f"P{patient}_{visit}_{outcome}"})
    else:
        rows = [{"region": str(region)} for region in range(10) for _ in range(8)]
    values = rng.poisson(3, (len(rows), 24))
    obs = pd.DataFrame(rows, index=pd.Index([f"cell{i}" for i in range(len(rows))], name="cell_id"))
    if dataset == "SCOPEseq2":
        obs["c2_Area"] = values[:, 0] * .3 + rng.normal(size=len(rows))
        obs["c2_Circ."] = values[:, 1] * -.3 + rng.normal(size=len(rows))
    else:
        values[:, 0] += obs.outcome.to_numpy() * 5
    var = pd.DataFrame({"gene_symbol": ["MT-FAKE"] + [f"GENE{i}" for i in range(1, 24)]},
                       index=[f"ENSG{i:011d}" for i in range(24)])
    data = ad.AnnData(sparse.csr_matrix(values), obs=obs, var=var)
    prepared = output / "prepared"
    prepared.mkdir(parents=True)
    data.write_h5ad(prepared / "counts.h5ad")
    seal(prepared, {"synthetic": True})
    source = output / "collections/end_to_end-fixture"
    source.mkdir(parents=True)
    if dataset == "ReSisTrace":
        identity = {"counts_sha256": sha256(prepared / "counts.h5ad"),
                    "code_sha256": adapter.common.scientific_code_hash()}
    else:
        identity = {"code": {name: sha256(adapter.directory / name) for name in
                    ("experiment.py", "features.py", "prepare.py", "collect_replicates.py")},
                    "prepared_files": read_json(prepared / "complete.json")["files"]}
        write_json(source / "prepared.json", {"directory": str(prepared), "receipt": verify(prepared)})
    write_json(source / "collection.json", {"settings": settings, "identity": identity})
    write_json(source / "model.json", {"embedding_dim": 2048, "model_version": "synthetic"})
    return data, settings, source


class SyntheticEncoder:
    """Mock pretraining solely to generate a source collection; sklearn is real."""
    dimension = 2048
    def __init__(self, data, adapter):
        self.data, self.adapter = data, adapter
        self.weights = np.random.default_rng(8).normal(size=(data.n_vars, self.dimension)).astype(np.float32)

    def features(self, indices, rate, draw):
        obs = self.data.obs.iloc[indices]
        counts = self.adapter.thin(self.data.X[indices], obs.index, rate, draw)
        raw = log_counts(counts)
        total = np.asarray(counts.sum(axis=1)).ravel()
        depth = np.column_stack([np.log1p(total), np.log1p(np.diff(counts.indptr))]).astype(np.float32)
        values = {"scunveil": np.asarray(raw @ self.weights), "raw_log1p": raw,
                  "depth": depth, "depth_qc": depth, "depth_mito": np.column_stack([depth, total > 0])}
        if "n_cells" in obs:
            values["clone_size"] = np.log1p(obs.n_cells.to_numpy())[:, None]
        return counts, {name: values[name] for name in self.adapter.experiment.METHODS}


def main():
    dataset, output = sys.argv[1], Path(sys.argv[2])
    adapter = Adapter(dataset)
    data, settings, source = fixture(dataset, output, adapter)
    encoder = SyntheticEncoder(data, adapter)
    partitions = adapter.partitions(data, settings)
    if dataset == "ReSisTrace":
        _, selected = adapter.experiment.calibrate(data, encoder, settings, source)
    else:
        selected = adapter.experiment.calibrate(data, encoder, settings, source, partitions)
    for index in range(3):
        adapter.experiment.collect_one(data, encoder, settings, source, partitions, selected, index)
    before = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
    root = run_comparison(adapter, source, 2, 1, {"requested_components": 16})
    assert len(completed(root)) == 2
    for path in completed(root):
        info = verify(path)
        assert info["embedding_dim"] == 2048 and info["pca_dimension"] == 16 and info["pairing_verified"]
        original = pd.read_csv(Path(info["source_repetition"]) / "metrics.tsv", sep="\t", dtype={"biological_replicate": str})
        actual = pd.read_csv(path / "metrics.tsv", sep="\t", dtype={"biological_replicate": str})
        pd.testing.assert_frame_equal(original, actual[actual.method != "pca_full"].reset_index(drop=True))
    # Verify report construction while the PCA collector lock is held.
    with lock(root / "collector.lock"):
        report = build_report(root, 2)
    old_report = report.resolve()
    assert report.is_symlink() and read_json(report / "report_metadata.json")["n_repetitions"] == 2
    assert list((report / "paper_figure").glob("*.pdf"))
    assert "SYNTHETIC CPU FIXTURE" in (report / "REPORT.md").read_text()
    summary = pd.read_csv(report / "summary.tsv", sep="\t")
    assert summary.sd.notna().all()
    primary = pd.read_csv(report / "primary_scores.tsv", sep="\t")
    keyed = primary.pivot(index=["repetition", "retention_rate", "task"], columns="method", values="score")
    expected_delta = (keyed.scunveil - keyed.pca_full).sort_index()
    delta = pd.read_csv(report / "paired_differences.tsv", sep="\t")
    actual_delta = delta[delta.baseline == "pca_full"].set_index(
        ["repetition", "retention_rate", "task"]).difference.sort_index()
    np.testing.assert_allclose(actual_delta, expected_delta, atol=1e-15)
    # Existing public report and plot commands must also accept the new format.
    subprocess.run([sys.executable, str(adapter.directory / "build_report.py"), "--collection", str(root),
                    "--max-repetitions", "2"], check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(adapter.directory / "plot_paper_figure.py"), "--report", str(report),
                    "--output", str(output / "separate_figure")], check=True, capture_output=True, text=True)
    assert list((output / "separate_figure").glob("*.pdf"))
    cal_hash, basis_hash = [sha256(root / sub / "complete.json") for sub in ("calibration", "pca_basis")]
    first_hash = sha256(completed(root)[0] / "complete.json")
    # Stop during a new repetition. No incomplete paired result is published.
    collect_one = adapter.collect_one
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    adapter.collect_one = interrupted
    try:
        run_comparison(adapter, source, 3, 1, {"requested_components": 16})
        raise AssertionError("Expected interruption")
    except KeyboardInterrupt:
        pass
    assert len(completed(root)) == 2
    adapter.collect_one = collect_one
    run_comparison(adapter, source, 3, 1, {"requested_components": 16})
    assert len(completed(root)) == 3
    assert cal_hash == sha256(root / "calibration/complete.json")
    assert basis_hash == sha256(root / "pca_basis/complete.json")
    assert first_hash == sha256(completed(root)[0] / "complete.json")
    build_report(root, 3)
    assert read_json(old_report / "report_metadata.json")["n_repetitions"] == 2
    assert read_json(report / "report_metadata.json")["n_repetitions"] == 3
    assert before == {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
    # Corrupt a reused source metric. The previous stable report stays published.
    original = source / "replicates/replicate_000000/metrics.tsv"
    original.write_text("corrupted")
    previous = report.resolve()
    try:
        build_report(root, 3)
        raise AssertionError("Expected corruption rejection")
    except ValueError as exc:
        assert "Corrupted" in str(exc)
    assert report.resolve() == previous
    assert "tensorflow" not in sys.modules, "PCA augmentation must not initialize the pretrained encoder"
    print(f"PASS {dataset}: paired replay, native sklearn fits, resume, integrity and atomic report")


if __name__ == "__main__":
    main()
