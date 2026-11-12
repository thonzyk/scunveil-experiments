"""Offline end-to-end checks with deterministic fake embeddings and real sklearn heads."""
import os
from pathlib import Path
import sys
import subprocess
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import benchmark_runner as runner
from benchmark_features import FeatureStore, clear_cache
from benchmark_report import build_report
from pca_comparison import Adapter, completed, lock, read_json, sha256, verify, write_json
from pca_fixture import fixture, SyntheticEncoder


def main():
    dataset, output = sys.argv[1], Path(sys.argv[2])
    adapter = Adapter(dataset)
    data, settings, source = fixture(dataset, output, adapter)
    partitions = adapter.partitions(data, settings)
    # One saved legacy draw audits that the new protocol preserves all cells,
    # roles, seeds, realized count matrices and empty-cell handling.
    original_encoder = SyntheticEncoder(data, adapter)
    if dataset == "ReSisTrace":
        _, old_selected = adapter.experiment.calibrate(data, original_encoder, settings, source)
    else:
        old_selected = adapter.experiment.calibrate(data, original_encoder, settings, source, partitions)
    adapter.experiment.collect_one(data, original_encoder, settings, source, partitions, old_selected, 0)
    source_before = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
    inference_calls = []

    class FakeBackend:
        def __init__(self, data, settings, root, factory=None):
            self.root = Path(root)
            self.weights = np.random.default_rng(8).normal(size=(data.n_vars, 2048)).astype(np.float32)

        def embeddings(self, counts, obs, *args):
            from full_data_pca import log_counts
            self.root.mkdir(parents=True, exist_ok=True)
            write_json(self.root / "model.json", read_json(source / "model.json"))
            inference_calls.append(tuple(obs.index))
            return np.asarray(log_counts(counts) @ self.weights), {}

    attribute = "FrozenEncoder" if dataset == "ReSisTrace" else "Encoder"
    setattr(adapter.experiment, attribute, FakeBackend)
    # Any accidental invocation of a legacy scaler is an immediate test failure.
    def forbidden(*args, **kwargs):
        raise AssertionError("A StandardScaler was invoked")
    adapter.experiment.StandardScaler = forbidden if hasattr(adapter.experiment, "StandardScaler") else None
    if dataset == "ReSisTrace":
        import evaluate
        evaluate.StandardScaler = forbidden
    fits = []
    fit = runner.fit_probe
    def checked(dataset, x, obs, parameter, settings):
        model = fit(dataset, x, obs, parameter, settings)
        assert len(model.steps) == 1 and model.steps[0][0] in ("ridge", "logisticregression")
        fits.append((len(obs), parameter))
        return model
    runner.fit_probe = checked
    kwargs = {"threads": 1, "device": "cpu", "allow_legacy": False, "adapter": adapter,
              "grids": {"c_values": [.0001, .01, 1], "alpha_values": [.001, 1, 1000]},
              "pca_settings": {"requested_components": 16}}
    root = output / "collections/unscaled-first"
    runner.run_dataset(dataset, source, root, target=1, **kwargs)
    assert len(completed(root)) == 1 and inference_calls
    info = verify(completed(root)[0])
    assert info["no_scaler"] and info["paired_source_verified"] and info["pca_dimension"] == 16
    cal = read_json(root / "calibration/selected.json")
    if dataset == "SCOPEseq2":
        assert all(len(v) == 2 for methods in cal.values() for v in methods.values())
    calibration_hash = sha256(root / "calibration/complete.json")
    rep_hash = sha256(completed(root)[0] / "complete.json")
    n_calls, n_fits = len(inference_calls), len(fits)
    runner.run_dataset(dataset, source, root, target=1, **kwargs)
    assert len(inference_calls) == n_calls and len(fits) == n_fits
    # Fresh collection: zero old models/calibration are reused, but all frozen
    # features for the same first draw are hits and trigger no inference.
    fresh = output / "collections/unscaled-fresh"
    runner.run_dataset(dataset, source, fresh, target=1, **kwargs)
    assert len(inference_calls) == n_calls and len(fits) > n_fits
    assert read_json(fresh / "calibration/selected.json") == cal
    pd.testing.assert_frame_equal(pd.read_csv(completed(root)[0] / "metrics.tsv", sep="\t"),
                                  pd.read_csv(completed(fresh)[0] / "metrics.tsv", sep="\t"))
    with lock(output / "benchmark.lock"):
        report = build_report(root, 100)
    snapshot = report.resolve()
    assert read_json(report / "report_metadata.json")["n_repetitions"] == 1
    assert "SYNTHETIC CPU FIXTURE" in (report / "REPORT.md").read_text()
    assert list((report / "paper_figure").glob("*.pdf"))
    # Same collection can grow, keeping its already calibrated C/alpha and reps.
    runner.run_dataset(dataset, source, root, target=2, **kwargs)
    assert len(completed(root)) == 2
    assert sha256(root / "calibration/complete.json") == calibration_hash
    assert sha256(completed(root)[0] / "complete.json") == rep_hash
    build_report(root, 100)
    assert read_json(snapshot / "report_metadata.json")["n_repetitions"] == 1
    assert read_json(report / "report_metadata.json")["n_repetitions"] == 2
    subprocess.run([sys.executable, str(adapter.directory / "build_report.py"), "--collection", str(root),
                    "--max-repetitions", "2"], check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(adapter.directory / "plot_paper_figure.py"), "--report", str(report),
                    "--output", str(output / "separate_figure")], check=True, capture_output=True, text=True)
    assert source_before == {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
    # Interrupted repetition cannot enter reports. Cache rebuild leaves all
    # historical result artifacts intact and forces actual feature recomputation.
    original_collect = adapter.experiment.collect_one
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    adapter.experiment.collect_one = interrupted
    try:
        runner.run_dataset(dataset, source, root, target=3, **kwargs)
        raise AssertionError("Expected interruption")
    except KeyboardInterrupt:
        pass
    assert len(completed(root)) == 2
    adapter.experiment.collect_one = original_collect
    cold = output / "collections/unscaled-cold"
    n_calls = len(inference_calls)
    runner.run_dataset(dataset, source, cold, target=1, cache_mode="rebuild", **kwargs)
    assert len(inference_calls) > n_calls
    pd.testing.assert_frame_equal(pd.read_csv(completed(root)[0] / "metrics.tsv", sep="\t"),
                                  pd.read_csv(completed(cold)[0] / "metrics.tsv", sep="\t"))
    assert sha256(completed(root)[0] / "complete.json") == rep_hash
    assert read_json(cold / "cache_policy.json")["allow_legacy_import"] is False
    # Completed reports remain readable with the feature cache entirely absent.
    clear_cache(output)
    build_report(root, 100)
    assert "tensorflow" not in sys.modules
    print(f"PASS {dataset}: no scalers, paired draws, independent fresh fits, safe resume, cache rebuild, report isolation")


if __name__ == "__main__":
    main()
