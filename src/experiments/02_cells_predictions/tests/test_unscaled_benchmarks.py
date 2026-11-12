"""Corrected benchmark protocol and control-plane checks, offline CPU only."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import benchmark_runner as runner
from benchmark_features import FeatureStore, clear_cache
from pca_comparison import DATASETS, read_json, write_json


@pytest.mark.parametrize("dataset", DATASETS)
def test_real_protocol_with_fake_frozen_embeddings(dataset, tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("unscaled_fixture.py")), dataset, str(tmp_path)],
                             capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-12000:]


def test_content_cache_identity_and_corruption(tmp_path):
    store = FeatureStore(tmp_path)
    calls = []
    def value():
        calls.append(1)
        return np.arange(12).reshape(3, 4)
    key = {"kind": "test", "data": "sha", "rows": ["a", "b", "c"], "seed": 4}
    np.testing.assert_array_equal(store.array(key, value), value())
    n = len(calls)
    store.array(key, value)
    assert len(calls) == n
    for field, changed in (("seed", 5), ("data", "changed"), ("rows", ["c", "b", "a"])):
        store.array({**key, field: changed}, value)
    assert len(calls) == n + 3
    path = store.path(key) / "values.npy"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted"):
        FeatureStore(tmp_path).array(key, value)


def test_grid_boundary_expansion_and_ties():
    selected, scores, location = runner.search_grid(lambda p: -(np.log10(p) + 4) ** 2, [.01, .1, 1], False, 6)
    assert np.isclose(selected, .0001) and location == "interior"
    assert min(scores) < selected
    assert runner.search_grid(lambda p: .5, [.01, .1, 1], False, 6)[0] == .01
    assert runner.search_grid(lambda p: .5, [.01, .1, 1], True, 6)[0] == 1
    with pytest.raises(RuntimeError, match="boundary"):
        runner.search_grid(lambda p: np.log10(p), [.01, .1, 1], False, 1)


def test_direct_lineage_probabilities_and_independent_ridge_penalties():
    obs = pd.DataFrame({"group": ["positive", "positive", "negative"], "outcome": [1, 1, 0]})
    # Prior-correcting each cell before averaging would reverse this ranking.
    assert runner.score_calibration("ReSisTrace", obs, np.array([.01, .99, .6])) == 0
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(17)
    x = rng.normal(size=(24, 8)) * np.arange(1, 9)
    obs = pd.DataFrame({"area": rng.normal(size=24), "circ": rng.normal(size=24)})
    settings = {"targets": ["area", "circ"], "ridge_tol": 1e-8, "max_iter": 3000}
    model = runner.fit_probe("SCOPEseq2", x, obs, runner.TargetAlphas([.01, 1000]), settings)
    for i, target in enumerate(settings["targets"]):
        expected = Ridge(alpha=[.01, 1000][i], solver="lsqr", tol=1e-8, max_iter=3000).fit(x, obs[target])
        np.testing.assert_allclose(model.predict(x)[:, i], expected.predict(x), atol=1e-6)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_resistrace_lbfgs_promotes_precision_without_rescaling(monkeypatch, sparse_input):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(42)
    x = rng.normal(loc=[7, 5], scale=[1.2, .6], size=(120, 2)).astype(np.float32)
    obs = pd.DataFrame({"group": [f"g{i // 2}" for i in range(120)],
                        "outcome": np.repeat(np.arange(60) % 2, 2)})
    values = sparse.csr_matrix(x) if sparse_input else x
    original_fit = LogisticRegression.fit
    seen = []
    def checked(self, fit_x, y, **kwargs):
        assert fit_x.dtype == np.float64
        np.testing.assert_array_equal(fit_x.toarray() if sparse.issparse(fit_x) else fit_x, x)
        seen.append(fit_x.dtype)
        return original_fit(self, fit_x, y, **kwargs)
    monkeypatch.setattr(LogisticRegression, "fit", checked)
    model = runner.fit_probe("ReSisTrace", values, obs, 1e-6, {"seed": 223003, "max_iter": 3000})
    assert seen and len(model.steps) == 1 and model[-1].solver == "lbfgs"
    assert values.dtype == np.float32
    assert np.isfinite(model.predict_proba(values)).all()


def test_fresh_resume_suite_pins_four_collections(tmp_path, monkeypatch):
    for dataset in DATASETS:
        source = tmp_path / dataset / "collections/original"
        source.mkdir(parents=True)
        write_json(source / "collection.json", {"identity": {"dataset": dataset}, "settings": {}})
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    args = SimpleNamespace(analysis_root=tmp_path, results="fresh", cache="reuse", max_replicates=3, threads=1, device="cpu", suite=None)
    runner.suite_main(args)
    first = Path(read_json(tmp_path / "benchmark_suites/LATEST_UNSCALED.json")["suite"])
    assert len(calls) == 4
    original = read_json(first)
    calls.clear(); args.results = "resume"; args.max_replicates = 5
    runner.suite_main(args)
    assert len(calls) == 4
    assert [c[c.index("--collection") + 1] for c in calls] == [original["datasets"][d]["collection"] for d in DATASETS]
    assert all(c[c.index("--max-replicates") + 1] == "5" for c in calls)
    calls.clear(); args.results = "fresh"; args.cache = "rebuild"
    runner.suite_main(args)
    second = Path(read_json(tmp_path / "benchmark_suites/LATEST_UNSCALED.json")["suite"])
    assert first != second and read_json(first) == original
    assert all("--no-legacy-import" in c for c in calls)


def test_cache_delete_is_scoped_and_rejects_symlinks(tmp_path):
    keep = tmp_path / "collections/keep"
    keep.mkdir(parents=True)
    (keep / "result").write_text("keep")
    managed = tmp_path / "feature_cache/unscaled_v2"
    managed.mkdir(parents=True)
    (managed / "cache").write_text("delete")
    clear_cache(tmp_path)
    assert not managed.exists() and (keep / "result").read_text() == "keep"
    managed.symlink_to(keep, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        clear_cache(tmp_path)
