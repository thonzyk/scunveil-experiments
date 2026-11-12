#!/usr/bin/env python3
"""Add full-data PCA to immutable saved paired repetitions, without encoder inference."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DATASETS = {
    "ReSisTrace": ("01_ReSisTrace", "replication_common", "replication"),
    "TumorReactiveTCells": ("02_TumorReactiveTCells", "tcr_common", "experiment"),
    "SCOPEseq2": ("03_SCOPEseq2", "scope_common", "experiment"),
    "MitochondrialPotential": ("04_MitochondrialPotential", "tmre_common", "experiment"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(directory, names=None):
    """Verify all PCA artifacts, or precisely the original artifacts being reused."""
    directory = Path(directory)
    info = read_json(directory / "complete.json")
    for name in info["files"] if names is None else names:
        if name not in info["files"] or sha256(directory / name) != info["files"][name]:
            raise ValueError(f"Corrupted completed artifact: {directory / name}")
    return info


def seal(directory, metadata):
    write_json(Path(directory) / "complete.json", {**metadata, "files": {
        str(p.relative_to(directory)): sha256(p) for p in sorted(Path(directory).rglob("*"))
        if p.is_file() and p.name != "complete.json" and not p.name.endswith(".part")}})


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed(root):
    paths = sorted((Path(root) / "replicates").glob("replicate_[0-9]*"))
    for index, path in enumerate(paths):
        if path.name != f"replicate_{index:06d}" or not (path / "complete.json").is_file():
            raise ValueError(f"Noncontiguous or uncommitted repetitions: {path}")
    return paths


class Adapter:
    """Reuse each original split, dilution, calibration, probe and scoring function.

    Each dataset runs in its own process: legacy modules share import names.
    Only the in-memory METHODS tuple is restricted; original files stay intact.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        folder, common_name, experiment_name = DATASETS[dataset]
        self.directory = HERE / folder
        previous = sys.modules.get(experiment_name)
        if previous and Path(previous.__file__).parent != self.directory:
            raise RuntimeError("Run different dataset adapters in separate Python processes")
        sys.path.insert(0, str(self.directory))
        self.common = importlib.import_module(common_name)
        self.experiment = importlib.import_module(experiment_name)
        if dataset == "ReSisTrace":
            self.thin = importlib.import_module("representations").binomial_umi_view
        else:
            self.thin = importlib.import_module("features").thin

    def check_source_code(self, manifest):
        identity = manifest["identity"]
        if self.dataset == "ReSisTrace":
            if identity["code_sha256"] != self.common.scientific_code_hash():
                raise ValueError("Original ReSisTrace scientific code changed; cannot replay safely")
        else:
            for name, expected in identity["code"].items():
                if sha256(self.directory / name) != expected:
                    raise ValueError(f"Original scientific code changed: {name}")

    def load_data(self, source, manifest):
        import anndata as ad
        if self.dataset == "ReSisTrace":
            path = Path(manifest["settings"]["output_root"]) / "prepared/counts.h5ad"
            expected = manifest["identity"]["counts_sha256"]
        else:
            prepared = read_json(source / "prepared.json")
            path = Path(prepared["directory"]) / "counts.h5ad"
            expected = manifest["identity"]["prepared_files"]["counts.h5ad"]
            if expected != prepared["receipt"]["files"]["counts.h5ad"]:
                raise ValueError("Prepared manifest and source collection disagree")
        print(f"Checking original counts: {path}", flush=True)
        if sha256(path) != expected:
            raise ValueError("Prepared integer counts changed since the source experiment")
        data = ad.read_h5ad(path)
        if self.dataset == "ReSisTrace":
            data.obs["replicate"] = data.obs.replicate.astype(str)
        elif self.dataset == "TumorReactiveTCells":
            clones = importlib.import_module("prepare").clone_table(data.obs)
            data.obs["n_cells"] = data.obs.clonotype_id.map(clones.n_cells).astype(int)
        return data, path, expected

    def partitions(self, data, settings):
        if self.dataset == "ReSisTrace":
            return self.experiment.partition_calibration(data.obs, settings)
        function = (self.experiment.region_partitions if self.dataset == "SCOPEseq2"
                    else self.experiment.patient_partitions)
        return function(data.obs, settings)

    @contextmanager
    def pca_only(self):
        from full_data_pca import PCA_METHOD
        methods = self.experiment.METHODS
        self.experiment.METHODS = (PCA_METHOD,)
        try:
            yield
        finally:
            self.experiment.METHODS = methods

    def calibrate(self, data, encoder, settings, root, partitions):
        with self.pca_only():
            if self.dataset == "ReSisTrace":
                groups, selected = self.experiment.calibrate(data, encoder, settings, root)
                import pandas as pd
                pd.testing.assert_frame_equal(groups.reset_index(drop=True), partitions.reset_index(drop=True),
                                              check_dtype=False, check_categorical=False)
                return selected
            return self.experiment.calibrate(data, encoder, settings, root, partitions)

    def collect_one(self, data, encoder, settings, root, partitions, selected, index):
        with self.pca_only():
            return self.experiment.collect_one(data, encoder, settings, root, partitions, selected, index)

    @property
    def paired_files(self):
        if self.dataset == "ReSisTrace":
            return ("lineage_splits.tsv", "cell_splits.tsv.gz", "measurement_audit.tsv")
        if self.dataset == "SCOPEseq2":
            return ("region_assignments.tsv", "cells.tsv", "measurement_audit.tsv")
        return ("patient_assignments.tsv", "selected_cells.tsv", "measurement_audit.tsv")

    @property
    def calibration_files(self):
        if self.dataset == "ReSisTrace":
            return ("lineage_partitions.tsv", "inner_splits.tsv.gz", "measurement_audit.tsv")
        return ("cells.tsv",) if self.dataset == "SCOPEseq2" else ("selected_cells.tsv",)


class PCAEncoder:
    def __init__(self, data, pca, thin):
        self.data, self.pca, self.thin = data, pca, thin
        self.dimension = pca.dimension

    def features(self, indices, rate, draw_seed):
        from full_data_pca import PCA_METHOD
        counts = self.thin(self.data.X[indices], self.data.obs_names[indices], rate, draw_seed)
        return counts, {PCA_METHOD: self.pca.transform(counts)}


def compare_frames(left, right, names):
    import pandas as pd
    for name in names:
        # Read IDs consistently, including ReSisTrace's biological replicate IDs.
        a, b = [pd.read_csv(Path(p) / name, sep="\t", dtype=str, keep_default_na=False)
                for p in (left, right)]
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as exc:
            raise ValueError(f"Pairing mismatch: {name}; refusing to publish PCA results") from exc


def resolve_source(output, explicit=None):
    if explicit:
        root = Path(explicit).resolve()
        if read_json(root / "collection.json").get("kind") == "full_data_pca_comparison":
            root = Path(read_json(root / "collection.json")["source_collection"])
        return root
    stable = Path(output) / "report"
    if stable.is_symlink():
        candidate = stable.resolve().parent.parent
        if (candidate / "collection.json").exists():
            manifest = read_json(candidate / "collection.json")
            return (Path(manifest["source_collection"]) if manifest.get("kind") == "full_data_pca_comparison"
                    else candidate)
    candidates = [p.parent for p in (Path(output) / "collections").glob("*/collection.json")
                  if read_json(p).get("kind") != "full_data_pca_comparison"]
    if len(candidates) > 1:
        raise ValueError("Multiple baseline collections and no stable report; supply --source-collection")
    return candidates[0] if candidates else None


def ensure_source(adapter, output, explicit, target, device):
    source = resolve_source(output, explicit)
    if source is not None and len(completed(source)) >= target:
        return source
    # Existing completed repetitions are normally enough: this branch is only
    # for a new installation or an explicitly higher repetition target.
    settings = read_json(source / "collection.json")["settings"] if source else adapter.common.load_settings()
    settings = {**settings, "output_root": str(output), "device": device}
    config = Path(output) / "pca_baseline_collection_config.json"
    write_json(config, settings)
    print(f"Collecting missing original-method repetitions up to TOTAL {target}", flush=True)
    subprocess.run([sys.executable, str(adapter.directory / "collect_replicates.py"),
                    "--config", str(config), "--device", device, "--max-replicates", str(target)], check=True)
    # Identify the collection produced by the unchanged original collector.
    candidates = []
    for path in (Path(output) / "collections").glob("*/collection.json"):
        manifest = read_json(path)
        if manifest.get("kind") == "full_data_pca_comparison" or len(completed(path.parent)) < target:
            continue
        a = {k: v for k, v in settings.items() if k not in ("device", "threads", "save_models", "batch_size")}
        b = {k: v for k, v in manifest["settings"].items() if k not in ("device", "threads", "save_models", "batch_size")}
        if a == b:
            candidates.append(path.parent)
    if source in candidates:
        return source
    if source is not None:
        raise RuntimeError("Collector did not extend the pinned source collection; source preserved. Check environment/code changes.")
    if len(candidates) != 1:
        raise RuntimeError("Cannot identify one completed baseline collection (collection may have been interrupted)")
    return candidates[0]


def comparison_root(source, dataset, pca_settings, threads):
    manifest = read_json(source / "collection.json")
    identity = {"protocol": "full_data_log1p_centered_PCA_augmentation_v1", "dataset": dataset,
                "source_collection": str(source.resolve()), "source_identity": manifest["identity"],
                "pca": pca_settings, "code": {name: sha256(HERE / name) for name in
                    ("full_data_pca.py", "pca_comparison.py")},
                "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "anndata", "scikit-learn")}}
    root = source.parent / f"pca_comparison-{digest(identity)[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "collection.json"
    with lock(root / "identity.lock"):
        if path.exists() and read_json(path)["identity"] != identity:
            raise ValueError("PCA collection identity mismatch")
        if not path.exists():
            write_json(path, {"kind": "full_data_pca_comparison", "identity": identity, "dataset": dataset,
                "source_collection": str(source.resolve()), "settings": manifest["settings"],
                "pca_settings": pca_settings, "threads": threads,
                "created_utc": datetime.now(timezone.utc).isoformat()})
    return root


def run_comparison(adapter, source, target, threads, pca_settings=None):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from full_data_pca import FullDataPCA, PCA_SETTINGS, progress
    pca_settings = {**PCA_SETTINGS, **(pca_settings or {})}
    manifest = read_json(source / "collection.json")
    adapter.check_source_code(manifest)
    source_dimension = read_json(source / "model.json")["embedding_dim"]
    if manifest["settings"]["model_version"] != "synthetic" and source_dimension != 2048:
        raise ValueError("Expected all 2048 scUNVEIL features in the source experiment")
    root = comparison_root(source, adapter.dataset, pca_settings, threads)
    print(f"PCA comparison: {root}\nReusing source: {source}", flush=True)
    settings = {**manifest["settings"], "threads": threads, "device": "cpu"}
    # Save only new PCA models according to the source's save_models preference.
    with lock(root / "collector.lock"), threadpool_limits(limits=threads):
        existing = completed(root)
        for path in existing:
            verify(path)
        if len(existing) >= target:
            print(f"Already have {len(existing)} paired PCA repetitions; target {target} reached", flush=True)
            return root
        data, counts_path, counts_hash = adapter.load_data(source, manifest)
        basis_dir = root / "pca_basis"
        if (basis_dir / "complete.json").exists():
            info = verify(basis_dir)
            if info["counts_sha256"] != counts_hash:
                raise ValueError("PCA cache input changed")
            pca = FullDataPCA.load(basis_dir / "basis.npz", info["pca"])
            print(f"Reusing fitted PCA basis ({pca.dimension} components)", flush=True)
        else:
            pending = root / "pending/basis"
            if pending.exists():
                shutil.rmtree(pending)
            pending.mkdir(parents=True)
            pca = FullDataPCA.fit(data.X, pca_settings)
            pca.save(pending / "basis.npz")
            write_json(pending / "pca.json", pca.info)
            seal(pending, {"counts_path": str(counts_path), "counts_sha256": counts_hash,
                           "cell_ids_sha256": digest(data.obs_names.tolist()),
                           "gene_ids_sha256": digest(data.var_names.tolist()), "pca": pca.info})
            pending.rename(basis_dir)
        encoder = PCAEncoder(data, pca, adapter.thin)
        partitions = adapter.partitions(data, settings)
        verify(source / "calibration")
        selected = adapter.calibrate(data, encoder, settings, root, partitions)
        compare_frames(source / "calibration", root / "calibration", adapter.calibration_files)
        source_paths = completed(source)
        if len(source_paths) < target:
            raise ValueError("Not enough completed source repetitions")
        # Original collectors publish into this private workspace. Nothing is
        # visible as a paired PCA repetition until all replay audits match.
        workspace = root / "replay_workspace"
        with progress(target - len(existing), f"{adapter.dataset}: remaining paired PCA repetitions") as bar:
            for index in range(len(existing), target):
                source_rep = source_paths[index]
                source_info = verify(source_rep, ["metrics.tsv", *adapter.paired_files])
                if source_info["embedding_dim"] != source_dimension:
                    raise ValueError("Source repetition/model embedding dimensions disagree")
                write_json(root / "status.json", {"state": "collecting PCA", "completed_repetitions": index,
                                                  "target": target, "current_repetition": index})
                replay = workspace / "replicates" / source_rep.name
                if replay.exists():
                    shutil.rmtree(replay)
                generated = adapter.collect_one(data, encoder, settings, workspace, partitions, selected, index)
                pca_info = verify(generated)
                compare_frames(source_rep, generated, adapter.paired_files)
                # Exact split, cell identity/order, counts hash (where saved),
                # dilution seed, library depth and empty-cell audits must match.
                for key, value in source_info.items():
                    if key.endswith("_seed") or key.endswith("_patients") or key.endswith("_regions") or key in (
                            "selected_cells_sha256", "cells_sha256"):
                        if pca_info.get(key) != value:
                            raise ValueError(f"Replay receipt mismatch: {key}")
                original = pd.read_csv(source_rep / "metrics.tsv", sep="\t",
                                       dtype={"biological_replicate": str})
                added = pd.read_csv(generated / "metrics.tsv", sep="\t",
                                    dtype={"biological_replicate": str})
                if set(added.method) != {"pca_full"} or "pca_full" in set(original.method):
                    raise ValueError("Unexpected PCA/original method set")
                # Validate every PCA metric key against the saved raw comparator.
                keys = [c for c in ("repetition", "retention_rate", "condition", "split",
                        "biological_replicate", "scope", "patient", "visit", "region", "target") if c in original]
                pd.testing.assert_frame_equal(original[original.method == "raw_log1p"][keys].reset_index(drop=True),
                                              added[keys].reset_index(drop=True), check_dtype=False)
                pd.concat([original, added], ignore_index=True).to_csv(generated / "metrics.tsv", sep="\t", index=False)
                metadata = {k: v for k, v in pca_info.items() if k != "files"}
                metadata.update(embedding_dim=source_info["embedding_dim"], pca_dimension=pca.dimension,
                    source_repetition=str(source_rep), source_receipt_sha256=sha256(source_rep / "complete.json"),
                    source_files={name: source_info["files"][name] for name in ("metrics.tsv", *adapter.paired_files)},
                    pairing_verified=True, original_metrics_reused=True,
                    predictions_scope="PCA only here; original-method predictions remain in source_repetition")
                seal(generated, metadata)
                destination = root / "replicates" / source_rep.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                generated.rename(destination)
                bar.update(1)
        write_json(root / "status.json", {"state": "target reached", "completed_repetitions": target})
    return root


def main(dataset_default=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=dataset_default, required=dataset_default is None)
    parser.add_argument("--max-replicates", type=int, default=100, help="TOTAL first paired repetitions (default 100)")
    parser.add_argument("--source-collection", type=Path, help="Pin an existing original-method collection")
    parser.add_argument("--output-root", type=Path, help="Dataset output directory; default /workspace/results/analysis/DATASET")
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu",
                        help="Only used if original-method repetitions are missing; PCA/sklearn use CPU")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report-only", action="store_true", help="Rebuild from completed PCA repetitions; no fitting")
    args = parser.parse_args()
    if args.max_replicates < 1 or args.threads < 1:
        parser.error("--max-replicates and --threads must be positive")
    # Import numerical libraries only after setting thread limits.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    output = args.output_root or Path("/workspace/results/analysis") / args.dataset
    adapter = Adapter(args.dataset)
    try:
        if args.report_only:
            source = resolve_source(output, args.source_collection)
            if source is None:
                raise ValueError("No original collection found")
            from full_data_pca import PCA_SETTINGS
            root = comparison_root(source, args.dataset, PCA_SETTINGS, args.threads)
        else:
            source = ensure_source(adapter, output, args.source_collection, args.max_replicates, args.device)
            root = run_comparison(adapter, source, args.max_replicates, args.threads)
        from pca_report import build_report
        report = build_report(root, args.max_replicates)
        print(f"Report: {report / 'REPORT.md'}\nFigure: {report / 'paper_figure'}", flush=True)
    except KeyboardInterrupt:
        print("Stopped safely. Completed repetitions are preserved; rerun the same command to resume.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
