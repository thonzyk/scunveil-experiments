"""Configuration and immutable, atomic artifacts for continuous replication."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import re

from common import (HERE, digest, local_model_hash, prepared_manifest, provenance,
                    read_json, sha256_file, write_json)


def load_settings(path=None):
    settings = read_json(HERE / "replication_config.json")
    overrides = read_json(path) if path else {}
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError(f"Unknown replication settings: {sorted(unknown)}")
    settings.update(overrides)
    if not Path(settings["output_root"]).is_absolute() or not re.fullmatch(r"[\w-]+", settings["run_name"]):
        raise ValueError("Use an absolute output_root and an alphanumeric run_name")
    for key in ("max_iter", "batch_size", "threads", "calibration_folds"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["calibration_folds"] < 2:
        raise ValueError("calibration_folds must be at least two")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("calibration_fraction", "train_fraction", "dev_fraction", "test_fraction"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{key} must be in (0, 1)")
    if not math.isclose(sum(settings[k] for k in ("train_fraction", "dev_fraction", "test_fraction")), 1):
        raise ValueError("train/dev/test fractions must sum to one")
    rates = settings["retention_rates"]
    if (not isinstance(rates, list) or not rates or
            any(type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1 for p in rates) or
            len(set(rates)) != len(rates)):
        raise ValueError("retention_rates must contain unique probabilities in (0, 1]")
    values = settings["c_values"]
    if (not isinstance(values, list) or not values or
            any(type(c) not in (int, float) or not math.isfinite(c) or c <= 0 for c in values) or
            len(set(values)) != len(values)):
        raise ValueError("c_values must contain unique positive values")
    if settings["device"] not in ("auto", "cpu", "gpu") or type(settings["save_models"]) is not bool:
        raise ValueError("Invalid device or save_models")
    return settings


def random_seed(settings, stream, replicate=None, rate=None):
    return int(digest({"seed": settings["seed"], "stream": stream,
                       "replicate": replicate, "retention_rate": rate})[:8], 16)


def rate_key(rate):
    return f"keep_{float(rate)}"


def scientific_code_hash():
    # Report formatting can change without invalidating collected experiments.
    names = ("replication.py", "replication_common.py", "collect_replicates.py",
             "common.py", "evaluate.py", "representations.py", "prepare.py")
    return digest({name: sha256_file(HERE / name) for name in names})


def collection_directory(settings):
    prepared = prepared_manifest(settings)
    environment = provenance()
    identity = {"protocol": "repeated_grouped_train_dev_test_v1",
                "settings": {k: v for k, v in settings.items() if k not in
                             ("output_root", "device", "threads", "batch_size", "save_models")},
                "embedding_policy": "all native pretrained features; no dimensional truncation",
                "counts_sha256": prepared["counts_sha256"], "code_sha256": scientific_code_hash(),
                "local_model_sha256": local_model_hash(),
                "python": environment["python"], "packages": environment["packages"]}
    root = Path(settings["output_root"]) / "collections" / f"{settings['run_name']}-{digest(identity)[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "collection.json"
    with exclusive_lock(root / "identity.lock"):
        if manifest.exists():
            if read_json(manifest)["identity"] != identity:
                raise ValueError("Collection identity mismatch")
        else:
            write_json(manifest, {"identity": identity, "settings": settings, "provenance": environment})
    return root


@contextmanager
def exclusive_lock(path):
    """OS lock releases even on SIGKILL; report readers need no collector lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, sep="\t", index=False,
                 compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_array(path, array):
    import numpy as np
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def seal(directory, metadata):
    files = {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob("*")
             if p.is_file() and p.name != "complete.json" and not p.name.endswith(".part")}
    write_json(directory / "complete.json", {**metadata, "files": files})


def verify(directory):
    info = read_json(directory / "complete.json")
    for name, expected in info["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Corrupted completed artifact: {directory / name}")
    return info


def completed_replicates(root, verify_checksums=True):
    result = []
    for path in sorted((Path(root) / "replicates").glob("replicate_[0-9]*")):
        # An atomic directory rename publishes a complete repetition.
        if not (path / "complete.json").is_file():
            raise ValueError(f"Uncommitted directory inside completed replicates: {path}")
        info = verify(path) if verify_checksums else read_json(path / "complete.json")
        if path.name != f"replicate_{info['replicate']:06d}":
            raise ValueError(f"Replication ID mismatch: {path}")
        result.append((path, info))
    if [r[1]["replicate"] for r in result] != list(range(len(result))):
        raise ValueError("Completed repetitions must form a contiguous sequence starting at zero")
    return result
