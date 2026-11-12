"""TCR provenance, configuration, deterministic seeds and committed artifacts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform

HERE = Path(__file__).resolve().parent
MODEL_SOURCE = HERE.parents[4] / "02_scunveil" / "src"
METHODS = ("scunveil", "raw_log1p", "depth", "clone_size")
LABELS = {"scunveil": "scUNVEIL", "raw_log1p": "Raw log1p RNA",
          "depth": "RNA summary", "clone_size": "Clonotype size"}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_array(path, values):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seal(directory, metadata):
    directory = Path(directory)
    write_json(directory / "complete.json", {**metadata, "files": {
        str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "complete.json" and not p.name.endswith(".part")}})


def verify(directory):
    directory = Path(directory)
    info = read_json(directory / "complete.json")
    for name, expected in info["files"].items():
        p = directory / name
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f"Corrupted completed artifact: {p}")
    return info


def completed(root, checksums=True):
    result = []
    for p in sorted((Path(root) / "replicates").glob("replicate_[0-9]*")):
        if not (p / "complete.json").is_file():
            raise ValueError(f"Uncommitted result in completed directory: {p}")
        info = verify(p) if checksums else read_json(p / "complete.json")
        if p.name != f"replicate_{info['repetition']:06d}":
            raise ValueError(f"Repetition ID mismatch: {p}")
        result.append((p, info))
    if [info["repetition"] for _, info in result] != list(range(len(result))):
        raise ValueError("Completed repetitions must be contiguous starting at zero")
    return result


def seed(settings, stream, repetition=None, rate=None):
    return int(digest([settings["seed"], stream, repetition, rate])[:8], 16)


def rate_key(rate):
    return f"keep_{float(rate):g}"


def load_settings(path=None):
    settings = read_json(HERE / "config.json")
    overrides = read_json(path) if path else {}
    if set(overrides) - set(settings):
        raise ValueError(f"Unknown settings: {sorted(set(overrides) - set(settings))}")
    settings.update(overrides)
    if not Path(settings["output_root"]).is_absolute():
        raise ValueError("output_root must be absolute")
    for key in ("calibration_patients", "min_calibration_clones_per_class", "test_patients",
                "dev_patients", "max_iter", "batch_size", "threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["calibration_patients"] < 2:
        raise ValueError("Need at least two independent calibration patients")
    fraction = settings["calibration_target_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration_target_fraction must lie in (0,1)")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("retention_rates", "c_values"):
        values = settings[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"Invalid {key}")
    if max(settings["retention_rates"]) > 1 or 1 not in settings["retention_rates"]:
        raise ValueError("retention_rates must include 1 and lie in (0,1]")
    if settings["device"] not in ("auto", "cpu", "gpu") or type(settings["save_models"]) is not bool:
        raise ValueError("Invalid runtime settings")
    return settings


def environment():
    return {"python": platform.python_version(), "packages": {
        name: version(name) for name in
        ("numpy", "scipy", "pandas", "anndata", "h5py", "scikit-learn", "tensorflow")}}


def collection_directory(settings, prepared):
    code = {name: sha256(HERE / name) for name in
            ("tcr_common.py", "prepare.py", "features.py", "experiment.py", "collect_replicates.py")}
    local = {p.name: sha256(p) for p in (MODEL_SOURCE / "scunveil").glob("*.py")}
    identity = {"protocol": "TCR_patient_disjoint_one_cell_molecular_retention_v1",
                "code": code, "local_inference": local, "prepared_files": prepared["files"],
                "environment": environment(),
                "settings": {k: v for k, v in settings.items() if k not in
                             ("output_root", "device", "batch_size", "threads", "save_models")}}
    root = Path(settings["output_root"]) / "collections" / f"end_to_end-{digest(identity)[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "identity.lock"):
        if (root / "collection.json").exists():
            if read_json(root / "collection.json")["identity"] != identity:
                raise ValueError("Collection identity mismatch")
        else:
            write_json(root / "collection.json", {"identity": identity, "settings": settings,
                       "created_utc": datetime.now(timezone.utc).isoformat()})
    return root
