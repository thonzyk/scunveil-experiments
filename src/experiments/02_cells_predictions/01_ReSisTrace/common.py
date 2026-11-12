"""Configuration, provenance and atomic outputs shared by the experiment."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_SOURCE = HERE.parents[4] / "02_scunveil" / "src"
UPSTREAM_REVISION = "26203470e69a311c25ab666310353391f3cc46bc"
CONDITIONS = ("Carboplatin", "Olaparib", "NK", "Control")
SAMPLES = [
    {"condition": condition, "replicate": replicate, "gsm": f"GSM{gsm}"}
    for condition, replicate, gsm in (
        ("Carboplatin", 1, 6938163), ("Carboplatin", 2, 6938165),
        ("Olaparib", 1, 6938167), ("Olaparib", 2, 6938169),
        ("NK", 1, 6938171), ("NK", 2, 6938173),
        ("Control", 1, 6938175), ("Control", 2, 6938177),
    )
]
SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
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


def read_json(path):
    return json.loads(Path(path).read_text())


def write_table(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def code_hash():
    return digest({**{p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))},
                   "retention_protocol.py": sha256_file(HERE.parent / "retention_protocol.py")})


def local_model_hash():
    if not MODEL_SOURCE.is_dir():
        raise FileNotFoundError(f"Local inference source is missing: {MODEL_SOURCE}")
    return digest({p.name: sha256_file(p) for p in sorted((MODEL_SOURCE / "scunveil").glob("*.py"))})


def provenance():
    packages = {}
    for name in ("numpy", "scipy", "pandas", "anndata", "scikit-learn", "tensorflow", "scunveil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "packages": packages,
        "git_revision": result.stdout.strip() if result.returncode == 0 else None,
        "experiment_code_sha256": code_hash(), "upstream_revision": UPSTREAM_REVISION,
    }


def load_config(path=None):
    default = read_json(HERE / "config.json")
    supplied = read_json(path) if path else {}
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = {**default, **supplied}
    if config["protocol"] not in ("retention", "legacy"):
        raise ValueError("protocol must be retention or legacy")
    rates = config["retention_rates"]
    if (not isinstance(rates, list) or not rates or
            any(type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1 for p in rates) or
            len(set(rates)) != len(rates)):
        raise ValueError("retention_rates must be unique probabilities in (0, 1]")
    if type(config["repeats"]) is not int or config["repeats"] < 1:
        raise ValueError("repeats must be a positive integer")
    for key in ("runtime_budget_minutes", "runtime_safety_factor", "runtime_reserve_seconds"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["runtime_safety_factor"] < 1:
        raise ValueError("runtime_safety_factor must be at least 1")
    if config["runtime_reserve_seconds"] >= config["runtime_budget_minutes"] * 60:
        raise ValueError("Report reserve must be smaller than the runtime budget")
    if type(config["minimum_repeats"]) is not int or config["minimum_repeats"] < 1:
        raise ValueError("minimum_repeats must be a positive integer")
    if config["protocol"] == "retention" and (
            config.get("train_cell_fraction", 1) != 1 or
            config.get("train_umi_keep_probability", 1) != 1 or
            config.get("permutation_repeats", 0) != 0):
        raise ValueError("Retention requires all training cells and matched train/test depths; use protocol=legacy for old setups")
    if not Path(config["output_root"]).is_absolute():
        raise ValueError("output_root must be an absolute path")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_name"]):
        raise ValueError("run_name must contain only letters, digits, '_' or '-'")
    for key in ("embedding_dim", "inner_folds", "max_iter", "batch_size", "threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["inner_folds"] < 2:
        raise ValueError("inner_folds must be at least two")
    for key in ("train_cell_fraction", "train_umi_keep_probability"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("bootstrap_repeats", "permutation_repeats", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not isinstance(config["c_values"], list) or not config["c_values"] or any(not math.isfinite(float(c)) or float(c) <= 0 for c in config["c_values"]):
        raise ValueError("c_values must contain positive regularization values")
    depths = config["depths"]
    if not isinstance(depths, list) or not depths or len({str(d) for d in depths}) != len(depths):
        raise ValueError("depths must be a nonempty, unique list")
    if any(d != "full" and (type(d) is not int or d <= 0) for d in depths):
        raise ValueError("depths may contain 'full' or positive integer UMI depths")
    if config["device"] not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be auto, cpu, or gpu")
    return config


def prepared_manifest(config, verify=True):
    root = Path(config["output_root"]) / "prepared"
    manifest = read_json(root / "manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Prepared data schema has changed; use a new output_root")
    if verify and sha256_file(root / "counts.h5ad") != manifest["counts_sha256"]:
        raise ValueError("Prepared counts checksum does not match its manifest")
    return manifest


def run_directory(config):
    manifest = prepared_manifest(config, verify=False)
    identity = {
        "config": {k: v for k, v in config.items() if k not in ("output_root", "device", "threads", "batch_size")},
        "counts_sha256": manifest["counts_sha256"], "code_sha256": code_hash(),
        "local_model_sha256": local_model_hash(),
    }
    identifier = digest(identity)
    root = Path(config["output_root"]) / "runs" / f'{config["run_name"]}-{identifier[:12]}'
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run.json"
    if path.exists():
        if read_json(path)["identity"] != identity:
            raise ValueError("Run identity mismatch")
    else:
        write_json(path, {"identity": identity, "config": config, "provenance": provenance()})
    return root
