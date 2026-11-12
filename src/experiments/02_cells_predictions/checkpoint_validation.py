"""Selected-model checkpoint validation and narrowly audited compatibility aliases.

Some historical factories hashed every model in a shared HF snapshot. Only files
under models/{selected_model}/ affect this experiment. Unrelated models may be
absent; missing or altered SELECTED files remain fatal.
"""
from pathlib import Path, PurePosixPath

from pca_comparison import read_json, sha256

HERE = Path(__file__).resolve().parent


def selected_checkpoint_files(info):
    files = info.get("checkpoint_files") or {}
    if not files:
        return {}
    version = info.get("model_version")
    if not version or PurePosixPath(version).name != version:
        raise ValueError("Invalid selected checkpoint model version")
    prefix = f"models/{version}/"
    selected = {name: checksum for name, checksum in files.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError(f"Checkpoint metadata has no files for selected model {version}")
    for name in selected:
        if ".." in PurePosixPath(name).parts:
            raise ValueError("Invalid checkpoint file path")
    return selected


def validate_selected_checkpoint(info):
    selected = selected_checkpoint_files(info)
    if info.get("checkpoint_path"):
        checkpoint = Path(info["checkpoint_path"])
        if selected:
            for name, expected in selected.items():
                path = checkpoint / name
                if not path.is_file():
                    raise FileNotFoundError(f"Required selected-model checkpoint file is missing: {path}")
                if sha256(path) != expected:
                    raise ValueError(f"Selected-model checkpoint file changed: {path}")
        else:
            # Legacy ReSisTrace recorded a revision but not per-file hashes.
            # Fingerprint just its selected model, never other models in the snapshot.
            directory = checkpoint / "models" / info["model_version"]
            if not directory.is_dir():
                raise FileNotFoundError(f"Required selected-model directory is missing: {directory}")
            selected = {str(p.relative_to(checkpoint)): sha256(p)
                        for p in directory.rglob("*") if p.is_file()}
            if not selected:
                raise ValueError("Selected checkpoint directory is empty")
    return selected


def computation_code_hash(path):
    """Preserve identity only for explicitly audited, byte-exact maintenance fixes.

    Unknown future edits use their real SHA and invalidate resume/cache identities
    as before. The allowlist itself is bound to this validator's exact code hash.
    No general 'ignore code changes' switch exists.
    """
    path = Path(path)
    actual = sha256(path)
    manifest = HERE / "checkpoint_validation_compatibility.json"
    if path.name not in ("benchmark_features.py", "benchmark_runner.py") or not manifest.exists():
        return actual
    rules = read_json(manifest)
    if rules.get("validator_sha256") != sha256(Path(__file__)):
        return actual
    rule = rules.get("files", {}).get(path.name, {})
    return rule.get("canonical_sha256", actual) if rule.get("actual_sha256") == actual else actual
