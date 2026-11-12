"""Checkpoint checks must bind the selected model, not unrelated cached models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from checkpoint_validation import computation_code_hash, selected_checkpoint_files, validate_selected_checkpoint
from pca_comparison import read_json, sha256


def info(tmp_path):
    selected = tmp_path / "models/main"
    selected.mkdir(parents=True)
    (selected / "config.json").write_text("selected model")
    return {"model_version": "main", "checkpoint_path": str(tmp_path), "checkpoint_files": {
        "models/main/config.json": sha256(selected / "config.json"),
        "models/unrelated/config.json": "irrelevant missing model hash"}}


def test_missing_unrelated_model_is_ignored(tmp_path):
    metadata = info(tmp_path)
    files = validate_selected_checkpoint(metadata)
    assert list(files) == ["models/main/config.json"]
    actual = {**metadata, "checkpoint_files": files}
    assert selected_checkpoint_files(actual) == selected_checkpoint_files(metadata)


def test_selected_missing_or_altered_still_rejected(tmp_path):
    metadata = info(tmp_path)
    path = tmp_path / "models/main/config.json"
    path.write_text("changed selected weights")
    with pytest.raises(ValueError, match="changed"):
        validate_selected_checkpoint(metadata)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="selected-model"):
        validate_selected_checkpoint(metadata)


def test_wrong_model_metadata_is_rejected(tmp_path):
    metadata = info(tmp_path)
    metadata["model_version"] = "absent"
    with pytest.raises(ValueError, match="no files"):
        selected_checkpoint_files(metadata)


def test_legacy_unhashed_snapshot_fingerprints_only_selected_model(tmp_path):
    metadata = info(tmp_path)
    metadata.pop("checkpoint_files")
    other = tmp_path / "models/unrelated"
    other.mkdir(); (other / "config.json").write_text("another model")
    assert list(validate_selected_checkpoint(metadata)) == ["models/main/config.json"]


def test_compatibility_alias_is_exact_and_preserves_existing_computation_hashes(tmp_path):
    base = Path(__file__).resolve().parents[1]
    rules = read_json(base / "checkpoint_validation_compatibility.json")
    assert rules["validator_sha256"] == sha256(base / "checkpoint_validation.py")
    for name, rule in rules["files"].items():
        assert sha256(base / name) == rule["actual_sha256"]
        assert computation_code_hash(base / name) == rule["canonical_sha256"]
        modified = tmp_path / name
        modified.write_text((base / name).read_text() + "\n# Unreviewed future change\n")
        assert computation_code_hash(modified) == sha256(modified)
