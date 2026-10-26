import pytest

from az_gomoku.assets import file_hash, rapfi_manifest, rapfi_paths
from az_gomoku import train


def test_external_assets_are_explicit_and_content_bound(tmp_path):
    cfg = train.Config()
    assert rapfi_paths(cfg, tmp_path) == (None, None)
    assert rapfi_manifest(cfg, tmp_path) == {"executable": None, "config": None}
    cfg.rapfi_executable = "engine"
    cfg.rapfi_config = "config.toml"
    cfg.rapfi_assets = ["weights"]
    for name in ("engine", "config.toml", "weights"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    original = rapfi_manifest(cfg, tmp_path)
    assert original["asset_0"] == file_hash(tmp_path / "weights")
    (tmp_path / "weights").write_bytes(b"replacement")
    assert rapfi_manifest(cfg, tmp_path) != original


def test_atomic_json_retries_transient_windows_lock(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text('{"state": "old"}', encoding="utf-8")
    replace = train.os.replace
    calls = []
    def locked_once(source, target):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("temporary reader lock")
        replace(source, target)
    monkeypatch.setattr(train.os, "replace", locked_once)
    monkeypatch.setattr(train.time, "sleep", lambda seconds: None)
    train.atomic_json(path, {"state": "new"})
    assert len(calls) == 2 and '"new"' in path.read_text()
    assert not path.with_suffix(".json.tmp").exists()


def test_atomic_json_does_not_hide_persistent_failure(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    path.write_text("old", encoding="utf-8")
    def blocked(*args):
        raise PermissionError("persistent lock")
    monkeypatch.setattr(train.os, "replace", blocked)
    monkeypatch.setattr(train.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        train.atomic_json(path, {"state": "new"})
    assert path.read_text() == "old"
