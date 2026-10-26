"""External evaluation assets are optional and identified by content hashes."""
import hashlib
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rapfi_paths(cfg, root):
    return tuple(Path(root) / value if value else None
                 for value in (cfg.rapfi_executable, cfg.rapfi_config))


def rapfi_manifest(cfg, root):
    executable, config = rapfi_paths(cfg, root)
    paths = {"executable": executable, "config": config}
    paths.update({f"asset_{i}": Path(root) / value
                  for i, value in enumerate(cfg.rapfi_assets)})
    return {key: file_hash(path) if path is not None and path.is_file() else None
            for key, path in paths.items()}
