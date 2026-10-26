"""Load the separately built native tree search; no compiler is invoked on import."""
import importlib.util
from pathlib import Path


def native():
    path = Path(__file__).resolve().parents[2] / "build" / "az_gomoku"
    files = list(path.glob("az_search*.pyd")) + list(path.glob("az_search*.so"))
    if len(files) != 1:
        raise RuntimeError("Build the native search with scripts/build_az_search.py first")
    spec = importlib.util.spec_from_file_location("az_search", files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
