"""Load unmodified kernels preserved from the baseline commit."""

import importlib.util
import sys
import tarfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=None)
def original(module: str) -> ModuleType:
    """Load a preserved module without replacing the installed engine."""
    root = Path(__file__).parent
    destination = root / ".reference"
    relative = "polars_ti/" + module.replace(".", "/") + ".py"
    path = destination / relative
    if not path.exists():
        with tarfile.open(root / "baseline.tar.gz") as archive:
            archive.extract(relative, destination, filter="data")
    name = "rolling_reference_" + module.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(module)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded
