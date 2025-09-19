from __future__ import annotations

import importlib
from typing import Any


def load_object(dotted: str) -> Any:
    """Load ``pkg.mod:ClassName`` or ``pkg.mod.ClassName`` objects dynamically."""

    if ":" in dotted:
        mod_name, obj_name = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        mod_name, obj_name = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(mod_name)
    return getattr(module, obj_name)
