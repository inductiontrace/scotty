"""Compatibility helpers for differing NumPy versions.

This module ensures that optional submodules expected by some third-party
libraries are available even when running on older NumPy releases.  It should
be imported before any library that requires these compatibility shims.
"""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - simple import guard
    import numpy.exceptions  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - executed only on older NumPy
    import numpy as _np

    # NumPy introduced ``numpy.exceptions.AxisError`` in newer versions. Older
    # releases (such as 1.24) still expose ``AxisError`` either as an attribute
    # of the top-level ``numpy`` package or from ``numpy.core._exceptions``.
    AxisError = getattr(_np, "AxisError", None)

    if AxisError is None:  # pragma: no cover - depends on NumPy version
        try:
            from numpy.core._exceptions import AxisError  # type: ignore
        except Exception:  # pragma: no cover - safeguard for unexpected setups
            AxisError = None

    shim = types.ModuleType("numpy.exceptions")

    if AxisError is not None:
        shim.AxisError = AxisError

    sys.modules.setdefault("numpy.exceptions", shim)
