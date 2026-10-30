#!/usr/bin/env python3
"""DSRL real-recording compatibility entrypoint.

The shared real_eval/v1 HDF5 writer intentionally accepts numeric tensors only
for decision fields. DSRL JSON diagnostics also contain strings and nested
PRNG metadata. Keep the full mapping in refill_stats.jsonl under q, while
passing only HDF5-safe numeric values as diagnostics. Exact latent/PRNG arrays
remain available through the upstream trace mapping.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np


def hdf5_safe_diagnostics(values: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return numeric HDF5 fields without mutating the full JSON diagnostics."""
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in values.items():
        name = str(key)
        if value is None or isinstance(value, (str, bytes, bytearray, memoryview, Mapping)):
            dropped.append(name)
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in "OUS":
            dropped.append(name)
            continue
        safe[name] = value
    return safe, tuple(sorted(dropped))


def _load_upstream():
    raw = os.environ.get("DSRL_UPSTREAM_SERVER_PY", "")
    if not raw:
        raise RuntimeError("DSRL_UPSTREAM_SERVER_PY is required")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"DSRL upstream server missing: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("dsrl_server_upstream", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DSRL upstream server: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def main(argv: list[str] | None = None) -> int:
    upstream, path = _load_upstream()
    original_plan = upstream.DSRLPolicy.plan
    warned = False

    def recording_safe_plan(self, state, jpeg1, jpeg2):
        nonlocal warned
        out = original_plan(self, state, jpeg1, jpeg2)
        full = out.get("diagnostics") or {}
        safe, dropped = hdf5_safe_diagnostics(full)
        # q is the JSON sidecar payload and must retain strings/nested PRNG.
        out["q"] = out.get("q") or full
        out["diagnostics"] = safe
        if dropped and not warned:
            upstream.log(
                "real_eval HDF5 compatibility: kept numeric diagnostics; "
                f"JSON-only fields={list(dropped)}"
            )
            warned = True
        return out

    upstream.DSRLPolicy.plan = recording_safe_plan
    print(f"[dsrl-recording-compat] upstream={path}", flush=True)
    return int(upstream.main(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
