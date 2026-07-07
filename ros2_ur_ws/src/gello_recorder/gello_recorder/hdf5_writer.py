#!/usr/bin/env python3
"""Growable HDF5 table writer -- a drop-in replacement for the CSV pattern used by
the GELLO/UR diagnostic recorder.

Each logical "table" becomes an HDF5 group; each column becomes its own resizable
1-D float64 dataset that is appended to one row at a time. This mirrors the old
`_open_csv(name, header)` + `writer.writerow([...])` usage so the recorder node can
swap CSV files for a single shared HDF5 file with a near 1:1 diff.

Standalone by design: depends ONLY on h5py + numpy (no rclpy / ROS), so it can be
unit-tested outside a ROS environment. Run `python3 hdf5_writer.py` for a self-test.
"""

import json

import h5py
import numpy as np


def _coerce(value) -> float:
    """Coerce a single cell to float64, mapping None/empty/non-numeric -> NaN.

    Tolerates the exact value types the recorder passes into writerow():
      * Python float / int              -> float(value)
      * numeric string like "1.2345"    -> float(value)   (call sites use f"{v:.4f}")
      * None                            -> np.nan
      * already-NaN float               -> np.nan (preserved)
      * empty string / unparseable str  -> np.nan (defensive, never raises)
    """
    if value is None:
        return np.nan
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return np.nan
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


class Hdf5TableWriter:
    """Growable HDF5 table: one resizable 1-D float64 dataset per column, appended row-by-row."""

    def __init__(self, h5file: h5py.File, table_name: str, header: list):
        """Create (or truncate) a group named `table_name` under `h5file`, with one
        resizable float64 dataset per column in `header`.

        Datasets are created with shape=(0,), maxshape=(None,), chunks=True (h5py auto
        picks a chunk shape) so appending row-by-row over a long session is cheap. The
        header is stored as a JSON-encoded group attribute (`attrs['columns']`) so the
        original column order is recoverable regardless of HDF5 key ordering.
        """
        self._header = list(header)

        # Truncate any pre-existing group of the same name so re-opening a table
        # starts clean (matching CSV open mode "w").
        if table_name in h5file:
            del h5file[table_name]
        self._group = h5file.create_group(table_name)
        self._group.attrs["columns"] = json.dumps(self._header)

        # One resizable float64 dataset per column. Dataset name == column name.
        # A parallel list keeps positional (column-index) access fast in writerow.
        self._datasets = []
        for col in self._header:
            dset = self._group.create_dataset(
                col,
                shape=(0,),
                maxshape=(None,),
                dtype="float64",
                chunks=True,
            )
            self._datasets.append(dset)

        self._nrows = 0

    def writerow(self, row: list) -> None:
        """Append one row (length must equal len(header)).

        Each value may be float / int / numeric-string / None; None (and any
        unparseable value) becomes np.nan. Each column dataset is resized by +1 and
        its new last element is assigned -- an O(1) amortised append, not a full copy.
        """
        if len(row) != len(self._datasets):
            raise ValueError(
                f"row has {len(row)} values but table has "
                f"{len(self._datasets)} columns ({self._header})"
            )

        new_len = self._nrows + 1
        for dset, value in zip(self._datasets, row):
            dset.resize((new_len,))
            dset[self._nrows] = _coerce(value)
        self._nrows = new_len

    def flush(self) -> None:
        """Flush the underlying file to disk (defensive; caller may also flush the
        shared h5py.File separately)."""
        self._group.file.flush()


def open_h5_table(h5file: h5py.File, table_name: str, header: list) -> Hdf5TableWriter:
    """Convenience constructor mirroring the old `_open_csv(name, header)` call site,
    so integrating into the recorder node is a near 1:1 swap."""
    return Hdf5TableWriter(h5file, table_name, header)


if __name__ == "__main__":
    import os
    import tempfile

    scratch = tempfile.mkdtemp(prefix="hdf5_writer_selftest_")
    path = os.path.join(scratch, "selftest.h5")
    print(f"self-test scratch dir : {scratch}")
    print(f"self-test hdf5 file    : {path}")

    # Two tables with different headers, mirroring recorder usage: some rows carry raw
    # floats, some carry f-string-formatted numbers, some carry None.
    header_a = ["t_rel_s", "q1", "q2", "qd1"]
    header_b = ["t_rel_s", "fx", "fy", "fz"]

    # Reference values we expect to read back (post-coercion), NaN where None was given.
    expected_a = []
    expected_b = []

    with h5py.File(path, "w") as f:
        wa = open_h5_table(f, "table_a", header_a)
        wb = open_h5_table(f, "table_b", header_b)

        for i in range(5):
            t = i * 0.01
            # Mix raw floats, formatted strings, ints and None -- exactly the shapes
            # the recorder passes (e.g. pos raw float, qd as f"{v:.5f}" or None).
            q1 = float(i)
            q2 = f"{1.2345 + i:.4f}"            # numeric string
            qd1 = None if i % 2 == 0 else f"{0.5 * i:.5f}"
            row_a = [f"{t:.4f}", q1, q2, qd1]
            wa.writerow(row_a)
            expected_a.append([
                t,
                float(q1),
                float(q2),
                np.nan if qd1 is None else float(qd1),
            ])

            fx = i * 10          # int
            fy = None
            fz = f"{-3.14159 * i:.5f}"
            row_b = [f"{t:.4f}", fx, fy, fz]
            wb.writerow(row_b)
            expected_b.append([
                t,
                float(fx),
                np.nan,
                float(fz),
            ])

        wa.flush()
        wb.flush()

    # Reopen read-only with a plain handle and verify round-trip.
    with h5py.File(path, "r") as f:
        for tname, header, expected in (
            ("table_a", header_a, expected_a),
            ("table_b", header_b, expected_b),
        ):
            grp = f[tname]
            cols = json.loads(grp.attrs["columns"])
            assert cols == header, f"{tname}: columns attr {cols} != {header}"

            for r, exp_row in enumerate(expected):
                for c, col in enumerate(header):
                    got = grp[col][r]
                    want = exp_row[c]
                    if np.isnan(want):
                        assert np.isnan(got), (
                            f"{tname}.{col}[{r}] = {got}, expected NaN"
                        )
                    else:
                        assert abs(got - want) < 1e-9, (
                            f"{tname}.{col}[{r}] = {got}, expected {want}"
                        )

            # Row count sanity: every column dataset has one element per written row.
            for col in header:
                assert grp[col].shape[0] == len(expected), (
                    f"{tname}.{col} has {grp[col].shape[0]} rows, "
                    f"expected {len(expected)}"
                )

    print("SELF-TEST OK")
