#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build conversion_table.npy from generated local-frame sensor samples.
"""

import argparse
import os
from pathlib import Path
import re
import time

import numpy as np

try:
    from . import horizon_detection_V6 as hd
except ImportError:
    import horizon_detection_V6 as hd

RX_STATIC = re.compile(r"p(?P<pitch>[-\d.]+)_r(?P<roll>[-\d.]+)\.npy$", re.I)
_MODULE_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _MODULE_DIR / "generated"
_DEFAULT_FRAMES_DIR = _GENERATED_DIR / "frames"
_DEFAULT_TABLE_PATH = _GENERATED_DIR / "conversion_table.npy"


def _iter_batches(items, batch_size):
    """Yield contiguous slices of `items` with the requested batch size."""
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _format_duration(seconds: float) -> str:
    """Format elapsed/ETA durations for progress output."""
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _build_rows_for_batch(batch_files, data_dir: Path) -> np.ndarray:
    """Load one batch of frames and convert them into conversion-table rows."""
    batch_rows = np.zeros((len(batch_files), 5), dtype=np.float64)

    for i, fname in enumerate(batch_files):
        m = RX_STATIC.fullmatch(fname)
        p_cmd = float(m["pitch"])
        r_cmd = float(m["roll"])
        data = np.load(data_dir / fname)
        result1 = hd.vector(data)
        result2 = hd.integrate_angles(data)
        batch_rows[i, :] = np.array([p_cmd, r_cmd, result1[0], result1[1], result2], dtype=np.float64)

    return batch_rows


def _build_canonical_roll_mask(conversion_table: np.ndarray) -> np.ndarray:
    """
    Detect a two-band roll table and keep the lower band as canonical.

    Uses the largest gap in sorted unique roll values as the split.
    If no clear split exists, keeps all rows.
    """
    roll = np.asarray(conversion_table[:, 1], dtype=np.float64)
    unique_roll = np.unique(roll)
    if unique_roll.size < 2:
        return np.ones(conversion_table.shape[0], dtype=bool)

    roll_sorted = np.sort(unique_roll)
    gaps = np.diff(roll_sorted)
    gap_idx = int(np.argmax(gaps))
    if float(gaps[gap_idx]) <= 1.0:
        return np.ones(conversion_table.shape[0], dtype=bool)

    split_upper = float(roll_sorted[gap_idx])
    return roll <= split_upper


def main():
    parser = argparse.ArgumentParser(description="Build conversion table from p*_r*.npy samples.")
    parser.add_argument(
        "--data-dir",
        default=str(_DEFAULT_FRAMES_DIR),
        help="Input directory containing p*_r*.npy files.",
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_TABLE_PATH),
        help="Output .npy table path.",
    )
    parser.add_argument(
        "--keep-all-roll-bands",
        action="store_true",
        help="Disable canonical-roll filtering and save all generated roll bands.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of frame files to process per progress-reporting batch.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    files = sorted(f for f in os.listdir(data_dir) if RX_STATIC.fullmatch(f))
    if not files:
        raise SystemExit(f"No static test files (p*_r*.npy) found in {data_dir}.")

    batch_size = max(1, int(args.batch_size))
    total_files = len(files)
    total_batches = (total_files + batch_size - 1) // batch_size
    conversion_table = np.zeros((len(files), 5), dtype=np.float64)
    start_time = time.perf_counter()

    for batch_index, (start, batch_files) in enumerate(_iter_batches(files, batch_size), start=1):
        end = start + len(batch_files)
        conversion_table[start:end, :] = _build_rows_for_batch(batch_files, data_dir)

        elapsed = time.perf_counter() - start_time
        processed = end
        rate = processed / elapsed if elapsed > 0.0 else 0.0
        remaining = total_files - processed
        eta_seconds = remaining / rate if rate > 0.0 else 0.0
        print(
            f"[batch {batch_index}/{total_batches}] "
            f"{processed}/{total_files} files "
            f"({processed / total_files:.1%}) "
            f"elapsed={_format_duration(elapsed)} "
            f"rate={rate:.0f} files/s "
            f"eta={_format_duration(eta_seconds)}",
            flush=True,
        )

    kept_rows = conversion_table.shape[0]
    if not args.keep_all_roll_bands:
        canonical_mask = _build_canonical_roll_mask(conversion_table)
        conversion_table = conversion_table[canonical_mask]
        kept_rows = conversion_table.shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, conversion_table)
    total_elapsed = time.perf_counter() - start_time
    if args.keep_all_roll_bands:
        print(
            f"Saved conversion table: {out_path} "
            f"({kept_rows} rows, all roll bands kept, total time {_format_duration(total_elapsed)})"
        )
    else:
        print(
            f"Saved conversion table: {out_path} "
            f"({kept_rows} rows, canonical roll band only, total time {_format_duration(total_elapsed)})"
        )


if __name__ == "__main__":
    main()
