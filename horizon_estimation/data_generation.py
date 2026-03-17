#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:04:14 2025

@author: emiel
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np

try:
    from .image_generation_V3 import simulate_earth
except ImportError:
    from image_generation_V3 import simulate_earth

try:
    from .image_generation_V3 import simulate_earth_batch
except ImportError:
    try:
        from image_generation_V3 import simulate_earth_batch
    except ImportError:
        simulate_earth_batch = None

threshold = 35.0
MIN_VALID_PIXELS = 20
MAX_VALID_PIXELS = 748
_MODULE_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _MODULE_DIR / "generated"
_DEFAULT_FRAMES_DIR = _GENERATED_DIR / "frames"
_DEFAULT_BANDS_JSON = _GENERATED_DIR / "roll_bands.json"


def build_range(start, end, step):
    """Inclusive float range with deterministic step count."""
    n = int(np.floor((end - start) / step)) + 1
    return start + np.arange(max(0, n), dtype=np.float64) * step


def generate_legacy(pitch_values, roll_values, out_dir, yaw, altitude_km, verbose=False):
    """
    Original slow mode: simulate each (pitch, roll) pair separately.
    """
    new_pitch = []
    new_roll = []

    for p in pitch_values:
        for r in roll_values:
            data = simulate_earth(
                p,
                r,
                yaw,
                altitude_km=altitude_km,
                verbose=verbose,
            )
            mask = data > threshold
            count = np.count_nonzero(mask)
            if (count > MIN_VALID_PIXELS) and (count < MAX_VALID_PIXELS):
                np.save(f"{out_dir}/p{p}_r{r}.npy", data)
                new_pitch.append(p)
                new_roll.append(r)

    return new_pitch, new_roll


def generate_batch(pitch_values, roll_values, out_dir, yaw, altitude_km, verbose=False):
    """
    Fast mode: simulate all roll values at once for each pitch.
    """
    if simulate_earth_batch is None:
        raise RuntimeError("simulate_earth_batch is not available in image_generation_V3.py")

    new_pitch = []
    new_roll = []

    for p, saved_rolls in _generate_batch_iter(
        pitch_values=pitch_values,
        roll_values=roll_values,
        out_dir=out_dir,
        yaw=yaw,
        altitude_km=altitude_km,
        verbose=verbose,
        workers=1,
    ):
        new_pitch.extend([p] * len(saved_rolls))
        new_roll.extend(saved_rolls.tolist())

    return new_pitch, new_roll


def generate_batch_parallel(pitch_values, roll_values, out_dir, yaw, altitude_km, workers=1, verbose=False):
    """
    Fast mode with optional process-level parallelism over pitch values.
    """
    if simulate_earth_batch is None:
        raise RuntimeError("simulate_earth_batch is not available in image_generation_V3.py")

    new_pitch = []
    new_roll = []

    for p, saved_rolls in _generate_batch_iter(
        pitch_values=pitch_values,
        roll_values=roll_values,
        out_dir=out_dir,
        yaw=yaw,
        altitude_km=altitude_km,
        verbose=verbose,
        workers=workers,
    ):
        new_pitch.extend([p] * len(saved_rolls))
        new_roll.extend(saved_rolls.tolist())

    return new_pitch, new_roll


def _generate_batch_iter(pitch_values, roll_values, out_dir, yaw, altitude_km, verbose, workers):
    """Yield `(pitch, saved_roll_values)` while generating and saving batch frames."""
    roll_values = np.asarray(roll_values, dtype=np.float32)
    pitch_values = np.asarray(pitch_values, dtype=np.float32)

    if int(workers) <= 1:
        for p in pitch_values:
            yield _generate_one_pitch_batch((float(p), roll_values, out_dir, float(yaw), float(altitude_km), verbose))
        return

    tasks = [
        (float(p), roll_values, out_dir, float(yaw), float(altitude_km), bool(verbose))
        for p in pitch_values
    ]
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        for result in executor.map(_generate_one_pitch_batch, tasks):
            yield result


def _generate_one_pitch_batch(task):
    """
    Generate and persist all valid roll frames for one pitch.

    Returns:
      `(pitch, saved_rolls)` where `saved_rolls` is a float32 array.
    """
    p, roll_values, out_dir, yaw, altitude_km, verbose = task
    data_stack = simulate_earth_batch(
        p,
        roll_values,
        yaw,
        altitude_km=altitude_km,
        verbose=verbose,
    )
    mask_stack = data_stack > threshold
    count_stack = np.count_nonzero(mask_stack, axis=(1, 2))
    valid = (count_stack > MIN_VALID_PIXELS) & (count_stack < MAX_VALID_PIXELS)
    saved_rolls = roll_values[valid]

    for r, data in zip(saved_rolls, data_stack[valid]):
        np.save(f"{out_dir}/p{p}_r{float(r)}.npy", data.astype(np.float32, copy=False))

    return p, saved_rolls.astype(np.float32, copy=False)


def merge_contiguous_values(values):
    """Merge sorted numeric values into contiguous [start, end] intervals."""
    if values.size == 0:
        return []
    vals = np.sort(np.unique(values.astype(np.float64)))
    if vals.size == 1:
        return [(float(vals[0]), float(vals[0]))]
    diffs = np.diff(vals)
    step = float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

    bands = []
    start = vals[0]
    prev = vals[0]
    for v in vals[1:]:
        if v <= prev + step * 1.01:
            prev = v
        else:
            bands.append((float(start), float(prev)))
            start = v
            prev = v
    bands.append((float(start), float(prev)))
    return bands


def discover_roll_bands(
    pitch_values,
    roll_candidates,
    yaw,
    altitude_km,
    valid_rate_threshold,
):
    """
    Auto-discover roll bands where horizon samples are often valid.

    A roll is considered usable when its valid ratio across sampled pitch values
    exceeds valid_rate_threshold.
    """
    if simulate_earth_batch is None:
        raise RuntimeError("simulate_earth_batch is not available in image_generation_V3.py")

    valid_counts = np.zeros(roll_candidates.size, dtype=np.int32)
    total = int(pitch_values.size)

    for p in pitch_values:
        data_stack = simulate_earth_batch(
            p,
            roll_candidates,
            yaw,
            altitude_km=altitude_km,
            verbose=False,
        )
        mask_stack = data_stack > threshold
        count_stack = np.count_nonzero(mask_stack, axis=(1, 2))
        valid_now = (count_stack > MIN_VALID_PIXELS) & (count_stack < MAX_VALID_PIXELS)
        valid_counts += valid_now.astype(np.int32, copy=False)

    valid_rate = valid_counts.astype(np.float64) / max(1, total)
    keep = valid_rate >= float(valid_rate_threshold)
    selected_rolls = roll_candidates[keep]
    bands = merge_contiguous_values(selected_rolls)
    return selected_rolls, valid_rate, bands


def expand_bands_to_rolls(bands, roll_step):
    """Convert roll bands into explicit roll values using the requested step."""
    rolls = []
    for start, end in bands:
        band_vals = build_range(float(start), float(end), float(roll_step))
        if band_vals.size:
            rolls.append(band_vals)
    if not rolls:
        return np.empty((0,), dtype=np.float64)
    return np.unique(np.concatenate(rolls))


def load_roll_bands_json(path, roll_step):
    """Load discovered bands JSON and expand to roll values."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    band_entries = payload.get("bands", [])
    bands = [(float(item["start"]), float(item["end"])) for item in band_entries]
    return expand_bands_to_rolls(bands, roll_step), payload


def clear_generated_frames(out_dir):
    """Remove previously generated frame files from output directory."""
    out_path = Path(out_dir)
    if not out_path.exists():
        return 0, []
    removed = 0
    locked = []
    for p in out_path.glob("p*_r*.npy"):
        deleted = False
        for attempt in range(5):
            try:
                p.unlink()
                removed += 1
                deleted = True
                break
            except PermissionError:
                # Windows indexing/viewers can hold short-lived locks.
                time.sleep(0.15 * (attempt + 1))
        if not deleted:
            locked.append(str(p))
    return removed, locked


def main():
    parser = argparse.ArgumentParser(description="Generate horizon-detection training data.")
    parser.add_argument(
        "--profile",
        choices=["full", "minimal-local"],
        default="full",
        help="full: scan full roll range candidates; minimal-local: generate only local valid-roll bands for a smaller single-sensor conversion table.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_DEFAULT_FRAMES_DIR),
        help="Output directory for generated .npy frames.",
    )
    parser.add_argument(
        "--clear-out-dir",
        action="store_true",
        help="Delete existing generated frame files (p*_r*.npy) in --out-dir before generation.",
    )
    parser.add_argument("--pitch-start", type=float, default=0.0)
    parser.add_argument("--pitch-end", type=float, default=359.0)
    parser.add_argument("--pitch-step", type=float, default=1.0)
    parser.add_argument("--roll-start", type=float, default=0.0)
    parser.add_argument("--roll-end", type=float, default=359.0)
    parser.add_argument("--roll-step", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angle used during generation (deg).")
    parser.add_argument("--altitude-km", type=float, default=550.0, help="Orbit altitude used for simulation (km).")
    parser.add_argument(
        "--auto-roll-bands",
        action="store_true",
        help="Automatically discover valid local roll bands for the current altitude/model.",
    )
    parser.add_argument(
        "--band-pitch-step",
        type=float,
        default=5.0,
        help="Pitch step for auto-band discovery (coarser is faster).",
    )
    parser.add_argument(
        "--band-valid-rate",
        type=float,
        default=0.10,
        help="Minimum valid ratio per roll to keep in auto-band discovery.",
    )
    parser.add_argument(
        "--bands-json",
        default=str(_DEFAULT_BANDS_JSON),
        help="Optional JSON output path for discovered roll bands and metadata.",
    )
    parser.add_argument(
        "--use-bands-json",
        default=None,
        help="Load roll bands from a previous --bands-json file and generate using those rolls.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Run auto roll-band discovery and exit without generating frames.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use original per-frame generation mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for batch generation (pitch-level parallelism).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print simulation progress from image generation",
    )
    args = parser.parse_args()

    pitch_values = build_range(args.pitch_start, args.pitch_end, args.pitch_step)
    if pitch_values.size == 0:
        raise SystemExit("Empty pitch range. Check --pitch-start/--pitch-end/--pitch-step.")

    if args.use_bands_json:
        roll_values, source_payload = load_roll_bands_json(args.use_bands_json, args.roll_step)
        source_altitude = source_payload.get("altitude_km", None)
        if source_altitude is not None and abs(float(source_altitude) - float(args.altitude_km)) > 1e-9:
            print(
                "Warning: --use-bands-json altitude_km "
                f"({float(source_altitude):.3f}) differs from --altitude-km ({float(args.altitude_km):.3f})."
            )
    elif args.profile == "minimal-local":
        # Local-frame valid roll bands observed for this sensor/FOV setup.
        roll_a = build_range(15.0, 119.0, args.roll_step)
        roll_b = build_range(243.0, 345.0, args.roll_step)
        roll_values = np.unique(np.concatenate([roll_a, roll_b]))
    else:
        roll_values = build_range(args.roll_start, args.roll_end, args.roll_step)

    if roll_values.size == 0:
        raise SystemExit("Empty roll range. Check roll options.")

    if args.auto_roll_bands:
        roll_candidates = roll_values.copy()
        discover_pitch = build_range(args.pitch_start, args.pitch_end, args.band_pitch_step)
        discovered_rolls, valid_rate, bands = discover_roll_bands(
            pitch_values=discover_pitch,
            roll_candidates=roll_candidates,
            yaw=float(args.yaw),
            altitude_km=float(args.altitude_km),
            valid_rate_threshold=float(args.band_valid_rate),
        )
        if discovered_rolls.size == 0:
            raise SystemExit(
                "Auto roll-band discovery found no usable rolls. "
                "Try lower --band-valid-rate or verify altitude/threshold settings."
            )
        roll_values = discovered_rolls
        print("Auto-discovered roll bands:")
        for start, end in bands:
            print(f"  {start:.3f} .. {end:.3f}")

        if args.bands_json:
            bands_json_path = Path(args.bands_json)
            payload = {
                "altitude_km": float(args.altitude_km),
                "yaw": float(args.yaw),
                "band_valid_rate": float(args.band_valid_rate),
                "band_pitch_step": float(args.band_pitch_step),
                "roll_step": float(args.roll_step),
                "bands": [{"start": float(a), "end": float(b)} for a, b in bands],
                "roll_count": int(roll_values.size),
                "roll_min": float(np.min(roll_values)),
                "roll_max": float(np.max(roll_values)),
                "avg_valid_rate_kept": float(np.mean(valid_rate[np.isin(roll_candidates, roll_values)])),
            }
            bands_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bands_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        if args.discover_only:
            return
    elif args.discover_only:
        raise SystemExit("--discover-only requires --auto-roll-bands.")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.clear_out_dir:
        removed, locked = clear_generated_frames(args.out_dir)
        print(f"Cleared {removed} existing frame files from {args.out_dir}")
        if locked:
            preview = ", ".join(locked[:3])
            if len(locked) > 3:
                preview += ", ..."
            print(
                f"Warning: {len(locked)} files could not be removed due to file locks: {preview}. "
                "Close programs using those files and rerun cleanup if needed."
            )

    if args.legacy:
        generate_legacy(
            pitch_values,
            roll_values,
            args.out_dir,
            yaw=float(args.yaw),
            altitude_km=float(args.altitude_km),
            verbose=args.verbose,
        )
    else:
        generate_batch_parallel(
            pitch_values,
            roll_values,
            args.out_dir,
            yaw=float(args.yaw),
            altitude_km=float(args.altitude_km),
            workers=max(1, int(args.workers)),
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
