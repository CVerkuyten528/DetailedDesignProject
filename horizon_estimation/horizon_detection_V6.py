#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on: 18/02/2026

@author: Rune
"""

from pathlib import Path

import numpy as np

try:
    from .image_generation_V3 import simulate_earth
except ImportError:
    from image_generation_V3 import simulate_earth

# Parameters
reference_vector = (0,-1) # Vector pointing down in sensor boresight
sensor = (24, 32)
origin = (11, 15)
_MODULE_DIR = Path(__file__).resolve().parent
_FOV_DIR = _MODULE_DIR / "FOV_Files"
_GENERATED_DIR = _MODULE_DIR / "generated"
_CONV_TABLE_PATH = _GENERATED_DIR / "conversion_table.npy"

# Load X and Y angles from text files.
X_angles_deg = np.loadtxt(_FOV_DIR / "horizontal_angles.txt", delimiter="\t") # Use 32x24
Y_angles_deg = np.loadtxt(_FOV_DIR / "vertical_angles.txt", delimiter="\t") # Use 32x24

# Convert to radians.
X_angles = np.radians(X_angles_deg)
Y_angles = np.radians(Y_angles_deg)

# Convert from pixel coordinates to fov angle coordinates in radians
origin = np.array([X_angles[origin], Y_angles[origin]])

# Load conversion table when available.
if _CONV_TABLE_PATH.exists():
    conv_table = np.load(_CONV_TABLE_PATH)
else:
    conv_table = np.empty((0, 5), dtype=np.float64)
# Stores the latest valid pitch/roll so invalid frames can still return a guess.
last_pitch_roll = (np.nan, np.nan)

# Precompute once
X_gradient = np.gradient(X_angles, axis=1)
Y_gradient = np.gradient(Y_angles, axis=0)
angle_area = np.abs(X_gradient * Y_gradient)

X_rel = X_angles - origin[0]
Y_rel = Y_angles - origin[1]

def vector(data, threshold=35):
    """
    Estimate a unit direction vector from the Earth/space mask.

    Returns (nan, nan) when too few or too many Earth pixels are visible.
    """
    mask = data > threshold
    count = np.count_nonzero(mask)

    # Sensor sees too little or too much of earth
    if (count < 20) or (count > 748):
        return np.nan, np.nan

    x_sum = np.sum(X_rel[mask], dtype=np.float64)
    y_sum = np.sum(Y_rel[mask], dtype=np.float64)
    norm = np.hypot(x_sum, y_sum)

    if norm == 0.0:
        return np.nan, np.nan
    unit_vector = np.array([x_sum / norm, y_sum / norm], dtype=np.float32)
    
    return unit_vector

def integrate_angles(data, threshold=35):
    """
    Integrate the angular area covered by pixels above threshold.
    """
    data_mask = data > threshold
    return np.sum(angle_area[data_mask], dtype=np.float64)

def convert_coordinates(vector, area):
    """
    Map (vector, area) features to pitch/roll using nearest neighbour lookup.

    Fallback behavior for invalid inputs:
    1) Reuse the previous valid pitch/roll.
    2) If area is still valid, use area-only nearest neighbour.
    3) Otherwise use the middle entry in the conversion table.
    """
    global last_pitch_roll
    if conv_table.size == 0:
        raise RuntimeError(
            "conversion_table.npy not found. Generate it with "
            "`python -m horizon_estimation.conversion_table` first."
        )
    
    vector_invalid = np.isnan(vector[0]) or np.isnan(vector[1])
    area_invalid = np.isnan(area)
    
    if vector_invalid or area_invalid:
        # First fallback: use last valid output if available.
        if not (np.isnan(last_pitch_roll[0]) or np.isnan(last_pitch_roll[1])):
            return last_pitch_roll
        
        # Second fallback: area-only nearest neighbour.
        if not area_invalid:
            diff_area = np.abs(conv_table[:, 4] - area)
            index = np.argmin(diff_area)
        else:
            # Last fallback when all features are invalid.
            index = conv_table.shape[0] // 2
    else:
        diff = conv_table[:, 2:] - [vector[0], vector[1], area]
        dist2 = diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2
        index = np.argmin(dist2)
    
    pitch, roll = conv_table[index, :2]
    last_pitch_roll = (pitch, roll)
    
    return pitch, roll

if __name__=="__main__":
    data = simulate_earth(0, 80, 0, plot=True)
    vec = vector(data)
    area = integrate_angles(data)
    p, r = convert_coordinates(vec,area)
    print(f"Vector: {vec}, area: {area}")
    print(f"Pitch: {p}, Roll: {r}")
