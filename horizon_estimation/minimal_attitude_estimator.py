#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal single-file attitude estimator.

This module keeps the same core estimation path as the current realtime
multi-sensor workflow:
  - 4 side sensors with roll offsets 0/90/180/270 deg
  - thermal Earth simulation from body pitch/roll/yaw
  - horizon feature extraction
  - nearest-neighbor lookup against conversion_table.npy
  - branch-aware pitch continuity gating
  - recent-valid fallback and switch-margin-based sensor handover
  - active-sensor pitch/roll resolution back to shared pitch/body roll
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EARTH_THRESHOLD = 35.0
SENSOR_ROLL_OFFSETS_DEG = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)

_MODULE_DIR = Path(__file__).resolve().parent
_FOV_DIR = _MODULE_DIR / "FOV_Files"
_GENERATED_DIR = _MODULE_DIR / "generated"


def _wrap_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _effective_horizon_switch_margin(active_ref_dist: float, switch_margin: float) -> float:
    margin = float(switch_margin)
    if margin < 1.0:
        return margin
    level = min(10.0, max(1.0, margin))
    rel = 0.005 + (level - 1.0) * (0.05 - 0.005) / 9.0
    return float(active_ref_dist) * rel


def _load_text_array(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t")


_X_ANGLES_HIGHRES_DEG = _load_text_array(_FOV_DIR / "horizontal_angles_highres_Scipy_Linear.txt")
_Y_ANGLES_HIGHRES_DEG = _load_text_array(_FOV_DIR / "vertical_angles_highres_Scipy_Linear.txt")
_X_ANGLES_HIGHRES = np.radians(_X_ANGLES_HIGHRES_DEG).astype(np.float32, copy=False)
_Y_ANGLES_HIGHRES = np.radians(_Y_ANGLES_HIGHRES_DEG).astype(np.float32, copy=False)

_X_ANGLES_DEG = _load_text_array(_FOV_DIR / "horizontal_angles.txt")
_Y_ANGLES_DEG = _load_text_array(_FOV_DIR / "vertical_angles.txt")
_X_ANGLES = np.radians(_X_ANGLES_DEG).astype(np.float32, copy=False)
_Y_ANGLES = np.radians(_Y_ANGLES_DEG).astype(np.float32, copy=False)
_ORIGIN_INDEX = (11, 15)
_ORIGIN = np.array([_X_ANGLES[_ORIGIN_INDEX], _Y_ANGLES[_ORIGIN_INDEX]], dtype=np.float32)
_X_GRADIENT = np.gradient(_X_ANGLES, axis=1).astype(np.float32, copy=False)
_Y_GRADIENT = np.gradient(_Y_ANGLES, axis=0).astype(np.float32, copy=False)
_ANGLE_AREA = np.abs(_X_GRADIENT * _Y_GRADIENT).astype(np.float32, copy=False)
_X_REL = (_X_ANGLES - _ORIGIN[0]).astype(np.float32, copy=False)
_Y_REL = (_Y_ANGLES - _ORIGIN[1]).astype(np.float32, copy=False)

_CONV_TABLE_PATH = _GENERATED_DIR / "conversion_table.npy"
if not _CONV_TABLE_PATH.exists():
    legacy_path = _MODULE_DIR / "conversion_table.npy"
    _CONV_TABLE_PATH = legacy_path if legacy_path.exists() else _CONV_TABLE_PATH

_CONV_TABLE_F32 = np.load(_CONV_TABLE_PATH).astype(np.float32, copy=False)
_CONV_FEATURES_F32 = _CONV_TABLE_F32[:, 2:]
_CONV_PITCH_ROLL_F32 = _CONV_TABLE_F32[:, :2]
_CONV_PITCH_F32 = _CONV_PITCH_ROLL_F32[:, 0]
_CONV_ROLL_F32 = _CONV_PITCH_ROLL_F32[:, 1]
_CONV_FEATURES_NORM2_F32 = np.sum(_CONV_FEATURES_F32 * _CONV_FEATURES_F32, axis=1, dtype=np.float32)


def earth_pixel_count(frame: np.ndarray, threshold: float = EARTH_THRESHOLD) -> int:
    return int(np.count_nonzero(frame > threshold))


def vector(frame: np.ndarray, threshold: float = EARTH_THRESHOLD) -> tuple[float, float]:
    mask = frame > threshold
    count = int(np.count_nonzero(mask))
    if (count < 20) or (count > 748):
        return np.nan, np.nan

    x_sum = np.sum(_X_REL[mask], dtype=np.float64)
    y_sum = np.sum(_Y_REL[mask], dtype=np.float64)
    norm = float(np.hypot(x_sum, y_sum))
    if norm == 0.0:
        return np.nan, np.nan
    return float(x_sum / norm), float(y_sum / norm)


def integrate_angles(frame: np.ndarray, threshold: float = EARTH_THRESHOLD) -> float:
    return float(np.sum(_ANGLE_AREA[frame > threshold], dtype=np.float64))


def build_pitch_prior_array(
    prior_est_pitch: np.ndarray | None = None,
    shared_pitch_prior: float = np.nan,
    n_items: int | None = None,
) -> np.ndarray | None:
    if np.isfinite(shared_pitch_prior):
        if n_items is None:
            if prior_est_pitch is None:
                raise ValueError("n_items is required when no per-sensor pitch prior is provided.")
            n_items = int(np.asarray(prior_est_pitch).size)
        return np.full(int(n_items), np.float32(float(shared_pitch_prior) % 360.0), dtype=np.float32)
    if prior_est_pitch is None:
        return None
    return np.asarray(prior_est_pitch, dtype=np.float32)


def _lookup_features_nn(
    features: np.ndarray,
    prior_pitch_for_nn: np.ndarray | None = None,
    pitch_continuity_deg: float = 10.0,
    pitch_neighbor_k: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(features.shape[0])
    horizon_dist_raw = np.full(n, np.float32(np.inf), dtype=np.float32)
    est_pitch = np.full(n, np.float32(np.nan), dtype=np.float32)
    est_roll = np.full(n, np.float32(np.nan), dtype=np.float32)

    valid_idx = np.where(np.isfinite(features).all(axis=1))[0]
    if valid_idx.size == 0:
        return horizon_dist_raw, est_pitch, est_roll

    query = features[valid_idx]
    query_norm2 = np.sum(query * query, axis=1, dtype=np.float32)
    dist2 = (
        query_norm2[:, None]
        + _CONV_FEATURES_NORM2_F32[None, :]
        - np.float32(2.0) * (query @ _CONV_FEATURES_F32.T)
    )
    np.maximum(dist2, np.float32(0.0), out=dist2)

    use_pitch_gate = (
        prior_pitch_for_nn is not None
        and float(pitch_continuity_deg) > 0.0
        and int(pitch_neighbor_k) > 1
    )
    rows = np.arange(valid_idx.size)
    if not use_pitch_gate:
        best_idx = np.argmin(dist2, axis=1)
        horizon_dist_raw[valid_idx] = dist2[rows, best_idx]
        est_pitch[valid_idx] = _CONV_PITCH_ROLL_F32[best_idx, 0]
        est_roll[valid_idx] = _CONV_PITCH_ROLL_F32[best_idx, 1]
        return horizon_dist_raw, est_pitch, est_roll

    prior_pitch_arr = np.asarray(prior_pitch_for_nn, dtype=np.float32)
    k_scan = int(max(2, min(int(pitch_neighbor_k), int(_CONV_FEATURES_F32.shape[0]))))
    pitch_gate = np.float32(float(pitch_continuity_deg))

    topk_idx_unsorted = np.argpartition(dist2, kth=k_scan - 1, axis=1)[:, :k_scan]
    topk_dist2_unsorted = np.take_along_axis(dist2, topk_idx_unsorted, axis=1)
    order = np.argsort(topk_dist2_unsorted, axis=1)
    topk_idx = np.take_along_axis(topk_idx_unsorted, order, axis=1)
    topk_dist2 = np.take_along_axis(topk_dist2_unsorted, order, axis=1)

    for row, qidx in enumerate(valid_idx):
        if qidx >= prior_pitch_arr.size or (not np.isfinite(prior_pitch_arr[qidx])):
            chosen_col = 0
        else:
            cand_pitch = _CONV_PITCH_F32[topk_idx[row]]
            pitch_diff_c = np.abs(
                (cand_pitch - prior_pitch_arr[qidx] + np.float32(180.0)) % np.float32(360.0) - np.float32(180.0)
            )
            cand_pitch_m = (cand_pitch + np.float32(180.0)) % np.float32(360.0)
            pitch_diff_m = np.abs(
                (cand_pitch_m - prior_pitch_arr[qidx] + np.float32(180.0)) % np.float32(360.0) - np.float32(180.0)
            )
            pitch_diff = np.minimum(pitch_diff_c, pitch_diff_m)
            ok = np.where(pitch_diff <= pitch_gate)[0]
            if ok.size == 0:
                continue
            chosen_col = int(ok[0])

        chosen_idx = int(topk_idx[row, chosen_col])
        horizon_dist_raw[qidx] = np.float32(topk_dist2[row, chosen_col])
        est_pitch[qidx] = _CONV_PITCH_ROLL_F32[chosen_idx, 0]
        est_roll[qidx] = _CONV_PITCH_ROLL_F32[chosen_idx, 1]

    return horizon_dist_raw, est_pitch, est_roll


def _resolve_local_mirror_with_prior(
    est_pitch: float,
    est_roll: float,
    prior_pitch: float,
    prior_roll: float,
) -> tuple[float, float]:
    if not (np.isfinite(est_pitch) and np.isfinite(est_roll)):
        return est_pitch, est_roll
    p_c = float(est_pitch) % 360.0
    r_c = float(est_roll) % 360.0
    p_m = (p_c + 180.0) % 360.0
    r_m = (360.0 - r_c) % 360.0

    s_c = abs(_wrap_angle_deg(p_c - float(prior_pitch))) + abs(_wrap_angle_deg(r_c - float(prior_roll)))
    s_m = abs(_wrap_angle_deg(p_m - float(prior_pitch))) + abs(_wrap_angle_deg(r_m - float(prior_roll)))
    if s_m < s_c:
        return p_m, r_m
    return p_c, r_c


def _resolve_local_mirror_with_prior_arrays(
    est_pitch: np.ndarray,
    est_roll: np.ndarray,
    prior_pitch: np.ndarray,
    prior_roll: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(est_pitch, dtype=np.float32).copy()
    r = np.asarray(est_roll, dtype=np.float32).copy()
    pp = np.asarray(prior_pitch, dtype=np.float32)
    pr = np.asarray(prior_roll, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    n = min(p.size, r.size, pp.size, pr.size, valid.size)
    for idx in range(n):
        if not valid[idx]:
            continue
        if not (np.isfinite(pp[idx]) and np.isfinite(pr[idx])):
            continue
        p[idx], r[idx] = _resolve_local_mirror_with_prior(p[idx], r[idx], pp[idx], pr[idx])
    return p, r


def apply_pitch_continuity_gate(
    dist_raw: np.ndarray,
    valid_now: np.ndarray,
    est_pitch_now: np.ndarray,
    est_roll_now: np.ndarray,
    prior_pitch: float | np.ndarray,
    pitch_continuity_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate = float(pitch_continuity_deg)
    if gate <= 0.0:
        return dist_raw, valid_now, est_pitch_now, est_roll_now

    dist_raw = np.asarray(dist_raw, dtype=np.float32).copy()
    valid_now = np.asarray(valid_now, dtype=bool).copy()
    est_pitch_now = np.asarray(est_pitch_now, dtype=np.float32).copy()
    est_roll_now = np.asarray(est_roll_now, dtype=np.float32).copy()

    if np.isscalar(prior_pitch):
        if not np.isfinite(prior_pitch):
            return dist_raw, valid_now, est_pitch_now, est_roll_now
        prior_arr = np.full(valid_now.size, np.float32(float(prior_pitch) % 360.0), dtype=np.float32)
    else:
        prior_arr = np.asarray(prior_pitch, dtype=np.float32)
        if prior_arr.size == 0:
            return dist_raw, valid_now, est_pitch_now, est_roll_now

    for idx in range(valid_now.size):
        if (not bool(valid_now[idx])) or idx >= prior_arr.size or (not np.isfinite(prior_arr[idx])):
            continue
        pitch_diff = abs(_wrap_angle_deg(float(est_pitch_now[idx]) - float(prior_arr[idx])))
        if pitch_diff > gate:
            valid_now[idx] = False
            dist_raw[idx] = np.float32(np.inf)
            est_pitch_now[idx] = np.float32(np.nan)
            est_roll_now[idx] = np.float32(np.nan)

    return dist_raw, valid_now, est_pitch_now, est_roll_now


def compute_sensor_metrics(
    frames: np.ndarray,
    threshold: float = EARTH_THRESHOLD,
    prior_est_pitch: np.ndarray | None = None,
    prior_est_roll: np.ndarray | None = None,
    shared_pitch_prior: float = np.nan,
    pitch_continuity_deg: float = 10.0,
    pitch_neighbor_k: int = 16,
) -> dict[str, np.ndarray]:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("frames must have shape (N, H, W)")

    n = int(frames.shape[0])
    pixel_scores = np.zeros(n, dtype=np.int32)
    features = np.full((n, 3), np.float32(np.nan), dtype=np.float32)

    for idx in range(n):
        frame = frames[idx]
        pixel_scores[idx] = earth_pixel_count(frame, threshold=threshold)
        vec = vector(frame, threshold=threshold)
        area = integrate_angles(frame, threshold=threshold)
        if not (np.isfinite(vec).all() and np.isfinite(area)):
            continue
        features[idx, :] = np.array([vec[0], vec[1], area], dtype=np.float32)

    pitch_prior = build_pitch_prior_array(
        prior_est_pitch=prior_est_pitch,
        shared_pitch_prior=shared_pitch_prior,
        n_items=n,
    )
    horizon_dist_raw, est_pitch, est_roll = _lookup_features_nn(
        features=features,
        prior_pitch_for_nn=pitch_prior,
        pitch_continuity_deg=pitch_continuity_deg,
        pitch_neighbor_k=pitch_neighbor_k,
    )
    horizon_valid = np.isfinite(horizon_dist_raw)

    if pitch_prior is not None and prior_est_roll is not None:
        est_pitch, est_roll = _resolve_local_mirror_with_prior_arrays(
            est_pitch=est_pitch,
            est_roll=est_roll,
            prior_pitch=pitch_prior,
            prior_roll=prior_est_roll,
            valid_mask=horizon_valid,
        )

    horizon_dist_raw, horizon_valid, est_pitch, est_roll = apply_pitch_continuity_gate(
        dist_raw=horizon_dist_raw,
        valid_now=horizon_valid,
        est_pitch_now=est_pitch,
        est_roll_now=est_roll,
        prior_pitch=pitch_prior if pitch_prior is not None else np.nan,
        pitch_continuity_deg=pitch_continuity_deg,
    )

    return {
        "pixel_scores": pixel_scores,
        "horizon_dist_raw": horizon_dist_raw,
        "horizon_valid": horizon_valid,
        "est_pitch": est_pitch,
        "est_roll": est_roll,
    }


def update_effective_horizon_dist(
    last_valid_horizon_dist: np.ndarray,
    horizon_dist_raw: np.ndarray,
    horizon_valid: np.ndarray,
    invalid_streak: np.ndarray | None = None,
    invalid_fallback_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    last_valid_horizon_dist = np.asarray(last_valid_horizon_dist, dtype=np.float32).copy()
    horizon_dist_raw = np.asarray(horizon_dist_raw, dtype=np.float32)
    horizon_valid = np.asarray(horizon_valid, dtype=bool)

    dist_effective = horizon_dist_raw.copy()
    invalid_now = ~horizon_valid
    dist_effective[invalid_now] = last_valid_horizon_dist[invalid_now]

    if invalid_streak is not None and invalid_fallback_frames is not None:
        streak = np.asarray(invalid_streak, dtype=np.int32)
        hold = int(max(0, invalid_fallback_frames))
        stale_invalid = invalid_now & (streak > hold)
        dist_effective[stale_invalid] = np.float32(np.nan)

    last_valid_horizon_dist[horizon_valid] = horizon_dist_raw[horizon_valid]
    return dist_effective, last_valid_horizon_dist


def select_active_sensor(
    pixel_scores: np.ndarray,
    horizon_dist_raw: np.ndarray,
    horizon_dist_effective: np.ndarray,
    horizon_valid_now: np.ndarray,
    horizon_recent_valid: np.ndarray | None = None,
    active_sensor: int | None = None,
    switch_margin: float = 7.0,
) -> int:
    pixel_scores = np.asarray(pixel_scores)
    horizon_dist_raw = np.asarray(horizon_dist_raw, dtype=np.float32)
    horizon_dist_effective = np.asarray(horizon_dist_effective, dtype=np.float32)
    horizon_valid_now = np.asarray(horizon_valid_now, dtype=bool)
    if horizon_recent_valid is None:
        horizon_recent_valid = horizon_valid_now.copy()
    else:
        horizon_recent_valid = np.asarray(horizon_recent_valid, dtype=bool)

    if active_sensor is None:
        if np.any(horizon_valid_now):
            return int(np.argmin(np.where(horizon_valid_now, horizon_dist_raw, np.inf)))
        if np.any(horizon_recent_valid):
            masked_recent = np.where(horizon_recent_valid, horizon_dist_effective, np.inf)
            masked_recent = np.where(np.isfinite(masked_recent), masked_recent, np.inf)
            if np.any(np.isfinite(masked_recent)):
                return int(np.argmin(masked_recent))
        if np.any(np.isfinite(horizon_dist_effective)):
            return int(np.argmin(np.where(np.isfinite(horizon_dist_effective), horizon_dist_effective, np.inf)))
        return int(np.argmax(pixel_scores))

    active_sensor = int(active_sensor)
    active_ref_dist = (
        float(horizon_dist_raw[active_sensor])
        if horizon_valid_now[active_sensor]
        else float(horizon_dist_effective[active_sensor])
    )

    candidates = np.where(horizon_recent_valid)[0]
    if candidates.size == 0:
        return active_sensor

    candidate_dist = np.where(horizon_valid_now[candidates], horizon_dist_raw[candidates], horizon_dist_effective[candidates])
    candidate_dist = np.where(np.isfinite(candidate_dist), candidate_dist, np.inf)
    if not np.any(np.isfinite(candidate_dist)):
        return active_sensor

    best_idx = int(candidates[np.argmin(candidate_dist)])
    best_dist = float(
        horizon_dist_raw[best_idx] if horizon_valid_now[best_idx] else horizon_dist_effective[best_idx]
    )
    if best_idx == active_sensor:
        return active_sensor
    if not np.isfinite(active_ref_dist):
        return best_idx

    required_improvement = _effective_horizon_switch_margin(active_ref_dist, switch_margin)
    if best_dist + required_improvement < active_ref_dist:
        return best_idx
    return active_sensor


def resolve_body_roll_with_prior(local_roll: float, sensor_roll_offset: float, prior_body_roll: float = np.nan) -> float:
    cand_c = _wrap_angle_deg(float(local_roll) - float(sensor_roll_offset))
    cand_m = _wrap_angle_deg((360.0 - float(local_roll)) - float(sensor_roll_offset))
    if not np.isfinite(prior_body_roll):
        return cand_c
    if abs(_wrap_angle_deg(cand_m - float(prior_body_roll))) < abs(_wrap_angle_deg(cand_c - float(prior_body_roll))):
        return cand_m
    return cand_c


def resolve_pitch_with_prior(local_pitch: float, prior_pitch: float = np.nan) -> float:
    p_c = float(local_pitch) % 360.0
    p_m = (p_c + 180.0) % 360.0
    if not np.isfinite(prior_pitch):
        return p_c
    if abs(_wrap_angle_deg(p_m - float(prior_pitch))) < abs(_wrap_angle_deg(p_c - float(prior_pitch))):
        return p_m
    return p_c


def _downsample_stack(images: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    n, old_rows, old_cols = images.shape
    new_rows, new_cols = new_shape
    factor_row = old_rows // new_rows
    factor_col = old_cols // new_cols
    return images.reshape(n, new_rows, factor_row, new_cols, factor_col).mean(axis=(2, 4), dtype=np.float32)


def simulate_sensor_frames(
    pitch_deg: float,
    roll_deg: float,
    yaw_deg: float,
    altitude_km: float = 550.0,
    earth_radius_km: float = 6378.14,
    earth_temp: float = 45.0,
    space_temp: float = 25.0,
) -> np.ndarray:
    roll_values = (float(roll_deg) + SENSOR_ROLL_OFFSETS_DEG).astype(np.float32, copy=False)

    pitch_rad = np.float32(np.radians(float(pitch_deg)))
    yaw_rad = np.float32(np.radians(float(yaw_deg)))
    roll_rad = np.radians(roll_values).astype(np.float32, copy=False).reshape(-1, 1, 1)

    earth_half_angle = np.float32(math.asin(float(earth_radius_km) / (float(earth_radius_km) + float(altitude_km))))
    earth_cos = np.float32(math.cos(float(earth_half_angle)))

    center_row = int(_X_ANGLES_HIGHRES.shape[0] / 2)
    center_col = int(_X_ANGLES_HIGHRES.shape[1] / 2)
    x_center = _X_ANGLES_HIGHRES[center_row, center_col]
    y_center = _Y_ANGLES_HIGHRES[center_row, center_col]

    x_rel = _X_ANGLES_HIGHRES - x_center
    y_rel = _Y_ANGLES_HIGHRES - y_center

    cos_pitch = np.float32(math.cos(float(pitch_rad)))
    sin_pitch = np.float32(math.sin(float(pitch_rad)))
    x_angles_rot = x_rel * cos_pitch - y_rel * sin_pitch + x_center
    y_angles_rot = x_rel * sin_pitch + y_rel * cos_pitch + y_center + roll_rad

    alpha_loc = x_angles_rot + yaw_rad
    beta_loc = y_angles_rot
    inside_earth = (np.cos(beta_loc) * np.cos(alpha_loc)) > earth_cos

    earth_fraction = _downsample_stack(inside_earth.astype(np.float32, copy=False), (24, 32))
    return np.float32(space_temp) + np.float32(earth_temp - space_temp) * earth_fraction


@dataclass
class AttitudeEstimate:
    estimated_pitch_deg: float
    estimated_roll_deg: float
    active_sensor: int | None
    did_poll: bool
    sensor_pitch_deg: np.ndarray
    sensor_roll_deg: np.ndarray
    sensor_valid: np.ndarray
    sensor_distance_raw: np.ndarray
    sensor_distance_effective: np.ndarray


class MinimalAttitudeEstimator:
    def __init__(
        self,
        yaw_deg: float = 5.0,
        sim_fps: float = 30.0,
        poll_rate: float = 5.0,
        switch_margin: float = 7.0,
        valid_window: int = 3,
        threshold: float = EARTH_THRESHOLD,
        pitch_continuity_deg: float = 10.0,
        pitch_neighbor_k: int = 16,
        invalid_fallback_frames: int = 1,
    ) -> None:
        if sim_fps <= 0.0:
            raise ValueError("sim_fps must be > 0.")
        if poll_rate <= 0.0:
            raise ValueError("poll_rate must be > 0.")

        self.yaw_deg = float(yaw_deg)
        self.sim_fps = float(sim_fps)
        self.poll_rate = float(poll_rate)
        self.sim_dt_default = 1.0 / self.sim_fps
        self.poll_dt = 1.0 / self.poll_rate
        self.switch_margin = float(switch_margin)
        self.valid_window = int(max(0, valid_window))
        self.threshold = float(threshold)
        self.pitch_continuity_deg = float(pitch_continuity_deg)
        self.pitch_neighbor_k = int(max(2, pitch_neighbor_k))
        self.invalid_fallback_frames = int(max(0, invalid_fallback_frames))

        self.active_sensor: int | None = None
        self.last_valid_horizon_dist = np.full(4, np.float32(np.inf), dtype=np.float32)
        self.last_valid_est_pitch = np.full(4, np.float32(np.nan), dtype=np.float32)
        self.last_valid_est_roll = np.full(4, np.float32(np.nan), dtype=np.float32)
        self.pitch_est_filtered = np.nan
        self.roll_est_filtered = np.nan
        self.frames_since_valid = np.full(4, 1_000_000, dtype=np.int32)
        self.poll_elapsed_s = 0.0
        self.have_poll = False
        self.last_sensor_pitch = np.full(4, np.float32(np.nan), dtype=np.float32)
        self.last_sensor_roll = np.full(4, np.float32(np.nan), dtype=np.float32)
        self.last_sensor_valid = np.zeros(4, dtype=bool)
        self.last_sensor_dist_raw = np.full(4, np.float32(np.inf), dtype=np.float32)
        self.last_sensor_dist_effective = np.full(4, np.float32(np.inf), dtype=np.float32)
        self._initialized = False

    def _seed_priors(self, true_pitch_deg: float, true_roll_deg: float) -> None:
        pitch_mod = np.float32(float(true_pitch_deg) % 360.0)
        roll_mod = np.mod(float(true_roll_deg) + SENSOR_ROLL_OFFSETS_DEG, 360.0).astype(np.float32)
        self.last_valid_est_pitch[:] = pitch_mod
        self.last_valid_est_roll[:] = roll_mod
        self._initialized = True

    def update(
        self,
        true_pitch_deg: float,
        true_roll_deg: float,
        dt_s: float | None = None,
        yaw_deg: float | None = None,
    ) -> AttitudeEstimate:
        if not self._initialized:
            self._seed_priors(true_pitch_deg, true_roll_deg)

        dt = self.sim_dt_default if dt_s is None else float(dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be > 0.")
        yaw_now = self.yaw_deg if yaw_deg is None else float(yaw_deg)

        self.poll_elapsed_s += dt
        did_poll = (not self.have_poll) or (self.poll_elapsed_s + 1e-12 >= self.poll_dt)
        poll_dt_used = self.poll_dt

        if did_poll:
            if self.have_poll:
                poll_dt_used = self.poll_elapsed_s
            self.poll_elapsed_s = 0.0
            self.have_poll = True

            frames = simulate_sensor_frames(
                pitch_deg=true_pitch_deg,
                roll_deg=true_roll_deg,
                yaw_deg=yaw_now,
            )
            shared_pitch_prior = self.pitch_est_filtered if np.isfinite(self.pitch_est_filtered) else np.nan
            metrics = compute_sensor_metrics(
                frames=frames,
                threshold=self.threshold,
                prior_est_pitch=self.last_valid_est_pitch,
                prior_est_roll=self.last_valid_est_roll,
                shared_pitch_prior=shared_pitch_prior,
                pitch_continuity_deg=self.pitch_continuity_deg,
                pitch_neighbor_k=self.pitch_neighbor_k,
            )

            dist_raw = np.asarray(metrics["horizon_dist_raw"], dtype=np.float32)
            valid_now = np.asarray(metrics["horizon_valid"], dtype=bool)
            est_pitch_now = np.asarray(metrics["est_pitch"], dtype=np.float32)
            est_roll_now = np.asarray(metrics["est_roll"], dtype=np.float32)

            self.frames_since_valid[valid_now] = 0
            self.frames_since_valid[~valid_now] += 1
            recent_valid = self.frames_since_valid <= self.valid_window

            self.last_valid_est_pitch[valid_now] = est_pitch_now[valid_now]
            self.last_valid_est_roll[valid_now] = est_roll_now[valid_now]
            dist_effective, self.last_valid_horizon_dist = update_effective_horizon_dist(
                last_valid_horizon_dist=self.last_valid_horizon_dist,
                horizon_dist_raw=dist_raw,
                horizon_valid=valid_now,
                invalid_streak=self.frames_since_valid,
                invalid_fallback_frames=self.invalid_fallback_frames,
            )

            self.active_sensor = select_active_sensor(
                pixel_scores=metrics["pixel_scores"],
                horizon_dist_raw=dist_raw,
                horizon_dist_effective=dist_effective,
                horizon_valid_now=valid_now,
                horizon_recent_valid=recent_valid,
                active_sensor=self.active_sensor,
                switch_margin=self.switch_margin,
            )

            self.last_sensor_pitch = est_pitch_now
            self.last_sensor_roll = est_roll_now
            self.last_sensor_valid = valid_now
            self.last_sensor_dist_raw = dist_raw
            self.last_sensor_dist_effective = dist_effective

            if self.active_sensor is not None:
                p_sensor = float(self.last_sensor_pitch[self.active_sensor])
                r_sensor = float(self.last_sensor_roll[self.active_sensor])
                if not np.isfinite(p_sensor):
                    p_sensor = float(self.last_valid_est_pitch[self.active_sensor])
                    r_sensor = float(self.last_valid_est_roll[self.active_sensor])

                if np.isfinite(p_sensor) and np.isfinite(r_sensor):
                    self.pitch_est_filtered = resolve_pitch_with_prior(
                        local_pitch=p_sensor,
                        prior_pitch=self.pitch_est_filtered,
                    )

                    self.roll_est_filtered = resolve_body_roll_with_prior(
                        local_roll=r_sensor,
                        sensor_roll_offset=float(SENSOR_ROLL_OFFSETS_DEG[self.active_sensor]),
                        prior_body_roll=self.roll_est_filtered,
                    )

        return AttitudeEstimate(
            estimated_pitch_deg=float(self.pitch_est_filtered),
            estimated_roll_deg=float(self.roll_est_filtered),
            active_sensor=self.active_sensor,
            did_poll=did_poll,
            sensor_pitch_deg=self.last_sensor_pitch.copy(),
            sensor_roll_deg=self.last_sensor_roll.copy(),
            sensor_valid=self.last_sensor_valid.copy(),
            sensor_distance_raw=self.last_sensor_dist_raw.copy(),
            sensor_distance_effective=self.last_sensor_dist_effective.copy(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file minimal attitude estimator.")
    parser.add_argument("--pitch", type=float, required=True, help="Current true body pitch in deg.")
    parser.add_argument("--roll", type=float, required=True, help="Current true body roll in deg.")
    parser.add_argument("--yaw", type=float, default=5.0, help="Fixed body yaw in deg.")
    parser.add_argument("--steps", type=int, default=1, help="Number of repeated updates to run.")
    parser.add_argument("--dt", type=float, default=None, help="Timestep in seconds per update. Default uses sim_fps.")
    parser.add_argument("--sim-fps", type=float, default=30.0, help="Simulation update rate in Hz.")
    parser.add_argument("--poll-rate", type=float, default=5.0, help="Estimator polling rate in Hz.")
    parser.add_argument("--switch-margin", type=float, default=7.0, help="Horizon switch margin.")
    parser.add_argument("--valid-window", type=int, default=3, help="Recent-valid window in poll frames.")
    parser.add_argument("--pitch-continuity-deg", type=float, default=10.0, help="Shared pitch continuity gate in deg.")
    args = parser.parse_args()

    estimator = MinimalAttitudeEstimator(
        yaw_deg=args.yaw,
        sim_fps=args.sim_fps,
        poll_rate=args.poll_rate,
        switch_margin=args.switch_margin,
        valid_window=args.valid_window,
        pitch_continuity_deg=args.pitch_continuity_deg,
    )

    result = None
    for _ in range(max(1, int(args.steps))):
        result = estimator.update(args.pitch, args.roll, dt_s=args.dt)

    assert result is not None
    print(f"estimated_pitch_deg={result.estimated_pitch_deg:.6f}")
    print(f"estimated_roll_deg={result.estimated_roll_deg:.6f}")
    print(f"active_sensor={result.active_sensor}")
    print(f"did_poll={int(result.did_poll)}")


if __name__ == "__main__":
    main()
