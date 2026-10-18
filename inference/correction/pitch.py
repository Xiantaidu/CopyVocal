from __future__ import annotations

import numpy as np


def sample_pitch(pitch: dict, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate log-F0 inside voiced runs; never bridge an unvoiced gap.

    UV comes from the nearest original frame independently of F0."""
    query = np.asarray(query_times, dtype=np.float64)
    times = np.asarray(pitch.get("times", []), dtype=np.float64)
    f0 = np.asarray(pitch.get("f0_hz", []), dtype=np.float64)
    voiced = np.asarray(pitch.get("voiced", f0 > 0), dtype=bool).copy()
    output = np.zeros(query.shape, dtype=np.float32)
    mask = np.zeros(query.shape, dtype=bool)
    if not len(times):
        if len(f0) or len(voiced):
            raise ValueError("Pitch timestamps are required.")
        return output, mask
    if times.ndim != 1 or f0.shape != times.shape or voiced.shape != times.shape:
        raise ValueError("Pitch times, f0_hz, and voiced must be equally sized one-dimensional arrays.")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Pitch times must be finite and strictly increasing.")
    voiced &= np.isfinite(f0) & (f0 > 0)
    if len(times) == 1:
        selected = np.isfinite(query) & np.isclose(query, times[0], atol=1e-7, rtol=0)
        if voiced[0]:
            output[selected], mask[selected] = f0[0], True
        return output, mask
    right = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    left = right - 1
    nearest = np.where(np.abs(query - times[left]) <= np.abs(times[right] - query), left, right)
    half_left, half_right = (times[1] - times[0]) / 2, (times[-1] - times[-2]) / 2
    valid = np.isfinite(query) & (query >= times[0] - half_left) & (query < times[-1] + half_right)
    mask = valid & voiced[nearest]
    # Restrict interpolation to the voiced run containing the nearest frame.
    edges = np.diff(np.pad(voiced.astype(np.int8), (1, 1)))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        selected = mask & (nearest >= start) & (nearest < stop)
        if selected.any():
            output[selected] = np.exp(np.interp(query[selected], times[start:stop], np.log(f0[start:stop])))
    return output, mask

