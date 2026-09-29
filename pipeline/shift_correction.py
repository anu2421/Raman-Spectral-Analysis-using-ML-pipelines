"""
Step 2 - Raman shift correction (only the x-axis moves, intensity is never changed).

peak1445 : each cell's ~1445 cm-1 peak is moved to exactly 1445 cm-1.
global   : each replicate mean is slid in 1 cm-1 steps to the step that best
           matches the fixed reference (Control mean), applied to all its cells.
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter, medfilt, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import f as f_dist, t as t_dist

from config import *


def to_grid(cell, common_x, shift=0.0):
    """New x = original x + shift (intensity unchanged), then onto common grid."""
    return np.interp(common_x, cell["x"] + shift, cell["y"])


def locate_peak(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    if m.sum() < 5:
        return None
    xs, ys = x[m], y[m]
    idx, _ = find_peaks(ys)
    if len(idx) == 0:
        return None
    i = int(idx[np.argmax(ys[idx])])
    a, b = max(0, i - 2), min(len(xs), i + 3)
    if b - a >= 3:
        c = np.polyfit(xs[a:b] - xs[i], ys[a:b], 2)
        if c[0] < 0:
            v = -c[1] / (2 * c[0])
            if xs[a] - xs[i] <= v <= xs[b - 1] - xs[i]:
                return float(xs[i] + v)
    return float(xs[i])


def _shift_from_1445(pos):
    if pos is None:
        return 0.0
    s = PEAK1445_TARGET_CM1 - pos
    return float(s) if abs(s) <= MAX_ALLOWED_SHIFT_CM1 else 0.0


def shifts_peak1445(data, common_x):
    shifts = {}
    for key, reps in data.items():
        shifts[key] = {}
        for rep, cells in reps.items():
            if PEAK1445_LEVEL == "replicate":
                mean = np.mean([to_grid(c, common_x) for c in cells], axis=0)
                s = _shift_from_1445(locate_peak(common_x, mean, PEAK1445_SEARCH_MIN_CM1, PEAK1445_SEARCH_MAX_CM1))
                shifts[key][rep] = np.full(len(cells), s)
            else:
                shifts[key][rep] = np.array([
                    _shift_from_1445(locate_peak(c["x"], c["y"], PEAK1445_SEARCH_MIN_CM1, PEAK1445_SEARCH_MAX_CM1))
                    for c in cells])
    return shifts


def _best_global_shift(x, y, reference, common_x, match_mask):
    candidates = np.arange(-GLOBAL_SHIFT_MAX_CM1, GLOBAL_SHIFT_MAX_CM1 + 1e-9, GLOBAL_SHIFT_STEP_CM1)
    ref = reference[match_mask]
    scores = []
    for s in candidates:
        moved = np.interp(common_x, x + s, y)[match_mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(moved, ref)[0, 1]
        scores.append(r if np.isfinite(r) else -np.inf)
    return float(candidates[int(np.argmax(scores))])


def shifts_global(data, common_x):
    match_mask = (common_x >= GLOBAL_SHIFT_MATCH_MIN_CM1) & (common_x <= GLOBAL_SHIFT_MATCH_MAX_CM1)
    shifts = {k: {r: np.zeros(len(c)) for r, c in reps.items()} for k, reps in data.items()}

    def reference_for(dataset, shifts):
        if GLOBAL_SHIFT_REFERENCE == "own_control":
            ref_keys = [(dataset, "control")]
        else:
            ref_keys = [k for k in data if k[1] == "control"]
        ref_keys = [k for k in ref_keys if k in data] or list(data.keys())
        return np.mean([to_grid(c, common_x, shifts[k][r][i])
                        for k in ref_keys for r, cells in data[k].items()
                        for i, c in enumerate(cells)], axis=0)

    for _ in range(max(1, GLOBAL_SHIFT_ITERATIONS)):
        refs = {ds: reference_for(ds, shifts) for ds in {k[0] for k in data}}
        for key, reps in data.items():
            reference = refs[key[0]]
            for rep, cells in reps.items():
                if GLOBAL_SHIFT_LEVEL == "replicate":
                    mean = np.mean([to_grid(c, common_x) for c in cells], axis=0)
                    shifts[key][rep][:] = _best_global_shift(common_x, mean, reference, common_x, match_mask)
                else:
                    for i, c in enumerate(cells):
                        shifts[key][rep][i] = _best_global_shift(c["x"], c["y"], reference, common_x, match_mask)
    return shifts
