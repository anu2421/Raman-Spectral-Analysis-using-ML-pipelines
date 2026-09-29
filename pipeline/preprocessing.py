"""
Step 1 - Reading the Raman CSVs, preprocessing and folder discovery.

cosmic spike removal -> ROI (600-1800 cm-1) -> Savitzky-Golay smoothing ->
ALS baseline correction -> normalization (area / L2).
Handles R1/R2/R3 subfolders AND "R1_..." flat files.
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


def read_raman_csv(path):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
        header_idx = None
        for i, line in enumerate(lines):
            if "RamanShift" in line and "Intensity" in line:
                header_idx = i
                break
        if header_idx is None:
            print(f"    [WARN] Could not find data header in {path}, skipping.")
            return None
        df = pd.read_csv(path, skiprows=header_idx, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        shift_col = [c for c in df.columns if "RamanShift" in c][0]
        intens_col = [c for c in df.columns if "Intensity" in c][0]
        shift = pd.to_numeric(df[shift_col], errors="coerce").to_numpy()
        intens = pd.to_numeric(df[intens_col], errors="coerce").to_numpy()
        mask = ~np.isnan(shift) & ~np.isnan(intens)
        shift, intens = shift[mask], intens[mask]
        order = np.argsort(shift)
        return shift[order], intens[order].astype(float)
    except Exception as e:
        print(f"    [WARN] Failed to read {path}: {e}")
        return None


def remove_cosmic_spikes(y, window=5, threshold=7.0):
    if window % 2 == 0:
        window += 1
    y = y.copy()
    med = medfilt(y, kernel_size=window)
    residual = y - med
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-9
    spike_mask = np.abs(residual) > threshold * mad * 1.4826
    y[spike_mask] = med[spike_mask]
    return y


def apply_roi(x, y, roi_min, roi_max):
    mask = (x >= roi_min) & (x <= roi_max)
    return x[mask], y[mask]


def smooth_savgol(y, window=11, polyorder=3):
    window = min(window, len(y) - (1 - len(y) % 2))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if window >= len(y):
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    D = D.tocsc()
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = np.zeros(L)
    for _ in range(niter):
        W.setdiag(w)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _trapz(y):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y)


def normalize_spectrum(y, method):
    if method == "area":
        area = _trapz(np.abs(y))
        return y / area if area != 0 else y
    if method == "l2":
        norm = np.linalg.norm(y)
        return y / norm if norm != 0 else y
    raise ValueError(f"Unknown normalization method: {method}")


def preprocess_until_baseline(x, y, label=""):
    y = remove_cosmic_spikes(y, window=SPIKE_WINDOW, threshold=SPIKE_THRESHOLD)
    x, y = apply_roi(x, y, ROI_MIN_CM1, ROI_MAX_CM1)
    if len(x) < max(SG_POLYORDER + 2, 5):
        print(f"    [WARN] {label}: too few points after ROI cut, skipping.")
        return None
    y = smooth_savgol(y, window=SG_WINDOW, polyorder=SG_POLYORDER)
    y = y - als_baseline(y, lam=ALS_LAMBDA, p=ALS_P, niter=ALS_NITER)
    return x, y


def _normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def find_group_folder(root, keywords):
    try:
        entries = sorted(os.listdir(root))
    except OSError as e:
        print(f"[ERROR] Could not list {root}: {e}")
        return None, None
    for entry in entries:
        full = os.path.join(root, entry)
        if os.path.isdir(full) and all(kw in _normalize_name(entry) for kw in keywords):
            return full, entry
    return None, None


def replicate_paths(group_dir):
    reps = {}
    for entry in sorted(os.listdir(group_dir)):
        full = os.path.join(group_dir, entry)
        m = re.match(REPLICATE_REGEX, entry.strip())
        if os.path.isdir(full) and m:
            reps[f"R{int(m.group(1))}"] = sorted(glob.glob(os.path.join(full, "**", "*.csv"), recursive=True))
    if reps:
        return reps, "subfolder"
    for p in sorted(glob.glob(os.path.join(group_dir, "**", "*.csv"), recursive=True)):
        m = re.match(FILENAME_REPLICATE_REGEX, os.path.basename(p))
        if m:
            reps.setdefault(f"R{int(m.group(1))}", []).append(p)
    return reps, "filename prefix"


def _rep_sort(reps):
    return sorted(reps, key=lambda r: int(r[1:]))


def load_group(group_dir, merge=None):
    """dict 'R1' -> list of {"x","y"} (baseline-corrected, not yet normalized).
    merge: optional {"R1": "R2"} -> files of R1 are counted as R2."""
    paths_by_rep, layout = replicate_paths(group_dir)
    for src, dst in (merge or {}).items():
        if src in paths_by_rep:
            print(f"    {src}: {len(paths_by_rep[src])} CSV(s) relabelled as {dst}")
            paths_by_rep.setdefault(dst, []).extend(paths_by_rep.pop(src))
    out = {}
    for rep in _rep_sort(paths_by_rep):
        cells = []
        for p in paths_by_rep[rep]:
            res = read_raman_csv(p)
            if res is None:
                continue
            pre = preprocess_until_baseline(res[0], res[1], f"{rep}/{os.path.basename(p)}")
            if pre is not None:
                cells.append({"x": pre[0], "y": pre[1]})
        if cells:
            out[rep] = cells
            print(f"    {rep}: {len(cells)} usable cell(s) [{layout}]")
    return out


# ---- used by norm_shift_comparison.py ----

def subtract_baseline(y, lam=1e5, p=0.01, niter=10):
    baseline = als_baseline(y, lam=lam, p=p, niter=niter)
    return y - baseline


def find_replicate_folders(group_dir):
    """Returns dict 'R1' -> path, 'R2' -> path, ... sorted by replicate number.
    Empty dict if the group uses flat files instead of replicate subfolders."""
    reps = {}
    for entry in sorted(os.listdir(group_dir)):
        full = os.path.join(group_dir, entry)
        if not os.path.isdir(full):
            continue
        m = re.match(REPLICATE_REGEX, entry.strip())
        if m:
            reps[f"R{int(m.group(1))}"] = full
    return dict(sorted(reps.items(), key=lambda kv: int(kv[0][1:])))


def group_files_by_replicate_prefix(paths):
    """Groups a flat list of CSV paths by an R<number> prefix in the filename."""
    groups = {}
    unmatched = []
    for p in paths:
        m = re.match(FILENAME_REPLICATE_REGEX, os.path.basename(p))
        if m:
            groups.setdefault(f"R{int(m.group(1))}", []).append(p)
        else:
            unmatched.append(p)
    return dict(sorted(groups.items(), key=lambda kv: int(kv[0][1:]))), unmatched
