"""
K562 + A549 stacked / overlap plots only (no ANOVA / PCA / LDA).
Use with EXPERIMENT = "K562_vs_A549_sonicated" or "K562_vs_A549_heat" in config.py.

Plots made for EACH of the 4 methods (AREA/L2 x GLOBAL/1445):
  1. K562 Control - R1 / R2 / R3 stacked
  2. A549 Control - R1 / R2 / R3 stacked
  3. Control K562 vs Control A549 - stacked
  4. Control K562 vs Control A549 - overlap
  5. all groups of both cell lines - stacked

Run:  python stacked_K562_vs_A549.py
"""

import os
import sys
import numpy as np

from config import *
from pipeline.preprocessing import find_group_folder, load_group, normalize_spectrum, _rep_sort
from pipeline.shift_correction import to_grid, shifts_global, shifts_peak1445
from pipeline.plotting import plot_stacked, plot_overlap


def main():
    common_x = np.linspace(ROI_MIN_CM1, ROI_MAX_CM1, COMMON_GRID_POINTS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- load every group from both folders ONCE ---------------------------
    raw = {}   # (dataset, group) -> {rep: [cells]}
    for ds, root in DATASETS.items():
        if not os.path.isdir(root):
            print(f"[ERROR] {ds} folder not found: {root}")
            continue
        for g, kws in GROUP_KEYWORDS.items():
            print(f"\n{'='*70}\n{ds} - {GROUP_DISPLAY[g]}\n{'='*70}")
            gdir, fname = find_group_folder(root, kws)
            if gdir is None:
                print(f"  [WARN] No folder found (keywords {kws}) under {root}.")
                continue
            print(f"  Found folder: '{fname}'")
            cells = load_group(gdir)
            if cells:
                raw[(ds, g)] = cells
    if not raw:
        print("[ERROR] No usable spectra found in either folder.")
        sys.exit(1)

    for norm, shift_method in METHODS:
        tag = f"{NORM_LABELS[norm]}_{SHIFT_LABELS[shift_method]}"
        mlabel = f"{NORM_LABELS[norm]} + {SHIFT_LABELS[shift_method]}"
        out = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(out, exist_ok=True)
        print(f"\n{'#'*70}\nMETHOD: {mlabel}   ->  {out}\n{'#'*70}")

        data = {k: {r: [{"x": c["x"], "y": normalize_spectrum(c["y"], norm)} for c in cells]
                    for r, cells in reps.items()} for k, reps in raw.items()}
        shifts = shifts_global(data, common_x) if shift_method == "global" else shifts_peak1445(data, common_x)
        for k in data:
            allv = np.concatenate(list(shifts[k].values()))
            print(f"  {k[0]} {GROUP_DISPLAY[k[1]]}: applied shift mean {allv.mean():+.2f} cm-1 "
                  f"(range {allv.min():+.2f} to {allv.max():+.2f})")

        # corrected, gridded spectra
        rep_arrays = {k: {r: np.vstack([to_grid(c, common_x, shifts[k][r][i]) for i, c in enumerate(cells)])
                          for r, cells in reps.items()} for k, reps in data.items()}
        pooled = {k: np.vstack([rep_arrays[k][r] for r in _rep_sort(reps)]) for k, reps in rep_arrays.items()}

        def trace(k):
            arr = pooled[k]
            return {"label": f"{GROUP_DISPLAY[k[1]]} {k[0]}", "color": COLORS[k],
                    "mean_y": arr.mean(axis=0),
                    "sem_y": arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0 * arr[0],
                    "n_cells": arr.shape[0]}

        # 1 + 2. each Control: R1 / R2 / R3 stacked
        for ds in DATASETS:
            k = (ds, "control")
            if k not in rep_arrays:
                continue
            reps = _rep_sort(rep_arrays[k])
            traces = [{"label": r, "color": COLORS[k], "mean_y": rep_arrays[k][r].mean(axis=0),
                       "n_cells": rep_arrays[k][r].shape[0]} for r in reps]
            plot_stacked(common_x, traces, f"Control {ds}: R1 / R2 / R3 stacked\n[{mlabel}]",
                         os.path.join(out, f"Control_{ds}_replicates_stacked.png"), shade_within=True)

        # 3 + 4. both Controls: stacked and overlap
        ctrl_keys = [(ds, "control") for ds in DATASETS if (ds, "control") in pooled]
        ctrl_traces = [trace(k) for k in ctrl_keys]
        plot_stacked(common_x, ctrl_traces, f"Control K562 vs Control A549 (stacked)\n[{mlabel}]",
                     os.path.join(out, "Controls_K562_vs_A549_stacked.png"))
        plot_overlap(common_x, ctrl_traces, f"Control K562 vs Control A549 (overlap)\n[{mlabel}]",
                     os.path.join(out, "Controls_K562_vs_A549_overlap.png"), NORM_LABELS[norm])

        # 5. all groups: K562 then A549 (order / title set in config.py)
        plot_stacked(common_x, [trace(k) for k in CROSS_ALL_ORDER if k in pooled],
                     f"{CROSS_ALL_TITLE}: K562 and A549 (stacked)\n[{mlabel}]",
                     os.path.join(out, f"{CROSS_ALL_FILE}_K562_A549_stacked.png"))

    print(f"\nAll plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
