"""
Full analysis for the experiment selected in config.py (EXPERIMENT).

For each cell line / group in the config:
  Step 1  load + preprocess every cell            (pipeline/preprocessing.py)
  Step 2  normalize + Raman shift correction      (pipeline/shift_correction.py)
  Step 3  stacked / overlap plots                 (pipeline/plotting.py)
  Step 4  cluster-permutation ANOVA + peaks       (pipeline/anova.py)
  Step 5  PCA, LDA, confusion matrices (CV, LORO) (pipeline/classification.py)
  Step 6  peak position shifts                    (pipeline/peak_shift.py)
and a summary table of all comparisons.

Run:  python main.py
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from pipeline.preprocessing import find_group_folder, load_group, normalize_spectrum, _rep_sort
from pipeline.shift_correction import to_grid, shifts_global, shifts_peak1445
from pipeline.plotting import (safe_name, plot_stacked, plot_overlap, plot_group_comparison_stacked,
                               plot_group_comparison_overlapped)
from pipeline.anova import run_anova_for_comparison, HAS_GAMES_HOWELL
from pipeline.peak_shift import run_peak_shift_for_comparison


def _wrapped_title(label, *args, **kwargs):
    """Display-only: wraps long title lines so they are not cut off.
    Calls Axes.set_title directly (what plt.title does internally), so it
    can never call itself, even when re-run in the same notebook kernel."""
    if isinstance(label, str):
        label = "\n".join(textwrap.fill(line, 45) if "$" not in line else line
                          for line in label.split("\n"))
    return plt.gca().set_title(label, *args, **kwargs)


def _read_accuracy(path):
    try:
        df = pd.read_csv(path, index_col=0)
        return float(df.loc["accuracy", "f1-score"]), float(df.loc["macro avg", "f1-score"])
    except Exception:
        return np.nan, np.nan


def run_all(common_x):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- load every group in GROUP_KEYWORDS ONCE -------------------------
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
            cells = load_group(gdir, REPLICATE_MERGE.get(g, {}))
            if cells:
                raw[(ds, g)] = cells
    if not raw:
        print("[ERROR] No usable spectra found.")
        sys.exit(1)

    summary = []
    for norm, shift_method in METHODS:
        tag = f"{NORM_LABELS[norm]}_{SHIFT_LABELS[shift_method]}"
        mlabel = f"{NORM_LABELS[norm]} + {SHIFT_LABELS[shift_method]}"
        out = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(out, exist_ok=True)
        print(f"\n{'#'*70}\nMETHOD: {mlabel}   ->  {out}\n{'#'*70}")

        # normalize, shift-correct, put on the common grid
        data = {k: {r: [{"x": c["x"], "y": normalize_spectrum(c["y"], norm)} for c in cells]
                    for r, cells in reps.items()} for k, reps in raw.items()}
        shifts = shifts_global(data, common_x) if shift_method == "global" else shifts_peak1445(data, common_x)
        log_rows = []
        for k in data:
            for r in _rep_sort(data[k]):
                for s in shifts[k][r]:
                    log_rows.append({"cell_line": k[0], "group": GROUP_DISPLAY[k[1]], "replicate": r,
                                     "applied_shift_cm-1": float(s)})
            allv = np.concatenate(list(shifts[k].values()))
            print(f"  {k[0]} {GROUP_DISPLAY[k[1]]}: applied shift mean {allv.mean():+.2f} cm-1 "
                  f"(range {allv.min():+.2f} to {allv.max():+.2f})")
        pd.DataFrame(log_rows).to_csv(os.path.join(out, "applied_shift_log.csv"), index=False)

        rep_arrays = {k: {r: np.vstack([to_grid(c, common_x, shifts[k][r][i]) for i, c in enumerate(reps[r])])
                          for r in _rep_sort(reps)} for k, reps in data.items()}
        pooled = {k: np.vstack([rep_arrays[k][r] for r in _rep_sort(rep_arrays[k])]) for k in rep_arrays}

        # ---------------- stacked / overlap plots ----------------
        stk = os.path.join(out, "stacked_plots")
        os.makedirs(stk, exist_ok=True)
        print(f"\n-- stacked / overlap plots --")

        def trace(k):
            arr = pooled[k]
            return {"label": f"{GROUP_DISPLAY[k[1]]} {k[0]}", "color": COLORS[k],
                    "mean_y": arr.mean(axis=0),
                    "sem_y": arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0 * arr[0],
                    "n_cells": arr.shape[0]}

        for ds in DATASETS:
            # each group: R1 / R2 / R3 stacked
            for g in GROUP_KEYWORDS:
                k = (ds, g)
                if k not in rep_arrays:
                    continue
                traces = [{"label": r, "color": COLORS[k], "mean_y": rep_arrays[k][r].mean(axis=0),
                           "n_cells": rep_arrays[k][r].shape[0]} for r in _rep_sort(rep_arrays[k])]
                plot_stacked(common_x, traces, f"{GROUP_DISPLAY[g]} {ds}: R1 / R2 / R3 stacked\n[{mlabel}]",
                             os.path.join(stk, f"{safe_name(GROUP_DISPLAY[g])}_{ds}_replicates_stacked.png"),
                             shade_within=True)
            # all groups of this cell line: stacked and overlap
            allg = [trace((ds, g)) for g in GROUP_KEYWORDS if (ds, g) in pooled]
            plot_stacked(common_x, allg, f"{ALL_GROUPS_TITLE} {ds} (stacked)\n[{mlabel}]",
                         os.path.join(stk, f"{ALL_GROUPS_FILE}_{ds}_stacked.png"))
            plot_overlap(common_x, allg, f"{ALL_GROUPS_TITLE} {ds} (overlap)\n[{mlabel}]",
                         os.path.join(stk, f"{ALL_GROUPS_FILE}_{ds}_overlap.png"), NORM_LABELS[norm])

        # ---------------- group_info in the format the analysis functions use ----
        group_info = {}
        for ds in DATASETS:
            for g in GROUP_KEYWORDS:
                gid = f"{ds}_{g}"
                k = (ds, g)
                present = k in pooled
                group_info[gid] = {"display": f"{GROUP_DISPLAY[g]} {ds}", "color": COLORS[k],
                                   "keywords": GROUP_KEYWORDS[g], "dir": None, "folder_name": None,
                                   "replicate_data": rep_arrays[k] if present else {},
                                   "pooled": pooled[k] if present else None}

        # ---------------- full analysis per comparison ----------------
        for comp in COMPARISONS:
            cdir = os.path.join(out, comp["suffix"])
            os.makedirs(cdir, exist_ok=True)
            comp_v = {**comp, "title": f"{comp['title']} [{mlabel}]"}
            print(f"\n{'='*70}\n{comp_v['title']}\n{'='*70}")
            plot_group_comparison_stacked(common_x, group_info, comp_v, cdir)
            plot_group_comparison_overlapped(common_x, group_info, comp_v, cdir)
            run_anova_for_comparison(common_x, group_info, comp_v, cdir)
            run_peak_shift_for_comparison(common_x, group_info, comp_v, cdir)

            s = comp["suffix"]
            cv_acc, cv_f1 = _read_accuracy(os.path.join(cdir, f"{s}_classification_report.csv"))
            lo_acc, lo_f1 = _read_accuracy(os.path.join(cdir, f"{s}_classification_report_LORO.csv"))
            try:
                cl = pd.read_csv(os.path.join(cdir, f"{s}_significant_clusters.csv"))
                n_cl = int(cl["significant"].astype(str).str.lower().eq("true").sum())
            except Exception:
                n_cl = 0
            try:
                st = pd.read_csv(os.path.join(cdir, f"{s}_anova_by_wavenumber.csv"))
                pct = 100 * float(st["in_significant_cluster"].astype(str).str.lower().eq("true").mean())
            except Exception:
                pct = np.nan
            try:
                pk = pd.read_csv(os.path.join(cdir, f"{s}_significant_peaks.csv"))
                peaks = ", ".join(f"{v:.0f}" for v in pk["wavenumber_cm-1"])
            except Exception:
                peaks = ""
            n_groups = len([g for g in comp["groups"] if group_info[g]["pooled"] is not None])
            summary.append({"method": mlabel, "comparison": comp["title"],
                            "n_groups": n_groups, "chance_accuracy": round(1 / n_groups, 3) if n_groups else np.nan,
                            "lda_accuracy_cell_level_cv": cv_acc, "lda_macro_f1_cell_level_cv": cv_f1,
                            "lda_accuracy_LORO": lo_acc, "lda_macro_f1_LORO": lo_f1,
                            "n_significant_clusters": n_cl, "pct_spectrum_in_sig_clusters": pct,
                            "effect_gated_peaks_cm-1": peaks})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUTPUT_DIR, "summary_all_comparisons.csv"), index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n", sdf.to_string(index=False))
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


def main():
    common_x = np.linspace(ROI_MIN_CM1, ROI_MAX_CM1, COMMON_GRID_POINTS)
    if not HAS_GAMES_HOWELL:
        print("[NOTE] statsmodels not found -> Games-Howell falls back to Bonferroni-corrected Welch t-tests.")
    previous_title = plt.title
    plt.title = _wrapped_title
    try:
        run_all(common_x)
    finally:
        plt.title = previous_title


if __name__ == "__main__":
    main()
