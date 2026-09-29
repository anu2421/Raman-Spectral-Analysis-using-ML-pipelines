"""
Step 6 - Peak position shift detection (bootstrapped).
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
from pipeline.plotting import plot_blank_placeholder


def _find_peaks_in_mean(y_mean, prominence, distance):
    rng = y_mean.max() - y_mean.min()
    y_disp = (y_mean - y_mean.min()) / rng if rng != 0 else y_mean
    idx, _ = find_peaks(y_disp, prominence=prominence, distance=distance)
    return idx


def _refine_peak_position(y, approx_idx, window):
    lo = max(0, approx_idx - window)
    hi = min(len(y), approx_idx + window + 1)
    local = y[lo:hi]
    return lo + int(np.argmax(local))


def _bootstrap_peak_positions(cell_matrix, common_x, approx_idx, window, n_boot, seed):
    rng = np.random.default_rng(seed)
    n_cells = cell_matrix.shape[0]
    positions = np.empty(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_cells, size=n_cells)
        boot_mean = cell_matrix[sample_idx].mean(axis=0)
        refined_idx = _refine_peak_position(boot_mean, approx_idx, window)
        positions[b] = common_x[refined_idx]
    return positions


def _match_peaks_across_groups(peaks_per_group, common_x, tolerance_cm1):
    labels = list(peaks_per_group.keys())
    anchor = max(labels, key=lambda l: len(peaks_per_group[l]))
    remaining = {l: list(peaks_per_group[l]) for l in labels}
    groups = []
    for anchor_idx in list(remaining[anchor]):
        group = {anchor: anchor_idx}
        anchor_wn = common_x[anchor_idx]
        for l in labels:
            if l == anchor or not remaining[l]:
                continue
            wns = common_x[np.array(remaining[l])]
            diffs = np.abs(wns - anchor_wn)
            best = int(np.argmin(diffs))
            if diffs[best] <= tolerance_cm1:
                group[l] = remaining[l][best]
        groups.append(group)
    remaining[anchor] = []
    leftover = {l: idxs for l, idxs in remaining.items() if idxs}
    return groups, leftover


def detect_group_peak_shifts(common_x, group_arrays, display_names, prominence, distance,
                              match_tol_cm1, refine_window, n_boot, seed):
    """
    group_arrays: dict internal_key -> (n_cells, P) array.
    display_names: dict internal_key -> human-readable name.
    Reference condition for each matched peak's shift is 'control' when
    present in that peak's group, otherwise the first key alphabetically.
    """
    labels = list(group_arrays.keys())
    peaks_per_group = {l: _find_peaks_in_mean(group_arrays[l].mean(axis=0), prominence, distance)
                        for l in labels}
    groups, leftover = _match_peaks_across_groups(peaks_per_group, common_x, match_tol_cm1)

    records = []
    for gi, group in enumerate(groups):
        if len(group) < 2:
            continue
        boot_positions = {
            l: _bootstrap_peak_positions(group_arrays[l], common_x, idx, refine_window, n_boot, seed + gi)
            for l, idx in group.items()
        }
        ref = "control" if "control" in group else sorted(group.keys())[0]
        for l in sorted(group.keys()):
            pos = boot_positions[l]
            row = {
                "peak_group": gi,
                "condition": display_names[l],
                "reference_condition": display_names[ref],
                "position_mean_cm-1": round(float(pos.mean()), 2),
                "position_sd_cm-1": round(float(pos.std(ddof=1)), 2),
            }
            if l != ref:
                diff = boot_positions[l] - boot_positions[ref]
                ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
                row["shift_vs_reference_cm-1"] = round(float(diff.mean()), 2)
                row["shift_ci_low"] = round(float(ci_low), 2)
                row["shift_ci_high"] = round(float(ci_high), 2)
                row["significant_shift"] = bool(ci_low > 0 or ci_high < 0)
            records.append(row)

    shift_cols = ["peak_group", "condition", "reference_condition", "position_mean_cm-1",
                  "position_sd_cm-1", "shift_vs_reference_cm-1", "shift_ci_low",
                  "shift_ci_high", "significant_shift"]
    shift_df = pd.DataFrame(records, columns=shift_cols) if records else pd.DataFrame(columns=shift_cols)

    leftover_records = [
        {"condition": display_names[l], "position_cm-1": round(float(common_x[idx]), 1),
         "note": "peak found only in this condition -- no match within tolerance elsewhere"}
        for l, idxs in leftover.items() for idx in idxs
    ]
    leftover_df = pd.DataFrame(leftover_records)
    return shift_df, leftover_df


def plot_group_shift_chart(shift_df, title, path, color_map):
    if shift_df.empty:
        plot_blank_placeholder(title, path, reason="No matched peaks across groups to chart")
        return

    group_order = shift_df.groupby("peak_group")["position_mean_cm-1"].mean().sort_values().index.tolist()
    fig_height = max(3, 0.6 * len(group_order) + 1.5)
    plt.figure(figsize=(9, fig_height))

    y_labels = []
    for row_i, gi in enumerate(group_order):
        g = shift_df[shift_df["peak_group"] == gi].sort_values("position_mean_cm-1")
        y = row_i
        xs = g["position_mean_cm-1"].to_numpy()
        xerr = g["position_sd_cm-1"].to_numpy()
        plt.plot([xs.min(), xs.max()], [y, y], color="0.75", linewidth=1, zorder=1)
        for x_val, err_val, cond, sig in zip(xs, xerr, g["condition"], g["significant_shift"]):
            is_sig = (sig == True)  # noqa: E712 -- NaN (reference rows) must compare False
            plt.errorbar(x_val, y, xerr=err_val, fmt="o", color=color_map.get(cond, "gray"),
                         markeredgecolor="red" if is_sig else "none",
                         markeredgewidth=2 if is_sig else 0, markersize=9, capsize=3, zorder=3)
        y_labels.append(f"~{xs.mean():.0f} cm$^{{-1}}$")

    plt.yticks(range(len(group_order)), y_labels)
    plt.xlabel("Raman shift, peak position (cm$^{-1}$)")
    plt.ylabel("Matched peak (reference-condition wavenumber)")
    plt.title(f"{title}\n(red outline = 95% CI on the shift excludes 0 cm$^{{-1}}$)")

    present = shift_df["condition"].unique()
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=9, label=name)
               for name, col in color_map.items() if name in present]
    if handles:
        plt.legend(handles=handles, loc="best", fontsize=9)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Peak position shift chart saved: {path}")


def run_peak_shift_for_comparison(common_x, group_info, comp, output_dir):
    keys = comp["groups"]
    present_keys = [k for k in keys if group_info[k]["pooled"] is not None]
    chart_path = os.path.join(output_dir, f"{comp['suffix']}_peak_position_shift_chart.png")

    if len(present_keys) < 2:
        plot_blank_placeholder(f"{comp['title']}: peak position shifts", chart_path,
                                reason=f"Need >=2 groups with data to detect a shift "
                                       f"(currently have {len(present_keys)})")
        return

    group_arrays = {k: group_info[k]["pooled"] for k in present_keys}
    display_names = {k: group_info[k]["display"] for k in present_keys}
    color_map = {group_info[k]["display"]: group_info[k]["color"] for k in present_keys}

    shift_df, leftover_df = detect_group_peak_shifts(
        common_x, group_arrays, display_names, PEAK_PROMINENCE, PEAK_DISTANCE,
        PEAK_SHIFT_MATCH_TOLERANCE_CM1, PEAK_SHIFT_REFINE_WINDOW,
        PEAK_SHIFT_BOOTSTRAP_N, PEAK_SHIFT_BOOTSTRAP_SEED
    )
    shift_csv = os.path.join(output_dir, f"{comp['suffix']}_peak_position_shifts.csv")
    shift_df.to_csv(shift_csv, index=False)
    n_sig = int(shift_df["significant_shift"].sum()) if "significant_shift" in shift_df else 0
    print(f"  -> Peak position shift table saved: {shift_csv} "
          f"({n_sig} condition-pair shift(s) with 95% CI excluding 0 cm-1)")

    plot_group_shift_chart(shift_df, f"{comp['title']}: peak position by condition", chart_path, color_map)

    if not leftover_df.empty:
        leftover_csv = os.path.join(output_dir, f"{comp['suffix']}_condition_specific_peaks.csv")
        leftover_df.to_csv(leftover_csv, index=False)
        print(f"  -> {len(leftover_df)} condition-specific peak(s) saved: {leftover_csv}")

    missing_keys = [k for k in keys if k not in present_keys]
    if missing_keys:
        missing_names = ", ".join(group_info[k]["display"] for k in missing_keys)
        print(f"  [INFO] {missing_names} not yet included here -- will appear automatically "
              f"once its folder is found.")
