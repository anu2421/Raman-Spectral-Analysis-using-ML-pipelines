"""
Step 3 - Stacked / overlapped spectra plots.
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


def safe_name(s):
    return re.sub(r'[^\w\-]+', '_', s)


def plot_blank_placeholder(title, path, reason):
    """
    Written whenever there isn't yet enough data to draw the real figure.
    Keeps the expected output filename populated with something readable
    instead of erroring or silently producing nothing, so a rerun after
    fixing the folder naming just overwrites this with the real plot.
    """
    plt.figure(figsize=(9, 4))
    plt.axis("off")
    plt.text(0.5, 0.55, "No data yet", ha="center", va="center", fontsize=15, color="0.3")
    plt.text(0.5, 0.4, reason, ha="center", va="center", fontsize=10, color="0.5", wrap=True)
    plt.text(0.5, 0.25, "This file will populate automatically once the matching\n"
                         "folder / naming is found -- just rerun the script.",
             ha="center", va="center", fontsize=9, color="0.55")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Blank placeholder saved ({reason}): {path}")


def _shade_color(base_color, factor):
    """Blends base_color toward white by `factor` (0 = base color, 1 = white).
    Used to give replicates within the same group a light-to-dark family of
    shades, like the reference figure's R1/R2/R3 blues."""
    r, g, b = mcolors.to_rgb(base_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def _detect_shared_peaks(display_traces, common_x, prominence=0.04, distance=30, max_peaks=6):
    """
    display_traces: list of 0-1-rescaled arrays (same length as common_x).
    Finds peaks on the grand mean of those traces so the same dashed
    reference lines appear consistently across every plot in a set.
    """
    grand = np.mean(np.vstack(display_traces), axis=0)
    peak_idx, props = find_peaks(grand, prominence=prominence, distance=distance)
    if len(peak_idx) > max_peaks:
        top = np.argsort(props["prominences"])[-max_peaks:]
        peak_idx = np.sort(peak_idx[top])
    return common_x[peak_idx]


def _draw_peak_markers(peak_positions, top_y):
    for wn in peak_positions:
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, top_y, f"{wn:.0f}", ha="center", va="bottom", fontsize=8)


def _style_stacked_axes(common_x):
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(common_x.min(), common_x.max())
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")


def plot_stacked_replicates(common_x, info, output_dir):
    """One figure per group: R1/R2/R3 mean spectra, offset-stacked, shaded
    from dark (R1) to light (later replicates) within the group's color,
    with dashed shared-peak markers -- matching the reference figure style."""
    fname = os.path.join(output_dir, f"{safe_name(info['display'])}_replicates_stacked.png")
    if info["dir"] is None:
        plot_blank_placeholder(f"{info['display']}: replicate spectra", fname,
                                reason=f"No folder found yet for '{info['display']}' "
                                       f"(looked for keywords {info['keywords']})")
        return
    if not info["replicate_data"]:
        plot_blank_placeholder(f"{info['display']}: replicate spectra", fname,
                                reason=f"Folder found ('{info['folder_name']}') but no usable "
                                       f"R1/R2/R3 spectra inside it")
        return

    reps = sorted(info["replicate_data"].keys(), key=lambda r: int(r[1:]))
    n = len(reps)

    # rescale each replicate's mean spectrum 0-1 for DISPLAY only (offset
    # stacking); the saved CSVs elsewhere still use the real preprocessed values.
    means_disp = {}
    for r in reps:
        y = info["replicate_data"][r].mean(axis=0)
        rng = y.max() - y.min()
        means_disp[r] = (y - y.min()) / rng if rng != 0 else y

    offset_step = 1.05
    peaks = _detect_shared_peaks(list(means_disp.values()), common_x)

    plt.figure(figsize=(9, 2.5 + 1.5 * n))
    for i, r in enumerate(reps):
        shade_factor = (i / max(n - 1, 1)) * 0.55
        color = _shade_color(info["color"], shade_factor)
        n_cells = info["replicate_data"][r].shape[0]
        plt.plot(common_x, means_disp[r] + i * offset_step, color=color, linewidth=1.4,
                  label=f"{r} (n={n_cells} cells)")

    _draw_peak_markers(peaks, top_y=(n - 1) * offset_step + 1.15)
    _style_stacked_axes(common_x)
    plt.title(info["display"], fontsize=16, fontweight="bold")
    plt.legend(fontsize=9, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  -> Stacked replicate plot saved: {fname}")


def plot_group_comparison_stacked(common_x, group_info, comp, output_dir):
    """One figure per comparison set: Control + the two named groups,
    offset-stacked in the same shared style (0-1 rescaled, shared-peak
    dashed markers, bold title, top-left legend)."""
    fname = os.path.join(output_dir, f"{comp['suffix']}_stacked.png")
    keys = comp["groups"]
    present_keys = [k for k in keys if group_info[k]["pooled"] is not None]
    missing_keys = [k for k in keys if k not in present_keys]

    if not present_keys:
        plot_blank_placeholder(f"{comp['title']}: stacked mean spectra", fname,
                                reason="None of this comparison's groups have data yet")
        return

    means_disp = {}
    for k in present_keys:
        y = group_info[k]["pooled"].mean(axis=0)
        rng = y.max() - y.min()
        means_disp[k] = (y - y.min()) / rng if rng != 0 else y

    offset_step = 1.05
    peaks = _detect_shared_peaks(list(means_disp.values()), common_x)
    n = len(present_keys)

    plt.figure(figsize=(9, 2.5 + 1.5 * n))
    for i, k in enumerate(present_keys):
        info = group_info[k]
        n_cells = info["pooled"].shape[0]
        plt.plot(common_x, means_disp[k] + i * offset_step, color=info["color"], linewidth=1.6,
                  label=f"{info['display']} (n={n_cells} cells)")

    _draw_peak_markers(peaks, top_y=(n - 1) * offset_step + 1.15)
    _style_stacked_axes(common_x)
    title = comp["title"]
    if missing_keys:
        missing_names = ", ".join(group_info[k]["display"] for k in missing_keys)
        title += f"\n(missing so far: {missing_names} -- appears automatically once found)"
    plt.title(title, fontsize=15, fontweight="bold")
    plt.legend(fontsize=9, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  -> Comparison stacked plot saved: {fname}")

    # underlying mean-spectra table, for anyone who wants the raw numbers
    table = pd.DataFrame({"RamanShift_cm-1": common_x})
    for k in present_keys:
        table[group_info[k]["display"]] = group_info[k]["pooled"].mean(axis=0)
    table.to_csv(os.path.join(output_dir, f"{comp['suffix']}_mean_spectra.csv"), index=False)


def plot_group_comparison_overlapped(common_x, group_info, comp, output_dir):
    """Same groups as plot_group_comparison_stacked, but drawn OVERLAPPED
    (no vertical offset) on a shared y-axis, so differences in shape and
    relative intensity between conditions are directly comparable instead
    of being separated out for readability."""
    fname = os.path.join(output_dir, f"{comp['suffix']}_overlapped.png")
    keys = comp["groups"]
    present_keys = [k for k in keys if group_info[k]["pooled"] is not None]
    missing_keys = [k for k in keys if k not in present_keys]

    if not present_keys:
        plot_blank_placeholder(f"{comp['title']}: overlapped mean spectra", fname,
                                reason="None of this comparison's groups have data yet")
        return

    means_disp = {}
    for k in present_keys:
        y = group_info[k]["pooled"].mean(axis=0)
        rng = y.max() - y.min()
        means_disp[k] = (y - y.min()) / rng if rng != 0 else y

    peaks = _detect_shared_peaks(list(means_disp.values()), common_x)

    plt.figure(figsize=(9, 5))
    for k in present_keys:
        info = group_info[k]
        n_cells = info["pooled"].shape[0]
        plt.plot(common_x, means_disp[k], color=info["color"], linewidth=1.6,
                  alpha=0.85, label=f"{info['display']} (n={n_cells} cells)")

    _draw_peak_markers(peaks, top_y=1.08)
    _style_stacked_axes(common_x)
    plt.ylim(-0.05, 1.2)
    title = f"{comp['title']} (overlapped)"
    if missing_keys:
        missing_names = ", ".join(group_info[k]["display"] for k in missing_keys)
        title += f"\n(missing so far: {missing_names} -- appears automatically once found)"
    plt.title(title, fontsize=15, fontweight="bold")
    plt.legend(fontsize=9, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  -> Comparison overlapped plot saved: {fname}")


def _rescale01(y):
    rng = y.max() - y.min()
    return (y - y.min()) / rng if rng != 0 else y


def _finish(title, path, common_x, ylabel):
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(common_x.min(), common_x.max())
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> saved: {path}")


def plot_stacked(common_x, traces, title, path, shade_within=False):
    """traces: list of {"label","color","mean_y","n_cells"}; first = bottom."""
    if not traces:
        print(f"  [WARN] nothing to plot for '{title}', skipped.")
        return
    n = len(traces)
    disp = [_rescale01(t["mean_y"]) for t in traces]
    offset = 1.05
    peaks = _detect_shared_peaks(disp, common_x)

    plt.figure(figsize=(9, 2.5 + 1.5 * n))
    for i, (t, d) in enumerate(zip(traces, disp)):
        color = _shade_color(traces[0]["color"], (i / max(n - 1, 1)) * 0.55) if shade_within else t["color"]
        plt.plot(common_x, d + i * offset, color=color, linewidth=1.5,
                 label=f"{t['label']} (n={t['n_cells']} cells)")
    for wn in peaks:
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, (n - 1) * offset + 1.15, f"{wn:.0f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(top=(n - 1) * offset + 1.55)
    # legend below the axes so it never covers peaks or peak labels
    plt.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=2 if n > 2 else n, frameon=False)
    _finish(title, path, common_x, "Intensity (a.u.)")


def plot_overlap(common_x, traces, title, path, norm_label):
    """Overlapped (no offset) mean spectra in their real normalized units,
    with +/- SEM shading, so the effect of AREA vs L2 is visible."""
    if not traces:
        print(f"  [WARN] nothing to plot for '{title}', skipped.")
        return
    plt.figure(figsize=(9, 5))
    for t in traces:
        plt.plot(common_x, t["mean_y"], color=t["color"], linewidth=1.5, alpha=0.9,
                 label=f"{t['label']} (n={t['n_cells']} cells)")
        plt.fill_between(common_x, t["mean_y"] - t["sem_y"], t["mean_y"] + t["sem_y"],
                         color=t["color"], alpha=0.2)
    peaks = _detect_shared_peaks([_rescale01(t["mean_y"]) for t in traces], common_x)
    top = max(float(t["mean_y"].max()) for t in traces)
    for wn in peaks:
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, top * 1.08, f"{wn:.0f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(top=top * 1.2)
    plt.legend(fontsize=9, loc="upper left", frameon=True)
    _finish(title, path, common_x, f"Mean {norm_label}-normalized intensity (a.u.)")
