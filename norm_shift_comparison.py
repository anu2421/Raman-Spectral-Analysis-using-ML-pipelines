"""
Normalization x shift-correction comparison.

For each analysis (heat-inactivated, sonicated) the data is built 6 ways:
    normalization : L2, AREA
    shift         : none, 1445 cm-1 peak alignment, global Raman shift
and every version is run through the same stacked plots / ANOVA / PCA / LDA /
confusion matrices / peak-shift analysis, each into its own subfolder.
Then COMPARE_* figures and a summary table compare the 6 versions.

Cell line: NORM_SHIFT_CELL_LINE in config.py.
Run:  python norm_shift_comparison.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from config import *
from pipeline.preprocessing import (read_raman_csv, remove_cosmic_spikes, apply_roi, smooth_savgol,
                                    subtract_baseline, normalize_spectrum, find_group_folder,
                                    find_replicate_folders, group_files_by_replicate_prefix)
from pipeline.plotting import (plot_stacked_replicates, plot_group_comparison_stacked,
                               plot_group_comparison_overlapped)
from pipeline.anova import run_anova_for_comparison, HAS_GAMES_HOWELL
from pipeline.peak_shift import run_peak_shift_for_comparison
from pipeline.classification import HAS_SKLEARN_DA
if HAS_SKLEARN_DA:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ########################################################################
# ########################################################################
# ##                                                                    ##
# ##   ADD-ON: HEAT-INACTIVATED  +  SONICATED (both run in one go)      ##
# ##   NORMALIZATION (L2 vs AREA)  x  SHIFT CORRECTION                  ##
# ##   (none / 1445 cm-1 peak alignment / global Raman shift)           ##
# ##                                                                    ##
# ##   Everything ABOVE this banner is the original pipeline, unchanged.##
# ##   For each analysis in ANALYSIS_RUNS the add-on (a) loads its     ##
# ##   groups once,                                                     ##
# ##   (b) builds 6 versions of the data (2 normalizations x 3 shift    ##
# ##   modes), and (c) feeds each version through the SAME original     ##
# ##   plotting / ANOVA / PCA / LDA / confusion-matrix / peak-shift     ##
# ##   functions, each into its own subfolder. Then it makes            ##
# ##   side-by-side comparison figures across the 6 versions.           ##
# ##                                                                    ##
# ########################################################################
# ########################################################################

import html as _html
import json as _json

# numpy >= 2.0 renamed np.trapz -> np.trapezoid. normalize_spectrum(...,
# "area") above uses np.trapz, so this keeps area normalization working on
# any numpy version without touching that function. Same math either way.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


# ========================================================================
# ======================= ADD-ON USER SETTINGS ==========================
# ========================================================================
# Both analyses are run, one after the other. Each gets its own output
# folder (one subfolder per variant + the COMPARE_* summary figures).
#   * Heat-inactivated: Control vs E.coli heat vs S.aureus heat
#   * Sonicated:        Control vs S.aureus sonication
#     (to also include E.coli sonication, add "ecoli_sonication" to its
#      "groups" list below)
ANALYSIS_RUNS = [
    {"output_dir": os.path.join("results", f"{NORM_SHIFT_CELL_LINE}_heat_inactivated_norm_shift_comparison"),
     "title": "Heat-Inactivated vs Control",
     "suffix": "heat_inactivated_vs_control",
     "groups": ["control", "ecoli_heat", "saureus_heat"]},
    {"output_dir": os.path.join("results", f"{NORM_SHIFT_CELL_LINE}_saureus_sonication_norm_shift_comparison"),
     "title": "S.aureus Sonication vs Control",
     "suffix": "saureus_sonication_vs_control",
     "groups": ["control", "saureus_sonication"]},
]

# These are set automatically for each run in ANALYSIS_RUNS (don't edit).
ADDON_OUTPUT_DIR = ANALYSIS_RUNS[0]["output_dir"]
ADDON_GROUP_KEYS = ANALYSIS_RUNS[0]["groups"]
COMPARISON_TITLE = ANALYSIS_RUNS[0]["title"]
ADDON_COMPARISON_SETS = [{"suffix": ANALYSIS_RUNS[0]["suffix"], "title": ANALYSIS_RUNS[0]["title"],
                          "groups": ANALYSIS_RUNS[0]["groups"]}]

# What to compare. Every combination is run -> 2 x 3 = 6 variants.
NORMALIZATION_METHODS_TO_COMPARE = ["l2", "area"]
SHIFT_METHODS_TO_COMPARE = ["none", "peak1445", "global"]

NORM_LABELS = {"l2": "L2 norm", "area": "Area norm", "minmax": "Min-max norm"}
SHIFT_LABELS = {"none": "no shift corr.", "peak1445": "1445 peak aligned",
                "global": "global shift corr."}

# Largest correction ever applied (cm-1). A spectrum whose required shift
# is bigger than this is left unshifted and flagged in the shift log.
MAX_ALLOWED_SHIFT_CM1 = 15.0

# --- Method A: 1445 cm-1 peak alignment (CH2 deformation) ---------------
# The 1445 peak's true maximum is located in the search window (with
# sub-point parabolic refinement), and the whole spectrum's x-axis is
# translated so that peak sits exactly at PEAK1445_TARGET_CM1.
PEAK1445_TARGET_CM1 = 1445.0
PEAK1445_SEARCH_MIN_CM1 = 1420.0
PEAK1445_SEARCH_MAX_CM1 = 1470.0
# "cell": every single-cell spectrum aligned on its own 1445 peak.
# "replicate": 1445 found on each replicate's MEAN spectrum, and that one
#              shift applied to every cell of the replicate (assumes the
#              drift is per measurement session, and is less noise-driven).
PEAK1445_LEVEL = "cell"

# --- Method B: global Raman shift correction ----------------------------
# Same principle as the Plotly "Live Global Raman Shift Correction" tool:
#   * one spectrum is the FIXED reference (there: GC50; here: Control mean)
#   * the other spectrum is moved as a WHOLE along the Raman-shift axis in
#     1 cm-1 steps (new x = original x + shift); intensity is never changed
#   * the chosen shift is saved with the corrected spectrum
# The difference: instead of you clicking +1 / -1 until it looks aligned,
# the script tries every step from -GLOBAL_SHIFT_MAX_CM1 to
# +GLOBAL_SHIFT_MAX_CM1 and keeps the one whose shifted spectrum correlates
# best with the reference. An HTML version of the clicking tool is still
# written (see MAKE_LIVE_SHIFT_HTML) so you can check / override by eye.
GLOBAL_SHIFT_REFERENCE_GROUP = "control"
GLOBAL_SHIFT_STEP_CM1 = 1.0
GLOBAL_SHIFT_MAX_CM1 = 15
# Only this part of the spectrum is used for the correlation score, so the
# edges (where a shifted spectrum runs out of data) don't bias the choice.
GLOBAL_SHIFT_MATCH_MIN_CM1 = 620.0
GLOBAL_SHIFT_MATCH_MAX_CM1 = 1780.0
# "replicate": replicate MEAN spectrum vs Control mean (closest to the
#              original tool, which moved one mean spectrum vs another);
#              the shift is then applied to every cell in that replicate.
# "cell":      every single-cell spectrum matched to the Control mean.
GLOBAL_SHIFT_LEVEL = "replicate"
# Rebuild the Control reference from the already-aligned Control spectra
# and re-match this many times (1 = no refinement).
GLOBAL_SHIFT_ITERATIONS = 2
# Manual overrides after checking the HTML tool, e.g.
#   {("ecoli_heat", "R2"): -3, ("saureus_heat", "R1"): 2}
# Applied to every cell of that replicate, for the "global" variants only.
GLOBAL_SHIFT_MANUAL_OVERRIDES = {}
# Write the interactive Plotly HTML (one per normalization method).
MAKE_LIVE_SHIFT_HTML = True

# --- QC: independent check of how well peaks line up after correction ---
# Phenylalanine ring breathing (~1003 cm-1) is sharp and should sit at the
# same position in every cell. Its spread is reported for every variant.
# (For the 1445-aligned variants, the 1445 spread is ~0 by construction,
# so the Phe spread is the fair, independent comparison.)
QC_PEAKS = {"phe1003": (990.0, 1015.0), "ch1445": (1420.0, 1470.0)}
# ========================================================================
# ========================================================================


# ----------------------------------------------------------------------
# Loading: same preprocessing as preprocess_one_spectrum(), but stopped
# right before normalization so both normalizations can be applied to the
# exact same baseline-corrected spectra, and the x-axis correction can be
# applied before the final interpolation onto the common grid.
# ----------------------------------------------------------------------
def _preprocess_until_baseline(x, y, sample_label=""):
    if SPIKE_REMOVAL_ENABLED:
        y = remove_cosmic_spikes(y, window=SPIKE_WINDOW, threshold=SPIKE_THRESHOLD)
    if ROI_ENABLED:
        x, y = apply_roi(x, y, ROI_MIN_CM1, ROI_MAX_CM1)
    if len(x) < max(SG_POLYORDER + 2, 5):
        print(f"    [WARN] {sample_label}: too few points after ROI cut, skipping.")
        return None
    if SMOOTHING_ENABLED:
        y = smooth_savgol(y, window=SG_WINDOW, polyorder=SG_POLYORDER)
    if BASELINE_ENABLED:
        y = subtract_baseline(y, lam=ALS_LAMBDA, p=ALS_P, niter=ALS_NITER)
    return x, y


def load_group_cells(group_dir):
    """Same folder layouts as load_group_data(). Returns
    dict 'R1' -> list of {"file", "x", "y"} (baseline-corrected, native x)."""
    rep_folders = find_replicate_folders(group_dir)
    if rep_folders:
        path_groups = {r: sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True))
                       for r, p in rep_folders.items()}
        layout = "subfolder"
    else:
        all_paths = sorted(glob.glob(os.path.join(group_dir, "**", "*.csv"), recursive=True))
        if not all_paths:
            print(f"    [WARN] No CSVs found under '{group_dir}' at all.")
            return {}
        path_groups, unmatched = group_files_by_replicate_prefix(all_paths)
        layout = "filename prefix"
        if unmatched:
            print(f"    [WARN] {len(unmatched)} CSV(s) had no R<number>_ prefix and were skipped.")

    out = {}
    for rep, paths in path_groups.items():
        cells = []
        for p in paths:
            res = read_raman_csv(p)
            if res is None:
                continue
            pre = _preprocess_until_baseline(res[0], res[1], f"{rep}/{os.path.basename(p)}")
            if pre is None:
                continue
            cells.append({"file": os.path.basename(p), "x": pre[0], "y": pre[1]})
        if cells:
            out[rep] = cells
            print(f"    {rep}: {len(cells)} usable cell(s) from {len(paths)} CSV(s) [{layout}]")
        else:
            print(f"    [WARN] {rep}: no usable spectra found ({len(paths)} CSV(s) seen).")
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def _normalize_cells(cells_by_rep, method):
    return {rep: [{"file": c["file"], "x": c["x"],
                   "y": normalize_spectrum(c["y"], method) if NORMALIZATION_ENABLED else c["y"]}
                  for c in cells]
            for rep, cells in cells_by_rep.items()}


def _to_grid(cell, common_x, shift=0.0):
    """Global x-axis translation (new x = x + shift), intensity unchanged,
    then interpolation onto the common grid."""
    return np.interp(common_x, cell["x"] + shift, cell["y"])


def _rep_sort(reps):
    return sorted(reps, key=lambda r: int(r[1:]))


# ----------------------------------------------------------------------
# Peak locating (used by 1445 alignment and by the QC metrics)
# ----------------------------------------------------------------------
def _locate_peak(x, y, lo, hi):
    """Highest local maximum inside [lo, hi], refined by a parabola through
    up to 5 points around it. Returns position in cm-1, or None."""
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


# ----------------------------------------------------------------------
# Shift method A: 1445 cm-1 peak alignment
# ----------------------------------------------------------------------
def _shift_from_1445(pos):
    if pos is None:
        return 0.0, "1445 peak not found in search window -> not shifted"
    s = PEAK1445_TARGET_CM1 - pos
    if abs(s) > MAX_ALLOWED_SHIFT_CM1:
        return 0.0, f"required shift {s:+.2f} cm-1 exceeds limit -> not shifted"
    return float(s), f"1445 peak found at {pos:.2f} cm-1"


def compute_shifts_peak1445(norm_cells, common_x):
    shifts, notes = {}, {}
    for g, reps in norm_cells.items():
        shifts[g], notes[g] = {}, {}
        for rep, cells in reps.items():
            if PEAK1445_LEVEL == "replicate":
                mean = np.mean([_to_grid(c, common_x) for c in cells], axis=0)
                s, note = _shift_from_1445(_locate_peak(common_x, mean, PEAK1445_SEARCH_MIN_CM1,
                                                        PEAK1445_SEARCH_MAX_CM1))
                shifts[g][rep] = np.full(len(cells), s)
                notes[g][rep] = [f"[replicate mean] {note}"] * len(cells)
            else:
                ss, nn = [], []
                for c in cells:
                    s, note = _shift_from_1445(_locate_peak(c["x"], c["y"], PEAK1445_SEARCH_MIN_CM1,
                                                            PEAK1445_SEARCH_MAX_CM1))
                    ss.append(s)
                    nn.append(note)
                shifts[g][rep] = np.array(ss)
                notes[g][rep] = nn
    return shifts, notes


# ----------------------------------------------------------------------
# Shift method B: global Raman shift correction (automated version of the
# +1 / -1 cm-1 button tool)
# ----------------------------------------------------------------------
def _best_global_shift(x, y, reference, common_x, match_mask):
    """Tries every whole-spectrum translation in GLOBAL_SHIFT_STEP_CM1 steps
    (exactly what the +1/-1 buttons do) and returns the one whose shifted
    spectrum correlates best with the fixed reference."""
    candidates = np.arange(-GLOBAL_SHIFT_MAX_CM1, GLOBAL_SHIFT_MAX_CM1 + 1e-9, GLOBAL_SHIFT_STEP_CM1)
    ref = reference[match_mask]
    scores = []
    for s in candidates:
        moved = np.interp(common_x, x + s, y)[match_mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(moved, ref)[0, 1]
        scores.append(r if np.isfinite(r) else -np.inf)
    scores = np.array(scores)
    best = int(np.argmax(scores))
    zero = int(np.argmin(np.abs(candidates)))
    return float(candidates[best]), float(scores[best]), float(scores[zero])


def compute_shifts_global(norm_cells, common_x):
    match_mask = (common_x >= GLOBAL_SHIFT_MATCH_MIN_CM1) & (common_x <= GLOBAL_SHIFT_MATCH_MAX_CM1)
    shifts = {g: {r: np.zeros(len(c)) for r, c in reps.items()} for g, reps in norm_cells.items()}
    notes = {g: {r: [""] * len(c) for r, c in reps.items()} for g, reps in norm_cells.items()}

    ref_key = GLOBAL_SHIFT_REFERENCE_GROUP if norm_cells.get(GLOBAL_SHIFT_REFERENCE_GROUP) else None
    if ref_key is None:
        print(f"  [WARN] Reference group '{GLOBAL_SHIFT_REFERENCE_GROUP}' has no data -> "
              f"using the mean of ALL groups as the fixed reference instead.")
    ref_groups = [ref_key] if ref_key else list(norm_cells.keys())

    reference = None
    for it in range(max(1, GLOBAL_SHIFT_ITERATIONS)):
        # fixed reference = (already aligned) Control mean spectrum
        reference = np.mean([_to_grid(c, common_x, shifts[g][r][i])
                             for g in ref_groups for r, cells in norm_cells[g].items()
                             for i, c in enumerate(cells)], axis=0)
        for g, reps in norm_cells.items():
            for r, cells in reps.items():
                if GLOBAL_SHIFT_LEVEL == "replicate":
                    mean = np.mean([_to_grid(c, common_x) for c in cells], axis=0)
                    s, sc, sc0 = _best_global_shift(common_x, mean, reference, common_x, match_mask)
                    if abs(s) >= GLOBAL_SHIFT_MAX_CM1:
                        note = (f"best shift hit the search limit ({s:+.0f}) -> check this replicate "
                                f"in the HTML tool; corr {sc0:.4f} -> {sc:.4f}")
                    else:
                        note = f"[replicate mean] corr vs reference {sc0:.4f} (0 cm-1) -> {sc:.4f}"
                    shifts[g][r][:] = s
                    notes[g][r] = [note] * len(cells)
                else:
                    for i, c in enumerate(cells):
                        s, sc, sc0 = _best_global_shift(c["x"], c["y"], reference, common_x, match_mask)
                        shifts[g][r][i] = s
                        notes[g][r][i] = f"corr vs reference {sc0:.4f} (0 cm-1) -> {sc:.4f}"

    for (g, r), s in GLOBAL_SHIFT_MANUAL_OVERRIDES.items():
        if g in shifts and r in shifts[g]:
            shifts[g][r][:] = float(s)
            notes[g][r] = [f"MANUAL OVERRIDE {float(s):+.0f} cm-1"] * len(shifts[g][r])
        else:
            print(f"  [WARN] Manual override for {(g, r)} ignored: no such group/replicate loaded.")

    # final reference after overrides, for the HTML tool
    reference = np.mean([_to_grid(c, common_x, shifts[g][r][i])
                         for g in ref_groups for r, cells in norm_cells[g].items()
                         for i, c in enumerate(cells)], axis=0)
    return shifts, notes, reference


# ----------------------------------------------------------------------
# Build the group_info structure the ORIGINAL functions expect
# ----------------------------------------------------------------------
def build_variant_group_info(discovered, norm_cells, shifts, common_x):
    group_info = {}
    for key in ADDON_GROUP_KEYS:
        cfg = GROUPS[key]
        rep_data = {}
        for r in _rep_sort(norm_cells.get(key, {}).keys()):
            cells = norm_cells[key][r]
            rep_data[r] = np.vstack([_to_grid(c, common_x, shifts[key][r][i])
                                     for i, c in enumerate(cells)])
        pooled = np.vstack(list(rep_data.values())) if rep_data else None
        group_info[key] = {**cfg, "dir": discovered[key]["dir"],
                           "folder_name": discovered[key]["folder_name"],
                           "replicate_data": rep_data, "pooled": pooled}
    return group_info


def shift_log_dataframe(norm_cells, shifts, notes):
    rows = []
    for g in ADDON_GROUP_KEYS:
        if g not in norm_cells:
            continue
        for r in _rep_sort(norm_cells[g].keys()):
            for i, c in enumerate(norm_cells[g][r]):
                rows.append({"group": GROUPS[g]["display"], "replicate": r, "file": c["file"],
                             "applied_shift_cm-1": round(float(shifts[g][r][i]), 3),
                             "note": notes[g][r][i]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Shift diagnostics figure (per shift-corrected variant)
# ----------------------------------------------------------------------
def plot_shift_diagnostics(norm_cells, shifts, common_x, variant_label, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    labels, data, cols = [], [], []
    for g in ADDON_GROUP_KEYS:
        if g not in norm_cells:
            continue
        for r in _rep_sort(norm_cells[g].keys()):
            labels.append(f"{GROUPS[g]['display']}\n{r}")
            data.append(shifts[g][r])
            cols.append(GROUPS[g]["color"])
    if data:
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Applied x-axis shift (cm$^{-1}$)")
    ax.set_title("Shift applied per group / replicate")

    for ax, (lo, hi, center, name) in zip(axes[1:], [(1380, 1510, 1445, "CH$_2$ ~1445"),
                                                     (960, 1045, 1003, "Phe ~1003")]):
        zoom = (common_x >= lo) & (common_x <= hi)
        for g in ADDON_GROUP_KEYS:
            if g not in norm_cells:
                continue
            before = np.mean([_to_grid(c, common_x) for r in norm_cells[g]
                              for c in norm_cells[g][r]], axis=0)
            after = np.mean([_to_grid(c, common_x, shifts[g][r][i]) for r in norm_cells[g]
                             for i, c in enumerate(norm_cells[g][r])], axis=0)
            col = GROUPS[g]["color"]
            ax.plot(common_x[zoom], before[zoom], color=col, linestyle=":", linewidth=1.3)
            ax.plot(common_x[zoom], after[zoom], color=col, linewidth=1.9, label=GROUPS[g]["display"])
        ax.axvline(center, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax.set_ylabel("Mean normalized intensity")
        ax.set_title(f"{name} region: before (dotted) vs after (solid)")
        handles = ax.get_legend_handles_labels()[0] + [
            Line2D([0], [0], color="0.3", linestyle=":", label="before correction"),
            Line2D([0], [0], color="0.3", linestyle="-", label="after correction")]
        ax.legend(handles=handles, fontsize=7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Shift-correction diagnostics [{variant_label}]", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> Shift diagnostics plot saved: {path}")


# ----------------------------------------------------------------------
# Interactive HTML: the "Live Global Raman Shift Correction" tool, adapted
# BLUE   = Control mean spectrum (FIXED reference, instead of GC50)
# ORANGE = selected group/replicate mean spectrum (MOVABLE, instead of a
#          Stem Loop), starting at the shift the pipeline chose.
# +1 / -1 buttons move it by exactly 1 cm-1; intensity never changes.
# SAVE downloads the shifted spectrum as CSV, exactly like the original.
# ----------------------------------------------------------------------
LIVE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Live Global Raman Shift Correction - __NORM__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 25px; background: white; }
h1 { text-align: center; font-size: 28px; margin-bottom: 20px; }
.controls { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; padding: 18px;
            border: 1px solid #ccc; border-radius: 10px; background: #f7f7f7; }
label { font-size: 18px; }
select { min-width: 360px; padding: 12px; font-size: 16px; border: 1px solid #999; border-radius: 5px; }
button { padding: 13px 22px; font-size: 16px; border-radius: 6px; border: 1px solid #999;
         cursor: pointer; background: #eee; }
button:hover { filter: brightness(0.90); }
.reset-button { background: #ff8c00; color: white; border: none; }
.auto-button { background: #4a6fd1; color: white; border: none; }
.save-button { background: #2e8b57; color: white; border: none; }
.shift-display { font-size: 21px; font-weight: bold; margin-left: 10px; }
.instructions { margin-top: 15px; margin-bottom: 10px; padding: 12px; font-size: 16px; line-height: 1.6; }
.status { font-size: 17px; font-weight: bold; margin: 8px 0; }
#ramanPlot { width: 100%; height: 750px; }
pre { background: #f4f4f4; padding: 12px; border-radius: 6px; font-size: 14px; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>Interactive Global Raman Shift Correction (__NORM__)</h1>
<div class="controls">
  <label><b>Spectrum:</b></label>
  <select id="spectrumSelect">__OPTIONS__</select>
  <button type="button" onclick="moveBy(-1)">&larr; -1 cm&#8315;&#185;</button>
  <button type="button" onclick="moveBy(1)">+1 cm&#8315;&#185; &rarr;</button>
  <button type="button" class="auto-button" onclick="toAuto()">AUTO (pipeline value)</button>
  <button type="button" class="reset-button" onclick="resetShift()">RESET to 0</button>
  <button type="button" class="save-button" onclick="saveCorrectedSpectrum()">SAVE CORRECTED SPECTRUM</button>
  <span class="shift-display" id="shiftDisplay"></span>
</div>
<div class="instructions">
  <b>Blue:</b> Control mean &mdash; fixed reference &nbsp;&nbsp;&nbsp; <b>Orange:</b> selected replicate mean &mdash; movable
  <br>Each click moves the complete orange spectrum by exactly 1 cm&#8315;&#185;. Only the Raman-shift coordinate changes; intensity is unchanged.
  <br>The orange spectrum starts at the shift the pipeline chose automatically (best correlation with the Control mean).
  Dashed grey lines mark 1003 and 1445 cm&#8315;&#185;.
</div>
<div class="status" id="status"></div>
<div id="ramanPlot"></div>
<p><b>Manual overrides</b> (only replicates you moved away from the automatic value). Paste into
<code>GLOBAL_SHIFT_MANUAL_OVERRIDES</code> in the script and rerun to use them:</p>
<pre id="overrides">{}</pre>
<script>
const refX = __REFX__;
const refY = __REFY__;
const items = __ITEMS__;
const xRange = __XRANGE__;
let selectedIndex = 0;
let shifts = items.map(function(it) { return it.auto_shift; });

function fmt(s) { return (s >= 0 ? "+" : "") + s + " cm\u207B\u00B9"; }

Plotly.newPlot("ramanPlot", [
  { x: refX, y: refY, mode: "lines", name: "Control mean (fixed)",
    line: { color: "#1f77b4", width: 3 },
    hovertemplate: "Control<br>Raman Shift: %{x:.2f} cm\u207B\u00B9<br>Intensity: %{y:.6g}<extra></extra>" },
  { x: items[0].x, y: items[0].y, mode: "lines", name: items[0].label,
    line: { color: "#ff7f0e", width: 3 } }
], {
  title: { text: "", x: 0.5, xanchor: "center", font: { size: 22 } },
  xaxis: { title: "Raman Shift (cm\u207B\u00B9)", range: xRange, showgrid: true, zeroline: false },
  yaxis: { title: "Normalized intensity", showgrid: true, zeroline: false },
  shapes: [
    { type: "line", x0: 1003, x1: 1003, yref: "paper", y0: 0, y1: 1, line: { dash: "dash", color: "gray", width: 1 } },
    { type: "line", x0: 1445, x1: 1445, yref: "paper", y0: 0, y1: 1, line: { dash: "dash", color: "gray", width: 1 } }
  ],
  legend: { x: 1.02, y: 1, xanchor: "left", yanchor: "top", bgcolor: "rgba(255,255,255,0.95)",
            bordercolor: "black", borderwidth: 1, font: { size: 14 } },
  hovermode: "x unified", height: 750, margin: { l: 100, r: 320, t: 120, b: 90 }
}, { responsive: true, displaylogo: false, scrollZoom: true });

function updateGraph() {
  const it = items[selectedIndex];
  const s = shifts[selectedIndex];
  const shiftedX = it.x.map(function(x) { return x + s; });   // NEW X = ORIGINAL X + shift
  Plotly.restyle("ramanPlot", {
    x: [shiftedX], name: [it.label + " \u2014 " + fmt(s)],
    hovertemplate: [it.label + "<br>Raman Shift: %{x:.2f} cm\u207B\u00B9<br>Intensity: %{y:.6g}<extra></extra>"]
  }, [1]);
  Plotly.relayout("ramanPlot", { "title.text": it.label + " vs Control mean<br><sup>Global Raman shift = "
    + fmt(s) + " &nbsp;(pipeline automatic value: " + fmt(it.auto_shift) + ")</sup>" });
  document.getElementById("shiftDisplay").innerText = "Current global shift: " + fmt(s);
  document.getElementById("status").innerText = (s === 0)
    ? it.label + " is at original position." : it.label + " shifted by " + fmt(s) + ".";
  let lines = [];
  for (let i = 0; i < items.length; i++) {
    if (shifts[i] !== items[i].auto_shift) {
      lines.push('    ("' + items[i].group_key + '", "' + items[i].rep + '"): ' + shifts[i] + ",");
    }
  }
  document.getElementById("overrides").innerText =
    lines.length ? "{\n" + lines.join("\n") + "\n}" : "{}";
}
function moveBy(d) { shifts[selectedIndex] = shifts[selectedIndex] + d; updateGraph(); }
function toAuto() { shifts[selectedIndex] = items[selectedIndex].auto_shift; updateGraph(); }
function resetShift() { shifts[selectedIndex] = 0; updateGraph(); }
document.getElementById("spectrumSelect").addEventListener("change", function() {
  selectedIndex = parseInt(this.value, 10); updateGraph();
});
function saveCorrectedSpectrum() {
  const it = items[selectedIndex];
  const s = shifts[selectedIndex];
  let csv = "RamanShift_cm-1,Mean_Normalized_Intensity\n";
  for (let i = 0; i < it.x.length; i++) { csv += (it.x[i] + s) + "," + it.y[i] + "\n"; }
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const shiftText = (s >= 0 ? "plus_" + s : "minus_" + Math.abs(s)) + "cm-1";
  link.href = url;
  link.download = it.label.replace(/[^A-Za-z0-9]+/g, "_") + "_ShiftCorrected_" + shiftText + ".csv";
  document.body.appendChild(link); link.click(); document.body.removeChild(link);
  URL.revokeObjectURL(url);
  document.getElementById("status").innerText = "\u2713 Corrected spectrum saved \u2014 Global shift = " + fmt(s) + ".";
}
updateGraph();
</script>
</body>
</html>
"""


def write_live_shift_html(norm_cells, shifts, reference, common_x, norm_method, path):
    items = []
    for g in ADDON_GROUP_KEYS:
        if g not in norm_cells:
            continue
        for r in _rep_sort(norm_cells[g].keys()):
            mean = np.mean([_to_grid(c, common_x) for c in norm_cells[g][r]], axis=0)
            items.append({"group_key": g, "rep": r,
                          "label": f"{GROUPS[g]['display']} {r}",
                          "auto_shift": float(np.round(shifts[g][r][0], 3)),
                          "x": [round(float(v), 4) for v in common_x],
                          "y": [float(v) for v in mean]})
    if not items:
        return
    options = "".join(f'<option value="{i}">{_html.escape(it["label"])} '
                      f'(auto {it["auto_shift"]:+.0f} cm&#8315;&#185;)</option>'
                      for i, it in enumerate(items))
    doc = (LIVE_HTML_TEMPLATE
           .replace("__NORM__", _html.escape(NORM_LABELS.get(norm_method, norm_method)))
           .replace("__OPTIONS__", options)
           .replace("__REFX__", _json.dumps([round(float(v), 4) for v in common_x]))
           .replace("__REFY__", _json.dumps([float(v) for v in reference]))
           .replace("__ITEMS__", _json.dumps(items))
           .replace("__XRANGE__", _json.dumps([float(ROI_MIN_CM1), float(ROI_MAX_CM1)])))
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"  -> Live global-shift HTML tool saved (open in a browser): {path}")


# ----------------------------------------------------------------------
# Metrics for the cross-variant summary (read from the CSVs the original
# functions already wrote, plus a peak-alignment QC)
# ----------------------------------------------------------------------
def _safe_read_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def _peak_position_spread(group_info, common_x, lo, hi):
    within, between = [], []
    for key in ADDON_GROUP_KEYS:
        rd = group_info[key]["replicate_data"]
        rep_medians = []
        for r, arr in rd.items():
            pos = [p for p in (_locate_peak(common_x, row, lo, hi) for row in arr) if p is not None]
            if len(pos) >= 2:
                within.append(np.std(pos, ddof=1))
            if pos:
                rep_medians.append(np.median(pos))
        if len(rep_medians) >= 2:
            between.append(np.std(rep_medians, ddof=1))
    return (float(np.mean(within)) if within else np.nan,
            float(np.mean(between)) if between else np.nan)


def collect_variant_metrics(out_dir, suffix, nm, sm, tag, group_info, shift_log, common_x):
    row = {"variant_folder": tag, "normalization": nm, "shift_correction": sm}

    rep = _safe_read_csv(os.path.join(out_dir, f"{suffix}_classification_report.csv"), index_col=0)
    if rep is not None and "accuracy" in rep.index:
        row["lda_cv_accuracy"] = float(rep.loc["accuracy", "f1-score"])
        row["lda_cv_macro_f1"] = float(rep.loc["macro avg", "f1-score"])
    loro = _safe_read_csv(os.path.join(out_dir, f"{suffix}_classification_report_LORO.csv"), index_col=0)
    if loro is not None and "accuracy" in loro.index:
        row["lda_loro_accuracy"] = float(loro.loc["accuracy", "f1-score"])
        row["lda_loro_macro_f1"] = float(loro.loc["macro avg", "f1-score"])

    stats = _safe_read_csv(os.path.join(out_dir, f"{suffix}_anova_by_wavenumber.csv"))
    if stats is not None:
        row["pct_spectrum_in_sig_clusters"] = 100 * float(
            stats["in_significant_cluster"].astype(str).str.lower().eq("true").mean())
    cl = _safe_read_csv(os.path.join(out_dir, f"{suffix}_significant_clusters.csv"))
    row["n_significant_clusters"] = (int(cl["significant"].astype(str).str.lower().eq("true").sum())
                                     if cl is not None and "significant" in cl else 0)
    pk = _safe_read_csv(os.path.join(out_dir, f"{suffix}_significant_peaks.csv"))
    row["n_significant_effect_gated_peaks"] = 0 if pk is None else len(pk)
    sh = _safe_read_csv(os.path.join(out_dir, f"{suffix}_peak_position_shifts.csv"))
    row["n_significant_peak_position_shifts"] = (
        int(sh["significant_shift"].astype(str).str.lower().eq("true").sum())
        if sh is not None and "significant_shift" in sh else 0)

    for qc_name, (lo, hi) in QC_PEAKS.items():
        w, b = _peak_position_spread(group_info, common_x, lo, hi)
        row[f"{qc_name}_within_replicate_sd_cm-1"] = round(w, 3) if np.isfinite(w) else np.nan
        row[f"{qc_name}_between_replicate_sd_cm-1"] = round(b, 3) if np.isfinite(b) else np.nan

    row["mean_abs_applied_shift_cm-1"] = (round(float(shift_log["applied_shift_cm-1"].abs().mean()), 3)
                                          if not shift_log.empty else 0.0)
    return row


# ----------------------------------------------------------------------
# Cross-variant comparison figures
# ----------------------------------------------------------------------
def _variant_X(group_info):
    keys = [k for k in ADDON_GROUP_KEYS if group_info[k]["pooled"] is not None]
    X = np.vstack([group_info[k]["pooled"] for k in keys])
    idx = np.concatenate([np.full(group_info[k]["pooled"].shape[0], i) for i, k in enumerate(keys)])
    return keys, X, idx


def plot_variant_grid(variants, kind, path):
    """kind = 'pca' or 'lda'. Rows = normalization, columns = shift method."""
    if not HAS_SKLEARN_DA:
        print(f"  [NOTE] scikit-learn not found -> skipping {kind.upper()} comparison grid.")
        return
    nrows, ncols = len(NORMALIZATION_METHODS_TO_COMPARE), len(SHIFT_METHODS_TO_COMPARE)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    handles = {}
    for i, nm in enumerate(NORMALIZATION_METHODS_TO_COMPARE):
        for j, sm in enumerate(SHIFT_METHODS_TO_COMPARE):
            ax = axes[i][j]
            gi = variants.get((nm, sm))
            title = f"{NORM_LABELS.get(nm, nm)} | {SHIFT_LABELS[sm]}"
            if gi is None:
                ax.axis("off")
                continue
            keys, X, idx = _variant_X(gi)
            if len(keys) < 2:
                ax.set_title(title + "\n(need >=2 groups)")
                continue
            if kind == "pca":
                model = PCA(n_components=2)
                S = model.fit_transform(X)
                ve = model.explained_variance_ratio_ * 100
                ax.set_xlabel(f"PC1 ({ve[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({ve[1]:.1f}%)")
            else:
                n_comp = min(2, len(keys) - 1)
                S = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X, idx)
                if n_comp == 1:
                    S = np.column_stack([S[:, 0], np.random.default_rng(0).normal(0, 0.02, len(S))])
                ax.set_xlabel("LD1")
                ax.set_ylabel("LD2" if n_comp == 2 else "(jitter)")
            for gi_i, k in enumerate(keys):
                m = idx == gi_i
                h = ax.scatter(S[m, 0], S[m, 1], s=12, alpha=0.65, color=GROUPS[k]["color"],
                               label=GROUPS[k]["display"])
                handles[GROUPS[k]["display"]] = h
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
               ncol=len(handles), fontsize=10, frameon=False)
    nice = "PCA (unsupervised)" if kind == "pca" else "LDA (supervised, fit on all cells - see CV accuracy for honest separation)"
    fig.suptitle(f"{COMPARISON_TITLE}: {nice}\nrows = normalization, columns = shift correction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {kind.upper()} comparison grid saved: {path}")


def plot_mean_spectra_grid(variants, common_x, path):
    nrows, ncols = len(NORMALIZATION_METHODS_TO_COMPARE), len(SHIFT_METHODS_TO_COMPARE)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), squeeze=False)
    for i, nm in enumerate(NORMALIZATION_METHODS_TO_COMPARE):
        for j, sm in enumerate(SHIFT_METHODS_TO_COMPARE):
            ax = axes[i][j]
            gi = variants.get((nm, sm))
            if gi is None:
                ax.axis("off")
                continue
            for k in ADDON_GROUP_KEYS:
                if gi[k]["pooled"] is None:
                    continue
                ax.plot(common_x, gi[k]["pooled"].mean(axis=0), color=GROUPS[k]["color"],
                        linewidth=1.2, label=GROUPS[k]["display"])
            for wn in (1003, 1445):
                ax.axvline(wn, color="gray", linestyle="--", linewidth=0.7)
            ax.set_title(f"{NORM_LABELS.get(nm, nm)} | {SHIFT_LABELS[sm]}", fontsize=11, fontweight="bold")
            ax.set_xlim(common_x.min(), common_x.max())
            ax.set_xlabel("Raman Shift (cm$^{-1}$)")
            ax.set_ylabel("Mean intensity")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{COMPARISON_TITLE}: mean spectra per variant\n"
                 "(y-scales differ between L2 and area rows - compare shapes, not absolute values)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> Mean-spectra comparison grid saved: {path}")


def plot_metric_summary(summary_df, path):
    metrics = [
        ("lda_cv_accuracy", "LDA cross-validated accuracy (cell-level 5-fold)", "higher = groups more separable"),
        ("lda_loro_accuracy", "LDA accuracy, leave-one-replicate-out", "stricter: tests a replicate never seen in training"),
        ("pct_spectrum_in_sig_clusters", "% of spectrum in significant ANOVA clusters", ""),
        ("n_significant_peak_position_shifts", "# significant peak-position shifts (vs Control)",
         "after correction, remaining shifts are relative to the aligned reference"),
        ("phe1003_within_replicate_sd_cm-1", "Phe ~1003 position SD within replicate (cm$^{-1}$)",
         "lower = cells better aligned (independent QC)"),
        ("phe1003_between_replicate_sd_cm-1", "Phe ~1003 position SD between replicates (cm$^{-1}$)",
         "lower = replicates better aligned (independent QC)"),
        ("mean_abs_applied_shift_cm-1", "Mean |applied shift| (cm$^{-1}$)", ""),
    ]
    metrics = [m for m in metrics if m[0] in summary_df.columns]
    if not metrics:
        return
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.3 * nrows), squeeze=False)
    shifts_order = SHIFT_METHODS_TO_COMPARE
    norms = NORMALIZATION_METHODS_TO_COMPARE
    width = 0.8 / len(norms)
    norm_colors = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5"]
    for ax, (col, title, sub) in zip(axes.flat, metrics):
        x = np.arange(len(shifts_order))
        for n_i, nm in enumerate(norms):
            vals = []
            for sm in shifts_order:
                r = summary_df[(summary_df["normalization"] == nm) & (summary_df["shift_correction"] == sm)]
                vals.append(float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan)
            off = (n_i - (len(norms) - 1) / 2) * width
            bars = ax.bar(x + off, vals, width=width, color=norm_colors[n_i % 4],
                          label=NORM_LABELS.get(nm, nm))
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([SHIFT_LABELS[s] for s in shifts_order], fontsize=9)
        ax.set_title(title + (f"\n({sub})" if sub else ""), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"{COMPARISON_TITLE}: normalization x shift-correction summary",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> Metric summary plot saved: {path}")


# ----------------------------------------------------------------------
# Add-on main
# ----------------------------------------------------------------------
import textwrap as _textwrap


def _wrapped_title(label, *args, **kwargs):
    """Display-only: wraps long title lines (the variant name makes some
    titles too wide for the original fixed figure sizes). Calls
    Axes.set_title directly - exactly what plt.title does internally - so
    it can never call itself, even when re-run in the same notebook kernel."""
    if isinstance(label, str):
        label = "\n".join(_textwrap.fill(line, 45) if "$" not in line else line
                           for line in label.split("\n"))
    return plt.gca().set_title(label, *args, **kwargs)


def main_all_norm_shift_comparisons():
    """Runs the full normalization x shift comparison for every entry in
    ANALYSIS_RUNS (heat-inactivated, then sonicated)."""
    global ADDON_OUTPUT_DIR, ADDON_GROUP_KEYS, COMPARISON_TITLE, ADDON_COMPARISON_SETS
    _previous_plt_title = plt.title
    plt.title = _wrapped_title
    try:
        for run in ANALYSIS_RUNS:
            ADDON_OUTPUT_DIR = run["output_dir"]
            ADDON_GROUP_KEYS = list(run["groups"])
            COMPARISON_TITLE = run["title"]
            ADDON_COMPARISON_SETS = [{"suffix": run["suffix"], "title": run["title"],
                                      "groups": list(run["groups"])}]
            print(f"\n\n{'*'*70}\nANALYSIS: {run['title']}\n{'*'*70}")
            _run_one_norm_shift_comparison()
    finally:
        plt.title = _previous_plt_title   # put pyplot back the way it was


def _run_one_norm_shift_comparison():
    if not os.path.isdir(ROOT_FOLDER):
        print(f"[ERROR] Folder not found: {ROOT_FOLDER}")
        sys.exit(1)
    if not ROI_ENABLED:
        print("[ERROR] ROI_ENABLED must be True so every spectrum shares a common wavenumber axis.")
        sys.exit(1)
    if not HAS_GAMES_HOWELL:
        print("[NOTE] statsmodels not found -> Games-Howell post-hoc will fall back to "
              "Bonferroni-corrected Welch t-tests. (pip install statsmodels for the real thing.)")

    os.makedirs(ADDON_OUTPUT_DIR, exist_ok=True)
    common_x = np.linspace(ROI_MIN_CM1, ROI_MAX_CM1, COMMON_GRID_POINTS)

    # --- load each selected group ONCE (up to baseline correction) ---
    discovered, raw_cells = {}, {}
    for key in ADDON_GROUP_KEYS:
        cfg = GROUPS[key]
        print(f"\n{'='*70}\nGROUP: {cfg['display']}\n{'='*70}")
        group_dir, folder_name = find_group_folder(ROOT_FOLDER, cfg["keywords"])
        discovered[key] = {"dir": group_dir, "folder_name": folder_name}
        if group_dir is None:
            print(f"  [WARN] No folder found for '{cfg['display']}' (keywords {cfg['keywords']}).")
            continue
        print(f"  Found folder: '{folder_name}'")
        cells = load_group_cells(group_dir)
        if cells:
            raw_cells[key] = cells
            print(f"  -> {sum(len(v) for v in cells.values())} cell(s) across {len(cells)} replicate(s)")
    if not raw_cells:
        print("[ERROR] No usable spectra in any of the selected groups.")
        sys.exit(1)

    variants, summary_rows = {}, []
    for nm in NORMALIZATION_METHODS_TO_COMPARE:
        norm_cells = {g: _normalize_cells(c, nm) for g, c in raw_cells.items()}
        for sm in SHIFT_METHODS_TO_COMPARE:
            tag = f"{nm}_{sm}"
            label = f"{NORM_LABELS.get(nm, nm)} | {SHIFT_LABELS[sm]}"
            out = os.path.join(ADDON_OUTPUT_DIR, tag)
            os.makedirs(out, exist_ok=True)
            print(f"\n{'#'*70}\nVARIANT: {label}   ->  {out}\n{'#'*70}")

            if sm == "none":
                shifts = {g: {r: np.zeros(len(c)) for r, c in reps.items()} for g, reps in norm_cells.items()}
                notes = {g: {r: ["no shift correction"] * len(c) for r, c in reps.items()}
                         for g, reps in norm_cells.items()}
            elif sm == "peak1445":
                shifts, notes = compute_shifts_peak1445(norm_cells, common_x)
            elif sm == "global":
                shifts, notes, reference = compute_shifts_global(norm_cells, common_x)
                if MAKE_LIVE_SHIFT_HTML:
                    write_live_shift_html(norm_cells, shifts, reference, common_x, nm,
                                          os.path.join(out, "LIVE_Global_Raman_Shift_Correction.html"))
            else:
                raise ValueError(f"Unknown shift method: {sm}")

            shift_log = shift_log_dataframe(norm_cells, shifts, notes)
            shift_log.to_csv(os.path.join(out, "applied_shift_log.csv"), index=False)
            if sm != "none":
                print(f"  Applied shifts: mean |shift| = {shift_log['applied_shift_cm-1'].abs().mean():.2f} cm-1, "
                      f"range {shift_log['applied_shift_cm-1'].min():+.2f} to "
                      f"{shift_log['applied_shift_cm-1'].max():+.2f} cm-1")
                plot_shift_diagnostics(norm_cells, shifts, common_x, label,
                                       os.path.join(out, "shift_correction_diagnostics.png"))

            group_info = build_variant_group_info(discovered, norm_cells, shifts, common_x)
            variants[(nm, sm)] = group_info

            # ---- the ORIGINAL pipeline, unchanged, on this variant's data ----
            print(f"\n-- per-group replicate plots --")
            for key in ADDON_GROUP_KEYS:
                plot_stacked_replicates(common_x, group_info[key], out)
            for comp in ADDON_COMPARISON_SETS:
                comp_v = {**comp, "title": f"{comp['title']} [{label}]"}
                print(f"\n-- {comp_v['title']} --")
                plot_group_comparison_stacked(common_x, group_info, comp_v, out)
                plot_group_comparison_overlapped(common_x, group_info, comp_v, out)
                run_anova_for_comparison(common_x, group_info, comp_v, out)
                run_peak_shift_for_comparison(common_x, group_info, comp_v, out)
                summary_rows.append(collect_variant_metrics(out, comp["suffix"], nm, sm, tag,
                                                            group_info, shift_log, common_x))

    # --- cross-variant comparison ---------------------------------------------
    print(f"\n{'='*70}\nCROSS-VARIANT COMPARISON\n{'='*70}")
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(ADDON_OUTPUT_DIR, "variant_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  -> Summary table saved: {summary_csv}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False))

    plot_metric_summary(summary_df, os.path.join(ADDON_OUTPUT_DIR, "COMPARE_metrics_summary.png"))
    plot_variant_grid(variants, "pca", os.path.join(ADDON_OUTPUT_DIR, "COMPARE_pca_grid.png"))
    plot_variant_grid(variants, "lda", os.path.join(ADDON_OUTPUT_DIR, "COMPARE_lda_grid.png"))
    plot_mean_spectra_grid(variants, common_x, os.path.join(ADDON_OUTPUT_DIR, "COMPARE_mean_spectra_grid.png"))

    print(f"\nAll add-on outputs saved in: {ADDON_OUTPUT_DIR}")


if __name__ == "__main__":
    # This runs the normalization/shift comparison for BOTH heat-inactivated and sonicated.
    main_all_norm_shift_comparisons()
