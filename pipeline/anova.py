"""
Step 4 - Cluster-based permutation ANOVA, effect-size gate (eta^2),
Games-Howell post-hoc, peak bar chart and ANOVA spectra plot.
run_anova_for_comparison() also calls the PCA / LDA / classification step.
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

try:
    from statsmodels.stats.libqsturng import psturng
    HAS_GAMES_HOWELL = True
except ImportError:
    HAS_GAMES_HOWELL = False

from config import *
from pipeline.plotting import plot_blank_placeholder
from pipeline.classification import (plot_pca_scores, plot_lda_scores,
                                     evaluate_classification, evaluate_classification_loro)


def anova_F_vectorized(X, group_idx, n_groups):
    """
    X: (N, P) array, one row per cell, one column per wavenumber.
    group_idx: (N,) int array of group index (0..n_groups-1) per row.
    Returns F (P,), eta_squared (P,), dfn, dfd.
    """
    N, P = X.shape
    grand_mean = X.mean(axis=0)
    ss_between = np.zeros(P)
    ss_within = np.zeros(P)
    for g in range(n_groups):
        mask = group_idx == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        Xg = X[mask]
        group_mean = Xg.mean(axis=0)
        ss_between += n_g * (group_mean - grand_mean) ** 2
        ss_within += ((Xg - group_mean) ** 2).sum(axis=0)

    dfn = n_groups - 1
    dfd = N - n_groups
    ms_between = ss_between / dfn
    ms_within = ss_within / dfd
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(ms_within > 0, ms_between / ms_within, 0.0)
        ss_total = ss_between + ss_within
        eta_sq = np.where(ss_total > 0, ss_between / ss_total, 0.0)
    return F, eta_sq, dfn, dfd


def find_clusters_above_threshold(values, threshold):
    above = values > threshold
    idx = np.where(above)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    return [(run, float(values[run].sum())) for run in runs]


def cluster_permutation_test(X, group_idx, n_groups, n_perms, cluster_alpha,
                              form_alpha, seed):
    """
    Cluster-mass permutation test (Maris & Oostenveld style), adapted from
    EEG/fMRI stats to handle the strong point-to-point correlation in a
    smoothed spectrum, which plain pointwise FDR ignores.

    NOTE (added): now also returns the null_max_mass array itself (the
    largest cluster mass seen in each of the n_perms label-shuffles), so
    it can be plotted -- this is the only change to this function's
    behavior; the test itself is unchanged.
    """
    F_obs, eta_sq_obs, dfn, dfd = anova_F_vectorized(X, group_idx, n_groups)
    F_thresh = f_dist.ppf(1 - form_alpha, dfn, dfd)
    obs_clusters = find_clusters_above_threshold(F_obs, F_thresh)

    rng = np.random.default_rng(seed)
    null_max_mass = np.zeros(n_perms)
    for i in range(n_perms):
        perm_idx = rng.permutation(group_idx)
        F_perm, _, _, _ = anova_F_vectorized(X, perm_idx, n_groups)
        perm_clusters = find_clusters_above_threshold(F_perm, F_thresh)
        null_max_mass[i] = max((m for _, m in perm_clusters), default=0.0)

    sig_mask = np.zeros(X.shape[1], dtype=bool)
    cluster_records = []
    for run, mass in obs_clusters:
        p_val = (1 + np.sum(null_max_mass >= mass)) / (n_perms + 1)
        if p_val < cluster_alpha:
            sig_mask[run] = True
        cluster_records.append((run, mass, p_val))

    return F_obs, eta_sq_obs, sig_mask, cluster_records, F_thresh, null_max_mass


def plot_permutation_null_distribution(null_max_mass, cluster_records, cluster_alpha, title, path):
    if len(cluster_records) == 0:
        print(f"  [INFO] No candidate clusters to show a null distribution for, skipping.")
        return None
    plt.figure(figsize=(9, 5))
    plt.hist(null_max_mass, bins=40, color="0.75", edgecolor="white")
    for run, mass, p in cluster_records:
        color = "green" if p < cluster_alpha else "0.4"
        plt.axvline(mass, color=color, linestyle="--", linewidth=1.5, alpha=0.85)
    handles = [
        Line2D([0], [0], color="0.75", lw=8, label=f"Null distribution ({len(null_max_mass)} shuffles)"),
        Line2D([0], [0], color="green", linestyle="--", lw=1.5, label=f"Significant cluster (p<{cluster_alpha})"),
        Line2D([0], [0], color="0.4", linestyle="--", lw=1.5, label="Non-significant candidate"),
    ]
    plt.legend(handles=handles, fontsize=9)
    plt.xlabel("Cluster mass (summed F-statistic within a contiguous run)")
    plt.ylabel("Count (out of permutations)")
    plt.title(title)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Permutation null-distribution plot saved: {path}")
    return path


def games_howell_pairwise(samples, labels, alpha):
    """Doesn't assume equal n or equal variance, unlike Tukey HSD --
    appropriate since group sizes here differ noticeably."""
    k = len(samples)
    n_pairs = k * (k - 1) // 2
    pvals = {}
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = len(samples[i]), len(samples[j])
            mi, mj = np.mean(samples[i]), np.mean(samples[j])
            vi, vj = np.var(samples[i], ddof=1), np.var(samples[j], ddof=1)
            se_sq = vi / ni + vj / nj
            if se_sq <= 0:
                continue
            se = np.sqrt(se_sq)
            t_stat = abs(mi - mj) / se
            df = se_sq ** 2 / ((vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1))

            if HAS_GAMES_HOWELL:
                q = t_stat * np.sqrt(2)
                p = psturng(q, k, df)
                p = float(p[0]) if isinstance(p, np.ndarray) else float(p)
            else:
                # Fallback: Welch t-test, Bonferroni-corrected across pairs
                p_raw = 2 * t_dist.sf(t_stat, df)
                p = min(p_raw * n_pairs, 1.0)

            pvals[f"{labels[i]}-{labels[j]}"] = p
    return pvals


def analyze_peaks(common_x, group_arrays, keys, display_names, eta_sq, sig_mask,
                   prominence, distance, window, min_eta_sq, alpha):
    """Finds peaks on the grand mean, keeps only ones inside a
    cluster-significant region AND passing the effect-size gate, then runs
    Games-Howell post-hoc at the best local wavenumber for each."""
    cond_means = np.vstack([group_arrays[k].mean(axis=0) for k in keys])
    grand_mean = cond_means.mean(axis=0)
    rng = grand_mean.max() - grand_mean.min()
    grand_mean_disp = (grand_mean - grand_mean.min()) / rng if rng != 0 else grand_mean
    peak_idx, _ = find_peaks(grand_mean_disp, prominence=prominence, distance=distance)

    records = []
    for idx in peak_idx:
        lo = max(0, idx - window)
        hi = min(len(sig_mask), idx + window + 1)
        local_sig = np.where(sig_mask[lo:hi])[0]
        if len(local_sig) == 0:
            continue  # not inside a cluster-significant region
        local_eta = eta_sq[lo:hi][local_sig]
        best_local = lo + local_sig[np.argmax(local_eta)]
        if eta_sq[best_local] < min_eta_sq:
            continue  # fails the effect-size gate

        samples = [group_arrays[k][:, best_local] for k in keys]
        means = {k: float(np.mean(s)) for k, s in zip(keys, samples)}
        sems = {k: float(np.std(s, ddof=1) / np.sqrt(len(s))) for k, s in zip(keys, samples)}
        dominant = max(means, key=means.get)
        gh = games_howell_pairwise(samples, keys, alpha)
        sig_pairs = []
        for pair, p in gh.items():
            if p < alpha:
                a, b = pair.split("-", 1)
                sig_pairs.append(f"{display_names[a]}-{display_names[b]}")

        rec = {
            "wavenumber_cm-1": round(float(common_x[best_local]), 1),
            "eta_squared": round(float(eta_sq[best_local]), 3),
        }
        for k in keys:
            rec[f"mean_{display_names[k]}"] = means[k]
            rec[f"sem_{display_names[k]}"] = sems[k]
        rec["dominant_condition"] = display_names[dominant]
        rec["significant_pairs_GamesHowell"] = ", ".join(sig_pairs) if sig_pairs else "(none)"
        records.append(rec)

    cols = ["wavenumber_cm-1", "eta_squared", "dominant_condition", "significant_pairs_GamesHowell"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records).sort_values("wavenumber_cm-1").reset_index(drop=True)


def plot_peak_bar_chart(peaks_df, keys, display_names, colors, title, safe_suffix, output_dir):
    """Grouped bar chart: one group per significant peak, one bar per condition."""
    if peaks_df.empty:
        print(f"  [INFO] No significant peaks to chart for {title}, skipping bar chart.")
        return None

    n_peaks = len(peaks_df)
    n_cond = len(keys)
    x = np.arange(n_peaks)
    width = 0.8 / n_cond

    plt.figure(figsize=(max(6, n_peaks * 1.4), 5))
    for i, k in enumerate(keys):
        label_name = display_names[k]
        means = peaks_df[f"mean_{label_name}"].to_numpy()
        sems = peaks_df[f"sem_{label_name}"].to_numpy()
        offset = (i - (n_cond - 1) / 2) * width
        plt.bar(x + offset, means, width=width, yerr=sems, capsize=3,
                label=label_name, color=colors[k], alpha=0.9)

    plt.xticks(x, [f"{wn:.0f}" for wn in peaks_df["wavenumber_cm-1"]])
    plt.xlabel("Raman Shift (cm$^{-1}$) of significant peak")
    plt.ylabel("Mean normalized intensity $\\pm$ SEM")
    plt.title(f"{title}: peak intensity by condition\n"
              f"(only peaks with eta$^2$>={MIN_ETA_SQUARED}, cluster-significant, shown)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{safe_suffix}_peak_bar_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Peak bar chart saved: {path}")
    return path


def plot_anova_spectra(common_x, group_arrays, keys, display_names, colors, peaks_df,
                        sig_mask, cluster_alpha, min_eta_sq, title, path):
    """Mean +/- SEM spectra per group, with yellow shading over
    cluster-significant regions and dashed markers at effect-size-gated
    peaks -- the ANOVA counterpart of the plain stacked/overlapped plots."""
    plt.figure(figsize=(10, 6))
    all_means = []
    for k in keys:
        mean_y = group_arrays[k].mean(axis=0)
        sem_y = group_arrays[k].std(axis=0, ddof=1) / np.sqrt(group_arrays[k].shape[0])
        all_means.append(mean_y)
        color = colors[k]
        plt.plot(common_x, mean_y, label=f"{display_names[k]} (n={group_arrays[k].shape[0]})",
                 color=color, linewidth=1.5)
        plt.fill_between(common_x, mean_y - sem_y, mean_y + sem_y, color=color, alpha=0.2)
    all_means = np.vstack(all_means)

    sig_idx = np.where(sig_mask)[0]
    if len(sig_idx) > 0:
        splits = np.where(np.diff(sig_idx) > 1)[0] + 1
        for run in np.split(sig_idx, splits):
            plt.axvspan(common_x[run[0]], common_x[run[-1]], color="yellow", alpha=0.15)

    data_max = float(all_means.max())
    data_min = float(min(0.0, all_means.min()))
    headroom = (data_max - data_min) * 0.25
    plt.ylim(data_min - 0.02 * (data_max - data_min), data_max + headroom)
    label_y = data_max + headroom * 0.3
    for _, row in peaks_df.iterrows():
        plt.axvline(row["wavenumber_cm-1"], color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(row["wavenumber_cm-1"], label_y, f"{row['wavenumber_cm-1']:.0f}",
                  ha="center", va="bottom", fontsize=7, rotation=90)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Normalized Intensity (a.u.)")
    plt.title(f"{title}\n(yellow = significant cluster, p<{cluster_alpha}; dashed = peak with "
              f"eta$^2$>={min_eta_sq})", pad=14)
    plt.legend(fontsize=9, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> ANOVA spectra plot saved: {path}")


def run_anova_for_comparison(common_x, group_info, comp, output_dir):
    """Runs the full cluster-permutation ANOVA (+ effect-size gate +
    Games-Howell post-hoc) for one comparison set (e.g. Control vs
    E.coli-Sonication vs S.aureus-Sonication) using the already-pooled,
    preprocessed cell data built earlier in the pipeline. NEW: also runs
    the permutation null-distribution plot, PCA, LDA, and a cross-validated
    classification report + confusion matrix for the same comparison."""
    keys_all = comp["groups"]
    present_keys = [k for k in keys_all if group_info[k]["pooled"] is not None]
    missing_keys = [k for k in keys_all if k not in present_keys]
    suffix = comp["suffix"]

    anova_plot_path = os.path.join(output_dir, f"{suffix}_anova_spectra.png")
    if len(present_keys) < 2:
        plot_blank_placeholder(f"{comp['title']}: ANOVA", anova_plot_path,
                                reason=f"Need >=2 groups with data to run ANOVA "
                                       f"(currently have {len(present_keys)})")
        return

    display_names = {k: group_info[k]["display"] for k in present_keys}
    colors = {k: group_info[k]["color"] for k in present_keys}
    group_arrays = {k: group_info[k]["pooled"] for k in present_keys}

    n_groups = len(present_keys)
    key_to_idx = {k: i for i, k in enumerate(present_keys)}
    X = np.vstack([group_arrays[k] for k in present_keys])
    group_idx = np.concatenate([np.full(group_arrays[k].shape[0], key_to_idx[k]) for k in present_keys])

    print(f"  Running cluster-permutation ANOVA ({N_PERMUTATIONS} permutations)...")
    F_obs, eta_sq, sig_mask, cluster_records, F_thresh, null_max_mass = cluster_permutation_test(
        X, group_idx, n_groups, N_PERMUTATIONS, CLUSTER_ALPHA, CLUSTER_FORMING_ALPHA, RANDOM_SEED
    )
    n_sig_clusters = sum(1 for _, _, p in cluster_records if p < CLUSTER_ALPHA)
    print(f"  -> {len(cluster_records)} candidate cluster(s) found, {n_sig_clusters} significant "
          f"(cluster_alpha={CLUSTER_ALPHA}); {int(sig_mask.sum())}/{len(sig_mask)} points covered.")

    # NEW: permutation null-distribution plot
    null_path = os.path.join(output_dir, f"{suffix}_permutation_null_distribution.png")
    plot_permutation_null_distribution(null_max_mass, cluster_records, CLUSTER_ALPHA,
                                        f"{comp['title']}: cluster-permutation null distribution",
                                        null_path)

    # point-by-point stats table
    stats_df = pd.DataFrame({
        "RamanShift_cm-1": common_x,
        "F_stat": F_obs,
        "eta_squared": eta_sq,
        "in_significant_cluster": sig_mask,
        "passes_effect_size_gate": eta_sq >= MIN_ETA_SQUARED,
    })
    stats_csv = os.path.join(output_dir, f"{suffix}_anova_by_wavenumber.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"  -> Point-by-point ANOVA stats saved: {stats_csv}")

    # cluster summary table
    cluster_summary = pd.DataFrame([
        {
            "start_cm-1": round(float(common_x[run[0]]), 1),
            "end_cm-1": round(float(common_x[run[-1]]), 1),
            "cluster_mass": round(mass, 2),
            "p_value": p,
            "significant": p < CLUSTER_ALPHA,
        }
        for run, mass, p in cluster_records
    ])
    cluster_csv = os.path.join(output_dir, f"{suffix}_significant_clusters.csv")
    cluster_summary.to_csv(cluster_csv, index=False)
    print(f"  -> Cluster summary saved: {cluster_csv}")

    # distinct peaks: cluster-significant AND effect-size gated, with post-hoc
    peaks_df = analyze_peaks(common_x, group_arrays, present_keys, display_names, eta_sq, sig_mask,
                              PEAK_PROMINENCE, PEAK_DISTANCE, PEAK_SEARCH_WINDOW,
                              MIN_ETA_SQUARED, POSTHOC_ALPHA)
    peaks_csv = os.path.join(output_dir, f"{suffix}_significant_peaks.csv")
    peaks_df.to_csv(peaks_csv, index=False)
    print(f"  -> {len(peaks_df)} significant, effect-size-gated peak(s) found "
          f"(eta^2 >= {MIN_ETA_SQUARED}). Saved: {peaks_csv}")

    plot_peak_bar_chart(peaks_df, present_keys, display_names, colors, comp["title"], suffix, output_dir)

    title = f"{comp['title']}: ANOVA across groups"
    if missing_keys:
        missing_names = ", ".join(group_info[k]["display"] for k in missing_keys)
        title += f"\n(missing so far: {missing_names} -- appears automatically once found)"
    plot_anova_spectra(common_x, group_arrays, present_keys, display_names, colors, peaks_df,
                        sig_mask, CLUSTER_ALPHA, MIN_ETA_SQUARED, title, anova_plot_path)

    # NEW: PCA / LDA scatter plots for this comparison
    pca_path = os.path.join(output_dir, f"{suffix}_pca_scores.png")
    plot_pca_scores(X, group_idx, present_keys, key_to_idx, display_names, colors,
                     f"{comp['title']}: PCA of processed spectra (unsupervised)", pca_path)
    lda_path = os.path.join(output_dir, f"{suffix}_lda_scores.png")
    plot_lda_scores(X, group_idx, present_keys, key_to_idx, display_names, colors,
                     f"{comp['title']}: LDA of processed spectra (supervised)", lda_path)

    # NEW: cross-validated classification report + confusion matrix
    evaluate_classification(X, group_idx, present_keys, key_to_idx, display_names, comp["title"],
                             suffix, output_dir, CLASSIFICATION_CV_FOLDS, RANDOM_SEED)

    # NEW: leave-one-replicate-out CV (same as the A549 pipeline). Per-cell
    # (condition, replicate) label, in the SAME row order as X.
    replicate_groups = np.concatenate([
        np.array([f"{k}::{rep_label}" for rep_label, mat in group_info[k]["replicate_data"].items()
                  for _ in range(mat.shape[0])])
        for k in present_keys
    ])
    n_folds = len(np.unique(replicate_groups))
    print(f"  Running leave-one-replicate-out cross-validation ({n_folds} fold(s), "
          f"one per condition-replicate)...")
    evaluate_classification_loro(X, group_idx, replicate_groups, present_keys, key_to_idx, display_names,
                                 comp["title"], suffix, output_dir)
    print(f"  -> Compare the two accuracy/macro-F1 numbers: the LORO one is the stricter, more "
          f"defensible number, since no test cell came from a replicate seen during training.")

    if missing_keys:
        missing_names = ", ".join(group_info[k]["display"] for k in missing_keys)
        print(f"  [INFO] {missing_names} not yet included here -- will appear automatically "
              f"once its folder is found.")
