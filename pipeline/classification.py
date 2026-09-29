"""
Step 5 - PCA / LDA score plots and LDA classification
(cell-level k-fold CV and leave-one-replicate-out CV) with confusion matrices.
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
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    HAS_SKLEARN_DA = True
except ImportError:
    HAS_SKLEARN_DA = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneGroupOut
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False

from config import *


def plot_pca_scores(X, group_idx, keys, key_to_idx, display_names, colors, title, path):
    if not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not found -> skipping PCA plot.")
        return None
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(7, 6))
    for k in keys:
        mask = group_idx == key_to_idx[k]
        plt.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.7,
                    color=colors[k], label=display_names[k])
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    plt.title(title)
    plt.legend(fontsize=9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> PCA scores plot saved: {path}")
    return path


def plot_lda_scores(X, group_idx, keys, key_to_idx, display_names, colors, title, path):
    if not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not found -> skipping LDA plot.")
        return None
    n_classes = len(keys)
    n_comp = min(2, n_classes - 1)
    if n_comp < 1:
        print("  [WARN] Need >=2 conditions for LDA, skipping.")
        return None
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    scores = lda.fit_transform(X, group_idx)
    plt.figure(figsize=(7, 6))
    rng = np.random.default_rng(0)
    for k in keys:
        mask = group_idx == key_to_idx[k]
        if n_comp == 1:
            jitter = rng.normal(0, 0.02, size=int(mask.sum()))
            plt.scatter(scores[mask, 0], jitter, s=18, alpha=0.7, color=colors[k], label=display_names[k])
        else:
            plt.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.7, color=colors[k], label=display_names[k])
    plt.xlabel("LD1")
    plt.ylabel("LD2" if n_comp == 2 else "(vertical jitter for display only)")
    plt.title(title)
    plt.legend(fontsize=9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> LDA scores plot saved: {path}")
    return path


def plot_confusion_matrix(cm, display_names_list, title, path, accuracy, macro_f1):
    plt.figure(figsize=(6, 5.5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="cells (count)")
    n = len(display_names_list)
    plt.xticks(range(n), display_names_list, rotation=30, ha="right")
    plt.yticks(range(n), display_names_list)
    plt.xlabel("Predicted condition")
    plt.ylabel("True condition")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.title(f"{title}\naccuracy={accuracy:.2f}, macro F1={macro_f1:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Confusion matrix plot saved: {path}")
    return path


def evaluate_classification(X, group_idx, keys, key_to_idx, display_names, title_prefix,
                             safe_prefix, output_dir, n_splits, seed):
    if not HAS_SKLEARN_METRICS or not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not fully available -> skipping classification report.")
        return None, None

    class_counts = {k: int((group_idx == key_to_idx[k]).sum()) for k in keys}
    n_splits_eff = max(2, min(n_splits, min(class_counts.values())))
    if n_splits_eff < 2:
        print("  [WARN] Not enough samples per class for cross-validated classification, skipping.")
        return None, None

    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X, group_idx, cv=cv)

    label_order = [key_to_idx[k] for k in keys]
    display_list = [display_names[k] for k in keys]
    cm = confusion_matrix(group_idx, y_pred, labels=label_order)
    report_dict = classification_report(group_idx, y_pred, labels=label_order,
                                         target_names=display_list, output_dict=True, zero_division=0)

    accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    report_no_acc = {k: v for k, v in report_dict.items() if k != "accuracy"}
    report_df = pd.DataFrame(report_no_acc).transpose()
    report_df.loc["accuracy"] = {"precision": np.nan, "recall": np.nan,
                                  "f1-score": accuracy, "support": cm.sum()}

    report_csv = os.path.join(output_dir, f"{safe_prefix}_classification_report.csv")
    report_df.to_csv(report_csv)
    print(f"  -> Classification report saved ({n_splits_eff}-fold stratified CV, LDA classifier): "
          f"{report_csv} (accuracy={accuracy:.2f}, macro F1={macro_f1:.2f})")

    cm_path = os.path.join(output_dir, f"{safe_prefix}_confusion_matrix.png")
    plot_confusion_matrix(cm, display_list, f"{title_prefix}: LDA confusion matrix (cross-validated)",
                           cm_path, accuracy, macro_f1)
    return report_df, cm


def evaluate_classification_loro(X, group_idx, replicate_groups, keys, key_to_idx, display_names,
                                  title_prefix, safe_prefix, output_dir):
    """
    NEW (ported from the A549 pipeline): leave-one-replicate-out (LORO)
    cross-validation. Each fold holds out EVERY cell from one
    (condition, replicate) pair, trains on all other replicates, and
    predicts only the held-out replicate's cells.

    Stricter than cell-level k-fold CV, which can put cells from the same
    replicate/day in both training and test folds and so partly reward the
    model for recognising that replicate's technical signature (batch
    effect, instrument drift) instead of real biological difference.
    """
    if not HAS_SKLEARN_METRICS or not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not fully available -> skipping LORO classification report.")
        return None, None

    unique_groups = np.unique(replicate_groups)
    if len(unique_groups) < 2:
        print("  [WARN] Need >=2 replicate groups for leave-one-replicate-out CV, skipping.")
        return None, None

    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X, group_idx, cv=logo, groups=replicate_groups)

    label_order = [key_to_idx[k] for k in keys]
    display_list = [display_names[k] for k in keys]
    cm = confusion_matrix(group_idx, y_pred, labels=label_order)
    report_dict = classification_report(group_idx, y_pred, labels=label_order,
                                        target_names=display_list, output_dict=True, zero_division=0)

    accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    report_no_acc = {k: v for k, v in report_dict.items() if k != "accuracy"}
    report_df = pd.DataFrame(report_no_acc).transpose()
    report_df.loc["accuracy"] = {"precision": np.nan, "recall": np.nan,
                                 "f1-score": accuracy, "support": cm.sum()}

    report_csv = os.path.join(output_dir, f"{safe_prefix}_classification_report_LORO.csv")
    report_df.to_csv(report_csv)
    print(f"  -> LORO classification report saved ({len(unique_groups)}-fold leave-one-replicate-out, "
          f"LDA classifier): {report_csv} (accuracy={accuracy:.2f}, macro F1={macro_f1:.2f})")

    cm_path = os.path.join(output_dir, f"{safe_prefix}_confusion_matrix_LORO.png")
    plot_confusion_matrix(cm, display_list,
                          f"{title_prefix}: LDA confusion matrix (leave-one-replicate-out CV)",
                          cm_path, accuracy, macro_f1)
    return report_df, cm
