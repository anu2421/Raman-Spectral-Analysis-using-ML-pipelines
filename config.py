"""
All settings in one place. Choose the experiment below, then run:
    python main.py                   full analysis (A549_* / K562_* experiments)
    python stacked_K562_vs_A549.py   stacked plots of both cell lines (K562_vs_A549_* experiments)
"""

import os

# ========================================================================
# ============================ USER SETTINGS ============================
# ========================================================================
# Folder that contains the 5 condition folders of each cell line
# (CONTROL, E.coli heat / sonication, S.aureus heat / sonication).
DATA_DIR_A549 = os.path.join("data", "A549")
DATA_DIR_K562 = os.path.join("data", "K562")      # the "All data" folder

# Which analysis to run:
#   A549_sonicated | A549_heat | K562_sonicated | K562_heat | K562_vs_A549_sonicated | K562_vs_A549_heat
EXPERIMENT = "A549_sonicated"

NORM_LABELS = {"area": "AREA", "l2": "L2"}
SHIFT_LABELS = {"global": "GLOBAL", "peak1445": "1445"}
METHODS = [("area", "global")]
REPLICATE_MERGE = {}
COMPARISONS = []
# Fixed reference for the global shift: "own_control" = this cell line's own
# Control mean; "both_controls" = mean of K562 + A549 Controls together.
GLOBAL_SHIFT_REFERENCE = "own_control"

if EXPERIMENT == "A549_sonicated":
    DATASETS = {
        "A549": DATA_DIR_A549,
    }

    OUTPUT_DIR = os.path.join("results", "A549_sonication_3groups_AREA_GLOBAL_full_analysis")

    # Folder matching (fuzzy: lowercased, punctuation/spaces removed, must
    # contain ALL keywords).
    # "E.coli sonication" -> "ecolisonication", "S.aureus sonication" -> "saureussonication"
    GROUP_KEYWORDS = {
        "control": ["control"],
        "ecoli_sonication": ["ecoli", "son"],
        "saureus_sonication": ["aureus", "son"],
    }
    GROUP_DISPLAY = {"control": "Control", "ecoli_sonication": "E.coli sonicated",
                     "saureus_sonication": "S.aureus sonicated"}

    # Replicate relabelling per group, applied before anything else.
    # E.coli sonicated: nearly all files are R2 with a few R1 -> all taken as R2
    # (one replicate).
    REPLICATE_MERGE = {
        "ecoli_sonication": {"R1": "R2"},
    }

    COLORS = {
        ("A549", "control"): "0.35",
        ("A549", "ecoli_sonication"): "tab:red",
        ("A549", "saureus_sonication"): "tab:blue",
    }

    METHODS = [("area", "global")]

    COMPARISONS = [
        {"suffix": "A549_Sonicated_Control_vs_Ecoli_vs_Saureus",
         "title": "A549 sonicated: Control vs E.coli vs S.aureus",
         "groups": ["A549_control", "A549_ecoli_sonication", "A549_saureus_sonication"]},
        # E.coli has only one replicate, so its leave-one-replicate-out result is
        # not meaningful (when its only replicate is held out, the model has never
        # seen E.coli). This 2-group comparison gives a valid LORO for S.aureus.
        {"suffix": "A549_Sonicated_Saureus_vs_Control",
         "title": "A549 sonicated: S.aureus vs Control",
         "groups": ["A549_control", "A549_saureus_sonication"]},
    ]

    ALL_GROUPS_TITLE = "Control vs E.coli vs S.aureus sonicated"
    ALL_GROUPS_FILE = "Control_Ecoli_Saureus_sonicated"

elif EXPERIMENT == "A549_heat":
    DATASETS = {
        "A549": DATA_DIR_A549,
    }

    OUTPUT_DIR = os.path.join("results", "A549_heat_inactivated_AREA_GLOBAL_full_analysis")

    # Folder matching (fuzzy: lowercased, punctuation/spaces removed, must
    # contain ALL keywords).
    # "E.COLI Heat inactivation" -> "ecoliheatinactivation", "S.aureus heat inactivation" -> "saureusheatinactivation"
    GROUP_KEYWORDS = {
        "control": ["control"],
        "ecoli_heat": ["ecoli", "heat"],
        "saureus_heat": ["aureus", "heat"],
    }
    GROUP_DISPLAY = {"control": "Control", "ecoli_heat": "E.coli heat-inactivated",
                     "saureus_heat": "S.aureus heat-inactivated"}

    COLORS = {
        ("A549", "control"): "0.35",
        ("A549", "ecoli_heat"): "firebrick",
        ("A549", "saureus_heat"): "navy",
    }

    METHODS = [("area", "global")]

    COMPARISONS = [
        {"suffix": "A549_Heat_inactivated_Control_vs_Ecoli_vs_Saureus",
         "title": "A549 heat-inactivated: Control vs E.coli vs S.aureus",
         "groups": ["A549_control", "A549_ecoli_heat", "A549_saureus_heat"]},
    ]

    ALL_GROUPS_TITLE = "Control vs E.coli vs S.aureus heat-inactivated"
    ALL_GROUPS_FILE = "Control_Ecoli_Saureus_heat"

elif EXPERIMENT == "K562_sonicated":
    DATASETS = {
        "K562": DATA_DIR_K562,
    }

    OUTPUT_DIR = os.path.join("results", "K562_sonication_AREA_GLOBAL_full_analysis")

    # Folder matching (fuzzy: lowercased, punctuation/spaces removed, must
    # contain ALL keywords).
    GROUP_KEYWORDS = {
        "control": ["control"],
        "saureus_sonication": ["aureus", "son"],
    }
    GROUP_DISPLAY = {"control": "Control", "saureus_sonication": "S.aureus sonicated"}

    COLORS = {
        ("K562", "control"): "0.35",
        ("K562", "saureus_sonication"): "tab:blue",
    }

    METHODS = [("area", "global")]

    COMPARISONS = [
        {"suffix": "K562_Saureus_sonicated_vs_Control",
         "title": "K562: S.aureus sonicated vs Control",
         "groups": ["K562_control", "K562_saureus_sonication"]},
    ]

    ALL_GROUPS_TITLE = "Control vs S.aureus sonicated"
    ALL_GROUPS_FILE = "Control_vs_Saureus_sonicated"

elif EXPERIMENT == "K562_heat":
    DATASETS = {
        "K562": DATA_DIR_K562,
    }

    OUTPUT_DIR = os.path.join("results", "K562_heat_inactivated_AREA_GLOBAL_full_analysis")

    # Folder matching (fuzzy: lowercased, punctuation/spaces removed, must
    # contain ALL keywords).
    # "E.COLI-heat inactivated" -> "ecoliheatinactivated", "S.aureus- heat inactivated" -> "saureusheatinactivated"
    GROUP_KEYWORDS = {
        "control": ["control"],
        "ecoli_heat": ["ecoli", "heat"],
        "saureus_heat": ["aureus", "heat"],
    }
    GROUP_DISPLAY = {"control": "Control", "ecoli_heat": "E.coli heat-inactivated",
                     "saureus_heat": "S.aureus heat-inactivated"}

    COLORS = {
        ("K562", "control"): "0.35",
        ("K562", "ecoli_heat"): "tab:orange",
        ("K562", "saureus_heat"): "tab:purple",
    }

    METHODS = [("area", "global")]

    COMPARISONS = [
        {"suffix": "K562_Heat_inactivated_Control_vs_Ecoli_vs_Saureus",
         "title": "K562 heat-inactivated: Control vs E.coli vs S.aureus",
         "groups": ["K562_control", "K562_ecoli_heat", "K562_saureus_heat"]},
    ]

    ALL_GROUPS_TITLE = "Control vs E.coli vs S.aureus heat-inactivated"
    ALL_GROUPS_FILE = "Control_Ecoli_Saureus_heat"

elif EXPERIMENT == "K562_vs_A549_sonicated":
    DATASETS = {
        "K562": DATA_DIR_K562,
        "A549": DATA_DIR_A549,
    }

    OUTPUT_DIR = os.path.join("results", "K562_A549_sonication_stacked_plots")

    # Folder matching (fuzzy: lowercased, punctuation/spaces removed, must
    # contain ALL keywords). Works for both "S.aureus- sonication" (K562) and
    # "S.aureus sonication" (A549).
    GROUP_KEYWORDS = {
        "control": ["control"],
        "saureus_sonication": ["aureus", "son"],
    }
    GROUP_DISPLAY = {"control": "Control", "saureus_sonication": "S.aureus sonicated"}

    # Line colours for every (dataset, group) trace.
    COLORS = {
        ("K562", "control"): "0.35",
        ("K562", "saureus_sonication"): "tab:blue",
        ("A549", "control"): "tab:olive",
        ("A549", "saureus_sonication"): "navy",
    }

    # The 4 methods: (normalization, shift correction)
    METHODS = [
        ("area", "global"),
        ("area", "peak1445"),
        ("l2", "global"),
        ("l2", "peak1445"),
    ]

    GLOBAL_SHIFT_REFERENCE = "both_controls"
    CROSS_ALL_ORDER = [("K562", "control"), ("K562", "saureus_sonication"),
                       ("A549", "control"), ("A549", "saureus_sonication")]
    CROSS_ALL_TITLE = "Control & S.aureus sonicated"
    CROSS_ALL_FILE = "Control_and_Saureus_sonicated"

elif EXPERIMENT == "K562_vs_A549_heat":
    DATASETS = {
        "K562": DATA_DIR_K562,
        "A549": DATA_DIR_A549,
    }

    OUTPUT_DIR = os.path.join("results", "K562_A549_heat_inactivated_stacked_plots")

    # Folder matching (fuzzy: lowercased, punctuation/spaces removed, must
    # contain ALL keywords). Works for "E.COLI-heat inactivated" /
    # "S.aureus- heat inactivated" (K562) and "E.COLI Heat inactivation" /
    # "S.aureus heat inactivation" (A549).
    GROUP_KEYWORDS = {
        "control": ["control"],
        "ecoli_heat": ["ecoli", "heat"],
        "saureus_heat": ["aureus", "heat"],
    }
    GROUP_DISPLAY = {"control": "Control", "ecoli_heat": "E.coli heat inactivated",
                     "saureus_heat": "S.aureus heat inactivated"}

    # Line colours for every (dataset, group) trace.
    COLORS = {
        ("K562", "control"): "0.35",
        ("K562", "ecoli_heat"): "tab:orange",
        ("K562", "saureus_heat"): "tab:purple",
        ("A549", "control"): "tab:olive",
        ("A549", "ecoli_heat"): "firebrick",
        ("A549", "saureus_heat"): "navy",
    }

    # The 4 methods: (normalization, shift correction)
    METHODS = [
        ("area", "global"),
        ("area", "peak1445"),
        ("l2", "global"),
        ("l2", "peak1445"),
    ]

    GLOBAL_SHIFT_REFERENCE = "both_controls"
    CROSS_ALL_ORDER = [("K562", "control"), ("K562", "ecoli_heat"), ("K562", "saureus_heat"),
                       ("A549", "control"), ("A549", "ecoli_heat"), ("A549", "saureus_heat")]
    CROSS_ALL_TITLE = "Control, E.coli & S.aureus heat inactivated"
    CROSS_ALL_FILE = "Control_Ecoli_Saureus_heat"

else:
    raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")


# --- Preprocessing (same values as your pipelines) ----------------------
SPIKE_WINDOW = 5
SPIKE_THRESHOLD = 7.0
ROI_MIN_CM1 = 600
ROI_MAX_CM1 = 1800
SG_WINDOW = 11
SG_POLYORDER = 3
ALS_LAMBDA = 1e5
ALS_P = 0.01
ALS_NITER = 10
COMMON_GRID_POINTS = 1000

REPLICATE_REGEX = r'^[Rr][\s\-_]?(\d+)$'               # R1 / R2 / R3 subfolders
FILENAME_REPLICATE_REGEX = r'^[Rr][_\-\s]?(\d+)[_\-\s]'  # "R1_..." flat files

# --- Shift correction settings (same as before) ---------------------------
MAX_ALLOWED_SHIFT_CM1 = 15.0
PEAK1445_TARGET_CM1 = 1445.0
PEAK1445_SEARCH_MIN_CM1 = 1420.0
PEAK1445_SEARCH_MAX_CM1 = 1470.0
PEAK1445_LEVEL = "cell"

GLOBAL_SHIFT_STEP_CM1 = 1.0
GLOBAL_SHIFT_MAX_CM1 = 15
GLOBAL_SHIFT_MATCH_MIN_CM1 = 620.0
GLOBAL_SHIFT_MATCH_MAX_CM1 = 1780.0
GLOBAL_SHIFT_LEVEL = "replicate"
GLOBAL_SHIFT_ITERATIONS = 2

# --- ANOVA / peak detection (same as your pipeline) -----------------------
PEAK_PROMINENCE = 0.03
PEAK_DISTANCE = 15
PEAK_SHIFT_MATCH_TOLERANCE_CM1 = 15.0
PEAK_SHIFT_BOOTSTRAP_N = 500
PEAK_SHIFT_BOOTSTRAP_SEED = 4242
PEAK_SHIFT_REFINE_WINDOW = 8
MIN_ETA_SQUARED = 0.06
N_PERMUTATIONS = 1000
CLUSTER_ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
RANDOM_SEED = 42
PEAK_SEARCH_WINDOW = 5
POSTHOC_ALPHA = 0.05

# --- Classification -------------------------------------------------------
CLASSIFICATION_CV_FOLDS = 5
# ========================================================================
# ========================================================================


# ========================================================================
# ============ SETTINGS FOR norm_shift_comparison.py ====================
# ========================================================================
# Which cell line to run the L2 / AREA x none / 1445 / global comparison on.
NORM_SHIFT_CELL_LINE = "K562"      # "K562" or "A549"

SPIKE_REMOVAL_ENABLED = True
ROI_ENABLED = True              # MUST stay True: every spectrum needs a shared x-axis
SMOOTHING_ENABLED = True
BASELINE_ENABLED = True
NORMALIZATION_ENABLED = True

if NORM_SHIFT_CELL_LINE == "K562":
    ROOT_FOLDER = DATA_DIR_K562
    GROUPS = {
        "control":            {"keywords": ["control"],        "display": "Control",
                                "color": "0.35"},
        "ecoli_sonication":    {"keywords": ["ecoli", "son"],    "display": "E.coli - Sonication",
                                "color": "tab:red"},
        "ecoli_heat":          {"keywords": ["ecoli", "heat"],   "display": "E.coli - Heat Inactivated",
                                "color": "tab:orange"},
        "saureus_sonication":  {"keywords": ["aureus", "son"],   "display": "S.aureus - Sonication",
                                "color": "tab:blue"},
        "saureus_heat":        {"keywords": ["aureus", "heat"],  "display": "S.aureus - Heat Inactivated",
                                "color": "tab:purple"},
    }
else:
    ROOT_FOLDER = DATA_DIR_A549
    GROUPS = {
        "control":            {"keywords": ["control"],            "display": "Control",              "color": "0.35"},
        "ecoli_heat":         {"keywords": ["ecoli", "heat"],       "display": "E.coli (Heat)",         "color": "firebrick"},
        "ecoli_sonication":   {"keywords": ["ecoli", "sonication"], "display": "E.coli (Sonication)",   "color": "tab:red"},
        "saureus_heat":       {"keywords": ["aureus", "heat"],      "display": "S.aureus (Heat)",       "color": "navy"},
        "saureus_sonication": {"keywords": ["aureus", "sonication"],"display": "S.aureus (Sonication)", "color": "tab:blue"},
    }
# ========================================================================
