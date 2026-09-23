"""Default configuration for the Cu3MS4 Bayesian optimization workflow.

Campaign notebooks override CURRENT_PRECURSORS and TRANSFER_MODE in-cell.
"""

import math


# Synthesis system constants
CU_PRECURSOR_MMOL = 0.25     # Fixed Cu precursor amount (mmol)
TOTAL_VOLUME_ML = 12.0       # Fixed total volume (mL)
DDT_MMOL_PER_ML = 4.17       # DDT conversion (MW=202.4, density=0.845)
OAM_MMOL_PER_ML = 3.04       # OAm conversion (MW=267.5, density=0.813)


# Defaults used when adding chemical features. Notebooks override these.
# The column "VOacac" is Group-5 metal precursor mmol for all campaigns.
CURRENT_PRECURSORS = {
    'Cu_precursor': 'CuCl',
    'S_precursor': 'DDT',
    'Metal_Precursor': 'VO(acac)2',
}


# Search bounds for raw synthesis parameters.
RAW_BOUNDS = {
    "Temp":   (260.0, 310.0),   # °C
    "Time":   (8.0, 90.0),      # min
    "VOacac": (0.08, 0.66),     # mmol
    "DDT":    (1.0, 5.0),       # mL
    "OAm":    (1.0, 7.0),       # mL
}

# Bounds for derived (chemical) features, derived analytically from RAW_BOUNDS so
# that any sample inside them back-transforms to within the experimental box.
_time_lo, _time_hi     = RAW_BOUNDS["Time"]
_voacac_lo, _voacac_hi = RAW_BOUNDS["VOacac"]
_ddt_lo, _ddt_hi       = RAW_BOUNDS["DDT"]
_oam_lo, _oam_hi       = RAW_BOUNDS["OAm"]

CHEMICAL_BOUNDS = {
    # Strict inversion of the Time bounds.
    "log_Time":          (math.log10(_time_lo),   math.log10(_time_hi)),

    # Cu_V_ratio = Cu_precursor / VOacac (bounds invert: larger VOacac => smaller ratio).
    "Cu_V_ratio":        (CU_PRECURSOR_MMOL / _voacac_hi,  CU_PRECURSOR_MMOL / _voacac_lo),

    # Metal_Conc (mM) = (Cu_precursor + VOacac) / total_vol * 1000.
    "Metal_Conc":        ((CU_PRECURSOR_MMOL + _voacac_lo) / TOTAL_VOLUME_ML * 1000,
                          (CU_PRECURSOR_MMOL + _voacac_hi) / TOTAL_VOLUME_ML * 1000),

    # S_Metal_ratio = DDT_mmol / total_metal — worst-case lo/hi from DDT and VOacac extremes.
    "S_Metal_ratio":     (_ddt_lo * DDT_MMOL_PER_ML / (CU_PRECURSOR_MMOL + _voacac_hi),
                          _ddt_hi * DDT_MMOL_PER_ML / (CU_PRECURSOR_MMOL + _voacac_lo)),

    # Ligand_Metal_ratio = OAm_mmol / total_metal.
    "Ligand_Metal_ratio":(_oam_lo * OAM_MMOL_PER_ML / (CU_PRECURSOR_MMOL + _voacac_hi),
                          _oam_hi * OAM_MMOL_PER_ML / (CU_PRECURSOR_MMOL + _voacac_lo)),
}

# Rounding precision for lab-practical recommendations
RAW_ROUNDING = {
    "Temp":   0,      # 1 °C
    "Time":   0,      # 1 min
    "VOacac": 3,      # 0.001 mmol
    "DDT":    1,      # 0.1 mL
    "OAm":    1,      # 0.1 mL
}

# Factor and feature definitions
RAW_FACTORS = ["Temp", "Time", "VOacac", "DDT", "OAm"]

CHEM_FEATURES_BASIC = [
    "Cu_V_ratio",          # Stoichiometry
    "S_Metal_ratio",       # Sulfur excess
    "Ligand_Metal_ratio",  # Capping density
    "Metal_Conc",          # Total metal concentration (mM)
    "log_Time",            # Linearized kinetics
]

# Enhanced chemical features. Toggled on individually; require
# chemical_constants.py. The Cu and Metal precursor descriptors are used
# for transfer learning across precursor types.
ENHANCED_FEATURE_CONFIG = {
    'effective_dielectric': False,
    'Cu_precursor_hardness': False,   # Cu precursor Pearson hardness
    'Cu_hsab_mismatch': False,        # Cu precursor |eta_cation - eta_anion|
    'Metal_ionic_potential': False,   # Group-5 metal Z/r (Shannon, precursor basis)
    'Metal_oxophilicity': False,      # Group-5 metal oxophilicity (MO2 vs MS2)
    'S_BDE': False,
}

# Default feature set. DDT and Metal_Conc are left out because they are
# nearly collinear with S_Metal_ratio and Cu_V_ratio when Cu amount is fixed.
SYNTHESIS_FEATURES = [
    "Temp",
    "Cu_V_ratio",
    "S_Metal_ratio",
    "Ligand_Metal_ratio",
    "log_Time",
]

# Transfer learning. When enabled, precursor descriptors are appended and
# GPs use an ARD kernel. target_*_precursor is what new experiments use;
# other precursors in the training set are the transfer source.
TRANSFER_MODE = {
    'enabled': True,
    'vary_cu_precursor': True,
    'vary_metal_precursor': False,
    'target_cu_precursor': 'CuCl',
    'target_metal_precursor': 'VO(acac)2',
    'use_ard_kernel': True,
}

# Auto-sync: turning on a transfer axis automatically enables the matching
# enhanced features so add_chemical_features() produces the right columns.
if TRANSFER_MODE.get('vary_cu_precursor'):
    ENHANCED_FEATURE_CONFIG['Cu_precursor_hardness'] = True
    ENHANCED_FEATURE_CONFIG['Cu_hsab_mismatch'] = True
if TRANSFER_MODE.get('vary_metal_precursor'):
    ENHANCED_FEATURE_CONFIG['Metal_ionic_potential'] = True
    ENHANCED_FEATURE_CONFIG['Metal_oxophilicity'] = True

TRANSFER_FEATURES = list(SYNTHESIS_FEATURES)
if TRANSFER_MODE.get('vary_cu_precursor'):
    TRANSFER_FEATURES += ['Cu_precursor_hardness', 'Cu_hsab_mismatch']
if TRANSFER_MODE.get('vary_metal_precursor'):
    TRANSFER_FEATURES += ['Metal_ionic_potential', 'Metal_oxophilicity']

# Constant per precursor — these are not sampled during LHS.
PRECURSOR_DESCRIPTOR_FEATURES = {
    'Cu_precursor_hardness', 'Cu_hsab_mismatch',
    'Metal_ionic_potential', 'Metal_oxophilicity',
}

# With only 2 precursors on an axis, the two descriptors are collinear
# (one binary contrast). Keep the primary until >= 3 precursors are pooled.
MIN_PRECURSORS_FOR_INDEPENDENT_DESCRIPTORS = 3

TRANSFER_DESCRIPTOR_AXES = {
    'cu': {
        'precursor_col': 'Cu_precursor',
        'descriptors': ['Cu_precursor_hardness', 'Cu_hsab_mismatch'],
        'primary': 'Cu_precursor_hardness',
    },
    'metal': {
        'precursor_col': 'Metal_precursor',
        'descriptors': ['Metal_ionic_potential', 'Metal_oxophilicity'],
        'primary': 'Metal_oxophilicity',
    },
}


# Objectives and feasibility.
# Size is the only regression objective in the acquisition. The CV and
# Squareness GPs are kept for predictions but do not enter the acquisition
# score.
OBJECTIVES = ["Size", "CV", "Squareness"]
FEAS_COLS = ["HasProduct", "PhasePure"]
POLYMORPH_COL = "Polymorph"
CUBIC_LABEL = "cubic"


# Squareness binning (feasibility-style morphology constraint).
#   'multipod'      — non-cubic polymorph
#   'highly_cubic'  — cubic with Squareness >= Otsu threshold
#   'poorly_cubic'  — cubic with Squareness <  Otsu threshold
# The Otsu threshold is computed once from the initial CSV import and persisted
# to disk so it never drifts as new experiments come in.
SQUARENESS_BINS = ['multipod', 'highly_cubic', 'poorly_cubic']
DEFAULT_SQUARENESS_BIN = 'highly_cubic'
OTSU_N_THRESHOLDS = 1000


# Optimization settings.
DEFAULT_FEATURE_MODE = 'synthesis'
N_RECOMMENDATIONS = 2
MIN_COMPLETED_FOR_ERROR_MODEL = 10


# Plot colors and style.
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2E8B57',
    'warning': '#E63946',
    'neutral': '#6C757D',
    'pending': '#F4A261',
    'completed': '#2A9D8F',
    'skipped': '#ADB5BD',
}

PUBLICATION_STYLE = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
}
