"""Configuration for the Cu3MS4 Bayesian optimization workflow.

Edit this file to change experimental parameters, feature settings,
optimization behavior, or plot styling.

Supports three material systems via ``CURRENT_PRECURSORS`` / ``TRANSFER_MODE``:
    Cu3VS4    (default)
    Cu3NbS4   (set ``Metal_Precursor`` to 'NbCl5')
    Cu3TaS4   (set ``Metal_Precursor`` to 'TaCl5')

Multiple Cu precursors are supported for transfer learning:
    CuI, CuBr, CuCl, Cu(OAc), Cu(OAc)2, CuCl2
Set ``Cu_precursor`` in ``CURRENT_PRECURSORS`` and configure
``TRANSFER_MODE`` to enable precursor-varying descriptors.
"""

import math


# Synthesis system constants
CU_PRECURSOR_MMOL = 0.25     # Fixed Cu precursor amount (mmol)
TOTAL_VOLUME_ML = 12.0       # Fixed total volume (mL)
DDT_MMOL_PER_ML = 4.17       # DDT conversion (MW=202.4, density=0.845)
OAM_MMOL_PER_ML = 3.04       # OAm conversion (MW=267.5, density=0.813)


# Precursor choices used by chemical feature calculation.
# The raw column "VOacac" is kept for backward compatibility with old CSV data:
# it refers to the Group-5 metal precursor mmol regardless of which metal is used.
CURRENT_PRECURSORS = {
    'Cu_precursor': 'CuBr',
    'S_precursor': 'DDT',
    'Metal_Precursor': 'VO(acac)2',  # 'NbCl5' or 'TaCl5' for other campaigns
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
    'Metal_ionic_potential': False,   # Group-5 metal Z/r (Shannon)
    'Metal_hsab_mismatch': False,     # Group-5 metal |Z/r - eta_anion|
    'S_BDE': False,
}

# Synthesis-driven feature set: 5 mechanistically orthogonal features.
#
# DDT and Metal_Conc are excluded to avoid severe collinearity:
#   DDT vs S_Metal_ratio   — r > 0.95 (CuI is fixed so S_Metal_ratio is
#                            essentially DDT * const).
#   Cu_V_ratio vs Metal_Conc — r ~ -0.99 (both depend only on VOacac).
# Keeping collinear pairs destabilises GP lengthscales; the chemical ratios
# already carry the relevant physical information.
#
#   Temp               — Arrhenius nucleation; dominant CV driver
#   Cu_V_ratio         — Metal stoichiometry; controls phase purity
#   S_Metal_ratio      — Sulfur excess; controls crystallisation / shape
#   Ligand_Metal_ratio — OAm passivation density; growth-rate control
#   log_Time           — Linearised reaction extent; Ostwald ripening
SYNTHESIS_FEATURES = [
    "Temp",
    "Cu_V_ratio",
    "S_Metal_ratio",
    "Ligand_Metal_ratio",
    "log_Time",
]

# Transfer learning mode.
#
# When enabled, precursor-specific descriptors are appended to the feature set
# and the GP regressors switch to an ARD kernel so synthesis features and
# precursor descriptors can have separate lengthscales.
#
#   vary_cu_precursor     — adds Cu_precursor_hardness, Cu_hsab_mismatch
#   vary_metal_precursor  — adds Metal_ionic_potential, Metal_hsab_mismatch
#
# `target_*_precursor` is what new experiments will use; existing data from
# other precursors becomes the transfer training signal.
TRANSFER_MODE = {
    'enabled': True,
    'vary_cu_precursor': True,
    'vary_metal_precursor': False,
    'target_cu_precursor': 'CuBr',
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
    ENHANCED_FEATURE_CONFIG['Metal_hsab_mismatch'] = True

TRANSFER_FEATURES = list(SYNTHESIS_FEATURES)
if TRANSFER_MODE.get('vary_cu_precursor'):
    TRANSFER_FEATURES += ['Cu_precursor_hardness', 'Cu_hsab_mismatch']
if TRANSFER_MODE.get('vary_metal_precursor'):
    TRANSFER_FEATURES += ['Metal_ionic_potential', 'Metal_hsab_mismatch']

# Constant per precursor — these are not sampled during LHS.
PRECURSOR_DESCRIPTOR_FEATURES = {
    'Cu_precursor_hardness', 'Cu_hsab_mismatch',
    'Metal_ionic_potential', 'Metal_hsab_mismatch',
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
