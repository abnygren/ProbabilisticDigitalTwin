"""
Configuration for Cu₃VS₄ Bayesian Optimization

Edit this file to change experimental parameters, feature settings,
optimization behavior, or plot styling. This is the single "knobs and dials"
file — everything user-facing lives here.
"""

# =============================================================================
# SYNTHESIS SYSTEM CONSTANTS
# =============================================================================
CUI_MMOL = 0.25              # Fixed CuI amount (mmol)
TOTAL_VOLUME_ML = 12.0       # Fixed total volume (mL)
DDT_MMOL_PER_ML = 4.17       # DDT conversion (MW=202.4, density=0.845)
OAM_MMOL_PER_ML = 3.04       # OAm conversion (MW=267.5, density=0.813)

# =============================================================================
# PRECURSOR CHOICES (used by chemical feature calculation)
# =============================================================================
CURRENT_PRECURSORS = {
    'Cu_precursor': 'CuI',
    'S_precursor': 'DDT',
    'V_precursor': 'VO(acac)2',
}

# =============================================================================
# SYNTHESIS PARAMETER BOUNDS (what the optimizer searches over)
# =============================================================================
RAW_BOUNDS = {
    "Temp":   (260.0, 310.0),   # °C
    "Time":   (8.0, 90.0),      # min
    "VOacac": (0.08, 0.66),     # mmol
    "DDT":    (1.0, 5.0),       # mL
    "OAm":    (1.0, 7.0),       # mL
}

# Rounding precision for lab-practical recommendations
RAW_ROUNDING = {
    "Temp":   0,      # 1 °C
    "Time":   0,      # 1 min
    "VOacac": 3,      # 0.001 mmol
    "DDT":    1,      # 0.1 mL
    "OAm":    1,      # 0.1 mL
}

# =============================================================================
# FACTOR & FEATURE DEFINITIONS
# =============================================================================
RAW_FACTORS = ["Temp", "Time", "VOacac", "DDT", "OAm"]

CHEM_FEATURES_BASIC = [
    "Cu_V_ratio",         # Stoichiometry
    "S_Metal_ratio",      # Sulfur excess
    "Ligand_Metal_ratio", # Capping density
    "Metal_Conc",         # Total metal concentration (mM)
    "log_Time",           # Linearized kinetics
]

# Enhanced chemical features (toggle on/off).
# These require chemical_constants.py and are computed only when enabled.
ENHANCED_FEATURE_CONFIG = {
    'effective_dielectric': True,
    'Cu_precursor_hardness': False,
    'hsab_mismatch': False,
    'S_BDE': False,
}

# =============================================================================
# OBJECTIVES & FEASIBILITY
# =============================================================================
OBJECTIVES = ["Size", "GSD", "Squareness"]
FEAS_COLS = ["HasProduct", "PhasePure"]
POLYMORPH_COL = "Polymorph"
CUBIC_LABEL = "cubic"

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================
DEFAULT_FEATURE_MODE = 'smart_hybrid'
VIF_THRESHOLD = 10.0
N_RECOMMENDATIONS = 2
MIN_COMPLETED_FOR_ERROR_MODEL = 10

# =============================================================================
# PLOT COLORS & STYLE
# =============================================================================
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
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
}
