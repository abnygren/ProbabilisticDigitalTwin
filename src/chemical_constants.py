"""
Chemical Constants for Cu₃VS₄ Nanoparticle Synthesis

This module contains physically meaningful descriptors for encoding
chemical reactivity in Bayesian optimization of nanocrystal synthesis.

References are provided for each constant where applicable.
"""

# =============================================================================
# PEARSON ABSOLUTE HARDNESS (η)
# =============================================================================
# Hard-Soft Acid-Base (HSAB) theory: η = (IP - EA) / 2
# where IP = ionization potential, EA = electron affinity
# Units: eV
#
# Reference: Pearson, R.G. (1988) Inorg. Chem. 27, 734-740
#            "Absolute Electronegativity and Hardness: Application to Inorganic Chemistry"
# Reference: Parr, R.G. & Pearson, R.G. (1983) J. Am. Chem. Soc. 105, 7512-7516

PEARSON_HARDNESS_CATIONS = {
    # Copper species (Cu⁺ is a soft acid)
    'Cu+': 6.28,      # Soft acid
    'Cu2+': 8.27,     # Borderline acid
    
    # Vanadium species
    'V3+': 8.0,       # Hard acid (estimated)
    'V4+': 9.0,       # Hard acid (estimated from VO²⁺)
    'V5+': 10.0,      # Hard acid (estimated)
}

PEARSON_HARDNESS_ANIONS = {
    # Halides
    'I-': 5.0,        # Soft base
    'Br-': 5.8,       # Soft base
    'Cl-': 6.3,       # Borderline base
    'F-': 7.0,        # Hard base
    
    # Oxygen donors
    'OAc-': 7.0,      # Acetate - Hard base (carboxylate O donor)
    'acac-': 6.5,     # Acetylacetonate - Borderline (chelating O donor)
    'OH-': 7.5,       # Hydroxide - Hard base
    
    # Sulfur donors  
    'S2-': 4.1,       # Sulfide - Soft base
    'RS-': 4.5,       # Thiolate - Soft base (approximate)
}

# Combined precursor hardness values (cation + anion average or effective)
# This represents the overall "hardness character" of the precursor
PRECURSOR_HARDNESS = {
    # Copper precursors
    'CuI': 5.64,          # (6.28 + 5.0) / 2 - Soft-soft pairing, LOW reactivity
    'Cu(OAc)': 6.64,      # (6.28 + 7.0) / 2 - Soft-hard mismatch, HIGH reactivity
    'Cu(OAc)2': 7.64,     # Cu²⁺ with acetate
    'CuCl': 6.29,         # Soft-borderline
    'CuBr': 6.04,         # Soft-soft
    'CuCl2': 7.29,        # Cu²⁺ with chloride
    
    # Vanadium precursors
    'VO(acac)2': 7.25,    # V⁴⁺ with acac ligands (your VOacac)
    'VCl3': 7.15,         # V³⁺ with chloride
}

# HSAB mismatch: larger values indicate more labile/reactive precursors
# Calculated as |η_cation - η_anion|
HSAB_MISMATCH = {
    'CuI': 1.28,          # |6.28 - 5.0| - Small mismatch, stable
    'Cu(OAc)': 0.72,      # |6.28 - 7.0| - Some mismatch
    'Cu(OAc)2': 1.27,     # |8.27 - 7.0|
    'CuCl': 0.02,         # |6.28 - 6.3| - Very small
    'CuBr': 0.48,         # |6.28 - 5.8|
    'VO(acac)2': 2.5,     # |9.0 - 6.5| - Significant mismatch
}


# =============================================================================
# BOND DISSOCIATION ENERGIES (BDE)
# =============================================================================
# For sulfur precursors - affects sulfur release kinetics
# Units: kJ/mol
#
# Reference: Luo, Y.-R. (2007) "Comprehensive Handbook of Chemical Bond Energies"
# Reference: NIST Chemistry WebBook (webbook.nist.gov)

BOND_DISSOCIATION_ENERGIES = {
    # Thiols (R-SH bond)
    'DDT': 365,           # Dodecanethiol C-S bond (~365 kJ/mol for primary thiols)
    'octanethiol': 365,   # Similar primary thiol
    'tert-butylthiol': 355,  # Tertiary, slightly weaker
    
    # Thioethers / Sulfides
    'dioctyl_sulfide': 307,   # C-S in dialkyl sulfide
    'thioanisole': 290,       # Ar-S bond
    
    # Inorganic sulfur
    'elemental_S8': 226,      # S-S bond in S₈ ring
    'CS2': 272,               # C=S double bond
    'H2S': 381,               # H-S bond
    
    # Dithiocarbamates (common in NC synthesis)
    'diethyl_dithiocarbamate': 250,  # Estimated, chelating
}

# Normalized BDE (scaled 0-1 for ML, higher = harder to release S)
BDE_NORMALIZED = {k: (v - 220) / (390 - 220) for k, v in BOND_DISSOCIATION_ENERGIES.items()}


# =============================================================================
# SOLVENT PROPERTIES
# =============================================================================
# Dielectric constants (ε) and polarity indices
# Higher ε → better ion dissociation, faster cation exchange
#
# Reference: Wohlfarth, C. (2008) "Static Dielectric Constants of Pure Liquids and Binary 
#            Liquid Mixtures" Landolt-Börnstein IV/17
# Reference: Snyder, L.R. (1974) J. Chromatogr. 92, 223 (polarity index)

SOLVENT_DIELECTRIC = {
    # Non-polar high-boiling solvents
    'ODE': 2.1,               # 1-octadecene (very non-polar)
    'squalane': 2.0,          # Hydrocarbon
    'hexadecane': 2.05,       # n-alkane
    
    # Coordinating solvents (amines)
    'OAm': 3.4,               # Oleylamine (weakly polar, basic)
    'octylamine': 3.2,        # Primary amine
    'trioctylamine': 2.8,     # Tertiary amine (less polar)
    'TOPO': 2.6,              # Trioctylphosphine oxide
    
    # Thiols (act as both solvent and S source)
    'DDT': 2.5,               # Dodecanethiol (approximate)
    'octanethiol': 2.6,       # Similar
    
    # Polar aprotic (for comparison)
    'DMF': 36.7,              # Dimethylformamide
    'DMSO': 46.7,             # Dimethyl sulfoxide
}

SOLVENT_POLARITY_INDEX = {
    # Snyder polarity index P' (0 = non-polar, 10 = very polar)
    'ODE': 0.0,
    'squalane': 0.0,
    'OAm': 1.2,               # Estimated (basic amine character)
    'DDT': 0.1,               # Very weakly polar
    'trioctylamine': 0.8,
    'DMF': 6.4,
    'DMSO': 7.2,
}

# Boiling points (°C) - relevant for reaction temperature limits
SOLVENT_BOILING_POINT = {
    'ODE': 315,
    'OAm': 364,
    'DDT': 266,               # Note: lower than typical reaction temps
    'squalane': 350,
    'trioctylamine': 365,
}


# =============================================================================
# MOLECULAR WEIGHTS AND DENSITIES
# =============================================================================
# For converting between volume and molar quantities

MOLECULAR_WEIGHTS = {
    # Solvents
    'ODE': 252.48,            # 1-octadecene
    'OAm': 267.49,            # Oleylamine
    'DDT': 202.40,            # Dodecanethiol
    
    # Precursors
    'CuI': 190.45,
    'Cu(OAc)': 122.60,        # Anhydrous copper acetate
    'Cu(OAc)2': 181.63,       # Copper(II) acetate
    'VO(acac)2': 265.16,      # Vanadyl acetylacetonate
    
    # Products
    'Cu3VS4': 356.33,         # Target phase
}

DENSITIES = {
    # g/mL at 25°C
    'ODE': 0.789,
    'OAm': 0.813,
    'DDT': 0.845,
    'CuI': 5.67,              # Solid
    'VO(acac)2': 1.50,        # Solid (approximate)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_precursor_hardness(precursor_name: str, default: float = 6.0) -> float:
    """
    Get Pearson hardness for a precursor.
    
    Parameters
    ----------
    precursor_name : str
        Name of precursor (e.g., 'CuI', 'Cu(OAc)', 'VO(acac)2')
    default : float
        Default value if precursor not found
        
    Returns
    -------
    float
        Pearson absolute hardness (eV)
    """
    # Handle common aliases
    aliases = {
        'VOacac': 'VO(acac)2',
        'copper_iodide': 'CuI',
        'copper_acetate': 'Cu(OAc)',
    }
    name = aliases.get(precursor_name, precursor_name)
    return PRECURSOR_HARDNESS.get(name, default)


def get_hsab_mismatch(precursor_name: str, default: float = 1.0) -> float:
    """
    Get HSAB mismatch value for a precursor.
    Larger values indicate more reactive/labile precursors.
    """
    aliases = {
        'VOacac': 'VO(acac)2',
        'copper_iodide': 'CuI',
        'copper_acetate': 'Cu(OAc)',
    }
    name = aliases.get(precursor_name, precursor_name)
    return HSAB_MISMATCH.get(name, default)


def get_sulfur_bde(precursor_name: str, default: float = 365.0) -> float:
    """
    Get bond dissociation energy for sulfur precursor.
    
    Parameters
    ----------
    precursor_name : str
        Name of S precursor (e.g., 'DDT', 'elemental_S8')
    default : float
        Default BDE in kJ/mol
        
    Returns
    -------
    float
        BDE in kJ/mol
    """
    return BOND_DISSOCIATION_ENERGIES.get(precursor_name, default)


def get_solvent_dielectric(solvent_name: str, default: float = 2.5) -> float:
    """Get dielectric constant for a solvent."""
    return SOLVENT_DIELECTRIC.get(solvent_name, default)


def calculate_mixture_dielectric(
    volumes: dict,
    total_volume: float = None,
    method: str = 'volume_weighted'
) -> float:
    """
    Calculate effective dielectric constant of a solvent mixture.
    
    Parameters
    ----------
    volumes : dict
        Dictionary of {solvent_name: volume_in_mL}
        e.g., {'DDT': 3.0, 'OAm': 4.0, 'ODE': 5.0}
    total_volume : float, optional
        Total volume (if None, calculated from sum of volumes)
    method : str
        'volume_weighted' - simple linear mixing (default)
        'log_weighted' - logarithmic mixing rule (better for large ε differences)
        
    Returns
    -------
    float
        Effective dielectric constant of mixture
        
    Notes
    -----
    The volume-weighted average is a first approximation. For more accurate
    mixing, consider Onsager or Clausius-Mossotti equations, but for the
    relatively similar ε values in NC synthesis solvents, linear is adequate.
    """
    if total_volume is None:
        total_volume = sum(volumes.values())
    
    if total_volume <= 0:
        raise ValueError("Total volume must be positive")
    
    if method == 'volume_weighted':
        eps_mix = 0.0
        for solvent, vol in volumes.items():
            eps = get_solvent_dielectric(solvent)
            eps_mix += (vol / total_volume) * eps
        return eps_mix
    
    elif method == 'log_weighted':
        # Logarithmic mixing: ln(ε_mix) = Σ φᵢ ln(εᵢ)
        import numpy as np
        ln_eps_mix = 0.0
        for solvent, vol in volumes.items():
            eps = get_solvent_dielectric(solvent)
            ln_eps_mix += (vol / total_volume) * np.log(eps)
        return np.exp(ln_eps_mix)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# VALIDATION / SANITY CHECKS
# =============================================================================

def validate_precursor(name: str) -> bool:
    """Check if precursor is in our database."""
    all_precursors = set(PRECURSOR_HARDNESS.keys()) | {'VOacac', 'copper_iodide', 'copper_acetate'}
    return name in all_precursors


def list_available_precursors() -> dict:
    """Return all available precursors with their hardness values."""
    return dict(PRECURSOR_HARDNESS)


def list_available_solvents() -> dict:
    """Return all available solvents with their dielectric constants."""
    return dict(SOLVENT_DIELECTRIC)


# =============================================================================
# MODULE INFO
# =============================================================================

__version__ = '1.0.0'
__author__ = 'Cu3VS4 BO Project'

if __name__ == '__main__':
    # Quick test / display
    print("Chemical Constants Module for Cu₃VS₄ Synthesis")
    print("=" * 50)
    print("\nAvailable Cu precursors:")
    for p, h in PRECURSOR_HARDNESS.items():
        if 'Cu' in p:
            print(f"  {p}: η = {h:.2f} eV")
    
    print("\nSolvent dielectric constants:")
    for s, e in SOLVENT_DIELECTRIC.items():
        if s in ['ODE', 'OAm', 'DDT']:
            print(f"  {s}: ε = {e:.1f}")
    
    print("\nExample mixture calculation:")
    vols = {'DDT': 3.0, 'OAm': 4.0, 'ODE': 5.0}
    eps = calculate_mixture_dielectric(vols)
    print(f"  {vols} → ε_mix = {eps:.2f}")
