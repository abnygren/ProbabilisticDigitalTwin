"""
Chemical Constants for Sulvanite (Cu₃MS₄) Nanoparticle Synthesis

This module contains physically meaningful descriptors used to encode
precursor / solvent reactivity as additional features for the GP models.
References are provided for each tabulated constant.

Supports three material systems:
  - Cu₃VS₄   (M = V,  precursor VO(acac)₂)
  - Cu₃NbS₄  (M = Nb, precursor NbCl₅)
  - Cu₃TaS₄  (M = Ta, precursor TaCl₅)

And multiple Cu precursors: CuI, CuCl, CuBr, Cu(OAc), Cu(OAc)₂, CuCl₂.

TRANSFER LEARNING
-----------------
When running a multi-precursor campaign (e.g. pooling CuI + CuCl data, or
VO(acac)₂ + NbCl₅ data), enable the relevant descriptors in
``ENHANCED_FEATURE_CONFIG`` so they become GP features.  The Cu precursor
descriptors use Pearson hardness (valid for Cu⁺ species).  The Group 5
metal precursor descriptors use **ionic potential** (Z/r from Shannon
ionic radii) because Pearson hardness breaks for d⁰ M⁵⁺ cations — see
the IONIC POTENTIAL section below.

STATUS
------
All enhanced features are **disabled by default** in ``src/config.py``
because single-precursor campaigns produce constant descriptor values.
Enable them when pooling data across precursors for transfer learning.
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
    'Cu+': 6.28,      # Pearson 1988 (Inorg. Chem. 27, 734)
    'Cu2+': 8.27,     # Pearson 1988 (Inorg. Chem. 27, 734)

    # Vanadium — V³⁺ and V⁴⁺ computed from NIST IE data (CRC Handbook);
    # V⁵⁺ is a d⁰ species where Pearson η breaks (see IONIC_POTENTIAL).
    'V3+': 8.70,      # (IE4 - IE3)/2 = (46.709 - 29.311)/2  — NIST
    'V4+': 9.29,      # (IE5 - IE4)/2 = (65.282 - 46.709)/2  — NIST
    'V5+': 31.42,     # (IE6 - IE5)/2 = (128.13 - 65.282)/2  — NIST; d⁰ core gap, DO NOT use directly

    # Niobium — computed from NIST IE data (CRC Handbook).
    # Nb⁵⁺ is d⁰; the Pearson value is formally correct but on an
    # unusable scale for GP features.  Use IONIC_POTENTIAL instead.
    'Nb3+': 6.63,     # (IE4 - IE3)/2 = (38.3 - 25.04)/2     — NIST
    'Nb4+': 6.13,     # (IE5 - IE4)/2 = (50.55 - 38.3)/2     — NIST
    'Nb5+': 25.75,    # (IE6 - IE5)/2 = (102.057 - 50.55)/2   — NIST; d⁰ core gap, DO NOT use directly

    # Tantalum — only IE1 and IE2 available in standard tables.
    # Higher IEs not in CRC Handbook; cannot compute Pearson η for Ta⁵⁺.
    'Ta5+': None,     # NOT AVAILABLE — use IONIC_POTENTIAL instead
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
    'CuI': 5.64,          # (6.28 + 5.0) / 2 — Soft-soft pairing, LOW reactivity
    'Cu(OAc)': 6.64,      # (6.28 + 7.0) / 2 — Soft-hard mismatch, HIGH reactivity
    'Cu(OAc)2': 7.64,     # Cu²⁺ with acetate
    'CuCl': 6.29,         # (6.28 + 6.3) / 2 — Soft-borderline
    'CuBr': 6.04,         # (6.28 + 5.8) / 2 — Soft-soft
    'CuCl2': 7.29,        # Cu²⁺ with chloride

    # Vanadium precursors
    'VO(acac)2': 7.25,    # V⁴⁺ with acac ligands (your VOacac)
    'VCl3': 7.15,         # V³⁺ with chloride

    # Group 5 metal chloride precursors — cation hardness unreliable for
    # d⁰ M⁵⁺; these use the same (η_cation + η_anion)/2 formula with the
    # NIST-derived values above, but prefer IONIC_POTENTIAL for GP features.
    'NbCl5': None,        # Nb⁵⁺ Pearson η = 25.75 → average = (25.75 + 6.3)/2 ≈ 16.0; not meaningful
    'TaCl5': None,        # Ta⁵⁺ Pearson η unavailable
}

# HSAB mismatch: larger values indicate a softer/harder mismatch between
# cation and anion in the precursor and (heuristically) more labile bonding.
# Calculated as |η_cation - η_anion| using the values above; entries that
# rely on the V hardness ESTIMATES inherit the same caveat.
HSAB_MISMATCH = {
    'CuI': 1.28,          # |6.28 - 5.0|  — Pearson 1988 values
    'Cu(OAc)': 0.72,      # |6.28 - 7.0|
    'Cu(OAc)2': 1.27,     # |8.27 - 7.0|
    'CuCl': 0.02,         # |6.28 - 6.3|
    'CuBr': 0.48,         # |6.28 - 5.8|
    'VO(acac)2': 2.5,     # |9.0  - 6.5|  — uses V⁴⁺ ESTIMATE; treat as approximate
    'NbCl5': None,        # d⁰ M⁵⁺ — Pearson η unreliable; use METAL_HSAB_MISMATCH
    'TaCl5': None,        # d⁰ M⁵⁺ — Pearson η unavailable; use METAL_HSAB_MISMATCH
}


# =============================================================================
# SHANNON IONIC RADII & IONIC POTENTIAL
# =============================================================================
# Shannon, R.D. (1976) Acta Cryst. A32, 751-767
#   "Effective Ionic Radii in Oxides and Fluorides"
# Coordination number is chosen to match tetrahedral sulfide (CN = 4) or
# the closest available value.  Units: Å
#
# Ionic Potential = Z / r  (charge / Shannon radius)
# A robust descriptor for comparing high-oxidation-state d⁰ cations where
# Pearson hardness is unreliable.  Higher ionic potential → harder Lewis
# acid → more reactive with soft bases (S²⁻).

SHANNON_IONIC_RADII = {
    # Cation     CN    radius (Å)   source
    'Cu+':      0.60,   # CN=4 tetrahedral  — Shannon 1976
    'Cu2+':     0.57,   # CN=4 tetrahedral  — Shannon 1976
    'V3+':      0.640,  # CN=6 octahedral   — Shannon 1976 (no CN=4 data)
    'V4+':      0.580,  # CN=6 octahedral   — Shannon 1976 (as VO²⁺)
    'V5+':      0.355,  # CN=4 tetrahedral  — Shannon 1976
    'Nb5+':     0.480,  # CN=4 tetrahedral  — Shannon 1976
    'Ta5+':     0.640,  # CN=6 octahedral   — Shannon 1976 (CN=4 not tabulated)
}

IONIC_POTENTIAL = {
    ion: _charge / SHANNON_IONIC_RADII[ion]
    for ion, _charge in [
        ('Cu+',  1), ('Cu2+', 2),
        ('V3+',  3), ('V4+',  4), ('V5+',  5),
        ('Nb5+', 5), ('Ta5+', 5),
    ]
}
# Result (Z/r, units = e/Å):
#   Cu+  ≈ 1.67,  Cu2+ ≈ 3.51
#   V3+  ≈ 4.69,  V4+  ≈ 6.90,  V5+  ≈ 14.08
#   Nb5+ ≈ 10.42, Ta5+ ≈ 7.81


# =============================================================================
# METAL PRECURSOR DESCRIPTORS (for transfer learning)
# =============================================================================
# These descriptors compare Group 5 metal precursors across campaigns.
# They use ionic potential (Z/r) rather than Pearson hardness for the
# metal cation, because η is ill-defined for d⁰ M⁵⁺.
#
# Metal_ionic_potential = Z / r of the Group 5 cation
# Metal_hsab_mismatch  = |Z/r_cation - η_anion|  (mixed-scale mismatch)
#
# The mixed-scale mismatch is intentional: the important thing is that
# the *ordering* V⁵⁺ > Nb⁵⁺ > Ta⁵⁺ is preserved on a GP-friendly scale,
# and the anion identity (acac⁻ vs Cl⁻) also contributes a shift.

METAL_PRECURSOR_IONIC_POTENTIAL = {
    'VO(acac)2': IONIC_POTENTIAL['V5+'],     # ≈ 14.08  (V⁵⁺ is formal ox. state in product)
    'NbCl5':     IONIC_POTENTIAL['Nb5+'],    # ≈ 10.42
    'TaCl5':     IONIC_POTENTIAL['Ta5+'],    # ≈  7.81
}

METAL_HSAB_MISMATCH = {
    'VO(acac)2': abs(IONIC_POTENTIAL['V5+']  - PEARSON_HARDNESS_ANIONS['acac-']),  # |14.08 - 6.5| ≈ 7.58
    'NbCl5':     abs(IONIC_POTENTIAL['Nb5+'] - PEARSON_HARDNESS_ANIONS['Cl-']),    # |10.42 - 6.3| ≈ 4.12
    'TaCl5':     abs(IONIC_POTENTIAL['Ta5+'] - PEARSON_HARDNESS_ANIONS['Cl-']),    # | 7.81 - 6.3| ≈ 1.51
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

    # Copper precursors
    'CuI': 190.45,
    'Cu(OAc)': 122.60,        # Anhydrous copper acetate
    'Cu(OAc)2': 181.63,       # Copper(II) acetate
    'CuCl': 98.99,            # Copper(I) chloride
    'CuBr': 143.45,           # Copper(I) bromide
    'CuCl2': 134.45,          # Copper(II) chloride

    # Group 5 metal precursors
    'VO(acac)2': 265.16,      # Vanadyl acetylacetonate
    'NbCl5': 270.17,          # Niobium(V) chloride
    'TaCl5': 358.21,          # Tantalum(V) chloride

    # Products
    'Cu3VS4': 356.33,         # Target phase
}

DENSITIES = {
    # g/mL at 25°C
    'ODE': 0.789,
    'OAm': 0.813,
    'DDT': 0.845,
    'CuI': 5.67,              # Solid
    'CuCl': 4.14,             # Solid
    'CuBr': 4.72,             # Solid
    'VO(acac)2': 1.50,        # Solid (approximate)
    'NbCl5': 2.75,            # Solid
    'TaCl5': 3.68,            # Solid
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_PRECURSOR_ALIASES = {
    'VOacac': 'VO(acac)2',
    'copper_iodide': 'CuI',
    'copper_acetate': 'Cu(OAc)',
    'copper_chloride': 'CuCl',
    'copper_bromide': 'CuBr',
    'niobium_chloride': 'NbCl5',
    'tantalum_chloride': 'TaCl5',
}


def _resolve_alias(name: str) -> str:
    """Map common aliases to canonical precursor names."""
    return _PRECURSOR_ALIASES.get(name, name)


def get_precursor_hardness(precursor_name: str, default: float = 6.0) -> float:
    """
    Get Pearson hardness for a Cu precursor.

    For Group 5 metal precursors (NbCl5, TaCl5) this returns None because
    Pearson η is ill-defined for d⁰ M⁵⁺ cations.  Use
    ``get_metal_ionic_potential`` instead.

    Parameters
    ----------
    precursor_name : str
        Name of precursor (e.g., 'CuI', 'CuCl', 'VO(acac)2')
    default : float
        Default value if precursor not found

    Returns
    -------
    float or None
    """
    name = _resolve_alias(precursor_name)
    val = PRECURSOR_HARDNESS.get(name, default)
    return val


def get_hsab_mismatch(precursor_name: str, default: float = 1.0) -> float:
    """
    Get HSAB mismatch value for a Cu precursor.

    For Group 5 metal precursors, returns None; use
    ``get_metal_hsab_mismatch`` instead.
    """
    name = _resolve_alias(precursor_name)
    val = HSAB_MISMATCH.get(name, default)
    return val


def get_shannon_radius(ion: str) -> float:
    """
    Get Shannon ionic radius for an ion.

    Parameters
    ----------
    ion : str
        Ion name with charge (e.g. 'V5+', 'Nb5+', 'Cu+')

    Returns
    -------
    float
        Shannon ionic radius in Å

    Raises
    ------
    KeyError
        If ion not in table
    """
    return SHANNON_IONIC_RADII[ion]


def get_ionic_potential(ion: str) -> float:
    """
    Get ionic potential (Z/r) for an ion.

    Parameters
    ----------
    ion : str
        Ion name with charge (e.g. 'V5+', 'Nb5+')

    Returns
    -------
    float
        Ionic potential in e/Å
    """
    return IONIC_POTENTIAL[ion]


def get_metal_ionic_potential(precursor_name: str, default: float = 10.0) -> float:
    """
    Get ionic potential descriptor for a Group 5 metal precursor.

    This is the primary cation descriptor for transfer learning across
    VO(acac)₂ / NbCl₅ / TaCl₅ campaigns.

    Parameters
    ----------
    precursor_name : str
        Metal precursor name (e.g. 'VO(acac)2', 'NbCl5', 'TaCl5')
    default : float
        Default value if precursor not found

    Returns
    -------
    float
        Z/r (e/Å) of the metal cation
    """
    name = _resolve_alias(precursor_name)
    return METAL_PRECURSOR_IONIC_POTENTIAL.get(name, default)


def get_metal_hsab_mismatch(precursor_name: str, default: float = 4.0) -> float:
    """
    Get the mixed-scale HSAB mismatch for a Group 5 metal precursor.

    Computed as |ionic_potential(M⁵⁺) − η(anion)|, giving a GP-friendly
    descriptor that captures both cation polarizing power and anion identity.

    Parameters
    ----------
    precursor_name : str
        Metal precursor name
    default : float
        Default value if precursor not found

    Returns
    -------
    float
    """
    name = _resolve_alias(precursor_name)
    return METAL_HSAB_MISMATCH.get(name, default)


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
    method: str = 'bruggeman'
) -> float:
    """
    Calculate effective dielectric constant of a solvent mixture.
    
    Parameters
    ----------
    volumes : dict
        Dictionary of {solvent_name: volume_in_mL}
    total_volume : float, optional
        Total volume (if None, calculated from sum)
    method : str
        'volume_weighted' - linear mixing (fast, approximate)
        'log_weighted' - logarithmic mixing
        'bruggeman' - Bruggeman effective medium (best for homogeneous mixtures)
        'maxwell_garnett' - Maxwell Garnett (requires dominant_solvent arg)
        
    Returns
    -------
    float
        Effective dielectric constant
    """
    if total_volume is None:
        total_volume = sum(volumes.values())
    
    if total_volume <= 0:
        raise ValueError("Total volume must be positive")

    vol_fractions = {s: v / total_volume for s, v in volumes.items()}

    if method == 'volume_weighted':
        return sum(vol_fractions[s] * get_solvent_dielectric(s)
                   for s in volumes.keys())

    elif method == 'log_weighted':
        import numpy as np
        ln_eps_mix = sum(vol_fractions[s] * np.log(get_solvent_dielectric(s))
                         for s in volumes.keys())
        return np.exp(ln_eps_mix)

    elif method == 'bruggeman':
        from scipy.optimize import fsolve

        components = {
            s: (get_solvent_dielectric(s), vol_fractions[s])
            for s in volumes.keys()
        }

        def bruggeman_eq(eps_eff):
            return sum(f * (eps - eps_eff) / (eps + 2 * eps_eff)
                       for eps, f in components.values())

        eps_init = sum(f * eps for eps, f in components.values())
        return fsolve(bruggeman_eq, eps_init)[0]

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# VALIDATION / SANITY CHECKS
# =============================================================================

def validate_precursor(name: str) -> bool:
    """Check if precursor name (or alias) is in our database."""
    canonical = _resolve_alias(name)
    all_known = set(PRECURSOR_HARDNESS.keys()) | set(METAL_PRECURSOR_IONIC_POTENTIAL.keys())
    return canonical in all_known


def list_available_precursors() -> dict:
    """Return all available precursors grouped by type."""
    cu_prec = {k: v for k, v in PRECURSOR_HARDNESS.items() if k.startswith('Cu')}
    metal_prec = dict(METAL_PRECURSOR_IONIC_POTENTIAL)
    return {'Cu_precursors': cu_prec, 'Metal_precursors': metal_prec}


def list_available_solvents() -> dict:
    """Return all available solvents with their dielectric constants."""
    return dict(SOLVENT_DIELECTRIC)


# =============================================================================
# MODULE INFO
# =============================================================================

__version__ = '2.0.0'
__author__ = 'Cu3VS4 BO Project'

if __name__ == '__main__':
    print("Chemical Constants Module for Cu₃MS₄ Synthesis")
    print("=" * 55)

    print("\nCu precursors (Pearson hardness):")
    for p, h in PRECURSOR_HARDNESS.items():
        if p.startswith('Cu') and h is not None:
            print(f"  {p:12s}: η = {h:.2f} eV,  HSAB mismatch = {HSAB_MISMATCH.get(p, '—')}")

    print("\nGroup 5 metal precursors (ionic potential):")
    for p, ip in METAL_PRECURSOR_IONIC_POTENTIAL.items():
        mm = METAL_HSAB_MISMATCH[p]
        print(f"  {p:12s}: Z/r = {ip:.2f} e/Å,  Metal HSAB mismatch = {mm:.2f}")

    print("\nShannon ionic radii → ionic potential:")
    for ion, r in SHANNON_IONIC_RADII.items():
        zr = IONIC_POTENTIAL[ion]
        print(f"  {ion:6s}: r = {r:.3f} Å,  Z/r = {zr:.2f}")

    print("\nSolvent dielectric constants:")
    for s in ['ODE', 'OAm', 'DDT']:
        print(f"  {s}: ε = {SOLVENT_DIELECTRIC[s]:.1f}")
