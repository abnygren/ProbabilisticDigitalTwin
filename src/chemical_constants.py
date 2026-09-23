"""Chemical descriptors for Cu3MS4 nanoparticle synthesis.

Tabulated constants used as optional GP features. Cu precursors use Pearson
hardness. Group-5 metal precursors use ionic potential (Z/r) and oxophilicity,
because Pearson hardness is unreliable for d⁰ M⁵⁺ cations.

Enhanced features are off by default. Enable them in config.py when pooling
data across precursors.
"""

# Pearson absolute hardness (η)
# Hard-Soft Acid-Base (HSAB) theory: η = (IP - EA) / 2
# where IP = ionization potential, EA = electron affinity
# Units: eV
#
# Reference: Pearson, R.G. (1988) Inorg. Chem. 27, 734-740
#            "Absolute Electronegativity and Hardness: Application to Inorganic Chemistry"
# Reference: Parr, R.G. & Pearson, R.G. (1983) J. Am. Chem. Soc. 105, 7512-7516

PEARSON_HARDNESS_CATIONS = {
    'Cu+': 6.28,      # Pearson 1988 (Inorg. Chem. 27, 734)
    'Cu2+': 8.27,     # Pearson 1988 (Inorg. Chem. 27, 734)
}

PEARSON_HARDNESS_ANIONS = {
    'I-': 3.70,       # Soft base
    'Br-': 4.24,      # Soft base
    'Cl-': 4.70,      # Borderline base
    'OAc-': 7.0,      # Acetate (carboxylate O donor)
    'acac-': 6.5,     # Acetylacetonate (chelating O donor)
}

# Precursor hardness: (η_cation + η_anion) / 2
PRECURSOR_HARDNESS = {
    'CuI': 4.99,          # (6.28 + 3.70) / 2
    'CuBr': 5.26,         # (6.28 + 4.24) / 2
    'CuCl': 5.49,         # (6.28 + 4.70) / 2
    'Cu(OAc)': 6.64,      # (6.28 + 7.0) / 2
    'Cu(OAc)2': 7.64,     # (8.27 + 7.0) / 2
    'VO(acac)2': 7.90,    # (9.29 + 6.5) / 2; V⁴⁺ NIST + acac⁻
}

# HSAB mismatch |η_cation − η_anion|
HSAB_MISMATCH = {
    'CuI': 2.58,          # |6.28 - 3.70|
    'CuBr': 2.04,         # |6.28 - 4.24|
    'CuCl': 1.58,         # |6.28 - 4.70|
    'Cu(OAc)': 0.72,      # |6.28 - 7.0|
    'Cu(OAc)2': 1.27,     # |8.27 - 7.0|
    'VO(acac)2': 2.79,    # |9.29 - 6.5|
}


# Shannon ionic radii and ionic potential (Z/r).
# Shannon, R.D. (1976) Acta Cryst. A32, 751-767
# Coordination: tetrahedral sulfide (CN = 4) or the closest tabulated value.
# Units: Å. Ionic potential is used for d⁰ M⁵⁺ cations, where Pearson η is unreliable.

SHANNON_IONIC_RADII = {
    # Cation     CN    radius (Å)   source
    'Cu+':      0.60,   # CN=4 tetrahedral  — Shannon 1976
    'Cu2+':     0.57,   # CN=4 tetrahedral  — Shannon 1976
    'V4+':      0.580,  # CN=6 octahedral   — Shannon 1976 (as VO²⁺)
    'V5+':      0.355,  # CN=4 tetrahedral  — Shannon 1976
    'Ta5+':     0.640,  # CN=6 octahedral   — Shannon 1976 (CN=4 not tabulated)
}

IONIC_POTENTIAL = {
    ion: _charge / SHANNON_IONIC_RADII[ion]
    for ion, _charge in [
        ('Cu+',  1), ('Cu2+', 2),
        ('V4+',  4), ('V5+',  5),
        ('Ta5+', 5),
    ]
}
# Result (Z/r, units = e/Å):
#   Cu+  ≈ 1.67,  Cu2+ ≈ 3.51
#   V4+  ≈ 6.90,  V5+  ≈ 14.08
#   Ta5+ ≈ 7.81


# Group-5 metal precursor descriptors for transfer learning.
# Ionic potential uses the oxidation state charged into the flask, not the
# product ion: VO(acac)2 is V(IV), TaCl5 is Ta(V).
# These precursor-basis values differ from the product-basis IONIC_POTENTIAL table.
METAL_PRECURSOR_IONIC_POTENTIAL = {
    'VO(acac)2': 7.5472,   # V(IV), CN=5, r=0.53 Å → 4/0.53
    'TaCl5':     7.8125,   # Ta(V), CN=6, r=0.64 Å → 5/0.64
}


# Oxophilicity: per-atom MO2 − MS2 formation-energy difference (eV/atom).
# More negative = more oxophilic. Trend V < Ta. The GP uses the z-scored
# value, so only the ordering matters. With two metals this axis is collinear
# with ionic potential.
METAL_OXOPHILICITY = {
    'VO(acac)2': -1.313,   # eV/atom; VO2 mp-19094 vs VS2 mp-557523
    'TaCl5':     -1.653,   # eV/atom; TaO2 mp-20994 vs TaS2 mp-1984
}

METAL_OXOPHILICITY_MP_IDS = {
    'VO2': 'mp-19094',
    'VS2': 'mp-557523',
    'TaO2': 'mp-20994',
    'TaS2': 'mp-1984',
}

METAL_OXOPHILICITY_SOURCE = (
    "Per-atom MO2 vs MS2 formation-energy difference from the Materials Project "
    "(Jain et al., APL Mater. 2013, 1, 011002): "
    "VO2 mp-19094, VS2 mp-557523, TaO2 mp-20994, TaS2 mp-1984."
)


# Bond dissociation energies (kJ/mol). DDT is the sulfur source in all campaigns.
# Luo, Y.-R. (2007) Comprehensive Handbook of Chemical Bond Energies.
BOND_DISSOCIATION_ENERGIES = {
    'DDT': 365,           # primary thiol C–S
}

# Solvent dielectric constants. ODE / OAm / DDT are the three liquids in the flask.
# Wohlfarth, C. (2008) Landolt-Börnstein IV/17.
SOLVENT_DIELECTRIC = {
    'ODE': 2.1,               # 1-octadecene
    'OAm': 3.4,               # oleylamine
    'DDT': 2.5,               # dodecanethiol (approximate)
}


# Molecular weights and densities
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

    # Group 5 metal precursors
    'VO(acac)2': 265.16,      # Vanadyl acetylacetonate
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
    'TaCl5': 3.68,            # Solid
}


# Helper functions

_PRECURSOR_ALIASES = {
    'VOacac': 'VO(acac)2',
    'copper_iodide': 'CuI',
    'copper_acetate': 'Cu(OAc)',
    'copper_chloride': 'CuCl',
    'copper_bromide': 'CuBr',
    'tantalum_chloride': 'TaCl5',
}


def _resolve_alias(name: str) -> str:
    """Map common aliases to canonical precursor names."""
    return _PRECURSOR_ALIASES.get(name, name)


def get_precursor_hardness(precursor_name: str, default: float = 6.0) -> float:
    """
    Get Pearson hardness for a precursor.

    TaCl5 is not in this table (Pearson η is ill-defined for d⁰ Ta⁵⁺);
    use ``get_metal_ionic_potential`` instead.

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
    if name in METAL_PRECURSOR_IONIC_POTENTIAL and name not in PRECURSOR_HARDNESS:
        return None
    return PRECURSOR_HARDNESS.get(name, default)


def get_hsab_mismatch(precursor_name: str, default: float = 1.0) -> float:
    """Get HSAB mismatch for a precursor."""
    name = _resolve_alias(precursor_name)
    if name in METAL_PRECURSOR_IONIC_POTENTIAL and name not in HSAB_MISMATCH:
        return None
    return HSAB_MISMATCH.get(name, default)


def get_shannon_radius(ion: str) -> float:
    """
    Get Shannon ionic radius for an ion.

    Parameters
    ----------
    ion : str
        Ion name with charge (e.g. 'V5+', 'Ta5+', 'Cu+')

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
        Ion name with charge (e.g. 'V5+', 'Ta5+')

    Returns
    -------
    float
        Ionic potential in e/Å
    """
    return IONIC_POTENTIAL[ion]


def get_metal_ionic_potential(precursor_name: str, default: float = 10.0) -> float:
    """
    Get ionic potential (Z/r) for a Group 5 metal precursor.

    Used with oxophilicity for V/Ta transfer. Oxophilicity is the primary
    metal-axis descriptor in ``config.TRANSFER_DESCRIPTOR_AXES``.

    Parameters
    ----------
    precursor_name : str
        Metal precursor name (e.g. 'VO(acac)2', 'TaCl5')
    default : float
        Default value if precursor not found

    Returns
    -------
    float
        Z/r (e/Å) of the metal cation
    """
    name = _resolve_alias(precursor_name)
    return METAL_PRECURSOR_IONIC_POTENTIAL.get(name, default)


def get_metal_oxophilicity(precursor_name: str, default: float = -1.5) -> float:
    """
    Get the oxophilicity descriptor for a Group 5 metal precursor.

    Oxophilicity is the per-atom (MO₂ − MS₂) energy (eV/atom); more negative
    means more oxophilic.  Used alongside ``get_metal_ionic_potential`` as the
    two metal-axis transfer-learning descriptors.

    Parameters
    ----------
    precursor_name : str
        Metal precursor name (e.g. 'VO(acac)2', 'TaCl5')
    default : float
        Default value if precursor not found

    Returns
    -------
    float
        Oxophilicity in eV/atom
    """
    name = _resolve_alias(precursor_name)
    return METAL_OXOPHILICITY.get(name, default)


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
        'bruggeman' - Bruggeman effective medium (default)
        
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


# Validation / sanity checks

def validate_precursor(name: str) -> bool:
    """Return True if the precursor name (or alias) is in the lookup tables."""
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


# Figure palette for method / skill colors (not precursor identity).
# Orange/brown is reserved for copper halides; purple for V/Ta.

DESCRIPTOR_SKILL_GAIN = '#238B45'
DESCRIPTOR_SKILL_GAIN_LT = '#74C476'   # lighter bars (Pearson r)

# LOPO: dark = centered R²; light = Pearson r
LOPO_SYNTHESIS = '#311C2F'
LOPO_SYNTHESIS_LT = '#A957A1'
LOPO_ONEHOT = '#6B354C'
LOPO_ONEHOT_LT = '#C06F91'
LOPO_HSAB = '#933A3D'
LOPO_HSAB_LT = '#D58789'
LOPO_LEGEND_DARK = '#4A4A48'
LOPO_LEGEND_LIGHT = '#B0B0AD'

if __name__ == '__main__':
    print("Chemical constants for Cu3MS4 synthesis")

    print("\nCu precursors (Pearson hardness):")
    for p, h in PRECURSOR_HARDNESS.items():
        if p.startswith('Cu') and h is not None:
            print(f"  {p:12s}: η = {h:.2f} eV,  HSAB mismatch = {HSAB_MISMATCH.get(p, '—')}")

    print("\nGroup 5 metal precursors (transfer descriptors):")
    for p, ip in METAL_PRECURSOR_IONIC_POTENTIAL.items():
        ox = METAL_OXOPHILICITY[p]
        print(f"  {p:12s}: Z/r = {ip:.4f} e/Å,  oxophilicity = {ox:.3f} eV/atom")
    print(f"  Oxophilicity source: {METAL_OXOPHILICITY_SOURCE}")

    print("\nShannon ionic radii → ionic potential:")
    for ion, r in SHANNON_IONIC_RADII.items():
        zr = IONIC_POTENTIAL[ion]
        print(f"  {ion:6s}: r = {r:.3f} Å,  Z/r = {zr:.2f}")

    print("\nSolvent dielectric constants:")
    for s in ['ODE', 'OAm', 'DDT']:
        print(f"  {s}: ε = {SOLVENT_DIELECTRIC[s]:.1f}")
