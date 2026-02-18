"""
Feature engineering for Cu₃VS₄ Bayesian Optimization

Transforms raw synthesis parameters (Temp, Time, VOacac, DDT, OAm) into
chemically meaningful features (ratios, concentrations, dielectric constants).
Also handles smart hybrid feature selection via VIF elimination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import (
    CUI_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, RAW_BOUNDS, RAW_ROUNDING,
    CHEM_FEATURES_BASIC, ENHANCED_FEATURE_CONFIG, CURRENT_PRECURSORS,
)

# Try to import chemical constants for enhanced features
try:
    import chemical_constants as chem_const
    CHEM_CONSTANTS_AVAILABLE = True
except ImportError:
    CHEM_CONSTANTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Build derived feature lists from config
# ---------------------------------------------------------------------------
CHEM_FEATURES_ENHANCED: List[str] = []
if CHEM_CONSTANTS_AVAILABLE:
    for feat, enabled in ENHANCED_FEATURE_CONFIG.items():
        if enabled:
            CHEM_FEATURES_ENHANCED.append(feat)

CHEM_FEATURES = CHEM_FEATURES_BASIC + CHEM_FEATURES_ENHANCED

HYBRID_FEATURES = RAW_FACTORS + ["Cu_V_ratio", "Metal_Conc"]
if 'effective_dielectric' in CHEM_FEATURES_ENHANCED:
    HYBRID_FEATURES.append('effective_dielectric')

# Smart hybrid is computed dynamically (see select_smart_hybrid_features)
SMART_HYBRID_FEATURES = None


# =============================================================================
# ROUNDING
# =============================================================================

def round_to_practical(params: Dict[str, float]) -> Dict[str, float]:
    """Round raw synthesis parameters to lab-practical precision."""
    return {
        k: round(float(v), RAW_ROUNDING.get(k, 3))
        for k, v in params.items()
    }


# =============================================================================
# RAW <-> CHEMICAL TRANSFORMS
# =============================================================================

def raw_to_chemical_features(
    Temp: np.ndarray,
    Time: np.ndarray,
    VOacac: np.ndarray,
    DDT: np.ndarray,
    OAm: np.ndarray,
    CuI: float = CUI_MMOL,
    total_vol: float = TOTAL_VOLUME_ML,
    ddt_conv: float = DDT_MMOL_PER_ML,
    oam_conv: float = OAM_MMOL_PER_ML,
    include_enhanced: bool = True,
    cu_precursor: str = None,
    s_precursor: str = None,
) -> Dict[str, np.ndarray]:
    """Transform raw synthesis parameters into chemically meaningful features."""
    Temp = np.atleast_1d(Temp).astype(float)
    Time = np.atleast_1d(Time).astype(float)
    VOacac = np.atleast_1d(VOacac).astype(float)
    DDT = np.atleast_1d(DDT).astype(float)
    OAm = np.atleast_1d(OAm).astype(float)

    n_samples = len(Temp)
    total_metal = CuI + VOacac
    DDT_mmol = DDT * ddt_conv
    OAm_mmol = OAm * oam_conv

    result = {
        'Temp': Temp,
        'Cu_V_ratio': CuI / VOacac,
        'S_Metal_ratio': DDT_mmol / total_metal,
        'Ligand_Metal_ratio': OAm_mmol / total_metal,
        'Metal_Conc': (total_metal / total_vol) * 1000,
        'log_Time': np.log10(np.maximum(Time, 1.0)),
    }

    if include_enhanced and CHEM_CONSTANTS_AVAILABLE:
        cu_prec = cu_precursor or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI')
        s_prec = s_precursor or CURRENT_PRECURSORS.get('S_precursor', 'DDT')

        if ENHANCED_FEATURE_CONFIG.get('effective_dielectric', False):
            ODE_vol = total_vol - DDT - OAm
            eps_mix = np.zeros(n_samples)
            for i in range(n_samples):
                volumes = {'DDT': DDT[i], 'OAm': OAm[i], 'ODE': ODE_vol[i]}
                eps_mix[i] = chem_const.calculate_mixture_dielectric(volumes, total_vol)
            result['effective_dielectric'] = eps_mix

        if ENHANCED_FEATURE_CONFIG.get('Cu_precursor_hardness', False):
            hardness = chem_const.get_precursor_hardness(cu_prec)
            result['Cu_precursor_hardness'] = np.full(n_samples, hardness)

        if ENHANCED_FEATURE_CONFIG.get('hsab_mismatch', False):
            mismatch = chem_const.get_hsab_mismatch(cu_prec)
            result['hsab_mismatch'] = np.full(n_samples, mismatch)

        if ENHANCED_FEATURE_CONFIG.get('S_BDE', False):
            bde = chem_const.get_sulfur_bde(s_prec)
            result['S_BDE'] = np.full(n_samples, bde)

    return result


def chemical_to_raw_features(
    Temp: np.ndarray,
    Cu_V_ratio: np.ndarray,
    S_Metal_ratio: np.ndarray,
    Ligand_Metal_ratio: np.ndarray,
    Metal_Conc: np.ndarray,
    log_Time: np.ndarray,
    CuI: float = CUI_MMOL,
    total_vol: float = TOTAL_VOLUME_ML,
    ddt_conv: float = DDT_MMOL_PER_ML,
    oam_conv: float = OAM_MMOL_PER_ML,
) -> Dict[str, np.ndarray]:
    """Convert chemical features back to raw lab parameters."""
    Temp = np.atleast_1d(Temp)
    Cu_V_ratio = np.atleast_1d(Cu_V_ratio)
    S_Metal_ratio = np.atleast_1d(S_Metal_ratio)
    Ligand_Metal_ratio = np.atleast_1d(Ligand_Metal_ratio)
    Metal_Conc = np.atleast_1d(Metal_Conc)
    log_Time = np.atleast_1d(log_Time)

    Time = 10 ** log_Time
    VOacac = CuI / Cu_V_ratio
    total_metal = CuI + VOacac
    DDT_mmol = S_Metal_ratio * total_metal
    OAm_mmol = Ligand_Metal_ratio * total_metal
    DDT = DDT_mmol / ddt_conv
    OAm = OAm_mmol / oam_conv

    return {
        'Temp': Temp,
        'Time': Time,
        'VOacac': VOacac,
        'DDT': DDT,
        'OAm': OAm,
    }


def add_chemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all chemical features to a dataframe."""
    df = df.copy()
    chem = raw_to_chemical_features(
        Temp=df['Temp'].values,
        Time=df['Time'].values,
        VOacac=df['VOacac'].values,
        DDT=df['DDT'].values,
        OAm=df['OAm'].values,
    )
    for name, values in chem.items():
        if name != 'Temp':
            df[name] = values
    return df


# =============================================================================
# SMART HYBRID FEATURE SELECTION (VIF-based)
# =============================================================================

def calculate_vif(X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity detection.

    VIF interpretation:
    - VIF = 1: No correlation with other features
    - VIF < 5: Low collinearity (acceptable)
    - VIF 5-10: Moderate collinearity (caution)
    - VIF > 10: High collinearity (problematic)
    """
    X = np.atleast_2d(X)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    vif_values = []
    for i in range(n_features):
        y = X[:, i]
        other_idx = [j for j in range(n_features) if j != i]
        X_other = X[:, other_idx]
        X_other_const = np.column_stack([np.ones(len(X)), X_other])

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_other_const, y, rcond=None)
            y_pred = X_other_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                r_squared = max(0, min(r_squared, 0.9999))
            else:
                r_squared = 0

            vif = 1 / (1 - r_squared)
        except (np.linalg.LinAlgError, ValueError):
            vif = np.inf

        vif_values.append(vif)

    def interpret_vif(v):
        if v < 5:
            return "✓ Low (acceptable)"
        elif v < 10:
            return "⚠ Moderate (caution)"
        else:
            return "✗ High (problematic)"

    results = pd.DataFrame({
        'Feature': feature_names,
        'VIF': vif_values,
    })
    results['Interpretation'] = results['VIF'].apply(interpret_vif)
    return results.sort_values('VIF', ascending=False)


def select_smart_hybrid_features(
    df: pd.DataFrame,
    vif_threshold: float = 10.0,
    verbose: bool = True
) -> List[str]:
    """
    Select hybrid features using iterative VIF elimination.

    Strategy:
    1. Start with all raw factors + all chemical features
    2. Compute VIF for every feature
    3. If highest VIF > threshold, drop that feature — preferring raw features
       when their chemical substitute is present
    4. Repeat until all VIF <= threshold
    """
    if 'Cu_V_ratio' not in df.columns:
        df = add_chemical_features(df)

    reconstruction_map = {
        'VOacac': 'Cu_V_ratio',
        'DDT': 'S_Metal_ratio',
        'OAm': 'Ligand_Metal_ratio',
        'Time': 'log_Time',
    }
    reverse_map = {v: k for k, v in reconstruction_map.items()}

    chemical_candidates = [
        "Cu_V_ratio", "S_Metal_ratio", "Ligand_Metal_ratio",
        "Metal_Conc", "log_Time",
    ]
    if 'effective_dielectric' in df.columns:
        chemical_candidates.append('effective_dielectric')

    candidates = RAW_FACTORS.copy() + chemical_candidates
    candidates = [f for f in candidates if f in df.columns]
    seen = set()
    candidates = [f for f in candidates if not (f in seen or seen.add(f))]

    dropped = []

    while True:
        X = df[candidates].values.astype(float)
        vif_df = calculate_vif(X, candidates)
        worst = vif_df.iloc[0]

        if worst['VIF'] <= vif_threshold:
            break

        droppable = []
        for _, row in vif_df.iterrows():
            feat = row['Feature']
            vif_val = row['VIF']
            if vif_val <= vif_threshold:
                break
            if feat in reconstruction_map and reconstruction_map[feat] in candidates:
                droppable.append((feat, vif_val, 'raw_with_substitute'))
            elif feat in reverse_map and reverse_map[feat] in candidates:
                droppable.append((feat, vif_val, 'chem_with_raw_present'))
            elif feat in RAW_FACTORS and feat not in reconstruction_map:
                droppable.append((feat, vif_val, 'raw_no_substitute'))
            elif feat not in RAW_FACTORS and feat not in reverse_map:
                droppable.append((feat, vif_val, 'chem_extra'))

        if not droppable:
            if verbose:
                print(f"  Cannot drop further without losing reconstruction ability "
                      f"(worst VIF: {worst['VIF']:.1f} on {worst['Feature']})")
            break

        priority = {'raw_with_substitute': 0, 'chem_extra': 1,
                     'chem_with_raw_present': 2, 'raw_no_substitute': 3}
        droppable.sort(key=lambda x: (priority[x[2]], -x[1]))
        to_drop, drop_vif, reason = droppable[0]

        candidates.remove(to_drop)
        dropped.append((to_drop, drop_vif, reason))
        if verbose:
            print(f"  ✗ Drop {to_drop:20s} (VIF={drop_vif:>7.1f}, {reason})")

        if to_drop in reconstruction_map:
            sub = reconstruction_map[to_drop]
            if sub not in candidates and sub in df.columns:
                candidates.append(sub)

    has_ratio = any(f in candidates for f in ['Cu_V_ratio', 'S_Metal_ratio', 'Ligand_Metal_ratio'])
    if has_ratio and 'Metal_Conc' not in candidates and 'Metal_Conc' in df.columns:
        candidates.append('Metal_Conc')

    if verbose:
        print(f"\n✓ Smart Hybrid Mode: {len(candidates)} features selected (VIF threshold={vif_threshold})")
        print(f"  Raw features kept: {[f for f in candidates if f in RAW_FACTORS]}")
        print(f"  Chemical features: {[f for f in candidates if f not in RAW_FACTORS]}")
        if dropped:
            print(f"  Dropped {len(dropped)} collinear features:")
            for feat, vif_val, reason in dropped:
                print(f"    {feat} (VIF={vif_val:.1f})")

    return candidates


# =============================================================================
# BOUNDS & FEATURE VECTOR HELPERS
# =============================================================================

def compute_feature_bounds(feature_list: List[str], df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute bounds for any feature list from observed data."""
    bounds = {}
    for feat in feature_list:
        if feat in RAW_BOUNDS:
            bounds[feat] = RAW_BOUNDS[feat]
        elif feat in df.columns:
            vals = df[feat].dropna()
            if len(vals) == 0:
                raise ValueError(f"No valid values found for feature '{feat}'")
            elif len(vals) == 1:
                val = vals.iloc[0]
                margin = abs(val) * 0.1 if abs(val) > 1e-6 else 0.1
                bounds[feat] = (val - margin, val + margin)
            else:
                margin = 0.05 * (vals.max() - vals.min())
                bounds[feat] = (vals.min() - margin, vals.max() + margin)
        else:
            raise ValueError(f"Feature '{feat}' not found in dataframe and not in RAW_BOUNDS")
    return bounds


def build_feature_vector_from_raw(
    conditions: Dict[str, float],
    feature_mode: str,
    feature_names: List[str]
) -> List[float]:
    """Build a feature vector in the base optimizer's feature space from raw conditions."""
    raw = {
        'Temp': float(conditions.get('Temp', 0)),
        'Time': float(conditions.get('Time', 0)),
        'VOacac': float(conditions.get('VOacac', 0)),
        'DDT': float(conditions.get('DDT', 0)),
        'OAm': float(conditions.get('OAm', 0)),
    }

    if feature_mode == 'raw':
        feat_dict = raw
    else:
        chem = raw_to_chemical_features(
            Temp=np.array([raw['Temp']]),
            Time=np.array([raw['Time']]),
            VOacac=np.array([raw['VOacac']]),
            DDT=np.array([raw['DDT']]),
            OAm=np.array([raw['OAm']])
        )
        feat_dict = raw.copy()
        for key, value in chem.items():
            feat_dict[key] = float(value[0])

    vector = []
    for feat in feature_names:
        if feat not in feat_dict:
            raise KeyError(f"Feature '{feat}' not available for mode '{feature_mode}'")
        vector.append(float(feat_dict[feat]))
    return vector
