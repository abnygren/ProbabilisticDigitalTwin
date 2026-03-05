"""
Feature engineering for Cu₃VS₄ Bayesian Optimization

Transforms raw synthesis parameters (Temp, Time, VOacac, DDT, OAm) into
chemically meaningful features (ratios, concentrations).
Provides VIF calculation for collinearity diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import (
    CUI_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, RAW_BOUNDS, RAW_ROUNDING, CHEMICAL_BOUNDS,
    CHEM_FEATURES_BASIC, ENHANCED_FEATURE_CONFIG, CURRENT_PRECURSORS,
    SYNTHESIS_FEATURES,
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
# COLLINEARITY DIAGNOSTICS (VIF)
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


# =============================================================================
# BOUNDS & FEATURE VECTOR HELPERS
# =============================================================================

def compute_feature_bounds(feature_list: List[str], df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Compute search bounds for a list of features.

    Priority order:
      1. RAW_BOUNDS      — hard lab limits for raw parameters (Temp, Time, DDT, ...)
      2. CHEMICAL_BOUNDS — analytically derived limits for chemical features; these
                           guarantee the reconstructed raw values stay inside RAW_BOUNDS
      3. Data-derived    — 5% outward margin on observed data (fallback only)
    """
    bounds = {}
    for feat in feature_list:
        if feat in RAW_BOUNDS:
            bounds[feat] = RAW_BOUNDS[feat]
        elif feat in CHEMICAL_BOUNDS:
            bounds[feat] = CHEMICAL_BOUNDS[feat]
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


# =============================================================================
# BOUNDS ROUND-TRIP VALIDATION
# =============================================================================

def validate_bounds_roundtrip(n_samples: int = 500, seed: int = 42):
    """Verify that points within CHEMICAL_BOUNDS back-transform to within RAW_BOUNDS.

    Tests the synthesis-mode reconstruction path:
      VOacac      = CuI / Cu_V_ratio
      Time        = 10^log_Time
      total_metal = CuI + VOacac
      DDT         = S_Metal_ratio * total_metal / DDT_MMOL_PER_ML
      OAm         = Ligand_Metal_ratio * total_metal / OAM_MMOL_PER_ML
    """
    rng = np.random.RandomState(seed)
    all_bounds = {**RAW_BOUNDS, **CHEMICAL_BOUNDS}
    synthesis_feats = list(SYNTHESIS_FEATURES)

    violations = []
    for _ in range(n_samples):
        feat_dict = {}
        for feat in synthesis_feats:
            lo, hi = all_bounds[feat]
            feat_dict[feat] = rng.uniform(lo, hi)

        raw = {}
        raw['Temp'] = feat_dict['Temp']
        raw['VOacac'] = CUI_MMOL / feat_dict['Cu_V_ratio']
        raw['Time'] = 10 ** feat_dict['log_Time']
        total_metal = CUI_MMOL + raw['VOacac']
        raw['DDT'] = (feat_dict['S_Metal_ratio'] * total_metal) / DDT_MMOL_PER_ML
        raw['OAm'] = (feat_dict['Ligand_Metal_ratio'] * total_metal) / OAM_MMOL_PER_ML

        for k in RAW_FACTORS:
            if k not in raw:
                continue
            lo, hi = RAW_BOUNDS[k]
            if raw[k] < lo - 1e-6 or raw[k] > hi + 1e-6:
                violations.append(k)

    if violations:
        from collections import Counter
        import warnings
        counts = Counter(violations)
        warnings.warn(
            f"Bounds round-trip: {len(violations)} out-of-bounds values in {n_samples} samples "
            f"(by param: {dict(counts)}). The optimizer clamps these, but CHEMICAL_BOUNDS "
            f"could be tightened to avoid.",
            stacklevel=2,
        )


validate_bounds_roundtrip()
