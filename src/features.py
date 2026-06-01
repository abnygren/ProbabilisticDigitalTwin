"""Feature engineering for the Cu3MS4 BO workflow.

Transforms raw synthesis parameters (Temp, Time, VOacac, DDT, OAm) into the
chemically meaningful descriptors used by the GP models (ratios, concentrations,
log-time, optional precursor descriptors).

The legacy column name ``VOacac`` refers to the Group-5 metal precursor mmol
regardless of which metal is actually used (V, Nb, Ta), kept for backward
compatibility with the original Cu3VS4 CSV exports.

Also provides the VIF calculation used by the collinearity diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import (
    CU_PRECURSOR_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, RAW_BOUNDS, RAW_ROUNDING, CHEMICAL_BOUNDS,
    CHEM_FEATURES_BASIC, ENHANCED_FEATURE_CONFIG, CURRENT_PRECURSORS,
    SYNTHESIS_FEATURES, TRANSFER_MODE, TRANSFER_FEATURES,
    PRECURSOR_DESCRIPTOR_FEATURES,
)

try:
    import chemical_constants as chem_const
    CHEM_CONSTANTS_AVAILABLE = True
except ImportError:
    CHEM_CONSTANTS_AVAILABLE = False


CHEM_FEATURES_ENHANCED: List[str] = []
if CHEM_CONSTANTS_AVAILABLE:
    for feat, enabled in ENHANCED_FEATURE_CONFIG.items():
        if enabled:
            CHEM_FEATURES_ENHANCED.append(feat)

CHEM_FEATURES = CHEM_FEATURES_BASIC + CHEM_FEATURES_ENHANCED

HYBRID_FEATURES = RAW_FACTORS + ["Cu_V_ratio", "Metal_Conc"]
if 'effective_dielectric' in CHEM_FEATURES_ENHANCED:
    HYBRID_FEATURES.append('effective_dielectric')


def round_to_practical(params: Dict[str, float]) -> Dict[str, float]:
    """Round raw synthesis parameters to lab-practical precision."""
    return {
        k: round(float(v), RAW_ROUNDING.get(k, 3))
        for k, v in params.items()
    }


def raw_to_chemical_features(
    Temp: np.ndarray,
    Time: np.ndarray,
    VOacac: np.ndarray,
    DDT: np.ndarray,
    OAm: np.ndarray,
    cu_mmol: float = CU_PRECURSOR_MMOL,
    total_vol: float = TOTAL_VOLUME_ML,
    ddt_conv: float = DDT_MMOL_PER_ML,
    oam_conv: float = OAM_MMOL_PER_ML,
    include_enhanced: bool = True,
    cu_precursor: str = None,
    s_precursor: str = None,
    metal_precursor: str = None,
) -> Dict[str, np.ndarray]:
    """Transform raw synthesis parameters into chemically meaningful features.

    Parameters
    ----------
    VOacac : array-like
        Group-5 metal precursor amount in mmol (legacy column name; applies to
        VO(acac)2, NbCl5, and TaCl5 alike).
    cu_precursor : str, optional
        Cu precursor name for per-sample descriptor lookup.
    metal_precursor : str, optional
        Group-5 metal precursor name for per-sample descriptor lookup.
    """
    Temp = np.atleast_1d(Temp).astype(float)
    Time = np.atleast_1d(Time).astype(float)
    VOacac = np.atleast_1d(VOacac).astype(float)
    DDT = np.atleast_1d(DDT).astype(float)
    OAm = np.atleast_1d(OAm).astype(float)

    n_samples = len(Temp)
    total_metal = cu_mmol + VOacac
    DDT_mmol = DDT * ddt_conv
    OAm_mmol = OAm * oam_conv

    result = {
        'Temp': Temp,
        'Cu_V_ratio': cu_mmol / VOacac,
        'S_Metal_ratio': DDT_mmol / total_metal,
        'Ligand_Metal_ratio': OAm_mmol / total_metal,
        'Metal_Conc': (total_metal / total_vol) * 1000,
        'log_Time': np.log10(np.maximum(Time, 1.0)),
    }

    if include_enhanced and CHEM_CONSTANTS_AVAILABLE:
        cu_prec = cu_precursor or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI')
        s_prec = s_precursor or CURRENT_PRECURSORS.get('S_precursor', 'DDT')
        metal_prec = metal_precursor or CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2')

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

        if ENHANCED_FEATURE_CONFIG.get('Cu_hsab_mismatch', False):
            mismatch = chem_const.get_hsab_mismatch(cu_prec)
            result['Cu_hsab_mismatch'] = np.full(n_samples, mismatch)

        if ENHANCED_FEATURE_CONFIG.get('Metal_ionic_potential', False):
            ip = chem_const.get_metal_ionic_potential(metal_prec)
            result['Metal_ionic_potential'] = np.full(n_samples, ip)

        if ENHANCED_FEATURE_CONFIG.get('Metal_hsab_mismatch', False):
            mm = chem_const.get_metal_hsab_mismatch(metal_prec)
            result['Metal_hsab_mismatch'] = np.full(n_samples, mm)

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
    cu_mmol: float = CU_PRECURSOR_MMOL,
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
    VOacac = cu_mmol / Cu_V_ratio
    total_metal = cu_mmol + VOacac
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
    """Add all chemical features to a dataframe.

    If a ``Cu_precursor`` or ``Metal_precursor`` column is present, per-row
    precursor descriptors are looked up so pooled multi-precursor datasets
    get the right values for each experiment. Otherwise the
    ``CURRENT_PRECURSORS`` defaults are used (single-precursor campaigns).
    """
    df = df.copy()

    has_cu_col = 'Cu_precursor' in df.columns
    has_metal_col = 'Metal_precursor' in df.columns

    chem = raw_to_chemical_features(
        Temp=df['Temp'].values,
        Time=df['Time'].values,
        VOacac=df['VOacac'].values,
        DDT=df['DDT'].values,
        OAm=df['OAm'].values,
        include_enhanced=not (has_cu_col or has_metal_col),
    )
    for name, values in chem.items():
        if name != 'Temp':
            df[name] = values

    if CHEM_CONSTANTS_AVAILABLE:
        if has_cu_col:
            if ENHANCED_FEATURE_CONFIG.get('Cu_precursor_hardness', False):
                df['Cu_precursor_hardness'] = df['Cu_precursor'].apply(
                    chem_const.get_precursor_hardness)
            if ENHANCED_FEATURE_CONFIG.get('Cu_hsab_mismatch', False):
                df['Cu_hsab_mismatch'] = df['Cu_precursor'].apply(
                    chem_const.get_hsab_mismatch)
        else:
            cu_prec = CURRENT_PRECURSORS.get('Cu_precursor', 'CuI')
            if ENHANCED_FEATURE_CONFIG.get('Cu_precursor_hardness', False):
                df['Cu_precursor_hardness'] = chem_const.get_precursor_hardness(cu_prec)
            if ENHANCED_FEATURE_CONFIG.get('Cu_hsab_mismatch', False):
                df['Cu_hsab_mismatch'] = chem_const.get_hsab_mismatch(cu_prec)

        if has_metal_col:
            if ENHANCED_FEATURE_CONFIG.get('Metal_ionic_potential', False):
                df['Metal_ionic_potential'] = df['Metal_precursor'].apply(
                    chem_const.get_metal_ionic_potential)
            if ENHANCED_FEATURE_CONFIG.get('Metal_hsab_mismatch', False):
                df['Metal_hsab_mismatch'] = df['Metal_precursor'].apply(
                    chem_const.get_metal_hsab_mismatch)
        else:
            metal_prec = CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2')
            if ENHANCED_FEATURE_CONFIG.get('Metal_ionic_potential', False):
                df['Metal_ionic_potential'] = chem_const.get_metal_ionic_potential(metal_prec)
            if ENHANCED_FEATURE_CONFIG.get('Metal_hsab_mismatch', False):
                df['Metal_hsab_mismatch'] = chem_const.get_metal_hsab_mismatch(metal_prec)

    return df


def calculate_vif(X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """Variance Inflation Factor for multicollinearity detection.

    VIF interpretation:
        VIF == 1   no correlation with other features
        VIF < 5    low collinearity (acceptable)
        5 <= VIF < 10  moderate collinearity (caution)
        VIF >= 10  high collinearity (problematic)
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
            return "Low (acceptable)"
        elif v < 10:
            return "Moderate (caution)"
        else:
            return "High (problematic)"

    results = pd.DataFrame({
        'Feature': feature_names,
        'VIF': vif_values,
    })
    results['Interpretation'] = results['VIF'].apply(interpret_vif)
    return results.sort_values('VIF', ascending=False)


def _resolve_precursor_feature_value(feat: str) -> float:
    """Look up the fixed value for a precursor descriptor feature.

    Uses the target precursor from ``TRANSFER_MODE`` (for recommendations)
    or falls back to ``CURRENT_PRECURSORS``.
    """
    cu_prec = (TRANSFER_MODE.get('target_cu_precursor')
               or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI'))
    metal_prec = (TRANSFER_MODE.get('target_metal_precursor')
                  or CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2'))

    lookup = {
        'Cu_precursor_hardness': lambda: chem_const.get_precursor_hardness(cu_prec),
        'Cu_hsab_mismatch':      lambda: chem_const.get_hsab_mismatch(cu_prec),
        'Metal_ionic_potential':  lambda: chem_const.get_metal_ionic_potential(metal_prec),
        'Metal_hsab_mismatch':   lambda: chem_const.get_metal_hsab_mismatch(metal_prec),
    }
    if feat not in lookup:
        raise ValueError(f"Unknown precursor descriptor feature: {feat}")
    return lookup[feat]()


def compute_feature_bounds(feature_list: List[str], df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute search bounds for a list of features.

    Priority order:
      1. Precursor descriptors -- pinned to the target precursor (point bounds).
      2. ``RAW_BOUNDS`` -- hard lab limits for raw parameters.
      3. ``CHEMICAL_BOUNDS`` -- analytic limits for chemical features so the
         reconstructed raw values stay inside ``RAW_BOUNDS``.
      4. Data-derived 5% margin on observed values (fallback only).
    """
    bounds = {}
    for feat in feature_list:
        if feat in PRECURSOR_DESCRIPTOR_FEATURES:
            val = _resolve_precursor_feature_value(feat)
            bounds[feat] = (val, val)
        elif feat in RAW_BOUNDS:
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
    feature_names: List[str],
    cu_precursor: str = None,
    metal_precursor: str = None,
) -> List[float]:
    """Build a feature vector in the base optimizer's feature space from raw conditions.

    For transfer mode, precursor descriptor features are resolved from the
    supplied precursor names (or TRANSFER_MODE / CURRENT_PRECURSORS defaults).
    """
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
            OAm=np.array([raw['OAm']]),
            cu_precursor=cu_precursor,
            metal_precursor=metal_precursor,
        )
        feat_dict = raw.copy()
        for key, value in chem.items():
            feat_dict[key] = float(value[0])

    if CHEM_CONSTANTS_AVAILABLE:
        for feat in feature_names:
            if feat in PRECURSOR_DESCRIPTOR_FEATURES and feat not in feat_dict:
                feat_dict[feat] = _resolve_precursor_feature_value(feat)

    vector = []
    for feat in feature_names:
        if feat not in feat_dict:
            raise KeyError(f"Feature '{feat}' not available for mode '{feature_mode}'")
        vector.append(float(feat_dict[feat]))
    return vector


def validate_bounds_roundtrip(
    n_samples: int = 500,
    seed: int = 42,
    warn: bool = True,
) -> Dict[str, int]:
    """Verify that points sampled uniformly inside ``CHEMICAL_BOUNDS`` back-transform
    to within ``RAW_BOUNDS`` under the synthesis-mode reconstruction.

    The reconstruction (matches ``Cu3VS4Optimizer._synthesis_to_raw``)::

        VOacac      = CuI / Cu_V_ratio
        Time        = 10 ** log_Time
        total_metal = CuI + VOacac
        DDT         = S_Metal_ratio      * total_metal / DDT_MMOL_PER_ML
        OAm         = Ligand_Metal_ratio * total_metal / OAM_MMOL_PER_ML

    Parameters
    ----------
    n_samples : int
        Number of uniform samples drawn inside the chemical-feature box.
    seed : int
        RNG seed.
    warn : bool
        Emit a ``UserWarning`` summarising violations. Set False for a silent
        diagnostic that only returns the count dict.

    Returns
    -------
    dict
        Mapping ``{raw_param: n_violations}``, populated only for params with
        at least one out-of-bounds back-transformed sample.

    Notes
    -----
    The optimizer clips back-transformed raw conditions to ``RAW_BOUNDS`` via
    ``np.clip`` before reporting recommendations, so a non-empty result does
    NOT mean unsafe recommendations are issued -- it only means the
    chemical-feature box is slightly larger than the strict pre-image of the
    raw box (because the ``S_Metal_ratio`` / ``Ligand_Metal_ratio`` bounds use
    worst-case extremes of VOacac). Tighten ``CHEMICAL_BOUNDS`` in
    ``config.py`` if you want a fully tight chemical box.

    Not run at import time -- call it explicitly when auditing the bounds.
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
        raw['VOacac'] = CU_PRECURSOR_MMOL / feat_dict['Cu_V_ratio']
        raw['Time'] = 10 ** feat_dict['log_Time']
        total_metal = CU_PRECURSOR_MMOL + raw['VOacac']
        raw['DDT'] = (feat_dict['S_Metal_ratio'] * total_metal) / DDT_MMOL_PER_ML
        raw['OAm'] = (feat_dict['Ligand_Metal_ratio'] * total_metal) / OAM_MMOL_PER_ML

        for k in RAW_FACTORS:
            if k not in raw:
                continue
            lo, hi = RAW_BOUNDS[k]
            if raw[k] < lo - 1e-6 or raw[k] > hi + 1e-6:
                violations.append(k)

    from collections import Counter
    counts = dict(Counter(violations))

    if warn and violations:
        import warnings
        warnings.warn(
            f"Bounds round-trip: {len(violations)} out-of-bounds values in "
            f"{n_samples} samples (by param: {counts}). The optimizer clips "
            "these to RAW_BOUNDS before reporting, but CHEMICAL_BOUNDS could "
            "be tightened to remove the slack.",
            stacklevel=2,
        )

    return counts
