"""
Cu3VS4 Bayesian Optimization Module

This module contains the Cu3VS4Optimizer class and all necessary helper functions
for Bayesian optimization of Cu3VS4 nanoparticle synthesis.

Import this in your notebooks instead of running multiple notebooks in sequence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import warnings

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Statistical imports
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.stats.qmc import LatinHypercube

warnings.filterwarnings('ignore')

# =============================================================================
# SYSTEM CONSTANTS
# =============================================================================
CUI_MMOL = 0.25              # Fixed CuI amount (mmol)
TOTAL_VOLUME_ML = 12.0       # Fixed total volume (mL)
DDT_MMOL_PER_ML = 4.17       # DDT conversion (MW=202.4, ρ=0.845)
OAM_MMOL_PER_ML = 3.04       # OAm conversion (MW=267.5, ρ=0.813)

# =============================================================================
# FACTOR DEFINITIONS
# =============================================================================
RAW_FACTORS = ["Temp", "Time", "VOacac", "DDT", "OAm"]

# Chemical features derived from raw factors (BASIC)
CHEM_FEATURES_BASIC = [
    "Cu_V_ratio",         # Stoichiometry
    "S_Metal_ratio",      # Sulfur excess
    "Ligand_Metal_ratio", # Capping density
    "Metal_Conc",         # Total metal concentration
    "log_Time",           # Linearized kinetics
]

# Try to import chemical constants
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import chemical_constants as chem_const
    CHEM_CONSTANTS_AVAILABLE = True
except ImportError:
    CHEM_CONSTANTS_AVAILABLE = False

# Enhanced feature configuration
ENHANCED_FEATURE_CONFIG = {
    'effective_dielectric': True if CHEM_CONSTANTS_AVAILABLE else False,
    'Cu_precursor_hardness': False,
    'hsab_mismatch': False,
    'S_BDE': False,
}

CURRENT_PRECURSORS = {
    'Cu_precursor': 'CuI',
    'S_precursor': 'DDT',
    'V_precursor': 'VO(acac)2',
}

# Build enhanced features list
CHEM_FEATURES_ENHANCED = []
if CHEM_CONSTANTS_AVAILABLE:
    for feat, enabled in ENHANCED_FEATURE_CONFIG.items():
        if enabled:
            CHEM_FEATURES_ENHANCED.append(feat)

# Combined feature lists
CHEM_FEATURES = CHEM_FEATURES_BASIC + CHEM_FEATURES_ENHANCED
HYBRID_FEATURES = RAW_FACTORS + ["Cu_V_ratio", "Metal_Conc"]
if 'effective_dielectric' in CHEM_FEATURES_ENHANCED:
    HYBRID_FEATURES.append('effective_dielectric')

# Smart hybrid: will be computed dynamically to avoid collinearity
# (see select_smart_hybrid_features function below)
SMART_HYBRID_FEATURES = None  # Computed at runtime

# Bounds
RAW_BOUNDS = {
    "Temp":   (260.0, 310.0),
    "Time":   (8.0, 90.0),
    "VOacac": (0.08, 0.66),
    "DDT":    (1.0, 5.0),
    "OAm":    (1.0, 7.0),
}

# Objectives & feasibility
OBJECTIVES = ["Size", "GSD", "Squareness"]
FEAS_COLS = ["HasProduct", "PhasePure"]
POLYMORPH_COL = "Polymorph"
CUBIC_LABEL = "cubic"


# =============================================================================
# FEATURE TRANSFORMATION
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
    
    # Basic chemical features
    result = {
        'Temp': Temp,
        'Cu_V_ratio': CuI / VOacac,
        'S_Metal_ratio': DDT_mmol / total_metal,
        'Ligand_Metal_ratio': OAm_mmol / total_metal,
        'Metal_Conc': (total_metal / total_vol) * 1000,  # Convert to mM
        'log_Time': np.log10(np.maximum(Time, 1.0)),
    }
    
    # Enhanced chemical descriptors (if available and enabled)
    if include_enhanced and CHEM_CONSTANTS_AVAILABLE:
        cu_prec = cu_precursor or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI')
        s_prec = s_precursor or CURRENT_PRECURSORS.get('S_precursor', 'DDT')
        
        # Effective dielectric constant
        if ENHANCED_FEATURE_CONFIG.get('effective_dielectric', False):
            ODE_vol = total_vol - DDT - OAm
            eps_mix = np.zeros(n_samples)
            for i in range(n_samples):
                volumes = {'DDT': DDT[i], 'OAm': OAm[i], 'ODE': ODE_vol[i]}
                eps_mix[i] = chem_const.calculate_mixture_dielectric(volumes, total_vol)
            result['effective_dielectric'] = eps_mix
        
        # Other enhanced features (if enabled)
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
        if name != 'Temp':  # Don't duplicate Temp
            df[name] = values
    
    return df


def select_smart_hybrid_features(
    df: pd.DataFrame,
    correlation_threshold: float = 0.8,
    verbose: bool = True
) -> List[str]:
    """
    Select hybrid features intelligently by removing raw features that are
    highly correlated with chemical features.
    
    Strategy:
    1. Start with all raw factors
    2. Add key chemical features (Cu_V_ratio, Metal_Conc, effective_dielectric)
    3. For each raw feature, check correlation with chemical features
    4. If |correlation| > threshold, drop the raw feature and keep the chemical one
    
    Parameters
    ----------
    df : DataFrame
        Data with both raw and chemical features
    correlation_threshold : float
        Correlation threshold above which to drop raw features (default: 0.8)
    verbose : bool
        Print feature selection decisions
        
    Returns
    -------
    list of str
        Selected feature names for smart hybrid mode
    """
    # Ensure we have chemical features
    if 'Cu_V_ratio' not in df.columns:
        df = add_chemical_features(df)
    
    # Start with candidate raw and chemical features
    raw_candidates = RAW_FACTORS.copy()
    
    # Include ALL chemical features needed for reconstruction
    # (not just Cu_V_ratio and Metal_Conc)
    chemical_candidates = [
        "Cu_V_ratio",       # To reconstruct VOacac
        "S_Metal_ratio",    # To reconstruct DDT
        "Ligand_Metal_ratio",  # To reconstruct OAm
        "Metal_Conc",       # For total metal calculation
        "log_Time"          # To reconstruct Time
    ]
    
    # Add enhanced features if available
    if 'effective_dielectric' in df.columns:
        chemical_candidates.append('effective_dielectric')
    
    # Compute correlation matrix
    all_features = raw_candidates + chemical_candidates
    available_features = [f for f in all_features if f in df.columns]
    corr_matrix = df[available_features].corr()
    
    # Map: which chemical feature can reconstruct which raw feature
    reconstruction_map = {
        'VOacac': 'Cu_V_ratio',
        'DDT': 'S_Metal_ratio',
        'OAm': 'Ligand_Metal_ratio',
        'Time': 'log_Time'
    }
    
    # Decide which raw features to keep
    selected_features = []
    dropped_raw = []
    required_chemical = set()  # Chemical features needed for reconstruction
    
    for raw_feat in raw_candidates:
        if raw_feat not in df.columns:
            continue
            
        # Check correlation with each chemical feature
        max_corr = 0.0
        culprit = None
        
        for chem_feat in chemical_candidates:
            if chem_feat not in df.columns:
                continue
            corr = abs(corr_matrix.loc[raw_feat, chem_feat])
            if corr > max_corr:
                max_corr = corr
                culprit = chem_feat
        
        # Decision: keep raw feature if correlation is below threshold
        if max_corr < correlation_threshold:
            selected_features.append(raw_feat)
            if verbose:
                print(f"  ✓ Keep {raw_feat:12s} (max |r| = {max_corr:.3f} with {culprit})")
        else:
            dropped_raw.append((raw_feat, culprit, max_corr))
            # Mark the corresponding reconstruction feature as required
            if raw_feat in reconstruction_map:
                required_chemical.add(reconstruction_map[raw_feat])
            if verbose:
                print(f"  ✗ Drop {raw_feat:12s} (|r| = {max_corr:.3f} with {culprit}) "
                      f"→ requires {reconstruction_map.get(raw_feat, '?')} for reconstruction")
    
    # Add required chemical features (needed for reconstruction)
    for chem_feat in required_chemical:
        if chem_feat in df.columns and chem_feat not in selected_features:
            selected_features.append(chem_feat)
    
    # Add Metal_Conc if any reconstruction is needed (it's needed for the calculation)
    if required_chemical and 'Metal_Conc' in df.columns and 'Metal_Conc' not in selected_features:
        selected_features.append('Metal_Conc')
    
    # Add optional enhanced features if not correlated with selected raw features
    for chem_feat in ['effective_dielectric']:
        if chem_feat in df.columns and chem_feat not in selected_features:
            # Only add if it's not redundant with kept raw features
            selected_features.append(chem_feat)
    
    if verbose:
        print(f"\n✓ Smart Hybrid Mode: {len(selected_features)} features selected")
        print(f"  Raw features kept: {[f for f in selected_features if f in RAW_FACTORS]}")
        print(f"  Chemical features: {[f for f in selected_features if f not in RAW_FACTORS]}")
        if dropped_raw:
            print(f"  Dropped {len(dropped_raw)} collinear raw features:")
            for raw, chem, corr in dropped_raw:
                print(f"    {raw} → replaced by {chem} (|r|={corr:.3f})")
    
    return selected_features


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


# =============================================================================
# GP BUILDERS
# =============================================================================

def make_gp_regressor(n_features: int, alpha: float = 1e-4) -> GaussianProcessRegressor:
    """
    Create GP regressor with Matérn 5/2 ARD kernel.
    
    Parameters
    ----------
    n_features : int
        Number of input features (for ARD lengthscales)
    alpha : float
        Regularization parameter (nugget). Default 1e-4 provides numerical
        stability and prevents overfitting. Increase to 1e-3 or 1e-2 if
        data is noisy or sample size is small relative to features.
        
    Notes
    -----
    The alpha parameter adds to the diagonal of the kernel matrix, acting as
    Tikhonov regularization. Too small (e.g., 1e-10) can cause numerical
    instability and overconfident predictions. Too large reduces model
    flexibility.
    """
    kernel = (
        C(1.0, (0.001, 1000.0)) * 
        Matern(
            length_scale=[1.0] * n_features,
            length_scale_bounds=(0.01, 100.0),
            nu=2.5
        ) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1.0))
    )
    
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=15,
        random_state=42,
        alpha=alpha
    )


def make_gp_classifier(n_features: int) -> GaussianProcessClassifier:
    """Create GP classifier for feasibility."""
    kernel = C(1.0, (0.01, 100.0)) * Matern(
        length_scale=[1.0] * n_features,
        length_scale_bounds=(0.1, 10.0),
        nu=2.5
    )
    
    return GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42,
        max_iter_predict=200
    )


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = 0.01,
    minimize: bool = True
) -> np.ndarray:
    """Expected Improvement acquisition function."""
    sigma = np.maximum(sigma, 1e-9)
    
    if minimize:
        improvement = y_best - mu - xi
    else:
        improvement = mu - y_best - xi
    
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    return np.maximum(ei, 0.0)


def prob_in_interval(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    tolerance: float
) -> np.ndarray:
    """P(target - tol ≤ Y ≤ target + tol)."""
    sigma = np.maximum(sigma, 1e-9)
    z_hi = (target + tolerance - mu) / sigma
    z_lo = (target - tolerance - mu) / sigma
    return norm.cdf(z_hi) - norm.cdf(z_lo)


# =============================================================================
# SAMPLING
# =============================================================================

def latin_hypercube_sample(
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    feature_order: List[str],
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate LHS samples."""
    n_dims = len(feature_order)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    
    lows = np.array([bounds[k][0] for k in feature_order])
    highs = np.array([bounds[k][1] for k in feature_order])
    
    return samples * (highs - lows) + lows


# =============================================================================
# WEIGHT CALCULATION
# =============================================================================

def calculate_objective_weights(
    df_success: pd.DataFrame,
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    method: str = 'comprehensive'
) -> Dict[str, float]:
    """Calculate statistically sound objective weights based on data and model performance."""
    weights = {}
    
    gsd_data = df_success['GSD'].values
    sq_data = df_success['Squareness'].values
    
    cv_gsd = np.std(gsd_data) / (np.mean(gsd_data) + 1e-10)
    cv_sq = np.std(sq_data) / (np.mean(sq_data) + 1e-10)
    
    gsd_range = (gsd_data.max() - gsd_data.min()) / (np.mean(gsd_data) + 1e-10)
    sq_range = (sq_data.max() - sq_data.min()) / (np.mean(sq_data) + 1e-10)
    
    gsd_ideal = 1.0
    sq_ideal = 1.0
    gsd_distance = np.abs(np.mean(gsd_data) - gsd_ideal) / (np.std(gsd_data) + 1e-10)
    sq_distance = np.abs(np.mean(sq_data) - sq_ideal) / (np.std(sq_data) + 1e-10)
    
    if method == 'variability':
        w_gsd = cv_gsd
        w_sq = cv_sq
    elif method == 'uncertainty':
        if model_metrics is None:
            raise ValueError("model_metrics required for 'uncertainty' method")
        rmse_gsd = model_metrics.get('GSD', {}).get('rmse', 0.1)
        rmse_sq = model_metrics.get('Squareness', {}).get('rmse', 0.1)
        w_gsd = rmse_gsd / (np.mean(gsd_data) + 1e-10)
        w_sq = rmse_sq / (np.mean(sq_data) + 1e-10)
    elif method == 'improvement':
        w_gsd = gsd_range
        w_sq = sq_range
    elif method == 'comprehensive':
        factors_gsd = [cv_gsd, gsd_range, gsd_distance]
        factors_sq = [cv_sq, sq_range, sq_distance]
        
        if model_metrics is not None:
            rmse_gsd = model_metrics.get('GSD', {}).get('rmse', 0.1)
            rmse_sq = model_metrics.get('Squareness', {}).get('rmse', 0.1)
            rmse_gsd_norm = rmse_gsd / (np.mean(gsd_data) + 1e-10)
            rmse_sq_norm = rmse_sq / (np.mean(sq_data) + 1e-10)
            factors_gsd.append(rmse_gsd_norm)
            factors_sq.append(rmse_sq_norm)
            
            r2_gsd = model_metrics.get('GSD', {}).get('r2', 0.0)
            r2_sq = model_metrics.get('Squareness', {}).get('r2', 0.0)
            reliability_gsd = 1.0 - r2_gsd if r2_gsd >= 0 else 1.0 + abs(r2_gsd)
            reliability_sq = 1.0 - r2_sq if r2_sq >= 0 else 1.0 + abs(r2_sq)
            factors_gsd.append(reliability_gsd)
            factors_sq.append(reliability_sq)
        
        w_gsd = np.prod([f for f in factors_gsd if f > 0]) ** (1.0 / len(factors_gsd))
        w_sq = np.prod([f for f in factors_sq if f > 0]) ** (1.0 / len(factors_sq))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    total = w_gsd + w_sq
    if total > 0:
        scale = 2.0 / total
        w_gsd *= scale
        w_sq *= scale
    else:
        w_gsd = 1.0
        w_sq = 1.0
    
    return {
        'GSD': float(w_gsd),
        'Squareness': float(w_sq)
    }


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    gp_factory: Callable,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """Leave-One-Out CV with proper re-fitting."""
    n = len(y)
    y_pred = np.zeros(n)
    y_std = np.zeros(n)
    
    for train_idx, test_idx in LeaveOneOut().split(X):
        gp = gp_factory()
        gp.fit(X[train_idx], y[train_idx])
        mu, std = gp.predict(X[test_idx], return_std=True)
        y_pred[test_idx] = mu
        y_std[test_idx] = std
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    z = np.abs(y - y_pred) / np.maximum(y_std, 1e-9)
    cal_95 = np.mean(z < 1.96)
    cal_68 = np.mean(z < 1.0)
    
    result = {'r2': r2, 'rmse': rmse, 'mae': mae, 'cal_95': cal_95, 'cal_68': cal_68}
    
    if return_predictions:
        result['y_pred'] = y_pred
        result['y_std'] = y_std
    
    return result


# =============================================================================
# FEATURE MODE COMPARISON
# =============================================================================

def compare_feature_modes(
    df: pd.DataFrame,
    modes: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare LOO-CV performance across different feature representations.
    
    This helps justify the choice of feature mode (raw, chemical, hybrid, smart_hybrid)
    based on actual predictive performance rather than assumptions.
    
    Parameters
    ----------
    df : DataFrame
        Full dataset with all required columns
    modes : list of str, optional
        Feature modes to compare. Default: ['raw', 'chemical', 'hybrid', 'smart_hybrid']
    verbose : bool
        Print results during comparison
        
    Returns
    -------
    DataFrame with columns: Mode, Property, R2, RMSE, MAE, Cal_68, Cal_95
    
    Example
    -------
    >>> comparison = compare_feature_modes(df)
    >>> print(comparison.pivot(index='Property', columns='Mode', values='R2'))
    """
    if modes is None:
        modes = ['raw', 'chemical', 'hybrid', 'smart_hybrid']
    
    results = []
    
    for mode in modes:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating {mode.upper()} feature mode...")
            print(f"{'='*50}")
        
        try:
            # Build optimizer with validation
            opt = Cu3VS4Optimizer(df, feature_mode=mode, validate=True)
            
            for prop in ['Size', 'GSD', 'Squareness']:
                if prop in opt.metrics:
                    m = opt.metrics[prop]
                    results.append({
                        'Mode': mode,
                        'Property': prop,
                        'N_Features': len(opt.features),
                        'R2': m['r2'],
                        'RMSE': m['rmse'],
                        'MAE': m['mae'],
                        'Cal_68': m['cal_68'],  # Should be ~0.68
                        'Cal_95': m['cal_95'],  # Should be ~0.95
                    })
        except Exception as e:
            if verbose:
                print(f"  Error with {mode} mode: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if verbose and not results_df.empty:
        print(f"\n{'='*60}")
        print("FEATURE MODE COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Pivot for nice display
        for metric in ['R2', 'RMSE']:
            print(f"\n{metric} by Mode and Property:")
            pivot = results_df.pivot(index='Property', columns='Mode', values=metric)
            print(pivot.round(3).to_string())
        
        # Recommendation
        mean_r2 = results_df.groupby('Mode')['R2'].mean()
        best_mode = mean_r2.idxmax()
        print(f"\n✓ Recommended mode based on mean R²: {best_mode} ({mean_r2[best_mode]:.3f})")
        
        # Calibration check
        print("\nCalibration Analysis (target: Cal_68 ≈ 0.68, Cal_95 ≈ 0.95):")
        cal_summary = results_df.groupby('Mode')[['Cal_68', 'Cal_95']].mean()
        print(cal_summary.round(3).to_string())
    
    return results_df


# =============================================================================
# EXTRAPOLATION DETECTION
# =============================================================================

def detect_extrapolation(
    X_new: np.ndarray,
    X_train: np.ndarray,
    scaler: StandardScaler = None,
    threshold: float = 2.0,
    method: str = 'nearest'
) -> Dict[str, Any]:
    """
    Detect if new points are in extrapolation regions (far from training data).
    
    GP predictions degrade rapidly in extrapolation regions. This function
    provides warnings and quantifies how far new points are from training data.
    
    Parameters
    ----------
    X_new : array (n_new, n_features)
        New points to check
    X_train : array (n_train, n_features)
        Training data points
    scaler : StandardScaler, optional
        If provided, scaling is applied. If None, assumes already scaled.
    threshold : float
        Distance threshold (in scaled space) above which a point is
        considered extrapolation. Default 2.0 means 2 standard deviations
        from nearest training point.
    method : str
        'nearest' - distance to nearest training point
        'hull' - whether point is inside convex hull (more conservative)
        
    Returns
    -------
    dict with:
        'is_extrapolation': bool array, True if point is extrapolating
        'distances': float array, distance to nearest training point
        'n_extrapolating': int, count of extrapolating points
        'warnings': list of str, human-readable warnings
    """
    X_new = np.atleast_2d(X_new)
    X_train = np.atleast_2d(X_train)
    
    # Scale if scaler provided
    if scaler is not None:
        X_new_scaled = scaler.transform(X_new)
        X_train_scaled = scaler.transform(X_train)
    else:
        X_new_scaled = X_new
        X_train_scaled = X_train
    
    # Compute distances to nearest training point
    distances = cdist(X_new_scaled, X_train_scaled).min(axis=1)
    
    # Determine extrapolation
    is_extrapolation = distances > threshold
    
    # Build warnings
    warnings = []
    n_extrap = is_extrapolation.sum()
    
    if n_extrap > 0:
        warnings.append(
            f"⚠️ {n_extrap}/{len(X_new)} points are in extrapolation regions "
            f"(distance > {threshold} from nearest training point)"
        )
        
        # Find which points are worst
        worst_idx = np.argmax(distances)
        warnings.append(
            f"   Worst case: point {worst_idx} is {distances[worst_idx]:.2f} "
            f"std from training data"
        )
        
        if n_extrap > len(X_new) * 0.5:
            warnings.append(
                "   ⚠️ CAUTION: Majority of candidates are extrapolating. "
                "Consider expanding training data in this region."
            )
    
    return {
        'is_extrapolation': is_extrapolation,
        'distances': distances,
        'n_extrapolating': int(n_extrap),
        'fraction_extrapolating': n_extrap / len(X_new),
        'max_distance': float(distances.max()),
        'mean_distance': float(distances.mean()),
        'threshold': threshold,
        'warnings': warnings,
    }


# =============================================================================
# COLLINEARITY DIAGNOSTICS (VIF)
# =============================================================================

def calculate_vif(X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity. High VIF (>5-10) indicates problematic collinearity.
    
    Parameters
    ----------
    X : array (n_samples, n_features)
        Feature matrix
    feature_names : list of str, optional
        Names for features. If None, uses Feature_0, Feature_1, etc.
        
    Returns
    -------
    DataFrame with columns: Feature, VIF, Interpretation
    
    Notes
    -----
    VIF interpretation:
    - VIF = 1: No correlation with other features
    - VIF < 5: Low collinearity (acceptable)
    - VIF 5-10: Moderate collinearity (caution)
    - VIF > 10: High collinearity (problematic)
    
    For Cu₃VS₄ synthesis, derived features like Cu_V_ratio and Metal_Conc
    are algebraically related to raw factors, which may cause high VIF.
    Consider using either raw OR chemical features, not both.
    """
    X = np.atleast_2d(X)
    n_features = X.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Add intercept for VIF calculation
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    vif_values = []
    
    for i in range(n_features):
        # Get the feature to analyze
        y = X[:, i]
        
        # Get other features (excluding the one we're analyzing)
        other_idx = [j for j in range(n_features) if j != i]
        X_other = X[:, other_idx]
        
        # Add intercept
        X_other_const = np.column_stack([np.ones(len(X)), X_other])
        
        # Fit OLS and get R²
        try:
            # Use least squares to get R²
            coeffs, residuals, rank, s = np.linalg.lstsq(X_other_const, y, rcond=None)
            y_pred = X_other_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                r_squared = max(0, min(r_squared, 0.9999))  # Clip to avoid division by zero
            else:
                r_squared = 0
            
            vif = 1 / (1 - r_squared)
        except:
            vif = np.inf
        
        vif_values.append(vif)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Feature': feature_names,
        'VIF': vif_values,
    })
    
    # Add interpretation
    def interpret_vif(v):
        if v < 5:
            return "✓ Low (acceptable)"
        elif v < 10:
            return "⚠ Moderate (caution)"
        else:
            return "✗ High (problematic)"
    
    results['Interpretation'] = results['VIF'].apply(interpret_vif)
    results = results.sort_values('VIF', ascending=False)
    
    return results


def diagnose_collinearity(
    df: pd.DataFrame,
    feature_mode: str = 'hybrid',
    verbose: bool = True,
    feature_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive collinearity diagnostics for a given feature mode.
    
    Parameters
    ----------
    df : DataFrame
        Dataset with synthesis parameters
    feature_mode : str
        'raw', 'chemical', 'hybrid', or 'smart_hybrid'
    verbose : bool
        Print detailed results
    feature_list : list of str, optional
        If provided, use this feature list instead of looking up by mode.
        Required for 'smart_hybrid' mode.
        
    Returns
    -------
    dict with VIF results, correlation matrix, and recommendations
    """
    # Get feature list
    if feature_list is not None:
        features = feature_list
    elif feature_mode == 'raw':
        features = RAW_FACTORS
    elif feature_mode == 'chemical':
        features = CHEM_FEATURES
    elif feature_mode == 'hybrid':
        features = HYBRID_FEATURES
    elif feature_mode == 'smart_hybrid':
        raise ValueError(
            "For smart_hybrid mode, you must provide feature_list parameter. "
            "Use optimizer.get_collinearity_diagnostics() which handles this automatically."
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    # Add chemical features if needed
    df_work = df.copy()
    needs_chem = feature_mode in ['chemical', 'hybrid', 'smart_hybrid'] or (
        feature_list is not None and any(f in CHEM_FEATURES for f in feature_list)
    )
    if needs_chem and 'Cu_V_ratio' not in df_work.columns:
        df_work = add_chemical_features(df_work)
    
    # Extract features
    available = [f for f in features if f in df_work.columns]
    X = df_work[available].values
    
    # Calculate VIF
    vif_df = calculate_vif(X, available)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, index=available, columns=available)
    
    # Find problematic pairs (|r| > 0.8)
    problematic_pairs = []
    for i, f1 in enumerate(available):
        for j, f2 in enumerate(available):
            if i < j and abs(corr_matrix[i, j]) > 0.8:
                problematic_pairs.append((f1, f2, corr_matrix[i, j]))
    
    # Build recommendations
    recommendations = []
    high_vif = vif_df[vif_df['VIF'] >= 10]['Feature'].tolist()
    
    if high_vif:
        recommendations.append(
            f"Features with VIF ≥ 10: {high_vif}. Consider removing or combining these."
        )
    
    if problematic_pairs:
        for f1, f2, r in problematic_pairs:
            recommendations.append(
                f"High correlation ({r:.2f}) between {f1} and {f2}. "
                f"Consider keeping only one."
            )
    
    if feature_mode == 'hybrid':
        # Check for specific known issues (only for standard hybrid, not smart_hybrid)
        raw_in_hybrid = [f for f in RAW_FACTORS if f in available]
        derived_in_hybrid = [f for f in available if f not in RAW_FACTORS]
        if raw_in_hybrid and derived_in_hybrid:
            recommendations.append(
                "Hybrid mode includes both raw factors and derived features. "
                "Derived features (Cu_V_ratio, Metal_Conc) are algebraically "
                "related to raw factors, which can inflate VIF. "
                "Consider using 'smart_hybrid' mode to auto-remove collinear features."
            )
    
    if feature_mode == 'smart_hybrid':
        # Note which features were selected
        raw_kept = [f for f in RAW_FACTORS if f in available]
        derived_used = [f for f in available if f not in RAW_FACTORS]
        recommendations.append(
            f"Smart hybrid mode: Kept {len(raw_kept)} raw features {raw_kept}, "
            f"using {len(derived_used)} chemical features {derived_used} "
            f"to replace collinear raw parameters."
        )
    
    if not recommendations:
        recommendations.append("✓ No major collinearity issues detected.")
    
    results = {
        'feature_mode': feature_mode,
        'features': available,
        'n_features': len(available),
        'vif': vif_df,
        'correlation_matrix': corr_df,
        'problematic_pairs': problematic_pairs,
        'recommendations': recommendations,
        'has_issues': len(high_vif) > 0 or len(problematic_pairs) > 0,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COLLINEARITY DIAGNOSTICS: {feature_mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"\nFeatures ({len(available)}): {available}")
        print(f"\nVariance Inflation Factors:")
        print(vif_df.to_string(index=False))
        
        if problematic_pairs:
            print(f"\nHighly Correlated Pairs (|r| > 0.8):")
            for f1, f2, r in problematic_pairs:
                print(f"  {f1} ↔ {f2}: r = {r:.3f}")
        
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    return results


# =============================================================================
# CLASSIFIER CALIBRATION METRICS
# =============================================================================

def classifier_calibration_metrics(
    clf,
    X: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics for a probabilistic classifier.
    
    Well-calibrated classifiers have predicted probabilities that match
    observed frequencies. E.g., among samples with P(y=1) = 0.8, about
    80% should actually have y=1.
    
    Parameters
    ----------
    clf : fitted classifier with predict_proba method
    X : array (n_samples, n_features)
        Test features (should be held-out or CV predictions)
    y_true : array (n_samples,)
        True binary labels (0 or 1)
    n_bins : int
        Number of bins for calibration curve
        
    Returns
    -------
    dict with:
        'brier_score': float, mean squared error of probabilities (lower is better)
        'log_loss': float, negative log-likelihood (lower is better)
        'calibration_bins': dict with bin centers and observed frequencies
        'ece': float, Expected Calibration Error
        'interpretation': str, human-readable assessment
    """
    from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss
    
    # Get predicted probabilities
    y_prob = clf.predict_proba(X)[:, 1]
    y_true = np.asarray(y_true)
    
    # Brier score (mean squared error of probabilities)
    brier = brier_score_loss(y_true, y_prob)
    
    # Log loss
    try:
        logloss = sk_log_loss(y_true, y_prob)
    except:
        logloss = np.nan
    
    # Calibration curve (reliability diagram data)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_freq = []
    predicted_freq = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        
        if mask.sum() > 0:
            observed_freq.append(y_true[mask].mean())
            predicted_freq.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            observed_freq.append(np.nan)
            predicted_freq.append(np.nan)
            bin_counts.append(0)
    
    # Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(y_true)
    for obs, pred, count in zip(observed_freq, predicted_freq, bin_counts):
        if count > 0 and not np.isnan(obs):
            ece += (count / total_samples) * abs(obs - pred)
    
    # Interpretation
    if brier < 0.1:
        interpretation = "✓ Excellent calibration (Brier < 0.1)"
    elif brier < 0.2:
        interpretation = "✓ Good calibration (Brier < 0.2)"
    elif brier < 0.3:
        interpretation = "⚠ Moderate calibration (Brier < 0.3)"
    else:
        interpretation = "✗ Poor calibration (Brier ≥ 0.3)"
    
    if ece > 0.15:
        interpretation += f"; High ECE ({ece:.3f}) suggests miscalibration"
    
    return {
        'brier_score': float(brier),
        'log_loss': float(logloss),
        'ece': float(ece),
        'calibration_bins': {
            'bin_centers': bin_centers.tolist(),
            'observed_frequency': observed_freq,
            'predicted_frequency': predicted_freq,
            'bin_counts': bin_counts,
        },
        'class_balance': float(y_true.mean()),
        'interpretation': interpretation,
    }


def evaluate_all_classifiers(
    optimizer,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate calibration of all classifiers in a Cu3VS4Optimizer.
    
    Parameters
    ----------
    optimizer : Cu3VS4Optimizer
        Fitted optimizer instance
    verbose : bool
        Print detailed results
        
    Returns
    -------
    dict mapping classifier name to calibration metrics
    """
    results = {}
    
    classifiers = [
        ('HasProduct', optimizer.clf_product),
        ('PhasePure', optimizer.clf_pure),
        ('IsCubic', optimizer.clf_cubic),
    ]
    
    for name, clf in classifiers:
        if clf is None:
            if verbose:
                print(f"\n{name}: Skipped (not fitted)")
            continue
        
        # Get true labels
        y_true = optimizer.df_all[name].values
        
        # Note: Using training data for calibration is not ideal, but
        # we're limited by sample size. In production, use LOO predictions.
        metrics = classifier_calibration_metrics(clf, optimizer.X_all_scaled, y_true)
        results[name] = metrics
        
        if verbose:
            print(f"\n{name} Classifier:")
            print(f"  Class balance: {metrics['class_balance']*100:.1f}% positive")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  {metrics['interpretation']}")
    
    if verbose and results:
        print(f"\n{'='*50}")
        print("CALIBRATION SUMMARY")
        print(f"{'='*50}")
        print("Well-calibrated: Brier < 0.2, ECE < 0.1")
        print("Note: Metrics computed on training data. Cross-validation")
        print("would give more realistic estimates for held-out data.")
    
    return results


# =============================================================================
# Cu₃VS₄ BAYESIAN OPTIMIZER
# =============================================================================

class Cu3VS4Optimizer:
    """
    Chemically-informed Bayesian Optimization for Cu₃VS₄ synthesis.
    
    Supports multiple feature representations:
    - 'raw': Original synthesis parameters
    - 'chemical': Fully transformed chemical features
    - 'hybrid': Raw parameters + key chemical insights
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_mode: str = 'hybrid',
        validate: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        smart_hybrid_threshold: float = 0.8
    ):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        df : DataFrame
            Experimental data
        feature_mode : str
            'raw', 'chemical', 'hybrid', or 'smart_hybrid' (default: 'hybrid')
        validate : bool
            Run LOO-CV validation
        objective_weights : dict, optional
            Custom weights for GSD and Squareness objectives
        smart_hybrid_threshold : float
            Correlation threshold for smart_hybrid mode (default: 0.8)
        """
        self.feature_mode = feature_mode
        
        # Store data and add chemical features if needed
        self.df_all = df.copy()
        if feature_mode in ['chemical', 'hybrid', 'smart_hybrid']:
            if 'Cu_V_ratio' not in self.df_all.columns:
                self.df_all = add_chemical_features(self.df_all)
        
        self.df_success = self.df_all[self.df_all["HasProduct"] == 1].copy()
        
        # Select features based on mode
        if feature_mode == 'raw':
            self.features = RAW_FACTORS
        elif feature_mode == 'chemical':
            self.features = CHEM_FEATURES
        elif feature_mode == 'hybrid':
            self.features = HYBRID_FEATURES
        elif feature_mode == 'smart_hybrid':
            print(f"\nSelecting smart hybrid features (threshold |r| > {smart_hybrid_threshold})...")
            self.features = select_smart_hybrid_features(
                self.df_all,
                correlation_threshold=smart_hybrid_threshold,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}. Use 'raw', 'chemical', 'hybrid', or 'smart_hybrid'.")
        
        if len(self.df_success) < 5:
            raise ValueError(f"Need ≥5 successful experiments")
        
        # Extract features
        self.X_all = self.df_all[self.features].values.astype(float)
        self.X_success = self.df_success[self.features].values.astype(float)
        
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_all)
        self.X_all_scaled = self.scaler.transform(self.X_all)
        self.X_success_scaled = self.scaler.transform(self.X_success)
        
        # Compute bounds
        self.bounds = compute_feature_bounds(self.features, self.df_all)
        
        # Build models
        self._build_models()
        
        # Validate
        self.metrics = {}
        if validate:
            self._validate()
        
        # Set objective weights
        if objective_weights is None:
            self.objective_weights = self._calculate_weights()
        else:
            self.objective_weights = objective_weights
    
    def _gp_factory(self):
        return make_gp_regressor(len(self.features))
    
    def _build_models(self):
        """Build and fit all models."""
        n = len(self.features)
        print(f"Building models with {n} features ({self.feature_mode} mode)...")
        
        # Regression
        self.gp_size = make_gp_regressor(n)
        self.gp_gsd = make_gp_regressor(n)
        self.gp_sq = make_gp_regressor(n)
        
        self.gp_size.fit(self.X_success_scaled, self.df_success["Size"].values)
        self.gp_gsd.fit(self.X_success_scaled, self.df_success["GSD"].values)
        self.gp_sq.fit(self.X_success_scaled, self.df_success["Squareness"].values)
        print(f"  Regression models fitted (n={len(self.df_success)})")
        
        # Classification
        self.clf_product = self._fit_clf("HasProduct")
        self.clf_pure = self._fit_clf("PhasePure")
        self.clf_cubic = self._fit_clf("IsCubic")
    
    def _fit_clf(self, col: str):
        y = self.df_all[col].values
        if len(np.unique(y)) < 2:
            print(f"  {col}: Single class, skipping")
            return None
        clf = make_gp_classifier(len(self.features))
        clf.fit(self.X_all_scaled, y.astype(int))
        print(f"  {col} classifier fitted")
        return clf
    
    def _validate(self):
        """Run LOO-CV."""
        print("\nRunning LOO cross-validation...")
        
        for name in ["Size", "GSD", "Squareness"]:
            y = self.df_success[name].values
            cv = loo_cv(self.X_success_scaled, y, self._gp_factory, return_predictions=True)
            self.metrics[name] = cv
            print(f"  {name}: R²={cv['r2']:.3f}, RMSE={cv['rmse']:.3f}")
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate objective weights from data and model performance."""
        return calculate_objective_weights(
            df_success=self.df_success,
            model_metrics=self.metrics,
            method='comprehensive'
        )
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict all properties and feasibility."""
        X_scaled = self.scaler.transform(X)
        
        size_mu, size_std = self.gp_size.predict(X_scaled, return_std=True)
        gsd_mu, gsd_std = self.gp_gsd.predict(X_scaled, return_std=True)
        sq_mu, sq_std = self.gp_sq.predict(X_scaled, return_std=True)
        
        n = len(X)
        p_product = np.ones(n) if self.clf_product is None else self.clf_product.predict_proba(X_scaled)[:, 1]
        p_pure = np.ones(n) if self.clf_pure is None else self.clf_pure.predict_proba(X_scaled)[:, 1]
        p_cubic = np.ones(n) if self.clf_cubic is None else self.clf_cubic.predict_proba(X_scaled)[:, 1]
        
        return {
            'size_mu': size_mu, 'size_std': size_std,
            'gsd_mu': gsd_mu, 'gsd_std': gsd_std,
            'sq_mu': sq_mu, 'sq_std': sq_std,
            'p_product': p_product, 'p_pure': p_pure, 'p_cubic': p_cubic,
            'p_feasible': p_product * p_pure * p_cubic
        }
    
    def acquisition(
        self,
        X: np.ndarray,
        target_size: float,
        size_tol: float
    ) -> Dict[str, np.ndarray]:
        """Compute acquisition function."""
        preds = self.predict(X)
        
        gsd_best = self.df_success["GSD"].min()
        sq_best = self.df_success["Squareness"].max()
        
        ei_gsd = expected_improvement(preds['gsd_mu'], preds['gsd_std'], gsd_best, minimize=True)
        ei_sq = expected_improvement(preds['sq_mu'], preds['sq_std'], sq_best, minimize=False)
        
        # Normalize
        ei_gsd_n = (ei_gsd - ei_gsd.min()) / (np.ptp(ei_gsd) + 1e-10)
        ei_sq_n = (ei_sq - ei_sq.min()) / (np.ptp(ei_sq) + 1e-10)
        
        # Use objective weights
        w_gsd = self.objective_weights.get("GSD", 1.0)
        w_sq = self.objective_weights.get("Squareness", 1.0)
        acq_obj = (w_gsd * ei_gsd_n + w_sq * ei_sq_n) / (w_gsd + w_sq)
        
        p_size = prob_in_interval(preds['size_mu'], preds['size_std'], target_size, size_tol)
        
        total = acq_obj * p_size * preds['p_feasible']
        
        return {'total': total, 'acq_obj': acq_obj, 'p_size': p_size, **preds}
    
    def recommend(
        self,
        target_size: float,
        size_tol: float = 2.5,
        p_size_min: float = 0.2,
        p_feas_min: float = 0.3,
        n_candidates: int = 20000,
        n_return: int = 2,
        min_distance: float = 0.3,
        seed: Optional[int] = None,
        warn_extrapolation: bool = True,
        extrapolation_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Recommend synthesis conditions for target size.
        
        Parameters
        ----------
        target_size : float
            Target particle size (nm)
        size_tol : float
            Acceptable tolerance around target
        p_size_min : float
            Minimum probability of hitting target size
        p_feas_min : float
            Minimum feasibility probability
        n_candidates : int
            Number of LHS candidates to generate
        n_return : int
            Number of recommendations to return
        min_distance : float
            Minimum distance between recommendations (in scaled space)
        seed : int, optional
            Random seed for reproducibility
        warn_extrapolation : bool
            If True, check if recommendations are far from training data
        extrapolation_threshold : float
            Distance threshold for extrapolation warning (in std units)
            
        Returns
        -------
        DataFrame with recommended conditions and predictions
        """
        # Sample
        X = latin_hypercube_sample(n_candidates, self.bounds, self.features, seed)
        
        # Acquisition
        acq = self.acquisition(X, target_size, size_tol)
        
        # Filter
        mask = (acq['p_size'] >= p_size_min) & (acq['p_feasible'] >= p_feas_min)
        
        if mask.sum() == 0:
            print(f"[WARNING] No candidates meet constraints. Relaxing...")
            mask = np.ones(len(X), dtype=bool)
        
        X_feas = X[mask]
        acq_feas = acq['total'][mask]
        
        # Sort and select diverse
        order = np.argsort(acq_feas)[::-1]
        X_scaled = self.scaler.transform(X_feas)
        
        selected = []
        selected_scaled = []
        
        for idx in order:
            x = X_scaled[idx]
            if selected_scaled and np.min(cdist([x], selected_scaled)) < min_distance:
                continue
            selected.append(idx)
            selected_scaled.append(x)
            if len(selected) >= n_return:
                break
        
        # Build results - always convert to raw parameters for lab use
        rows = []
        for rank, idx in enumerate(selected, 1):
            row = {'Rank': rank}
            
            # Get feature values
            feat_dict = {feat: X_feas[idx, i] for i, feat in enumerate(self.features)}
            
            # Convert to raw parameters based on feature mode
            if self.feature_mode == 'raw':
                raw_params = feat_dict
            elif self.feature_mode == 'chemical':
                raw_params = chemical_to_raw_features(
                    Temp=feat_dict['Temp'],
                    Cu_V_ratio=feat_dict['Cu_V_ratio'],
                    S_Metal_ratio=feat_dict['S_Metal_ratio'],
                    Ligand_Metal_ratio=feat_dict['Ligand_Metal_ratio'],
                    Metal_Conc=feat_dict['Metal_Conc'],
                    log_Time=feat_dict['log_Time']
                )
            elif self.feature_mode in ['hybrid', 'smart_hybrid']:
                raw_params = {}
                # First, copy any raw features that are present
                for raw_feat in RAW_FACTORS:
                    if raw_feat in feat_dict:
                        raw_params[raw_feat] = feat_dict[raw_feat]
                
                # Reconstruct missing raw features from chemical features
                # 1. VOacac from Cu_V_ratio
                if 'Cu_V_ratio' in feat_dict and 'VOacac' not in raw_params:
                    raw_params['VOacac'] = CUI_MMOL / feat_dict['Cu_V_ratio']
                
                # 2. Time from log_Time
                if 'log_Time' in feat_dict and 'Time' not in raw_params:
                    raw_params['Time'] = 10 ** feat_dict['log_Time']
                
                # 3. Calculate total_metal for DDT/OAm reconstruction
                # We need this for S_Metal_ratio and Ligand_Metal_ratio conversions
                if 'Metal_Conc' in feat_dict:
                    # Metal_Conc = (total_metal / TOTAL_VOLUME_ML) * 1000 in mM
                    total_metal = (feat_dict['Metal_Conc'] / 1000.0) * TOTAL_VOLUME_ML
                elif 'VOacac' in raw_params or 'Cu_V_ratio' in feat_dict:
                    # Calculate from stoichiometry: total_metal = CuI + VOacac
                    VOacac_val = raw_params.get('VOacac', CUI_MMOL / feat_dict.get('Cu_V_ratio', 1.0))
                    total_metal = CUI_MMOL + VOacac_val
                else:
                    total_metal = None
                
                # 4. DDT from S_Metal_ratio
                if 'S_Metal_ratio' in feat_dict and 'DDT' not in raw_params and total_metal is not None:
                    DDT_mmol = feat_dict['S_Metal_ratio'] * total_metal
                    raw_params['DDT'] = DDT_mmol / DDT_MMOL_PER_ML
                
                # 5. OAm from Ligand_Metal_ratio
                if 'Ligand_Metal_ratio' in feat_dict and 'OAm' not in raw_params and total_metal is not None:
                    OAm_mmol = feat_dict['Ligand_Metal_ratio'] * total_metal
                    raw_params['OAm'] = OAm_mmol / OAM_MMOL_PER_ML
            else:
                raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
            
            # Add raw parameters to row
            for key in RAW_FACTORS:
                if key not in raw_params:
                    raise RuntimeError(
                        f"Could not reconstruct raw parameter '{key}' from features in {self.feature_mode} mode. "
                        f"Available features: {list(feat_dict.keys())}"
                    )
                row[key] = round(raw_params[key], 3)
            
            # Predictions
            row.update({
                'Pred_Size': round(acq['size_mu'][mask][idx], 2),
                'Pred_Size_Std': round(acq['size_std'][mask][idx], 2),
                'Pred_GSD': round(acq['gsd_mu'][mask][idx], 3),
                'Pred_Squareness': round(acq['sq_mu'][mask][idx], 3),
                'P_Size': round(acq['p_size'][mask][idx], 3),
                'P_Feasible': round(acq['p_feasible'][mask][idx], 3),
                'Acquisition': round(acq_feas[idx], 4),
            })
            rows.append(row)
        
        result_df = pd.DataFrame(rows)
        
        # Check for extrapolation
        if warn_extrapolation and len(selected) > 0:
            X_selected = X_feas[selected]
            extrap_check = detect_extrapolation(
                X_selected, 
                self.X_all,
                self.scaler,
                threshold=extrapolation_threshold
            )
            
            # Add extrapolation info to dataframe
            result_df['Extrapolation_Distance'] = extrap_check['distances']
            result_df['Is_Extrapolating'] = extrap_check['is_extrapolation']
            
            # Print warnings if any
            for warning in extrap_check['warnings']:
                print(warning)
        
        return result_df
    
    def get_lengthscales(self) -> pd.DataFrame:
        """Extract learned lengthscales (inverse = importance)."""
        results = []
        for name, gp in [("Size", self.gp_size), ("GSD", self.gp_gsd), ("Squareness", self.gp_sq)]:
            try:
                ls = gp.kernel_.k1.k2.length_scale
                for feat, l in zip(self.features, ls):
                    results.append({'Model': name, 'Feature': feat, 'Lengthscale': l, 'Importance': 1/l})
            except:
                pass
        return pd.DataFrame(results)
    
    def get_collinearity_diagnostics(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Get collinearity diagnostics for the current feature set.
        
        Returns VIF values, correlation matrix, and recommendations.
        High VIF (>10) indicates problematic multicollinearity that may
        affect lengthscale interpretation and model stability.
        
        Parameters
        ----------
        verbose : bool
            Print detailed results
            
        Returns
        -------
        dict with VIF, correlations, and recommendations
        """
        return diagnose_collinearity(
            self.df_all, 
            self.feature_mode, 
            verbose, 
            feature_list=self.features
        )
    
    def get_classifier_calibration(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate calibration of feasibility classifiers.
        
        Well-calibrated classifiers produce probabilities that match
        observed frequencies. E.g., if P(HasProduct)=0.8, then 80%
        of such samples should actually have product.
        
        Parameters
        ----------
        verbose : bool
            Print detailed results
            
        Returns
        -------
        dict mapping classifier name to calibration metrics
        """
        return evaluate_all_classifiers(self, verbose)
    
    def full_diagnostics(self) -> Dict[str, Any]:
        """
        Run all diagnostic checks and return comprehensive report.
        
        Includes:
        - LOO-CV metrics for regressors
        - Collinearity diagnostics (VIF)
        - Classifier calibration (Brier score, ECE)
        - Feature importance from lengthscales
        
        Returns
        -------
        dict with all diagnostic results
        """
        print(f"\n{'='*70}")
        print("COMPREHENSIVE MODEL DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"\nFeature mode: {self.feature_mode}")
        print(f"Features ({len(self.features)}): {self.features}")
        print(f"Training samples: {len(self.df_all)} total, {len(self.df_success)} successful")
        
        # Samples per feature ratio
        ratio = len(self.df_success) / len(self.features)
        if ratio < 5:
            print(f"⚠️ Warning: Only {ratio:.1f} samples per feature (recommend ≥10)")
        else:
            print(f"✓ Samples per feature ratio: {ratio:.1f}")
        
        diagnostics = {
            'feature_mode': self.feature_mode,
            'n_features': len(self.features),
            'n_samples': len(self.df_all),
            'n_successful': len(self.df_success),
            'samples_per_feature': ratio,
        }
        
        # LOO-CV metrics
        print(f"\n--- LOO-CV Regression Metrics ---")
        if self.metrics:
            diagnostics['loo_cv'] = self.metrics
            for prop, m in self.metrics.items():
                print(f"{prop}: R²={m['r2']:.3f}, RMSE={m['rmse']:.3f}, "
                      f"Cal_68={m['cal_68']:.2f} (target: 0.68), "
                      f"Cal_95={m['cal_95']:.2f} (target: 0.95)")
        else:
            print("No LOO-CV metrics available. Run with validate=True.")
        
        # Collinearity
        print(f"\n--- Collinearity Diagnostics ---")
        diagnostics['collinearity'] = self.get_collinearity_diagnostics(verbose=False)
        vif_df = diagnostics['collinearity']['vif']
        high_vif = vif_df[vif_df['VIF'] >= 10]
        if len(high_vif) > 0:
            print(f"⚠️ High VIF features: {high_vif['Feature'].tolist()}")
        else:
            print("✓ No severe collinearity detected")
        print(vif_df.to_string(index=False))
        
        # Classifier calibration
        print(f"\n--- Classifier Calibration ---")
        diagnostics['classifier_calibration'] = self.get_classifier_calibration(verbose=False)
        for name, cal in diagnostics['classifier_calibration'].items():
            print(f"{name}: Brier={cal['brier_score']:.4f}, ECE={cal['ece']:.4f}")
            print(f"  {cal['interpretation']}")
        
        print(f"\n{'='*70}")
        
        return diagnostics


print("✓ Cu3VS4 Optimizer module loaded")
print(f"  Features available: raw, chemical, hybrid")
print(f"  Enhanced features: {CHEM_FEATURES_ENHANCED if CHEM_FEATURES_ENHANCED else 'None'}")
print(f"  Diagnostic functions: compare_feature_modes, detect_extrapolation, diagnose_collinearity")
print(f"  Calibration: classifier_calibration_metrics, evaluate_all_classifiers")
