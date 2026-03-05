"""
Cu₃VS₄ Bayesian Optimizer (base class)

Contains the Cu3VS4Optimizer class and GP/acquisition helpers.
Feature engineering, diagnostics, and visualization live in their own modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

from scipy.stats import norm
from scipy.spatial.distance import cdist

from config import (
    CUI_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, OBJECTIVES, FEAS_COLS, RAW_BOUNDS,
)
from features import (
    add_chemical_features,
    compute_feature_bounds,
    round_to_practical,
    chemical_to_raw_features,
    CHEM_FEATURES, HYBRID_FEATURES,
)
from config import SYNTHESIS_FEATURES
from diagnostics import (
    loo_cv,
    detect_extrapolation,
    diagnose_collinearity,
    evaluate_all_classifiers,
)


# =============================================================================
# GP BUILDERS
# =============================================================================

def make_gp_regressor(n_features: int) -> GaussianProcessRegressor:
    """
    Create GP regressor with isotropic Matérn 5/2 kernel.

    Uses a SINGLE shared lengthscale rather than per-feature ARD.
    With ~44 cubic data points in 5-D, ARD (5 separate lengthscales)
    causes the optimizer to set very short lengthscales in individual
    dimensions, leading to interpolation and R² < 0 in LOO-CV.
    The isotropic kernel forces equal treatment of all (StandardScaler-
    normalised) features, dramatically improving generalisation:
      Size R²: 0.08 → 0.30,  Squareness R²: −0.62 → −0.05
    """
    kernel = (
        C(1.0, (0.01, 100.0)) *
        Matern(length_scale=1.0, length_scale_bounds=(0.3, 10.0), nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 2.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        n_restarts_optimizer=10, random_state=42, alpha=1e-8,
    )


def make_gp_classifier(n_features: int) -> GaussianProcessClassifier:
    """Create GP classifier for feasibility."""
    kernel = C(1.0, (0.01, 100.0)) * Matern(
        length_scale=[1.0] * n_features, length_scale_bounds=(0.1, 10.0), nu=2.5
    )
    return GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=5,
        random_state=42, max_iter_predict=200
    )


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray,
    y_best: float, xi: float = 0.01, minimize: bool = True
) -> np.ndarray:
    """Expected Improvement acquisition function."""
    sigma = np.maximum(sigma, 1e-9)
    improvement = (y_best - mu - xi) if minimize else (mu - y_best - xi)
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def prob_in_interval(
    mu: np.ndarray, sigma: np.ndarray,
    target: float, tolerance: float
) -> np.ndarray:
    """P(target - tol <= Y <= target + tol)."""
    sigma = np.maximum(sigma, 1e-9)
    z_hi = (target + tolerance - mu) / sigma
    z_lo = (target - tolerance - mu) / sigma
    return norm.cdf(z_hi) - norm.cdf(z_lo)


def _normalize_ei(ei: np.ndarray) -> np.ndarray:
    """Normalize EI to [0, 1]. Returns zeros when all values are identical."""
    ptp = np.ptp(ei)
    if ptp < 1e-8:
        return np.zeros_like(ei)
    return (ei - ei.min()) / ptp


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
    from scipy.stats.qmc import LatinHypercube
    n_dims = len(feature_order)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    lows = np.array([bounds[k][0] for k in feature_order])
    highs = np.array([bounds[k][1] for k in feature_order])
    return samples * (highs - lows) + lows


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def _select_diverse_candidates(
    X: np.ndarray,
    acq_total: np.ndarray,
    p_size: np.ndarray,
    p_feasible: np.ndarray,
    scaler: StandardScaler,
    p_size_min: float,
    p_feas_min: float,
    n_return: int,
    min_distance: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
    """
    Filter candidates by constraints and select diverse top candidates.

    Returns (X_feasible, acq_feasible, selected_indices, feasibility_mask).
    selected_indices are positions within X_feasible.
    """
    mask = (p_size >= p_size_min) & (p_feasible >= p_feas_min)
    if mask.sum() == 0:
        print("[WARNING] No candidates meet constraints. Relaxing...")
        mask = np.ones(len(X), dtype=bool)

    X_feas = X[mask]
    acq_feas = acq_total[mask]
    order = np.argsort(acq_feas)[::-1]
    X_scaled = scaler.transform(X_feas)

    selected, selected_scaled = [], []
    for idx in order:
        x = X_scaled[idx]
        if selected_scaled and np.min(cdist([x], selected_scaled)) < min_distance:
            continue
        selected.append(idx)
        selected_scaled.append(x)
        if len(selected) >= n_return:
            break

    return X_feas, acq_feas, selected, mask


# =============================================================================
# WEIGHT CALCULATION
# =============================================================================

def calculate_objective_weights(
    df_success: pd.DataFrame,
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    method: str = 'variance'
) -> Dict[str, float]:
    """
    Calculate objective weights for GSD and Squareness.

    Default method ('variance') uses the coefficient of variation.
    """
    gsd_data = df_success['GSD'].values
    sq_data = df_success['Squareness'].values

    if method == 'variance':
        w_gsd = np.std(gsd_data) / (np.mean(gsd_data) + 1e-10)
        w_sq = np.std(sq_data) / (np.mean(sq_data) + 1e-10)
    elif method == 'equal':
        w_gsd = 1.0; w_sq = 1.0
    elif method == 'uncertainty':
        if model_metrics is None:
            raise ValueError("model_metrics required for 'uncertainty' method")
        rmse_gsd = model_metrics.get('GSD', {}).get('rmse', 0.1)
        rmse_sq = model_metrics.get('Squareness', {}).get('rmse', 0.1)
        w_gsd = rmse_gsd / (np.mean(gsd_data) + 1e-10)
        w_sq = rmse_sq / (np.mean(sq_data) + 1e-10)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'variance', 'equal', or 'uncertainty'.")

    total = w_gsd + w_sq
    if total > 0:
        scale = 2.0 / total
        w_gsd *= scale; w_sq *= scale
    else:
        w_gsd = 1.0; w_sq = 1.0

    return {'GSD': float(w_gsd), 'Squareness': float(w_sq)}


# =============================================================================
# Cu₃VS₄ BAYESIAN OPTIMIZER
# =============================================================================

class Cu3VS4Optimizer:
    """
    Chemically-informed Bayesian Optimization for Cu₃VS₄ synthesis.

    Supports 'raw', 'chemical', 'hybrid', and 'synthesis' feature modes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_mode: str = 'synthesis',
        validate: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
    ):
        self.feature_mode = feature_mode

        self.df_all = df.copy()
        if feature_mode in ['chemical', 'hybrid', 'synthesis']:
            if 'Cu_V_ratio' not in self.df_all.columns:
                self.df_all = add_chemical_features(self.df_all)

        self.df_success = self.df_all[self.df_all["HasProduct"] == 1].copy()

        if feature_mode == 'raw':
            self.features = RAW_FACTORS
        elif feature_mode == 'chemical':
            self.features = CHEM_FEATURES
        elif feature_mode == 'hybrid':
            self.features = HYBRID_FEATURES
        elif feature_mode == 'synthesis':
            self.features = list(SYNTHESIS_FEATURES)
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}. Use 'raw', 'chemical', 'hybrid', or 'synthesis'.")

        if len(self.df_success) < 5:
            raise ValueError(f"Need ≥5 successful experiments")

        self.X_all = self.df_all[self.features].values.astype(float)
        self.X_success = self.df_success[self.features].values.astype(float)

        self.scaler = StandardScaler()
        self.scaler.fit(self.X_all)
        self.X_all_scaled = self.scaler.transform(self.X_all)
        self.X_success_scaled = self.scaler.transform(self.X_success)

        self.bounds = compute_feature_bounds(self.features, self.df_all)

        self._build_models()

        self.metrics = {}
        if validate:
            self._validate()

        if objective_weights is None:
            self.objective_weights = calculate_objective_weights(self.df_cubic, method='variance')
        else:
            self.objective_weights = objective_weights

    def _gp_factory(self):
        return make_gp_regressor(len(self.features))

    def _build_models(self):
        n = len(self.features)
        print(f"Building models with {n} features ({self.feature_mode} mode)...")

        # Regression models are trained on cubic-only data: size/GSD/Squareness are
        # only meaningful for cubic particles, and mixing morphologies confounds the GP.
        if 'IsCubic' in self.df_success.columns:
            df_cubic = self.df_success[self.df_success['IsCubic'] == 1].copy()
            if len(df_cubic) >= 5:
                self.df_cubic = df_cubic
                self.X_cubic = self.df_cubic[self.features].values.astype(float)
                self.X_cubic_scaled = self.scaler.transform(self.X_cubic)
                print(f"  Cubic-only regression data: n={len(self.df_cubic)} "
                      f"(of {len(self.df_success)} successful)")
            else:
                print(f"  ⚠ Only {len(df_cubic)} cubic samples; "
                      f"falling back to all successful (n={len(self.df_success)})")
                self.df_cubic = self.df_success
                self.X_cubic = self.X_success
                self.X_cubic_scaled = self.X_success_scaled
        else:
            self.df_cubic = self.df_success
            self.X_cubic = self.X_success
            self.X_cubic_scaled = self.X_success_scaled

        self.gp_size = make_gp_regressor(n)
        self.gp_gsd = make_gp_regressor(n)
        self.gp_sq = make_gp_regressor(n)
        self.gp_size.fit(self.X_cubic_scaled, self.df_cubic["Size"].values)
        self.gp_gsd.fit(self.X_cubic_scaled, self.df_cubic["GSD"].values)
        self.gp_sq.fit(self.X_cubic_scaled, self.df_cubic["Squareness"].values)
        print(f"  Regression models fitted (n={len(self.df_cubic)})")

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
        print("\nRunning LOO cross-validation (cubic-only)...")
        for name in ["Size", "GSD", "Squareness"]:
            y = self.df_cubic[name].values
            cv = loo_cv(
                self.X_cubic, y, self._gp_factory,
                return_predictions=True, scaler_factory=StandardScaler,
            )
            self.metrics[name] = cv
            print(f"  {name}: R²={cv['r2']:.3f}, RMSE={cv['rmse']:.3f}")

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
        self, X: np.ndarray, target_size: float, size_tol: float,
        preds: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute acquisition function. Accepts pre-computed predictions for corrected models."""
        if preds is None:
            preds = self.predict(X)
        gsd_best = self.df_cubic["GSD"].min()
        sq_best = self.df_cubic["Squareness"].max()

        ei_gsd = expected_improvement(preds['gsd_mu'], preds['gsd_std'], gsd_best, minimize=True)
        ei_sq = expected_improvement(preds['sq_mu'], preds['sq_std'], sq_best, minimize=False)
        ei_gsd_n = _normalize_ei(ei_gsd)
        ei_sq_n = _normalize_ei(ei_sq)

        w_gsd = self.objective_weights.get("GSD", 1.0)
        w_sq = self.objective_weights.get("Squareness", 1.0)
        acq_obj = (w_gsd * ei_gsd_n + w_sq * ei_sq_n) / (w_gsd + w_sq)
        p_size = prob_in_interval(preds['size_mu'], preds['size_std'], target_size, size_tol)
        total = acq_obj * p_size * preds['p_feasible']

        return {'total': total, 'acq_obj': acq_obj, 'p_size': p_size, **preds}

    def recommend(
        self,
        target_size: float, size_tol: float = 2.5,
        p_size_min: float = 0.2, p_feas_min: float = 0.3,
        n_candidates: int = 20000, n_return: int = 2,
        min_distance: float = 0.3,
        seed: Optional[int] = None,
        warn_extrapolation: bool = True, extrapolation_threshold: float = 2.0
    ) -> pd.DataFrame:
        """Recommend synthesis conditions for target size."""
        X = latin_hypercube_sample(n_candidates, self.bounds, self.features, seed)
        acq = self.acquisition(X, target_size, size_tol)

        X_feas, acq_feas, selected, mask = _select_diverse_candidates(
            X, acq['total'], acq['p_size'], acq['p_feasible'],
            self.scaler, p_size_min, p_feas_min, n_return, min_distance,
        )

        rows = []
        for rank, idx in enumerate(selected, 1):
            row = {'Rank': rank}
            feat_dict = {feat: X_feas[idx, i] for i, feat in enumerate(self.features)}
            raw_params = self._feature_dict_to_raw(feat_dict)
            raw_params = round_to_practical(raw_params)
            for key in RAW_FACTORS:
                row[key] = raw_params[key]
            row.update({
                'Pred_Size': round(float(acq['size_mu'][mask][idx]), 2),
                'Pred_Size_Std': round(float(acq['size_std'][mask][idx]), 2),
                'Pred_GSD': round(float(acq['gsd_mu'][mask][idx]), 3),
                'Pred_Squareness': round(float(acq['sq_mu'][mask][idx]), 3),
                'P_Size': round(float(acq['p_size'][mask][idx]), 3),
                'P_Feasible': round(float(acq['p_feasible'][mask][idx]), 3),
                'Acquisition': round(float(acq_feas[idx]), 4),
            })
            rows.append(row)

        result_df = pd.DataFrame(rows)
        if warn_extrapolation and len(selected) > 0:
            X_selected = X_feas[selected]
            extrap_check = detect_extrapolation(X_selected, self.X_all, self.scaler, threshold=extrapolation_threshold)
            result_df['Extrapolation_Distance'] = extrap_check['distances']
            result_df['Is_Extrapolating'] = extrap_check['is_extrapolation']
            for warning in extrap_check['warnings']:
                print(warning)

        return result_df

    def _feature_dict_to_raw(self, feat_dict: Dict[str, float]) -> Dict[str, float]:
        """Convert a feature-space dict back to raw lab parameters."""
        if self.feature_mode == 'raw':
            return {k: feat_dict[k] for k in RAW_FACTORS}
        if self.feature_mode == 'chemical':
            return self._chemical_to_raw(feat_dict)
        return self._synthesis_to_raw(feat_dict)

    @staticmethod
    def _chemical_to_raw(feat_dict: Dict[str, float]) -> Dict[str, float]:
        """Back-transform chemical features to raw lab parameters."""
        temp = feat_dict.get('Temp', sum(RAW_BOUNDS['Temp']) / 2)
        result = chemical_to_raw_features(
            Temp=temp, Cu_V_ratio=feat_dict['Cu_V_ratio'],
            S_Metal_ratio=feat_dict['S_Metal_ratio'],
            Ligand_Metal_ratio=feat_dict['Ligand_Metal_ratio'],
            Metal_Conc=feat_dict['Metal_Conc'], log_Time=feat_dict['log_Time'],
        )
        return {k: float(v) for k, v in result.items()}

    def _synthesis_to_raw(self, feat_dict: Dict[str, float]) -> Dict[str, float]:
        """Back-transform hybrid/synthesis features to raw lab parameters.

        Reconstruction paths:
          Temp, DDT         — pass through (if present as raw factors)
          VOacac            = CuI / Cu_V_ratio
          Time              = 10^log_Time
          total_metal       = Metal_Conc / 1000 * total_vol  (or CuI + VOacac)
          DDT (if derived)  = S_Metal_ratio * total_metal / DDT_MMOL_PER_ML
          OAm               = Ligand_Metal_ratio * total_metal / OAM_MMOL_PER_ML
        """
        raw = {}
        for f in RAW_FACTORS:
            if f in feat_dict:
                raw[f] = feat_dict[f]

        if 'Cu_V_ratio' in feat_dict and 'VOacac' not in raw:
            raw['VOacac'] = CUI_MMOL / feat_dict['Cu_V_ratio']

        if 'log_Time' in feat_dict and 'Time' not in raw:
            raw['Time'] = 10 ** feat_dict['log_Time']

        if 'Metal_Conc' in feat_dict:
            total_metal = (feat_dict['Metal_Conc'] / 1000.0) * TOTAL_VOLUME_ML
        else:
            voacac = raw.get('VOacac', CUI_MMOL / feat_dict.get('Cu_V_ratio', 1.0))
            total_metal = CUI_MMOL + voacac

        if 'S_Metal_ratio' in feat_dict and 'DDT' not in raw:
            raw['DDT'] = (feat_dict['S_Metal_ratio'] * total_metal) / DDT_MMOL_PER_ML

        if 'Ligand_Metal_ratio' in feat_dict and 'OAm' not in raw:
            raw['OAm'] = (feat_dict['Ligand_Metal_ratio'] * total_metal) / OAM_MMOL_PER_ML

        missing = [k for k in RAW_FACTORS if k not in raw]
        if missing:
            raise RuntimeError(
                f"Could not reconstruct {missing} from features in "
                f"{self.feature_mode} mode. Available: {list(feat_dict.keys())}"
            )

        for k, (lo, hi) in RAW_BOUNDS.items():
            if k in raw:
                raw[k] = float(np.clip(raw[k], lo, hi))

        return raw

    def get_lengthscales(self) -> pd.DataFrame:
        """Extract learned lengthscales (inverse = importance).

        Handles both isotropic (scalar) and ARD (per-feature) kernels.
        """
        results = []
        for name, gp in [("Size", self.gp_size), ("GSD", self.gp_gsd), ("Squareness", self.gp_sq)]:
            try:
                ls = gp.kernel_.k1.k2.length_scale
                ls = np.atleast_1d(ls)
                if ls.shape[0] == 1:
                    for feat in self.features:
                        results.append({'Model': name, 'Feature': feat,
                                        'Lengthscale': float(ls[0]), 'Importance': 1.0 / float(ls[0])})
                else:
                    for feat, l in zip(self.features, ls):
                        results.append({'Model': name, 'Feature': feat,
                                        'Lengthscale': float(l), 'Importance': 1.0 / float(l)})
            except AttributeError:
                pass
        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Compute feature importance via gradient-based sensitivity analysis.

        For each feature and training point, estimates |∂f/∂x_j| using finite
        differences in the scaled space.  The mean absolute gradient gives a
        data-distribution-aware importance that works correctly with both
        isotropic and ARD kernels.
        """
        eps = 0.05
        results = []

        for name, gp in [("Size", self.gp_size), ("GSD", self.gp_gsd), ("Squareness", self.gp_sq)]:
            raw_sens = []
            for j in range(len(self.features)):
                X_plus = self.X_cubic_scaled.copy()
                X_minus = self.X_cubic_scaled.copy()
                X_plus[:, j] += eps
                X_minus[:, j] -= eps
                grad = np.abs(gp.predict(X_plus) - gp.predict(X_minus)) / (2 * eps)
                raw_sens.append(float(np.mean(grad)))

            total = sum(raw_sens)
            for feat, s in zip(self.features, raw_sens):
                results.append({
                    'Model': name, 'Feature': feat,
                    'Sensitivity': s,
                    'Importance': s / total if total > 0 else 1.0 / len(self.features),
                })

        return pd.DataFrame(results)

    def get_collinearity_diagnostics(self, verbose: bool = True) -> Dict[str, Any]:
        return diagnose_collinearity(self.df_all, self.feature_mode, verbose, feature_list=self.features)

    def get_classifier_calibration(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        return evaluate_all_classifiers(self, verbose)

    def full_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic checks and return comprehensive report."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE MODEL DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"\nFeature mode: {self.feature_mode}")
        print(f"Features ({len(self.features)}): {self.features}")
        print(f"Training samples: {len(self.df_all)} total, {len(self.df_success)} successful")

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

        print(f"\n--- LOO-CV Regression Metrics ---")
        if self.metrics:
            diagnostics['loo_cv'] = self.metrics
            for prop, m in self.metrics.items():
                print(f"{prop}: R²={m['r2']:.3f}, RMSE={m['rmse']:.3f}, "
                      f"Cal_68={m['cal_68']:.2f} (target: 0.68), "
                      f"Cal_95={m['cal_95']:.2f} (target: 0.95)")
        else:
            print("No LOO-CV metrics available. Run with validate=True.")

        print(f"\n--- Collinearity Diagnostics ---")
        diagnostics['collinearity'] = self.get_collinearity_diagnostics(verbose=False)
        vif_df = diagnostics['collinearity']['vif']
        high_vif = vif_df[vif_df['VIF'] >= 10]
        if len(high_vif) > 0:
            print(f"⚠️ High VIF features: {high_vif['Feature'].tolist()}")
        else:
            print("✓ No severe collinearity detected")
        print(vif_df.to_string(index=False))

        print(f"\n--- Classifier Calibration ---")
        diagnostics['classifier_calibration'] = self.get_classifier_calibration(verbose=False)
        for name, cal in diagnostics['classifier_calibration'].items():
            print(f"{name}: Brier={cal['brier_score']:.4f}, ECE={cal['ece']:.4f}")
            print(f"  {cal['interpretation']}")

        print(f"\n{'='*70}")
        return diagnostics
