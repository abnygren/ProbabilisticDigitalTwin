"""Base Cu3VS4 Bayesian optimizer.

Contains the ``Cu3VS4Optimizer`` class together with the GP factories and the
acquisition / sampling / candidate-selection helpers that it uses.
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
    CU_PRECURSOR_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, OBJECTIVES, FEAS_COLS, RAW_BOUNDS,
    SQUARENESS_BINS, DEFAULT_SQUARENESS_BIN, OTSU_N_THRESHOLDS,
    SYNTHESIS_FEATURES, TRANSFER_MODE, TRANSFER_FEATURES,
    PRECURSOR_DESCRIPTOR_FEATURES,
)
from features import (
    add_chemical_features,
    compute_feature_bounds,
    round_to_practical,
    chemical_to_raw_features,
    CHEM_FEATURES, HYBRID_FEATURES,
)
from diagnostics import (
    loo_cv,
    detect_extrapolation,
    diagnose_collinearity,
    evaluate_all_classifiers,
)


def make_gp_regressor(n_features: int, use_ard: bool = False) -> GaussianProcessRegressor:
    """GP regressor with a Matern-5/2 kernel.

    Parameters
    ----------
    n_features : int
        Number of input features.
    use_ard : bool
        Use per-feature lengthscales (ARD). Helpful when transfer learning is
        active and synthesis features sit on different scales than precursor
        descriptors. The default (False) uses a single shared lengthscale,
        which is better behaved on small homogeneous datasets.
    """
    if use_ard:
        length_scale = [1.0] * n_features
        ls_bounds = (0.1, 10.0)
    else:
        length_scale = 1.0
        ls_bounds = (0.3, 10.0)

    kernel = (
        C(1.0, (0.01, 100.0)) *
        Matern(length_scale=length_scale, length_scale_bounds=ls_bounds, nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 2.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        n_restarts_optimizer=10, random_state=42, alpha=1e-8,
    )


def make_gp_classifier(n_features: int) -> GaussianProcessClassifier:
    """GP classifier used for the feasibility (HasProduct / PhasePure / IsCubic) targets."""
    kernel = C(1.0, (0.01, 100.0)) * Matern(
        length_scale=[1.0] * n_features, length_scale_bounds=(0.1, 10.0), nu=2.5
    )
    return GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=5,
        random_state=42, max_iter_predict=200
    )


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray,
    y_best: float, xi: float = 0.01, minimize: bool = True
) -> np.ndarray:
    """Expected Improvement acquisition."""
    sigma = np.maximum(sigma, 1e-9)
    improvement = (y_best - mu - xi) if minimize else (mu - y_best - xi)
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def prob_in_interval(
    mu: np.ndarray, sigma: np.ndarray,
    target: float, tolerance: float
) -> np.ndarray:
    """``P(target - tol <= Y <= target + tol)`` under N(mu, sigma)."""
    sigma = np.maximum(sigma, 1e-9)
    z_hi = (target + tolerance - mu) / sigma
    z_lo = (target - tolerance - mu) / sigma
    return norm.cdf(z_hi) - norm.cdf(z_lo)


def _normalize_ei(ei: np.ndarray) -> np.ndarray:
    """Min-max normalize EI to [0, 1]; returns zeros when all values are equal."""
    ptp = np.ptp(ei)
    if ptp < 1e-8:
        return np.zeros_like(ei)
    return (ei - ei.min()) / ptp


def latin_hypercube_sample(
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    feature_order: List[str],
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate Latin-hypercube samples on the box defined by ``bounds``."""
    from scipy.stats.qmc import LatinHypercube
    n_dims = len(feature_order)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    lows = np.array([bounds[k][0] for k in feature_order])
    highs = np.array([bounds[k][1] for k in feature_order])
    return samples * (highs - lows) + lows


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
    X_history_scaled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
    """Filter candidates by P(size)/P(feasible) thresholds, then greedily pick
    diverse top-acquisition points.

    Candidates must be at least ``min_distance`` (scaled-space) away from each
    other and from any points in ``X_history_scaled`` (previously recommended
    conditions).

    Returns
    -------
    (X_feasible, acq_feasible, selected_indices, feasibility_mask)
        ``selected_indices`` are positions within ``X_feasible``.
    """
    mask = (p_size >= p_size_min) & (p_feasible >= p_feas_min)
    if mask.sum() == 0:
        print("[WARNING] No candidates meet constraints. Relaxing...")
        mask = np.ones(len(X), dtype=bool)

    X_feas = X[mask]
    acq_feas = acq_total[mask]
    order = np.argsort(acq_feas)[::-1]
    X_scaled = scaler.transform(X_feas)

    # Seed the exclusion set with past recommendations so new candidates are
    # forced away from previously explored conditions.
    if X_history_scaled is not None and len(X_history_scaled) > 0:
        excluded_scaled = list(X_history_scaled)
    else:
        excluded_scaled = []

    selected, selected_scaled = [], []
    for idx in order:
        x = X_scaled[idx]
        all_existing = selected_scaled + excluded_scaled
        if all_existing and np.min(cdist([x], all_existing)) < min_distance:
            continue
        selected.append(idx)
        selected_scaled.append(x)
        if len(selected) >= n_return:
            break

    return X_feas, acq_feas, selected, mask


def otsu_threshold(values: np.ndarray, n_thresholds: int = OTSU_N_THRESHOLDS) -> float:
    """Otsu threshold (between-class variance maximizer).

    Used on the squareness values of cubic-only particles to split them into
    ``highly_cubic`` (>= threshold) and ``poorly_cubic`` (< threshold) bins.
    """
    thresholds = np.linspace(values.min(), values.max(), n_thresholds)
    best_thresh = thresholds[0]
    best_variance = 0.0

    for t in thresholds:
        below = values[values <= t]
        above = values[values > t]
        if len(below) == 0 or len(above) == 0:
            continue
        w_below = len(below) / len(values)
        w_above = len(above) / len(values)
        between_variance = w_below * w_above * (below.mean() - above.mean()) ** 2
        if between_variance > best_variance:
            best_variance = between_variance
            best_thresh = t

    return float(best_thresh)


def compute_bin_probability(
    sq_mu: np.ndarray,
    sq_std: np.ndarray,
    p_cubic: np.ndarray,
    threshold: float,
    squareness_bin: str = DEFAULT_SQUARENESS_BIN,
) -> np.ndarray:
    """Probability that a candidate falls in the requested squareness bin.

    Combines the squareness GP posterior with the IsCubic classifier::

        highly_cubic = P(IsCubic) * P(Squareness >= threshold)
        poorly_cubic = P(IsCubic) * P(Squareness <  threshold)
        multipod     = 1 - P(IsCubic)
    """
    sq_std = np.maximum(sq_std, 1e-9)
    if squareness_bin == 'highly_cubic':
        p_above = 1.0 - norm.cdf((threshold - sq_mu) / sq_std)
        return p_cubic * p_above
    elif squareness_bin == 'poorly_cubic':
        p_below = norm.cdf((threshold - sq_mu) / sq_std)
        return p_cubic * p_below
    elif squareness_bin == 'multipod':
        return 1.0 - p_cubic
    else:
        raise ValueError(
            f"Unknown squareness_bin '{squareness_bin}'. "
            f"Choose from {SQUARENESS_BINS}"
        )


class Cu3VS4Optimizer:
    """Chemically-informed Bayesian optimizer for Cu3MS4 synthesis.

    Size is the only regression objective in the acquisition. The CV and
    Squareness GPs are kept for informational predictions only. Squareness
    binning (``multipod`` / ``highly_cubic`` / ``poorly_cubic``) enters as a
    feasibility-style constraint with the Otsu threshold.

    Feature modes: ``raw``, ``chemical``, ``hybrid``, ``synthesis``, ``transfer``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_mode: str = 'synthesis',
        validate: bool = True,
        frozen_otsu_threshold: Optional[float] = None,
    ):
        self.feature_mode = feature_mode
        self.use_ard = (
            feature_mode == 'transfer'
            and TRANSFER_MODE.get('use_ard_kernel', True)
        )

        self.df_all = df.copy()
        # Backwards compatibility: allow callers that still pass a 'GSD' column
        # by aliasing it to 'CV'.
        if 'CV' not in self.df_all.columns and 'GSD' in self.df_all.columns:
            self.df_all['CV'] = self.df_all['GSD']
        if feature_mode in ['chemical', 'hybrid', 'synthesis', 'transfer']:
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
        elif feature_mode == 'transfer':
            self.features = list(TRANSFER_FEATURES)
        else:
            raise ValueError(
                f"Unknown feature_mode: {feature_mode}. "
                f"Use 'raw', 'chemical', 'hybrid', 'synthesis', or 'transfer'."
            )

        if len(self.df_success) < 5:
            raise ValueError(f"Need ≥5 successful experiments")

        self.X_all = self.df_all[self.features].values.astype(float)
        self.X_success = self.df_success[self.features].values.astype(float)

        self.scaler = StandardScaler()
        self.scaler.fit(self.X_all)
        self.X_all_scaled = self.scaler.transform(self.X_all)
        self.X_success_scaled = self.scaler.transform(self.X_success)

        self.bounds = compute_feature_bounds(self.features, self.df_all)

        self._frozen_otsu_threshold = frozen_otsu_threshold
        self._build_models()

        self.metrics = {}
        if validate:
            self._validate()

    def _gp_factory(self):
        return make_gp_regressor(len(self.features), use_ard=self.use_ard)

    def _build_models(self):
        n = len(self.features)
        ard_str = " [ARD]" if self.use_ard else ""
        print(f"Building models with {n} features ({self.feature_mode}{ard_str} mode)...")

        # Regression GPs are trained on cubic-only data: size, CV and squareness
        # are only meaningful for cubic particles, and mixing morphologies
        # confounds the GP.
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

        # Use the frozen Otsu threshold if one was supplied, else compute it
        # from the current cubic data (standalone usage without
        # SelfValidatingOptimizer).
        if self._frozen_otsu_threshold is not None:
            self.sq_threshold = self._frozen_otsu_threshold
        else:
            sq_vals = self.df_cubic['Squareness'].dropna().values
            self.sq_threshold = otsu_threshold(sq_vals) if len(sq_vals) >= 4 else 0.81
        print(f"  Squareness Otsu threshold: {self.sq_threshold:.3f}")

        self.gp_size = make_gp_regressor(n, use_ard=self.use_ard)
        self.gp_cv = make_gp_regressor(n, use_ard=self.use_ard)
        self.gp_sq = make_gp_regressor(n, use_ard=self.use_ard)
        self.gp_size.fit(self.X_cubic_scaled, self.df_cubic["Size"].values)
        self.gp_cv.fit(self.X_cubic_scaled, self.df_cubic["CV"].values)
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
        for name in ["Size", "CV", "Squareness"]:
            y = self.df_cubic[name].values
            cv = loo_cv(
                self.X_cubic, y, self._gp_factory,
                return_predictions=True, scaler_factory=StandardScaler,
            )
            self.metrics[name] = cv
            print(f"  {name}: R²={cv['r2']:.3f}, RMSE={cv['rmse']:.3f}")

        if (self.feature_mode == 'transfer'
                and 'Cu_precursor' in self.df_cubic.columns
                and self.df_cubic['Cu_precursor'].nunique() > 1):
            self._validate_per_precursor()

    def _validate_per_precursor(self):
        """Print LOO-CV R² broken out by Cu precursor."""
        from sklearn.metrics import r2_score, mean_squared_error
        print("\n  Per-precursor LOO breakdown:")
        cu_labels = self.df_cubic['Cu_precursor'].values
        for name in ["Size", "CV", "Squareness"]:
            cv = self.metrics.get(name)
            if cv is None or 'y_pred' not in cv:
                continue
            y_true = self.df_cubic[name].values
            y_pred = cv['y_pred']
            for prec in sorted(self.df_cubic['Cu_precursor'].unique()):
                mask = cu_labels == prec
                n_prec = mask.sum()
                if n_prec < 3:
                    print(f"    {name}/{prec}: n={n_prec} (too few for R²)")
                    continue
                r2 = r2_score(y_true[mask], y_pred[mask])
                rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                print(f"    {name}/{prec}: R²={r2:.3f}, RMSE={rmse:.3f} (n={n_prec})")

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict all properties and feasibility."""
        X_scaled = self.scaler.transform(X)
        size_mu, size_std = self.gp_size.predict(X_scaled, return_std=True)
        cv_mu, cv_std = self.gp_cv.predict(X_scaled, return_std=True)
        sq_mu, sq_std = self.gp_sq.predict(X_scaled, return_std=True)

        n = len(X)
        p_product = (np.ones(n) if self.clf_product is None
                     else self.clf_product.predict_proba(X_scaled)[:, 1])
        p_pure = (np.ones(n) if self.clf_pure is None
                  else self.clf_pure.predict_proba(X_scaled)[:, 1])
        p_cubic = (np.ones(n) if self.clf_cubic is None
                   else self.clf_cubic.predict_proba(X_scaled)[:, 1])

        return {
            'size_mu': size_mu, 'size_std': size_std,
            'cv_mu': cv_mu, 'cv_std': cv_std,
            'sq_mu': sq_mu, 'sq_std': sq_std,
            'p_product': p_product, 'p_pure': p_pure, 'p_cubic': p_cubic,
            'p_feasible': p_product * p_pure * p_cubic
        }

    def acquisition(
        self, X: np.ndarray, target_size: float, size_tol: float,
        preds: Optional[Dict[str, np.ndarray]] = None,
        squareness_bin: str = DEFAULT_SQUARENESS_BIN,
    ) -> Dict[str, np.ndarray]:
        """Compute the acquisition score.

        ``total = P(size in target interval) * P(HasProduct) * P(PhasePure) * P(squareness_bin)``

        CV and squareness GP predictions are included in the return dict for
        informational display but do not enter the acquisition.
        """
        if preds is None:
            preds = self.predict(X)

        p_size = prob_in_interval(
            preds['size_mu'], preds['size_std'], target_size, size_tol
        )
        p_feasible = preds['p_product'] * preds['p_pure']
        p_bin = compute_bin_probability(
            preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
            self.sq_threshold, squareness_bin,
        )

        total = p_size * p_feasible * p_bin

        return {
            **preds,
            'total': total, 'p_size': p_size,
            'p_feasible': p_feasible, 'p_bin': p_bin,
        }

    def recommend(
        self,
        target_size: float, size_tol: float = 2.5,
        squareness_bin: str = DEFAULT_SQUARENESS_BIN,
        p_size_min: float = 0.2, p_feas_min: float = 0.3,
        n_candidates: int = 20000, n_return: int = 2,
        min_distance: float = 0.3,
        seed: Optional[int] = None,
        warn_extrapolation: bool = True, extrapolation_threshold: float = 2.0
    ) -> pd.DataFrame:
        """Recommend synthesis conditions for target size and squareness bin."""
        if self.feature_mode == 'transfer':
            synth_feats = [f for f in self.features if f not in PRECURSOR_DESCRIPTOR_FEATURES]
            synth_bounds = {k: v for k, v in self.bounds.items() if k in synth_feats}
            X_synth = latin_hypercube_sample(n_candidates, synth_bounds, synth_feats, seed)
            extra_cols = []
            for feat in self.features:
                if feat in PRECURSOR_DESCRIPTOR_FEATURES:
                    val = self.bounds[feat][0]  # point bound (lo == hi)
                    extra_cols.append(np.full(n_candidates, val))
            X = np.column_stack([X_synth] + extra_cols) if extra_cols else X_synth
        else:
            X = latin_hypercube_sample(n_candidates, self.bounds, self.features, seed)
        acq = self.acquisition(X, target_size, size_tol, squareness_bin=squareness_bin)

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
                'Pred_CV': round(float(acq['cv_mu'][mask][idx]), 3),
                'Pred_Squareness': round(float(acq['sq_mu'][mask][idx]), 3),
                'Squareness_Bin': squareness_bin,
                'P_Size': round(float(acq['p_size'][mask][idx]), 3),
                'P_Feasible': round(float(acq['p_feasible'][mask][idx]), 3),
                'P_Bin': round(float(acq['p_bin'][mask][idx]), 3),
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
        # 'synthesis', 'hybrid', and 'transfer' all go through _synthesis_to_raw
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
        """Back-transform hybrid/synthesis/transfer features to raw lab parameters.

        Precursor descriptor features (``Cu_precursor_hardness`` etc.) are
        dropped silently -- they describe precursor identity, not a synthesis
        parameter that maps to a raw lab setting.

        Reconstruction::

            Temp, DDT         pass through if present as raw factors
            VOacac            = CuI / Cu_V_ratio
            Time              = 10 ** log_Time
            total_metal       = Metal_Conc / 1000 * total_vol   (else CuI + VOacac)
            DDT (if derived)  = S_Metal_ratio      * total_metal / DDT_MMOL_PER_ML
            OAm               = Ligand_Metal_ratio * total_metal / OAM_MMOL_PER_ML
        """
        feat_dict = {k: v for k, v in feat_dict.items()
                     if k not in PRECURSOR_DESCRIPTOR_FEATURES}
        raw = {}
        for f in RAW_FACTORS:
            if f in feat_dict:
                raw[f] = feat_dict[f]

        if 'Cu_V_ratio' in feat_dict and 'VOacac' not in raw:
            raw['VOacac'] = CU_PRECURSOR_MMOL / feat_dict['Cu_V_ratio']

        if 'log_Time' in feat_dict and 'Time' not in raw:
            raw['Time'] = 10 ** feat_dict['log_Time']

        if 'Metal_Conc' in feat_dict:
            total_metal = (feat_dict['Metal_Conc'] / 1000.0) * TOTAL_VOLUME_ML
        else:
            voacac = raw.get('VOacac', CU_PRECURSOR_MMOL / feat_dict.get('Cu_V_ratio', 1.0))
            total_metal = CU_PRECURSOR_MMOL + voacac

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
        """Extract learned lengthscales (inverse interpreted as importance).

        Handles both isotropic (scalar) and ARD (per-feature) kernels.
        """
        results = []
        gp_models = [("Size", self.gp_size), ("CV", self.gp_cv),
                     ("Squareness", self.gp_sq)]
        for name, gp in gp_models:
            try:
                ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale)
                if ls.shape[0] == 1:
                    for feat in self.features:
                        results.append({
                            'Model': name, 'Feature': feat,
                            'Lengthscale': float(ls[0]),
                            'Importance': 1.0 / float(ls[0]),
                        })
                else:
                    for feat, l in zip(self.features, ls):
                        results.append({
                            'Model': name, 'Feature': feat,
                            'Lengthscale': float(l),
                            'Importance': 1.0 / float(l),
                        })
            except AttributeError:
                pass
        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Gradient-based feature importance.

        For every feature and training point we estimate ``|df/dx_j|`` with
        central finite differences in the scaled space; the mean absolute
        gradient is a data-distribution-aware importance that works for both
        isotropic and ARD kernels.
        """
        eps = 0.05
        results = []
        gp_models = [("Size", self.gp_size), ("CV", self.gp_cv),
                     ("Squareness", self.gp_sq)]

        for name, gp in gp_models:
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
        """Run all diagnostic checks and return a single report dict."""
        print(f"\n{'='*70}")
        print("MODEL DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"\nFeature mode: {self.feature_mode}")
        print(f"Features ({len(self.features)}): {self.features}")
        print(f"Training samples: {len(self.df_all)} total, "
              f"{len(self.df_success)} successful, "
              f"{len(self.df_cubic)} cubic for regression")

        ratio = len(self.df_cubic) / len(self.features)
        if ratio < 5:
            print(f"[warning] Only {ratio:.1f} cubic regression samples per feature (recommend >= 10)")
        else:
            print(f"Cubic regression samples per feature: {ratio:.1f}")

        diagnostics = {
            'feature_mode': self.feature_mode,
            'n_features': len(self.features),
            'n_samples': len(self.df_all),
            'n_successful': len(self.df_success),
            'n_cubic_regression': len(self.df_cubic),
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
            print(f"[warning] High VIF features: {high_vif['Feature'].tolist()}")
        else:
            print("No severe collinearity detected")
        print(vif_df.to_string(index=False))

        print(f"\n--- Classifier Calibration ---")
        diagnostics['classifier_calibration'] = self.get_classifier_calibration(verbose=False)
        for name, cal in diagnostics['classifier_calibration'].items():
            print(f"{name}: Brier={cal['brier_score']:.4f}, ECE={cal['ece']:.4f}")
            print(f"  {cal['interpretation']}")

        print(f"\n{'='*70}")
        return diagnostics
