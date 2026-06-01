"""Self-validating Bayesian optimization for Cu3MS4 nanoparticle synthesis.

Extends the base ``Cu3VS4Optimizer`` with:

    - experiment tracking with source attribution,
    - frozen per-recommendation prediction snapshots,
    - residual-GP bias correction and per-property uncertainty calibration,
    - transfer learning across Cu and Group-5 metal precursors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

from config import (
    CU_PRECURSOR_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, N_RECOMMENDATIONS, MIN_COMPLETED_FOR_ERROR_MODEL,
    DEFAULT_SQUARENESS_BIN, TRANSFER_MODE, CURRENT_PRECURSORS,
)
from features import (
    add_chemical_features,
    round_to_practical,
    build_feature_vector_from_raw,
    chemical_to_raw_features,
)
from experiment_store import ExperimentStore, RecommendationStore
from optimizer import (
    Cu3VS4Optimizer,
    otsu_threshold,
    latin_hypercube_sample,
    _select_diverse_candidates,
)
from diagnostics import detect_extrapolation, compare_feature_modes


class ErrorLearner:
    """Post-hoc error / calibration learner for the base GP regressors.

    Fit on the *completed recommendations only* -- the points where the
    optimizer made a prediction (frozen snapshot in ``recommendations.json``)
    and we later observed the actual outcome. For each property in
    ``{Size, CV, Squareness}`` it learns two things:

    1. A **residual GP** mapping ``X_scaled -> (actual - predicted)`` using
       the base optimizer's feature scaler. At prediction time its mean is
       *added* to the base GP mean, which captures any spatially varying
       systematic bias the base model couldn't absorb (e.g. a direction in
       feature space where the model consistently under- or over-predicts).

       Because it is itself a GP fit on residuals it is not an oracle: when
       extrapolating away from completed recommendations its bias estimate
       falls back to the prior (~ ``mean_bias`` for the property), which is
       the correct conservative behaviour. The residual GP's own posterior
       variance is intentionally not folded into the predictive sigma -- the
       calibration factor below does that.

    2. A scalar **uncertainty calibration factor** per property, fit so that
       the empirical RMS z-score of completed recommendations is ~1. At
       prediction time the base GP sigma is multiplied by this factor.
       ``factor == 1``  base GP is already well calibrated.
       ``factor > 1``   base GP was overconfident.
       ``factor < 1``   base GP was underconfident.
       The factor is floored at 0.5 so one fluky run cannot collapse the
       intervals to near zero.

    Both corrections only switch on after ``MIN_COMPLETED_FOR_ERROR_MODEL``
    (default 10) recommendations have been completed. Until then,
    ``SelfValidatingOptimizer._warm_start_calibration_from_loo`` can still
    seed the calibration factor from LOO-CV coverage of the base training
    set.
    """

    def __init__(self, min_samples: int = MIN_COMPLETED_FOR_ERROR_MODEL):
        self.min_samples = min_samples
        self.residual_models = {'Size': None, 'CV': None, 'Squareness': None}
        self.calibration_factors = {'Size': 1.0, 'CV': 1.0, 'Squareness': 1.0}
        self.mean_bias = {'Size': 0.0, 'CV': 0.0, 'Squareness': 0.0}
        self.training_data = {'X': None, 'errors': {'Size': None, 'CV': None, 'Squareness': None}}
        self.is_fitted = False
        self.n_training_samples = 0
        self.scaler = None

    def fit(
        self,
        completed_recommendations: List[Dict],
        scaler: StandardScaler,
        feature_mode: str,
        feature_names: List[str]
    ):
        self.scaler = scaler
        n = len(completed_recommendations)

        if n < self.min_samples:
            print(f"[ErrorLearner] Insufficient data: {n}/{self.min_samples} completed recommendations")
            self.is_fitted = False
            return

        print(f"[ErrorLearner] Fitting on {n} completed recommendations...")

        X_list = []
        errors = {'Size': [], 'CV': [], 'Squareness': []}
        z_scores = {'Size': [], 'CV': [], 'Squareness': []}

        for rec in completed_recommendations:
            cond = rec['conditions']
            prec = rec.get('precursors', {})
            X_list.append(build_feature_vector_from_raw(
                cond, feature_mode=feature_mode, feature_names=feature_names,
                cu_precursor=prec.get('Cu_precursor'),
                metal_precursor=prec.get('Metal_precursor'),
            ))
            if rec['errors']:
                for prop in ['Size', 'CV', 'Squareness']:
                    err = rec['errors'].get(f'{prop.lower()}_error')
                    z = rec['errors'].get(f'{prop.lower()}_z_score')
                    if err is not None:
                        errors[prop].append(err)
                    if z is not None:
                        z_scores[prop].append(z)

        X = np.array(X_list)
        X_scaled = scaler.transform(X)
        self.training_data['X'] = X_scaled
        self.n_training_samples = n

        for prop in ['Size', 'CV', 'Squareness']:
            err_array = np.array(errors[prop])
            z_array = np.array(z_scores[prop])

            if len(err_array) < self.min_samples:
                print(f"  {prop}: Insufficient error data ({len(err_array)} samples)")
                continue

            self.training_data['errors'][prop] = err_array
            self.mean_bias[prop] = float(np.mean(err_array))

            if len(z_array) >= self.min_samples:
                rms_z = np.sqrt(np.mean(z_array**2))
                self.calibration_factors[prop] = max(0.5, rms_z)
                print(f"  {prop}: Mean bias = {self.mean_bias[prop]:.3f}, "
                      f"Calibration factor = {self.calibration_factors[prop]:.2f}")

            try:
                kernel = (
                    C(1.0, (0.01, 100.0)) *
                    Matern(length_scale=[1.0] * X_scaled.shape[1], length_scale_bounds=(0.1, 10.0), nu=2.5) +
                    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
                )
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)
                gp.fit(X_scaled, err_array)
                self.residual_models[prop] = gp
            except Exception as e:
                print(f"  {prop}: Could not fit residual GP: {e}")
                self.residual_models[prop] = None

        self.is_fitted = True
        print(f"[ErrorLearner] Fitting complete")

    def predict_bias(self, X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        bias = {}
        for prop in ['Size', 'CV', 'Squareness']:
            if self.residual_models[prop] is not None and self.is_fitted:
                bias[prop] = self.residual_models[prop].predict(X_scaled)
            elif self.is_fitted:
                bias[prop] = np.full(len(X_scaled), self.mean_bias[prop])
            else:
                bias[prop] = np.zeros(len(X_scaled))
        return bias

    def get_calibration_factor(self, prop: str) -> float:
        return self.calibration_factors.get(prop, 1.0)

    def has_enough_data(self) -> bool:
        return self.n_training_samples >= self.min_samples

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            'is_fitted': self.is_fitted,
            'n_training_samples': self.n_training_samples,
            'min_samples_required': self.min_samples,
            'mean_bias': self.mean_bias.copy(),
            'calibration_factors': self.calibration_factors.copy(),
            'residual_models_fitted': {prop: (model is not None) for prop, model in self.residual_models.items()},
        }

    def save(self, json_path: Path):
        from experiment_store import _atomic_json_save
        data = {
            'is_fitted': self.is_fitted,
            'n_training_samples': self.n_training_samples,
            'min_samples': self.min_samples,
            'mean_bias': self.mean_bias,
            'calibration_factors': self.calibration_factors,
        }
        _atomic_json_save(json_path, data)

    def load(self, json_path: Path):
        import json
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.is_fitted = data.get('is_fitted', False)
            self.n_training_samples = data.get('n_training_samples', 0)
            self.min_samples = data.get('min_samples', MIN_COMPLETED_FOR_ERROR_MODEL)
            self.mean_bias = data.get('mean_bias', {'Size': 0.0, 'CV': 0.0, 'Squareness': 0.0})
            self.calibration_factors = data.get('calibration_factors', {'Size': 1.0, 'CV': 1.0, 'Squareness': 1.0})

    def __repr__(self):
        if self.is_fitted:
            return f"ErrorLearner(fitted on {self.n_training_samples} samples)"
        return f"ErrorLearner(not fitted, need {self.min_samples} samples)"


class SelfValidatingOptimizer:
    """Cu3MS4 Bayesian optimizer with self-validation.

    Tracks experiments, stores frozen prediction snapshots per recommendation,
    learns from prediction errors, and applies bias correction together with
    a per-property uncertainty calibration.

    When ``TRANSFER_MODE`` is enabled in ``config.py`` the optimizer uses the
    ``'transfer'`` feature mode, which appends precursor descriptors to the
    synthesis features and switches the GP kernel to ARD.
    """

    def __init__(
        self,
        data_dir: Path = None,
        initialize_from_csv: bool = True,
        initial_csv_imports: List[Dict] = None,
        feature_mode: str = 'hybrid',
        base_optimizer: Any = None
    ):
        """
        Parameters
        ----------
        data_dir : Path
            Directory for experiments.json, recommendations.json, etc.
        initialize_from_csv : bool
            Import CSV data on first run (when experiments.json is empty).
        initial_csv_imports : list of dict, optional
            Explicit list of CSVs to import on first run. Each dict has:
            ``{'path': Path, 'cu_precursor': str, 'metal_precursor': str}``.
            When provided, these override the default CSV auto-detection.
            Use this for transfer-learning campaigns that pool data from
            multiple precursor CSVs.
        feature_mode : str
            Feature mode ('raw', 'chemical', 'synthesis', 'hybrid', 'transfer').
        base_optimizer : Any
            Pre-built Cu3VS4Optimizer (rarely used; normally built internally).
        """
        if data_dir is None:
            raise ValueError("data_dir must be provided")

        self.data_dir = Path(data_dir)

        if TRANSFER_MODE.get('enabled', False):
            self.feature_mode = 'transfer'
        else:
            self.feature_mode = feature_mode

        self.exp_store = ExperimentStore(self.data_dir / "experiments.json")
        self.rec_store = RecommendationStore(self.data_dir / "recommendations.json")
        self.error_learner = ErrorLearner()

        if initialize_from_csv and len(self.exp_store) == 0:
            if initial_csv_imports is not None:
                for spec in initial_csv_imports:
                    csv_path = Path(spec['path'])
                    if csv_path.exists():
                        n = self.exp_store.import_from_csv(
                            csv_path, source='imported',
                            default_cu_precursor=spec.get('cu_precursor', 'CuI'),
                            default_metal_precursor=spec.get('metal_precursor', 'VO(acac)2'),
                        )
                        print(f"Imported {n} experiments from {csv_path.name} "
                              f"(Cu={spec.get('cu_precursor', 'CuI')})")
                    else:
                        print(f"[WARNING] CSV not found: {csv_path}")
            else:
                csv_cv = self.data_dir / "COMPLETE_CUVS_DATA_CV.csv"
                csv_legacy = self.data_dir / "COMPLETE_CUVS_DATA_SIDE2.csv"
                if csv_cv.exists():
                    csv_path = csv_cv
                elif csv_legacy.exists():
                    csv_path = csv_legacy
                else:
                    csv_path = csv_cv

                if csv_path.exists():
                    n = self.exp_store.import_from_csv(
                        csv_path, source='imported',
                        default_cu_precursor=CURRENT_PRECURSORS.get('Cu_precursor', 'CuI'),
                        default_metal_precursor=CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2'),
                    )
                    print(f"Imported {n} experiments from CSV as 'imported' source")

        # Compute or load the frozen Otsu threshold from initial (imported) data
        self._frozen_otsu_threshold = self._load_or_compute_otsu_threshold()

        self.base_optimizer = base_optimizer
        self._feature_mode_cache = None
        self._loo_calibration_done = False

        if len(self.exp_store) > 0:
            self._build_models()
            self._update_error_learner()

        print(f"\n{'='*60}")
        print("SELF-VALIDATING OPTIMIZER INITIALIZED")
        print(f"{'='*60}")
        print(f"Experiments: {self.exp_store}")
        print(f"Recommendations: {self.rec_store}")
        print(f"Otsu threshold (frozen): {self._frozen_otsu_threshold:.3f}")
        print(f"Error Learner: {self.error_learner}")
        if self.base_optimizer is not None:
            print(f"Base Optimizer: Cu3VS4Optimizer ({self.base_optimizer.feature_mode} mode)")
        else:
            print(f"Base Optimizer: Not initialized (need data)")

    def _load_or_compute_otsu_threshold(self) -> float:
        """Load the persisted Otsu threshold or compute it from imported (CSV) data.

        The threshold is computed once from the initial dataset and saved to
        ``otsu_threshold.json`` so it never drifts as new experiments arrive.
        """
        import json
        threshold_path = self.data_dir / "otsu_threshold.json"

        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                data = json.load(f)
            val = data.get('otsu_threshold')
            if val is not None:
                print(f"[Otsu] Loaded frozen threshold from disk: {val:.3f}")
                return float(val)

        # First time through: compute the threshold from the imported rows only
        # (the original CSV) so it is anchored to the initial dataset.
        df_imported = self.exp_store.get_by_source('imported')
        if df_imported.empty:
            print("[Otsu] No imported data available; using fallback 0.810")
            val = 0.810
            from experiment_store import _atomic_json_save
            _atomic_json_save(threshold_path, {'otsu_threshold': round(val, 4)})
            print(f"[Otsu] Saved fallback to {threshold_path}")
            return val

        cubic_mask = df_imported['Polymorph'].fillna('').astype(str).str.lower().str.strip() == 'cubic'
        cubic_sq = df_imported.loc[cubic_mask, 'Squareness'].dropna().values

        if len(cubic_sq) < 4:
            print(f"[Otsu] Only {len(cubic_sq)} cubic squareness values; using fallback 0.810")
            val = 0.810
        else:
            val = otsu_threshold(cubic_sq)
            print(f"[Otsu] Computed threshold from {len(cubic_sq)} cubic samples: {val:.3f}")

        # Persist so we never recompute against future data.
        from experiment_store import _atomic_json_save
        _atomic_json_save(threshold_path, {'otsu_threshold': round(val, 4)})
        print(f"[Otsu] Saved to {threshold_path}")
        return float(val)

    def _build_models(self):
        self._feature_mode_cache = None
        df_all = self.exp_store.get_all()
        df_success = self.exp_store.get_training_data()

        if len(df_success) < 5:
            print(f"[WARNING] Only {len(df_success)} successful experiments. Need at least 5.")
            self.base_optimizer = None
            return

        print(f"\nBuilding base optimizer (Cu3VS4Optimizer) on {len(df_success)} successful experiments...")

        if 'IsCubic' not in df_all.columns:
            df_all['IsCubic'] = (
                df_all.get('Polymorph', pd.Series()).fillna('').astype(str).str.lower().str.strip() == 'cubic'
            ).astype(int)
        if 'Cu_V_ratio' not in df_all.columns:
            df_all = add_chemical_features(df_all)

        try:
            self.base_optimizer = Cu3VS4Optimizer(
                df=df_all, feature_mode=self.feature_mode, validate=False,
                frozen_otsu_threshold=self._frozen_otsu_threshold,
            )
            print(f"  ✓ Base optimizer created ({self.feature_mode} mode)")
            print(f"  Features: {self.base_optimizer.features}")
            if self.feature_mode == 'transfer':
                print(f"  Transfer mode: vary_cu={TRANSFER_MODE.get('vary_cu_precursor', False)}, "
                      f"vary_metal={TRANSFER_MODE.get('vary_metal_precursor', False)}")
                print(f"  Target precursors: Cu={TRANSFER_MODE.get('target_cu_precursor', '—')}, "
                      f"Metal={TRANSFER_MODE.get('target_metal_precursor', '—')}")
                if TRANSFER_MODE.get('vary_cu_precursor') and df_all['Cu_precursor'].nunique() <= 1:
                    print("  ⚠ vary_cu_precursor is True but all experiments use the same Cu precursor. "
                          "Precursor features will have zero variance until mixed-precursor data is added.")
                if TRANSFER_MODE.get('vary_metal_precursor') and df_all['Metal_precursor'].nunique() <= 1:
                    print("  ⚠ vary_metal_precursor is True but all experiments use the same Metal precursor. "
                          "Precursor features will have zero variance until mixed-precursor data is added.")
        except Exception as e:
            print(f"  ✗ Failed to create base optimizer: {e}")
            self.base_optimizer = None
            raise

    def _get_scaler(self):
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")
        return self.base_optimizer.scaler

    def _get_active_precursors(self) -> Dict[str, str]:
        """Return the precursors that new recommendations should target.

        Uses the ``TRANSFER_MODE`` targets in transfer mode, else falls back to
        ``CURRENT_PRECURSORS``.
        """
        if TRANSFER_MODE.get('enabled'):
            return {
                'Cu_precursor': (TRANSFER_MODE.get('target_cu_precursor')
                                 or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI')),
                'Metal_precursor': (TRANSFER_MODE.get('target_metal_precursor')
                                    or CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2')),
            }
        return {
            'Cu_precursor': CURRENT_PRECURSORS.get('Cu_precursor', 'CuI'),
            'Metal_precursor': CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2'),
        }

    def _update_error_learner(self):
        completed = self.rec_store.get_completed()
        if len(completed) >= self.error_learner.min_samples:
            scaler = self._get_scaler() if self.base_optimizer is not None else None
            if scaler is not None:
                self.error_learner.fit(
                    completed, scaler,
                    feature_mode=self.base_optimizer.feature_mode,
                    feature_names=self.base_optimizer.features
                )
        elif self.base_optimizer is not None and not self._loo_calibration_done:
            self._warm_start_calibration_from_loo()
            self._loo_calibration_done = True

    def _warm_start_calibration_from_loo(self):
        """Seed calibration factors from LOO-CV coverage before any completions exist.

        Uses ``cal_68`` (fraction of LOO predictions within 1 sigma): if the
        model captures only 53% instead of the expected 68%, the intervals are
        too tight by a factor of ``0.68 / 0.53 ~ 1.28`` and we inflate sigma
        accordingly.
        """
        if self.base_optimizer is None:
            return

        if not self.base_optimizer.metrics:
            print("[ErrorLearner] Running LOO-CV for calibration warm-start...")
            self.base_optimizer._validate()

        updated = {}
        for prop in ['Size', 'CV', 'Squareness']:
            if prop not in self.base_optimizer.metrics:
                continue
            cal_68 = self.base_optimizer.metrics[prop].get('cal_68', 0.0)
            if cal_68 < 0.1:
                continue
            factor = round(max(0.5, 0.68 / cal_68), 3)
            self.error_learner.calibration_factors[prop] = factor
            updated[prop] = (cal_68, factor)

        if updated:
            print("[ErrorLearner] Warm-started calibration factors from LOO-CV coverage:")
            for prop, (cal, fac) in updated.items():
                direction = "overconfident → inflating" if fac > 1.0 else "underconfident → shrinking"
                print(f"  {prop}: cal_68={cal:.2f} (target 0.68) → factor={fac:.3f}x ({direction})")

    def _build_history_exclusion_set(self) -> Optional[np.ndarray]:
        """Collect every non-skipped past recommendation, in scaled feature space.

        Used by ``_select_diverse_candidates`` to enforce a minimum distance
        between new recommendations and any previously suggested conditions
        (completed or pending).
        """
        all_recs = self.rec_store.get_all()
        active_recs = [r for r in all_recs if r['status'] in ('completed', 'pending')]
        if not active_recs:
            return None

        X_list = []
        for rec in active_recs:
            try:
                prec = rec.get('precursors', {})
                vec = build_feature_vector_from_raw(
                    rec['conditions'],
                    feature_mode=self.base_optimizer.feature_mode,
                    feature_names=self.base_optimizer.features,
                    cu_precursor=prec.get('Cu_precursor'),
                    metal_precursor=prec.get('Metal_precursor'),
                )
                X_list.append(vec)
            except Exception:
                continue

        if not X_list:
            return None

        X_history = np.array(X_list)
        return self.base_optimizer.scaler.transform(X_history)

    def predict(self, X: np.ndarray, apply_correction: bool = True) -> Dict[str, np.ndarray]:
        """Predict with optional self-validation corrections.

        Pipeline:

        1. Get base GP posterior mean ``mu`` and standard deviation ``sigma``
           for each of ``{Size, CV, Squareness}`` from
           ``self.base_optimizer.predict``.
        2. If the ``ErrorLearner`` is fitted (>= ``MIN_COMPLETED_FOR_ERROR_MODEL``
           completed recommendations)::

               mu    <- mu    + residual_GP_mean(X_scaled)
               sigma <- sigma * calibration_factor[property]

        3. Otherwise, if a LOO-CV warm-start of the calibration factor is
           available (set during ``__init__``), only the sigma rescaling
           runs and the means are left unchanged.

        ``preds['correction_applied']`` records whether step 2 or 3 fired.

        The residual GP's own posterior variance is intentionally not folded
        into sigma here -- uncertainty inflation is handled by the
        calibration factor, which is fit on observed z-scores and is therefore
        already the empirically correct scale.
        """
        if self.base_optimizer is None:
            raise RuntimeError(
                "Base optimizer not built. Need at least 5 successful experiments. "
                f"Currently have {len(self.exp_store.get_training_data())} valid experiments."
            )
        preds = self.base_optimizer.predict(X)

        calibration_active = apply_correction and self.error_learner.is_fitted
        loo_calibration_active = apply_correction and self._loo_calibration_done and not self.error_learner.is_fitted

        if calibration_active:
            X_scaled = self.base_optimizer.scaler.transform(X)
            bias = self.error_learner.predict_bias(X_scaled)
            preds['size_mu'] = preds['size_mu'] + bias['Size']
            preds['cv_mu'] = preds['cv_mu'] + bias['CV']
            preds['sq_mu'] = preds['sq_mu'] + bias['Squareness']
            preds['size_std'] = preds['size_std'] * self.error_learner.get_calibration_factor('Size')
            preds['cv_std'] = preds['cv_std'] * self.error_learner.get_calibration_factor('CV')
            preds['sq_std'] = preds['sq_std'] * self.error_learner.get_calibration_factor('Squareness')
        elif loo_calibration_active:
            preds['size_std'] = preds['size_std'] * self.error_learner.get_calibration_factor('Size')
            preds['cv_std'] = preds['cv_std'] * self.error_learner.get_calibration_factor('CV')
            preds['sq_std'] = preds['sq_std'] * self.error_learner.get_calibration_factor('Squareness')

        preds['correction_applied'] = calibration_active or loo_calibration_active
        return preds

    def validate_models(self) -> Dict[str, Any]:
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")
        self.base_optimizer._validate()
        return self.base_optimizer.metrics

    def _raw_to_feature_array(self, raw_conditions: Dict[str, float]) -> np.ndarray:
        vec = build_feature_vector_from_raw(
            raw_conditions,
            feature_mode=self.base_optimizer.feature_mode,
            feature_names=self.base_optimizer.features,
        )
        return np.array(vec)

    def recommend(
        self,
        target_size: float, size_tol: float = 2.5,
        squareness_bin: str = DEFAULT_SQUARENESS_BIN,
        p_size_min: float = 0.15, p_feas_min: float = 0.25,
        n_candidates: int = 20000, min_distance: float = 0.3,
        n_return: int = None, seed: Optional[int] = None,
        warn_extrapolation: bool = True, extrapolation_threshold: float = 2.0
    ) -> pd.DataFrame:
        """Generate recommendations (with error correction) and persist them."""
        if n_return is None:
            n_return = N_RECOMMENDATIONS
        if self.base_optimizer is None:
            raise RuntimeError(
                "Cannot generate recommendations: base optimizer not built. "
                f"Need at least 5 successful experiments. "
                f"Currently have {len(self.exp_store.get_training_data())} valid experiments."
            )

        pending = self.rec_store.get_pending()
        same_request_pending = [
            r for r in pending
            if abs(r['target']['size'] - target_size) < 0.01
            and abs(r['target'].get('tolerance', size_tol) - size_tol) < 0.01
            and r.get('target', {}).get('squareness_bin', DEFAULT_SQUARENESS_BIN) == squareness_bin
        ]
        if same_request_pending:
            print(
                f"[INFO] Auto-skipping {len(same_request_pending)} existing recommendations "
                f"for {target_size}nm, tol={size_tol}, bin={squareness_bin}..."
            )
            for rec in same_request_pending:
                self.rec_store.skip(
                    rec['rec_id'],
                    reason=f"Superseded by new request for {target_size}nm ± {size_tol} ({squareness_bin})",
                )

        other_pending = len(pending) - len(same_request_pending)
        if other_pending > 0:
            print(f"[INFO] Keeping {other_pending} pending recommendations for other targets or bins")

        active_precursors = self._get_active_precursors()

        print(f"\n{'='*60}")
        print(f"GENERATING RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Target: {target_size} ± {size_tol} nm")
        print(f"Squareness bin: {squareness_bin}")
        print(f"Precursors: Cu={active_precursors['Cu_precursor']}, "
              f"Metal={active_precursors['Metal_precursor']}")
        print(f"Otsu threshold: {self._frozen_otsu_threshold:.3f}")
        print(f"Base optimizer: {self.base_optimizer.feature_mode} mode")
        print(f"Correction applied: {self.error_learner.is_fitted or self._loo_calibration_done}")

        base = self.base_optimizer
        X = latin_hypercube_sample(n_candidates, base.bounds, base.features, seed)
        preds = self.predict(X, apply_correction=True)
        acq = base.acquisition(X, target_size, size_tol, preds=preds, squareness_bin=squareness_bin)

        # Exclusion set so we never re-recommend a previously suggested point.
        X_history_scaled = self._build_history_exclusion_set()

        X_feas, acq_feas, selected, mask = _select_diverse_candidates(
            X, acq['total'], acq['p_size'], acq['p_feasible'],
            base.scaler, p_size_min, p_feas_min, n_return, min_distance,
            X_history_scaled=X_history_scaled,
        )

        rows, rec_ids, X_selected_list = [], [], []
        for rank, sel_idx in enumerate(selected, 1):
            feat_dict = {feat: X_feas[sel_idx, i] for i, feat in enumerate(base.features)}
            raw_params = base._feature_dict_to_raw(feat_dict)
            raw_params = round_to_practical(raw_params)

            predictions = {
                'size_mu': float(acq['size_mu'][mask][sel_idx]),
                'size_std': float(acq['size_std'][mask][sel_idx]),
                'cv_mu': float(acq['cv_mu'][mask][sel_idx]),
                'cv_std': float(acq['cv_std'][mask][sel_idx]),
                'sq_mu': float(acq['sq_mu'][mask][sel_idx]),
                'sq_std': float(acq['sq_std'][mask][sel_idx]),
                'p_feasible': float(acq['p_feasible'][mask][sel_idx]),
                'p_bin': float(acq['p_bin'][mask][sel_idx]),
                'correction_applied': bool(preds.get('correction_applied', False)),
            }

            rec_id = self.rec_store.save_recommendation(
                target_size=target_size,
                size_tolerance=size_tol,
                squareness_bin=squareness_bin,
                conditions=raw_params,
                predictions=predictions,
                feature_mode=self.base_optimizer.feature_mode,
                correction_applied=bool(preds.get('correction_applied', False)),
                rank=rank,
                precursors=active_precursors,
            )
            rec_ids.append(rec_id)
            X_selected_list.append(X_feas[sel_idx])

            rows.append({
                'Rank': rank, 'Rec_ID': rec_id, **raw_params,
                'Pred_Size': round(predictions['size_mu'], 2),
                'Pred_Size_Std': round(predictions['size_std'], 2),
                'Pred_CV': round(predictions['cv_mu'], 3),
                'Pred_Squareness': round(predictions['sq_mu'], 3),
                'Squareness_Bin': squareness_bin,
                'P_Feasible': round(predictions['p_feasible'], 3),
                'P_Bin': round(predictions['p_bin'], 3),
            })

        result_df = pd.DataFrame(rows)
        if warn_extrapolation and X_selected_list:
            X_sel = np.array(X_selected_list)
            extrap_check = detect_extrapolation(X_sel, base.X_all, base.scaler, threshold=extrapolation_threshold)
            result_df['Extrapolation_Distance'] = extrap_check['distances']
            result_df['Is_Extrapolating'] = extrap_check['is_extrapolation']
            for warning in extrap_check['warnings']:
                print(warning)

        print(f"\n✓ Generated {len(result_df)} recommendations (IDs: {rec_ids})")
        print(f"  These are now PENDING in the recommendation store.")
        print(f"  After running experiments, use complete_recommendation() to log results.")
        return result_df

    def complete_recommendation(
        self, rec_id: str,
        Size: float, CV: float = None, Squareness: float = None,
        HasProduct: int = 1, PhasePure: int = 1, Polymorph: str = 'cubic',
        GSD: float = None,
    ) -> Dict[str, Any]:
        """Close out a recommendation with actual results and refit the models."""
        # Accept either CV or GSD for backwards compatibility with old call sites.
        cv_value = CV if CV is not None else GSD
        if cv_value is None:
            raise ValueError("Must provide CV (or legacy GSD) value")
        if Squareness is None:
            raise ValueError("Must provide Squareness value")

        print(f"\n{'='*60}")
        print(f"COMPLETING RECOMMENDATION: {rec_id}")
        print(f"{'='*60}")

        rec = self.rec_store.get_by_id(rec_id)
        if rec is None:
            raise ValueError(f"Recommendation {rec_id} not found")

        actual_results = {
            'Size': Size, 'CV': cv_value, 'Squareness': Squareness,
            'HasProduct': HasProduct, 'PhasePure': PhasePure, 'Polymorph': Polymorph,
        }
        errors = self.rec_store.complete(rec_id, actual_results)
        rec_prec = rec.get('precursors', {})
        cu_prec = (rec_prec.get('Cu_precursor')
                   or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI'))
        metal_prec = (rec_prec.get('Metal_precursor')
                      or CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2'))
        exp_id = self.exp_store.add_experiment(
            conditions=rec['conditions'], results=actual_results,
            source='recommendation', recommendation_id=rec_id,
            cu_precursor=cu_prec, metal_precursor=metal_prec,
        )

        print(f"\nResults recorded:")
        print(f"  Experiment ID: {exp_id}")
        pred = rec['predictions']
        print(f"  Size: {Size} nm (predicted: {pred['size_mu']:.2f} ± {pred['size_std']:.2f})")
        print(f"  CV: {cv_value} (predicted: {pred['cv_mu']:.3f})")
        print(f"  Squareness: {Squareness} (predicted: {pred['sq_mu']:.3f})")
        print(f"\nPrediction errors:")
        for key, val in errors.items():
            if 'error' in key and val is not None:
                print(f"  {key}: {val:+.3f}")

        print(f"\nRebuilding models with new data...")
        self._build_models()
        self._update_error_learner()
        return errors

    def skip_recommendation(self, rec_id: str, reason: str = None):
        self.rec_store.skip(rec_id, reason)
        print(f"Recommendation {rec_id} marked as skipped. Reason: {reason or 'Not specified'}")

    def delete_recommendation(self, rec_id: str):
        """Remove a pending recommendation from the store permanently."""
        self.rec_store.delete(rec_id)
        print(f"Recommendation {rec_id} permanently deleted.")

    def add_manual_experiment(
        self,
        Temp: float, Time: float, VOacac: float, DDT: float, OAm: float,
        Size: float, CV: float = None, Squareness: float = None,
        HasProduct: int = 1, PhasePure: int = 1, Polymorph: str = 'cubic',
        GSD: float = None,
        Cu_precursor: str = None,
        Metal_precursor: str = None,
    ) -> str:
        cv_value = CV if CV is not None else GSD
        if cv_value is None:
            raise ValueError("Must provide CV (or legacy GSD) value")
        if Squareness is None:
            raise ValueError("Must provide Squareness value")
        conditions = {'Temp': Temp, 'Time': Time, 'VOacac': VOacac, 'DDT': DDT, 'OAm': OAm}
        results = {
            'Size': Size, 'CV': cv_value, 'Squareness': Squareness,
            'HasProduct': HasProduct, 'PhasePure': PhasePure, 'Polymorph': Polymorph
        }
        cu_prec = Cu_precursor or CURRENT_PRECURSORS.get('Cu_precursor', 'CuI')
        metal_prec = Metal_precursor or CURRENT_PRECURSORS.get('Metal_Precursor', 'VO(acac)2')
        exp_id = self.exp_store.add_experiment(
            conditions, results, source='manual',
            cu_precursor=cu_prec, metal_precursor=metal_prec,
        )
        print(f"Added manual experiment: {exp_id}")
        self._build_models()
        self._update_error_learner()
        return exp_id

    def get_pending_recommendations(self) -> pd.DataFrame:
        pending = self.rec_store.get_pending()
        if not pending:
            print("No pending recommendations.")
            return pd.DataFrame()
        rows = []
        for rec in pending:
            rows.append({
                'Rec_ID': rec['rec_id'],
                'Date': rec['timestamp'][:10],
                'Target_Size': rec['target']['size'],
                'Target_Tol': rec['target'].get('tolerance'),
                'Squareness_Bin': rec['target'].get('squareness_bin', DEFAULT_SQUARENESS_BIN),
                'Temp': rec['conditions']['Temp'],
                'Time': rec['conditions']['Time'],
                'VOacac': rec['conditions']['VOacac'],
                'DDT': rec['conditions']['DDT'],
                'OAm': rec['conditions']['OAm'],
                'Feature_Mode': rec.get('model_context', {}).get('feature_mode', self.feature_mode),
                'Cu_Precursor': rec.get('precursors', {}).get('Cu_precursor', '—'),
                'Metal_Precursor': rec.get('precursors', {}).get('Metal_precursor', '—'),
                'Correction': rec.get('model_context', {}).get('correction_applied', False),
                'Pred_Size': f"{rec['predictions']['size_mu']:.1f} ± {rec['predictions']['size_std']:.1f}",
                'Pred_CV': f"{rec['predictions']['cv_mu']:.3f}",
                'Pred_Sq': f"{rec['predictions']['sq_mu']:.3f}",
                'P_Bin': f"{rec['predictions'].get('p_bin', 0):.3f}",
            })
        return pd.DataFrame(rows)

    def get_model_assessment(self) -> Dict[str, Any]:
        stats = self.rec_store.get_error_statistics()
        stats['error_learner'] = self.error_learner.get_diagnostics()
        stats['experiment_counts'] = self.exp_store.count()
        stats['recommendation_counts'] = self.rec_store.count()
        return stats

    def predict_from_conditions(
        self,
        Temp: float, Time: float, VOacac: float, DDT: float, OAm: float,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Predict outcomes for a single set of raw synthesis conditions."""
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")

        conditions = {'Temp': Temp, 'Time': Time, 'VOacac': VOacac, 'DDT': DDT, 'OAm': OAm}
        X = np.array([self._raw_to_feature_array(conditions)])
        preds = self.predict(X, apply_correction=True)

        from optimizer import compute_bin_probability
        threshold = self.base_optimizer.sq_threshold
        p_cubic = preds['p_cubic'][0]
        sq_mu = preds['sq_mu'][0]
        sq_std = preds['sq_std'][0]

        sq_arr = np.array([sq_mu])
        std_arr = np.array([sq_std])
        cubic_arr = np.array([p_cubic])

        p_highly = float(compute_bin_probability(
            sq_arr, std_arr, cubic_arr, threshold, 'highly_cubic')[0])
        p_poorly = float(compute_bin_probability(
            sq_arr, std_arr, cubic_arr, threshold, 'poorly_cubic')[0])
        p_multipod = float(compute_bin_probability(
            sq_arr, std_arr, cubic_arr, threshold, 'multipod')[0])

        predicted_bin = max(
            [('highly_cubic', p_highly), ('poorly_cubic', p_poorly),
             ('multipod', p_multipod)],
            key=lambda x: x[1],
        )[0]

        result = {
            'conditions': conditions,
            'Size': (preds['size_mu'][0], preds['size_std'][0]),
            'CV': (preds['cv_mu'][0], preds['cv_std'][0]),
            'Squareness': (preds['sq_mu'][0], preds['sq_std'][0]),
            'P_Product': preds['p_product'][0],
            'P_PhasePure': preds['p_pure'][0],
            'P_Cubic': preds['p_cubic'][0],
            'P_Feasible': preds['p_product'][0] * preds['p_pure'][0],
            'Predicted_Bin': predicted_bin,
            'P_Bin': {'highly_cubic': p_highly, 'poorly_cubic': p_poorly, 'multipod': p_multipod},
            'correction_applied': preds['correction_applied'],
        }

        if verbose:
            print(f"\n{'='*60}")
            print("PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"\nConditions:")
            print(f"  Temp: {Temp}°C, Time: {Time} min, VOacac: {VOacac} mmol")
            print(f"  DDT: {DDT} mL, OAm: {OAm} mL")
            print(f"\nBase Optimizer: {self.base_optimizer.feature_mode} mode")
            print(f"\nPredictions {'(with error correction)' if result['correction_applied'] else '(base model)'}:")
            print(f"  Size:       {result['Size'][0]:.2f} ± {result['Size'][1]:.2f} nm")
            print(f"  CV:         {result['CV'][0]:.3f} ± {result['CV'][1]:.3f}")
            print(f"  Squareness: {result['Squareness'][0]:.3f} ± {result['Squareness'][1]:.3f}")
            print(f"\nFeasibility:")
            print(f"  P(Product):    {result['P_Product']:.3f}")
            print(f"  P(Phase Pure): {result['P_PhasePure']:.3f}")
            print(f"  P(Cubic):      {result['P_Cubic']:.3f}")
            print(f"\nSquareness Binning (threshold = {threshold:.3f}):")
            print(f"  P(highly_cubic): {p_highly:.3f}")
            print(f"  P(poorly_cubic): {p_poorly:.3f}")
            print(f"  P(multipod):     {p_multipod:.3f}")
            print(f"  Predicted bin:   {predicted_bin}")

        return result

    def compare_feature_modes(self, modes=None, verbose=True, force_recompute=False):
        cache_key = tuple(modes) if modes is not None else None
        if (not force_recompute and self._feature_mode_cache is not None
                and self._feature_mode_cache[0] == cache_key):
            if verbose:
                print("[INFO] Returning cached feature-mode comparison (use force_recompute=True to retrain)")
            return self._feature_mode_cache[1]

        df_all = self.exp_store.get_all()
        if 'IsCubic' not in df_all.columns:
            df_all['IsCubic'] = (
                df_all.get('Polymorph', pd.Series()).fillna('').astype(str).str.lower().str.strip() == 'cubic'
            ).astype(int)
        if 'Cu_V_ratio' not in df_all.columns:
            df_all = add_chemical_features(df_all)

        result_df = compare_feature_modes(df_all, modes, verbose)
        self._feature_mode_cache = (cache_key, result_df)
        return result_df

    def get_collinearity_diagnostics(self, verbose=True):
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")
        return self.base_optimizer.get_collinearity_diagnostics(verbose)

    def get_classifier_calibration(self, verbose=True):
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")
        return self.base_optimizer.get_classifier_calibration(verbose)

    def full_diagnostics(self):
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")
        diagnostics = self.base_optimizer.full_diagnostics()
        diagnostics['error_learner'] = self.error_learner.get_diagnostics()
        diagnostics['experiment_counts'] = self.exp_store.count()
        diagnostics['recommendation_counts'] = self.rec_store.count()
        print(f"\n--- Self-Validation Status ---")
        el = diagnostics['error_learner']
        if el['is_fitted']:
            print(f"✓ Error learner ACTIVE (trained on {el['n_training_samples']} samples)")
            print(f"  Mean bias corrections: {el['mean_bias']}")
            print(f"  Calibration factors: {el['calibration_factors']}")
        else:
            print(f"⏳ Error learner inactive (need {el['min_samples_required']} completed recommendations)")
        return diagnostics

    def print_status(self):
        print(f"\n{'='*60}")
        print("SYSTEM STATUS")
        print(f"{'='*60}")
        exp_counts = self.exp_store.count()
        rec_counts = self.rec_store.count()
        print(f"\nExperiments:")
        print(f"  Total: {exp_counts['total']}")
        print(f"  Imported: {exp_counts.get('imported', 0)}")
        print(f"  From recommendations: {exp_counts.get('recommendation', 0)}")
        print(f"  Manual: {exp_counts.get('manual', 0)}")
        print(f"\nRecommendations:")
        print(f"  Total: {rec_counts['total']}")
        print(f"  Pending: {rec_counts['pending']}")
        print(f"  Completed: {rec_counts['completed']}")
        print(f"  Skipped: {rec_counts['skipped']}")
        print(f"\nError Learning:")
        if self.error_learner.is_fitted:
            diag = self.error_learner.get_diagnostics()
            print(f"  Status: ACTIVE (trained on {diag['n_training_samples']} samples)")
            for prop, bias in diag['mean_bias'].items():
                print(f"    {prop} bias: {bias:+.3f}")
            for prop, cal in diag['calibration_factors'].items():
                print(f"    {prop} calibration: {cal:.2f}x")
        else:
            print(f"  Status: INACTIVE (need {self.error_learner.min_samples} completed recommendations)")
            print(f"  Current: {rec_counts['completed']} completed")
