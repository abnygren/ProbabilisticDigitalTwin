"""
Self-Validating Bayesian Optimization for Cu₃VS₄ Nanoparticle Synthesis

Extends the base Cu3VS4Optimizer with:
- Experiment tracking with source attribution
- Prediction snapshot storage
- Error learning and bias correction
- Calibrated uncertainties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

from config import (
    CUI_MMOL, TOTAL_VOLUME_ML, DDT_MMOL_PER_ML, OAM_MMOL_PER_ML,
    RAW_FACTORS, N_RECOMMENDATIONS, MIN_COMPLETED_FOR_ERROR_MODEL,
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
    expected_improvement,
    prob_in_interval,
    latin_hypercube_sample,
)
from diagnostics import detect_extrapolation, compare_feature_modes

from scipy.spatial.distance import cdist


# =============================================================================
# ERROR LEARNER
# =============================================================================

class ErrorLearner:
    """
    Learns from prediction errors to improve future predictions.

    Two correction mechanisms:
    1. Residual Model: GP trained on (X -> actual - predicted) to learn systematic biases
    2. Calibration: Multiplier for uncertainty based on historical z-scores
    """

    def __init__(self, min_samples: int = MIN_COMPLETED_FOR_ERROR_MODEL):
        self.min_samples = min_samples
        self.residual_models = {'Size': None, 'GSD': None, 'Squareness': None}
        self.calibration_factors = {'Size': 1.0, 'GSD': 1.0, 'Squareness': 1.0}
        self.mean_bias = {'Size': 0.0, 'GSD': 0.0, 'Squareness': 0.0}
        self.training_data = {'X': None, 'errors': {'Size': None, 'GSD': None, 'Squareness': None}}
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
        errors = {'Size': [], 'GSD': [], 'Squareness': []}
        z_scores = {'Size': [], 'GSD': [], 'Squareness': []}

        for rec in completed_recommendations:
            cond = rec['conditions']
            X_list.append(build_feature_vector_from_raw(cond, feature_mode=feature_mode, feature_names=feature_names))
            if rec['errors']:
                for prop in ['Size', 'GSD', 'Squareness']:
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

        for prop in ['Size', 'GSD', 'Squareness']:
            err_array = np.array(errors[prop])
            z_array = np.array(z_scores[prop])

            if len(err_array) < self.min_samples:
                print(f"  {prop}: Insufficient error data ({len(err_array)} samples)")
                continue

            self.training_data['errors'][prop] = err_array
            self.mean_bias[prop] = float(np.mean(err_array))

            if len(z_array) >= self.min_samples:
                rms_z = np.sqrt(np.mean(z_array**2))
                self.calibration_factors[prop] = max(1.0, rms_z)
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
        for prop in ['Size', 'GSD', 'Squareness']:
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
        import json
        data = {
            'is_fitted': self.is_fitted,
            'n_training_samples': self.n_training_samples,
            'min_samples': self.min_samples,
            'mean_bias': self.mean_bias,
            'calibration_factors': self.calibration_factors,
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, json_path: Path):
        import json
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.is_fitted = data.get('is_fitted', False)
            self.n_training_samples = data.get('n_training_samples', 0)
            self.min_samples = data.get('min_samples', MIN_COMPLETED_FOR_ERROR_MODEL)
            self.mean_bias = data.get('mean_bias', {'Size': 0.0, 'GSD': 0.0, 'Squareness': 0.0})
            self.calibration_factors = data.get('calibration_factors', {'Size': 1.0, 'GSD': 1.0, 'Squareness': 1.0})

    def __repr__(self):
        if self.is_fitted:
            return f"ErrorLearner(fitted on {self.n_training_samples} samples)"
        return f"ErrorLearner(not fitted, need {self.min_samples} samples)"


# =============================================================================
# SELF-VALIDATING OPTIMIZER
# =============================================================================

class SelfValidatingOptimizer:
    """
    Bayesian Optimization for Cu₃VS₄ synthesis with self-validation.

    Tracks experiments, stores prediction snapshots, learns from errors,
    and applies bias correction + calibrated uncertainties.
    """

    def __init__(
        self,
        data_dir: Path = None,
        initialize_from_csv: bool = True,
        feature_mode: str = 'hybrid',
        base_optimizer: Any = None
    ):
        if data_dir is None:
            raise ValueError("data_dir must be provided")

        self.data_dir = Path(data_dir)
        self.feature_mode = feature_mode

        self.exp_store = ExperimentStore(self.data_dir / "experiments.json")
        self.rec_store = RecommendationStore(self.data_dir / "recommendations.json")
        self.error_learner = ErrorLearner()

        if initialize_from_csv and len(self.exp_store) == 0:
            csv_path = self.data_dir / "COMPLETE_CUVS_DATA_SIDE2.csv"
            if csv_path.exists():
                n = self.exp_store.import_from_csv(csv_path, source='imported')
                print(f"Imported {n} experiments from CSV as 'imported' source")

        self.base_optimizer = base_optimizer
        self._feature_mode_cache = None

        if len(self.exp_store) > 0:
            self._build_models()
            self._update_error_learner()

        print(f"\n{'='*60}")
        print("SELF-VALIDATING OPTIMIZER INITIALIZED")
        print(f"{'='*60}")
        print(f"Experiments: {self.exp_store}")
        print(f"Recommendations: {self.rec_store}")
        print(f"Error Learner: {self.error_learner}")
        if self.base_optimizer is not None:
            print(f"Base Optimizer: Cu3VS4Optimizer ({self.base_optimizer.feature_mode} mode)")
        else:
            print(f"Base Optimizer: Not initialized (need data)")

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
            self.base_optimizer = Cu3VS4Optimizer(df=df_all, feature_mode=self.feature_mode, validate=False)
            print(f"  ✓ Base optimizer created ({self.feature_mode} mode)")
            print(f"  Features: {self.base_optimizer.features}")
        except Exception as e:
            print(f"  ✗ Failed to create base optimizer: {e}")
            self.base_optimizer = None
            raise

    def _get_scaler(self):
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")
        return self.base_optimizer.scaler

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

    def predict(self, X: np.ndarray, apply_correction: bool = True) -> Dict[str, np.ndarray]:
        """Predict with optional error correction."""
        if self.base_optimizer is None:
            raise RuntimeError(
                "Base optimizer not built. Need at least 5 successful experiments. "
                f"Currently have {len(self.exp_store.get_training_data())} valid experiments."
            )
        preds = self.base_optimizer.predict(X)

        if apply_correction and self.error_learner.is_fitted:
            X_scaled = self.base_optimizer.scaler.transform(X)
            bias = self.error_learner.predict_bias(X_scaled)
            preds['size_mu'] = preds['size_mu'] + bias['Size']
            preds['gsd_mu'] = preds['gsd_mu'] + bias['GSD']
            preds['sq_mu'] = preds['sq_mu'] + bias['Squareness']
            preds['size_std'] = preds['size_std'] * self.error_learner.get_calibration_factor('Size')
            preds['gsd_std'] = preds['gsd_std'] * self.error_learner.get_calibration_factor('GSD')
            preds['sq_std'] = preds['sq_std'] * self.error_learner.get_calibration_factor('Squareness')

        preds['correction_applied'] = apply_correction and self.error_learner.is_fitted
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
        p_size_min: float = 0.15, p_feas_min: float = 0.25,
        n_candidates: int = 20000, min_distance: float = 0.3,
        n_return: int = None, seed: Optional[int] = None,
        warn_extrapolation: bool = True, extrapolation_threshold: float = 2.0
    ) -> pd.DataFrame:
        """Generate recommendations with error correction and store them."""
        if n_return is None:
            n_return = N_RECOMMENDATIONS
        if self.base_optimizer is None:
            raise RuntimeError(
                "Cannot generate recommendations: base optimizer not built. "
                f"Need at least 5 successful experiments. "
                f"Currently have {len(self.exp_store.get_training_data())} valid experiments."
            )

        pending = self.rec_store.get_pending()
        same_target_pending = [r for r in pending if abs(r['target']['size'] - target_size) < 0.01]
        if same_target_pending:
            print(f"[INFO] Auto-skipping {len(same_target_pending)} existing recommendations for target {target_size}nm...")
            for rec in same_target_pending:
                self.rec_store.skip(rec['rec_id'], reason=f'Superseded by new request for {target_size}nm')

        other_pending = len(pending) - len(same_target_pending)
        if other_pending > 0:
            print(f"[INFO] Keeping {other_pending} pending recommendations for other target sizes")

        print(f"\n{'='*60}")
        print(f"GENERATING RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Target: {target_size} ± {size_tol} nm")
        print(f"Base optimizer: {self.base_optimizer.feature_mode} mode")
        print(f"Correction applied: {self.error_learner.is_fitted}")

        base = self.base_optimizer
        X = latin_hypercube_sample(n_candidates, base.bounds, base.features, seed)
        preds = self.predict(X, apply_correction=True)

        df_success = self.exp_store.get_training_data()
        gsd_best = df_success["GSD"].min()
        sq_best = df_success["Squareness"].max()

        ei_gsd = expected_improvement(preds['gsd_mu'], preds['gsd_std'], gsd_best, minimize=True)
        ei_sq = expected_improvement(preds['sq_mu'], preds['sq_std'], sq_best, minimize=False)
        ei_gsd_n = (ei_gsd - ei_gsd.min()) / (np.ptp(ei_gsd) + 1e-10)
        ei_sq_n = (ei_sq - ei_sq.min()) / (np.ptp(ei_sq) + 1e-10)

        w_gsd = base.objective_weights.get('GSD', 1.0)
        w_sq = base.objective_weights.get('Squareness', 1.0)
        acq_obj = (w_gsd * ei_gsd_n + w_sq * ei_sq_n) / (w_gsd + w_sq)
        p_size = prob_in_interval(preds['size_mu'], preds['size_std'], target_size, size_tol)
        total_acq = acq_obj * p_size * preds['p_feasible']

        mask = (p_size >= p_size_min) & (preds['p_feasible'] >= p_feas_min)
        if mask.sum() == 0:
            print("[WARNING] No candidates meet constraints. Relaxing...")
            mask = np.ones(len(X), dtype=bool)

        X_feas = X[mask]; acq_feas = total_acq[mask]
        preds_feas = {k: v[mask] if isinstance(v, np.ndarray) else v for k, v in preds.items()}

        order = np.argsort(acq_feas)[::-1]
        X_scaled = base.scaler.transform(X_feas)
        selected, selected_scaled = [], []
        for idx in order:
            x = X_scaled[idx]
            if selected_scaled and np.min(cdist([x], selected_scaled)) < min_distance:
                continue
            selected.append(idx); selected_scaled.append(x)
            if len(selected) >= n_return:
                break

        rows, rec_ids, X_selected_list = [], [], []
        for rank, sel_idx in enumerate(selected, 1):
            feat_dict = {feat: X_feas[sel_idx, i] for i, feat in enumerate(base.features)}
            raw_params = base._feature_dict_to_raw(feat_dict)
            raw_params = round_to_practical(raw_params)

            predictions = {
                'size_mu': float(preds_feas['size_mu'][sel_idx]),
                'size_std': float(preds_feas['size_std'][sel_idx]),
                'gsd_mu': float(preds_feas['gsd_mu'][sel_idx]),
                'gsd_std': float(preds_feas['gsd_std'][sel_idx]),
                'sq_mu': float(preds_feas['sq_mu'][sel_idx]),
                'sq_std': float(preds_feas['sq_std'][sel_idx]),
                'p_feasible': float(preds_feas['p_feasible'][sel_idx]),
            }

            rec_id = self.rec_store.save_recommendation(
                target_size=target_size, size_tolerance=size_tol,
                conditions=raw_params, predictions=predictions, rank=rank
            )
            rec_ids.append(rec_id)
            X_selected_list.append(X_feas[sel_idx])

            rows.append({
                'Rank': rank, 'Rec_ID': rec_id, **raw_params,
                'Pred_Size': round(predictions['size_mu'], 2),
                'Pred_Size_Std': round(predictions['size_std'], 2),
                'Pred_GSD': round(predictions['gsd_mu'], 3),
                'Pred_Squareness': round(predictions['sq_mu'], 3),
                'P_Feasible': round(predictions['p_feasible'], 3),
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
        Size: float, GSD: float, Squareness: float,
        HasProduct: int = 1, PhasePure: int = 1, Polymorph: str = 'cubic'
    ) -> Dict[str, Any]:
        """Complete a recommendation with actual results. Retrains models."""
        print(f"\n{'='*60}")
        print(f"COMPLETING RECOMMENDATION: {rec_id}")
        print(f"{'='*60}")

        rec = self.rec_store.get_by_id(rec_id)
        if rec is None:
            raise ValueError(f"Recommendation {rec_id} not found")

        actual_results = {
            'Size': Size, 'GSD': GSD, 'Squareness': Squareness,
            'HasProduct': HasProduct, 'PhasePure': PhasePure, 'Polymorph': Polymorph,
        }
        errors = self.rec_store.complete(rec_id, actual_results)
        exp_id = self.exp_store.add_experiment(
            conditions=rec['conditions'], results=actual_results,
            source='recommendation', recommendation_id=rec_id
        )

        print(f"\nResults recorded:")
        print(f"  Experiment ID: {exp_id}")
        print(f"  Size: {Size} nm (predicted: {rec['predictions']['size_mu']:.2f} ± {rec['predictions']['size_std']:.2f})")
        print(f"  GSD: {GSD} (predicted: {rec['predictions']['gsd_mu']:.3f})")
        print(f"  Squareness: {Squareness} (predicted: {rec['predictions']['sq_mu']:.3f})")
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

    def add_manual_experiment(
        self,
        Temp: float, Time: float, VOacac: float, DDT: float, OAm: float,
        Size: float, GSD: float, Squareness: float,
        HasProduct: int = 1, PhasePure: int = 1, Polymorph: str = 'cubic'
    ) -> str:
        conditions = {'Temp': Temp, 'Time': Time, 'VOacac': VOacac, 'DDT': DDT, 'OAm': OAm}
        results = {
            'Size': Size, 'GSD': GSD, 'Squareness': Squareness,
            'HasProduct': HasProduct, 'PhasePure': PhasePure, 'Polymorph': Polymorph
        }
        exp_id = self.exp_store.add_experiment(conditions, results, source='manual')
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
                'Temp': rec['conditions']['Temp'],
                'Time': rec['conditions']['Time'],
                'VOacac': rec['conditions']['VOacac'],
                'DDT': rec['conditions']['DDT'],
                'OAm': rec['conditions']['OAm'],
                'Pred_Size': f"{rec['predictions']['size_mu']:.1f} ± {rec['predictions']['size_std']:.1f}",
                'Pred_GSD': f"{rec['predictions']['gsd_mu']:.3f}",
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
        """Predict outcomes from raw synthesis conditions."""
        if self.base_optimizer is None:
            raise RuntimeError("Base optimizer not initialized")

        conditions = {'Temp': Temp, 'Time': Time, 'VOacac': VOacac, 'DDT': DDT, 'OAm': OAm}
        X = np.array([self._raw_to_feature_array(conditions)])
        preds = self.predict(X, apply_correction=True)

        result = {
            'conditions': conditions,
            'Size': (preds['size_mu'][0], preds['size_std'][0]),
            'GSD': (preds['gsd_mu'][0], preds['gsd_std'][0]),
            'Squareness': (preds['sq_mu'][0], preds['sq_std'][0]),
            'P_Product': preds['p_product'][0],
            'P_PhasePure': preds['p_pure'][0],
            'P_Cubic': preds['p_cubic'][0],
            'P_Feasible': preds['p_feasible'][0],
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
            print(f"  GSD:        {result['GSD'][0]:.3f} ± {result['GSD'][1]:.3f}")
            print(f"  Squareness: {result['Squareness'][0]:.3f} ± {result['Squareness'][1]:.3f}")
            print(f"\nFeasibility:")
            print(f"  P(Product):    {result['P_Product']:.3f}")
            print(f"  P(Phase Pure): {result['P_PhasePure']:.3f}")
            print(f"  P(Cubic):      {result['P_Cubic']:.3f}")
            print(f"  P(Feasible):   {result['P_Feasible']:.3f}")

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
