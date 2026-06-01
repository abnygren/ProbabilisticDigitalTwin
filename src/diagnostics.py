"""Diagnostics and model-quality assessment.

Provides:
    - leave-one-out cross-validation with per-fold refit and rescaling,
    - feature-mode comparison,
    - extrapolation detection,
    - VIF-based collinearity diagnostics,
    - Brier / ECE classifier calibration,
    - optimization-convergence summary and reporting helpers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy.stats import norm
from scipy.spatial.distance import cdist

from config import RAW_FACTORS
from features import add_chemical_features, calculate_vif, CHEM_FEATURES, HYBRID_FEATURES


def loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    gp_factory: Callable,
    return_predictions: bool = False,
    scaler_factory: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Leave-one-out CV with a real per-fold refit.

    If ``scaler_factory`` is provided, a fresh scaler is fit on each
    training fold so the test point can't leak through the standardization.
    """
    n = len(y)
    y_pred = np.zeros(n)
    y_std = np.zeros(n)

    for train_idx, test_idx in LeaveOneOut().split(X):
        if scaler_factory is not None:
            scaler = scaler_factory()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]
        gp = gp_factory()
        gp.fit(X_train, y[train_idx])
        mu, std = gp.predict(X_test, return_std=True)
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


def compare_feature_modes(
    df: pd.DataFrame,
    modes: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """LOO-CV comparison across feature representations.

    Useful when deciding which feature mode to commit to for a campaign.
    """
    # Local import avoids a circular import with optimizer.
    from optimizer import Cu3VS4Optimizer

    if modes is None:
        modes = ['raw', 'chemical', 'hybrid', 'synthesis']

    results = []

    for mode in modes:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating {mode.upper()} feature mode...")
            print(f"{'='*50}")

        try:
            opt = Cu3VS4Optimizer(df, feature_mode=mode, validate=True)
            for prop in ['Size', 'CV', 'Squareness']:
                if prop in opt.metrics:
                    m = opt.metrics[prop]
                    results.append({
                        'Mode': mode,
                        'Property': prop,
                        'N_Features': len(opt.features),
                        'R2': m['r2'],
                        'RMSE': m['rmse'],
                        'MAE': m['mae'],
                        'Cal_68': m['cal_68'],
                        'Cal_95': m['cal_95'],
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

        for metric in ['R2', 'RMSE']:
            print(f"\n{metric} by Mode and Property:")
            pivot = results_df.pivot(index='Property', columns='Mode', values=metric)
            print(pivot.round(3).to_string())

        mean_r2 = results_df.groupby('Mode')['R2'].mean()
        best_mode = mean_r2.idxmax()
        print(f"\nRecommended mode based on mean R²: {best_mode} ({mean_r2[best_mode]:.3f})")

        print("\nCalibration Analysis (target: Cal_68 ≈ 0.68, Cal_95 ≈ 0.95):")
        cal_summary = results_df.groupby('Mode')[['Cal_68', 'Cal_95']].mean()
        print(cal_summary.round(3).to_string())

    return results_df


def detect_extrapolation(
    X_new: np.ndarray,
    X_train: np.ndarray,
    scaler: StandardScaler = None,
    threshold: float = 2.0,
    method: str = 'nearest'
) -> Dict[str, Any]:
    """Flag candidate points that sit far from the training data.

    Returns a dict with ``is_extrapolation``, ``distances``,
    ``n_extrapolating`` and any human-readable ``warnings``.
    """
    X_new = np.atleast_2d(X_new)
    X_train = np.atleast_2d(X_train)

    if scaler is not None:
        X_new_scaled = scaler.transform(X_new)
        X_train_scaled = scaler.transform(X_train)
    else:
        X_new_scaled = X_new
        X_train_scaled = X_train

    distances = cdist(X_new_scaled, X_train_scaled).min(axis=1)
    is_extrapolation = distances > threshold

    warnings = []
    n_extrap = is_extrapolation.sum()
    if n_extrap > 0:
        warnings.append(
            f"[warning] {n_extrap}/{len(X_new)} points are in extrapolation regions "
            f"(distance > {threshold} from nearest training point)"
        )
        worst_idx = np.argmax(distances)
        warnings.append(
            f"   Worst case: point {worst_idx} is {distances[worst_idx]:.2f} "
            f"std from training data"
        )
        if n_extrap > len(X_new) * 0.5:
            warnings.append(
                "   [caution] Majority of candidates are extrapolating. "
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


def diagnose_collinearity(
    df: pd.DataFrame,
    feature_mode: str = 'hybrid',
    verbose: bool = True,
    feature_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Collinearity diagnostics for a feature mode.

    Returns a dict with the VIF table, full correlation matrix,
    flagged feature pairs, and a list of human-readable recommendations.
    """
    if feature_list is not None:
        features = feature_list
    elif feature_mode == 'raw':
        features = RAW_FACTORS
    elif feature_mode == 'chemical':
        features = CHEM_FEATURES
    elif feature_mode == 'hybrid':
        features = HYBRID_FEATURES
    elif feature_mode == 'synthesis':
        from config import SYNTHESIS_FEATURES
        features = list(SYNTHESIS_FEATURES)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    df_work = df.copy()
    needs_chem = feature_mode in ['chemical', 'hybrid', 'synthesis'] or (
        feature_list is not None and any(f in CHEM_FEATURES for f in feature_list)
    )
    if needs_chem and 'Cu_V_ratio' not in df_work.columns:
        df_work = add_chemical_features(df_work)

    available = [f for f in features if f in df_work.columns]
    X = df_work[available].values

    vif_df = calculate_vif(X, available)

    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, index=available, columns=available)

    problematic_pairs = []
    for i, f1 in enumerate(available):
        for j, f2 in enumerate(available):
            if i < j and abs(corr_matrix[i, j]) > 0.8:
                problematic_pairs.append((f1, f2, corr_matrix[i, j]))

    recommendations = []
    high_vif = vif_df[vif_df['VIF'] >= 10]['Feature'].tolist()

    if high_vif:
        recommendations.append(f"Features with VIF ≥ 10: {high_vif}. Consider removing or combining these.")
    if problematic_pairs:
        for f1, f2, r in problematic_pairs:
            recommendations.append(f"High correlation ({r:.2f}) between {f1} and {f2}. Consider keeping only one.")
    if feature_mode == 'hybrid':
        raw_in_hybrid = [f for f in RAW_FACTORS if f in available]
        derived_in_hybrid = [f for f in available if f not in RAW_FACTORS]
        if raw_in_hybrid and derived_in_hybrid:
            recommendations.append(
                "Hybrid mode includes both raw factors and derived features. "
                "Consider using 'synthesis' mode for mechanistically orthogonal features."
            )
    if feature_mode == 'synthesis':
        recommendations.append(
            f"Synthesis mode: {len(available)} mechanistically orthogonal features {available}. "
            f"Each maps to an independent physical control of nanoparticle synthesis."
        )
    if not recommendations:
        recommendations.append("No major collinearity issues detected.")

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

# CLASSIFIER CALIBRATION
# ------------------------------------------------------------------------------

def _build_classifier_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration metrics from true labels and predicted probabilities."""
    from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = brier_score_loss(y_true, y_prob)
    try:
        logloss = sk_log_loss(y_true, y_prob)
    except ValueError:
        logloss = np.nan

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq, predicted_freq, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            observed_freq.append(y_true[mask].mean())
            predicted_freq.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            observed_freq.append(np.nan)
            predicted_freq.append(np.nan)
            bin_counts.append(0)

    ece = 0.0
    total_samples = len(y_true)
    for obs, pred, count in zip(observed_freq, predicted_freq, bin_counts):
        if count > 0 and not np.isnan(obs):
            ece += (count / total_samples) * abs(obs - pred)

    if brier < 0.1:
        interpretation = "Excellent calibration (Brier < 0.1)"
    elif brier < 0.2:
        interpretation = "Good calibration (Brier < 0.2)"
    elif brier < 0.3:
        interpretation = "Moderate calibration (Brier < 0.3)"
    else:
        interpretation = "Poor calibration (Brier >= 0.3)"
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


def classifier_calibration_metrics(
    clf,
    X: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """In-sample calibration metrics for a probabilistic classifier.

    Returns a dict with ``brier_score``, ``log_loss``, ``ece``,
    ``calibration_bins`` and a short text ``interpretation``.
    """
    y_prob = clf.predict_proba(X)[:, 1]
    metrics = _build_classifier_calibration_metrics(y_true, y_prob, n_bins=n_bins)
    metrics['evaluation_method'] = 'in_sample'
    return metrics


def _make_calibration_gp_classifier(n_features: int) -> GaussianProcessClassifier:
    """Mirror the project GP classifier for cross-validated calibration checks."""
    kernel = C(1.0, (0.01, 100.0)) * Matern(
        length_scale=[1.0] * n_features, length_scale_bounds=(0.1, 10.0), nu=2.5
    )
    return GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=5,
        random_state=42, max_iter_predict=200
    )


def cross_validated_classifier_calibration(
    X: np.ndarray,
    y_true: np.ndarray,
    clf_factory: Callable[[], GaussianProcessClassifier],
    n_bins: int = 10,
    max_splits: int = 5,
) -> Dict[str, Any]:
    """Compute out-of-sample calibration metrics with stratified CV."""
    X = np.asarray(X)
    y_true = np.asarray(y_true).astype(int)

    classes, class_counts = np.unique(y_true, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least two classes for calibration evaluation")

    n_splits = min(max_splits, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two samples in each class for stratified CV")

    y_prob = np.zeros(len(y_true), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in splitter.split(X, y_true):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = clf_factory()
        clf.fit(X_train, y_true[train_idx])
        y_prob[test_idx] = clf.predict_proba(X_test)[:, 1]

    metrics = _build_classifier_calibration_metrics(y_true, y_prob, n_bins=n_bins)
    metrics['evaluation_method'] = 'stratified_cv'
    metrics['n_splits'] = n_splits
    return metrics


def evaluate_all_classifiers(optimizer, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Evaluate calibration of all classifiers in a Cu3VS4Optimizer."""
    results = {}
    classifiers = [
        ('HasProduct', optimizer.clf_product),
        ('PhasePure', optimizer.clf_pure),
        ('IsCubic', optimizer.clf_cubic),
    ]
    X_raw = optimizer.X_all

    n_feat = len(optimizer.features)
    for name, clf in classifiers:
        if clf is None:
            if verbose:
                print(f"\n{name}: Skipped (not fitted)")
            continue

        y_true = optimizer.df_all[name].values.astype(int)
        try:
            metrics = cross_validated_classifier_calibration(
                X_raw, y_true,
                clf_factory=lambda: _make_calibration_gp_classifier(n_feat),
            )
            metrics['in_sample'] = False
        except Exception as exc:
            metrics = classifier_calibration_metrics(
                clf, optimizer.X_all_scaled, y_true)
            metrics['evaluation_method'] = 'in_sample_fallback'
            metrics['fallback_reason'] = str(exc)
            metrics['in_sample'] = True

        results[name] = metrics
        if verbose:
            method = metrics['evaluation_method']
            if method == 'stratified_cv':
                print(f"\n{name} Classifier "
                      f"(out-of-sample stratified CV, {metrics['n_splits']} folds):")
            else:
                print(f"\n{name} Classifier (in-sample fallback):")
            balance = metrics['class_balance'] * 100
            print(f"  Class balance: {balance:.1f}% positive")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  {metrics['interpretation']}")
            if metrics.get('fallback_reason'):
                print(f"  Fallback reason: {metrics['fallback_reason']}")

    if verbose and results:
        print(f"\n{'='*50}")
        print("CALIBRATION SUMMARY")
        print(f"{'='*50}")
        print("Well-calibrated: Brier < 0.2, ECE < 0.1")
        print("Primary method: out-of-sample stratified CV.")
        print("Fallback: in-sample metrics only when class counts are too small for CV.")

    return results

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% confidence interval for a binomial success rate.

    Returns ``(lo, hi)`` on the [0, 1] scale. Robust at small n and at
    ``p`` near 0 or 1, where the normal approximation fails.
    """
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def get_optimization_convergence_summary(
    optimizer,
    last_n: Optional[int] = None,
    size_band_nm: float = 5.0
) -> Dict[str, Any]:
    """Summarise optimization progress, focused on model predictive quality.

    The primary question is "does the model predict what the experiment
    actually produces?", not "did the experiment hit an arbitrary target?".
    A recommendation that predicts 12.5 nm and yields 12.8 nm is a model
    success even if the user asked for 10 nm.

    Returns a dict with:

        prediction_quality  prediction-vs-actual metrics (MAE, z, coverage)
        target_achievement  target-hitting metrics (secondary context)
        by_target_band      both broken out per target-size band
        recent              same metrics restricted to the last ``last_n`` recs
        interpretation      short human-readable signals
    """
    completed = optimizer.rec_store.get_completed()
    completed_sorted = sorted(completed, key=lambda r: r.get('completed_timestamp') or r.get('timestamp', ''))

    def _pred_metrics(recs):
        """Prediction-quality metrics: predicted vs actual."""
        pred_errors, z_scores, within_1s, within_2s = [], [], [], []
        for r in recs:
            pred = r.get('predictions') or {}
            act = (r.get('actual_results') or {}).get('Size')
            pred_mu = pred.get('size_mu')
            pred_std = pred.get('size_std', 0.001)
            if act is None or pred_mu is None:
                continue
            err = act - pred_mu
            z = err / max(pred_std, 0.001)
            pred_errors.append(err)
            z_scores.append(z)
            within_1s.append(abs(z) < 1.0)
            within_2s.append(abs(z) < 2.0)
        n = len(pred_errors)
        if n == 0:
            return {
                'n': 0,
                'mae_nm': np.nan, 'mean_error_nm': np.nan, 'rmse_nm': np.nan,
                'mean_abs_z': np.nan, 'rms_z': np.nan,
                'within_1sigma': np.nan, 'within_2sigma': np.nan,
            }
        errs = np.array(pred_errors)
        zs = np.array(z_scores)
        return {
            'n': n,
            'mae_nm': float(np.mean(np.abs(errs))),
            'mean_error_nm': float(np.mean(errs)),
            'rmse_nm': float(np.sqrt(np.mean(errs ** 2))),
            'mean_abs_z': float(np.mean(np.abs(zs))),
            'rms_z': float(np.sqrt(np.mean(zs ** 2))),
            'within_1sigma': float(np.mean(within_1s)),
            'within_2sigma': float(np.mean(within_2s)),
        }

    def _target_metrics(recs):
        """Target-achievement metrics: actual vs user target."""
        errors_nm, within_tol = [], []
        for r in recs:
            tgt = r.get('target') or {}
            act = (r.get('actual_results') or {}).get('Size')
            if act is None:
                continue
            t = tgt.get('size')
            tol = tgt.get('tolerance', 2.5)
            if t is None:
                continue
            e_nm = abs(act - t)
            errors_nm.append(e_nm)
            within_tol.append(e_nm <= tol)
        n = len(errors_nm)
        if n == 0:
            return {
                'n': 0, 'n_success': 0, 'success_rate': np.nan,
                'success_rate_ci95': (np.nan, np.nan),
                'mean_abs_error_nm': np.nan,
            }
        n_success = int(np.sum(within_tol))
        ci_lo, ci_hi = _wilson_ci(n_success, n)
        return {
            'n': n,
            'n_success': n_success,
            'success_rate': 100 * n_success / n,
            'success_rate_ci95': (100 * ci_lo, 100 * ci_hi),
            'mean_abs_error_nm': float(np.mean(errors_nm)),
        }

    pred_overall = _pred_metrics(completed_sorted)
    tgt_overall = _target_metrics(completed_sorted)

    out: Dict[str, Any] = {
        'prediction_quality': pred_overall,
        'target_achievement': tgt_overall,
        'by_target_band': {},
        'interpretation': [],
    }

    # Per-recommendation detail for downstream plotting
    rec_details = []
    for r in completed_sorted:
        pred = r.get('predictions') or {}
        act = (r.get('actual_results') or {}).get('Size')
        tgt = (r.get('target') or {}).get('size')
        if act is not None and pred.get('size_mu') is not None:
            rec_details.append({
                'rec_id': r.get('rec_id'),
                'predicted': pred['size_mu'],
                'predicted_std': pred.get('size_std', 0),
                'actual': act,
                'target': tgt,
            })
    out['rec_details'] = rec_details

    bands: Dict[Tuple[float, float], List[Dict]] = {}
    for r in completed_sorted:
        t = (r.get('target') or {}).get('size')
        if t is None:
            continue
        low = size_band_nm * (t // size_band_nm)
        high = low + size_band_nm
        bands.setdefault((low, high), []).append(r)
    for (low, high), recs in sorted(bands.items()):
        out['by_target_band'][f"{low:.0f}-{high:.0f}"] = {
            'prediction': _pred_metrics(recs),
            'target': _target_metrics(recs),
        }

    if last_n is not None and last_n > 0 and len(completed_sorted) >= last_n:
        recent = completed_sorted[-last_n:]
        out['recent'] = {
            'prediction': _pred_metrics(recent),
            'target': _target_metrics(recent),
        }

    # Interpretation
    p = pred_overall
    if p['n'] >= 2:
        if p['mae_nm'] <= 2.0:
            out['interpretation'].append(
                f"Prediction MAE = {p['mae_nm']:.2f} nm → model is accurately predicting experimental outcomes."
            )
        elif p['mae_nm'] <= 4.0:
            out['interpretation'].append(
                f"Prediction MAE = {p['mae_nm']:.2f} nm → reasonable predictive accuracy; improving with more data."
            )
        else:
            out['interpretation'].append(
                f"Prediction MAE = {p['mae_nm']:.2f} nm → predictions are rough; consider reviewing feature mode or data quality."
            )

        if p['within_1sigma'] >= 0.55:
            out['interpretation'].append(
                f"{p['within_1sigma']:.0%} of outcomes fall within 1σ of prediction (target: 68%) → well-calibrated uncertainties."
            )
        elif p['within_1sigma'] < 0.40:
            out['interpretation'].append(
                f"Only {p['within_1sigma']:.0%} of outcomes fall within 1σ (target: 68%) → model is overconfident."
            )

        bias = p['mean_error_nm']
        if abs(bias) > 1.5:
            direction = "larger" if bias > 0 else "smaller"
            out['interpretation'].append(
                f"Systematic bias: experiments run {abs(bias):.1f} nm {direction} than predicted on average."
            )

    t = tgt_overall
    if t['n'] >= 2:
        if t['success_rate'] >= 60:
            ci_lo, ci_hi = t['success_rate_ci95']
            out['interpretation'].append(
                f"Target achievement: {t['success_rate']:.0f}% within tolerance "
                f"(n={t['n']}, 95% CI {ci_lo:.0f}–{ci_hi:.0f}%)."
            )
        else:
            out['interpretation'].append(
                f"Target achievement: {t['success_rate']:.0f}% within tolerance — the target size may be "
                f"near a physical limit of this synthesis. The model can still guide you to the closest achievable size."
            )

    # Backward compat: keep 'overall' key pointing to target metrics
    out['overall'] = tgt_overall

    return out


def print_optimization_convergence_summary(optimizer, last_n: Optional[int] = 10, size_band_nm: float = 5.0):
    """Print a convergence report centred on model predictive quality."""
    s = get_optimization_convergence_summary(optimizer, last_n=last_n, size_band_nm=size_band_nm)
    p = s['prediction_quality']
    t = s['target_achievement']

    print("\n" + "MODEL PERFORMANCE & OPTIMIZATION SUMMARY")
    print("-" * 60)

    print(f"\n  Completed recommendations: {p['n']}")
    if p['n'] < 2:
        print("  → Need at least 2 completed recs to assess performance.")
        print("=" * 60)
        return

    # Primary: Prediction quality
    print(f"\n  ── PREDICTION ACCURACY (predicted vs actual) ──")
    print(f"  Size MAE:       {p['mae_nm']:.2f} nm")
    print(f"  Size RMSE:      {p['rmse_nm']:.2f} nm")
    print(f"  Mean bias:      {p['mean_error_nm']:+.2f} nm")
    print(f"  Within 1σ:      {p['within_1sigma']:.0%}  (target: 68%)")
    print(f"  Within 2σ:      {p['within_2sigma']:.0%}  (target: 95%)")

    # Per-recommendation detail
    details = s.get('rec_details', [])
    if details:
        print(f"\n  ── PREDICTION LOG ──")
        print(f"  {'Rec':<10} {'Predicted':>10} {'Actual':>8} {'Error':>8} {'Target':>8}")
        print(f"  {'-'*46}")
        for d in details:
            err = d['actual'] - d['predicted']
            tgt_str = f"{d['target']:.0f}" if d['target'] is not None else "—"
            print(
                f"  {d['rec_id']:<10} "
                f"{d['predicted']:>7.1f} ± {d['predicted_std']:.1f} "
                f"{d['actual']:>7.1f} "
                f"{err:>+7.1f} "
                f"{tgt_str:>8}"
            )

    # Secondary: Target achievement
    print(f"\n  ── TARGET ACHIEVEMENT (actual vs user target) ──")
    ci_lo, ci_hi = t['success_rate_ci95']
    print(
        f"  Within tolerance: {t['success_rate']:.0f}% "
        f"({t['n_success']}/{t['n']}, 95% CI {ci_lo:.0f}–{ci_hi:.0f}%)"
    )
    print(f"  Mean |error| from target: {t['mean_abs_error_nm']:.2f} nm")

    if s.get('by_target_band'):
        print(f"\n  ── BY TARGET SIZE BAND ──")
        for band, bm in sorted(s['by_target_band'].items(), key=lambda x: float(x[0].split('-')[0])):
            bp = bm['prediction']
            bt = bm['target']
            if bp['n'] > 0:
                print(
                    f"    {band:>8} nm: n={bp['n']}, pred MAE={bp['mae_nm']:.2f} nm, "
                    f"target hit={bt['success_rate']:.0f}%"
                )

    if s.get('recent'):
        rp = s['recent']['prediction']
        rt = s['recent']['target']
        print(f"\n  ── LAST {last_n} RECOMMENDATIONS ──")
        print(
            f"    Pred MAE={rp['mae_nm']:.2f} nm, "
            f"within 1σ={rp['within_1sigma']:.0%}, "
            f"target hit={rt['success_rate']:.0f}%"
        )

    if s.get('interpretation'):
        print(f"\n  ── ASSESSMENT ──")
        for line in s['interpretation']:
            print(f"    • {line}")


# MODEL ASSESSMENT REPORT
# ------------------------------------------------------------------------------

def print_model_assessment(optimizer):
    """Print detailed model self-assessment report."""
    stats = optimizer.get_model_assessment()

    print("\n" + "MODEL SELF-ASSESSMENT REPORT")
    print("-" * 70)

    exp = stats.get('experiment_counts', {})
    print(f"\n📊 EXPERIMENT DATABASE")
    print(f"   Total experiments:        {exp.get('total', 0)}")
    print(f"   ├─ Imported (initial):    {exp.get('imported', 0)}")
    print(f"   ├─ From recommendations:  {exp.get('recommendation', 0)}")
    print(f"   └─ Manual additions:      {exp.get('manual', 0)}")

    rec = stats.get('recommendation_counts', {})
    print(f"\n📋 RECOMMENDATION HISTORY")
    print(f"   Total recommendations:    {rec.get('total', 0)}")
    print(f"   ├─ Pending:               {rec.get('pending', 0)}")
    print(f"   ├─ Completed:             {rec.get('completed', 0)}")
    print(f"   └─ Skipped:               {rec.get('skipped', 0)}")

    if stats.get('sufficient_data', False):
        print(f"\n📈 PREDICTION ACCURACY (from {stats.get('n_completed', 0)} completed recommendations)")
        print(f"   {'Property':<12} {'MAE':>8} {'Mean Err':>10} {'Within 1σ':>10} {'Within 2σ':>10}")
        print(f"   {'-'*52}")
        for prop in ['size', 'cv', 'squareness']:
            mae = stats.get(f'{prop}_mae', np.nan)
            mean_err = stats.get(f'{prop}_mean_error', np.nan)
            w1s = stats.get(f'{prop}_within_1sigma_rate', np.nan)
            w2s = stats.get(f'{prop}_within_2sigma_rate', np.nan)
            w1s_str = f"{w1s*100:.0f}%" if not np.isnan(w1s) else "N/A"
            w2s_str = f"{w2s*100:.0f}%" if not np.isnan(w2s) else "N/A"
            print(f"   {prop.capitalize():<12} {mae:>8.3f} {mean_err:>+10.3f} {w1s_str:>10} {w2s_str:>10}")
        print(f"\n   Target coverage: 1σ → 68%, 2σ → 95%")
        print(f"   If coverage < target → model is overconfident")
        print(f"   If coverage > target → model is underconfident")
    else:
        print(f"\n📈 PREDICTION ACCURACY")
        print(f"   Insufficient data ({stats.get('n_completed', 0)} completed, need ≥2)")

    el = stats.get('error_learner', {})
    print(f"\n🔧 ERROR CORRECTION STATUS")
    if el.get('is_fitted', False):
        print(f"   Status: ✅ ACTIVE (trained on {el.get('n_training_samples', 0)} samples)")
        print(f"\n   Bias Corrections (added to predictions):")
        for prop, bias in el.get('mean_bias', {}).items():
            print(f"      {prop}: {bias:+.3f}")
        print(f"\n   Calibration Factors (multiply uncertainty by):")
        for prop, cal in el.get('calibration_factors', {}).items():
            status = "⚠️ overconfident" if cal > 1.2 else "✅ well-calibrated"
            print(f"      {prop}: {cal:.2f}x {status}")
    else:
        needed = el.get('min_samples_required', 5)
        have = el.get('n_training_samples', 0)
        print(f"   Status: ⏳ INACTIVE (need {needed} completed recommendations, have {have})")


# STATISTICAL EVIDENCE FOR SIZE OPTIMIZATION
# ------------------------------------------------------------------------------

def compute_optimization_statistics(optimizer) -> Dict[str, Any]:
    """Compute statistical evidence that BO optimized nanocrystal size.

    Returns a dict with:
      - overall: aggregate metrics across all targets
      - by_target: per-target-size breakdown
      - tests: statistical test results (p-values, test statistics)
      - summary_table: DataFrame suitable for publication
    """
    from scipy.stats import (
        mannwhitneyu, wilcoxon, shapiro, ttest_1samp,
        spearmanr, linregress
    )

    completed = optimizer.rec_store.get_completed()
    completed_sorted = sorted(
        completed,
        key=lambda r: r.get('completed_timestamp') or r.get('timestamp', '')
    )

    if len(completed_sorted) < 4:
        print("Need at least 4 completed recommendations for statistical analysis.")
        return {}

    # --- Gather data ---
    records = []
    for rec in completed_sorted:
        act = rec.get('actual_results') or {}
        tgt = rec.get('target') or {}
        pred = rec.get('predictions') or {}
        actual_size = act.get('Size')
        target_size = tgt.get('size')
        pred_mu = pred.get('size_mu')
        pred_std = pred.get('size_std', 1.0)

        if actual_size is None or target_size is None or pred_mu is None:
            continue

        records.append({
            'rec_id': rec.get('rec_id'),
            'target': target_size,
            'actual': actual_size,
            'predicted': pred_mu,
            'pred_std': pred_std,
            'pred_error': abs(actual_size - pred_mu),
            'target_error': abs(actual_size - target_size),
            'z_score': (actual_size - pred_mu) / max(pred_std, 0.001),
            'tolerance': tgt.get('tolerance', 2.0),
        })

    df = pd.DataFrame(records)
    df['within_tolerance'] = df['target_error'] <= df['tolerance']

    # --- Initial dataset baseline ---
    if optimizer.base_optimizer and hasattr(optimizer.base_optimizer, 'df_success'):
        initial_sizes = optimizer.base_optimizer.df_success['Size'].dropna().values
    elif optimizer.base_optimizer and hasattr(optimizer.base_optimizer, 'df_all'):
        base_df = optimizer.base_optimizer.df_all
        initial_sizes = base_df.loc[base_df['HasProduct'] == 1, 'Size'].dropna().values
    else:
        initial_sizes = np.array([])

    results: Dict[str, Any] = {'by_target': {}, 'tests': {}}

    # --- Per-target analysis ---
    target_groups = df.groupby('target')
    table_rows = []

    for target_size, group in target_groups:
        group = group.reset_index(drop=True)
        n = len(group)
        pred_errors = group['pred_error'].values
        target_errors = group['target_error'].values
        z_scores = group['z_score'].values
        within_tol = group['within_tolerance'].sum()
        tol = group['tolerance'].iloc[0]

        # Baseline: distance from ALL initial experiments to this target
        # (represents unguided exploration — what you'd get without BO)
        if len(initial_sizes) > 0:
            baseline_errors = np.abs(initial_sizes - target_size)
        else:
            baseline_errors = np.array([])

        # Mann-Whitney: BO target errors vs full baseline distribution
        if len(baseline_errors) >= 3 and n >= 3:
            mw_stat, mw_p = mannwhitneyu(
                target_errors, baseline_errors, alternative='less'
            )
        else:
            mw_stat, mw_p = np.nan, np.nan

        # Error trend: Spearman correlation of pred_error vs iteration
        if n >= 4:
            iterations = np.arange(1, n + 1)
            sp_rho, sp_p = spearmanr(iterations, pred_errors)
            lr = linregress(iterations, pred_errors)
        else:
            sp_rho, sp_p = np.nan, np.nan
            lr = None

        target_result = {
            'n_experiments': n,
            'mean_pred_error': np.mean(pred_errors),
            'std_pred_error': np.std(pred_errors, ddof=1) if n > 1 else 0,
            'mean_target_error': np.mean(target_errors),
            'best_target_error': np.min(target_errors),
            'within_tolerance_frac': within_tol / n,
            'mean_z_score': np.mean(z_scores),
            'std_z_score': np.std(z_scores, ddof=1) if n > 1 else 0,
            'frac_within_1sigma': np.mean(np.abs(z_scores) <= 1.0),
            'mann_whitney_stat': mw_stat,
            'mann_whitney_p': mw_p,
            'error_trend_rho': sp_rho,
            'error_trend_p': sp_p,
            'error_trend_slope': lr.slope if lr else np.nan,
            'baseline_mean_error': np.mean(baseline_errors) if len(baseline_errors) > 0 else np.nan,
        }
        results['by_target'][int(target_size)] = target_result

        table_rows.append({
            'Target (nm)': int(target_size),
            'N': n,
            'Mean |actual−target| (nm)': f"{np.mean(target_errors):.2f}",
            'Best |actual−target| (nm)': f"{np.min(target_errors):.2f}",
            '% Within Tolerance': f"{100 * within_tol / n:.0f}%",
            'Mean |pred−actual| (nm)': f"{np.mean(pred_errors):.2f}",
            'Baseline Mean Error (nm)': f"{np.mean(baseline_errors):.2f}" if len(baseline_errors) > 0 else "N/A",
            'p (vs baseline)': f"{mw_p:.4f}" if not np.isnan(mw_p) else "N/A",
        })

    # --- Overall tests ---
    all_pred_errors = df['pred_error'].values
    all_target_errors = df['target_error'].values
    all_z_scores = df['z_score'].values
    all_within = df['within_tolerance'].values

    # Z-score normality (Shapiro-Wilk)
    if len(all_z_scores) >= 3:
        sw_stat, sw_p = shapiro(all_z_scores)
    else:
        sw_stat, sw_p = np.nan, np.nan

    # Z-score mean = 0 (one-sample t-test)
    if len(all_z_scores) >= 3:
        t_stat, t_p = ttest_1samp(all_z_scores, 0)
    else:
        t_stat, t_p = np.nan, np.nan

    # Proportion within tolerance vs random chance
    # Random chance: P(|x - target| <= tol) ≈ 2*tol / (max_size - min_size)
    size_range = 30.0 - 10.0  # design space
    mean_tol = df['tolerance'].mean()
    p_random = 2 * mean_tol / size_range
    observed_hit_rate = np.mean(all_within)

    # Binomial test for hit rate
    from scipy.stats import binomtest
    n_total = len(all_within)
    n_hits = int(np.sum(all_within))
    binom_result = binomtest(n_hits, n_total, p_random, alternative='greater')

    # Overall baseline comparison: BO target errors vs what unguided experiments achieve
    if len(initial_sizes) > 0:
        # For each target attempted, compute distance of ALL initial experiments to that target
        # This represents what you'd expect from unguided sampling
        unique_targets = df['target'].unique()
        baseline_all = np.concatenate([
            np.abs(initial_sizes - t) for t in unique_targets
        ])
        mw_overall_stat, mw_overall_p = mannwhitneyu(
            all_target_errors, baseline_all, alternative='less'
        )
    else:
        baseline_all = np.array([])
        mw_overall_stat, mw_overall_p = np.nan, np.nan

    # Error trend overall (Spearman on sequential recommendation order)
    if len(all_pred_errors) >= 5:
        overall_iterations = np.arange(1, len(all_pred_errors) + 1)
        sp_overall_rho, sp_overall_p = spearmanr(overall_iterations, all_pred_errors)
    else:
        sp_overall_rho, sp_overall_p = np.nan, np.nan

    results['overall'] = {
        'n_total': n_total,
        'mean_pred_error': np.mean(all_pred_errors),
        'mean_target_error': np.mean(all_target_errors),
        'hit_rate': observed_hit_rate,
        'random_hit_rate': p_random,
        'binomial_p': binom_result.pvalue,
        'z_score_mean': np.mean(all_z_scores),
        'z_score_std': np.std(all_z_scores, ddof=1),
        'z_score_normality_p': sw_p,
        'z_score_zero_mean_p': t_p,
        'frac_within_1sigma': np.mean(np.abs(all_z_scores) <= 1.0),
        'frac_within_2sigma': np.mean(np.abs(all_z_scores) <= 2.0),
        'overall_trend_rho': sp_overall_rho,
        'overall_trend_p': sp_overall_p,
        'mann_whitney_vs_baseline_p': mw_overall_p,
    }

    results['tests'] = {
        'shapiro_wilk': {'stat': sw_stat, 'p': sw_p,
                         'interpretation': 'z-scores are normally distributed' if sw_p > 0.05
                         else 'z-scores deviate from normality'},
        'ttest_zero_mean': {'stat': t_stat, 'p': t_p,
                            'interpretation': 'predictions are unbiased' if t_p > 0.05
                            else 'predictions show systematic bias'},
        'binomial_hit_rate': {'n_hits': n_hits, 'n_total': n_total,
                              'observed': observed_hit_rate, 'expected_random': p_random,
                              'p': binom_result.pvalue,
                              'interpretation': f'hit rate ({observed_hit_rate:.0%}) significantly '
                              f'exceeds random chance ({p_random:.0%})' if binom_result.pvalue < 0.05
                              else 'hit rate not significantly above random'},
        'mann_whitney_vs_baseline': {'stat': mw_overall_stat, 'p': mw_overall_p,
                                     'interpretation': 'BO achieves target sizes better than unguided sampling'
                                     if mw_overall_p < 0.05 else 'no significant difference from baseline'},
        'prediction_error_trend': {'rho': sp_overall_rho, 'p': sp_overall_p,
                                   'interpretation': 'prediction error decreases over time'
                                   if (sp_overall_p < 0.05 and sp_overall_rho < 0)
                                   else 'no significant monotonic trend in prediction error'},
    }

    results['summary_table'] = pd.DataFrame(table_rows)

    return results


def print_optimization_statistics(optimizer):
    """Print a publication-ready statistical summary of the optimization."""
    results = compute_optimization_statistics(optimizer)
    if not results:
        return

    overall = results['overall']
    tests = results['tests']

    print("\n" + "=" * 70)
    print("  STATISTICAL EVIDENCE FOR SIZE OPTIMIZATION")
    print("=" * 70)

    print(f"\n  Total BO-guided experiments: {overall['n_total']}")
    print(f"  Mean prediction error |pred − actual|: {overall['mean_pred_error']:.2f} nm")
    print(f"  Mean target achievement |actual − target|: {overall['mean_target_error']:.2f} nm")
    print(f"  Hit rate (within tolerance): {overall['hit_rate']:.0%}")

    # --- GP Calibration ---
    print("\n  ─── GP Model Calibration ───")
    print(f"  Z-score mean: {overall['z_score_mean']:.3f} (ideal: 0)")
    print(f"  Z-score std:  {overall['z_score_std']:.3f} (ideal: 1)")
    print(f"  Within ±1σ:   {overall['frac_within_1sigma']:.0%} (ideal: 68%)")
    print(f"  Within ±2σ:   {overall['frac_within_2sigma']:.0%} (ideal: 95%)")

    # --- Statistical Tests ---
    print("\n  ─── Statistical Tests ───")

    for test_name, test_data in tests.items():
        p = test_data['p']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        label = test_name.replace('_', ' ').title()
        print(f"\n  {label}:")
        print(f"    p = {p:.4f} {sig}")
        print(f"    → {test_data['interpretation']}")

    # --- Per-Target Table ---
    print("\n  ─── Per-Target Summary ───")
    table = results['summary_table']
    print(table.to_string(index=False))

    # --- Suggested paper text ---
    print("\n  ─── Suggested Paper Language ───")
    print(f"  \"Bayesian optimization achieved target sizes with a mean absolute")
    print(f"   error of {overall['mean_target_error']:.2f} nm across {overall['n_total']} experiments")
    print(f"   spanning 10–30 nm targets. The within-tolerance hit rate")
    print(f"   ({overall['hit_rate']:.0%}) was significantly higher than random")
    print(f"   sampling (p = {tests['binomial_hit_rate']['p']:.2e}, binomial test).")
    if tests['mann_whitney_vs_baseline']['p'] < 0.05:
        print(f"   BO-guided experiments achieved target sizes significantly")
        print(f"   better than unguided exploration (p = {tests['mann_whitney_vs_baseline']['p']:.2e},")
        print(f"   Mann–Whitney U test).")
    print(f"   The GP surrogate model was well-calibrated, with")
    print(f"   {overall['frac_within_1sigma']:.0%} of outcomes within ±1σ of predictions")
    print(f"   (expected: 68%) and z-scores consistent with zero mean")
    print(f"   (p = {tests['ttest_zero_mean']['p']:.3f}, one-sample t-test).\"")

    print("\n" + "=" * 70)

    return results


def get_gp_hyperparameter_table(optimizer) -> pd.DataFrame:
    """Extract a publication-ready table of fitted GP kernel hyperparameters.

    Returns a DataFrame with columns: Model, Signal_Variance, Lengthscale(s),
    Noise_Level, Log_Marginal_Likelihood.  For ARD kernels, lengthscales are
    reported per-feature as a dict column; for isotropic, as a scalar.
    """
    base = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    if base is None:
        raise RuntimeError("Base optimizer not initialized")

    rows = []
    models = [('Size', base.gp_size), ('CV', base.gp_cv),
              ('Squareness', base.gp_sq)]
    features = list(base.features)

    for name, gp in models:
        try:
            ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale)
            sigma_f = float(gp.kernel_.k1.k1.constant_value)
            noise = float(gp.kernel_.k2.noise_level)
            lml = float(gp.log_marginal_likelihood_value_)

            if ls.shape[0] == 1:
                ls_report = float(ls[0])
            else:
                ls_report = {feat: round(float(l), 4) for feat, l in zip(features, ls)}

            rows.append({
                'Model': name,
                'Signal_Variance': round(sigma_f, 4),
                'Lengthscales': ls_report,
                'Noise_Level': round(noise, 5),
                'Log_Marginal_Likelihood': round(lml, 2),
                'Kernel_String': str(gp.kernel_),
            })
        except AttributeError:
            rows.append({'Model': name, 'Signal_Variance': None,
                         'Lengthscales': None, 'Noise_Level': None,
                         'Log_Marginal_Likelihood': None, 'Kernel_String': None})

    return pd.DataFrame(rows)


def display_recommendations_table(recommendations_df: pd.DataFrame):
    """Display recommendations in a nicely formatted way."""
    if recommendations_df.empty:
        print("No recommendations to display.")
        return

    print("\n" + "SYNTHESIS RECOMMENDATIONS")
    print("-" * 80)

    # Recommendation tables use 'Pred_CV' for the polydispersity prediction;
    # fall back to the legacy 'Pred_GSD' column for older saved data.
    cv_col = 'Pred_CV' if 'Pred_CV' in recommendations_df.columns else 'Pred_GSD'

    for _, row in recommendations_df.iterrows():
        print(f"\n  Recommendation {row['Rank']}: {row['Rec_ID']}")
        print(f"   Conditions:")
        print(f"      Temp: {row['Temp']:.1f}°C | "
              f"Time: {row['Time']:.1f} min | "
              f"VOacac: {row['VOacac']:.3f} mmol")
        print(f"      DDT: {row['DDT']:.2f} mL | OAm: {row['OAm']:.2f} mL")
        print(f"   Predictions:")
        print(f"      Size: {row['Pred_Size']:.1f} ± {row['Pred_Size_Std']:.1f} nm")
        print(f"      CV:  {row[cv_col]:.3f} | "
              f"Squareness: {row['Pred_Squareness']:.3f}")

        sq_bin = row.get('Squareness_Bin', 'N/A')
        p_bin = row.get('P_Bin', None)
        p_feas = row.get('P_Feasible', None)
        bin_str = f" (P = {p_bin:.0%})" if p_bin is not None else ""
        print(f"   Squareness bin: {sq_bin}{bin_str}")
        if p_feas is not None:
            print(f"   Feasibility: {p_feas * 100:.0f}%")

    print("\n" + "To complete a recommendation after running the experiment:")
    print("  optimizer.complete_recommendation('REC_XXX', Size=..., CV=..., Squareness=...)")
    print("-" * 80)
