"""
Diagnostics and model-quality assessment for Cu₃VS₄ Bayesian Optimization

Includes:
- LOO-CV with proper re-fitting
- Feature-mode comparison
- Extrapolation detection
- Collinearity diagnostics (VIF-based)
- Classifier calibration metrics (Brier, ECE)
- Optimization convergence summary
- Model assessment reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy.stats import norm
from scipy.spatial.distance import cdist

from config import RAW_FACTORS
from features import add_chemical_features, calculate_vif, CHEM_FEATURES, HYBRID_FEATURES


# =============================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
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
# FEATURE-MODE COMPARISON
# =============================================================================

def compare_feature_modes(
    df: pd.DataFrame,
    modes: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare LOO-CV performance across different feature representations.

    Helps justify the choice of feature mode based on actual predictive performance.
    """
    # Import here to avoid circular imports
    from optimizer import Cu3VS4Optimizer

    if modes is None:
        modes = ['raw', 'chemical', 'hybrid', 'smart_hybrid']

    results = []

    for mode in modes:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating {mode.upper()} feature mode...")
            print(f"{'='*50}")

        try:
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
        print(f"\n✓ Recommended mode based on mean R²: {best_mode} ({mean_r2[best_mode]:.3f})")

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

    Returns dict with is_extrapolation, distances, n_extrapolating, warnings.
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
            f"⚠️ {n_extrap}/{len(X_new)} points are in extrapolation regions "
            f"(distance > {threshold} from nearest training point)"
        )
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
# COLLINEARITY DIAGNOSTICS
# =============================================================================

def diagnose_collinearity(
    df: pd.DataFrame,
    feature_mode: str = 'hybrid',
    verbose: bool = True,
    feature_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive collinearity diagnostics for a given feature mode.

    Returns dict with VIF results, correlation matrix, and recommendations.
    """
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

    df_work = df.copy()
    needs_chem = feature_mode in ['chemical', 'hybrid', 'smart_hybrid'] or (
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
                "Consider using 'smart_hybrid' mode to auto-remove collinear features."
            )
    if feature_mode == 'smart_hybrid':
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
# CLASSIFIER CALIBRATION
# =============================================================================

def classifier_calibration_metrics(
    clf,
    X: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics for a probabilistic classifier.
    Returns dict with brier_score, log_loss, ece, calibration_bins, interpretation.
    """
    from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

    y_prob = clf.predict_proba(X)[:, 1]
    y_true = np.asarray(y_true)

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


def evaluate_all_classifiers(optimizer, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Evaluate calibration of all classifiers in a Cu3VS4Optimizer."""
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
        y_true = optimizer.df_all[name].values
        metrics = classifier_calibration_metrics(clf, optimizer.X_all_scaled, y_true)
        metrics['in_sample'] = True
        results[name] = metrics
        if verbose:
            print(f"\n{name} Classifier (in-sample calibration):")
            print(f"  Class balance: {metrics['class_balance']*100:.1f}% positive")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  {metrics['interpretation']}")

    if verbose and results:
        print(f"\n{'='*50}")
        print("CALIBRATION SUMMARY (IN-SAMPLE)")
        print(f"{'='*50}")
        print("Well-calibrated: Brier < 0.2, ECE < 0.1")
        print("Note: All metrics are in-sample (same data used to fit classifiers).")

    return results


# =============================================================================
# CONVERGENCE SUMMARY
# =============================================================================

def get_optimization_convergence_summary(
    optimizer,
    last_n: Optional[int] = None,
    size_band_nm: float = 5.0
) -> Dict[str, Any]:
    """
    Summarize optimization progress for judging convergence, especially across target sizes.

    Returns dict with overall, by_target_band, recent, and interpretation.
    """
    completed = optimizer.rec_store.get_completed()
    completed_sorted = sorted(completed, key=lambda r: r.get('completed_timestamp', r.get('created_at', '')))

    def _metrics(recs):
        if not recs:
            return {'n': 0, 'success_rate': np.nan, 'mean_abs_error_nm': np.nan, 'mean_abs_error_pct': np.nan}
        errors_nm, errors_pct, within_tol = [], [], []
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
            errors_pct.append(100 * e_nm / t if t > 0 else 0)
            within_tol.append(e_nm <= tol)
        n = len(errors_nm)
        if n == 0:
            return {'n': 0, 'success_rate': np.nan, 'mean_abs_error_nm': np.nan, 'mean_abs_error_pct': np.nan}
        return {
            'n': n,
            'success_rate': np.mean(within_tol) * 100,
            'mean_abs_error_nm': np.mean(errors_nm),
            'mean_abs_error_pct': np.mean(errors_pct),
        }

    overall = _metrics(completed_sorted)
    out: Dict[str, Any] = {'overall': overall, 'by_target_band': {}, 'interpretation': []}

    bands: Dict[Tuple[float, float], List[Dict]] = {}
    for r in completed_sorted:
        t = (r.get('target') or {}).get('size')
        if t is None:
            continue
        low = size_band_nm * (t // size_band_nm)
        high = low + size_band_nm
        bands.setdefault((low, high), []).append(r)
    for (low, high), recs in sorted(bands.items()):
        out['by_target_band'][f"{low:.0f}-{high:.0f}"] = _metrics(recs)

    if last_n is not None and last_n > 0 and len(completed_sorted) >= last_n:
        recent = completed_sorted[-last_n:]
        out['recent'] = _metrics(recent)
        if overall['n'] >= last_n:
            if out['recent']['mean_abs_error_nm'] >= overall['mean_abs_error_nm'] * 0.9:
                out['interpretation'].append(
                    f"Last {last_n} recs have similar or worse error than overall → possible plateau."
                )
            if out['recent']['success_rate'] >= 80:
                out['interpretation'].append(
                    f"Recent success rate {out['recent']['success_rate']:.0f}% → optimization may be converged for those sizes."
                )

    if overall['n'] >= 2:
        if overall['success_rate'] >= 80:
            out['interpretation'].append(f"Overall {overall['success_rate']:.0f}% within tolerance → strong performance.")
        if overall['mean_abs_error_nm'] <= 2.0:
            out['interpretation'].append(f"Mean |error| = {overall['mean_abs_error_nm']:.2f} nm → good target hitting.")

    return out


def print_optimization_convergence_summary(optimizer, last_n: Optional[int] = 10, size_band_nm: float = 5.0):
    """Print a short convergence report."""
    s = get_optimization_convergence_summary(optimizer, last_n=last_n, size_band_nm=size_band_nm)
    overall = s['overall']
    print("\n" + "=" * 60)
    print("OPTIMIZATION CONVERGENCE SUMMARY")
    print("=" * 60)
    print(f"\n  Completed recommendations: {overall['n']}")
    if overall['n'] < 2:
        print("  → Need at least 2 completed recs to assess convergence.")
        print("=" * 60)
        return
    print(f"  Success rate (within tolerance): {overall['success_rate']:.0f}%")
    print(f"  Mean |error| from target:        {overall['mean_abs_error_nm']:.2f} nm ({overall['mean_abs_error_pct']:.1f}%)")
    if s.get('by_target_band'):
        print("\n  By target size band (nm):")
        for band, m in sorted(s['by_target_band'].items(), key=lambda x: float(x[0].split('-')[0])):
            if m['n'] > 0:
                print(f"    {band:>8}: n={m['n']}, success={m['success_rate']:.0f}%, mean |err|={m['mean_abs_error_nm']:.2f} nm")
    if s.get('recent'):
        r = s['recent']
        print(f"\n  Last {last_n} recommendations:")
        print(f"    success={r['success_rate']:.0f}%, mean |err|={r['mean_abs_error_nm']:.2f} nm")
    if s.get('interpretation'):
        print("\n  Signals:")
        for line in s['interpretation']:
            print(f"    • {line}")
    print("\n" + "=" * 60)


# =============================================================================
# MODEL ASSESSMENT REPORT
# =============================================================================

def print_model_assessment(optimizer):
    """Print detailed model self-assessment report."""
    stats = optimizer.get_model_assessment()

    print("\n" + "=" * 70)
    print("MODEL SELF-ASSESSMENT REPORT")
    print("=" * 70)

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
        for prop in ['size', 'gsd', 'squareness']:
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

    print("\n" + "=" * 70)


def display_recommendations_table(recommendations_df: pd.DataFrame):
    """Display recommendations in a nicely formatted way."""
    if recommendations_df.empty:
        print("No recommendations to display.")
        return

    print("\n" + "=" * 80)
    print("SYNTHESIS RECOMMENDATIONS")
    print("=" * 80)

    for _, row in recommendations_df.iterrows():
        print(f"\n🔬 Recommendation {row['Rank']}: {row['Rec_ID']}")
        print(f"   Conditions:")
        print(f"      Temp: {row['Temp']:.1f}°C | Time: {row['Time']:.1f} min | VOacac: {row['VOacac']:.3f} mmol")
        print(f"      DDT: {row['DDT']:.2f} mL | OAm: {row['OAm']:.2f} mL")
        print(f"   Predictions:")
        print(f"      Size: {row['Pred_Size']:.1f} ± {row['Pred_Size_Std']:.1f} nm")
        print(f"      GSD: {row['Pred_GSD']:.3f} | Squareness: {row['Pred_Squareness']:.3f}")
        print(f"   Feasibility: {row['P_Feasible']*100:.0f}%")

    print("\n" + "-" * 80)
    print("To complete a recommendation after running the experiment:")
    print("  optimizer.complete_recommendation('REC_XXX', Size=..., GSD=..., Squareness=...)")
    print("=" * 80)
