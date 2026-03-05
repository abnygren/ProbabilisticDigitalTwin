"""
Visualization functions for Cu₃VS₄ Bayesian Optimization

All plot_* functions live here. They only read from the optimizer — they never
modify state — so editing a plot can never break the optimization logic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from typing import Tuple, Optional
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error

from config import COLORS, RAW_BOUNDS, RAW_FACTORS


# =============================================================================
# RECOMMENDATION HISTORY
# =============================================================================

def plot_recommendation_history(optimizer, figsize: Tuple[int, int] = (14, 6)):
    """Plot recommendation timeline and size-error distribution."""
    history_df = optimizer.rec_store.get_history_df()
    if history_df.empty:
        print("No recommendations to display.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    status_colors = {'pending': COLORS['pending'], 'completed': COLORS['completed'], 'skipped': COLORS['skipped']}
    for i, row in history_df.iterrows():
        color = status_colors.get(row['status'], COLORS['neutral'])
        ax1.barh(i, 1, color=color, edgecolor='black', linewidth=0.5)
        label = f"{row['rec_id']}: Target {row['target_size']:.0f}nm"
        if row['status'] == 'completed' and 'size_error' in row and pd.notna(row.get('size_error')):
            label += f" (err: {row['size_error']:+.1f})"
        ax1.text(0.5, i, label, ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.set_yticks(range(len(history_df)))
    ax1.set_yticklabels([f"{row['timestamp'][:10]}" for _, row in history_df.iterrows()])
    ax1.set_xlabel('Recommendation')
    ax1.set_title('Recommendation Timeline')
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    legend_elements = [Patch(facecolor=COLORS['pending'], label='Pending'),
                       Patch(facecolor=COLORS['completed'], label='Completed'),
                       Patch(facecolor=COLORS['skipped'], label='Skipped')]
    ax1.legend(handles=legend_elements, loc='lower right')

    ax2 = axes[1]
    completed = history_df[history_df['status'] == 'completed']
    if len(completed) >= 2:
        errors = completed['size_error'].dropna()
        if len(errors) > 0:
            ax2.hist(errors, bins=min(10, len(errors)), color=COLORS['primary'], edgecolor='black', alpha=0.7)
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
            ax2.axvline(errors.mean(), color='orange', linestyle='-', linewidth=2,
                       label=f'Mean error: {errors.mean():.2f}')
            ax2.set_xlabel('Size Prediction Error (nm)')
            ax2.set_ylabel('Count')
            ax2.set_title('Size Prediction Error Distribution')
            ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Need ≥2 completed\nrecommendations\nfor error analysis',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Size Prediction Error Distribution')

    plt.tight_layout()
    return fig


# =============================================================================
# PARITY PLOTS
# =============================================================================

def plot_parity(optimizer, figsize: Tuple[int, int] = (12, 4)):
    """Predicted vs actual values for completed recommendations."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations for parity plots. Currently have {len(completed)}.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (prop, pred_key) in zip(axes, [('Size', 'size_mu'), ('GSD', 'gsd_mu'), ('Squareness', 'sq_mu')]):
        predicted, actual, pred_std = [], [], []
        for rec in completed:
            if rec['actual_results'] and rec['actual_results'].get(prop) is not None:
                predicted.append(rec['predictions'][pred_key])
                actual.append(rec['actual_results'][prop])
                pred_std.append(rec['predictions'].get(pred_key.replace('_mu', '_std'), 0))

        if len(predicted) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {prop}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prop)
            continue

        predicted = np.array(predicted)
        actual = np.array(actual)
        pred_std = np.array(pred_std)

        ax.errorbar(actual, predicted, yerr=1.96*pred_std, fmt='o',
                   color=COLORS['primary'], capsize=3, markersize=8, alpha=0.7)
        lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        margin = 0.1 * (lims[1] - lims[0])
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', lw=1, label='Perfect')

        if len(predicted) >= 3:
            r2 = r2_score(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}',
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Actual {prop}'); ax.set_ylabel(f'Predicted {prop}')
        ax.set_title(f'{prop} Parity'); ax.set_aspect('equal')
        ax.legend(loc='lower right')

    plt.suptitle('Prediction Accuracy (Completed Recommendations Only)', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def plot_calibration(optimizer, figsize: Tuple[int, int] = (10, 4)):
    """Z-score distribution and confidence-interval coverage."""
    from scipy.stats import norm as _norm
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 3:
        print(f"Need at least 3 completed recommendations for calibration analysis. Have {len(completed)}.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    z_scores = {'Size': [], 'GSD': [], 'Squareness': []}
    for rec in completed:
        if rec['errors']:
            for prop in ['Size', 'GSD', 'Squareness']:
                z = rec['errors'].get(f'{prop.lower()}_z_score')
                if z is not None:
                    z_scores[prop].append(z)
    all_z = [z for zs in z_scores.values() for z in zs]
    if all_z:
        ax1.hist(all_z, bins=min(15, len(all_z)), color=COLORS['primary'], edgecolor='black', alpha=0.7, density=True)
        x = np.linspace(-4, 4, 100)
        ax1.plot(x, _norm.pdf(x), 'r-', lw=2, label='Standard Normal\n(ideal calibration)')
        ax1.axvline(-1, color='orange', linestyle=':', alpha=0.7)
        ax1.axvline(1, color='orange', linestyle=':', alpha=0.7, label='±1σ bounds')
        ax1.axvline(-2, color='red', linestyle=':', alpha=0.5)
        ax1.axvline(2, color='red', linestyle=':', alpha=0.5, label='±2σ bounds')
        ax1.set_xlabel('Z-score (actual - predicted) / std')
        ax1.set_ylabel('Density')
        ax1.set_title('Prediction Z-Score Distribution')
        ax1.legend(fontsize=8)

    ax2 = axes[1]
    stats = optimizer.rec_store.get_error_statistics()
    props = ['size', 'gsd', 'squareness']
    within_1s = [stats.get(f'{p}_within_1sigma_rate', 0) * 100 for p in props]
    within_2s = [stats.get(f'{p}_within_2sigma_rate', 0) * 100 for p in props]
    x = np.arange(len(props))
    width = 0.35
    bars1 = ax2.bar(x - width/2, within_1s, width, label='Within 1σ', color=COLORS['primary'])
    bars2 = ax2.bar(x + width/2, within_2s, width, label='Within 2σ', color=COLORS['secondary'])
    ax2.axhline(68, color='blue', linestyle='--', alpha=0.5, label='Target 1σ (68%)')
    ax2.axhline(95, color='purple', linestyle='--', alpha=0.5, label='Target 2σ (95%)')
    ax2.set_ylabel('Coverage Rate (%)')
    ax2.set_title('Confidence Interval Coverage')
    ax2.set_xticks(x); ax2.set_xticklabels(['Size', 'GSD', 'Squareness'])
    ax2.set_ylim(0, 105); ax2.legend(fontsize=8, loc='lower right')
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


# =============================================================================
# ERROR LEARNING PROGRESS
# =============================================================================

def plot_error_learning_progress(optimizer, figsize: Tuple[int, int] = (12, 4)):
    """Cumulative MAE over completed recommendations for each property."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 3:
        print(f"Need at least 3 completed recommendations. Have {len(completed)}.")
        return None

    completed_sorted = sorted(completed, key=lambda x: x.get('completed_timestamp', ''))
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, prop in zip(axes, ['Size', 'GSD', 'Squareness']):
        errors, cumulative_mae = [], []
        for rec in completed_sorted:
            if rec['errors'] and rec['errors'].get(f'{prop.lower()}_error') is not None:
                errors.append(rec['errors'][f'{prop.lower()}_error'])
                cumulative_mae.append(np.mean(np.abs(errors)))
        if len(errors) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {prop}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{prop} Learning Progress')
            continue

        x = range(1, len(cumulative_mae) + 1)
        ax.plot(x, cumulative_mae, 'o-', color=COLORS['primary'], markersize=6, label='Cumulative MAE')
        if len(list(x)) >= 3:
            z = np.polyfit(list(x), cumulative_mae, 1)
            p = np.poly1d(z)
            ax.plot(list(x), p(list(x)), '--', color=COLORS['secondary'], alpha=0.7, label=f'Trend (slope: {z[0]:.3f})')
            if z[0] < 0:
                ax.text(0.95, 0.95, 'Improving', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='green')
            elif z[0] > 0.01:
                ax.text(0.95, 0.95, 'Degrading', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='red')
        ax.set_xlabel('# Completed Recommendations'); ax.set_ylabel('Cumulative MAE')
        ax.set_title(f'{prop} Learning Progress'); ax.legend(fontsize=8)

    plt.suptitle('Error Learning Progress Over Time', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# ERROR CORRECTION IMPACT
# =============================================================================

def plot_error_correction_impact(optimizer, figsize: Tuple[int, int] = (14, 4)):
    """Compare predictions with vs without error correction."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations. Have {len(completed)}.")
        return None
    if not optimizer.error_learner.is_fitted:
        print("Error correction not active yet. Need at least 5 completed recommendations.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (prop, pred_key) in zip(axes, [('Size', 'size_mu'), ('GSD', 'gsd_mu'), ('Squareness', 'sq_mu')]):
        X_list, actual_vals = [], []
        for rec in completed:
            if rec['actual_results'] and rec['actual_results'].get(prop) is not None:
                cond = rec['conditions']
                X_list.append([cond['Temp'], cond['Time'], cond['VOacac'], cond['DDT'], cond['OAm']])
                actual_vals.append(rec['actual_results'][prop])

        if len(X_list) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {prop}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prop)
            continue

        actual = np.array(actual_vals)
        feat_arrays = []
        for x in X_list:
            cond = {'Temp': x[0], 'Time': x[1], 'VOacac': x[2], 'DDT': x[3], 'OAm': x[4]}
            feat_arrays.append(optimizer._raw_to_feature_array(cond))
        X_feat = np.array(feat_arrays)

        preds_no_corr = optimizer.base_optimizer.predict(X_feat)
        pred_no_corr = preds_no_corr[pred_key]
        preds_with_corr = optimizer.predict(X_feat, apply_correction=True)
        pred_with_corr = preds_with_corr[pred_key]

        lims = [min(actual.min(), pred_no_corr.min(), pred_with_corr.min()),
                max(actual.max(), pred_no_corr.max(), pred_with_corr.max())]
        margin = 0.1 * (lims[1] - lims[0])
        lims = [lims[0] - margin, lims[1] + margin]

        ax.scatter(actual, pred_no_corr, alpha=0.6, s=60, color=COLORS['warning'], label='Without correction', marker='o')
        ax.scatter(actual, pred_with_corr, alpha=0.6, s=60, color=COLORS['success'], label='With correction', marker='s')
        ax.plot(lims, lims, 'k--', lw=1, label='Perfect')

        r2_no = r2_score(actual, pred_no_corr)
        r2_with = r2_score(actual, pred_with_corr)
        mae_no = mean_absolute_error(actual, pred_no_corr)
        mae_with = mean_absolute_error(actual, pred_with_corr)
        improvement = ((mae_no - mae_with) / mae_no) * 100

        ax.text(0.05, 0.95,
               f'No correction:\nR² = {r2_no:.3f}\nMAE = {mae_no:.3f}\n\n'
               f'With correction:\nR² = {r2_with:.3f}\nMAE = {mae_with:.3f}\n\n'
               f'Improvement: {improvement:.1f}%',
               transform=ax.transAxes, va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Actual {prop}'); ax.set_ylabel(f'Predicted {prop}')
        ax.set_title(f'{prop}: Correction Impact'); ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize=8)

    plt.suptitle('Error Correction Impact: With vs Without Self-Validation', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# TARGET ACHIEVEMENT TRACKING
# =============================================================================

def plot_target_achievement(optimizer, figsize: Tuple[int, int] = (12, 5)):
    """Track how well recommendations achieve target sizes over time."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations. Have {len(completed)}.")
        return None

    completed_sorted = sorted(completed, key=lambda x: x.get('completed_timestamp', ''))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    targets, achieved, errors, indices = [], [], [], []
    for i, rec in enumerate(completed_sorted):
        if rec['actual_results'] and rec['actual_results'].get('Size') is not None:
            targets.append(rec['target']['size'])
            achieved.append(rec['actual_results']['Size'])
            errors.append(rec['actual_results']['Size'] - rec['target']['size'])
            indices.append(i + 1)

    if len(targets) < 2:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Target Achievement Over Time')
    else:
        ax1.plot(indices, targets, 'o-', color=COLORS['primary'], markersize=8, label='Target Size', linewidth=2)
        ax1.plot(indices, achieved, 's-', color=COLORS['success'], markersize=8, label='Achieved Size', linewidth=2)
        for idx, target in zip(indices, targets):
            tol_list = [r['target']['tolerance'] for r in completed_sorted
                       if r['actual_results'] and r['actual_results'].get('Size') is not None]
            if tol_list:
                ax1.axhspan(target - tol_list[0], target + tol_list[0], alpha=0.2, color=COLORS['primary'])
        ax1.set_xlabel('# Completed Recommendation'); ax1.set_ylabel('Size (nm)')
        ax1.set_title('Target vs Achieved Size Over Time'); ax1.legend(loc='best'); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if len(errors) >= 2:
        ax2.hist(errors, bins=min(10, len(errors)), color=COLORS['primary'], edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect target')
        ax2.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean error: {np.mean(errors):.2f} nm')
        avg_tol = np.mean([r['target']['tolerance'] for r in completed_sorted
                         if r['actual_results'] and r['actual_results'].get('Size') is not None])
        ax2.axvspan(-avg_tol, avg_tol, alpha=0.2, color=COLORS['success'],
                   label=f'Target tolerance (±{avg_tol:.1f} nm)')
        ax2.set_xlabel('Error from Target (nm)'); ax2.set_ylabel('Count')
        ax2.set_title('Target Achievement Error Distribution'); ax2.legend()
        within_tol = [abs(e) <= r['target']['tolerance']
                     for r, e in zip(completed_sorted, errors)
                     if r['actual_results'] and r['actual_results'].get('Size') is not None]
        if within_tol:
            success_rate = np.mean(within_tol) * 100
            ax2.text(0.95, 0.95, f'Success rate:\n{success_rate:.0f}% within tolerance',
                    transform=ax2.transAxes, va='top', ha='right', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Target Achievement Error Distribution')

    plt.suptitle('Target Achievement Tracking', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(optimizer, figsize: Tuple[int, int] = (10, 6)):
    """Visualize feature importance from GP sensitivity analysis.

    Uses gradient-based sensitivity (mean |∂f/∂x_j| over training data) which
    gives meaningful per-feature importance even with isotropic kernels.
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    try:
        importance_df = optimizer.base_optimizer.get_feature_importance()
    except (AttributeError, ValueError):
        print("Could not compute feature importance from GP models.")
        return None
    if importance_df.empty:
        print("No feature importance data available.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, prop in zip(axes, ['Size', 'GSD', 'Squareness']):
        prop_data = importance_df[importance_df['Model'] == prop].copy()
        if prop_data.empty:
            ax.text(0.5, 0.5, f'No data\nfor {prop}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prop); continue
        prop_data = prop_data.sort_values('Importance', ascending=True)
        y_pos = np.arange(len(prop_data))
        bars = ax.barh(y_pos, prop_data['Importance'], color=COLORS['primary'], alpha=0.7)
        for i, (idx, row) in enumerate(prop_data.iterrows()):
            if row['Feature'] in RAW_FACTORS:
                bars[i].set_color(COLORS['primary'])
            elif 'ratio' in row['Feature'].lower() or 'conc' in row['Feature'].lower():
                bars[i].set_color(COLORS['secondary'])
            else:
                bars[i].set_color(COLORS['tertiary'])
        ax.set_yticks(y_pos); ax.set_yticklabels(prop_data['Feature'], fontsize=9)
        ax.set_xlabel('Relative Importance'); ax.set_title(f'{prop} Feature Importance')
        ax.invert_yaxis()
        for i, (idx, row) in enumerate(prop_data.iterrows()):
            ax.text(row['Importance'], i, f" {row['Importance']:.0%}", va='center', fontsize=8)

    plt.suptitle('Feature Importance (GP Sensitivity Analysis)', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# CLASSIFIER CALIBRATION RELIABILITY DIAGRAMS
# =============================================================================

def plot_classifier_calibration(optimizer, figsize: Tuple[int, int] = (12, 4)):
    """Plot reliability diagrams for classifier calibration."""
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    cal_results = optimizer.get_classifier_calibration(verbose=False)
    if not cal_results:
        print("No classifier calibration data available.")
        return None

    n_classifiers = len(cal_results)
    fig, axes = plt.subplots(1, n_classifiers, figsize=figsize)
    if n_classifiers == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, cal_results.items()):
        bins = metrics['calibration_bins']
        obs = np.array(bins['observed_frequency'])
        pred = np.array(bins['predicted_frequency'])
        counts = np.array(bins['bin_counts'])
        mask = counts > 0
        if mask.sum() > 0:
            sizes = 50 + 200 * (counts[mask] / counts[mask].max())
            ax.scatter(pred[mask], obs[mask], s=sizes, alpha=0.7,
                      color=COLORS['primary'], edgecolor='black', linewidth=0.5)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        ax.text(0.05, 0.95,
               f"Brier: {metrics['brier_score']:.4f}\nECE: {metrics['ece']:.4f}\n"
               f"Class balance: {metrics['class_balance']*100:.0f}%",
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Observed Frequency')
        ax.set_title(f'{name} Calibration'); ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Classifier Reliability Diagrams', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# COLLINEARITY HEATMAP
# =============================================================================

def plot_collinearity_heatmap(optimizer, figsize: Tuple[int, int] = (8, 6)):
    """Plot correlation heatmap for collinearity visualization."""
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    diag = optimizer.get_collinearity_diagnostics(verbose=False)
    corr_df = diag['correlation_matrix']

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Correlation')
    features = corr_df.columns.tolist()
    ax.set_xticks(range(len(features))); ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(features, fontsize=9)
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr_df.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    ax.set_title(f'Feature Correlation Matrix ({optimizer.base_optimizer.feature_mode} mode)')
    if diag['problematic_pairs']:
        print("\n⚠️ Highly correlated pairs (|r| > 0.8):")
        for f1, f2, r in diag['problematic_pairs']:
            print(f"   {f1} ↔ {f2}: r = {r:.3f}")
    plt.tight_layout()
    return fig


# =============================================================================
# DATASET QUALITY DASHBOARD
# =============================================================================

def plot_dataset_quality_dashboard(optimizer, figsize: Tuple[int, int] = (14, 10), save_path: Path = None):
    """Comprehensive 4-panel dataset quality dashboard for presentations."""
    df_all = optimizer.exp_store.get_all()
    df_success = optimizer.exp_store.get_training_data()
    if df_all.empty:
        print("No experiments available for dashboard.")
        return None

    df_all = df_all.copy()
    df_all['Success'] = df_all['HasProduct'].map({1: 'Success', 0: 'Failed'})

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Design Space Coverage
    ax1 = fig.add_subplot(gs[0, 0])
    if 'Temp' in df_all.columns and 'Time' in df_all.columns:
        success_mask = df_all['HasProduct'] == 1
        fail_mask = df_all['HasProduct'] == 0
        ax1.scatter(df_all.loc[fail_mask, 'Temp'], df_all.loc[fail_mask, 'Time'],
                   c=COLORS['warning'], s=80, alpha=0.6, label='Failed', edgecolor='black', linewidth=0.5)
        ax1.scatter(df_all.loc[success_mask, 'Temp'], df_all.loc[success_mask, 'Time'],
                   c=COLORS['success'], s=80, alpha=0.7, label='Success', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Temperature (°C)'); ax1.set_ylabel('Time (min)')
        ax1.set_title('Design Space Coverage: Temp vs Time')
        temp_bounds = RAW_BOUNDS.get('Temp', (260, 310))
        time_bounds = RAW_BOUNDS.get('Time', (8, 90))
        rect = Rectangle((temp_bounds[0], time_bounds[0]),
                         temp_bounds[1] - temp_bounds[0], time_bounds[1] - time_bounds[0],
                         fill=False, edgecolor=COLORS['primary'], linestyle='--', linewidth=1.5, label='Search bounds')
        ax1.add_patch(rect); ax1.legend(loc='upper right')

    # Panel 2: Outcome Distributions
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_success.empty:
        for (offset, col, color, label) in [
            (0.7, 'Size', COLORS['primary'], 'Size'),
            (0.35, 'GSD', COLORS['secondary'], 'GSD'),
            (0.0, 'Squareness', COLORS['tertiary'], 'Squareness'),
        ]:
            ax_sub = ax2.inset_axes([0, offset, 1, 0.25])
            ax_sub.hist(df_success[col], bins=12, color=color, edgecolor='black', alpha=0.7)
            ax_sub.set_ylabel('Count', fontsize=8)
            if col != 'Size':
                ax_sub.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_sub.set_title(f'{col}: {df_success[col].min():.2f}–{df_success[col].max():.2f}', fontsize=9, loc='left')
            ax_sub.tick_params(axis='both', labelsize=7)
        ax2.set_axis_off()
        ax2.set_title('Outcome Property Distributions', fontsize=11, fontweight='bold', y=1.02)
    else:
        ax2.text(0.5, 0.5, 'No successful experiments', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Outcome Property Distributions')

    # Panel 3: Dataset Composition
    ax3 = fig.add_subplot(gs[1, 0])
    n_total = len(df_all); n_success = len(df_success); n_failed = n_total - n_success
    success_rate = (n_success / n_total * 100) if n_total > 0 else 0
    categories = ['Total\nExperiments', 'Successful\n(HasProduct=1)', 'Failed\n(No Product)']
    counts = [n_total, n_success, n_failed]
    colors_bars = [COLORS['primary'], COLORS['success'], COLORS['warning']]
    bars = ax3.bar(categories, counts, color=colors_bars, edgecolor='black', alpha=0.8)
    for bar, count in zip(bars, counts):
        ax3.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Experiments')
    ax3.set_title(f'Dataset Composition (Success Rate: {success_rate:.0f}%)')
    ax3.set_ylim(0, max(counts) * 1.15)
    n_features_active = len(optimizer.base_optimizer.features) if optimizer.base_optimizer else 6
    ax3.text(0.98, 0.95,
            f'Samples per feature:\n  Raw: {n_success/5:.1f}\n  Synthesis: {n_success/7:.1f}\n  Active: {n_success/n_features_active:.1f}\n  (recommend ≥10)',
            transform=ax3.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Panel 4: LOO-CV Performance by Mode
    ax4 = fig.add_subplot(gs[1, 1])
    try:
        comparison_df = optimizer.compare_feature_modes(modes=['raw', 'chemical', 'hybrid', 'synthesis'], verbose=False)
        if not comparison_df.empty:
            pivot = comparison_df.pivot(index='Property', columns='Mode', values='R2')
            x = np.arange(len(pivot.index)); n_modes = 4; width = 0.8 / n_modes
            mode_colors = {'raw': COLORS['primary'], 'chemical': COLORS['secondary'],
                          'hybrid': COLORS['tertiary'], 'synthesis': '#2ecc71'}
            mode_labels = {'raw': 'Raw', 'chemical': 'Chemical', 'hybrid': 'Hybrid', 'synthesis': 'Synthesis'}
            for i, mode in enumerate(['raw', 'chemical', 'hybrid', 'synthesis']):
                if mode in pivot.columns:
                    ax4.bar(x + i*width, pivot[mode], width, label=mode_labels[mode], color=mode_colors[mode], edgecolor='black', alpha=0.8)
            ax4.axhline(0, color='black', linewidth=0.8)
            ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Target R² > 0.5')
            ax4.set_ylabel('LOO-CV R²'); ax4.set_title('Model Predictive Performance by Feature Mode')
            ax4.set_xticks(x + width*1.5); ax4.set_xticklabels(pivot.index)
            ax4.legend(loc='lower right', fontsize=7)
            ax4.set_ylim(min(-0.5, pivot.min().min()-0.1), max(1.0, pivot.max().max()+0.1))
        else:
            ax4.text(0.5, 0.5, 'Run compare_feature_modes() first', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Predictive Performance')
    except Exception as e:
        ax4.text(0.5, 0.5, f'Could not compute\nmodel comparison:\n{str(e)[:50]}',
                ha='center', va='center', transform=ax4.transAxes, fontsize=9)
        ax4.set_title('Model Predictive Performance')

    fig.suptitle('Dataset Quality Dashboard for Cu₃VS₄ Bayesian Optimization', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Dashboard saved to {save_path}")
    return fig


# =============================================================================
# LOO-CV RESIDUALS
# =============================================================================

def plot_loo_residuals(optimizer, figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
    """LOO-CV parity and residuals: assess model fit and homoscedasticity."""
    if optimizer.base_optimizer is None:
        return None
    base = optimizer.base_optimizer
    if not base.metrics or 'y_pred' not in base.metrics.get('Size', {}):
        optimizer.validate_models()
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for row, (prop, _) in enumerate([('Size', 'size_mu'), ('GSD', 'gsd_mu'), ('Squareness', 'sq_mu')]):
        if prop not in base.metrics or 'y_pred' not in base.metrics[prop]:
            continue
        y_actual = base.df_cubic[prop].values
        y_pred = base.metrics[prop]['y_pred']
        residual = y_actual - y_pred

        ax_par = axes[row, 0]
        ax_par.scatter(y_actual, y_pred, c=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
        lim_lo = min(y_actual.min(), y_pred.min()); lim_hi = max(y_actual.max(), y_pred.max())
        ax_par.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', label='Perfect')
        ax_par.set_xlabel(f'Actual {prop}'); ax_par.set_ylabel(f'LOO Predicted {prop}')
        ax_par.set_title(f'{prop}: Parity (R²={base.metrics[prop]["r2"]:.3f})'); ax_par.legend(loc='upper left')

        ax_res = axes[row, 1]
        ax_res.scatter(y_pred, residual, c=COLORS['secondary'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_res.axhline(0, color='black', linestyle='--')
        ax_res.set_xlabel(f'LOO Predicted {prop}'); ax_res.set_ylabel('Residual')
        ax_res.set_title(f'{prop}: Residuals vs Predicted')

    plt.suptitle('LOO-CV: Parity and Residuals by Property', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# PROPERTY CORRELATIONS
# =============================================================================

def plot_property_correlations(optimizer, figsize: Tuple[int, int] = (10, 4)) -> Optional[plt.Figure]:
    """Pairwise correlations of outcome properties (Size, GSD, Squareness)."""
    df_success = optimizer.exp_store.get_training_data()
    if df_success.empty or len(df_success) < 3:
        return None
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    pairs = [('Size', 'GSD'), ('Size', 'Squareness'), ('GSD', 'Squareness')]
    for ax, (xcol, ycol) in zip(axes, pairs):
        if 'PhasePure' in df_success.columns:
            for val, c in [(1, COLORS['success']), (0, COLORS['warning'])]:
                mask = df_success['PhasePure'] == val
                if mask.any():
                    ax.scatter(df_success.loc[mask, xcol], df_success.loc[mask, ycol],
                               c=c, alpha=0.7, label='Phase pure' if val == 1 else 'Impure', edgecolor='black', linewidth=0.5)
        else:
            ax.scatter(df_success[xcol], df_success[ycol], c=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(f'{xcol} vs {ycol}')
        if 'PhasePure' in df_success.columns:
            ax.legend(loc='best', fontsize=8)
    plt.suptitle('Outcome Property Correlations (Successful Experiments)', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# ACQUISITION SLICE
# =============================================================================

def plot_acquisition_slice(
    optimizer,
    target_size: float = 20.0,
    size_tol: float = 2.5,
    param1: str = 'Temp',
    param2: Optional[str] = None,
    n_grid: int = 30,
    figsize: Tuple[int, int] = (8, 6)
) -> Optional[plt.Figure]:
    """2D slice of acquisition function over two parameters (others at median)."""
    if optimizer.base_optimizer is None:
        return None
    base = optimizer.base_optimizer
    features = base.features
    if param2 is None:
        param2 = 'log_Time' if 'log_Time' in features else 'Time'
    if param1 not in features or param2 not in features:
        return None
    bounds = base.bounds
    medians = base.df_success[features].median()
    i1, i2 = features.index(param1), features.index(param2)
    v1 = np.linspace(bounds[param1][0], bounds[param1][1], n_grid)
    v2 = np.linspace(bounds[param2][0], bounds[param2][1], n_grid)
    V1, V2 = np.meshgrid(v1, v2)
    n_pts = V1.size
    X = np.tile(medians.values, (n_pts, 1))
    X[:, i1] = V1.ravel(); X[:, i2] = V2.ravel()
    acq = base.acquisition(X, target_size, size_tol)
    Z = acq['total'].reshape(V1.shape)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(V1, V2, Z, shading='auto', cmap='viridis')
    ax.set_xlabel(param1); ax.set_ylabel(param2)
    ax.set_title(f'Acquisition (target size {target_size}±{size_tol} nm)')
    plt.colorbar(im, ax=ax, label='Acquisition value')
    ax.scatter(base.df_success[param1], base.df_success[param2],
               c='white', s=20, alpha=0.8, edgecolor='black', linewidth=0.5, label='Training data')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    return fig


# =============================================================================
# RECOMMENDATION REGRET
# =============================================================================

def plot_recommendation_regret(optimizer, figsize: Tuple[int, int] = (10, 4)) -> Optional[plt.Figure]:
    """Simple regret over completed recommendations: |actual - target| per property."""
    completed = [r for r in optimizer.rec_store.get_all() if r.get('status') == 'completed' and r.get('actual_results')]
    if len(completed) < 2:
        return None
    df_success = optimizer.exp_store.get_training_data()
    gsd_best = df_success['GSD'].min() if not df_success.empty else 1.0
    sq_best = df_success['Squareness'].max() if not df_success.empty else 1.0
    completed_sorted = sorted(completed, key=lambda r: r.get('created_at', ''))
    indices = np.arange(1, len(completed_sorted) + 1)
    err_size, err_gsd, err_sq = [], [], []
    for r in completed_sorted:
        act = r.get('actual_results') or {}
        tgt = r.get('target') or {}
        err_size.append(abs(act.get('Size', np.nan) - tgt.get('size', np.nan)))
        err_gsd.append(act.get('GSD', np.nan) - gsd_best if act.get('GSD') is not None else np.nan)
        err_sq.append(sq_best - act.get('Squareness', np.nan) if act.get('Squareness') is not None else np.nan)
    err_size, err_gsd, err_sq = np.array(err_size), np.array(err_gsd), np.array(err_sq)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, vals, label, color in [
        (axes[0], err_size, '|Size − target| (nm)', COLORS['primary']),
        (axes[1], err_gsd, 'GSD − best (minimize)', COLORS['secondary']),
        (axes[2], err_sq, 'best − Squareness (maximize)', COLORS['tertiary']),
    ]:
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(indices[valid], np.maximum(vals[valid], 0), 'o-', color=color, markersize=6)
        ax.set_xlabel('Recommendation index'); ax.set_ylabel(label)
        ax.set_title(label.split('(')[0].strip())
    plt.suptitle('Simple Regret Over Completed Recommendations', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
