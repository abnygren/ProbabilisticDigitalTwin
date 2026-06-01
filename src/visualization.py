"""Plotting helpers for the Cu3MS4 BO workflow.

Every ``plot_*`` function in here is read-only with respect to the optimizer
(it inspects state but never mutates it), so editing a plot cannot break the
optimization logic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize, to_rgba
from typing import Tuple, Optional
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error

from config import COLORS, RAW_BOUNDS, RAW_FACTORS, SYNTHESIS_FEATURES


_ANN_BOX = dict(boxstyle='round,pad=0.4', facecolor='white',
                edgecolor='#CCCCCC', alpha=0.95, linewidth=0.8)
_LEGEND_FONT_SIZE = 12
_STATS_FONT_SIZE = 9


def _style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply consistent presentation styling to an axes object."""
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, labelpad=6)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.grid(True, color='#EEEEEE', linewidth=0.8, zorder=0)
    ax.set_facecolor('white')


def _styled_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=_LEGEND_FONT_SIZE, frameon=True, framealpha=0.95,
                  edgecolor='#CCCCCC', **kwargs)


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=17, fontweight='bold', y=1.02)


# Squareness-bin helpers shared across the classification / probability plots.

_BIN_NAMES = ['Multipod', 'Highly cubic', 'Poorly cubic']
_BIN_KEYS = ['multipod', 'highly_cubic', 'poorly_cubic']
_BIN_MARKERS = ['X', 'o', 's']


def _bin_colors():
    return [COLORS['warning'], COLORS['success'], COLORS['tertiary']]


def _get_is_cubic(df):
    """Resolve an IsCubic boolean Series from a DataFrame."""
    if 'IsCubic' in df.columns:
        return df['IsCubic']
    if 'Polymorph' in df.columns:
        return (df['Polymorph'] == 'cubic').astype(int)
    return pd.Series(1, index=df.index)


def _get_bin_masks(df, is_cubic, threshold):
    """Return [multipod_mask, highly_cubic_mask, poorly_cubic_mask]."""
    return [
        is_cubic == 0,
        (is_cubic == 1) & (df['Squareness'] >= threshold),
        (is_cubic == 1) & (df['Squareness'] < threshold),
    ]


def _bin_marker_legend_handles():
    """Shared legend handles for squareness-bin scatter markers."""
    from matplotlib.lines import Line2D
    colors = _bin_colors()
    return [
        Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
               markeredgecolor='black', markersize=9, label=f'{n} (data)')
        for m, c, n in zip(_BIN_MARKERS, colors, _BIN_NAMES)
    ]


# RECOMMENDATION HISTORY
# ------------------------------------------------------------------------------

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
        ax2.text(0.5, 0.5, 'Need $\\geq$2 completed\nrecommendations\nfor error analysis',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Size Prediction Error Distribution')

    plt.tight_layout()
    return fig


# PARITY PLOTS
# ------------------------------------------------------------------------------

def plot_parity(optimizer, figsize: Tuple[int, int] = (12, 4)):
    """Predicted vs actual values for completed recommendations."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations for parity plots. Currently have {len(completed)}.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (prop, pred_key) in zip(axes, [('Size', 'size_mu'), ('CV', 'cv_mu'), ('Squareness', 'sq_mu')]):
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


# CALIBRATION ANALYSIS
# ------------------------------------------------------------------------------

def plot_calibration(optimizer, figsize: Tuple[int, int] = (10, 4)):
    """Z-score distribution and confidence-interval coverage."""
    from scipy.stats import norm as _norm
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 3:
        print(f"Need at least 3 completed recommendations for calibration analysis. Have {len(completed)}.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    z_scores = {'Size': [], 'CV': [], 'Squareness': []}
    for rec in completed:
        if rec['errors']:
            for prop in ['Size', 'CV', 'Squareness']:
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
    props = ['size', 'cv', 'squareness']
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
    ax2.set_xticks(x); ax2.set_xticklabels(['Size', 'CV', 'Squareness'])
    ax2.set_ylim(0, 105); ax2.legend(fontsize=8, loc='lower right')
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


# ERROR LEARNING PROGRESS
# ------------------------------------------------------------------------------

def plot_error_learning_progress(optimizer, figsize: Tuple[int, int] = (13, 4)):
    """Cumulative MAE over completed recommendations for each property."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 3:
        print(f"Need at least 3 completed recommendations. Have {len(completed)}.")
        return None

    completed_sorted = sorted(completed, key=lambda x: x.get('completed_timestamp', ''))
    prop_colors = {'Size': COLORS['primary'], 'CV': COLORS['secondary'], 'Squareness': COLORS['tertiary']}
    prop_units  = {'Size': 'nm', 'CV': '', 'Squareness': ''}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    for ax, prop in zip(axes, ['Size', 'CV', 'Squareness']):
        color = prop_colors[prop]
        errors, cumulative_mae = [], []
        for rec in completed_sorted:
            if rec['errors'] and rec['errors'].get(f'{prop.lower()}_error') is not None:
                errors.append(rec['errors'][f'{prop.lower()}_error'])
                cumulative_mae.append(np.mean(np.abs(errors)))

        unit = f' ({prop_units[prop]})' if prop_units[prop] else ''
        _style_ax(ax, title=f'{prop} Prediction Error',
                  xlabel='Completed Recommendations',
                  ylabel=f'Cumulative MAE{unit}')

        if len(errors) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {prop}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            continue

        x = list(range(1, len(cumulative_mae) + 1))
        ax.fill_between(x, cumulative_mae, alpha=0.15, color=color)
        ax.plot(x, cumulative_mae, 'o-', color=color, markersize=7,
                linewidth=2, label='Cumulative MAE', zorder=3)

        if len(x) >= 3:
            z = np.polyfit(x, cumulative_mae, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--', color='#555555', linewidth=1.5,
                    alpha=0.8, label=f'Trend ({z[0]:+.3f}/rec)')
            improving = z[0] < 0
            badge_color = '#2ecc71' if improving else '#e74c3c'
            badge_text  = '↓ Improving' if improving else '↑ Degrading'
            ax.text(0.97, 0.97, badge_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, fontweight='bold',
                    color=badge_color)

        _styled_legend(ax, loc='best')

    _suptitle(fig, 'Error Learning Progress Over Completed Recommendations')
    fig.tight_layout(pad=1.5)
    return fig


# ERROR CORRECTION IMPACT
# ------------------------------------------------------------------------------

def plot_error_correction_impact(optimizer, figsize: Tuple[int, int] = (14, 4.5)):
    """Compare stored prediction snapshots (at recommendation time) against actuals.

    Uses the predictions that were actually stored when each recommendation was
    made — not re-predictions from the current retrained model.  The current
    model has been trained on the very experiments being evaluated, so
    re-predicting produces near-perfect in-sample fits that make any additive
    correction look bad by comparison.  Stored snapshots are immune to that
    data-leakage artefact.

    Points are split by whether error correction was active at recommendation
    time.  If only one group exists the comparison is still shown as a single
    series so calibration accuracy remains visible.
    """
    completed = optimizer.rec_store.get_completed()
    ann_box = dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#CCCCCC', alpha=0.95, linewidth=0.8)
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations. Have {len(completed)}.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    for ax, (prop, pred_key, actual_key) in zip(
        axes,
        [('Size', 'size_mu', 'Size'), ('CV', 'cv_mu', 'CV'), ('Squareness', 'sq_mu', 'Squareness')],
    ):
        _style_ax(ax, title=f'{prop}', xlabel=f'Actual {prop}', ylabel=f'Predicted {prop}')

        # Collect stored snapshot predictions split by correction flag
        no_corr_actual, no_corr_pred = [], []
        with_corr_actual, with_corr_pred = [], []

        for rec in completed:
            if rec['actual_results'] is None:
                continue
            actual_val = rec['actual_results'].get(actual_key)
            pred_val   = rec['predictions'].get(pred_key)
            if actual_val is None or pred_val is None:
                continue
            was_corrected = rec['predictions'].get('correction_applied', False)
            if was_corrected:
                with_corr_actual.append(actual_val)
                with_corr_pred.append(pred_val)
            else:
                no_corr_actual.append(actual_val)
                no_corr_pred.append(pred_val)

        all_actual = no_corr_actual + with_corr_actual
        all_pred   = no_corr_pred   + with_corr_pred

        if len(all_actual) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {prop}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            continue

        all_vals = np.array(all_actual + all_pred)
        margin = 0.08 * (all_vals.max() - all_vals.min())
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, color='#444444', lw=1.5, linestyle='--',
                label='Perfect prediction', zorder=2)

        has_both = len(no_corr_actual) >= 1 and len(with_corr_actual) >= 1

        if no_corr_actual:
            ax.scatter(no_corr_actual, no_corr_pred, alpha=0.75, s=70,
                       color=COLORS['warning'], marker='o', edgecolor='white',
                       linewidth=0.8, zorder=3,
                       label='Without correction' if has_both else 'Stored prediction')
        if with_corr_actual:
            ax.scatter(with_corr_actual, with_corr_pred, alpha=0.75, s=70,
                       color=COLORS['success'] if has_both else COLORS['warning'],
                       marker='s' if has_both else 'o',
                       edgecolor='white', linewidth=0.8, zorder=4,
                       label='With correction' if has_both else 'Base prediction (stored)')

        # Stats annotation
        lines = []
        if has_both:
            for label, act, pred in [
                ('Without', no_corr_actual, no_corr_pred),
                ('With   ', with_corr_actual, with_corr_pred),
            ]:
                r2  = r2_score(act, pred) if len(act) >= 2 else float('nan')
                mae = mean_absolute_error(act, pred)
                lines.append(f'{label}: R²={r2:.2f}  MAE={mae:.3f}')
            if len(no_corr_actual) >= 2 and len(with_corr_actual) >= 2:
                mae_no   = mean_absolute_error(no_corr_actual, no_corr_pred)
                mae_with = mean_absolute_error(with_corr_actual, with_corr_pred)
                improv   = ((mae_no - mae_with) / (mae_no + 1e-9)) * 100
                lines.append(f'Improvement: {improv:.1f}%')
        else:
            act_arr  = np.array(all_actual)
            pred_arr = np.array(all_pred)
            r2  = r2_score(act_arr, pred_arr) if len(act_arr) >= 2 else float('nan')
            mae = mean_absolute_error(act_arr, pred_arr)
            corrected = len(with_corr_actual) > 0
            lines.append(f'{"With" if corrected else "Without"} correction')
            lines.append(f'R²={r2:.2f}  MAE={mae:.3f}')

        ax.text(0.04, 0.96, '\n'.join(lines),
                transform=ax.transAxes, va='top', fontsize=_STATS_FONT_SIZE, bbox=ann_box)

        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        if prop == 'Size':
            ax.yaxis.set_major_locator(MultipleLocator(2.5))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=10, frameon=True, framealpha=0.95,
                      edgecolor='#CCCCCC', loc='lower right')

    _suptitle(fig, 'Self-Validation: Stored Predictions vs Actual')
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.07, right=0.98, wspace=0.35)
    return fig


# ERROR LEARNER CORRECTION ARROWS
# ------------------------------------------------------------------------------

def plot_bias_correction_arrows(optimizer, figsize: Tuple[int, int] = (14, 4.5)):
    """Show how the ErrorLearner shifts predictions toward actual values.

    Each completed recommendation is plotted as:
      - A circle (○) at the base model prediction (stored snapshot at rec time,
        before any mean correction — equivalent to the uncorrected model because
        all stored recs were made under LOO-only calibration).
      - A square (■) at the ErrorLearner-corrected prediction.
      - An arrow connecting them, coloured by whether the correction moved the
        prediction closer (green) or further (red) from the actual.

    Because the ErrorLearner is trained on these same residuals this is an
    in-sample demonstration — it shows the systematic biases the model has
    learned and will apply prospectively to future recommendations.
    """
    ann_box = dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#CCCCCC', alpha=0.95, linewidth=0.8)
    if not optimizer.error_learner.is_fitted:
        print("ErrorLearner is not yet fitted. Need at least "
              f"{optimizer.error_learner.min_samples} completed recommendations.")
        return None

    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations. Have {len(completed)}.")
        return None

    props = [
        ('Size',       'size_mu',  'Size',       'Size (nm)'),
        ('CV',         'cv_mu',    'CV',         'CV'),
        ('Squareness', 'sq_mu',    'Squareness', 'Squareness'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    for ax, (label, pred_key, actual_key, axis_label) in zip(axes, props):
        _style_ax(ax, title=label,
                  xlabel=f'Actual {axis_label}',
                  ylabel=f'Predicted {axis_label}')

        actuals, base_preds, corr_preds = [], [], []

        for rec in completed:
            if rec['actual_results'] is None:
                continue
            actual_val = rec['actual_results'].get(actual_key)
            base_pred  = rec['predictions'].get(pred_key)
            if actual_val is None or base_pred is None:
                continue

            cond = rec['conditions']
            feat = optimizer._raw_to_feature_array(cond)
            X_scaled = optimizer.base_optimizer.scaler.transform([feat])
            bias = optimizer.error_learner.predict_bias(X_scaled)

            corr_pred = base_pred + bias[label][0]

            actuals.append(actual_val)
            base_preds.append(base_pred)
            corr_preds.append(corr_pred)

        if len(actuals) < 2:
            ax.text(0.5, 0.5, 'Insufficient data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            continue

        actuals    = np.array(actuals)
        base_preds = np.array(base_preds)
        corr_preds = np.array(corr_preds)

        # Use only actuals + base_preds (same formula as plot_error_correction_impact)
        # so both figures have identical axis limits for overlay animation
        all_vals = np.concatenate([actuals, base_preds])
        margin = 0.08 * (all_vals.max() - all_vals.min())
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        ax.plot(lims, lims, color='#444444', lw=1.5, linestyle='--',
                label='Perfect prediction', zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        if label == 'Size':
            ax.yaxis.set_major_locator(MultipleLocator(2.5))

        # Draw arrows first (behind points)
        for a, b, c in zip(actuals, base_preds, corr_preds):
            err_before = abs(b - a)
            err_after  = abs(c - a)
            color = COLORS['success'] if err_after < err_before else COLORS['warning']
            ax.annotate(
                '', xy=(a, c), xytext=(a, b),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=1.4,
                    shrinkA=4, shrinkB=4,
                ),
                zorder=3,
            )

        ax.scatter(actuals, base_preds, s=70, color=COLORS['warning'],
                   marker='o', edgecolor='white', linewidth=0.8,
                   zorder=4, label='Base prediction (stored)')
        ax.scatter(actuals, corr_preds, s=70, color=COLORS['primary'],
                   marker='s', edgecolor='white', linewidth=0.8,
                   zorder=5, label='After bias correction')

        mae_base = mean_absolute_error(actuals, base_preds)
        mae_corr = mean_absolute_error(actuals, corr_preds)
        r2_base  = r2_score(actuals, base_preds)
        r2_corr  = r2_score(actuals, corr_preds)

        ax.text(0.04, 0.96,
                f'Base:      R²={r2_base:.2f}  MAE={mae_base:.3f}\n'
                f'Corrected: R²={r2_corr:.2f}  MAE={mae_corr:.3f}',
                transform=ax.transAxes, va='top', fontsize=_STATS_FONT_SIZE, bbox=ann_box)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=10, frameon=True, framealpha=0.95,
                      edgecolor='#CCCCCC', loc='lower right')

    _suptitle(fig, 'ErrorLearner Bias Correction: Base → Corrected vs Actual\n'
              '(in-sample — shows learned biases applied to future recommendations)')
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.07, right=0.98, wspace=0.35)
    return fig


# BIAS CORRECTION — SHARED DATA EXTRACTION
# ------------------------------------------------------------------------------

def _extract_bias_data(optimizer, loo_cv: bool = False):
    """Gather actuals, base predictions, and corrected predictions per property.

    Parameters
    ----------
    optimizer : SelfValidatingOptimizer
    loo_cv : bool
        If True, corrected predictions are computed via leave-one-out
        cross-validation of the residual GP (honest, out-of-sample).
        If False, uses the full-data ErrorLearner (in-sample).

    Returns a dict keyed by display label ('Size', 'CV', 'Squareness'),
    each containing arrays (actuals, base_preds, corr_preds) plus
    computed R² / MAE for both base and corrected.  Returns ``None`` if
    the ErrorLearner is not fitted or data is insufficient.
    """
    if not optimizer.error_learner.is_fitted:
        return None

    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        return None

    props = [
        ('Size',       'size_mu',  'Size',       'Size (nm)'),
        ('CV',         'cv_mu',    'CV',         'CV'),
        ('Squareness', 'sq_mu',    'Squareness', 'Squareness'),
    ]

    # Pre-collect valid indices and shared feature matrix for LOO
    valid_recs = []
    for rec in completed:
        if rec['actual_results'] is None:
            continue
        if all(rec['actual_results'].get(ak) is not None and
               rec['predictions'].get(pk) is not None
               for _, pk, ak, _ in props):
            valid_recs.append(rec)

    if len(valid_recs) < 2:
        return None

    X_features = []
    for rec in valid_recs:
        feat = optimizer._raw_to_feature_array(rec['conditions'])
        X_features.append(feat)
    X_scaled_all = optimizer.base_optimizer.scaler.transform(np.array(X_features))

    # Collect errors per property (for LOO refitting)
    all_errors = {}
    for label, pred_key, actual_key, _ in props:
        errs = []
        for rec in valid_recs:
            actual_val = rec['actual_results'][actual_key]
            base_pred = rec['predictions'][pred_key]
            errs.append(actual_val - base_pred)
        all_errors[label] = np.array(errs)

    result = {}
    for label, pred_key, actual_key, axis_label in props:
        actuals = np.array([r['actual_results'][actual_key] for r in valid_recs])
        base_preds = np.array([r['predictions'][pred_key] for r in valid_recs])
        errors = all_errors[label]

        if loo_cv:
            corr_preds = _loo_cv_bias(X_scaled_all, base_preds, errors)
        else:
            bias = optimizer.error_learner.predict_bias(X_scaled_all)
            corr_preds = base_preds + bias[label]

        result[label] = dict(
            actuals=actuals, base_preds=base_preds, corr_preds=corr_preds,
            axis_label=axis_label,
            r2_base=r2_score(actuals, base_preds),
            mae_base=mean_absolute_error(actuals, base_preds),
            r2_corr=r2_score(actuals, corr_preds),
            mae_corr=mean_absolute_error(actuals, corr_preds),
        )

    return result if result else None


def _loo_cv_bias(X_scaled, base_preds, errors):
    """Leave-one-out cross-validated bias correction.

    For each point i, fits a residual GP on all points except i,
    predicts the bias for i, and returns corrected predictions.
    Falls back to mean-of-others if a fold fails to fit.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        Matern, ConstantKernel as C, WhiteKernel,
    )

    n = len(base_preds)
    loo_corr = np.copy(base_preds)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X_scaled[mask]
        y_train = errors[mask]

        try:
            kernel = (
                C(1.0, (0.01, 100.0)) *
                Matern(
                    length_scale=[1.0] * X_scaled.shape[1],
                    length_scale_bounds=(0.1, 10.0), nu=2.5,
                ) +
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                n_restarts_optimizer=3, random_state=42,
            )
            gp.fit(X_train, y_train)
            bias_i = gp.predict(X_scaled[i:i+1])[0]
        except Exception:
            bias_i = np.mean(y_train)

        loo_corr[i] = base_preds[i] + bias_i

    return loo_corr


# BIAS CORRECTION — SUMMARY BAR CHART
# ------------------------------------------------------------------------------

def plot_bias_correction_summary(optimizer, figsize: Tuple[int, int] = (11, 4.5),
                                  loo_cv: bool = False):
    """Side-by-side grouped bar charts of R² and MAE before / after correction.

    Parameters
    ----------
    loo_cv : bool
        If True, corrected metrics are computed via leave-one-out CV
        (honest, out-of-sample).  Default uses in-sample correction.
    """
    data = _extract_bias_data(optimizer, loo_cv=loo_cv)
    if data is None:
        print("ErrorLearner not fitted or insufficient data.")
        return None

    labels = [k for k in ('Size', 'CV', 'Squareness') if k in data]
    r2_base = [data[k]['r2_base'] for k in labels]
    r2_corr = [data[k]['r2_corr'] for k in labels]
    mae_base = [data[k]['mae_base'] for k in labels]
    mae_corr = [data[k]['mae_corr'] for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    x = np.arange(len(labels))
    w = 0.32

    # --- R² panel ---
    bars_b = ax1.bar(x - w/2, r2_base, w, label='Base model',
                     color=COLORS['warning'], edgecolor='white', linewidth=0.8, zorder=3)
    bars_c = ax1.bar(x + w/2, r2_corr, w, label='After correction',
                     color=COLORS['primary'], edgecolor='white', linewidth=0.8, zorder=3)

    for bar, val in zip(bars_b, r2_base):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color=COLORS['warning'])
    for bar, val in zip(bars_c, r2_corr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color=COLORS['primary'])

    _style_ax(ax1, title='R²', ylabel='R² score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=13)
    ax1.set_ylim(min(min(r2_base), 0) - 0.15, 1.15)
    ax1.axhline(0, color='#999999', lw=0.8, ls='-', zorder=1)
    ax1.legend(fontsize=11, frameon=True, framealpha=0.95, edgecolor='#CCCCCC',
               loc='upper left')

    # --- MAE panel ---
    bars_b2 = ax2.bar(x - w/2, mae_base, w, label='Base model',
                      color=COLORS['warning'], edgecolor='white', linewidth=0.8, zorder=3)
    bars_c2 = ax2.bar(x + w/2, mae_corr, w, label='After correction',
                      color=COLORS['primary'], edgecolor='white', linewidth=0.8, zorder=3)

    for bar, val in zip(bars_b2, mae_base):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color=COLORS['warning'])
    for bar, val in zip(bars_c2, mae_corr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color=COLORS['primary'])

    _style_ax(ax2, title='Mean Absolute Error', ylabel='MAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=13)
    ax2.legend(fontsize=11, frameon=True, framealpha=0.95, edgecolor='#CCCCCC',
               loc='upper left')

    eval_tag = 'LOO-CV (out-of-sample)' if loo_cv else 'In-sample'
    _suptitle(fig, f'ErrorLearner Bias Correction — {eval_tag}')
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.82)
    return fig


# BIAS CORRECTION — BEFORE / AFTER SCATTER GRID
# ------------------------------------------------------------------------------

def plot_bias_correction_grid(optimizer, figsize: Tuple[int, int] = (14, 8),
                               loo_cv: bool = False):
    """2×3 scatter grid: top row = base predictions, bottom row = corrected.

    Parameters
    ----------
    loo_cv : bool
        If True, corrected predictions use leave-one-out CV (honest).
    """
    data = _extract_bias_data(optimizer, loo_cv=loo_cv)
    if data is None:
        print("ErrorLearner not fitted or insufficient data.")
        return None

    ordered = [k for k in ('Size', 'CV', 'Squareness') if k in data]
    ncols = len(ordered)

    fig, axes = plt.subplots(2, ncols, figsize=figsize)
    fig.patch.set_facecolor('white')
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, label in enumerate(ordered):
        d = data[label]
        actuals    = d['actuals']
        base_preds = d['base_preds']
        corr_preds = d['corr_preds']
        axis_label = d['axis_label']

        all_vals = np.concatenate([actuals, base_preds, corr_preds])
        margin = 0.08 * (all_vals.max() - all_vals.min())
        lims = [all_vals.min() - margin, all_vals.max() + margin]

        for row, (preds, r2, mae, row_label, color) in enumerate([
            (base_preds, d['r2_base'], d['mae_base'], 'Base model',      COLORS['warning']),
            (corr_preds, d['r2_corr'], d['mae_corr'], 'After correction', COLORS['primary']),
        ]):
            ax = axes[row, col]

            ax.plot(lims, lims, color='#444444', lw=1.5, linestyle='--', zorder=2)
            ax.scatter(actuals, preds, s=80, color=color,
                       marker='o', edgecolor='white', linewidth=0.8, zorder=4, alpha=0.85)

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

            title = label if row == 0 else ''
            ylabel = f'Predicted {axis_label}' if col == 0 else ''
            xlabel = f'Actual {axis_label}' if row == 1 else ''
            _style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)

            if label == 'Size':
                ax.yaxis.set_major_locator(MultipleLocator(2.5))
                ax.xaxis.set_major_locator(MultipleLocator(5))

            ax.text(0.04, 0.96,
                    f'R² = {r2:.2f}\nMAE = {mae:.3f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    fontweight='bold', bbox=_ANN_BOX)

    # Row labels on the left margin
    fig.text(0.005, 0.72, 'Base Model', fontsize=14, fontweight='bold',
             rotation=90, va='center', color=COLORS['warning'])
    fig.text(0.005, 0.30, 'Corrected', fontsize=14, fontweight='bold',
             rotation=90, va='center', color=COLORS['primary'])

    eval_tag = 'LOO-CV (out-of-sample)' if loo_cv else 'In-sample'
    _suptitle(fig, f'Bias Correction: Base vs Corrected — {eval_tag}')
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.08, top=0.88)
    return fig


# TARGET ACHIEVEMENT TRACKING
# ------------------------------------------------------------------------------

def plot_target_achievement(optimizer, figsize: Tuple[int, int] = (13, 5)):
    """Track how well recommendations achieve target sizes over time."""
    completed = optimizer.rec_store.get_completed()
    if len(completed) < 2:
        print(f"Need at least 2 completed recommendations. Have {len(completed)}.")
        return None

    completed_sorted = sorted(completed, key=lambda x: x.get('completed_timestamp', ''))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    targets, achieved, errors, indices, tolerances = [], [], [], [], []
    for i, rec in enumerate(completed_sorted):
        if rec['actual_results'] and rec['actual_results'].get('Size') is not None:
            targets.append(rec['target']['size'])
            achieved.append(rec['actual_results']['Size'])
            errors.append(rec['actual_results']['Size'] - rec['target']['size'])
            tolerances.append(rec['target']['tolerance'])
            indices.append(i + 1)

    ax1 = axes[0]
    _style_ax(ax1, title='Target vs Achieved Particle Size',
              xlabel='Recommendation #', ylabel='Size (nm)')
    if len(targets) >= 2:
        for tgt, tol in zip(targets, tolerances):
            ax1.axhspan(tgt - tol, tgt + tol, alpha=0.10, color=COLORS['primary'])
        ax1.plot(indices, targets,  'o--', color=COLORS['primary'],  markersize=8,
                 linewidth=2, label='Target Size',   zorder=3)
        ax1.plot(indices, achieved, 's-',  color=COLORS['success'],  markersize=8,
                 linewidth=2, label='Achieved Size', zorder=4)
        _styled_legend(ax1, loc='best')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=11)

    ax2 = axes[1]
    _style_ax(ax2, title='Error Distribution from Target',
              xlabel='Error from Target (nm)', ylabel='Count')
    if len(errors) >= 2:
        avg_tol = float(np.mean(tolerances))
        within  = [abs(e) <= t for e, t in zip(errors, tolerances)]
        success_rate = np.mean(within) * 100

        ax2.axvspan(-avg_tol, avg_tol, alpha=0.18, color=COLORS['success'],
                    label=f'Tolerance ±{avg_tol:.1f} nm', zorder=1)
        ax2.axvline(0, color='#444444', linestyle='--', linewidth=1.5,
                    label='Perfect', zorder=2)
        ax2.axvline(float(np.mean(errors)), color=COLORS['warning'], linestyle='-',
                    linewidth=2, label=f'Mean {np.mean(errors):+.2f} nm', zorder=3)
        ax2.hist(errors, bins=min(10, len(errors)), color=COLORS['primary'],
                 edgecolor='white', linewidth=0.8, alpha=0.80, zorder=4)
        ax2.text(0.97, 0.97,
                 f'{success_rate:.0f}% within\ntolerance',
                 transform=ax2.transAxes, va='top', ha='right',
                 fontsize=11, fontweight='bold',
                 color=COLORS['success'] if success_rate >= 50 else COLORS['warning'],
                 bbox=_ANN_BOX)
        _styled_legend(ax2, loc='upper left')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=11)

    _suptitle(fig, 'Target Achievement Tracking')
    fig.tight_layout(pad=1.5)
    return fig


# FEATURE IMPORTANCE
# ------------------------------------------------------------------------------

def plot_feature_importance(optimizer, figsize: Tuple[int, int] = (14, 5)):
    """Visualize feature importance from GP sensitivity analysis.

    Uses gradient-based sensitivity (mean |∂f/∂x_j| over training data, in
    standardised feature space, then normalised so the bars sum to 1 per
    property) which gives meaningful per-feature importance even with
    isotropic kernels. The x-axis is therefore a relative share of total
    sensitivity, NOT a percent variance explained.
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

    prop_colors = {'Size': COLORS['primary'], 'CV': COLORS['secondary'], 'Squareness': COLORS['tertiary']}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    for ax, prop in zip(axes, ['Size', 'CV', 'Squareness']):
        color = prop_colors[prop]
        prop_data = importance_df[importance_df['Model'] == prop].copy()

        _style_ax(ax, title=f'{prop}',
                  xlabel='Normalised mean |∂f/∂x| (per-property share)')
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='#EEEEEE', linewidth=0.8, zorder=0)
        ax.grid(axis='y', visible=False)

        if prop_data.empty:
            ax.text(0.5, 0.5, f'No data for {prop}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            continue

        prop_data = prop_data.sort_values('Importance', ascending=True)
        feat_labels = [_FEATURE_LABELS.get(f, f) for f in prop_data['Feature']]
        y_pos = np.arange(len(prop_data))

        bars = ax.barh(y_pos, prop_data['Importance'],
                       color=color, alpha=0.85, edgecolor='white',
                       linewidth=0.6, height=0.65)

        # Highlight top bar
        bars[-1].set_alpha(1.0)
        bars[-1].set_edgecolor('#333333')
        bars[-1].set_linewidth(1.2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, prop_data['Importance'].max() * 1.25)

        for i, (_, row) in enumerate(prop_data.iterrows()):
            ax.text(row['Importance'] + prop_data['Importance'].max() * 0.02,
                    i, f"{row['Importance']:.0%}",
                    va='center', fontsize=9,
                    fontweight='bold' if i == len(prop_data) - 1 else 'normal')

    _suptitle(fig, 'Feature Importance  (GP Gradient Sensitivity)')
    fig.tight_layout(pad=1.5)
    return fig


# CLASSIFIER CALIBRATION RELIABILITY DIAGRAMS
# ------------------------------------------------------------------------------

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
               f"Class balance: {metrics['class_balance']*100:.0f}%\n"
               f"Method: {metrics.get('evaluation_method', 'unknown')}",
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Observed Frequency')
        ax.set_title(f'{name} Calibration'); ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Classifier Reliability Diagrams', y=1.02)
    plt.tight_layout()
    return fig


# COLLINEARITY HEATMAP
# ------------------------------------------------------------------------------

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


# DATASET QUALITY DASHBOARD
# ------------------------------------------------------------------------------

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
            (0.35, 'CV', COLORS['secondary'], 'CV'),
            (0.0, 'Squareness', COLORS['tertiary'], 'Squareness'),
        ]:
            ax_sub = ax2.inset_axes([0, offset, 1, 0.25])
            ax_sub.hist(df_success[col], bins=12, color=color, edgecolor='black', alpha=0.7)
            ax_sub.set_ylabel('Count', fontsize=8)
            if col != 'Size':
                ax_sub.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_sub.set_title(f'{label}: {df_success[col].min():.2f}–{df_success[col].max():.2f}', fontsize=9, loc='left')
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
    n_features_active = len(optimizer.base_optimizer.features) if optimizer.base_optimizer else len(SYNTHESIS_FEATURES)
    n_regression = len(optimizer.base_optimizer.df_cubic) if optimizer.base_optimizer else n_success
    ax3.text(0.98, 0.95,
            f'Regression samples / feature (cubic-only):\n'
            f'  Raw: {n_regression/len(RAW_FACTORS):.1f}\n'
            f'  Synthesis: {n_regression/len(SYNTHESIS_FEATURES):.1f}\n'
            f'  Active: {n_regression/n_features_active:.1f}\n'
            f'  (recommend $\\geq$10)',
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

    fig.patch.set_facecolor('white')
    for ax_ in [ax1, ax3, ax4]:
        _style_ax(ax_)
    _suptitle(fig, 'Dataset Quality Dashboard  ·  Cu₃VS₄ Bayesian Optimization')
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=1.5)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Dashboard saved to {save_path}")
    return fig


# LOO-CV RESIDUALS
# ------------------------------------------------------------------------------

def plot_loo_residuals(optimizer, figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
    """LOO-CV parity and residuals: assess model fit and homoscedasticity."""
    if optimizer.base_optimizer is None:
        return None
    base = optimizer.base_optimizer
    if not base.metrics or 'y_pred' not in base.metrics.get('Size', {}):
        optimizer.validate_models()
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for row, (prop, _) in enumerate([('Size', 'size_mu'), ('CV', 'cv_mu'), ('Squareness', 'sq_mu')]):
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


# PROPERTY CORRELATIONS
# ------------------------------------------------------------------------------

def plot_property_correlations(optimizer, figsize: Tuple[int, int] = (13, 4.5)) -> Optional[plt.Figure]:
    """Pairwise correlations of outcome properties (Size, CV, Squareness)."""
    df_success = optimizer.exp_store.get_training_data()
    if df_success.empty or len(df_success) < 3:
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    pairs = [('Size', 'CV'), ('Size', 'Squareness'), ('CV', 'Squareness')]
    x_labels = ['Size (nm)', 'Size (nm)', 'CV']
    y_labels  = ['CV',       'Squareness', 'Squareness']

    for ax, (xcol, ycol), xl, yl in zip(axes, pairs, x_labels, y_labels):
        _style_ax(ax, title=f'{xl} vs {yl}', xlabel=xl, ylabel=yl)

        if 'PhasePure' in df_success.columns:
            for val, color, lbl in [(1, COLORS['success'], 'Phase pure'),
                                    (0, COLORS['warning'], 'Impure')]:
                mask = df_success['PhasePure'] == val
                if mask.any():
                    ax.scatter(df_success.loc[mask, xcol], df_success.loc[mask, ycol],
                               c=color, s=70, alpha=0.80, edgecolor='white',
                               linewidth=0.8, label=lbl, zorder=3)
        else:
            ax.scatter(df_success[xcol], df_success[ycol],
                       c=COLORS['primary'], s=70, alpha=0.80,
                       edgecolor='white', linewidth=0.8, zorder=3)

        # Pearson r annotation
        valid = df_success[[xcol, ycol]].dropna()
        if len(valid) > 2:
            r = float(np.corrcoef(valid[xcol], valid[ycol])[0, 1])
            ax.text(0.04, 0.96, f'r = {r:+.2f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    fontweight='bold', bbox=_ANN_BOX)

        _styled_legend(ax, loc='lower right')

    _suptitle(fig, 'Outcome Property Correlations  (Successful Experiments)')
    fig.tight_layout(pad=1.5)
    return fig


# ACQUISITION SLICE
# ------------------------------------------------------------------------------

def plot_acquisition_slice(
    optimizer,
    target_size: float = 20.0,
    size_tol: float = 2.5,
    squareness_bin: str = 'highly_cubic',
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
    acq = base.acquisition(X, target_size, size_tol, squareness_bin=squareness_bin)
    Z = acq['total'].reshape(V1.shape)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(V1, V2, Z, shading='auto', cmap='viridis')
    ax.set_xlabel(param1); ax.set_ylabel(param2)
    ax.set_title(f'Acquisition (target {target_size}±{size_tol} nm, bin={squareness_bin})')
    plt.colorbar(im, ax=ax, label='Acquisition value')
    ax.scatter(base.df_success[param1], base.df_success[param2],
               c='white', s=20, alpha=0.8, edgecolor='black', linewidth=0.5, label='Training data')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    return fig


# 3D RESPONSE & CLASSIFICATION SURFACES
# ------------------------------------------------------------------------------

_FEATURE_LABELS = {
    'Temp': 'Temperature (°C)',
    'Cu_V_ratio': 'Cu/V Ratio',
    'S_Metal_ratio': 'S/Metal Ratio',
    'Ligand_Metal_ratio': 'Ligand/Metal Ratio',
    'log_Time': r'log$_{10}$(Time)',
    'Time': 'Time (min)',
    'VOacac': r'VO(acac)$_2$ (mmol)',
    'DDT': 'DDT (mL)',
    'OAm': 'OAm (mL)',
    'Metal_Conc': 'Metal Conc. (mM)',
}


def _build_cube_faces(base, top_features, n_grid):
    """Construct feature-space grid points for all 6 faces of a design-space
    cube spanned by *top_features*.

    Returns ``(faces, X_all)`` where *faces* is a list of dicts with 3-D
    plotting coordinates and grid shape, and *X_all* is the stacked
    ``(N_total, n_features)`` array ready for ``base.predict()``.
    Non-top features are fixed at their cubic-training-data median.
    """
    features = base.features
    bounds = base.bounds
    medians = base.df_cubic[features].median()

    f_x, f_y, f_z = top_features
    idx = {f: features.index(f) for f in top_features}
    ranges = {f: np.linspace(bounds[f][0], bounds[f][1], n_grid)
              for f in top_features}

    face_specs = [
        (f_x, f_y, f_z, bounds[f_z][0]),
        (f_x, f_y, f_z, bounds[f_z][1]),
        (f_y, f_z, f_x, bounds[f_x][0]),
        (f_y, f_z, f_x, bounds[f_x][1]),
        (f_x, f_z, f_y, bounds[f_y][0]),
        (f_x, f_z, f_y, bounds[f_y][1]),
    ]

    faces, all_X = [], []
    for dim_a, dim_b, dim_fix, fix_val in face_specs:
        A, B = np.meshgrid(ranges[dim_a], ranges[dim_b])
        X_feat = np.tile(medians.values, (A.size, 1))
        X_feat[:, idx[dim_a]] = A.ravel()
        X_feat[:, idx[dim_b]] = B.ravel()
        X_feat[:, idx[dim_fix]] = fix_val

        coords = {dim_a: A, dim_b: B, dim_fix: np.full_like(A, fix_val)}
        faces.append({
            'X_plot': coords[f_x], 'Y_plot': coords[f_y],
            'Z_plot': coords[f_z], 'shape': A.shape,
        })
        all_X.append(X_feat)

    return faces, np.vstack(all_X)


def _build_interior_slices(base, top_features, n_grid):
    """Three axis-aligned mid-plane slices through the interior of the design
    cube.  One slice fixes each of the three top features at its midpoint,
    letting the other two vary freely — exposing interior regions invisible
    on the outer faces.
    """
    features = base.features
    bounds   = base.bounds
    medians  = base.df_cubic[features].median()

    f_x, f_y, f_z = top_features
    idx    = {f: features.index(f) for f in top_features}
    ranges = {f: np.linspace(bounds[f][0], bounds[f][1], n_grid)
              for f in top_features}
    mids   = {f: (bounds[f][0] + bounds[f][1]) / 2.0 for f in top_features}

    slice_specs = [
        (f_x, f_y, f_z, mids[f_z]),
        (f_y, f_z, f_x, mids[f_x]),
        (f_x, f_z, f_y, mids[f_y]),
    ]

    slices, all_X = [], []
    for dim_a, dim_b, dim_fix, fix_val in slice_specs:
        A, B = np.meshgrid(ranges[dim_a], ranges[dim_b])
        X_feat = np.tile(medians.values, (A.size, 1))
        X_feat[:, idx[dim_a]]   = A.ravel()
        X_feat[:, idx[dim_b]]   = B.ravel()
        X_feat[:, idx[dim_fix]] = fix_val

        coords = {dim_a: A, dim_b: B, dim_fix: np.full_like(A, fix_val)}
        slices.append({
            'X_plot': coords[f_x], 'Y_plot': coords[f_y],
            'Z_plot': coords[f_z], 'shape': A.shape,
        })
        all_X.append(X_feat)

    return slices, np.vstack(all_X)


def _color_surfaces(surfaces, p_mp, p_hc, p_pc, c_mp, c_hc, c_pc,
                    alpha, offset):
    """Assign discrete (winner-takes-all) colours to a list of surface dicts.
    Returns the updated offset into the flat probability arrays."""
    for surf in surfaces:
        n = surf['shape'][0] * surf['shape'][1]
        pm = p_mp[offset:offset + n].reshape(surf['shape'])
        ph = p_hc[offset:offset + n].reshape(surf['shape'])
        pp = p_pc[offset:offset + n].reshape(surf['shape'])

        winner = np.argmax(np.stack([pm, ph, pp], axis=-1), axis=-1)
        rgba = np.zeros((*surf['shape'], 4))
        for ch in range(3):
            rgba[..., ch] = np.where(winner == 0, c_mp[ch],
                             np.where(winner == 1, c_hc[ch], c_pc[ch]))
        rgba[..., 3] = alpha
        surf['colors'] = rgba
        offset += n
    return offset


def _get_top_features(base, model_name, n=3):
    """Return the *n* most important features for *model_name*."""
    df = base.get_feature_importance()
    return (df[df['Model'] == model_name]
            .sort_values('Importance', ascending=False)['Feature']
            .head(n).tolist())


def _overlay_recommendations(ax, optimizer, base, top_3):
    """Add recommended conditions as diamond markers to a 3D axes."""
    from features import build_feature_vector_from_raw

    recs = [r for r in optimizer.rec_store.get_all()
            if r['status'] != 'skipped']
    if not recs:
        return

    feat_idx = {f: base.features.index(f) for f in top_3}
    completed, pending = [], []
    for r in recs:
        vec = build_feature_vector_from_raw(
            r['conditions'], base.feature_mode, base.features)
        pt = [vec[feat_idx[f]] for f in top_3]
        (completed if r['status'] == 'completed' else pending).append(pt)

    if completed:
        arr = np.array(completed)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                   c=COLORS['completed'], s=120, marker='D', alpha=1.0,
                   edgecolor='black', linewidth=1.5,
                   label='Completed recs', depthshade=False, zorder=6)
    if pending:
        arr = np.array(pending)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                   c=COLORS['pending'], s=120, marker='D', alpha=1.0,
                   edgecolor='black', linewidth=1.5,
                   label='Pending recs', depthshade=False, zorder=6)


def plot_response_surface(
    optimizer,
    response: str = 'Size',
    n_grid: int = 25,
    elev: float = 25,
    azim: float = 135,
    figsize: Tuple[int, int] = (12, 9),
) -> Optional[plt.Figure]:
    """3D cube response surface coloured by a GP prediction.

    The three axes are the top-3 most important features (gradient-based
    sensitivity) for the chosen response model.  Non-top features are held
    at their median cubic-training-data values.

    Parameters
    ----------
    response : {'Size', 'CV', 'Squareness', 'Feasibility'}
        Which model output to visualise.  *Feasibility* shows the joint
        classifier probability P(HasProduct) * P(PhasePure) * P(IsCubic)
        and uses the Size model's top features for the axes.
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    cfg = {
        'Size':        ('Size',       'size_mu',    'Predicted Size (nm)',   'viridis'),
        'CV':          ('CV',         'cv_mu',      'Predicted CV',         'viridis_r'),
        'Squareness':  ('Squareness', 'sq_mu',      'Predicted Squareness', 'viridis'),
        'Feasibility': ('Size',       'p_feasible', 'P(feasible)',          'RdYlGn'),
    }
    if response not in cfg:
        print(f"Unknown response '{response}'. Choose from {list(cfg.keys())}")
        return None

    model_name, pred_key, cbar_label, cmap_name = cfg[response]

    top_3 = _get_top_features(base, model_name)
    if len(top_3) < 3:
        print(f"Need at least 3 features for response surface. Have {len(top_3)}.")
        return None

    faces, X_all = _build_cube_faces(base, top_3, n_grid)
    preds = base.predict(X_all)
    R_all = preds[pred_key]

    offset = 0
    for face in faces:
        n = face['shape'][0] * face['shape'][1]
        face['R'] = R_all[offset:offset + n].reshape(face['shape'])
        offset += n

    vmin = min(f['R'].min() for f in faces)
    vmax = max(f['R'].max() for f in faces)
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        ax.plot_surface(face['X_plot'], face['Y_plot'], face['Z_plot'],
                        facecolors=colormap(norm(face['R'])),
                        rstride=1, cstride=1,
                        alpha=0.55, shade=False, antialiased=True,
                        edgecolor=(0.4, 0.4, 0.4, 0.08), linewidth=0.3)

    f_x, f_y, f_z = top_3
    bounds = base.bounds
    ax.set_xlim(bounds[f_x][0], bounds[f_x][1])
    ax.set_ylim(bounds[f_y][0], bounds[f_y][1])
    ax.set_zlim(bounds[f_z][0], bounds[f_z][1])

    df_pts = base.df_success if response == 'Feasibility' else base.df_cubic
    ax.scatter(df_pts[f_x], df_pts[f_y], df_pts[f_z],
               c='white', s=80, alpha=1.0, edgecolor='black', linewidth=1.5,
               label='Training data', depthshade=False, zorder=5)

    _overlay_recommendations(ax, optimizer, base, top_3)

    ax.set_xlabel(_FEATURE_LABELS.get(f_x, f_x), fontsize=15, labelpad=8)
    ax.set_ylabel(_FEATURE_LABELS.get(f_y, f_y), fontsize=15, labelpad=8)
    ax.set_zlabel(_FEATURE_LABELS.get(f_z, f_z), fontsize=15, labelpad=8)
    ax.tick_params(labelsize=12)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(cbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    display = 'CV' if response == 'CV' else response
    ax.set_title(f'{display} Response Surface\n({f_x} × {f_y} × {f_z})',
                 fontsize=17, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    return fig


def plot_classification_surface(
    optimizer,
    n_grid: int = 25,
    elev: float = 30,
    azim: float = 315,
    figsize: Tuple[int, int] = (12, 9),
    show_slices: bool = True,
) -> Optional[plt.Figure]:
    """3D classification surface for squareness bins.

    Uses the Squareness model's top-3 features as axes.  Each cube-face
    point is coloured by its most probable bin: *multipod*, *highly cubic*,
    or *poorly cubic*.  Training data are overlaid with markers matching
    their observed bin assignment.

    When *show_slices* is True (default), three additional mid-plane cross-
    sections are drawn through the interior of the cube, making poorly-cubic
    regions that sit inside the parameter space visible.
    """
    from optimizer import compute_bin_probability

    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    top_3 = _get_top_features(base, 'Squareness')
    if len(top_3) < 3:
        print(f"Need at least 3 features for classification surface. "
              f"Have {len(top_3)}.")
        return None

    faces, X_faces = _build_cube_faces(base, top_3, n_grid)

    if show_slices:
        slices, X_slices = _build_interior_slices(base, top_3, n_grid)
        X_all = np.vstack([X_faces, X_slices])
    else:
        X_all = X_faces

    preds = base.predict(X_all)

    p_mp = compute_bin_probability(preds['sq_mu'], preds['sq_std'],
                                   preds['p_cubic'], base.sq_threshold,
                                   'multipod')
    p_hc = compute_bin_probability(preds['sq_mu'], preds['sq_std'],
                                   preds['p_cubic'], base.sq_threshold,
                                   'highly_cubic')
    p_pc = compute_bin_probability(preds['sq_mu'], preds['sq_std'],
                                   preds['p_cubic'], base.sq_threshold,
                                   'poorly_cubic')

    c_mp = np.array(to_rgba(COLORS['warning']))[:3]
    c_hc = np.array(to_rgba(COLORS['success']))[:3]
    c_pc = np.array(to_rgba(COLORS['tertiary']))[:3]

    offset = _color_surfaces(faces, p_mp, p_hc, p_pc,
                             c_mp, c_hc, c_pc, alpha=0.55, offset=0)
    if show_slices:
        _color_surfaces(slices, p_mp, p_hc, p_pc,
                        c_mp, c_hc, c_pc, alpha=0.75, offset=offset)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        ax.plot_surface(face['X_plot'], face['Y_plot'], face['Z_plot'],
                        facecolors=face['colors'],
                        rstride=1, cstride=1,
                        shade=False, antialiased=True,
                        edgecolor=(0.4, 0.4, 0.4, 0.05), linewidth=0.2)

    if show_slices:
        for sl in slices:
            ax.plot_surface(sl['X_plot'], sl['Y_plot'], sl['Z_plot'],
                            facecolors=sl['colors'],
                            rstride=1, cstride=1,
                            shade=False, antialiased=True,
                            edgecolor=(0.4, 0.4, 0.4, 0.05), linewidth=0.2)

    f_x, f_y, f_z = top_3
    bounds = base.bounds
    ax.set_xlim(bounds[f_x][0], bounds[f_x][1])
    ax.set_ylim(bounds[f_y][0], bounds[f_y][1])
    ax.set_zlim(bounds[f_z][0], bounds[f_z][1])

    df_s = base.df_success
    threshold = base.sq_threshold
    is_cubic = _get_is_cubic(df_s)
    bin_masks = _get_bin_masks(df_s, is_cubic, threshold)
    colors = _bin_colors()

    for label, mask, marker, color in zip(_BIN_NAMES, bin_masks,
                                          _BIN_MARKERS, colors):
        sub = df_s[mask]
        if not sub.empty:
            ax.scatter(sub[f_x], sub[f_y], sub[f_z],
                       c=color, s=80, marker=marker, alpha=1.0,
                       edgecolor='black', linewidth=1.5, label=label,
                       depthshade=False, zorder=5)

    _overlay_recommendations(ax, optimizer, base, top_3)

    ax.set_xlabel(_FEATURE_LABELS.get(f_x, f_x))
    ax.set_ylabel(_FEATURE_LABELS.get(f_y, f_y))
    ax.set_zlabel(_FEATURE_LABELS.get(f_z, f_z))
    ax.set_title(f'Squareness Bin Classification Surface\n'
                 f'({f_x} × {f_y} × {f_z})',
                 fontsize=12, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

    legend_elements = [
        Patch(facecolor=COLORS['warning'],  label='Multipod'),
        Patch(facecolor=COLORS['success'],  label='Highly cubic'),
        Patch(facecolor=COLORS['tertiary'], label='Poorly cubic'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    plt.tight_layout()
    return fig


def plot_classification_facets(
    optimizer,
    n_grid: int = 50,
    figsize: Tuple[int, int] = (16, 5),
) -> Optional[plt.Figure]:
    """Faceted 2D classification maps for squareness bins.

    Fixes the third most-important Squareness feature at three levels
    (low / mid / high) and draws a 2-D heatmap of the most-probable bin over
    the top-2 feature axes for each level.  Decision boundaries are drawn as
    crisp contour lines.  All training experiments are overlaid as markers
    (projected onto the 2-D plane) coloured by their observed bin.
    """
    from optimizer import compute_bin_probability
    from matplotlib.lines import Line2D

    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    top_3 = _get_top_features(base, 'Squareness')
    if len(top_3) < 3:
        print(f"Need at least 3 features. Have {len(top_3)}.")
        return None

    f_x, f_y, f_z = top_3
    features = base.features
    bounds   = base.bounds
    medians  = base.df_cubic[features].median()
    idx      = {f: features.index(f) for f in top_3}
    threshold = base.sq_threshold

    x_range = np.linspace(bounds[f_x][0], bounds[f_x][1], n_grid)
    y_range = np.linspace(bounds[f_y][0], bounds[f_y][1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    z_lo, z_hi = bounds[f_z][0], bounds[f_z][1]
    z_levels   = [z_lo, (z_lo + z_hi) / 2, z_hi]
    z_labels   = [
        f"Low  ({z_lo:.2g})",
        f"Mid  ({(z_lo + z_hi) / 2:.2g})",
        f"High ({z_hi:.2g})",
    ]

    c_mp = np.array(to_rgba(COLORS['warning']))[:3]
    c_hc = np.array(to_rgba(COLORS['success']))[:3]
    c_pc = np.array(to_rgba(COLORS['tertiary']))[:3]

    df_s = base.df_success
    is_cubic = _get_is_cubic(df_s)
    bin_masks = _get_bin_masks(df_s, is_cubic, threshold)
    colors = _bin_colors()

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig.patch.set_facecolor('white')

    z_range = z_hi - z_lo

    for col, (ax, z_val, z_label) in enumerate(zip(axes, z_levels, z_labels)):
        X_feat = np.tile(medians.values, (X_grid.size, 1))
        X_feat[:, idx[f_x]] = X_grid.ravel()
        X_feat[:, idx[f_y]] = Y_grid.ravel()
        X_feat[:, idx[f_z]] = z_val

        preds = base.predict(X_feat)
        p_mp = compute_bin_probability(preds['sq_mu'], preds['sq_std'],
                                       preds['p_cubic'], threshold, 'multipod')
        p_hc = compute_bin_probability(preds['sq_mu'], preds['sq_std'],
                                       preds['p_cubic'], threshold, 'highly_cubic')
        p_pc = compute_bin_probability(preds['sq_mu'], preds['sq_std'],
                                       preds['p_cubic'], threshold, 'poorly_cubic')

        winner = np.argmax(
            np.stack([p_mp, p_hc, p_pc], axis=1), axis=1
        ).reshape(n_grid, n_grid)

        img = np.zeros((n_grid, n_grid, 4))
        for ch in range(3):
            img[..., ch] = np.where(winner == 0, c_mp[ch],
                           np.where(winner == 1, c_hc[ch], c_pc[ch]))
        img[..., 3] = 0.85

        ax.imshow(img, origin='lower', aspect='auto',
                  extent=[bounds[f_x][0], bounds[f_x][1],
                          bounds[f_y][0], bounds[f_y][1]],
                  interpolation='nearest')

        ax.contour(X_grid, Y_grid, winner,
                   levels=[0.5, 1.5], colors='white',
                   linewidths=2.0, linestyles='--', zorder=3)

        for mask, marker, color in zip(bin_masks, _BIN_MARKERS, colors):
            sub = df_s[mask]
            if sub.empty:
                continue
            dist  = np.abs(sub[f_z] - z_val) / (z_range + 1e-9)
            near  = dist <= 0.25
            far   = ~near
            if near.any():
                ax.scatter(sub.loc[near, f_x], sub.loc[near, f_y],
                           c=color, s=90, marker=marker, alpha=1.0,
                           edgecolor='black', linewidth=1.2, zorder=5)
            if far.any():
                ax.scatter(sub.loc[far, f_x], sub.loc[far, f_y],
                           c=color, s=50, marker=marker, alpha=0.25,
                           edgecolor='none', zorder=4)

        f_z_label = _FEATURE_LABELS.get(f_z, f_z)
        ax.set_title(f"{f_z_label}\n{z_label}",
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(_FEATURE_LABELS.get(f_x, f_x), fontsize=11)
        if col == 0:
            ax.set_ylabel(_FEATURE_LABELS.get(f_y, f_y), fontsize=11)

        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')

    region_handles = [
        Patch(facecolor=c, edgecolor='black', label=f'{n} region')
        for c, n in zip(_bin_colors(), _BIN_NAMES)
    ]
    marker_handles = _bin_marker_legend_handles() + [
        Line2D([0], [0], linestyle='--', color='white', linewidth=2,
               label='Decision boundary'),
    ]
    fig.legend(handles=region_handles + marker_handles,
               loc='lower center', ncol=7, fontsize=9.5,
               frameon=True, framealpha=0.95, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, -0.10))

    fig.suptitle(
        'Squareness Bin Classification Map\n'
        '(Most probable bin · faded markers = experiments outside this slice)',
        fontsize=13, fontweight='bold', y=1.03,
    )
    fig.tight_layout(w_pad=0.5)
    return fig


def plot_bin_probability_facets(
    optimizer,
    n_grid: int = 50,
    figsize: Tuple[int, int] = (14, 11),
    fixed_feature: Optional[str] = None,
) -> Optional[plt.Figure]:
    """3×3 probability heatmap grid for all three squareness bins.

    Rows = bins (Multipod / Highly cubic / Poorly cubic).
    Columns = a chosen feature fixed at low / mid / high.

    X and Y axes are always the two most important features for Squareness.
    The column-slice variable is *fixed_feature* if supplied, otherwise the
    3rd most-important feature (original default behaviour).

    Each panel shows the continuous GP-derived probability (0–1) for that bin
    as a colourmap, with training data overlaid.  Unlike the winner-takes-all
    map, this makes regions of non-zero poorly-cubic probability visible even
    when that bin never achieves a plurality.
    """
    from optimizer import compute_bin_probability
    from matplotlib.lines import Line2D

    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    all_top = _get_top_features(base, 'Squareness', n=len(base.features))
    if len(all_top) < 3:
        print(f"Need at least 3 features. Have {len(all_top)}.")
        return None

    f_x, f_y = all_top[0], all_top[1]

    if fixed_feature is not None:
        if fixed_feature not in base.features:
            print(f"'{fixed_feature}' not in feature set: {base.features}")
            return None
        f_z = fixed_feature
    else:
        f_z = all_top[2]

    features  = base.features
    bounds    = base.bounds
    medians   = base.df_cubic[features].median()
    idx       = {f: features.index(f) for f in [f_x, f_y, f_z]}
    threshold = base.sq_threshold

    x_range = np.linspace(bounds[f_x][0], bounds[f_x][1], n_grid)
    y_range = np.linspace(bounds[f_y][0], bounds[f_y][1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    z_lo, z_hi = bounds[f_z][0], bounds[f_z][1]
    z_levels = [z_lo, (z_lo + z_hi) / 2, z_hi]
    z_labels = [f"Low  ({z_lo:.2g})", f"Mid  ({(z_lo+z_hi)/2:.2g})", f"High ({z_hi:.2g})"]
    z_range  = z_hi - z_lo

    bin_cmaps = [plt.get_cmap('Reds'), plt.get_cmap('Greens'),
                 plt.get_cmap('Oranges')]
    colors = _bin_colors()

    df_s = base.df_success
    is_cubic = _get_is_cubic(df_s)
    bin_masks = _get_bin_masks(df_s, is_cubic, threshold)

    fig, axes = plt.subplots(3, 3, figsize=figsize,
                             sharex='col', sharey='row')
    fig.patch.set_facecolor('white')

    for col, (z_val, z_label) in enumerate(zip(z_levels, z_labels)):
        X_feat = np.tile(medians.values, (X_grid.size, 1))
        X_feat[:, idx[f_x]] = X_grid.ravel()
        X_feat[:, idx[f_y]] = Y_grid.ravel()
        X_feat[:, idx[f_z]] = z_val

        preds = base.predict(X_feat)
        probs = {
            'multipod':     compute_bin_probability(
                preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
                threshold, 'multipod'),
            'highly_cubic': compute_bin_probability(
                preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
                threshold, 'highly_cubic'),
            'poorly_cubic': compute_bin_probability(
                preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
                threshold, 'poorly_cubic'),
        }

        for row, (bin_name, bin_key, cmap, d_color, marker, mask) in enumerate(
            zip(_BIN_NAMES, _BIN_KEYS, bin_cmaps, colors, _BIN_MARKERS, bin_masks)
        ):
            ax = axes[row][col]
            ax.set_facecolor('#f5f5f5')

            prob_grid = probs[bin_key].reshape(n_grid, n_grid)

            im = ax.imshow(prob_grid, origin='lower', aspect='auto',
                           extent=[bounds[f_x][0], bounds[f_x][1],
                                   bounds[f_y][0], bounds[f_y][1]],
                           cmap=cmap, vmin=0, vmax=1,
                           interpolation='bilinear')

            try:
                ax.contour(X_grid, Y_grid, prob_grid,
                           levels=[0.33, 0.5],
                           colors=['#333333', '#333333'],
                           linewidths=[1.0, 2.0],
                           linestyles=[':', '--'],
                           zorder=3)
            except Exception:
                pass

            sub_all = df_s[mask]
            if not sub_all.empty:
                dist = np.abs(sub_all[f_z] - z_val) / (z_range + 1e-9)
                near = dist <= 0.25
                if near.any():
                    ax.scatter(sub_all.loc[near, f_x], sub_all.loc[near, f_y],
                               c=d_color, s=80, marker=marker, alpha=1.0,
                               edgecolor='black', linewidth=1.1, zorder=5)
                if (~near).any():
                    ax.scatter(sub_all.loc[~near, f_x], sub_all.loc[~near, f_y],
                               c=d_color, s=40, marker=marker, alpha=0.2,
                               edgecolor='none', zorder=4)

            if col == 2:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('Probability', fontsize=12)
                cb.ax.tick_params(labelsize=11)

            if col == 0:
                ax.set_ylabel(
                    f'{bin_name}\n{_FEATURE_LABELS.get(f_y, f_y)}',
                    fontsize=16)
            else:
                ax.set_ylabel('')

            if row == 0:
                f_z_label = _FEATURE_LABELS.get(f_z, f_z)
                ax.set_title(f'{f_z_label}\n{z_label}',
                             fontsize=18, fontweight='bold', pad=8)

            if row == 2:
                ax.set_xlabel(_FEATURE_LABELS.get(f_x, f_x), fontsize=16)

            ax.tick_params(axis='both', labelsize=12)

            for spine in ax.spines.values():
                spine.set_edgecolor('#888888')

    marker_handles = _bin_marker_legend_handles() + [
        Line2D([0], [0], linestyle=':', color='#555555',
               linewidth=1.5, label='P = 0.33 contour'),
        Line2D([0], [0], linestyle='--', color='#555555',
               linewidth=2.0, label='P = 0.50 contour'),
    ]
    fig.legend(handles=marker_handles, loc='lower center', ncol=5,
               fontsize=14, frameon=True, framealpha=0.95,
               edgecolor='#CCCCCC', bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8, w_pad=0.4)
    return fig


# BIN PROBABILITY FACETS — ALL REMAINING FEATURES AS COLUMN FACETS
# ------------------------------------------------------------------------------

def plot_bin_probability_facets_all(
    optimizer,
    n_grid: int = 50,
    figsize: Tuple[int, int] = (14, 11),
) -> list:
    """Produce one 3×3 probability-facet figure for each non-top-2 feature.

    The x and y axes are always the two most important Squareness features.
    Each figure uses a different remaining feature (ranked 3rd, 4th, 5th …)
    as the column-slice variable, so you can see how the landscape shifts as
    each secondary variable is varied.

    Returns a list of figures (one per binning variable).
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return []
    base = optimizer.base_optimizer

    all_top = _get_top_features(base, 'Squareness', n=len(base.features))
    if len(all_top) < 3:
        print(f"Need at least 3 features. Have {len(all_top)}.")
        return []

    remaining = all_top[2:]  # 3rd, 4th, 5th … most important
    figs = []
    for feat in remaining:
        fig = plot_bin_probability_facets(
            optimizer, n_grid=n_grid, figsize=figsize, fixed_feature=feat
        )
        if fig is not None:
            figs.append(fig)
    return figs


# BIN PROBABILITY FACETS — Cu/V RATIO FIXED AS COLUMN FACET
# ------------------------------------------------------------------------------

def plot_bin_probability_facets_cuv(
    optimizer,
    n_grid: int = 50,
    figsize: Tuple[int, int] = (14, 11),
) -> Optional[plt.Figure]:
    """3×3 probability heatmap grid with Cu/V ratio fixed as the column facet.

    Identical to plot_bin_probability_facets, but Cu_V_ratio is always used
    as the column-slice variable.  The two x/y axes are still chosen
    dynamically as the top-2 most important features for Squareness
    (excluding Cu_V_ratio), so the plot updates as experiments accumulate.
    """
    from optimizer import compute_bin_probability
    from matplotlib.lines import Line2D

    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    f_z = 'Cu_V_ratio'
    if f_z not in base.features:
        print(f"'{f_z}' is not in the current feature set: {base.features}")
        return None

    # Top-2 most important features for Squareness, excluding Cu_V_ratio
    all_top = _get_top_features(base, 'Squareness', n=len(base.features))
    top_xy = [f for f in all_top if f != f_z][:2]
    if len(top_xy) < 2:
        print("Not enough features for x/y axes after fixing Cu_V_ratio as column facet.")
        return None
    f_x, f_y = top_xy

    features  = base.features
    bounds    = base.bounds
    medians   = base.df_cubic[features].median()
    idx       = {f: features.index(f) for f in [f_x, f_y, f_z]}
    threshold = base.sq_threshold

    x_range = np.linspace(bounds[f_x][0], bounds[f_x][1], n_grid)
    y_range = np.linspace(bounds[f_y][0], bounds[f_y][1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    z_lo, z_hi = bounds[f_z][0], bounds[f_z][1]
    z_levels = [z_lo, (z_lo + z_hi) / 2, z_hi]
    z_labels = [f"Low  ({z_lo:.2g})", f"Mid  ({(z_lo+z_hi)/2:.2g})", f"High ({z_hi:.2g})"]
    z_range  = z_hi - z_lo

    bin_cmaps = [plt.get_cmap('Reds'), plt.get_cmap('Greens'),
                 plt.get_cmap('Oranges')]
    colors = _bin_colors()

    df_s = base.df_success
    is_cubic = _get_is_cubic(df_s)
    bin_masks = _get_bin_masks(df_s, is_cubic, threshold)

    fig, axes = plt.subplots(3, 3, figsize=figsize,
                             sharex='col', sharey='row')
    fig.patch.set_facecolor('white')

    for col, (z_val, z_label) in enumerate(zip(z_levels, z_labels)):
        X_feat = np.tile(medians.values, (X_grid.size, 1))
        X_feat[:, idx[f_x]] = X_grid.ravel()
        X_feat[:, idx[f_y]] = Y_grid.ravel()
        X_feat[:, idx[f_z]] = z_val

        preds = base.predict(X_feat)
        probs = {
            'multipod':     compute_bin_probability(
                preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
                threshold, 'multipod'),
            'highly_cubic': compute_bin_probability(
                preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
                threshold, 'highly_cubic'),
            'poorly_cubic': compute_bin_probability(
                preds['sq_mu'], preds['sq_std'], preds['p_cubic'],
                threshold, 'poorly_cubic'),
        }

        for row, (bin_name, bin_key, cmap, d_color, marker, mask) in enumerate(
            zip(_BIN_NAMES, _BIN_KEYS, bin_cmaps, colors, _BIN_MARKERS, bin_masks)
        ):
            ax = axes[row][col]
            ax.set_facecolor('#f5f5f5')

            prob_grid = probs[bin_key].reshape(n_grid, n_grid)

            im = ax.imshow(prob_grid, origin='lower', aspect='auto',
                           extent=[bounds[f_x][0], bounds[f_x][1],
                                   bounds[f_y][0], bounds[f_y][1]],
                           cmap=cmap, vmin=0, vmax=1,
                           interpolation='bilinear')

            try:
                ax.contour(X_grid, Y_grid, prob_grid,
                           levels=[0.33, 0.5],
                           colors=['#333333', '#333333'],
                           linewidths=[1.0, 2.0],
                           linestyles=[':', '--'],
                           zorder=3)
            except Exception:
                pass

            sub_all = df_s[mask]
            if not sub_all.empty:
                dist = np.abs(sub_all[f_z] - z_val) / (z_range + 1e-9)
                near = dist <= 0.25
                if near.any():
                    ax.scatter(sub_all.loc[near, f_x], sub_all.loc[near, f_y],
                               c=d_color, s=80, marker=marker, alpha=1.0,
                               edgecolor='black', linewidth=1.1, zorder=5)
                if (~near).any():
                    ax.scatter(sub_all.loc[~near, f_x], sub_all.loc[~near, f_y],
                               c=d_color, s=40, marker=marker, alpha=0.2,
                               edgecolor='none', zorder=4)

            if col == 2:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('Probability', fontsize=12)
                cb.ax.tick_params(labelsize=11)

            if col == 0:
                ax.set_ylabel(
                    f'{bin_name}\n{_FEATURE_LABELS.get(f_y, f_y)}',
                    fontsize=16)
            else:
                ax.set_ylabel('')

            if row == 0:
                f_z_label = _FEATURE_LABELS.get(f_z, f_z)
                ax.set_title(f'{f_z_label}\n{z_label}',
                             fontsize=18, fontweight='bold', pad=8)

            if row == 2:
                ax.set_xlabel(_FEATURE_LABELS.get(f_x, f_x), fontsize=16)

            ax.tick_params(axis='both', labelsize=12)

            for spine in ax.spines.values():
                spine.set_edgecolor('#888888')

    marker_handles = _bin_marker_legend_handles() + [
        Line2D([0], [0], linestyle=':', color='#555555',
               linewidth=1.5, label='P = 0.33 contour'),
        Line2D([0], [0], linestyle='--', color='#555555',
               linewidth=2.0, label='P = 0.50 contour'),
    ]
    fig.legend(handles=marker_handles, loc='lower center', ncol=5,
               fontsize=14, frameon=True, framealpha=0.95,
               edgecolor='#CCCCCC', bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8, w_pad=0.4)
    return fig


# BO TRAJECTORY
# ------------------------------------------------------------------------------

def plot_bo_trajectory(
    optimizer,
    figsize: Tuple[int, int] = (11, 5),
) -> Optional[plt.Figure]:
    """BO recommendation trajectory: target, predicted ± 2σ, and actual size.

    Each completed recommendation is shown at its index:
    - Dashed line + shaded band = target ± tolerance
    - Hollow diamond with error bar = GP-predicted size ± 2σ
    - Filled circle = actual measured size, coloured by squareness bin
    """
    completed = [r for r in optimizer.rec_store.get_all()
                 if r.get('status') == 'completed' and r.get('actual_results')]
    if len(completed) < 2:
        print("Need at least 2 completed recommendations.")
        return None

    completed_sorted = sorted(completed,
                              key=lambda r: r.get('completed_timestamp') or r.get('timestamp', ''))

    threshold = getattr(optimizer.base_optimizer, 'sq_threshold', 0.81) \
        if optimizer.base_optimizer else 0.81

    indices, targets, tols, preds, pred_stds, actuals, bins_ = [], [], [], [], [], [], []

    for i, rec in enumerate(completed_sorted):
        act      = rec.get('actual_results') or {}
        tgt      = rec.get('target') or {}
        pred_mu  = (rec.get('predictions') or {}).get('size_mu',  np.nan)
        pred_std = (rec.get('predictions') or {}).get('size_std', np.nan)
        actual   = act.get('Size', np.nan)
        sq       = act.get('Squareness', np.nan)
        is_cubic = act.get('IsCubic', 1)

        if np.isnan(actual):
            continue

        if is_cubic == 0:
            b = 'multipod'
        elif not np.isnan(sq) and sq >= threshold:
            b = 'highly_cubic'
        else:
            b = 'poorly_cubic'

        indices.append(i + 1)
        targets.append(tgt.get('size', np.nan))
        tols.append(tgt.get('tolerance', 2.0))
        preds.append(pred_mu)
        pred_stds.append(pred_std)
        actuals.append(actual)
        bins_.append(b)

    if not indices:
        print("No completed recommendations with size data.")
        return None

    indices   = np.array(indices, dtype=float)
    targets   = np.array(targets)
    tols      = np.array(tols)
    preds     = np.array(preds)
    pred_stds = np.array(pred_stds)
    actuals   = np.array(actuals)

    bin_color_map = {
        'highly_cubic': COLORS['success'],
        'poorly_cubic': COLORS['tertiary'],
        'multipod':     COLORS['warning'],
    }
    bin_label_map = {
        'highly_cubic': 'Highly cubic',
        'poorly_cubic': 'Poorly cubic',
        'multipod':     'Multipod',
    }

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    _style_ax(ax, title='Bayesian Optimization Trajectory',
              xlabel='Recommendation #', ylabel='Particle Size (nm)')

    # Target band (shaded, one per unique target)
    seen_tgt_keys = set()
    for idx, tgt, tol in zip(indices, targets, tols):
        key = (round(float(tgt), 1), round(float(tol), 1))
        lbl = f'Target ± tolerance' if not seen_tgt_keys else '_nolegend_'
        seen_tgt_keys.add(key)
        ax.fill_between([idx - 0.35, idx + 0.35], tgt - tol, tgt + tol,
                        alpha=0.18, color=COLORS['primary'], zorder=1)
        ax.hlines(tgt, idx - 0.35, idx + 0.35,
                  colors=COLORS['primary'], linewidths=2.0,
                  linestyles='--', zorder=2, label=lbl)

    # Connect actuals
    ax.plot(indices, actuals, '-', color='#CCCCCC', linewidth=1.2, zorder=3)

    # Predicted ± 2σ
    valid_pred = ~np.isnan(preds)
    if valid_pred.any():
        ax.errorbar(indices[valid_pred], preds[valid_pred],
                    yerr=2 * pred_stds[valid_pred],
                    fmt='D', color='#555555', markersize=9,
                    markerfacecolor='white', markeredgewidth=2.0,
                    capsize=5, capthick=1.5, elinewidth=1.5,
                    label='GP predicted ± 2σ', zorder=4)

    # Actual outcomes coloured by bin
    seen_bins: set = set()
    for idx, actual, b in zip(indices, actuals, bins_):
        color = bin_color_map[b]
        lbl   = bin_label_map[b] if b not in seen_bins else '_nolegend_'
        seen_bins.add(b)
        ax.scatter(idx, actual, s=140, color=color,
                   edgecolor='black', linewidth=1.2, zorder=5, label=lbl)

    # Rec ID labels
    for idx, actual, rec in zip(indices, actuals, completed_sorted):
        ax.annotate(rec.get('rec_id', ''), (idx, actual),
                    textcoords='offset points', xytext=(0, 10),
                    fontsize=8, ha='center', color='#555555')

    ax.set_xticks(indices.astype(int))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=_LEGEND_FONT_SIZE,
                  frameon=True, framealpha=0.95, edgecolor='#CCCCCC',
                  loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=len(handles))
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.22)
    return fig


# BO TRAJECTORY BY TARGET SIZE
# ------------------------------------------------------------------------------

def plot_bo_trajectory_by_size(
    optimizer,
    figsize: Tuple[int, int] = (15, 10),
) -> Optional[plt.Figure]:
    """BO trajectory split into subplots by target size.

    Same information as plot_bo_trajectory but with one panel per target size,
    showing how predictions and actuals converge over iterations within each
    size campaign.
    """
    completed = [r for r in optimizer.rec_store.get_all()
                 if r.get('status') == 'completed' and r.get('actual_results')]
    if len(completed) < 2:
        print("Need at least 2 completed recommendations.")
        return None

    completed_sorted = sorted(completed,
                              key=lambda r: r.get('completed_timestamp') or r.get('timestamp', ''))

    threshold = getattr(optimizer.base_optimizer, 'sq_threshold', 0.81) \
        if optimizer.base_optimizer else 0.81

    bin_color_map = {
        'highly_cubic': COLORS['success'],
        'poorly_cubic': COLORS['tertiary'],
        'multipod':     COLORS['warning'],
    }

    groups: dict = {}
    for rec in completed_sorted:
        act = rec.get('actual_results') or {}
        tgt = rec.get('target') or {}
        target_size = tgt.get('size')
        if target_size is None:
            continue

        pred_mu  = (rec.get('predictions') or {}).get('size_mu', np.nan)
        pred_std = (rec.get('predictions') or {}).get('size_std', np.nan)
        actual   = act.get('Size', np.nan)
        sq       = act.get('Squareness', np.nan)
        is_cubic = act.get('IsCubic', 1)

        if np.isnan(actual):
            continue

        if is_cubic == 0:
            b = 'multipod'
        elif not np.isnan(sq) and sq >= threshold:
            b = 'highly_cubic'
        else:
            b = 'poorly_cubic'

        key = int(target_size)
        if key not in groups:
            groups[key] = []
        groups[key].append({
            'rec_id': rec.get('rec_id', ''),
            'target': target_size,
            'tolerance': tgt.get('tolerance', 2.0),
            'pred_mu': pred_mu,
            'pred_std': pred_std,
            'actual': actual,
            'bin': b,
        })

    if not groups:
        print("No completed recommendations with size data.")
        return None

    target_sizes = sorted(groups.keys())
    n_panels = len(target_sizes)
    n_cols = min(n_panels, 3)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor('white')
    fig.suptitle('Bayesian Optimization Trajectory by Target Size',
                 fontsize=15, fontweight='bold', y=0.98)

    for ax in axes.flat:
        ax.set_visible(False)

    for panel_idx, target_size in enumerate(target_sizes):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row, col]
        ax.set_visible(True)

        recs = groups[target_size]
        n = len(recs)
        iterations = np.arange(1, n + 1)
        tol = recs[0]['tolerance']

        ax.axhline(target_size, color=COLORS['primary'], linestyle='--',
                   linewidth=2.0, alpha=0.8, zorder=1)
        ax.axhspan(target_size - tol, target_size + tol,
                   alpha=0.12, color=COLORS['primary'], zorder=0)

        pred_mus = np.array([r['pred_mu'] for r in recs])
        pred_stds = np.array([r['pred_std'] for r in recs])
        actuals_arr = np.array([r['actual'] for r in recs])

        ax.plot(iterations, actuals_arr, '-', color='#CCCCCC', linewidth=1.2, zorder=3)

        valid_pred = ~np.isnan(pred_mus)
        if valid_pred.any():
            ax.errorbar(iterations[valid_pred], pred_mus[valid_pred],
                        yerr=2 * pred_stds[valid_pred],
                        fmt='D', color='#555555', markersize=8,
                        markerfacecolor='white', markeredgewidth=1.8,
                        capsize=4, capthick=1.3, elinewidth=1.3,
                        zorder=4)

        for i, r in enumerate(recs):
            color = bin_color_map[r['bin']]
            ax.scatter(iterations[i], r['actual'], s=120, color=color,
                       edgecolor='black', linewidth=1.0, zorder=5)
            ax.annotate(r['rec_id'], (iterations[i], r['actual']),
                        textcoords='offset points', xytext=(0, 10),
                        fontsize=7, ha='center', color='#555555')

        _style_ax(ax, title=f'Target: {target_size} nm',
                  xlabel='Iteration', ylabel='Particle Size (nm)')
        ax.set_xticks(iterations.astype(int))

        y_pad = max(max(pred_stds[valid_pred]) * 2 if valid_pred.any() else 3, 3)
        all_vals = np.concatenate([pred_mus[valid_pred], actuals_arr,
                                   [target_size - tol, target_size + tol]])
        ax.set_ylim(np.nanmin(all_vals) - y_pad, np.nanmax(all_vals) + y_pad)

    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['primary'], linestyle='--',
                   linewidth=2, label='Target ± tolerance'),
        plt.Line2D([0], [0], marker='D', color='#555555', markerfacecolor='white',
                   markeredgewidth=1.8, markersize=8, linestyle='None',
                   label='GP predicted ± 2σ'),
        plt.scatter([], [], s=120, color=COLORS['success'],
                    edgecolor='black', linewidth=1.0, label='Highly cubic'),
        plt.scatter([], [], s=120, color=COLORS['tertiary'],
                    edgecolor='black', linewidth=1.0, label='Poorly cubic'),
    ]

    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=_LEGEND_FONT_SIZE, frameon=True,
               framealpha=0.95, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=2.0, rect=[0, 0.05, 1, 0.95])
    return fig


# RECOMMENDATION REGRET
# ------------------------------------------------------------------------------

def plot_recommendation_regret(optimizer, figsize: Tuple[int, int] = (13, 4.5)) -> Optional[plt.Figure]:
    """Simple regret over completed recommendations: |actual - target| per property."""
    completed = [r for r in optimizer.rec_store.get_all()
                 if r.get('status') == 'completed' and r.get('actual_results')]
    if len(completed) < 2:
        return None

    df_success = optimizer.exp_store.get_training_data()
    cv_best = df_success['CV'].min() if not df_success.empty else 1.0
    sq_best  = df_success['Squareness'].max() if not df_success.empty else 1.0
    completed_sorted = sorted(completed,
                              key=lambda r: r.get('completed_timestamp') or r.get('timestamp', ''))
    indices = np.arange(1, len(completed_sorted) + 1)

    err_size, err_cv, err_sq = [], [], []
    for r in completed_sorted:
        act = r.get('actual_results') or {}
        tgt = r.get('target') or {}
        err_size.append(abs(act.get('Size', np.nan) - tgt.get('size', np.nan)))
        err_cv.append(act.get('CV', np.nan) - cv_best if act.get('CV') is not None else np.nan)
        err_sq.append(sq_best - act.get('Squareness', np.nan) if act.get('Squareness') is not None else np.nan)

    err_size = np.array(err_size)
    err_cv   = np.array(err_cv)
    err_sq   = np.array(err_sq)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    specs = [
        (axes[0], err_size, 'Size Error from Target (nm)', COLORS['primary']),
        (axes[1], err_cv,   'CV Excess above Best',        COLORS['secondary']),
        (axes[2], err_sq,   'Squareness Gap to Best',      COLORS['tertiary']),
    ]
    titles = ['Size Convergence', 'CV Convergence', 'Squareness Convergence']

    for ax, (_, vals, ylabel, color), title in zip(axes, specs, titles):
        valid = ~np.isnan(vals)
        _style_ax(ax, title=title, xlabel='Recommendation #', ylabel=ylabel)
        if valid.any():
            v = np.maximum(vals[valid], 0)
            x = indices[valid]
            ax.fill_between(x, v, alpha=0.15, color=color)
            ax.plot(x, v, 'o-', color=color, markersize=7, linewidth=2, zorder=3)
            # Running minimum (best so far)
            running_min = np.minimum.accumulate(v)
            ax.plot(x, running_min, '--', color='#444444', linewidth=1.5,
                    alpha=0.7, label='Best so far')
            ax.axhline(0, color='#888888', linewidth=1, linestyle=':')
        _styled_legend(ax, loc='upper right')

    _suptitle(fig, 'Optimization Convergence Over Completed Recommendations')
    fig.tight_layout(pad=1.5)
    return fig


# LOO-CV PARITY
# ------------------------------------------------------------------------------

def plot_loo_parity(
    optimizer,
    figsize: Tuple[int, int] = (13, 4.5),
) -> Optional[plt.Figure]:
    """Three-panel LOO-CV parity comparing Size, CV, and Squareness.

    Shows actual vs leave-one-out predicted values for all cubic training
    experiments.  Error bars are the GP posterior σ at each training point
    (in-sample — included as a proxy for model confidence, not true LOO σ).

    The contrast between Size (typically high R²) and CV / Squareness
    (typically much lower R²) justifies treating the latter two as bin
    constraints in the acquisition rather than direct GP regression targets.
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    if not base.metrics or 'y_pred' not in base.metrics.get('Size', {}):
        print("Running LOO-CV validation…")
        optimizer.validate_models()
    if not base.metrics:
        print("No LOO-CV metrics available after validation.")
        return None

    gp_map  = {'Size': base.gp_size, 'CV': base.gp_cv, 'Squareness': base.gp_sq}
    specs   = [
        ('Size',       'Size (nm)',  COLORS['primary']),
        ('CV',         'CV',         COLORS['secondary']),
        ('Squareness', 'Squareness', COLORS['tertiary']),
    ]
    X_scaled = base.scaler.transform(base.df_cubic[base.features].values)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')

    for ax, (prop, label, color) in zip(axes, specs):
        if prop not in base.metrics or 'y_pred' not in base.metrics[prop]:
            _style_ax(ax, title=label,
                      xlabel=f'Actual {label}', ylabel=f'LOO Predicted {label}')
            ax.text(0.5, 0.5, 'No LOO data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            continue

        y_actual = base.df_cubic[prop].values
        y_loo    = base.metrics[prop]['y_pred']
        r2_loo   = float(base.metrics[prop]['r2'])
        mae_loo  = mean_absolute_error(y_actual, y_loo)

        try:
            _, gp_std = gp_map[prop].predict(X_scaled, return_std=True)
        except Exception:
            gp_std = np.zeros(len(y_actual))

        _style_ax(ax, title=label,
                  xlabel=f'Actual {label}', ylabel=f'LOO Predicted {label}')

        all_vals = np.concatenate([y_actual, y_loo])
        margin   = 0.08 * (all_vals.max() - all_vals.min())
        lims     = [all_vals.min() - margin, all_vals.max() + margin]

        ax.plot(lims, lims, color='#444444', lw=1.5, linestyle='--',
                label='Perfect prediction', zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')

        ax.errorbar(y_actual, y_loo, yerr=gp_std,
                    fmt='o', color=color, markersize=7, alpha=0.75,
                    capsize=3, capthick=1.2, elinewidth=1.0,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label='LOO prediction ± GP σ', zorder=4)

        ax.text(0.04, 0.96,
                f'LOO R² = {r2_loo:.2f}\nMAE = {mae_loo:.3f}',
                transform=ax.transAxes, va='top', fontsize=10,
                fontweight='bold', bbox=_ANN_BOX)

        _styled_legend(ax, loc='lower right')

    _suptitle(fig, 'LOO-CV Parity  ·  Size, CV, and Squareness GP Surrogates')
    fig.tight_layout(pad=1.5)
    return fig


# FEASIBILITY LANDSCAPE (2-D)
# ------------------------------------------------------------------------------

def plot_feasibility_landscape(
    optimizer,
    n_grid: int = 40,
    figsize: Tuple[int, int] = (16, 4),
) -> Optional[plt.Figure]:
    """Four-panel 2-D heatmap of individual classifier probabilities and their product.

    Panels (left → right):
        P(Has Product)  ·  P(Phase Pure)  ·  P(Is Cubic)  ·  P(Feasible) [joint]

    Axes span the two most-important Size-model features; all other features
    are fixed at their cubic-training-data medians.  Dashed contours mark
    P = 0.5.  Training data are overlaid: white circles = successful
    experiments, orange crosses = failed (no product).
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    top_2 = _get_top_features(base, 'Size', n=2)
    if len(top_2) < 2:
        print("Need at least 2 features for the feasibility landscape.")
        return None
    f_x, f_y = top_2

    features = base.features
    bounds   = base.bounds
    medians  = base.df_cubic[features].median()
    ix, iy   = features.index(f_x), features.index(f_y)

    x_range = np.linspace(bounds[f_x][0], bounds[f_x][1], n_grid)
    y_range = np.linspace(bounds[f_y][0], bounds[f_y][1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    X_feat = np.tile(medians.values, (X_grid.size, 1))
    X_feat[:, ix] = X_grid.ravel()
    X_feat[:, iy] = Y_grid.ravel()

    preds = base.predict(X_feat)

    panels = [
        ('p_product',  'P(Has Product)'),
        ('p_pure',     'P(Phase Pure)'),
        ('p_cubic',    'P(Is Cubic)'),
        ('p_feasible', 'P(Feasible)  [joint]'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    fig.patch.set_facecolor('white')

    xl = _FEATURE_LABELS.get(f_x, f_x)
    yl = _FEATURE_LABELS.get(f_y, f_y)

    df_plot = base.df_all
    has_feat_cols = (f_x in df_plot.columns and f_y in df_plot.columns)

    for col, (ax, (key, title)) in enumerate(zip(axes, panels)):
        Z = preds[key].reshape(n_grid, n_grid)

        im = ax.imshow(
            Z, origin='lower', aspect='auto',
            extent=[bounds[f_x][0], bounds[f_x][1],
                    bounds[f_y][0], bounds[f_y][1]],
            cmap='RdYlGn', vmin=0, vmax=1,
            interpolation='bilinear', zorder=1,
        )
        try:
            ax.contour(X_grid, Y_grid, Z, levels=[0.5],
                       colors=['#333333'], linewidths=[1.5],
                       linestyles=['--'], zorder=3)
        except Exception:
            pass

        if has_feat_cols:
            if 'HasProduct' in df_plot.columns:
                fail = df_plot['HasProduct'] == 0
                succ = ~fail
                if fail.any():
                    ax.scatter(df_plot.loc[fail, f_x], df_plot.loc[fail, f_y],
                               c=COLORS['warning'], s=55, marker='X', alpha=0.85,
                               edgecolor='black', linewidth=0.8, zorder=5,
                               label='Failed' if col == 0 else '_nolegend_')
                if succ.any():
                    ax.scatter(df_plot.loc[succ, f_x], df_plot.loc[succ, f_y],
                               c='white', s=40, marker='o', alpha=0.75,
                               edgecolor='black', linewidth=0.8, zorder=5,
                               label='Success' if col == 0 else '_nolegend_')
            else:
                ax.scatter(df_plot[f_x], df_plot[f_y],
                           c='white', s=40, marker='o', alpha=0.75,
                           edgecolor='black', linewidth=0.8, zorder=5,
                           label='Training data' if col == 0 else '_nolegend_')
        else:
            ax.scatter(base.df_success[f_x], base.df_success[f_y],
                       c='white', s=40, marker='o', alpha=0.75,
                       edgecolor='black', linewidth=0.8, zorder=5,
                       label='Training data' if col == 0 else '_nolegend_')

        _style_ax(ax, title=title, xlabel=xl)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=10)

    axes[0].set_ylabel(yl, fontsize=13, labelpad=6)
    _styled_legend(axes[0], loc='lower right')

    _suptitle(fig, 'Feasibility Landscape  ·  GP Classifier Probability Components')
    fig.tight_layout(pad=1.5)
    return fig


# SQUARENESS BINNING
# ------------------------------------------------------------------------------

def plot_squareness_binning(
    optimizer,
    figsize: Tuple[int, int] = (12, 4.5),
) -> Optional[plt.Figure]:
    """Two-panel figure motivating the Squareness → discrete-bin strategy.

    Left:  histogram of Squareness values coloured by morphology bin
           (multipod / highly cubic / poorly cubic) with the Otsu threshold
           marked as a dashed vertical line.
    Right: Squareness vs Particle Size scatter coloured by bin, with the
           Otsu threshold shown as a horizontal dashed line.

    Together the panels show that Squareness separates naturally at the
    Otsu threshold, making it a robust binary constraint rather than a
    reliable continuous regression target in the acquisition.
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    df_s = base.df_success
    if df_s.empty or len(df_s) < 3:
        print("Insufficient data for squareness binning plot.")
        return None

    threshold = base.sq_threshold
    is_cubic = _get_is_cubic(df_s)
    bin_masks = _get_bin_masks(df_s, is_cubic, threshold)
    colors = _bin_colors()
    bin_specs = list(zip(bin_masks, colors, _BIN_NAMES, _BIN_MARKERS))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    # --- Left: Squareness histogram by bin ---
    ax1 = axes[0]
    _style_ax(ax1, title='Squareness Distribution',
              xlabel='Squareness', ylabel='Count')

    sq_valid = df_s['Squareness'].dropna()
    bins = np.linspace(float(sq_valid.min()), float(sq_valid.max()), 18)

    for mask, color, label, _ in bin_specs:
        sub = df_s.loc[mask, 'Squareness'].dropna()
        if not sub.empty:
            ax1.hist(sub, bins=bins, color=color, alpha=0.75,
                     edgecolor='white', linewidth=0.6, zorder=2, label=label)

    ax1.axvline(threshold, color='#333333', linestyle='--',
                linewidth=2.0, zorder=5,
                label=f'Otsu threshold ({threshold:.3f})')
    ax1.text(0.97, 0.97, f'Otsu threshold\n= {threshold:.3f}',
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=9, bbox=_ANN_BOX)
    _styled_legend(ax1, loc='upper left')

    # --- Right: Squareness vs Size scatter ---
    ax2 = axes[1]
    _style_ax(ax2, title='Squareness vs Particle Size',
              xlabel='Particle Size (nm)', ylabel='Squareness')

    for mask, color, label, marker in bin_specs:
        sub = df_s[mask]
        if not sub.empty:
            ax2.scatter(sub['Size'], sub['Squareness'],
                        c=color, s=70, marker=marker, alpha=0.80,
                        edgecolor='white', linewidth=0.8, zorder=4, label=label)

    ax2.axhline(threshold, color='#333333', linestyle='--',
                linewidth=2.0, zorder=5)
    ax2.text(0.97, 0.04, f'Otsu threshold ({threshold:.3f})',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=9, color='#333333', bbox=_ANN_BOX)
    _styled_legend(ax2, loc='lower right')

    _suptitle(fig, 'Squareness Binning  ·  Otsu Threshold Strategy')
    fig.tight_layout(pad=1.5)
    return fig


# ACQUISITION DECOMPOSITION
# ------------------------------------------------------------------------------

def plot_acquisition_decomposition(
    optimizer,
    target_size: float = 20.0,
    size_tol: float = 2.5,
    squareness_bin: str = 'highly_cubic',
    n_grid: int = 40,
    figsize: Tuple[int, int] = (16, 4),
) -> Optional[plt.Figure]:
    """Four-panel decomposition of the BO acquisition function.

    Each panel is a 2-D heatmap over the two most-important Size-model
    features (all others fixed at training-data medians):

        P(size ∈ target)  ·  P(product ∧ pure)  ·  P(bin)  ·  Total score

    The total acquisition is the element-wise product of the first three
    terms.  Training data are overlaid as white circles.

    Parameters
    ----------
    target_size : float
        Target particle size in nm (used for P(size) panel).
    size_tol : float
        Half-width of the size acceptance band in nm.
    squareness_bin : {'highly_cubic', 'poorly_cubic', 'multipod'}
        Morphology constraint shown in the P(bin) panel.
    """
    if optimizer.base_optimizer is None:
        print("Base optimizer not initialized.")
        return None
    base = optimizer.base_optimizer

    top_2 = _get_top_features(base, 'Size', n=2)
    if len(top_2) < 2:
        print("Need at least 2 features for acquisition decomposition.")
        return None
    f_x, f_y = top_2

    features = base.features
    bounds   = base.bounds
    medians  = base.df_success[features].median()
    ix, iy   = features.index(f_x), features.index(f_y)

    x_range = np.linspace(bounds[f_x][0], bounds[f_x][1], n_grid)
    y_range = np.linspace(bounds[f_y][0], bounds[f_y][1], n_grid)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    X_feat = np.tile(medians.values, (X_grid.size, 1))
    X_feat[:, ix] = X_grid.ravel()
    X_feat[:, iy] = Y_grid.ravel()

    acq = base.acquisition(X_feat, target_size, size_tol,
                           squareness_bin=squareness_bin)

    bin_label = squareness_bin.replace('_', ' ').title()
    panels = [
        ('p_size',     f'P(size $\\in$ {target_size}\u00b1{size_tol} nm)', 'viridis',  (0, 1)),
        ('p_feasible', 'P(product $\\wedge$ pure)',                          'viridis',  (0, 1)),
        ('p_bin',      f'P(bin = {bin_label})',                   'viridis',  (0, 1)),
        ('total',      'Total Acquisition  [product]',            'inferno',  None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    fig.patch.set_facecolor('white')

    xl = _FEATURE_LABELS.get(f_x, f_x)
    yl = _FEATURE_LABELS.get(f_y, f_y)

    for col, (ax, (key, title, cmap_name, vlim)) in enumerate(zip(axes, panels)):
        Z = acq[key].reshape(n_grid, n_grid)
        vmin_val = vlim[0] if vlim else 0.0
        vmax_val = vlim[1] if vlim else float(np.nanmax(Z)) or 1.0

        im = ax.imshow(
            Z, origin='lower', aspect='auto',
            extent=[bounds[f_x][0], bounds[f_x][1],
                    bounds[f_y][0], bounds[f_y][1]],
            cmap=cmap_name, vmin=vmin_val, vmax=vmax_val,
            interpolation='bilinear', zorder=1,
        )

        ax.scatter(base.df_success[f_x], base.df_success[f_y],
                   c='white', s=40, marker='o', alpha=0.70,
                   edgecolor='black', linewidth=0.8, zorder=5,
                   label='Training data' if col == 0 else '_nolegend_')

        _style_ax(ax, title=title, xlabel=xl)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    axes[0].set_ylabel(yl, fontsize=11, labelpad=6)
    _styled_legend(axes[0], loc='lower right')

    bin_str = squareness_bin.replace('_', ' ')
    _suptitle(fig,
              f'Acquisition Decomposition  ·  target {target_size} nm ± {size_tol} nm  ·  bin = {bin_str}')
    fig.tight_layout(pad=1.5)
    return fig


# OPTIMIZATION PROGRESS (CONVERGENCE EVIDENCE)
# ------------------------------------------------------------------------------

def plot_optimization_progress(
    optimizer,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[plt.Figure]:
    """Publication-ready evidence that BO optimized particle synthesis.

    Focuses on the three quantities the acquisition function actually uses:
      A) Size: prediction error decreasing over BO iterations (model learning)
      B) Size: cumulative fraction within target tolerance (hit rate)
      C) Feasibility: product formation rate comparison (initial vs BO)
      D) Squareness binning: morphology bin composition (initial vs BO)
    """
    completed = [r for r in optimizer.rec_store.get_all()
                 if r.get('status') == 'completed' and r.get('actual_results')]
    if len(completed) < 4:
        print("Need at least 4 completed recommendations.")
        return None

    completed_sorted = sorted(
        completed,
        key=lambda r: r.get('completed_timestamp') or r.get('timestamp', ''))

    df_all = optimizer.exp_store.get_all()
    if df_all.empty:
        print("No experiments in store.")
        return None

    threshold = getattr(optimizer.base_optimizer, 'sq_threshold', 0.81) \
        if optimizer.base_optimizer else 0.81

    # --- Extract BO recommendation data ---
    pred_errors, target_errors, tolerances = [], [], []
    for rec in completed_sorted:
        act = rec.get('actual_results') or {}
        tgt = rec.get('target') or {}
        pred = rec.get('predictions') or {}
        actual_size = act.get('Size')
        target_size = tgt.get('size')
        pred_mu = pred.get('size_mu')
        tol = tgt.get('tolerance', 2.0)

        if actual_size is None or target_size is None:
            pred_errors.append(np.nan)
            target_errors.append(np.nan)
            tolerances.append(tol)
            continue

        pred_errors.append(abs(actual_size - pred_mu) if pred_mu is not None else np.nan)
        target_errors.append(abs(actual_size - target_size))
        tolerances.append(tol)

    pred_errors = np.array(pred_errors)
    target_errors = np.array(target_errors)
    tolerances = np.array(tolerances)
    indices = np.arange(1, len(completed_sorted) + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    # --- Panel A: Size prediction error over BO iterations ---
    ax = axes[0, 0]
    _style_ax(ax, title='GP Model Learning: Size Prediction Error',
              xlabel='BO Iteration', ylabel='|Actual − Predicted| (nm)')

    valid_pred = ~np.isnan(pred_errors)
    if valid_pred.any():
        x_valid = indices[valid_pred]
        y_valid = pred_errors[valid_pred]
        running_min = np.minimum.accumulate(y_valid)

        ax.bar(x_valid, y_valid, width=0.7, color=COLORS['primary'],
               alpha=0.5, edgecolor='white', linewidth=0.5, label='Per-iteration error')
        ax.plot(x_valid, running_min, '-', color=COLORS['warning'],
                linewidth=2.5, zorder=5, label='Running best (min)')

        window = min(5, len(y_valid))
        if len(y_valid) >= window:
            rolling_mean = pd.Series(y_valid).rolling(window, min_periods=1).mean().values
            ax.plot(x_valid, rolling_mean, '-', color=COLORS['secondary'],
                    linewidth=2.0, zorder=4, label=f'Rolling mean (n={window})')

        ax.axhline(0, color='#888888', linewidth=0.8, linestyle=':')
        mean_err = np.nanmean(y_valid)
        ax.text(0.97, 0.95,
                f'Mean |error|: {mean_err:.1f} nm\n'
                f'Last 5 mean: {np.nanmean(y_valid[-5:]):.1f} nm',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=_ANN_BOX)
    _styled_legend(ax, loc='upper left')

    # --- Panel B: Cumulative target hit rate ---
    ax = axes[0, 1]
    _style_ax(ax, title='Size Target Achievement (Cumulative Hit Rate)',
              xlabel='BO Iteration', ylabel='Fraction Within Tolerance')

    valid_tgt = ~np.isnan(target_errors)
    if valid_tgt.any():
        hits = (target_errors[valid_tgt] <= tolerances[valid_tgt]).astype(float)
        cumulative_hit_rate = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        x_tgt = indices[valid_tgt]

        ax.plot(x_tgt, cumulative_hit_rate, 'o-', color=COLORS['success'],
                markersize=7, linewidth=2.0, markeredgecolor='white',
                markeredgewidth=0.8, zorder=4)
        ax.fill_between(x_tgt, 0, cumulative_hit_rate,
                        alpha=0.12, color=COLORS['success'])
        ax.axhline(1.0, color='#888888', linewidth=0.8, linestyle=':')
        ax.set_ylim(-0.05, 1.1)

        final_rate = cumulative_hit_rate[-1]
        total_hits = int(hits.sum())
        ax.text(0.97, 0.15,
                f'{total_hits}/{len(hits)} within tolerance\n'
                f'Overall hit rate: {final_rate:.0%}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, bbox=_ANN_BOX)

    # --- Panel C: Feasibility comparison ---
    ax = axes[1, 0]
    _style_ax(ax, title='Feasibility: Product Formation Rate',
              xlabel='', ylabel='Fraction Producing Product')

    df_init = df_all[df_all['source'] == 'imported']
    df_bo = df_all[df_all['source'] != 'imported']

    init_rate = df_init['HasProduct'].mean() if len(df_init) > 0 else 0
    bo_rate = df_bo['HasProduct'].mean() if len(df_bo) > 0 else 0
    n_init = len(df_init)
    n_bo = len(df_bo)

    bars = ax.bar([0, 1], [init_rate, bo_rate], width=0.55,
                  color=[COLORS['neutral'], COLORS['success']],
                  edgecolor='white', linewidth=1.5, alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Initial\n(n={n_init})', f'BO-guided\n(n={n_bo})'])
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color='#888888', linewidth=0.8, linestyle=':')

    for bar, rate, n in zip(bars, [init_rate, bo_rate], [n_init, n_bo]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{rate:.0%}\n({int(rate*n)}/{n})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Panel D: Squareness bin composition ---
    ax = axes[1, 1]
    _style_ax(ax, title='Morphology Bin Composition',
              xlabel='', ylabel='Fraction of Experiments')

    def classify_bins(df, thresh):
        has_product = df[df['HasProduct'] == 1]
        sq = has_product['Squareness'].values
        poly = has_product.get('Polymorph', pd.Series(dtype=str))
        n_total = len(has_product)
        if n_total == 0:
            return {'highly_cubic': 0, 'poorly_cubic': 0, 'multipod': 0, 'no_product': 0}
        n_multipod = ((poly == 'multipod') | (sq < 0.7)).sum() if len(poly) > 0 else 0
        cubic_mask = ~((poly == 'multipod') | (sq < 0.7)) if len(poly) > 0 else np.ones(n_total, dtype=bool)
        cubic_sq = sq[cubic_mask] if cubic_mask.any() else np.array([])
        n_highly = int((cubic_sq >= thresh).sum()) if len(cubic_sq) > 0 else 0
        n_poorly = int((cubic_sq < thresh).sum()) if len(cubic_sq) > 0 else 0
        return {
            'highly_cubic': n_highly / n_total,
            'poorly_cubic': n_poorly / n_total,
            'multipod': n_multipod / n_total,
        }

    init_bins = classify_bins(df_init, threshold)
    bo_bins = classify_bins(df_bo, threshold)

    bin_names = ['Highly cubic', 'Poorly cubic', 'Multipod']
    bin_keys = ['highly_cubic', 'poorly_cubic', 'multipod']
    bin_colors = [COLORS['success'], COLORS['tertiary'], COLORS['warning']]

    x_pos = np.array([0, 1])
    bar_width = 0.2
    for i, (key, name, color) in enumerate(zip(bin_keys, bin_names, bin_colors)):
        vals = [init_bins[key], bo_bins[key]]
        offset = (i - 1) * bar_width
        ax.bar(x_pos + offset, vals, bar_width * 0.9, color=color,
               edgecolor='white', linewidth=0.8, label=name, alpha=0.85)

    ax.set_xticks([0, 1])
    n_init_prod = int(df_init['HasProduct'].sum())
    n_bo_prod = int(df_bo['HasProduct'].sum())
    ax.set_xticklabels([f'Initial\n(n={n_init_prod} with product)',
                        f'BO-guided\n(n={n_bo_prod} with product)'])
    ax.set_ylim(0, 1.05)
    _styled_legend(ax, loc='upper left')

    hc_init = init_bins['highly_cubic']
    hc_bo = bo_bins['highly_cubic']
    ax.text(0.97, 0.95,
            f'Highly cubic: {hc_init:.0%} → {hc_bo:.0%}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold', bbox=_ANN_BOX)

    _suptitle(fig, 'Bayesian Optimization Convergence: Size, Feasibility & Morphology')
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.92)
    return fig


# GP HYPERPARAMETER TRACKING & LENGTHSCALE EVOLUTION
# ------------------------------------------------------------------------------

def log_gp_hyperparameters(optimizer, iteration: int, log_path: Path) -> dict:
    """Record fitted GP kernel hyperparameters at the current BO iteration.

    Appends a JSONL record to ``log_path`` containing lengthscales,
    signal variance, noise level, and log-marginal likelihood for each
    regression GP.  Returns the record dict.
    """
    import json

    base = optimizer.base_optimizer
    if base is None:
        return {}

    record = {
        'iteration': iteration,
        'n_training': len(base.df_cubic),
        'feature_mode': base.feature_mode,
        'features': list(base.features),
        'use_ard': base.use_ard,
    }

    for name, gp in [('Size', base.gp_size),
                     ('CV', base.gp_cv),
                     ('Squareness', base.gp_sq)]:
        try:
            ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale).tolist()
            sigma_f = float(gp.kernel_.k1.k1.constant_value)
            noise = float(gp.kernel_.k2.noise_level)
            lml = float(gp.log_marginal_likelihood_value_)

            record[name] = {
                'lengthscales': ls,
                'signal_variance': sigma_f,
                'noise_level': noise,
                'log_marginal_likelihood': lml,
            }
        except AttributeError:
            pass

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    return record


def plot_lengthscale_evolution(
    log_path: Path,
    figsize: Tuple[int, int] = (14, 5),
    properties: Optional[list] = None,
) -> Optional[plt.Figure]:
    """Plot GP lengthscale evolution over BO iterations from a JSONL log.

    Parameters
    ----------
    log_path : Path
        Path to the JSONL file written by ``log_gp_hyperparameters``.
    properties : list, optional
        Which GP models to plot.  Default: ['Size', 'CV', 'Squareness'].

    Returns
    -------
    fig : matplotlib Figure or None if log file is missing/empty.
    """
    import json

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"No hyperparameter log found at {log_path}")
        return None

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("Hyperparameter log is empty.")
        return None

    if properties is None:
        properties = ['Size', 'CV', 'Squareness']

    features = records[0].get('features', [])
    n_props = len(properties)
    fig, axes = plt.subplots(1, n_props, figsize=figsize, sharex=True)
    if n_props == 1:
        axes = [axes]

    prop_colors = {'Size': COLORS['primary'], 'CV': COLORS['secondary'],
                   'Squareness': COLORS['tertiary']}
    feat_cmap = plt.cm.Set2(np.linspace(0, 1, max(len(features), 1)))

    for ax, prop in zip(axes, properties):
        iterations = []
        ls_data = []
        noise_data = []
        for rec in records:
            if prop in rec:
                iterations.append(rec['iteration'])
                ls_data.append(rec[prop]['lengthscales'])
                noise_data.append(rec[prop]['noise_level'])

        if not iterations:
            _style_ax(ax, title=f'{prop} GP', xlabel='BO Iteration',
                      ylabel='Lengthscale')
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='grey')
            continue

        ls_array = np.array(ls_data)
        _style_ax(ax, title=f'{prop} GP', xlabel='BO Iteration',
                  ylabel='Lengthscale (standardized space)')

        if ls_array.ndim == 1 or ls_array.shape[1] == 1:
            ls_flat = ls_array.flatten()
            ax.plot(iterations, ls_flat, 'o-',
                    color=prop_colors.get(prop, COLORS['primary']),
                    markersize=6, linewidth=1.5, label='shared $\\ell$')
        else:
            for j, feat in enumerate(features):
                ax.plot(iterations, ls_array[:, j], 'o-',
                        color=feat_cmap[j], markersize=4, linewidth=1.2,
                        label=feat)

        # Bounds reference lines
        use_ard = records[0].get('use_ard', False)
        lb = 0.1 if use_ard else 0.3
        ax.axhline(lb, color='#999999', ls='--', alpha=0.5, linewidth=0.8,
                   label=f'lower bound ({lb})')
        ax.axhline(10.0, color='#999999', ls=':', alpha=0.5, linewidth=0.8,
                   label='upper bound (10)')

        ax.set_yscale('log')
        ax.set_ylim(lb * 0.8, 12)
        _styled_legend(ax, fontsize=8, loc='best')

    _suptitle(fig, 'GP Lengthscale Evolution Across BO Campaign')
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.88)
    return fig


def plot_gp_hyperparameter_summary(
    optimizer,
    figsize: Tuple[int, int] = (14, 5),
) -> Optional[plt.Figure]:
    """Snapshot of current GP kernel hyperparameters as a bar chart.

    Shows per-feature lengthscales (or shared lengthscale) for each
    regression GP, plus noise levels.  Useful for a single-iteration
    summary in the SI.
    """
    base = optimizer.base_optimizer
    if base is None:
        print("Base optimizer not initialized.")
        return None

    features = list(base.features)
    n_feat = len(features)
    models = [('Size', base.gp_size), ('CV', base.gp_cv),
              ('Squareness', base.gp_sq)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    prop_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for ax, (name, gp), color in zip(axes, models, prop_colors):
        try:
            ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale)
            noise = float(gp.kernel_.k2.noise_level)
            sigma_f = float(gp.kernel_.k1.k1.constant_value)
        except AttributeError:
            _style_ax(ax, title=f'{name} GP')
            ax.text(0.5, 0.5, 'Cannot extract kernel params',
                    transform=ax.transAxes, ha='center', fontsize=10)
            continue

        if ls.shape[0] == 1:
            labels = features
            vals = [float(ls[0])] * n_feat
            ax.axhline(float(ls[0]), color=color, ls='--', alpha=0.4,
                       linewidth=1.5)
        else:
            labels = features
            vals = ls.tolist()

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, vals, color=color, alpha=0.7, edgecolor='white',
                      linewidth=0.8)

        _style_ax(ax, title=f'{name} GP\n$\\sigma_f^2$={sigma_f:.2f}, '
                  f'$\\sigma_n^2$={noise:.4f}',
                  xlabel='Feature', ylabel='Lengthscale')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)

        use_ard = base.use_ard
        lb = 0.1 if use_ard else 0.3
        ax.axhline(lb, color='#999999', ls='--', alpha=0.5, linewidth=0.8)
        ax.axhline(10.0, color='#999999', ls=':', alpha=0.5, linewidth=0.8)
        ax.set_ylim(0, max(vals) * 1.3)

    _suptitle(fig, 'Fitted GP Kernel Hyperparameters (Current Iteration)')
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.82)
    return fig
