"""
Replay the completed BO campaign to reconstruct GP hyperparameter evolution.

THIS SCRIPT IS PURELY READ-ONLY:
  - Reads from data/experiments.json (never writes to it)
  - Builds temporary in-memory optimizers (nothing persisted to data/)
  - Outputs ONLY to outputs/si_hyperparameter_log.jsonl and outputs/figures/

Usage:
    cd CuVS_BO_Code
    python scripts/replay_lengthscale_evolution.py

Produces:
    outputs/si_hyperparameter_log.jsonl   — JSONL with kernel params per iteration
    outputs/figures/SI_lengthscale_evolution.pdf
    outputs/figures/SI_hyperparameter_final.pdf
    outputs/si_hyperparameter_table.csv   — Final fitted values for SI table
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

sys.path.insert(0, str(SRC_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')

from optimizer import Cu3VS4Optimizer, make_gp_regressor
from features import add_chemical_features
from config import SYNTHESIS_FEATURES, PUBLICATION_STYLE

plt.rcParams.update(PUBLICATION_STYLE)

# Output paths (NEVER writes to DATA_DIR)
HP_LOG = OUTPUT_DIR / "si_hyperparameter_log.jsonl"
HP_TABLE = OUTPUT_DIR / "si_hyperparameter_table.csv"
FIG_EVOLUTION = FIG_DIR / "SI_lengthscale_evolution.pdf"
FIG_FINAL = FIG_DIR / "SI_hyperparameter_final.pdf"


def load_experiments():
    """Load experiments from JSON (read-only)."""
    with open(DATA_DIR / "experiments.json", 'r') as f:
        store = json.load(f)

    imported = []
    bo_guided = []

    for exp in store['experiments']:
        row = {
            'exp_id': exp['exp_id'],
            'source': exp['source'],
            'timestamp': exp['timestamp'],
            **exp['conditions'],
            **exp['results'],
        }
        if 'precursors' in exp:
            row['Cu_precursor'] = exp['precursors'].get('Cu_precursor', 'CuI')
            row['Metal_precursor'] = exp['precursors'].get('Metal_precursor', 'VO(acac)2')
        else:
            row['Cu_precursor'] = 'CuI'
            row['Metal_precursor'] = 'VO(acac)2'

        if exp['source'] == 'imported':
            imported.append(row)
        elif exp['source'] == 'recommendation':
            bo_guided.append(row)

    df_imported = pd.DataFrame(imported)
    df_bo = pd.DataFrame(bo_guided)

    # Sort BO experiments by timestamp
    if not df_bo.empty:
        df_bo = df_bo.sort_values('timestamp').reset_index(drop=True)

    return df_imported, df_bo


def build_optimizer_snapshot(df):
    """Build a temporary optimizer from a DataFrame (in-memory only)."""
    df_work = df.copy()

    if 'Polymorph' in df_work.columns:
        df_work['IsCubic'] = (
            df_work['Polymorph'].fillna('').astype(str).str.lower().str.strip() == 'cubic'
        ).astype(int)
    else:
        df_work['IsCubic'] = 0

    if 'Cu_V_ratio' not in df_work.columns:
        df_work = add_chemical_features(df_work)

    # Backwards compatibility: older snapshots may store this under 'GSD'.
    if 'CV' not in df_work.columns and 'GSD' in df_work.columns:
        df_work['CV'] = df_work['GSD']

    opt = Cu3VS4Optimizer(
        df=df_work,
        feature_mode='synthesis',
        validate=False,
    )
    return opt


def extract_hyperparameters(opt):
    """Extract kernel hyperparameters from fitted GPs."""
    record = {}
    for name, gp in [('Size', opt.gp_size), ('CV', opt.gp_cv),
                     ('Squareness', opt.gp_sq)]:
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
                'kernel_string': str(gp.kernel_),
            }
        except AttributeError as e:
            record[name] = {'error': str(e)}
    return record


def replay_campaign():
    """Replay the BO campaign, logging hyperparameters at each stage."""
    print("=" * 70)
    print("REPLAYING BO CAMPAIGN FOR HYPERPARAMETER EVOLUTION")
    print("  Read-only: outputs go to outputs/ only")
    print("=" * 70)

    df_imported, df_bo = load_experiments()
    print(f"\n  Imported experiments: {len(df_imported)}")
    print(f"  BO-guided experiments: {len(df_bo)}")

    if HP_LOG.exists():
        HP_LOG.unlink()
        print(f"  Cleared old log: {HP_LOG}")

    records = []

    # Iteration 0: initial dataset only
    print(f"\n  [Iter 0] Building model on {len(df_imported)} imported experiments...")
    try:
        opt = build_optimizer_snapshot(df_imported)
        hp = extract_hyperparameters(opt)
        entry = {
            'iteration': 0,
            'n_total': len(df_imported),
            'n_cubic': len(opt.df_cubic),
            'label': 'initial dataset',
            **hp,
        }
        records.append(entry)
        print(f"    n_cubic={len(opt.df_cubic)}, Size ℓ={hp['Size']['lengthscales']}")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Incremental additions: add BO experiments in pairs (as they were recommended)
    # Group by recommendation batch (same timestamp prefix = same batch)
    if not df_bo.empty:
        # Identify batches by clustering timestamps within 30 min of each other
        df_bo['ts'] = pd.to_datetime(df_bo['timestamp'])
        df_bo = df_bo.sort_values('ts').reset_index(drop=True)

        batches = []
        current_batch = [0]
        for i in range(1, len(df_bo)):
            delta = (df_bo.loc[i, 'ts'] - df_bo.loc[current_batch[-1], 'ts']).total_seconds()
            if delta < 1800:  # 30 min
                current_batch.append(i)
            else:
                batches.append(current_batch)
                current_batch = [i]
        batches.append(current_batch)

        print(f"\n  Detected {len(batches)} BO batches:")
        for i, batch_idx in enumerate(batches):
            batch_ids = df_bo.loc[batch_idx, 'exp_id'].tolist()
            print(f"    Batch {i+1}: {batch_ids}")

        # Replay incrementally
        df_cumulative = df_imported.copy()
        for batch_num, batch_idx in enumerate(batches, 1):
            batch_rows = df_bo.loc[batch_idx]
            df_cumulative = pd.concat([df_cumulative, batch_rows], ignore_index=True)

            print(f"\n  [Iter {batch_num}] +{len(batch_idx)} experiments "
                  f"→ {len(df_cumulative)} total...")
            try:
                opt = build_optimizer_snapshot(df_cumulative)
                hp = extract_hyperparameters(opt)
                entry = {
                    'iteration': batch_num,
                    'n_total': len(df_cumulative),
                    'n_cubic': len(opt.df_cubic),
                    'label': f'batch {batch_num} ({df_bo.loc[batch_idx[0], "exp_id"]}–'
                             f'{df_bo.loc[batch_idx[-1], "exp_id"]})',
                    **hp,
                }
                records.append(entry)
                print(f"    n_cubic={len(opt.df_cubic)}, "
                      f"Size ℓ={hp['Size']['lengthscales']}")
            except Exception as e:
                print(f"    FAILED: {e}")

    # Write JSONL log
    with open(HP_LOG, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + '\n')
    print(f"\n  ✓ Hyperparameter log written to: {HP_LOG}")

    return records


def make_evolution_figure(records):
    """Generate the SI lengthscale evolution figure."""
    from config import COLORS

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    properties = ['Size', 'CV', 'Squareness']
    prop_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for ax, prop, color in zip(axes, properties, prop_colors):
        iterations = []
        lengthscales = []
        noise_levels = []

        for rec in records:
            if prop in rec and 'lengthscales' in rec[prop]:
                iterations.append(rec['iteration'])
                lengthscales.append(rec[prop]['lengthscales'][0])  # isotropic: single value
                noise_levels.append(rec[prop]['noise_level'])

        if not iterations:
            continue

        ax.plot(iterations, lengthscales, 'o-', color=color,
                markersize=7, linewidth=1.8, label='lengthscale $\\ell$',
                zorder=5)

        # Noise level on secondary axis
        ax2 = ax.twinx()
        ax2.plot(iterations, noise_levels, 's--', color='grey',
                 markersize=4, linewidth=1.0, alpha=0.6, label='noise $\\sigma_n^2$')
        ax2.set_ylabel('Noise level', fontsize=10, color='grey')
        ax2.tick_params(axis='y', labelcolor='grey', labelsize=9)
        ax2.spines['right'].set_color('grey')
        ax2.spines['right'].set_alpha(0.5)

        # Bounds
        ax.axhline(0.3, color='#999', ls='--', alpha=0.4, linewidth=0.8)
        ax.axhline(10.0, color='#999', ls=':', alpha=0.4, linewidth=0.8)
        ax.fill_between([min(iterations)-0.5, max(iterations)+0.5],
                        0.3, 10.0, alpha=0.03, color=color)

        ax.set_xlabel('BO Iteration', fontsize=12)
        ax.set_ylabel('Lengthscale (std. space)', fontsize=12)
        ax.set_title(f'{prop} GP', fontsize=13, fontweight='bold')
        ax.set_xticks(iterations)
        ax.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')

    fig.suptitle('GP Kernel Hyperparameter Evolution Across BO Campaign',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIG_EVOLUTION, bbox_inches='tight', dpi=300)
    print(f"  ✓ Evolution figure saved: {FIG_EVOLUTION}")
    return fig


def make_final_snapshot_figure(records):
    """Bar chart of final iteration hyperparameters."""
    from config import COLORS

    if not records:
        return None

    final = records[-1]
    features = list(SYNTHESIS_FEATURES)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    properties = ['Size', 'CV', 'Squareness']
    prop_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for ax, prop, color in zip(axes, properties, prop_colors):
        if prop not in final or 'lengthscales' not in final[prop]:
            continue

        ls = final[prop]['lengthscales']
        sigma_f = final[prop]['signal_variance']
        noise = final[prop]['noise_level']
        lml = final[prop]['log_marginal_likelihood']

        # For isotropic kernel, all features share same lengthscale
        if len(ls) == 1:
            vals = [ls[0]] * len(features)
        else:
            vals = ls

        x_pos = np.arange(len(features))
        ax.bar(x_pos, vals, color=color, alpha=0.75, edgecolor='white', linewidth=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Lengthscale', fontsize=11)
        ax.set_title(f'{prop} GP\n$\\sigma_f^2$={sigma_f:.3f}, '
                     f'$\\sigma_n^2$={noise:.4f}, LML={lml:.1f}',
                     fontsize=11, fontweight='bold')
        ax.axhline(0.3, color='#999', ls='--', alpha=0.5, linewidth=0.8)
        ax.axhline(10.0, color='#999', ls=':', alpha=0.5, linewidth=0.8)
        ax.set_ylim(0, max(vals) * 1.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

        n_cubic = final['n_cubic']
        n_total = final['n_total']
        ax.text(0.97, 0.95, f'n={n_cubic} cubic\n({n_total} total)',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9))

    fig.suptitle(f'Final Fitted GP Hyperparameters (Iteration {final["iteration"]})',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIG_FINAL, bbox_inches='tight', dpi=300)
    print(f"  ✓ Final snapshot figure saved: {FIG_FINAL}")
    return fig


def make_summary_table(records):
    """Create CSV table of hyperparameters across iterations for SI."""
    rows = []
    for rec in records:
        for prop in ['Size', 'CV', 'Squareness']:
            if prop in rec and 'lengthscales' in rec[prop]:
                rows.append({
                    'Iteration': rec['iteration'],
                    'N_Total': rec['n_total'],
                    'N_Cubic': rec['n_cubic'],
                    'Label': rec['label'],
                    'Property': prop,
                    'Lengthscale': rec[prop]['lengthscales'][0],
                    'Signal_Variance': rec[prop]['signal_variance'],
                    'Noise_Level': rec[prop]['noise_level'],
                    'Log_Marginal_Likelihood': rec[prop]['log_marginal_likelihood'],
                    'Kernel_String': rec[prop].get('kernel_string', ''),
                })
    df = pd.DataFrame(rows)
    df.to_csv(HP_TABLE, index=False)
    print(f"  ✓ Summary table saved: {HP_TABLE}")
    return df


if __name__ == '__main__':
    records = replay_campaign()

    if records:
        print(f"\n{'=' * 70}")
        print("GENERATING SI FIGURES")
        print(f"{'=' * 70}")
        make_evolution_figure(records)
        make_final_snapshot_figure(records)
        table = make_summary_table(records)

        print(f"\n{'=' * 70}")
        print("FINAL HYPERPARAMETER VALUES (for SI Table)")
        print(f"{'=' * 70}")
        final = records[-1]
        for prop in ['Size', 'CV', 'Squareness']:
            if prop in final and 'lengthscales' in final[prop]:
                hp = final[prop]
                print(f"\n  {prop} GP:")
                print(f"    Kernel: {hp.get('kernel_string', 'N/A')}")
                print(f"    σ_f² = {hp['signal_variance']:.4f}")
                print(f"    ℓ = {hp['lengthscales'][0]:.4f} (isotropic)")
                print(f"    σ_n² = {hp['noise_level']:.5f}")
                print(f"    Log-marginal-likelihood = {hp['log_marginal_likelihood']:.2f}")

        print(f"\n{'=' * 70}")
        print("DONE. All outputs in outputs/ — data/ was NOT modified.")
        print(f"{'=' * 70}")
