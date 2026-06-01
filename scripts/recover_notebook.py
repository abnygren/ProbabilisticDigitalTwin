"""Rewrite the source of specific cells in Cu3VS4_BO_Execute.ipynb.
Preserves all outputs and other metadata, only updating cell['source'].
"""
import json
import sys
from pathlib import Path

NB_PATH = Path("Notebooks/Cu3VS4_BO_Execute.ipynb")


def src(text: str) -> list:
    """Convert a multi-line string into the list-of-strings format ipynb expects."""
    lines = text.splitlines(keepends=True)
    return lines


# Map cell_idx -> new source text
NEW_SOURCES: dict[int, str] = {
    0: """# Self-Validating Bayesian Optimization for Cu\u2083VS\u2084 Nanoparticle Synthesis

This notebook is the front-end for the BO workflow: it loads experiments, fits the GP surrogate models, generates new recommendations, and visualizes everything once results come back from the lab.

The acquisition function only optimizes **Size** (single regression GP).
CV and Squareness GPs are fit alongside it for inspection but do not enter the acquisition.
Squareness instead enters as a bin-based feasibility constraint (`multipod` / `highly_cubic` / `poorly_cubic`), with the threshold pinned by Otsu's method on the initial dataset.

All model code lives in `src/` (see `config.py`, `features.py`, `optimizer.py`, `selfvalidating.py`, `experiment_store.py`, `diagnostics.py`, `visualization.py`, `chemical_constants.py`); this notebook only orchestrates and plots.

To switch precursor systems (e.g. Cu\u2083VS\u2084 \u2192 Cu\u2083NbS\u2084), edit `CURRENT_PRECURSORS` / `TRANSFER_MODE` in `src/config.py` and point `DATA_DIR` at the matching campaign folder.
""",

    1: """import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("..").resolve() / "src"))

from selfvalidating import SelfValidatingOptimizer
from config import COLORS, PUBLICATION_STYLE

from visualization import (
    plot_recommendation_history,
    plot_parity,
    plot_calibration,
    plot_error_learning_progress,
    plot_error_correction_impact,
    plot_target_achievement,
    plot_feature_importance,
    plot_classifier_calibration,
    plot_collinearity_heatmap,
    plot_dataset_quality_dashboard,
    plot_loo_residuals,
    plot_property_correlations,
    plot_acquisition_slice,
    plot_recommendation_regret,
    plot_response_surface,
    plot_classification_surface,
    plot_classification_facets,
    plot_loo_parity,
    plot_feasibility_landscape,
    plot_squareness_binning,
    plot_acquisition_decomposition,
    plot_bias_correction_arrows,
    plot_bin_probability_facets_all,
    plot_bin_probability_facets_cuv,
    plot_bo_trajectory,
    plot_bo_trajectory_by_size,
    plot_optimization_progress,
)

from diagnostics import (
    compare_feature_modes,
    detect_extrapolation,
    diagnose_collinearity,
    print_model_assessment,
    display_recommendations_table,
    get_optimization_convergence_summary,
    print_optimization_convergence_summary,
    print_optimization_statistics,
    compute_optimization_statistics,
)

DATA_DIR = Path("..") / "data"
OUTPUT_DIR = Path("..") / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(PUBLICATION_STYLE)

# Silence GP fitting noise that we know is benign here; real numerical issues
# still raise.
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')

np.random.seed(42)

print("\u2713 All imports successful!")
print(f"\u2713 Data directory: {DATA_DIR.resolve()}")
print(f"\u2713 Output directory: {OUTPUT_DIR.resolve()}")
""",

    2: """## 1. Initialize the optimizer

This step:
- Loads experiments from `experiments.json` (or imports the CSV on first run)
- Loads or computes the frozen Otsu squareness threshold from the imported CSV
- Fits GP regressors (Size objective; CV and Squareness for diagnostics)
- Fits feasibility classifiers (HasProduct, PhasePure, IsCubic)
- Loads any existing recommendation history
- Initializes the error learner (active once \u226510 recommendations are completed)
""",

    3: """# Recommended: 'synthesis' mode (5 mechanistically orthogonal features).
# See docs/FEATURE_MODES.md for trade-offs across modes.
optimizer = SelfValidatingOptimizer(
    data_dir=DATA_DIR,
    initialize_from_csv=True,   # only triggers if experiments.json doesn't exist
    feature_mode='synthesis',
)

# Other modes (uncomment one of these to try):
#   'raw'       \u2014 Temp, Time, VOacac, DDT, OAm
#   'chemical'  \u2014 derived chemical descriptors only
#   'hybrid'    \u2014 raw + chemical (collinearity warning)

optimizer.print_status()
""",

    4: """## 1b. Model diagnostics (recommended)

Runs the standard checks before generating recommendations:

- **LOO-CV** \u2014 R\u00b2, RMSE, and uncertainty calibration for Size / CV / Squareness
- **VIF** \u2014 collinearity in the feature set
- **Classifier calibration** \u2014 stratified CV when class counts allow, in-sample fallback otherwise
- **Sample-size adequacy** \u2014 warning if there are too few cubic regression samples per feature
""",

    5: """diagnostics = optimizer.full_diagnostics()

# Optional feature-mode comparison (re-fits each mode \u2014 ~2-3 min):
comparison_df = optimizer.compare_feature_modes(
    modes=['raw', 'chemical', 'hybrid', 'synthesis']
)
""",

    6: """# Reliability diagrams for the feasibility classifiers.
fig_cal = plot_classifier_calibration(optimizer)
if fig_cal:
    plt.savefig(OUTPUT_DIR / "classifier_calibration.png", dpi=150, bbox_inches='tight')
    plt.show()

# Pairwise feature correlation heatmap.
fig_corr = plot_collinearity_heatmap(optimizer)
if fig_corr:
    plt.savefig(OUTPUT_DIR / "feature_correlations.png", dpi=150, bbox_inches='tight')
    plt.show()
""",

    7: """## 1c. Dataset quality dashboard

Four-panel summary of the current training set:

1. Design-space coverage in parameter space
2. Outcome distributions (Size, GSD, Squareness)
3. Dataset composition (success rate, samples-per-feature)
4. LOO-CV R\u00b2 across feature modes
""",

    8: """fig_dashboard = plot_dataset_quality_dashboard(
    optimizer,
    figsize=(14, 10),
    save_path=OUTPUT_DIR / "dataset_quality_dashboard.png",
)

if fig_dashboard:
    plt.show()
    print(f"\\n\u2713 Dashboard saved to: {OUTPUT_DIR / 'dataset_quality_dashboard.png'}")
""",

    9: """## 2. Generate recommendations

Request synthesis recommendations for a target particle size and a squareness bin.

Squareness bins:
- `'highly_cubic'` \u2014 cubic with Squareness \u2265 Otsu threshold (\u22480.810)
- `'poorly_cubic'` \u2014 cubic with Squareness < Otsu threshold
- `'multipod'`     \u2014 non-cubic polymorph
""",

    10: """# TARGET_SIZE = 30.0
# SIZE_TOLERANCE = 2.0
# SQUARENESS_BIN = 'highly_cubic'  # 'highly_cubic' | 'poorly_cubic' | 'multipod'

# recommendations = optimizer.recommend(
#     target_size=TARGET_SIZE,
#     size_tol=SIZE_TOLERANCE,
#     squareness_bin=SQUARENESS_BIN,
#     seed=42,
# )
# display_recommendations_table(recommendations)
""",

    11: """## 3. View pending recommendations

Recommendations waiting on lab results.
""",

    12: """pending = optimizer.get_pending_recommendations()
display(pending)
""",

    13: """## 4. Complete recommendations

Log measured results back into the optimizer once the synthesis is done.
""",

    14: """# Uncomment and fill in with measured results:

# errors = optimizer.complete_recommendation(
#     rec_id='REC_024',         # recommendation ID
#     Size=15.758,              # measured size (nm)
#     CV=0.189,                 # measured CV
#     Squareness=0.823,         # measured squareness
#     HasProduct=1,             # 1 if product formed
#     PhasePure=0,              # 1 if phase pure
#     Polymorph='cubic',        # 'cubic', 'multipod', etc.
# )

# print_model_assessment(optimizer)
""",

    15: """## 5. Quick results (after experiments)

Recommendation timeline, BO trajectory toward the target, parity vs stored predictions, and achieved sizes vs targets.
""",

    16: """fig = plot_recommendation_history(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'recommendation_history.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    17: """fig = plot_bo_trajectory(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'bo_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    18: """fig = plot_bo_trajectory_by_size(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / "bo_trajectory_by_size.png", dpi=300, bbox_inches="tight")
    plt.show()
""",

    19: """fig = plot_target_achievement(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'target_achievement.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    20: """fig = plot_parity(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'parity_plots_recommendations.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    21: """## 6. Model quality (GP & classifiers)

How well the surrogate models fit and calibrate. If you already ran section 1b, the classifier reliability diagrams and feature correlation heatmap were generated there.

The cells below add: residual checks, the three-panel LOO-CV parity (Size / CV / Squareness), calibration coverage from completed recommendations, and gradient-based feature importance.
""",

    22: """fig = plot_calibration(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    23: """fig = plot_loo_residuals(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'loo_residuals.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    24: """fig = plot_loo_parity(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'loo_parity.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    25: """fig = plot_feature_importance(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    26: """## 7. Error correction & self-validation

Diagnostics for the error learner: systematic bias, correction impact, and the trajectory of learning from prediction errors.
""",

    27: """print_model_assessment(optimizer)
""",

    28: """fig = plot_error_learning_progress(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'learning_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    29: """fig = plot_error_correction_impact(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'error_correction_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    30: """# Requires the ErrorLearner to be fitted (\u226510 completed recommendations).
fig = plot_bias_correction_arrows(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'bias_correction_arrows.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    31: """## 8. Design space & acquisition geometry

Where the experiments live in parameter space, and how feasibility / morphology / acquisition vary across the GP-defined box.

The 2-D acquisition slice cell is optional \u2014 uncomment and pick two features to vary while everything else sits at successful-run medians.
""",

    32: """fig = plot_property_correlations(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'property_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    33: """fig = plot_feasibility_landscape(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'feasibility_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    34: """fig = plot_squareness_binning(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'squareness_binning.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    35: """# Acquisition decomposition: P(size band) \u00d7 P(feasible) \u00d7 P(bin).
# Edit kwargs to match the campaign you want to inspect.
fig = plot_acquisition_decomposition(
    optimizer,
    target_size=20.0,
    size_tol=2.5,
    squareness_bin='highly_cubic',
)
if fig:
    plt.savefig(OUTPUT_DIR / 'acquisition_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    36: """# Optional 2-D acquisition slice. Uncomment and pick the axes to vary.
# fig = plot_acquisition_slice(
#     optimizer,
#     target_size=20.0,
#     size_tol=2.5,
#     squareness_bin='highly_cubic',
#     param1='Temp',
#     param2=None,
# )
# if fig:
#     plt.savefig(OUTPUT_DIR / 'acquisition_slice.png', dpi=300, bbox_inches='tight')
#     plt.show()
""",

    37: """### Response & classification surfaces

3-D cubes use the top-3 gradient-importance features per model; the remaining features sit at the cubic-training medians.
""",

    38: """for resp in ['Size', 'CV', 'Squareness', 'Feasibility']:
    fig = plot_response_surface(optimizer, response=resp, n_grid=25)
    if fig:
        plt.savefig(
            OUTPUT_DIR / f'response_surface_{resp.lower()}.png',
            dpi=300,
            bbox_inches='tight',
        )
        plt.show()
""",

    39: """fig = plot_classification_surface(optimizer, n_grid=25)
if fig:
    plt.savefig(OUTPUT_DIR / 'classification_surface_squareness.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    40: """# 3\u00d73 facet grid per remaining feature (top-2 axes \u00d7 slice dimension).
figs = plot_bin_probability_facets_all(optimizer, n_grid=50)
for i, fig in enumerate(figs or []):
    if fig:
        plt.savefig(
            OUTPUT_DIR / f'bin_probability_facets_all_{i}.png',
            dpi=300,
            bbox_inches='tight',
        )
        plt.show()
""",

    41: """# 3\u00d73 facets with Cu/V ratio fixed as the column slice.
fig = plot_bin_probability_facets_cuv(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'bin_probability_facets_cuv.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    42: """## 8a. Optimization progress (convergence evidence)

Visual evidence that BO improved particle quality relative to the initial unguided dataset: running-best convergence for GSD and Squareness, plus distributional comparison with Mann-Whitney U tests.
""",

    43: """fig = plot_optimization_progress(optimizer)
if fig:
    fig.savefig(OUTPUT_DIR / 'optimization_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
""",

    44: """## 8b. Statistical evidence for optimization

Publication-style statistical tests showing BO optimized size: Mann-Whitney U vs baseline, binomial hit-rate test, GP calibration, and error-trend analysis.
""",

    45: """stats_results = print_optimization_statistics(optimizer)

# Optional: export the summary table.
# stats_results["summary_table"].to_csv(
#     OUTPUT_DIR / "optimization_statistics.csv",
#     index=False,
# )
""",

    46: """## 9. Convergence & stopping

Simple regret curves and the optimization convergence summary (prediction quality vs targets, by band, last-N window).
""",

    47: """fig = plot_recommendation_regret(optimizer)
if fig:
    plt.savefig(OUTPUT_DIR / 'recommendation_regret.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print('Need at least 2 completed recommendations to plot regret.')
""",

    48: """print_optimization_convergence_summary(optimizer, last_n=5)
""",
}


def main():
    with NB_PATH.open() as f:
        nb = json.load(f)

    for idx, new_text in NEW_SOURCES.items():
        if idx >= len(nb["cells"]):
            print(f"WARN: cell idx {idx} out of range ({len(nb['cells'])})")
            continue
        nb["cells"][idx]["source"] = src(new_text)

    with NB_PATH.open("w") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated {len(NEW_SOURCES)} cells in {NB_PATH}")


if __name__ == "__main__":
    main()
