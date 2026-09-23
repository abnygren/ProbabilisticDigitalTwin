# Cu3VS4 Bayesian Optimization

Code and processed data for closed-loop Bayesian optimization of Cu3VS4 nanocrystal synthesis, and for transfer to CuBr, CuCl, and Cu3TaS4.

Size is the optimization target. Product formation, phase purity, and morphology are constraints.

The CuI campaign maps the design space (68 initial reactions) and then selects phase-pure cubic Cu3VS4 from 12 to 30 nm. The same surrogate is then reused along two chemical axes:

- Copper halide (CuI, CuBr, CuCl), encoded by HSAB hardness
- Group 5 metal center (V to Ta), encoded by oxophilicity

## Campaigns

| Campaign | Notebook | Data |
|---|---|---|
| CuI | `Notebooks/Cu3VS4_BO_Execute_CuI.ipynb` | `data_CuI/` |
| CuBr | `Notebooks/Cu3VS4_BO_Execute_CuBr.ipynb` | `data_CuBr/` |
| CuCl | `Notebooks/Cu3VS4_BO_Execute_CuCl.ipynb` | `data_CuCl/` |
| Ta | `Notebooks/Cu3VS4_BO_Execute_Ta.ipynb` | `data_Ta/` |

Each notebook sets its own precursors and transfer flags in the first code cell. You do not need to edit `src/config.py` to switch campaigns.

Processed experiments and recommendations are stored as JSON in each `data_*/` folder, with CSV seed/transfer files used only if `experiments.json` is empty. Coefficient of variation (CV) is stored on every experiment for diagnostics; it does not enter the acquisition function.

## Setup

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

NumPy, SciPy, scikit-learn, pandas, and matplotlib.

## Reproducing a campaign

Open the notebook for that campaign and run cells from the top.

1. Initialize the optimizer (loads `experiments.json`, fits GPs and classifiers)
2. Run diagnostics
3. Generate recommendations, or skip those cells and go to the plots
4. After new syntheses, log results with `complete_recommendation`

Leave-one-precursor-out (withhold CuCl, train on CuI+CuBr) is in `src/diagnostics.py` as `leave_one_precursor_out`.

## How recommendations are chosen

```
alpha(x) = P(size in target) * P(has product) * P(phase pure) * P(target morphology)
```

After ten completed recommendations, a residual GP corrects systematic bias and a calibration factor rescales uncertainty.

## Layout

```
src/         optimizer, features, diagnostics, plotting, chemical constants
Notebooks/   one execute notebook per campaign
data_CuI/    CuI experiments and recommendations
data_CuBr/   CuBr transfer campaign
data_CuCl/   CuCl transfer campaign
data_Ta/     TaCl5 transfer campaign
scripts/     figure-generation helpers
tests/
```

## Tests

```bash
python tests/test_import.py
python tests/test_selfvalidating.py
python tests/verify_notebooks.py
```
