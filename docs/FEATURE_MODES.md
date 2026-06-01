# Feature Modes in Cu₃VS₄ Bayesian Optimization

This document explains the different feature representation modes available in the optimizer.

---

## Available Modes

### 1. **Raw Mode** (5 features)
Uses only the original synthesis parameters as measured in the lab.

**Features:**
- `Temp` — Temperature (°C)
- `Time` — Reaction time (min)
- `VOacac` — Vanadium precursor amount (mmol)
- `DDT` — Dodecanethiol volume (mL)
- `OAm` — Oleylamine volume (mL)

**Pros:**
- Directly interpretable by experimentalists
- No derived feature assumptions
- Lowest risk of collinearity

**Cons:**
- Ignores chemical intuition (e.g., stoichiometry, concentration effects)
- May require more data to learn relationships

---

### 2. **Chemical Mode** (6-7 features)
Uses only chemically meaningful derived features, discarding raw parameters.

**Features:**
- `Temp` — Temperature (kept from raw)
- `Cu_V_ratio` — Metal stoichiometry (CuI / VOacac)
- `S_Metal_ratio` — Sulfur excess (DDT_mmol / total_metal)
- `Ligand_Metal_ratio` — Capping ligand coverage (OAm_mmol / total_metal)
- `Metal_Conc` — Total metal concentration (mM)
- `log_Time` — Linearized kinetics (log₁₀(Time))
- `effective_dielectric` *(optional)* — Solvent mixture polarity

**Pros:**
- Incorporates chemical knowledge (HSAB theory, nucleation/growth mechanisms)
- Can reveal causal relationships (e.g., stoichiometry → phase purity)
- Potentially more generalizable to new precursors

**Cons:**
- Strong assumptions (e.g., all DDT converts to sulfur at same rate)
- Less interpretable for non-chemists
- Requires validation of chemical model

---

### 3. **Hybrid Mode** (8 features)
Combines raw parameters with key chemical features.

**Features:**
- All 5 raw factors: `Temp`, `Time`, `VOacac`, `DDT`, `OAm`
- Plus: `Cu_V_ratio`, `Metal_Conc`
- Plus: `effective_dielectric` *(if enabled)*

**Pros:**
- Balances interpretability and chemical insight
- Model can "choose" which representation to use via lengthscales
- Flexibility to capture both linear and nonlinear effects

**Cons:**
- **High collinearity**: Raw factors are algebraically related to derived features
  - `VOacac` appears in `Cu_V_ratio` and `Metal_Conc`
  - `DDT` and `OAm` appear in `Metal_Conc` and `effective_dielectric`
- Inflates VIF (Variance Inflation Factor) → unreliable lengthscale interpretation
- Can destabilize GP hyperparameter optimization
- Only ~5 samples per feature at n=47

---

### 4. **Synthesis Mode** (5 features) — **RECOMMENDED**
A reduced, mechanistically orthogonal feature set obtained by dropping the raw
`DDT` volume and `Metal_Conc` from the chemical mode. This is the default
(`DEFAULT_FEATURE_MODE = 'synthesis'` in `src/config.py`) and exactly matches
`SYNTHESIS_FEATURES` in the code.

**Features:**

| Feature | Physical Role | Controls |
|---|---|---|
| `Temp` | Arrhenius-driven burst nucleation | Dominant driver of size dispersion (CV) |
| `Cu_V_ratio` | Metal stoichiometry (CuI / VOacac) | Cu₃VS₄ phase purity, IsCubic |
| `S_Metal_ratio` | Sulfur excess (DDT_mmol / total_metal) | Crystallisation / shape (cubic vs multipod) |
| `Ligand_Metal_ratio` | OAm surface passivation density | Growth rate, colloidal stability |
| `log_Time` | Linearised reaction extent | Ostwald ripening at long times |

**Why DDT and Metal_Conc are excluded:**
Including them produces severe collinearity in this fixed-volume / fixed-CuI
system, which destabilises GP lengthscales and degrades LOO-CV performance:

- `DDT ↔ S_Metal_ratio`: `S_Metal_ratio = DDT * DDT_MMOL_PER_ML / (CuI + VOacac)`.
  Because `CuI` is fixed at 0.25 mmol and `VOacac` varies over a small range,
  these two are near-perfectly correlated (r > 0.95).
- `Cu_V_ratio ↔ Metal_Conc`: both are determined solely by `VOacac` (since
  `CuI` and total volume are fixed), giving r ≈ −0.99.

The five chemical ratios above already encode the same physical information
in a more interpretable, lower-collinearity form, and empirically give
the best LOO-CV R² of the four modes in this codebase.

**Pros:**
- Mechanistically motivated — every feature has a clear chemical meaning
- Lowest collinearity of any chemical mode (all VIFs < 5)
- Best samples-per-feature ratio (~9.4 at n = 47 successful experiments)
- Stable GP hyperparameter fitting with interpretable lengthscales

**Cons:**
- Requires the raw → chemical transformation (inverse is provided in
  `features.chemical_to_raw_features`)
- Feature set is fixed (not data-adaptive), but this is intentional for
  reproducibility across the campaign
- Drops the small amount of independent information carried by the raw
  `DDT` volume in addition to its contribution to `S_Metal_ratio`

---

### 5. **Transfer Mode** (5–9 features)
Extends synthesis mode with precursor descriptor features for transfer
learning across different Cu and/or Group 5 metal precursors. Enabled by
setting `TRANSFER_MODE['enabled'] = True` in `src/config.py`.

**Base features** (always included):
Same 5 as synthesis mode: `Temp`, `Cu_V_ratio`, `S_Metal_ratio`,
`Ligand_Metal_ratio`, `log_Time`.

**Additional precursor descriptor features** (toggled via `TRANSFER_MODE`):

| Toggle | Added Features | What They Capture |
|---|---|---|
| `vary_cu_precursor` | `Cu_precursor_hardness`, `Cu_hsab_mismatch` | Pearson hardness of the Cu precursor, cation-anion mismatch |
| `vary_metal_precursor` | `Metal_ionic_potential`, `Metal_hsab_mismatch` | Ionic potential (Z/r) of M⁵⁺, cation-anion mismatch |

**Kernel**: Uses ARD (Automatic Relevance Determination) Matérn 5/2 with
per-feature lengthscales, so the GP can learn different correlation
structures for synthesis parameters vs. precursor descriptors.

**Recommendation behavior**: During candidate generation, precursor
descriptor features are fixed to the target precursor values (point bounds),
and only the 5 synthesis dimensions are sampled via LHS.

**Pros:**
- Pools data across precursors to accelerate optimization of new systems
- GP learns how synthesis trends transfer from one precursor to another
- ARD lengthscales reveal which features are most important across precursors
- Target-precursor-specific recommendations — no wasted candidates

**Cons:**
- Requires experiments with multiple precursors in the training set
- ARD kernel has more hyperparameters to fit — needs adequate data
- Mixed-scale descriptors (Pearson η in eV vs Z/r in e/Å) — handled by
  StandardScaler but still an approximation

**Example usage:**
```python
# In config.py, set:
TRANSFER_MODE = {
    'enabled': True,
    'vary_cu_precursor': True,     # Pool CuI + CuCl data
    'vary_metal_precursor': False,
    'target_cu_precursor': 'CuCl',
    'target_metal_precursor': 'NbCl5',
    'use_ard_kernel': True,
}
```

---

## Recommendations

### For Your Current Dataset (~47 successful experiments)

**Best choice: `synthesis`**

**Rationale:**
1. Five features with clear physical meaning, zero redundancy
2. Best samples-per-feature ratio of any chemical mode
3. Eliminates collinearity issues that plagued hybrid mode
4. Interpretable GP lengthscales map directly to synthesis mechanisms

**How to use:**
```python
optimizer = SelfValidatingOptimizer(
    data_dir=DATA_DIR,
    initialize_from_csv=True,
    feature_mode='synthesis'
)
```

### For Conservative Baseline

**Use: `raw` mode**

If you want the most defensible model with fewest assumptions, stick to raw factors.
This is safest for publication if reviewers question your chemical feature engineering.

---

## How to Compare Modes

Use the built-in comparison function:

```python
comparison_df = optimizer.compare_feature_modes(
    modes=['raw', 'chemical', 'hybrid', 'synthesis']
)

# View results
print(comparison_df.pivot(index='Property', columns='Mode', values='R2'))
```

Look for:
- **Highest R²** for each property (Size, CV, Squareness)
- **Good calibration**: `Cal_68 ≈ 0.68`, `Cal_95 ≈ 0.95`
- **Lowest VIF** in collinearity diagnostics

---

## Summary Table

| Mode | # Features | Collinearity | Samples/Feature (n=47) | Best For |
|------|------------|--------------|------------------------|----------|
| `raw` | 5 | ✓ None | 9.4 | Conservative baseline, interpretability |
| `chemical` | 6-7 | ✓ Low | 6.7–7.8 | Chemical insight, generalization |
| `hybrid` | 8 | ✗ Severe (VIF>10k) | 5.9 | ❌ Not recommended (collinearity) |
| `synthesis` | 5 | ✓ Low (all VIF<5) | 9.4 | **Recommended** — mechanistic + low collinearity |
| `transfer` | 5–9 | ✓ Low (ARD handles scale) | varies | Multi-precursor campaigns, transfer learning |

---

## Related Files

- **Feature definitions**: `src/config.py` (see `SYNTHESIS_FEATURES`)
- **Feature transforms**: `src/features.py`
- **Constants**: `src/chemical_constants.py` (chemical descriptors)
- **Notebook**: `Notebooks/Cu3VS4_BO_Execute.ipynb`
