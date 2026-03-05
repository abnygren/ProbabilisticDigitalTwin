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

### 4. **Synthesis Mode** (7 features) — **RECOMMENDED**
Uses mechanistically motivated features that combine chemical ratios with the raw DDT
volume needed for GP predictive performance.

**Features:**

| Feature | Physical Role | Controls |
|---|---|---|
| `Temp` | Arrhenius-driven burst nucleation | Dominant driver of GSD |
| `DDT` | Raw sulfur-source volume (mL) | Near-linear relationship with Size |
| `Cu_V_ratio` | Metal stoichiometry (CuI / VOacac) | Cu₃VS₄ phase purity, IsCubic |
| `S_Metal_ratio` | Sulfur excess (DDT_mmol / total_metal) | Shape control (cubic vs multipod) |
| `Ligand_Metal_ratio` | OAm surface passivation density | Growth rate, colloidal stability |
| `Metal_Conc` | Total ion concentration (mM) | Supersaturation → nucleation → size |
| `log_Time` | Linearised reaction extent | Ostwald ripening at long times |

**Why DDT is kept as a raw feature:**
DDT volume has a near-linear relationship with particle size. The ratio form
(`S_Metal_ratio = DDT_mmol / total_metal`) divides by `total_metal` which varies with
VOacac, introducing a nonlinearity the GP struggles to resolve with limited data (n~47).
The GP's ARD kernel handles the moderate collinearity between DDT and S_Metal_ratio
gracefully via lengthscales. Removing DDT to achieve theoretical orthogonality actually
degrades predictive performance.

**Pros:**
- Mechanistically motivated — every feature has a clear chemical meaning
- DDT provides the GP with a direct, learnable signal for Size prediction
- 6.7 samples per feature at n=47 (better than hybrid's 5.9)
- Stable GP hyperparameter fitting with interpretable lengthscales

**Cons:**
- Moderate collinearity between DDT and S_Metal_ratio (VIF ~7, acceptable)
- Requires the raw→chemical transformation (inverse is provided)
- Feature set is fixed (not data-adaptive), but this is intentional for reproducibility

---

## Recommendations

### For Your Current Dataset (~47 successful experiments)

**Best choice: `synthesis`**

**Rationale:**
1. Six features with clear physical meaning, zero redundancy
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
- **Highest R²** for each property (Size, GSD, Squareness)
- **Good calibration**: `Cal_68 ≈ 0.68`, `Cal_95 ≈ 0.95`
- **Lowest VIF** in collinearity diagnostics

---

## Summary Table

| Mode | # Features | Collinearity | Samples/Feature (n=47) | Best For |
|------|------------|--------------|------------------------|----------|
| `raw` | 5 | ✓ None | 9.4 | Conservative baseline, interpretability |
| `chemical` | 6-7 | ✓ Low | 6.7–7.8 | Chemical insight, generalization |
| `hybrid` | 8 | ✗ Severe (VIF>10k) | 5.9 | ❌ Not recommended (collinearity) |
| `synthesis` | 7 | ✓ Low–moderate | 6.7 | **Recommended** — mechanistic + empirically validated |

---

## Related Files

- **Feature definitions**: `src/config.py` (see `SYNTHESIS_FEATURES`)
- **Feature transforms**: `src/features.py`
- **Constants**: `src/chemical_constants.py` (chemical descriptors)
- **Notebook**: `Notebooks/Cu3VS4_BO_Execute.ipynb`
