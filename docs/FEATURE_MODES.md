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

### 3. **Hybrid Mode** (8 features) — **Default**
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
- **Only 4.8 samples per feature** with 38 successful experiments (recommend ≥10)

**Current Issues (from diagnostics):**
```
High VIF features: VOacac, DDT, OAm, Metal_Conc, effective_dielectric
All have VIF = 10,000+ (perfect collinearity)
```

---

### 4. **Smart Hybrid Mode** (4-6 features) — **NEW**
Automatically removes raw features that are highly correlated with chemical features.

**Algorithm:**
1. Start with all raw factors
2. Add chemical features: `Cu_V_ratio`, `Metal_Conc`, `effective_dielectric`
3. For each raw feature, compute correlation with chemical features
4. If `|correlation| > threshold` (default: 0.8), **drop** the raw feature
5. Keep all chemical features (they replace the dropped raw features)

**Example Output:**
```
✓ Keep Temp        (max |r| = 0.12 with Metal_Conc)
✓ Keep Time        (max |r| = 0.05 with Cu_V_ratio)
✗ Drop VOacac      (|r| = 0.98 with Cu_V_ratio)  → replaced
✗ Drop DDT         (|r| = 0.95 with Metal_Conc)  → replaced
✗ Drop OAm         (|r| = 0.92 with Metal_Conc)  → replaced

Smart Hybrid: 5 features selected
  Raw features kept: ['Temp', 'Time']
  Chemical features: ['Cu_V_ratio', 'Metal_Conc', 'effective_dielectric']
```

**Resulting Features (typical):**
- `Temp`, `Time` — Kept (low correlation)
- `Cu_V_ratio` — Replaces `VOacac`
- `Metal_Conc` — Replaces `DDT` and `OAm`
- `effective_dielectric` — Additional chemical insight

**Pros:**
- **Eliminates collinearity automatically**
- Better samples-per-feature ratio (38/5 = 7.6 vs 38/8 = 4.8)
- More stable GP hyperparameter fitting
- Interpretable lengthscales
- Retains chemical insight where it's orthogonal to raw factors

**Cons:**
- Selection is data-dependent (may vary if dataset changes)
- Threshold choice (0.8) is somewhat arbitrary
- Still makes chemical assumptions for derived features

---

## Recommendations

### For Your Current Dataset (38 successful experiments)

**Best choice: `smart_hybrid`**

**Rationale:**
1. Your LOO-CV shows weak predictive performance across all modes (many negative R²)
2. Hybrid mode has severe collinearity (VIF = 10,000+)
3. Only 4.8 samples/feature in hybrid mode is underpowered
4. Smart hybrid reduces to ~5 features → 7.6 samples/feature
5. Removes redundant raw features while keeping chemical insight

**How to use:**
```python
optimizer = SelfValidatingOptimizer(
    data_dir=DATA_DIR,
    initialize_from_csv=True,
    feature_mode='smart_hybrid'  # Auto-removes collinear features
)
```

### If You Collect More Data (>100 successful experiments)

**Consider: `chemical` mode**

With more data, you can afford to test if chemical features truly generalize better than raw factors. The chemical mode makes the strongest assumptions but may reward you with better extrapolation to new precursors or unseen regions of the design space.

### For Conservative Baseline

**Use: `raw` mode**

If you want the most defensible model with fewest assumptions, stick to raw factors. This is safest for publication if reviewers question your chemical feature engineering.

---

## How to Compare Modes

Use the built-in comparison function:

```python
comparison_df = optimizer.compare_feature_modes(
    modes=['raw', 'chemical', 'hybrid', 'smart_hybrid']
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

| Mode | # Features | Collinearity | Samples/Feature | Best For |
|------|------------|--------------|-----------------|----------|
| `raw` | 5 | ✓ None | 7.6 | Conservative baseline, interpretability |
| `chemical` | 6-7 | ✓ Low | 5.4-6.3 | Chemical insight, generalization |
| `hybrid` | 8 | ✗ Severe (VIF>10k) | 4.8 | ❌ Not recommended (collinearity) |
| `smart_hybrid` | 4-6 | ✓ Low (auto-removed) | 6.3-9.5 | **Recommended** — balances insight & stability |

---

## Related Files

- **Implementation**: `src/cuvs_optimizer.py` (see `select_smart_hybrid_features`)
- **Constants**: `src/chemical_constants.py` (chemical descriptors)
- **Notebook**: `Notebooks/Cu3VS4_BO_Execute.ipynb` (cell 3 for initialization)
