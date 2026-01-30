# Refactoring Plan for Publication

## Goal: Transform notebooks into publication-ready Python package

## Phase 1: Extract Core Functions (Week 1)

### 1.1 Create Package Structure
```bash
mkdir -p cu3vs4_bo
touch cu3vs4_bo/__init__.py
```

### 1.2 Extract Feature Engineering (`cu3vs4_bo/features.py`)
**From:** `Cu3VS4_BO_Final.ipynb` cells 6-7

**Extract:**
- `raw_to_chemical_features()` → `compute_chemical_features()`
- `chemical_to_raw_features()` → `invert_chemical_features()`
- `add_chemical_features()` → `add_features_to_dataframe()`
- `compute_feature_bounds()` → `get_feature_bounds()`

**Changes:**
- Remove hardcoded constants (use config)
- Add type hints
- Add docstrings
- Add input validation

### 1.3 Extract GP Models (`cu3vs4_bo/models.py`)
**From:** `Cu3VS4_BO_Final.ipynb` cell 11

**Extract:**
- `make_gp_regressor()` → `create_gp_regressor()`
- `make_gp_classifier()` → `create_gp_classifier()`

**Changes:**
- Make hyperparameters configurable
- Add docstrings explaining kernel choices
- Add validation

### 1.4 Extract Acquisition Functions (`cu3vs4_bo/acquisition.py`)
**From:** `Cu3VS4_BO_Final.ipynb` cell 12

**Extract:**
- `expected_improvement()` → `compute_expected_improvement()`
- `prob_in_interval()` → `compute_probability_in_interval()`

**Changes:**
- Add docstrings with equations
- Add input validation
- Handle edge cases

## Phase 2: Refactor Main Class (Week 2)

### 2.1 Fix Cu3VS4Optimizer (`cu3vs4_bo/optimizer.py`)

**Critical Fixes:**

1. **Remove Monkey Patching**
   ```python
   # BEFORE (BAD):
   model.acquisition = types.MethodType(patched_acquisition, model)
   
   # AFTER (GOOD):
   class Cu3VS4Optimizer:
       def __init__(self, ..., objective_weights=None):
           self.objective_weights = objective_weights or self._calculate_weights()
       
       def acquisition(self, X, target_size, size_tol):
           # Use self.objective_weights directly
   ```

2. **Fix Feature Mode Handling**
   ```python
   # Create FeatureConverter class
   class FeatureConverter:
       def __init__(self, mode='hybrid'):
           self.mode = mode
           self.feature_list = self._get_feature_list()
       
       def convert_raw_to_features(self, raw_params):
           # Single method, no duplication
   ```

3. **Add Proper Configuration**
   ```python
   @dataclass
   class OptimizerConfig:
       feature_mode: str = 'hybrid'
       gp_restarts_regression: int = 15
       gp_restarts_classification: int = 5
       n_candidates: int = 20000
       min_distance: float = 0.3
       # ... all configurable parameters
   ```

### 2.2 Extract Utility Functions (`cu3vs4_bo/utils.py`)

**Extract:**
- `latin_hypercube_sample()` → `sample_lhs()`
- `loo_cv()` → `cross_validate_loo()`
- `calculate_objective_weights()` → `compute_objective_weights()`

## Phase 3: Clean Notebooks (Week 3)

### 3.1 Create Clean Example Notebooks

**Notebook 1: `01_data_exploration.ipynb`**
- Load data
- Basic statistics
- Visualizations
- Data quality checks

**Notebook 2: `02_model_training.ipynb`**
- Train models
- Compare feature modes
- Cross-validation
- Model diagnostics

**Notebook 3: `03_optimization.ipynb`**
- Generate recommendations
- Size sweeps
- Analyze recommendations

**Notebook 4: `04_results_analysis.ipynb`**
- Parity plots
- Feature importance
- Partial dependence plots
- Export results

### 3.2 Remove from Publication Notebooks:
- [ ] All debugging code
- [ ] Commented-out code blocks
- [ ] Hardcoded paths
- [ ] Personal notes
- [ ] Failed experiments
- [ ] Temporary visualizations

## Phase 4: Documentation (Week 4)

### 4.1 Add Docstrings

**Template:**
```python
def compute_chemical_features(
    temp: np.ndarray,
    time: np.ndarray,
    voacac: np.ndarray,
    ddt: np.ndarray,
    oam: np.ndarray,
    cui_mmol: float = 0.25,
    total_vol_ml: float = 12.0
) -> Dict[str, np.ndarray]:
    """
    Transform raw synthesis parameters into chemically meaningful features.
    
    Computes stoichiometric ratios, concentrations, and derived descriptors
    that encode chemical reactivity principles (HSAB theory, nucleation kinetics).
    
    Parameters
    ----------
    temp : array-like
        Reaction temperature (°C), shape (n_samples,)
    time : array-like
        Reaction time (minutes), shape (n_samples,)
    voacac : array-like
        Vanadium precursor amount (mmol), shape (n_samples,)
    ddt : array-like
        Dodecanethiol volume (mL), shape (n_samples,)
    oam : array-like
        Oleylamine volume (mL), shape (n_samples,)
    cui_mmol : float, default=0.25
        Fixed CuI amount (mmol)
    total_vol_ml : float, default=12.0
        Total reaction volume (mL)
    
    Returns
    -------
    features : dict
        Dictionary with feature names as keys and arrays as values:
        - 'Cu_V_ratio': Stoichiometry (target = 3.0 for Cu₃VS₄)
        - 'S_Metal_ratio': Sulfur excess
        - 'Ligand_Metal_ratio': Surface capping density
        - 'Metal_Conc': Total metal concentration (mM)
        - 'log_Time': log₁₀(Time) for linearized kinetics
    
    Examples
    --------
    >>> features = compute_chemical_features(
    ...     temp=[280, 290],
    ...     time=[30, 45],
    ...     voacac=[0.2, 0.25],
    ...     ddt=[3.0, 3.5],
    ...     oam=[4.0, 5.0]
    ... )
    >>> features['Cu_V_ratio']
    array([1.25, 1.0])
    
    Notes
    -----
    Chemical feature derivations based on:
    - Stoichiometry: Cu₃VS₄ requires Cu:V = 3:1
    - Nucleation: S excess drives nucleation (S:Metal ratio)
    - Surface capping: Ligand:Metal affects growth kinetics
    
    References
    ----------
    Pearson, R.G. (1988) Inorg. Chem. 27, 734-740 (HSAB theory)
    """
```

### 4.2 Create Methodology Documentation

**File: `docs/methodology.md`**

Sections:
1. Chemical Feature Engineering
   - Derivation of each feature
   - Chemical rationale
   - References

2. Gaussian Process Models
   - Kernel choice (Matérn 5/2)
   - Hyperparameter selection
   - ARD explanation

3. Acquisition Function
   - Expected Improvement derivation
   - Multi-objective weighting
   - Constraint handling

4. Validation Strategy
   - Leave-One-Out CV
   - Calibration checks
   - Uncertainty quantification

## Phase 5: Testing (Week 5)

### 5.1 Unit Tests

**File: `tests/test_features.py`**
```python
def test_compute_chemical_features():
    """Test chemical feature computation."""
    features = compute_chemical_features(
        temp=[280], time=[30], voacac=[0.2], ddt=[3.0], oam=[4.0]
    )
    assert 'Cu_V_ratio' in features
    assert features['Cu_V_ratio'][0] == pytest.approx(1.25, rel=1e-3)

def test_feature_inversion():
    """Test that features can be inverted back to raw."""
    raw = {'Temp': 280, 'Time': 30, 'VOacac': 0.2, 'DDT': 3.0, 'OAm': 4.0}
    features = compute_chemical_features(**raw)
    raw_reconstructed = invert_chemical_features(**features)
    assert raw_reconstructed['Temp'] == pytest.approx(raw['Temp'])
```

**File: `tests/test_optimizer.py`**
```python
def test_optimizer_initialization():
    """Test optimizer can be initialized."""
    optimizer = Cu3VS4Optimizer(df_sample, feature_mode='hybrid')
    assert optimizer.feature_mode == 'hybrid'
    assert len(optimizer.features) > 0

def test_recommendations():
    """Test recommendation generation."""
    optimizer = Cu3VS4Optimizer(df_sample)
    recs = optimizer.recommend(target_size=20.0, n_return=2)
    assert len(recs) == 2
    assert 'Temp' in recs.columns
```

## Phase 6: Final Polish (Week 6)

### 6.1 Code Review Checklist

- [ ] All functions have docstrings
- [ ] All classes have docstrings
- [ ] Type hints added
- [ ] No hardcoded paths
- [ ] No monkey patching
- [ ] Consistent error handling
- [ ] Input validation everywhere
- [ ] Tests pass
- [ ] Notebooks run end-to-end
- [ ] README complete

### 6.2 Pre-Submission Checklist

- [ ] Code archived on Zenodo (get DOI)
- [ ] Data archived separately
- [ ] LICENSE file added
- [ ] CITATION.cff created
- [ ] All dependencies in requirements.txt
- [ ] Python version specified
- [ ] Random seeds documented
- [ ] Methodology documented
- [ ] Examples work
- [ ] Tested on clean environment

## Migration Strategy

### Step-by-Step Migration:

1. **Don't break existing notebooks** - Keep them working during refactor
2. **Create new package alongside** - Don't delete old code yet
3. **Test new package** - Ensure it produces same results
4. **Update notebooks gradually** - One at a time
5. **Remove old code** - Only after everything works

### Compatibility Layer:

```python
# cu3vs4_bo/compat.py
"""Compatibility layer for old notebook code."""

def make_gp_regressor(n_features):
    """Legacy function name."""
    from cu3vs4_bo.models import create_gp_regressor
    return create_gp_regressor(n_features)
```

## Timeline

- **Week 1-2**: Core refactoring (package structure, extract functions)
- **Week 3**: Clean notebooks
- **Week 4**: Documentation
- **Week 5**: Testing
- **Week 6**: Final polish and submission prep

**Total: 6 weeks** (can be compressed to 3-4 weeks if full-time)
