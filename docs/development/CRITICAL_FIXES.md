# Critical Fixes for Publication (Do These First!)

## 🔴 Must Fix Before Submission

### 1. Remove Monkey Patching (30 minutes)

**File**: `Notebooks/Cu3VS4_BO_Final.ipynb`

**Find**: Cells 22 and 30 (where `types.MethodType` is used)

**Replace**:
```python
# OLD CODE (Cell 22):
def patched_acquisition(self, X, target_size, size_tol):
    # ... code ...
model_hybrid.acquisition = types.MethodType(patched_acquisition, model_hybrid)
```

**With**:
```python
# NEW CODE - Modify Cu3VS4Optimizer class (Cell 17):
class Cu3VS4Optimizer:
    def __init__(self, df, feature_mode='hybrid', validate=True, objective_weights=None):
        # ... existing code ...
        # Add this:
        if objective_weights is None:
            self.objective_weights = self._calculate_weights()
        else:
            self.objective_weights = objective_weights
    
    def _calculate_weights(self):
        """Calculate objective weights from data."""
        # Move weight calculation logic here
        return calculate_objective_weights(
            df_success=self.df_success,
            model_metrics=self.metrics,
            method='comprehensive'
        )
    
    def acquisition(self, X, target_size, size_tol):
        """Compute acquisition function."""
        preds = self.predict(X)
        gsd_best = self.df_success["GSD"].min()
        sq_best = self.df_success["Squareness"].max()
        
        ei_gsd = expected_improvement(preds['gsd_mu'], preds['gsd_std'], gsd_best, minimize=True)
        ei_sq = expected_improvement(preds['sq_mu'], preds['sq_std'], sq_best, minimize=False)
        
        # Normalize
        ei_gsd_n = (ei_gsd - ei_gsd.min()) / (np.ptp(ei_gsd) + 1e-10)
        ei_sq_n = (ei_sq - ei_sq.min()) / (np.ptp(ei_sq) + 1e-10)
        
        # Use self.objective_weights (no monkey patching!)
        w_gsd = self.objective_weights.get("GSD", 1.0)
        w_sq = self.objective_weights.get("Squareness", 1.0)
        acq_obj = (w_gsd * ei_gsd_n + w_sq * ei_sq_n) / (w_gsd + w_sq)
        
        p_size = prob_in_interval(preds['size_mu'], preds['size_std'], target_size, size_tol)
        total = acq_obj * p_size * preds['p_feasible']
        
        return {'total': total, 'acq_obj': acq_obj, 'p_size': p_size, **preds}
```

**Then delete**: Cells 22 and 30 (the patching code)

**Update**: Cell 21 where model is created:
```python
# Calculate weights first
calculated_weights = calculate_objective_weights(
    df_success=df_success,
    model_metrics=model_hybrid.metrics,
    method='comprehensive'
)

# Pass to optimizer
model_hybrid = Cu3VS4Optimizer(
    df, 
    feature_mode='hybrid', 
    validate=True,
    objective_weights=calculated_weights  # Pass here!
)
```

---

### 2. Create requirements.txt (15 minutes)

**Create file**: `requirements.txt` in project root

**Content**:
```txt
numpy>=1.21.0,<1.25.0
pandas>=1.3.0,<2.1.0
scipy>=1.7.0,<1.12.0
scikit-learn>=1.0.0,<1.4.0
matplotlib>=3.4.0,<3.8.0
seaborn>=0.11.0,<0.13.0
statsmodels>=0.13.0,<0.15.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

**Test**: 
```bash
pip install -r requirements.txt
```

---

### 3. Fix Hardcoded Paths (30 minutes)

**File**: `Notebooks/Cu3VS4_BO_Final.ipynb` Cell 4

**Replace**:
```python
ABSOLUTE_PATH = Path("/Users/ariananygren/Desktop/phd files/Projects/Cu3VS4/CuVS_BO_Code/Data/COMPLETE_CUVS_DATA_SIDE2.csv")
```

**With**:
```python
# Use environment variable or relative path only
DATA_PATH = Path(os.getenv("CUVS_DATA_PATH", "..") / "Data" / "COMPLETE_CUVS_DATA_SIDE2.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
```

**Do same for**: OUTPUT_DIR

---

### 4. Create README.md (2-3 hours)

**Create file**: `README.md` in project root

**Copy from**: `README_TEMPLATE.md` and customize:

**Minimum sections**:
1. Title and brief description
2. Installation (copy from template)
3. Quick start example
4. Data description
5. Citation information
6. License

**Test**: Have someone who hasn't seen your code try to install and run it

---

### 5. Fix SelfValidatingOptimizer Dependencies (1-2 hours)

**File**: `Notebooks/Cu3VS4_SelfValidating_BO.ipynb` Cell 3

**Problem**: Requires running another notebook first

**Quick Fix**: Add clear error message
```python
# Cell 3 - Replace the check with:
try:
    from cu3vs4_bo.optimizer import Cu3VS4Optimizer
    BASE_OPTIMIZER_AVAILABLE = True
except ImportError:
    # Try to import from notebook namespace
    if 'Cu3VS4Optimizer' in globals():
        BASE_OPTIMIZER_AVAILABLE = True
        print("✓ Using Cu3VS4Optimizer from notebook namespace")
    else:
        BASE_OPTIMIZER_AVAILABLE = False
        raise ImportError(
            "Cu3VS4Optimizer not found. Please either:\n"
            "1. Run Cu3VS4_BO_Final.ipynb first, OR\n"
            "2. Install cu3vs4_bo package: pip install -e ."
        )
```

**Better Fix**: Extract optimizer to module (see REFACTORING_PLAN.md)

---

### 6. Add LICENSE File (5 minutes)

**Create file**: `LICENSE` in project root

**Content** (MIT License):
```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 7. Clean One Example Notebook (2-3 hours)

**Pick**: `Cu3VS4_BO_Final.ipynb` (main notebook)

**Remove**:
- [ ] All commented-out code blocks
- [ ] Debugging print statements (keep important ones)
- [ ] Hardcoded absolute paths
- [ ] Personal notes
- [ ] Failed experiments
- [ ] Temporary visualizations

**Add**:
- [ ] Clear markdown explanations
- [ ] Section headers
- [ ] Brief methodology notes
- [ ] Output cells (so reviewers can see results)

**Test**: Run notebook from start to finish, ensure it works

---

### 8. Create CITATION.cff (10 minutes)

**Create file**: `CITATION.cff` in project root

**Content**:
```yaml
cff-version: 1.2.0
title: "Chemically-Informed Bayesian Optimization for Cu₃VS₄ Nanoparticle Synthesis"
authors:
  - given-names: Your
    family-names: Name
    affiliation: Your Institution
version: 1.0.0
date-released: 2024-01-20
license: MIT
repository-code: https://github.com/yourusername/CuVS_BO_Code
```

**Update**: After publication, add DOI:
```yaml
doi: 10.XXXX/XXXXX
```

---

## Quick Test Checklist

After making fixes, test:

1. **Clean Environment Test**:
   ```bash
   # Create new virtual environment
   python -m venv test_env
   source test_env/bin/activate
   pip install -r requirements.txt
   # Try to run notebook
   ```

2. **Import Test**:
   ```python
   # In Python
   import numpy as np
   import pandas as pd
   from sklearn.gaussian_process import GaussianProcessRegressor
   # All imports should work
   ```

3. **Notebook Test**:
   - Open notebook
   - Run "Restart & Run All"
   - Should complete without errors

4. **Path Test**:
   - Move project to different location
   - Should still work (no hardcoded paths)

---

## Priority Order

**Do these in order**:

1. ✅ Remove monkey patching (30 min) - **CRITICAL**
2. ✅ Create requirements.txt (15 min) - **CRITICAL**
3. ✅ Fix hardcoded paths (30 min) - **CRITICAL**
4. ✅ Create README.md (2-3 hours) - **IMPORTANT**
5. ✅ Add LICENSE (5 min) - **REQUIRED**
6. ✅ Create CITATION.cff (10 min) - **REQUIRED**
7. ✅ Clean one notebook (2-3 hours) - **IMPORTANT**
8. ✅ Fix dependencies (1-2 hours) - **IMPORTANT**

**Total time**: ~8-10 hours for critical fixes

---

## After Critical Fixes

Once these are done, you have a **minimally viable publication package**.

**Next steps** (see PUBLICATION_ACTION_PLAN.md):
- Add docstrings
- Create more example notebooks
- Add tests
- Archive code (get DOI)
- Full documentation

But the critical fixes above are the **minimum** needed for submission.

---

## Need Help?

**Common Issues**:

1. **"Module not found"**: Check requirements.txt, install dependencies
2. **"Path not found"**: Use relative paths, check file locations
3. **"Method not found"**: Check that monkey patching was removed
4. **"Can't reproduce results"**: Check random seeds, document all parameters

**Resources**:
- See `PUBLICATION_ACTION_PLAN.md` for detailed plan
- See `REFACTORING_PLAN.md` for code organization
- See `README_TEMPLATE.md` for documentation

Good luck! 🚀
