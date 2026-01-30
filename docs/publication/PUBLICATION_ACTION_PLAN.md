# Publication Action Plan: Cu₃VS₄ Bayesian Optimization

## Executive Summary

**Goal**: Make code publication-ready for chemistry journal submission

**Timeline**: 4-6 weeks (can be compressed to 2-3 weeks full-time)

**Priority**: Fix critical issues first, then polish

---

## Critical Issues (Must Fix Before Submission)

### 🔴 Priority 1: Code Architecture (Week 1)

#### 1.1 Remove Monkey Patching
**Location**: `Cu3VS4_BO_Final.ipynb` cells 22, 30

**Problem**: Using `types.MethodType` to patch methods at runtime
```python
# BAD - Current code
model.acquisition = types.MethodType(patched_acquisition, model)
```

**Fix**: Refactor `Cu3VS4Optimizer.__init__` to accept `objective_weights`
```python
# GOOD - Refactored
class Cu3VS4Optimizer:
    def __init__(self, df, feature_mode='hybrid', objective_weights=None):
        self.objective_weights = objective_weights or self._calculate_weights()
    
    def acquisition(self, X, target_size, size_tol):
        # Use self.objective_weights directly
        w_gsd = self.objective_weights.get("GSD", 1.0)
        w_sq = self.objective_weights.get("Squareness", 1.0)
        # ... rest of method
```

**Time**: 2-3 hours

---

#### 1.2 Extract Code to Package
**Problem**: Code duplicated across notebooks, hard to maintain

**Fix**: Create `cu3vs4_bo/` package structure
```
cu3vs4_bo/
├── __init__.py
├── features.py      # Feature engineering (from cell 6)
├── models.py        # GP builders (from cell 11)
├── acquisition.py   # Acquisition functions (from cell 12)
├── optimizer.py     # Main class (from cell 17)
└── utils.py         # Helpers (sampling, CV, etc.)
```

**Steps**:
1. Create package directory
2. Extract functions from notebooks
3. Add proper imports
4. Update notebooks to import from package

**Time**: 1-2 days

---

#### 1.3 Fix Path Handling
**Problem**: Hardcoded absolute paths, complex fallback logic

**Fix**: Use configuration file or environment variables
```python
# config.py
DATA_PATH = Path(os.getenv("CUVS_DATA_PATH", "data/COMPLETE_CUVS_DATA_SIDE2.csv"))
OUTPUT_DIR = Path(os.getenv("CUVS_OUTPUT_DIR", "outputs"))
```

**Time**: 1-2 hours

---

### 🔴 Priority 2: Dependency Management (Week 1)

#### 2.1 Create requirements.txt
**Problem**: No dependency list, reviewers can't reproduce

**Fix**: Create `requirements.txt` with pinned versions
```bash
numpy>=1.21.0,<1.25.0
pandas>=1.3.0,<2.1.0
scikit-learn>=1.0.0,<1.4.0
# ... see requirements_template.txt
```

**Time**: 30 minutes

---

#### 2.2 Fix Import Dependencies
**Problem**: `SelfValidatingOptimizer` depends on notebook execution order

**Fix**: Make imports explicit
```python
# cu3vs4_bo/self_validating.py
from cu3vs4_bo.optimizer import Cu3VS4Optimizer
from cu3vs4_bo.features import compute_chemical_features
```

**Time**: 2-3 hours

---

### 🟡 Priority 3: Documentation (Week 2)

#### 3.1 Create README.md
**Problem**: No installation/usage instructions

**Fix**: Use `README_TEMPLATE.md` as starting point

**Must Include**:
- Installation instructions
- Quick start example
- Data description
- Citation information
- License

**Time**: 4-6 hours

---

#### 3.2 Add Docstrings
**Problem**: Functions lack documentation

**Fix**: Add comprehensive docstrings to all public functions
- Parameters with types
- Return values
- Examples
- References (for chemical features)

**Time**: 1-2 days

---

#### 3.3 Create Methodology Documentation
**Problem**: Chemical feature derivations not documented

**Fix**: Create `docs/methodology.md` with:
- Chemical feature equations
- GP kernel choices
- Acquisition function details
- Validation procedures

**Time**: 1 day

---

### 🟡 Priority 4: Clean Notebooks (Week 2-3)

#### 4.1 Create Clean Example Notebooks
**Problem**: Notebooks contain debugging code, comments, failed experiments

**Fix**: Create 4 clean notebooks:
1. `01_data_exploration.ipynb` - Load and explore data
2. `02_model_training.ipynb` - Train and validate models
3. `03_optimization.ipynb` - Generate recommendations
4. `04_results_analysis.ipynb` - Analyze results

**Remove**:
- [ ] All commented-out code
- [ ] Debugging print statements
- [ ] Failed experiments
- [ ] Personal notes
- [ ] Hardcoded paths

**Time**: 2-3 days

---

### 🟢 Priority 5: Testing (Week 3)

#### 5.1 Add Basic Tests
**Problem**: No tests, can't verify code works

**Fix**: Add pytest tests for:
- Feature engineering functions
- Model initialization
- Recommendation generation
- Data loading

**Time**: 1-2 days

---

### 🟢 Priority 6: Final Polish (Week 4)

#### 6.1 Add License File
**Fix**: Create `LICENSE` file (MIT recommended)

**Time**: 15 minutes

---

#### 6.2 Create CITATION.cff
**Fix**: Create citation metadata file

**Time**: 30 minutes

---

#### 6.3 Archive Code
**Fix**: Upload to Zenodo/Figshare, get DOI

**Time**: 1-2 hours

---

## Detailed Week-by-Week Plan

### Week 1: Core Refactoring
**Goal**: Fix critical architecture issues

**Day 1-2**: Extract code to package
- [ ] Create `cu3vs4_bo/` directory structure
- [ ] Extract feature engineering functions
- [ ] Extract GP model builders
- [ ] Extract acquisition functions
- [ ] Extract optimizer class

**Day 3**: Fix monkey patching
- [ ] Refactor `Cu3VS4Optimizer.__init__`
- [ ] Remove all `types.MethodType` usage
- [ ] Test that functionality unchanged

**Day 4**: Fix dependencies
- [ ] Create `requirements.txt`
- [ ] Fix import statements
- [ ] Test on clean environment

**Day 5**: Path handling
- [ ] Create config module
- [ ] Remove hardcoded paths
- [ ] Use environment variables

---

### Week 2: Documentation & Cleanup
**Goal**: Make code understandable and usable

**Day 1-2**: README and documentation
- [ ] Write comprehensive README
- [ ] Create methodology docs
- [ ] Add docstrings to all functions

**Day 3-4**: Clean notebooks
- [ ] Create clean example notebooks
- [ ] Remove debugging code
- [ ] Add clear markdown explanations
- [ ] Ensure notebooks run end-to-end

**Day 5**: Review and test
- [ ] Test all notebooks
- [ ] Review documentation
- [ ] Fix any issues

---

### Week 3: Testing & Validation
**Goal**: Ensure code works correctly

**Day 1-2**: Add tests
- [ ] Write unit tests
- [ ] Test feature engineering
- [ ] Test model fitting
- [ ] Test recommendations

**Day 3**: Integration testing
- [ ] Test full workflow
- [ ] Test on different Python versions
- [ ] Test on different OS

**Day 4-5**: Fix issues
- [ ] Fix any bugs found
- [ ] Improve error messages
- [ ] Add input validation

---

### Week 4: Final Polish & Submission Prep
**Goal**: Prepare for journal submission

**Day 1**: Final code review
- [ ] Review all code
- [ ] Check documentation
- [ ] Verify reproducibility

**Day 2**: Create submission files
- [ ] Create LICENSE
- [ ] Create CITATION.cff
- [ ] Update README with DOI

**Day 3**: Archive code
- [ ] Upload to Zenodo
- [ ] Get DOI
- [ ] Update README with DOI

**Day 4-5**: Final checks
- [ ] Test on clean environment
- [ ] Verify all examples work
- [ ] Check documentation completeness
- [ ] Prepare submission package

---

## Quick Start Guide (If Time-Crunched)

**Minimum Viable Publication Package** (1-2 weeks):

1. **Fix Critical Issues** (3-4 days):
   - [ ] Remove monkey patching
   - [ ] Create requirements.txt
   - [ ] Fix import dependencies
   - [ ] Remove hardcoded paths

2. **Basic Documentation** (2-3 days):
   - [ ] Write README.md
   - [ ] Add docstrings to main functions
   - [ ] Create one clean example notebook

3. **Submission Prep** (1 day):
   - [ ] Add LICENSE
   - [ ] Archive code (get DOI)
   - [ ] Final review

**Total: 1-2 weeks** for minimal viable package

---

## Checklist for Journal Submission

### Code Quality
- [ ] No monkey patching
- [ ] Code organized in package structure
- [ ] All functions have docstrings
- [ ] Type hints added (optional but recommended)
- [ ] Input validation added
- [ ] Error handling consistent
- [ ] No hardcoded paths
- [ ] Random seeds documented

### Documentation
- [ ] README.md complete
- [ ] Installation instructions work
- [ ] Usage examples provided
- [ ] Methodology documented
- [ ] API reference (if applicable)
- [ ] Citation information included

### Reproducibility
- [ ] requirements.txt with pinned versions
- [ ] Python version specified
- [ ] Data files included or DOI provided
- [ ] All notebooks run end-to-end
- [ ] Results reproducible with documented seed
- [ ] Tested on clean environment

### Submission Files
- [ ] LICENSE file added
- [ ] CITATION.cff created
- [ ] Code archived on Zenodo/Figshare
- [ ] DOI obtained and added to README
- [ ] Data archived separately (if large)

### Journal-Specific
- [ ] Code availability statement prepared
- [ ] Computational environment described
- [ ] All hyperparameters documented
- [ ] Figures reproducible from code
- [ ] Supplementary materials organized

---

## Common Journal Requirements

### Nature/Science
- Code review by independent researcher
- Full reproducibility test
- Performance benchmarks
- Detailed methodology

### ACS Journals (JACS, etc.)
- Detailed methodology section
- Code comments explaining chemistry
- Validation against known results
- Clear data availability statement

### RSC Journals
- Code archived with DOI
- Clear documentation
- Reproducibility emphasized

### Chemistry of Materials
- Detailed experimental methods
- Code well-documented
- Data availability

---

## Resources

### Templates Created:
- `README_TEMPLATE.md` - Use as starting point
- `requirements_template.txt` - Dependency list
- `REFACTORING_PLAN.md` - Detailed refactoring guide
- `PUBLICATION_CHECKLIST.md` - Complete checklist

### External Resources:
- [Zenodo](https://zenodo.org/) - Code archiving
- [Figshare](https://figshare.com/) - Data/code archiving
- [Citation File Format](https://citation-file-format.github.io/) - CITATION.cff
- [Software Carpentry](https://software-carpentry.org/) - Best practices

---

## Questions to Answer Before Submission

1. **Can a colleague reproduce your results?**
   - Test with someone who hasn't seen the code

2. **Can reviewers run your code?**
   - Test on clean environment
   - Provide clear instructions

3. **Is your methodology clear?**
   - Document all assumptions
   - Explain chemical features
   - Justify model choices

4. **Is your code maintainable?**
   - Well-organized
   - Documented
   - Tested

5. **Is your code citable?**
   - DOI assigned
   - Citation information provided
   - License allows use

---

## Final Notes

**Remember**: 
- Code doesn't need to be perfect, but it needs to be:
  - Reproducible
  - Understandable
  - Usable
  - Citable

**Focus on**:
- Making it work for reviewers
- Clear documentation
- Reproducibility
- Following journal guidelines

**Don't worry about**:
- Perfect code architecture (good enough is fine)
- Extensive testing (basic tests sufficient)
- Advanced features (core functionality is enough)

**Timeline**: 4-6 weeks is realistic, but can be compressed to 2-3 weeks if focused

Good luck with your submission! 🚀
