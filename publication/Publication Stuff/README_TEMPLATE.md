# Chemically-Informed Bayesian Optimization for Cu₃VS₄ Nanoparticle Synthesis

[![DOI](https://zenodo.org/badge/DOI/10.XXXX/XXXXX.svg)](https://doi.org/10.XXXX/XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for Bayesian optimization of Cu₃VS₄ nanoparticle synthesis conditions, incorporating chemically meaningful features derived from synthesis parameters.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024cu3vs4,
  title={Chemically-Informed Bayesian Optimization for Cu₃VS₄ Nanoparticle Synthesis},
  author={Your Name and Coauthors},
  journal={Journal Name},
  year={2024},
  doi={10.XXXX/XXXXX}
}
```

## Overview

This work implements a hierarchical Bayesian optimization framework that:

- **Combines raw synthesis parameters** (temperature, time, precursor amounts) with **chemically-derived features** (stoichiometric ratios, concentrations)
- Uses **Gaussian Process models** with Matérn kernels for property prediction
- Implements **Expected Improvement** acquisition function for multi-objective optimization
- Provides **uncertainty quantification** and **interpretability tools** (partial dependence plots)

### Key Features

- Hierarchical modeling: Feasibility classification → Property regression
- Chemical feature engineering: Cu:V ratio, metal concentration, effective dielectric constant
- Multi-objective optimization: Size targeting + GSD minimization + Squareness maximization
- Cross-validation and model comparison (raw vs chemical vs hybrid features)

## Installation

### Requirements

- Python 3.8 or higher
- See `requirements.txt` for full dependency list

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/CuVS_BO_Code.git
cd CuVS_BO_Code

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Load Data and Train Model

```python
from cu3vs4_bo import Cu3VS4Optimizer
import pandas as pd

# Load data
df = pd.read_csv('data/processed/COMPLETE_CUVS_DATA_SIDE2.csv')

# Initialize optimizer (hybrid features recommended)
optimizer = Cu3VS4Optimizer(
    df=df,
    feature_mode='hybrid',  # 'raw', 'chemical', or 'hybrid'
    validate=True
)

# View model performance
optimizer.print_metrics()
```

### 2. Generate Recommendations

```python
# Get recommendations for target size
recommendations = optimizer.recommend(
    target_size=20.0,  # nm
    size_tol=2.5,      # ± tolerance
    n_return=2         # Number of recommendations
)

print(recommendations)
```

### 3. Predict Outcomes

```python
# Predict properties for specific conditions
result = optimizer.predict_from_conditions(
    Temp=285.0,
    Time=30.0,
    VOacac=0.20,
    DDT=3.0,
    OAm=4.5
)
```

## Data

The dataset (`data/processed/COMPLETE_CUVS_DATA_SIDE2.csv`) contains:

- **43 experiments** with synthesis conditions and measured properties
- **Raw factors**: Temperature (°C), Time (min), VOacac (mmol), DDT (mL), OAm (mL)
- **Responses**: Size (nm), GSD, Squareness
- **Feasibility**: HasProduct, PhasePure, Polymorph

See `data/README.md` for detailed data description.

## Usage Examples

### Example 1: Model Comparison

Compare raw, chemical, and hybrid feature representations:

```python
from cu3vs4_bo import compare_feature_modes

results = compare_feature_modes(df)
results.plot_comparison()
```

### Example 2: Partial Dependence Analysis

Visualize how features affect predictions:

```python
from cu3vs4_bo.visualization import plot_partial_dependence

plot_partial_dependence(optimizer, target='size')
```

### Example 3: Size Sweep

Generate recommendations across multiple target sizes:

```python
targets = [15, 18, 20, 25, 30]  # nm
for target in targets:
    recs = optimizer.recommend(target_size=target, n_return=1)
    print(f"{target}nm: {recs.iloc[0]}")
```

## Notebooks

See `notebooks/` directory for detailed examples:

1. `01_data_exploration.ipynb` - Data loading and exploration
2. `02_model_training.ipynb` - Model training and validation
3. `03_optimization.ipynb` - Generating recommendations
4. `04_results_analysis.ipynb` - Analyzing results and visualizations

## Methodology

### Chemical Features

| Feature | Formula | Chemical Meaning |
|---------|---------|------------------|
| Cu_V_ratio | CuI / VOacac | Stoichiometry (target = 3.0 for Cu₃VS₄) |
| S_Metal_ratio | DDT_mmol / (CuI + VOacac) | Sulfur excess driving nucleation |
| Ligand_Metal_ratio | OAm_mmol / (CuI + VOacac) | Surface capping density |
| Metal_Conc | (CuI + VOacac) / V_total | Total metal concentration (mM) |
| log_Time | log₁₀(Time) | Linearized reaction kinetics |
| effective_dielectric | Volume-weighted ε | Solvent mixture polarity |

### Model Architecture

- **GP Regression**: Matérn 5/2 kernel with ARD (Automatic Relevance Determination)
- **GP Classification**: For feasibility (HasProduct, PhasePure, IsCubic)
- **Acquisition**: Expected Improvement with multi-objective weighting
- **Validation**: Leave-One-Out Cross-Validation

See `docs/methodology.md` for detailed methodology.

## Project Structure

```
CuVS_BO_Code/
├── cu3vs4_bo/          # Main package
│   ├── models.py       # GP model builders
│   ├── features.py     # Feature engineering
│   ├── optimizer.py    # Main optimizer class
│   └── acquisition.py  # Acquisition functions
├── notebooks/          # Example notebooks
├── data/               # Data files
├── docs/               # Documentation
└── tests/              # Unit tests
```

## Reproducibility

All results can be reproduced using:

```bash
# Set random seed (set in code: np.random.seed(42))
python scripts/reproduce_results.py
```

Key parameters:
- Random seed: 42
- LOO-CV: Leave-One-Out
- GP restarts: 15 (regression), 5 (classification)
- LHS samples: 20,000 candidates

## Testing

Run unit tests:

```bash
pytest tests/
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Your Name
- **Email**: your.email@institution.edu
- **Institution**: Your Institution

## Acknowledgments

- References to relevant papers
- Acknowledgments to collaborators
- Funding sources

## References

1. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.

2. Gelbart, M. A., Snoek, J., & Adams, R. P. (2014). Bayesian optimization with unknown constraints. *arXiv preprint arXiv:1403.5607*.

3. Raccuglia, P., et al. (2016). Machine-learning-assisted materials discovery using failed experiments. *Nature*, 533(7601), 73-76.

4. Pearson, R. G. (1988). Absolute electronegativity and hardness: application to inorganic chemistry. *Inorganic Chemistry*, 27(4), 734-740.
