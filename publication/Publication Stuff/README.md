# Chemically-Informed Bayesian Optimization for Cu₃VS₄ Nanoparticle Synthesis

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

### Using the Main Optimization Notebook

1. **Open the notebook**: `Notebooks/Cu3VS4_BO_Final.ipynb`

2. **Run cells sequentially**:
   - Cell 1-4: Imports and configuration
   - Cell 5-8: Feature engineering and data loading
   - Cell 9-17: Model building (GP models)
   - Cell 18-24: Model comparison and selection
   - Cell 25-37: Optimization and recommendations

3. **Generate recommendations**:
   ```python
   recommendations = model.recommend(
       target_size=20.0,  # nm
       size_tol=2.5,      # ± tolerance
       n_return=2         # Number of recommendations
   )
   ```

### Data Path Configuration

The code uses relative paths by default. To use a custom data path:

```bash
export CUVS_DATA_PATH="/path/to/your/data.csv"
```

Or modify the path in Cell 4 of the notebook.

## Data

The dataset (`Data/COMPLETE_CUVS_DATA_SIDE2.csv`) contains:

- **43 experiments** with synthesis conditions and measured properties
- **Raw factors**: Temperature (°C), Time (min), VOacac (mmol), DDT (mL), OAm (mL)
- **Responses**: Size (nm), GSD, Squareness
- **Feasibility**: HasProduct, PhasePure, Polymorph

## Notebooks

- **`Cu3VS4_BO_Final.ipynb`**: Main Bayesian optimization notebook
  - Model training and comparison
  - Recommendation generation
  - Partial dependence analysis
  - Feature importance

- **`Cu3VS4_SelfValidating_BO.ipynb`**: Self-validating optimization system
  - Tracks recommendation lifecycle
  - Learns from prediction errors
  - Requires running `Cu3VS4_BO_Final.ipynb` first

- **`Initial_Dataset_VSCode.ipynb`**: Design of Experiments analysis
  - ANOVA analysis
  - Response surface visualization
  - Factor screening

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

## Reproducibility

All results can be reproduced using:

- Random seed: 42 (set in code)
- LOO-CV: Leave-One-Out Cross-Validation
- GP restarts: 15 (regression), 5 (classification)
- LHS samples: 20,000 candidates

## Project Structure

```
CuVS_BO_Code/
├── Notebooks/          # Jupyter notebooks
│   ├── Cu3VS4_BO_Final.ipynb
│   ├── Cu3VS4_SelfValidating_BO.ipynb
│   └── Initial_Dataset_VSCode.ipynb
├── Data/               # Data files
├── Outputs/            # Generated outputs
├── chemical_constants.py  # Chemical feature constants
├── requirements.txt    # Python dependencies
├── LICENSE            # MIT License
└── README.md          # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Your Name
- **Email**: your.email@institution.edu
- **Institution**: Your Institution

## References

1. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.

2. Gelbart, M. A., Snoek, J., & Adams, R. P. (2014). Bayesian optimization with unknown constraints. *arXiv preprint arXiv:1403.5607*.

3. Raccuglia, P., et al. (2016). Machine-learning-assisted materials discovery using failed experiments. *Nature*, 533(7601), 73-76.

4. Pearson, R. G. (1988). Absolute electronegativity and hardness: application to inorganic chemistry. *Inorganic Chemistry*, 27(4), 734-740.
