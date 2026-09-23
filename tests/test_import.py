#!/usr/bin/env python3
"""Check that the src package imports and that feature transforms run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import RAW_FACTORS, OBJECTIVES, CU_PRECURSOR_MMOL
from features import raw_to_chemical_features, CHEM_FEATURES
from optimizer import Cu3VS4Optimizer, make_gp_regressor
from selfvalidating import SelfValidatingOptimizer, ErrorLearner
from experiment_store import ExperimentStore, RecommendationStore
from diagnostics import loo_cv, compare_feature_modes, print_model_assessment
from visualization import plot_parity, plot_calibration, plot_feature_importance

import numpy as np

result = raw_to_chemical_features(
    Temp=np.array([280.0]),
    Time=np.array([30.0]),
    VOacac=np.array([0.20]),
    DDT=np.array([3.0]),
    OAm=np.array([4.0]),
)

assert RAW_FACTORS == ["Temp", "Time", "VOacac", "DDT", "OAm"]
assert OBJECTIVES == ["Size", "CV", "Squareness"]
assert abs(result["Cu_V_ratio"][0] - CU_PRECURSOR_MMOL / 0.20) < 1e-9
assert "Cu_V_ratio" in CHEM_FEATURES
assert Cu3VS4Optimizer is not None
assert SelfValidatingOptimizer is not None
assert ErrorLearner is not None
assert ExperimentStore is not None
assert RecommendationStore is not None
assert callable(loo_cv)
assert callable(compare_feature_modes)
assert callable(print_model_assessment)
assert callable(make_gp_regressor)
assert callable(plot_parity)
assert callable(plot_calibration)
assert callable(plot_feature_importance)

print("Imports and feature transform check passed")
print(f"  RAW_FACTORS: {RAW_FACTORS}")
print(f"  Cu_V_ratio at 0.20 mmol V: {result['Cu_V_ratio'][0]:.3f}")
print(f"  Metal_Conc: {result['Metal_Conc'][0]:.2f} mM")
