#!/usr/bin/env python3
"""
Test script to verify the reorganized src/ package imports correctly.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*70)
print("TESTING REORGANIZED src/ PACKAGE")
print("="*70)

try:
    print("\n1. Importing config...")
    from config import RAW_FACTORS, RAW_BOUNDS, OBJECTIVES, COLORS, CUI_MMOL
    print(f"   ✓ RAW_FACTORS: {RAW_FACTORS}")
    print(f"   ✓ OBJECTIVES: {OBJECTIVES}")

    print("\n2. Importing features...")
    from features import raw_to_chemical_features, add_chemical_features, CHEM_FEATURES
    print(f"   ✓ CHEM_FEATURES: {CHEM_FEATURES}")

    print("\n3. Testing feature transformation...")
    import numpy as np
    result = raw_to_chemical_features(
        Temp=np.array([280.0]),
        Time=np.array([30.0]),
        VOacac=np.array([0.20]),
        DDT=np.array([3.0]),
        OAm=np.array([4.0])
    )
    print(f"   ✓ Cu_V_ratio: {result['Cu_V_ratio'][0]:.3f}")
    print(f"   ✓ Metal_Conc: {result['Metal_Conc'][0]:.2f} mM")

    print("\n4. Importing optimizer...")
    from optimizer import Cu3VS4Optimizer, make_gp_regressor, expected_improvement
    print(f"   ✓ Cu3VS4Optimizer: {Cu3VS4Optimizer}")

    print("\n5. Importing selfvalidating...")
    from selfvalidating import SelfValidatingOptimizer, ErrorLearner
    print(f"   ✓ SelfValidatingOptimizer: {SelfValidatingOptimizer}")

    print("\n6. Importing experiment_store...")
    from experiment_store import ExperimentStore, RecommendationStore
    print(f"   ✓ ExperimentStore: {ExperimentStore}")

    print("\n7. Importing diagnostics...")
    from diagnostics import loo_cv, compare_feature_modes, print_model_assessment
    print(f"   ✓ loo_cv, compare_feature_modes, print_model_assessment loaded")

    print("\n8. Importing visualization...")
    from visualization import (
        plot_parity, plot_calibration, plot_feature_importance,
        plot_dataset_quality_dashboard, plot_loo_residuals,
    )
    print(f"   ✓ All visualization functions loaded")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nThe reorganized package is working correctly!")

except ImportError as e:
    print("\n" + "="*70)
    print("❌ IMPORT ERROR")
    print("="*70)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR")
    print("="*70)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
