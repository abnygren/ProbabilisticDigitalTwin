#!/usr/bin/env python3
"""
Quick test to verify the reorganized modules import correctly and the
SelfValidating notebook setup works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*70)
print("TESTING: Self-Validating Optimizer Setup (reorganized)")
print("="*70)

try:
    print("\n1. Testing module imports...")
    from optimizer import Cu3VS4Optimizer
    from features import raw_to_chemical_features, add_chemical_features, CHEM_FEATURES, HYBRID_FEATURES
    from config import RAW_FACTORS, RAW_BOUNDS, OBJECTIVES, CU_PRECURSOR_MMOL, TOTAL_VOLUME_ML
    print("   ✓ Module imports successful")

    print("\n2. Checking configuration constants...")
    DATA_DIR = Path(__file__).parent.parent / "data"
    OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
    print(f"   ✓ DATA_DIR: {DATA_DIR}")

    print("\n3. Testing Cu3VS4Optimizer availability...")
    print(f"   ✓ Cu3VS4Optimizer class: {Cu3VS4Optimizer}")

    print("\n4. Testing feature definitions...")
    print(f"   ✓ RAW_FACTORS: {RAW_FACTORS}")
    print(f"   ✓ OBJECTIVES: {OBJECTIVES}")
    print(f"   ✓ Chemical features available: {len(CHEM_FEATURES)} features")

    print("\n5. Testing SelfValidatingOptimizer import...")
    from selfvalidating import SelfValidatingOptimizer
    print(f"   ✓ SelfValidatingOptimizer: {SelfValidatingOptimizer}")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nYour notebooks should run without errors with the new module structure!")

except Exception as e:
    print("\n" + "="*70)
    print(f"❌ TEST FAILED: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
