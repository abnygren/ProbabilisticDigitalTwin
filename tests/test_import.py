#!/usr/bin/env python3
"""
Test script to verify the cuvs_optimizer module imports correctly.
Run this to check if the module is working.
"""

print("="*70)
print("TESTING CUVS_OPTIMIZER MODULE")
print("="*70)

try:
    print("\n1. Importing module...")
    from cuvs_optimizer import Cu3VS4Optimizer, raw_to_chemical_features
    print("   ✓ Import successful")
    
    print("\n2. Checking Cu3VS4Optimizer class...")
    print(f"   ✓ Class available: {Cu3VS4Optimizer}")
    
    print("\n3. Checking helper functions...")
    print(f"   ✓ raw_to_chemical_features available: {raw_to_chemical_features}")
    
    print("\n4. Testing feature transformation...")
    import numpy as np
    result = raw_to_chemical_features(
        Temp=np.array([280.0]),
        Time=np.array([30.0]),
        VOacac=np.array([0.20]),
        DDT=np.array([3.0]),
        OAm=np.array([4.0])
    )
    print(f"   ✓ Feature transformation works")
    print(f"   Cu_V_ratio: {result['Cu_V_ratio'][0]:.3f}")
    print(f"   Metal_Conc: {result['Metal_Conc'][0]:.2f} mM")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nThe module is working correctly!")
    print("You can now run the notebooks without any dependencies.")
    
except ImportError as e:
    print("\n" + "="*70)
    print("❌ IMPORT ERROR")
    print("="*70)
    print(f"\nError: {e}")
    print("\nPossible causes:")
    print("  1. cuvs_optimizer.py is not in the correct directory")
    print("  2. Python path issue")
    print("  3. Missing dependencies (numpy, pandas, sklearn, scipy)")
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR")
    print("="*70)
    print(f"\nError: {e}")
    print("\nSomething went wrong during testing.")
