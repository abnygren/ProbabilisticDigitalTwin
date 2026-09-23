#!/usr/bin/env python3
"""Check that SelfValidatingOptimizer and campaign data directories import."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from optimizer import Cu3VS4Optimizer
from features import CHEM_FEATURES
from config import RAW_FACTORS, OBJECTIVES
from selfvalidating import SelfValidatingOptimizer

campaigns = {
    "CuI": ROOT / "data_CuI" / "experiments.json",
    "CuBr": ROOT / "data_CuBr" / "experiments.json",
    "CuCl": ROOT / "data_CuCl" / "experiments.json",
    "Ta": ROOT / "data_Ta" / "experiments.json",
}

missing = [name for name, path in campaigns.items() if not path.exists()]
if missing:
    raise SystemExit(f"Missing campaign data: {missing}")

assert Cu3VS4Optimizer is not None
assert SelfValidatingOptimizer is not None
assert RAW_FACTORS == ["Temp", "Time", "VOacac", "DDT", "OAm"]
assert OBJECTIVES == ["Size", "CV", "Squareness"]
assert len(CHEM_FEATURES) >= 5

print("SelfValidatingOptimizer import check passed")
print(f"  Campaign data present: {', '.join(campaigns)}")
print(f"  Chemical features: {len(CHEM_FEATURES)}")
