#!/usr/bin/env python3
"""Check that the four campaign notebooks are valid JSON with the expected cells."""

import json
from pathlib import Path

NOTEBOOKS = [
    "Cu3VS4_BO_Execute_CuI.ipynb",
    "Cu3VS4_BO_Execute_CuBr.ipynb",
    "Cu3VS4_BO_Execute_CuCl.ipynb",
    "Cu3VS4_BO_Execute_Ta.ipynb",
]


def check_notebook(path: Path) -> None:
    with open(path) as f:
        nb = json.load(f)
    if "cells" not in nb:
        raise ValueError(f"{path.name}: missing cells")
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    if not code:
        raise ValueError(f"{path.name}: no code cells")
    joined = "\n".join("".join(c.get("source", [])) for c in code)
    if "SelfValidatingOptimizer" not in joined:
        raise ValueError(f"{path.name}: SelfValidatingOptimizer not found")
    print(f"  {path.name}: {len(nb['cells'])} cells ({len(code)} code)")


def main():
    notebooks_dir = Path(__file__).parent.parent / "Notebooks"
    print("Campaign notebooks")
    for name in NOTEBOOKS:
        path = notebooks_dir / name
        if not path.exists():
            raise SystemExit(f"Missing notebook: {path}")
        check_notebook(path)
    print("Notebook check passed")


if __name__ == "__main__":
    main()
