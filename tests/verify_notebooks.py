#!/usr/bin/env python3
"""
Notebook Verification Script
Run this to check if your notebooks are properly structured.
"""

import json
from pathlib import Path

def verify_notebook(nb_path):
    """Verify a Jupyter notebook structure."""
    print(f"\n{'='*70}")
    print(f"Checking: {nb_path.name}")
    print(f"{'='*70}")
    
    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)
        
        # Basic structure checks
        assert 'cells' in nb, "Missing 'cells' key"
        assert isinstance(nb['cells'], list), "'cells' must be a list"
        
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
        
        print(f"✓ Valid JSON structure")
        print(f"✓ Total cells: {len(nb['cells'])}")
        print(f"  - Code cells: {len(code_cells)}")
        print(f"  - Markdown cells: {len(md_cells)}")
        
        # Check for %run commands
        run_cells = []
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if '%run' in source and not source.strip().startswith('#'):
                    run_cells.append(i)
        
        if run_cells:
            print(f"⚠️  Contains %run commands in cells: {run_cells}")
        else:
            print(f"✓ No %run commands (good)")
        
        # Check for empty cells
        empty_cells = []
        for i, cell in enumerate(nb['cells']):
            source = ''.join(cell['source']).strip()
            if not source or source == '```':
                empty_cells.append(i)
        
        if empty_cells:
            print(f"⚠️  Empty cells at indices: {empty_cells}")
        else:
            print(f"✓ No empty cells")
        
        print(f"\n✅ {nb_path.name} is valid and ready to use!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("="*70)
    print("NOTEBOOK VERIFICATION TOOL")
    print("="*70)
    
    notebooks_dir = Path(__file__).parent.parent / "Notebooks"
    
    if not notebooks_dir.exists():
        print(f"❌ ERROR: Notebooks directory not found at {notebooks_dir}")
        return
    
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"❌ ERROR: No notebooks found in {notebooks_dir}")
        return
    
    results = {}
    for nb_path in notebooks:
        if nb_path.exists():
            results[nb_path.name] = verify_notebook(nb_path)
        else:
            print(f"\n❌ NOT FOUND: {nb_path}")
            results[nb_path.name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_good = all(results.values())
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    if all_good:
        print("\n🎉 All notebooks are valid and ready to use!")
        print("\nNEXT STEPS:")
        print("1. Open Jupyter Lab/Notebook")
        print("2. Open Notebooks/Cu3VS4_BO_Execute.ipynb")
        print("3. Run the setup and diagnostics cells first")
        print("4. Generate recommendations only after reviewing diagnostics")
    else:
        print("\n⚠️  Some notebooks have issues. Please check the errors above.")
    
    print("="*70)


if __name__ == "__main__":
    main()
