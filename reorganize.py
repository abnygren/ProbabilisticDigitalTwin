#!/usr/bin/env python3
"""
Reorganization script for CuVS_BO_Code project.

This script reorganizes the project structure for better maintainability.
Run with: python3 reorganize.py --dry-run  (to preview)
Run with: python3 reorganize.py            (to execute)
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime

# Project root
ROOT = Path(__file__).parent

# Define new structure
STRUCTURE = {
    'src/': [
        'cuvs_optimizer.py',
        'cuvs_selfvalidating.py',
        'chemical_constants.py',
    ],
    'notebooks/': [
        'Notebooks/Cu3VS4_BO_Execute.ipynb',
        'Notebooks/Initial_Dataset_VSCode.ipynb',
    ],
    'notebooks_archive/': [
        'Notebooks/Cu3VS4_BO_Final.ipynb',
        'Notebooks/Cu3VS4_SelfValidating_BO.ipynb',
        'old notebooks/',
    ],
    'data/': [
        'Data/',
    ],
    'outputs/figures/': [
        # Will move PNG files from Outputs/
    ],
    'outputs/tables/': [
        # Will move CSV files from Outputs/
    ],
    'outputs/archive/': [
        'Outputs_BO/',
        'Outputs_ID/',
    ],
    'docs/development/': [
        'Other/CRITICAL_FIXES.md',
        'Other/FIXES_COMPLETED.md',
        'Other/REFACTORING_PLAN.md',
        'ALL_FIXED_SUMMARY.md',
        'FIXED_BO_FINAL.md',
    ],
    'docs/analysis/': [
        'Other/EI_vs_UCB_ANALYSIS.md',
    ],
    'docs/publication/': [
        'Other/PUBLICATION_CHECKLIST.md',
        'Other/PUBLICATION_ACTION_PLAN.md',
        'Other/PEER_REVIEW_REPORT.md',
        'Other/PEER_REVIEW_Cu3VS4_BO_Final.md',
    ],
    'tests/': [
        'test_import.py',
        'test_selfvalidating.py',
        'verify_notebooks.py',
    ],
    'publication/': [
        'Publication Stuff/',
    ],
}

def move_files(source, dest, dry_run=True):
    """Move file or directory from source to dest."""
    source_path = ROOT / source
    dest_path = ROOT / dest
    
    if not source_path.exists():
        print(f"  ⚠️  Source not found: {source}")
        return False
    
    if dry_run:
        print(f"  📦 Would move: {source} → {dest}")
        return True
    else:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.move(str(source_path), str(dest_path))
        else:
            shutil.move(str(source_path), str(dest_path))
        print(f"  ✓ Moved: {source} → {dest}")
        return True

def organize_outputs(dry_run=True):
    """Organize Outputs/ directory by file type."""
    outputs_dir = ROOT / 'Outputs'
    if not outputs_dir.exists():
        return
    
    figures_dir = ROOT / 'outputs' / 'figures'
    tables_dir = ROOT / 'outputs' / 'tables'
    
    if not dry_run:
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
    
    moved_figures = 0
    moved_tables = 0
    
    for file in outputs_dir.iterdir():
        if file.is_file():
            if file.suffix.lower() == '.png':
                dest = figures_dir / file.name
                if dry_run:
                    print(f"  📦 Would move: Outputs/{file.name} → outputs/figures/{file.name}")
                else:
                    shutil.move(str(file), str(dest))
                    print(f"  ✓ Moved figure: {file.name}")
                moved_figures += 1
            elif file.suffix.lower() == '.csv':
                dest = tables_dir / file.name
                if dry_run:
                    print(f"  📦 Would move: Outputs/{file.name} → outputs/tables/{file.name}")
                else:
                    shutil.move(str(file), str(dest))
                    print(f"  ✓ Moved table: {file.name}")
                moved_tables += 1
    
    if not dry_run and moved_figures + moved_tables > 0:
        # Remove empty Outputs directory
        try:
            outputs_dir.rmdir()
            print(f"  ✓ Removed empty Outputs/ directory")
        except:
            pass
    
    return moved_figures, moved_tables

def create_symlinks(dry_run=True):
    """Create symlinks for backward compatibility (optional)."""
    if dry_run:
        print("\n📌 Would create symlinks for backward compatibility:")
        print("  - src/ → . (for imports)")
        print("  - data/ → Data/")
        print("  - outputs/ → Outputs/")
        return
    
    # Note: Symlinks might not work on all systems, so we'll update imports instead
    pass

def main():
    parser = argparse.ArgumentParser(description='Reorganize CuVS_BO_Code project structure')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without executing')
    args = parser.parse_args()
    
    dry_run = args.dry_run
    
    print("=" * 70)
    print("CuVS_BO_Code Project Reorganization")
    print("=" * 70)
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'EXECUTE'}")
    print()
    
    # Create directory structure
    print("📁 Creating directory structure...")
    for dir_path in STRUCTURE.keys():
        full_path = ROOT / dir_path
        if dry_run:
            print(f"  📦 Would create: {dir_path}")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {dir_path}")
    
    print()
    print("📦 Moving files...")
    
    # Move files according to structure
    for dest_dir, sources in STRUCTURE.items():
        for source in sources:
            if '/' in source and source.endswith('/'):
                # Directory move
                dest = dest_dir + source.rstrip('/').split('/')[-1]
            else:
                # File move - keep same name
                dest = dest_dir + source.split('/')[-1]
            
            move_files(source, dest, dry_run)
    
    print()
    print("📊 Organizing Outputs/ directory...")
    moved_figures, moved_tables = organize_outputs(dry_run)
    if dry_run:
        print(f"  Would move {moved_figures} figures and {moved_tables} tables")
    else:
        print(f"  ✓ Moved {moved_figures} figures and {moved_tables} tables")
    
    print()
    print("=" * 70)
    if dry_run:
        print("DRY RUN COMPLETE - No files were moved")
        print("Run without --dry-run to execute the reorganization")
    else:
        print("REORGANIZATION COMPLETE!")
        print()
        print("⚠️  IMPORTANT: Update import paths in notebooks:")
        print("   Change: sys.path.insert(0, str(Path('..').resolve()))")
        print("   To:     sys.path.insert(0, str(Path('..').resolve() / 'src'))")
    print("=" * 70)

if __name__ == '__main__':
    main()
