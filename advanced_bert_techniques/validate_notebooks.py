#!/usr/bin/env python3
"""
Script to validate that all notebooks can be executed without syntax errors
"""

import json
import os
import sys
import traceback
from pathlib import Path

def validate_notebook_syntax(notebook_path):
    """
    Validate that a notebook's code cells have valid Python syntax
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        errors = []
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source
                
                # Skip empty cells
                if not code.strip():
                    continue
                
                # Check for basic syntax errors
                try:
                    compile(code, f'<cell-{i}>', 'exec')
                except SyntaxError as e:
                    errors.append(f"Cell {i}: {e}")
                except Exception as e:
                    # Other compilation errors
                    errors.append(f"Cell {i}: {type(e).__name__}: {e}")
        
        return errors
        
    except json.JSONDecodeError as e:
        return [f"JSON parsing error: {e}"]
    except Exception as e:
        return [f"Unexpected error: {e}"]

def main():
    """Validate all notebooks in the current directory"""
    
    print("[VALIDATION] BERT TECHNIQUE NOTEBOOKS")
    print("=" * 50)
    
    notebook_files = list(Path('.').glob('*.ipynb'))
    notebook_files.sort()
    
    if not notebook_files:
        print("[ERROR] No notebook files found!")
        return 1
    
    total_notebooks = len(notebook_files)
    valid_notebooks = 0
    validation_results = []
    
    for notebook_path in notebook_files:
        print(f"[CHECK] {notebook_path.name}...", end=" ")
        
        errors = validate_notebook_syntax(notebook_path)
        
        if errors:
            print("ERRORS FOUND")
            validation_results.append({
                'notebook': notebook_path.name,
                'status': 'FAILED',
                'errors': errors
            })
        else:
            print("OK")
            valid_notebooks += 1
            validation_results.append({
                'notebook': notebook_path.name,
                'status': 'PASSED',
                'errors': []
            })
    
    print("\n" + "=" * 50)
    print("[SUMMARY] VALIDATION RESULTS")
    print("=" * 50)
    
    print(f"Total notebooks: {total_notebooks}")
    print(f"Valid notebooks: {valid_notebooks}")
    print(f"Invalid notebooks: {total_notebooks - valid_notebooks}")
    print(f"Success rate: {(valid_notebooks / total_notebooks) * 100:.1f}%")
    
    # Show detailed results for failed notebooks
    failed_notebooks = [r for r in validation_results if r['status'] == 'FAILED']
    
    if failed_notebooks:
        print("\n[FAILED] NOTEBOOKS WITH ERRORS:")
        for result in failed_notebooks:
            print(f"\n{result['notebook']}:")
            for error in result['errors']:
                print(f"  - {error}")
    else:
        print("\n[SUCCESS] ALL NOTEBOOKS PASSED VALIDATION!")
    
    # Create summary report
    with open('validation_report.txt', 'w') as f:
        f.write(f"Notebook Validation Report\\n")
        f.write(f"Generated: {__import__('datetime').datetime.now()}\\n")
        f.write(f"Total notebooks: {total_notebooks}\\n")
        f.write(f"Valid notebooks: {valid_notebooks}\\n")
        f.write(f"Success rate: {(valid_notebooks / total_notebooks) * 100:.1f}%\\n\\n")
        
        for result in validation_results:
            f.write(f"{result['notebook']}: {result['status']}\\n")
            if result['errors']:
                for error in result['errors']:
                    f.write(f"  - {error}\\n")
                f.write("\\n")
    
    print("\\n[REPORT] Detailed report saved to 'validation_report.txt'")
    
    # Return exit code
    return 0 if valid_notebooks == total_notebooks else 1

if __name__ == "__main__":
    sys.exit(main())