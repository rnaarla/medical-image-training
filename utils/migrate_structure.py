#!/usr/bin/env python3
"""
Project Structure Migration Script

Updates all import statements to work with the new organized structure.
Provides backward compatibility and validates the reorganization.
"""

import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def update_imports_in_file(file_path: Path):
    """Update import statements in a single file."""
    if not file_path.exists() or file_path.suffix != '.py':
        return
    
    print(f"📝 Updating imports in {file_path.name}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update imports
        replacements = [
            # Utils imports -> src.core imports
            (r'from utils\.model import', 'from src.core.model import'),
            (r'from utils\.data import', 'from src.core.data import'),
            (r'from utils\.metrics import', 'from src.core.metrics import'),
            (r'import utils\.', 'import src.core.'),
            
            # Grayscale wrapper imports -> src.core imports
            (r'from grayscale_wrapper import', 'from src.core.grayscale_wrapper import'),
            (r'import grayscale_wrapper', 'import src.core.grayscale_wrapper'),
            
            # Medical pipeline imports -> src.medical imports
            (r'from medical_data_pipeline import', 'from src.medical.medical_data_pipeline import'),
            (r'import medical_data_pipeline', 'import src.medical.medical_data_pipeline'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Add project config import if needed
        if 'from src.' in content and 'from project_config import' not in content:
            import_lines = []
            other_lines = []
            in_imports = False
            
            for line in content.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_lines.append(line)
                    in_imports = True
                elif in_imports and line.strip() == '':
                    import_lines.append(line)
                else:
                    if in_imports and import_lines:
                        # Add project config import before first non-import line
                        import_lines.insert(-1, '# Project configuration for reorganized structure')
                        import_lines.insert(-1, 'sys.path.insert(0, str(Path(__file__).parent))')
                        import_lines.insert(-1, 'from project_config import setup_imports')
                        import_lines.append('')
                        in_imports = False
                    other_lines.append(line)
            
            if import_lines:
                content = '\n'.join(import_lines + other_lines)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Updated {file_path.name}")
        else:
            print(f"📋 No changes needed for {file_path.name}")
            
    except Exception as e:
        print(f"❌ Error updating {file_path.name}: {e}")

def update_script_files():
    """Update shell script files with new paths."""
    script_files = [
        PROJECT_ROOT / "scripts" / "setup_env.sh",
        PROJECT_ROOT / "scripts" / "deploy_aws.sh",
        PROJECT_ROOT / "scripts" / "deploy_training.sh"
    ]
    
    for script_file in script_files:
        if not script_file.exists():
            continue
            
        print(f"📝 Updating script {script_file.name}")
        
        try:
            with open(script_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Update Python import tests
            content = content.replace(
                'from utils.model import get_model',
                'from src.core.model import get_model'
            )
            content = content.replace(
                'from utils.metrics import MetricsTracker',
                'from src.core.metrics import MetricsTracker'
            )
            
            if content != original_content:
                with open(script_file, 'w') as f:
                    f.write(content)
                print(f"✅ Updated {script_file.name}")
                
        except Exception as e:
            print(f"❌ Error updating {script_file.name}: {e}")

def create_compatibility_layer():
    """Create backward compatibility imports."""
    
    # Create compatibility imports in root directory
    compat_files = [
        ('grayscale_wrapper.py', 'from src.core.grayscale_wrapper import *'),
        ('medical_data_pipeline.py', 'from src.medical.medical_data_pipeline import *'),
    ]
    
    for filename, import_statement in compat_files:
        compat_path = PROJECT_ROOT / filename
        if not compat_path.exists():
            with open(compat_path, 'w') as f:
                f.write(f'"""Backward compatibility import for {filename}"""\n')
                f.write('import warnings\n')
                f.write(f'warnings.warn("Importing from root is deprecated. Use new structure: {import_statement}", DeprecationWarning)\n')
                f.write(f'{import_statement}\n')
            print(f"✅ Created compatibility layer: {filename}")

def validate_new_structure():
    """Validate that the new structure works correctly."""
    print("\n🔍 Validating new structure...")
    
    try:
        # Test core imports
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from src.core.grayscale_wrapper import EnterpriseGrayscaleProcessor
        print("✅ Core grayscale module import successful")
        
        # Test medical imports
        from src.medical.medical_data_pipeline import MedicalImageValidator
        print("✅ Medical pipeline module import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import validation failed: {e}")
        return False

def main():
    """Main migration function."""
    print("🏗️  Starting Project Structure Migration")
    print("=" * 50)
    
    # Update Python files
    python_files = [
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "evaluate.py", 
        PROJECT_ROOT / "onnx_export.py",
        PROJECT_ROOT / "infer_triton.py",
        PROJECT_ROOT / "encrypt_model.py",
    ]
    
    # Update files in tests directory
    tests_dir = PROJECT_ROOT / "tests"
    if tests_dir.exists():
        python_files.extend(tests_dir.glob("*.py"))
    
    # Update files in scripts directory  
    scripts_dir = PROJECT_ROOT / "scripts"
    if scripts_dir.exists():
        python_files.extend(scripts_dir.glob("*.py"))
    
    for file_path in python_files:
        if file_path.exists():
            update_imports_in_file(file_path)
    
    # Update shell scripts
    update_script_files()
    
    # Create backward compatibility layer
    create_compatibility_layer()
    
    # Validate structure
    if validate_new_structure():
        print("\n🎉 Migration completed successfully!")
        print("\n📁 NEW PROJECT STRUCTURE:")
        print("├── src/")
        print("│   ├── core/           # Core processing modules")
        print("│   └── medical/        # Medical-specific modules")  
        print("├── tests/             # All test files")
        print("├── scripts/           # Deployment and utility scripts")
        print("├── docs/              # Documentation")
        print("├── infrastructure/    # Terraform and deployment configs")
        print("├── cuda/              # CUDA kernels and setup")
        print("└── charts/            # Helm charts")
        
    else:
        print("\n⚠️  Migration completed with warnings. Check import errors above.")

if __name__ == "__main__":
    main()
