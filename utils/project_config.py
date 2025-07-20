"""
Enterprise Project Configuration

Centralized configuration for the Medical Image Training Platform.
Manages paths, imports, and dependencies across the reorganized structure.
"""

import os
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add source directories to Python path
SRC_DIR = PROJECT_ROOT / "src"
CORE_DIR = SRC_DIR / "core"
MEDICAL_DIR = SRC_DIR / "medical"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CUDA_DIR = PROJECT_ROOT / "cuda"
INFRASTRUCTURE_DIR = PROJECT_ROOT / "infrastructure"
DOCS_DIR = PROJECT_ROOT / "docs"

# Add to Python path for imports
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(MEDICAL_DIR))

# Project metadata
PROJECT_NAME = "Medical Image Training Platform"
VERSION = "2.0.0"
AUTHOR = "Medical AI Platform Team"

# Directory structure validation
def validate_project_structure():
    """Validate that all required directories exist."""
    required_dirs = [
        SRC_DIR, CORE_DIR, MEDICAL_DIR, TESTS_DIR, 
        SCRIPTS_DIR, CUDA_DIR, INFRASTRUCTURE_DIR, DOCS_DIR
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        raise RuntimeError(f"Missing required directories: {missing_dirs}")
    
    return True

# Import aliases for backward compatibility
def setup_imports():
    """Setup import aliases for the reorganized structure."""
    try:
        # Core processing imports
        from src.core.grayscale_wrapper import (
            EnterpriseGrayscaleProcessor,
            EnterpriseBatchProcessor,
            GrayscaleProcessorFactory,
            grayscale_normalize_tensor
        )
        
        # Medical pipeline imports  
        from src.medical.medical_data_pipeline import (
            MedicalImageValidator,
            MedicalDataUploader
        )
        
        return True
    except ImportError as e:
        print(f"Warning: Could not import modules after reorganization: {e}")
        return False

if __name__ == "__main__":
    print(f"🏥 {PROJECT_NAME} v{VERSION}")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"🐍 Python Path: {sys.path[:3]}")
    
    try:
        validate_project_structure()
        print("✅ Project structure validation passed")
        
        if setup_imports():
            print("✅ Import setup completed successfully")
        else:
            print("⚠️  Some imports failed - check file locations")
            
    except Exception as e:
        print(f"❌ Project setup failed: {e}")
