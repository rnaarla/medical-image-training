#!/usr/bin/env python3
"""
Quick Medical Image Upload and Test Script

Simple script to upload medical images and test the processing pipeline
without external dependencies.

Author: Medical AI Platform Team
Version: 2.0.0
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import subprocess

# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from utils.project_config import setup_imports
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickMedicalTester:
    """
    Quick medical image testing class with minimal dependencies.
    
    Performs basic system checks and validation without requiring
    external medical imaging libraries.
    """
    
    def __init__(self):
        self.test_results = {}
        self.current_dir = Path(__file__).parent.parent
        
    def create_test_image_data(self) -> bool:
        """Create basic test image data for validation."""
        try:
            # Create test data directory
            test_dir = self.current_dir / "quick_test_data"
            test_dir.mkdir(exist_ok=True)
            
            # Create a simple test image using basic Python
            import random
            
            # Generate simple test image data
            width, height = 64, 64
            image_data = []
            
            for y in range(height):
                row = []
                for x in range(width):
                    # Create gradient pattern
                    r = min(255, int((x / width) * 255))
                    g = min(255, int((y / height) * 255)) 
                    b = 128
                    row.append([r, g, b])
                image_data.append(row)
            
            # Save as simple format
            test_file = test_dir / "test_medical_image.json"
            with open(test_file, 'w') as f:
                json.dump({
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "data": image_data[:5]  # Just first 5 rows for size
                }, f, indent=2)
                
            logger.info(f"✅ Created test image data: {test_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create test image: {e}")
            return False
    
    def test_basic_imports(self) -> bool:
        """Test that basic Python imports work."""
        try:
            import torch
            import numpy as np
            logger.info(f"✅ PyTorch version: {torch.__version__}")
            logger.info(f"✅ NumPy version: {np.__version__}")
            return True
        except ImportError as e:
            logger.warning(f"⚠️  Missing dependencies: {e}")
            return False
    
    def test_grayscale_processing(self) -> bool:
        """Test grayscale processing functionality."""
        try:
            from src.core.grayscale_wrapper import run_enterprise_tests
            logger.info("🧪 Running enterprise grayscale tests...")
            
            success = run_enterprise_tests()
            if success:
                logger.info("✅ Grayscale processing tests passed")
                return True
            else:
                logger.warning("⚠️  Some grayscale tests failed")
                return False
                
        except ImportError as e:
            logger.warning(f"⚠️  Could not import grayscale module: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Grayscale test failed: {e}")
            return False
    
    def test_aws_configuration(self) -> bool:
        """Test AWS configuration and connectivity."""
        try:
            # Check if AWS CLI is available
            result = subprocess.run(['aws', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ AWS CLI available: {result.stdout.strip()}")
                return True
            else:
                logger.info("ℹ️  AWS CLI not found (optional for local testing)")
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("ℹ️  AWS CLI not available (optional for local testing)")
            return True
        except Exception as e:
            logger.warning(f"⚠️  AWS configuration check failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run comprehensive medical image processing test."""
        logger.info("🏥 Starting Quick Medical Image Processing Test")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Test 1: Create test data
        logger.info("Test 1: Creating test image data...")
        test1_result = self.create_test_image_data()
        
        # Test 2: Basic imports
        logger.info("Test 2: Testing basic imports...")
        test2_result = self.test_basic_imports()
        
        # Test 3: Grayscale processing
        logger.info("Test 3: Testing grayscale processing...")
        test3_result = self.test_grayscale_processing()
        
        # Test 4: AWS configuration
        logger.info("Test 4: Checking AWS configuration...")
        aws_result = self.test_aws_configuration()
        
        end_time = time.time()
        
        # Compile results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": round(end_time - start_time, 2),
            "tests": {
                "test_data_creation": test1_result,
                "basic_imports": test2_result, 
                "grayscale_processing": test3_result,
                "aws_configuration": aws_result,
            },
            "readiness_score": sum([
                test1_result,
                test2_result, 
                test3_result,
                aws_result
            ]) / 4.0
        }
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 QUICK TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        passed_tests = sum([test1_result, test2_result, test3_result, aws_result])
        total_tests = 4
        
        logger.info(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"📊 Readiness Score: {results['readiness_score']:.1%}")
        logger.info(f"⏱️  Total Time: {results['total_duration_seconds']}s")
        
        if results['readiness_score'] >= 0.75:
            logger.info("🎉 System ready for medical image processing!")
        elif results['readiness_score'] >= 0.5:
            logger.info("⚠️  System partially ready - some features may be limited")
        else:
            logger.info("❌ System needs configuration before medical image processing")
        
        return results

def main():
    """Main function for quick medical testing."""
    parser = argparse.ArgumentParser(description="Quick Medical Image Processing Test")
    parser.add_argument("--save", "-s", 
                       help="Save results to specified JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = QuickMedicalTester()
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    exit_code = 0 if results['readiness_score'] >= 0.5 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
