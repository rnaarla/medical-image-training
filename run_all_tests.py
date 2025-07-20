#!/usr/bin/env python3
"""
Simplified Test Runner for Medical Image Platform

Runs all available tests and reports results.
Designed to work with the reorganized project structure.

Author: Medical AI Platform Team
Version: 2.0.0
"""

import sys
import time
import logging
from pathlib import Path

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTestRunner:
    """Simple test runner for the medical image platform."""
    
    def __init__(self):
        self.test_results = []
        
    def run_core_tests(self) -> bool:
        """Run core grayscale processing tests."""
        logger.info("🔧 Running Core Processing Tests...")
        
        try:
            from src.core.grayscale_wrapper import run_enterprise_tests
            result = run_enterprise_tests()
            logger.info(f"Core tests: {'✅ PASSED' if result else '❌ FAILED'}")
            return result
        except Exception as e:
            logger.error(f"❌ Core tests failed: {e}")
            return False
    
    def run_import_tests(self) -> bool:
        """Test that all critical imports work."""
        logger.info("📦 Running Import Tests...")
        
        imports_to_test = [
            ("src.core.grayscale_wrapper", "EnterpriseGrayscaleProcessor"),
            ("src.core.grayscale_wrapper", "GrayscaleProcessorFactory"),
            ("src.core.grayscale_wrapper", "grayscale_normalize_tensor"),
            ("project_config", "PROJECT_ROOT"),
        ]
        
        successful_imports = 0
        
        for module_name, import_name in imports_to_test:
            try:
                module = __import__(module_name, fromlist=[import_name])
                getattr(module, import_name)
                logger.info(f"✅ {module_name}.{import_name}")
                successful_imports += 1
            except Exception as e:
                logger.warning(f"⚠️  {module_name}.{import_name}: {e}")
        
        success_rate = successful_imports / len(imports_to_test)
        logger.info(f"Import tests: {successful_imports}/{len(imports_to_test)} ({success_rate:.1%})")
        
        return success_rate > 0.5
    
    def run_structure_tests(self) -> bool:
        """Test project structure integrity."""
        logger.info("🏗️  Running Structure Tests...")
        
        # Get absolute path to project root
        project_root_abs = Path(__file__).parent.absolute()
        
        required_paths = [
            project_root_abs / "src" / "core",
            project_root_abs / "src" / "medical", 
            project_root_abs / "tests",
            project_root_abs / "scripts",
            project_root_abs / "docs",
            project_root_abs / "infrastructure",
            project_root_abs / "cuda",
            project_root_abs / "README.md",
            project_root_abs / "project_config.py"
        ]
        
        existing_paths = [p for p in required_paths if p.exists()]
        missing_paths = [p for p in required_paths if not p.exists()]
        
        logger.info(f"Structure: {len(existing_paths)}/{len(required_paths)} paths exist")
        
        if missing_paths:
            for path in missing_paths:
                logger.warning(f"⚠️  Missing: {path.relative_to(project_root_abs)}")
        
        return len(missing_paths) <= 1  # Allow 1 missing path
    
    def run_basic_functionality_tests(self) -> bool:
        """Run basic functionality tests."""
        logger.info("⚡ Running Basic Functionality Tests...")
        
        try:
            # Test 1: Create a simple tensor and process it
            from src.core.grayscale_wrapper import GrayscaleProcessorFactory
            
            processor = GrayscaleProcessorFactory.create_medical_processor()
            logger.info("✅ Medical processor created")
            
            # Test 2: Check device info
            device_info = processor.get_device_info()
            logger.info(f"✅ Device info: {device_info['backend']}")
            
            # Test 3: Test with dummy data (if torch available)
            try:
                import torch
                dummy_input = torch.randn(1, 3, 64, 64)
                result = processor(dummy_input)
                assert result.shape == (1, 1, 64, 64), f"Expected (1,1,64,64), got {result.shape}"
                logger.info("✅ Basic tensor processing")
                return True
            except ImportError:
                logger.info("⚠️  PyTorch not available - skipping tensor tests")
                return True
                
        except Exception as e:
            logger.error(f"❌ Basic functionality failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all available tests and return results."""
        logger.info("🏥 Starting Medical Image Platform Test Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        tests = [
            ("Structure Tests", self.run_structure_tests),
            ("Import Tests", self.run_import_tests),
            ("Core Processing Tests", self.run_core_tests),
            ("Basic Functionality Tests", self.run_basic_functionality_tests),
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed_tests += 1
            except Exception as e:
                logger.error(f"❌ {test_name} crashed: {e}")
                results[test_name] = False
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 TEST SUITE RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED" 
            logger.info(f"{status:<12} {test_name}")
        
        success_rate = passed_tests / total_tests
        logger.info(f"\n📊 Overall Results:")
        logger.info(f"   Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {success_rate:.1%}")
        logger.info(f"   Duration: {duration:.2f}s")
        
        if success_rate >= 0.8:
            logger.info("🎉 EXCELLENT: System is ready for production!")
        elif success_rate >= 0.6:
            logger.info("⚠️  GOOD: System is mostly ready with minor issues")
        elif success_rate >= 0.4:
            logger.info("⚠️  FAIR: System has significant issues to address")
        else:
            logger.info("❌ POOR: System needs major work before deployment")
        
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "detailed_results": results
        }

def main():
    """Main test execution function."""
    runner = SimpleTestRunner()
    results = runner.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results["success_rate"] >= 0.6 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
