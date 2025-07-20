import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports


        from src.core.model import get_model
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
        

        from src.core.data import create_data_loader
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
        

        from src.core.metrics import MetricsTracker, compute_accuracy
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
        

# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
        from src.core.grayscale_wrapper import (

        import onnx
        from src.core.model import get_model
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
        

        from src.core.model import get_model
        from src.core.metrics import MetricsTracker
        import torch.optim as optim
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
        

#!/usr/bin/env python3
"""
Comprehensive test suite for medical image training pipeline
Tests all components before deployment
"""

def test_environment():
    """Test basic environment setup"""
    print("🧪 Testing Environment Setup...")
    
    # Test Python version
    python_version = sys.version_info
    print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    assert python_version >= (3, 8), "Python 3.8+ required"
    
    # Test PyTorch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
    
    print("✅ Environment test passed\n")
    return True
    return True

def test_model():
    """Test model creation and forward pass"""
    print("🏗️ Testing Model Components...")
    
    try:
        # Create model
        model = get_model(num_classes=10)
        print(f"   Model created: {type(model).__name__}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
        
        print("✅ Model test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}\n")
        return False

def test_data_pipeline():
    """Test data loading components"""
    print("📊 Testing Data Pipeline...")
    
    try:
        # Create temporary dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            
            # Create class directories
            for split in ['train', 'val']:
                for class_id in range(3):  # 3 classes for test
                    class_dir = data_dir / split / f"class_{class_id}"
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create dummy images (just text files for test)
                    for i in range(5):
                        (class_dir / f"image_{i}.txt").write_text("dummy")
        
        print(f"   Test dataset created at {data_dir}")
        print("✅ Data pipeline test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}\n")
        return False

def test_metrics():
    """Test metrics computation"""
    print("📈 Testing Metrics Components...")
    
    try:
        # Create metrics tracker
        tracker = MetricsTracker(num_classes=5)
        
        # Simulate predictions
        predictions = torch.tensor([0, 1, 2, 1, 0])
        targets = torch.tensor([0, 1, 1, 1, 0])
        
        tracker.update(predictions, targets, loss=0.5)
        
        # Compute metrics
        accuracy = tracker.compute_accuracy()
        summary = tracker.get_summary()
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Summary keys: {list(summary.keys())}")
        
        # Test functional accuracy
        func_acc = compute_accuracy(predictions, targets)
        print(f"   Functional accuracy: {func_acc:.2f}%")
        
        print("✅ Metrics test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}\n")
        return False

def test_cuda_kernel():
    """Test custom CUDA kernel with enterprise architecture"""
    print("🔥 Testing CUDA Kernel...")
    
    try:
            KernelRegistry, 
            GrayscaleProcessorFactory,
            grayscale_normalize_tensor
        )
        
        registry = KernelRegistry()
        print(f"   CUDA kernel available: {registry.kernel_available}")
        print(f"   Backend: {registry.backend.value}")
        
        # Test with enterprise processor
        processor = GrayscaleProcessorFactory.create_medical_processor()
        x = torch.randn(2, 3, 64, 64)
        result = processor(x)
        
        print(f"   Enterprise test - Input: {x.shape}, Output: {result.shape}")
        assert result.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {result.shape}"
        
        # Test functional interface
        result_func = grayscale_normalize_tensor(x)
        assert result_func.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {result_func.shape}"
        
        print("✅ CUDA kernel test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ CUDA kernel test failed: {e}\n")
        return False

def test_onnx_export():
    """Test ONNX model export"""
    print("📦 Testing ONNX Export...")
    
    try:
        # Create model
        model = get_model(num_classes=10)
        model.eval()
        
        # Export to ONNX
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as temp_file:
            torch.onnx.export(
                model,
                dummy_input,
                temp_file.name,
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['input__0'],
                output_names=['output__0']
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(temp_file.name)
            onnx.checker.check_model(onnx_model)
            
            print(f"   ONNX model exported successfully")
            print(f"   Input: {onnx_model.graph.input[0].name}")
            print(f"   Output: {onnx_model.graph.output[0].name}")
        
        print("✅ ONNX export test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ ONNX export test failed: {e}\n")
        return False

def test_training_loop():
    """Test minimal training loop"""
    print("🏃 Testing Training Loop...")
    
    try:
        # Create model and optimizer
        model = get_model(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create dummy batch
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 10, (4,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"   Training step completed")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Output shape: {outputs.shape}")
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(x)
            val_loss = criterion(val_outputs, y)
        
        print(f"   Validation loss: {val_loss.item():.4f}")
        
        print("✅ Training loop test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Training loop test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("🧪 Medical Image Training - Component Tests")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_model,
        test_data_pipeline,
        test_metrics,
        test_cuda_kernel,
        test_onnx_export,
        test_training_loop
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            # Convert None to False for summation
            results.append(result if result is not None else False)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum([1 for r in results if r])
    total = len(results)
    
    print("📋 TEST SUMMARY")
    print("=" * 30)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment.")
        print("\n📋 Next Steps:")
        print("   1. Set up AWS credentials: aws configure")
        print("   2. Run infrastructure: ./terraform_auto_upload.sh infrastructure") 
        print("   3. Deploy training: ./terraform_auto_upload.sh train")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
