#!/usr/bin/env python3
"""
CUDA-Accelerated Grayscale Conversion Module

Enterprise-grade PyTorch extension for high-performance medical image preprocessing.
Implements custom CUDA kernels with graceful CPU fallbacks for production MLOps pipelines.

Architecture:
- Custom CUDA kernels for GPU acceleration
- Automatic fallback to PyTorch operations
- Memory-efficient tensor operations
- Comprehensive error handling and logging

Author: Medical AI Platform Team
Version: 2.0.0
License: Proprietary
"""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KernelBackend(Enum):
    """Enumeration of available computation backends."""
    CUDA_CUSTOM = "cuda_custom"
    CUDA_PYTORCH = "cuda_pytorch" 
    CPU = "cpu"

@dataclass
class PerformanceMetrics:
    """Performance tracking for kernel operations."""
    backend: KernelBackend
    execution_time_ms: float
    throughput_gb_s: float
    memory_usage_mb: float
    batch_size: int
    tensor_shape: Tuple[int, ...]

class KernelRegistry:
    """Singleton registry for managing CUDA kernel availability."""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._cuda_available = False
            self._kernel_available = False
            self._backend = KernelBackend.CPU
            self._initialize_kernels()
            KernelRegistry._initialized = True
    
    def _initialize_kernels(self) -> None:
        """Initialize CUDA kernels with comprehensive error handling."""
        try:
            import grayscale_ops
            self._grayscale_ops = grayscale_ops
            self._kernel_available = True
            self._cuda_available = torch.cuda.is_available()
            
            if self._cuda_available:
                self._backend = KernelBackend.CUDA_CUSTOM
                logger.info("✅ Custom CUDA kernels initialized successfully")
            else:
                self._backend = KernelBackend.CPU
                logger.warning("⚠️  CUDA unavailable, custom kernels loaded for CPU fallback")
                
        except ImportError as e:
            logger.warning(f"⚠️  Custom CUDA kernels unavailable: {e}")
            self._kernel_available = False
            self._cuda_available = torch.cuda.is_available()
            self._backend = KernelBackend.CUDA_PYTORCH if self._cuda_available else KernelBackend.CPU
            
        except Exception as e:
            logger.error(f"❌ Kernel initialization failed: {e}")
            self._kernel_available = False
            self._cuda_available = False
            self._backend = KernelBackend.CPU
    
    @property
    def backend(self) -> KernelBackend:
        return self._backend
    
    @property
    def cuda_available(self) -> bool:
        return self._cuda_available
    
    @property
    def kernel_available(self) -> bool:
        return self._kernel_available
    
    @property
    def grayscale_ops(self):
        if not self._kernel_available:
            raise RuntimeError("Custom CUDA kernels not available")
        return self._grayscale_ops

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        execution_time = (end_time - start_time) * 1000  # ms
        memory_delta = (end_memory - start_memory) / (1024 ** 2)  # MB
        
        logger.debug(f"{operation_name}: {execution_time:.3f}ms, Memory: {memory_delta:+.2f}MB")

class BaseGrayscaleProcessor(nn.Module, ABC):
    """Abstract base class for grayscale processing operations."""
    
    def __init__(self, 
                 mean: Union[float, Tuple[float, ...]] = 0.5,
                 std: Union[float, Tuple[float, ...]] = 0.5,
                 device_strategy: str = "auto"):
        super().__init__()
        self.device_strategy = device_strategy
        self.registry = KernelRegistry()
        self._setup_normalization(mean, std)
        self._initialize_backend()
    
    @abstractmethod
    def _setup_normalization(self, mean, std):
        """Setup normalization parameters."""
        pass
    
    @abstractmethod
    def _initialize_backend(self):
        """Initialize processing backend."""
        pass
    
    @abstractmethod
    def _process_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Core tensor processing logic."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with comprehensive error handling and monitoring."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {x.shape}")
        
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {x.size(1)} channels")
        
        with performance_monitor(f"{self.__class__.__name__}.forward"):
            return self._process_tensor(x)
    
    def get_device_info(self) -> Dict[str, Union[str, bool, int]]:
        """Get current device configuration information."""
        return {
            "backend": self.registry.backend.value,
            "cuda_available": self.registry.cuda_available,
            "kernel_available": self.registry.kernel_available,
            "current_device": str(next(self.parameters()).device) if list(self.parameters()) else "cpu",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

class EnterpriseGrayscaleProcessor(BaseGrayscaleProcessor):
    """
    Enterprise-grade grayscale conversion with adaptive backend selection.
    
    Features:
    - Automatic backend selection (CUDA custom > CUDA PyTorch > CPU)
    - Comprehensive error handling and logging
    - Performance monitoring and metrics
    - Memory-efficient operations
    """
    
    def _setup_normalization(self, mean: float, std: float):
        """Setup single-channel normalization parameters."""
        self.mean = mean
        self.std = std
        
        # RGB to grayscale conversion weights (ITU-R BT.709 standard)
        self.register_buffer('rgb_weights', torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1))
    
    def _initialize_backend(self):
        """Initialize processing backend based on availability."""
        self.backend = self.registry.backend
        logger.info(f"Initialized {self.__class__.__name__} with backend: {self.backend.value}")
    
    def _process_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Process tensor using appropriate backend."""
        if self.backend == KernelBackend.CUDA_CUSTOM:
            return self._cuda_custom_forward(x)
        elif self.backend == KernelBackend.CUDA_PYTORCH:
            return self._cuda_pytorch_forward(x)
        else:
            return self._cpu_forward(x)
    
    def _cuda_custom_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom CUDA kernel."""
        x = self._prepare_tensor_for_cuda(x)
        
        try:
            ops = self.registry.grayscale_ops
            return ops.grayscale_normalize(x, self.mean, self.std)
        except Exception as e:
            logger.warning(f"Custom CUDA kernel failed: {e}, falling back to PyTorch")
            return self._cuda_pytorch_forward(x)
    
    def _cuda_pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using PyTorch CUDA operations."""
        if not x.is_cuda:
            x = x.cuda()
        
        return self._apply_grayscale_conversion(x)
    
    def _cpu_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using CPU operations."""
        if x.is_cuda:
            x = x.cpu()
        
        return self._apply_grayscale_conversion(x)
    
    def _prepare_tensor_for_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare tensor for CUDA operations."""
        if x.dtype != torch.float32:
            x = x.float()
        
        if not x.is_cuda:
            x = x.cuda()
        
        if not x.is_contiguous():
            x = x.contiguous()
        
        return x
    
    def _apply_grayscale_conversion(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grayscale conversion using PyTorch operations."""
        # Convert to grayscale using weighted sum
        weights = self.rgb_weights.to(x.device, x.dtype)
        gray = torch.sum(x * weights, dim=1, keepdim=True)
        
        # Apply normalization
        normalized = (gray - self.mean) / self.std
        
        return normalized

class EnterpriseBatchProcessor(BaseGrayscaleProcessor):
    """
    Enterprise batch grayscale processor with per-channel normalization.
    
    Optimized for high-throughput medical image processing pipelines.
    Supports ImageNet normalization statistics and custom configurations.
    """
    
    def _setup_normalization(self, 
                           mean: Tuple[float, float, float], 
                           std: Tuple[float, float, float]):
        """Setup per-channel normalization parameters."""
        self.register_buffer('channel_mean', torch.tensor(mean).view(3))
        self.register_buffer('channel_std', torch.tensor(std).view(3))
        
        # RGB to grayscale conversion weights (ITU-R BT.709 standard)
        self.register_buffer('rgb_weights', torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1))
    
    def _initialize_backend(self):
        """Initialize processing backend based on availability."""
        self.backend = self.registry.backend
        logger.info(f"Initialized {self.__class__.__name__} with backend: {self.backend.value}")
    
    def _process_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Process tensor using appropriate backend."""
        if self.backend == KernelBackend.CUDA_CUSTOM:
            return self._cuda_custom_forward(x)
        else:
            return self._fallback_forward(x)
    
    def _cuda_custom_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom CUDA kernel."""
        x = self._prepare_tensor_for_cuda(x)
        
        try:
            ops = self.registry.grayscale_ops
            return ops.batch_grayscale_normalize(x, self.channel_mean, self.channel_std)
        except Exception as e:
            logger.warning(f"Custom CUDA batch kernel failed: {e}, falling back")
            return self._fallback_forward(x)
    
    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback implementation using PyTorch operations."""
        # Normalize per channel first
        mean = self.channel_mean.view(1, 3, 1, 1).to(x.device, x.dtype)
        std = self.channel_std.view(1, 3, 1, 1).to(x.device, x.dtype)
        normalized = (x - mean) / std
        
        # Convert to grayscale
        weights = self.rgb_weights.to(x.device, x.dtype)
        gray = torch.sum(normalized * weights, dim=1, keepdim=True)
        
        return gray
    
    def _prepare_tensor_for_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare tensor for CUDA operations."""
        if x.dtype != torch.float32:
            x = x.float()
        
        if not x.is_cuda:
            x = x.cuda()
        
        if not x.is_contiguous():
            x = x.contiguous()
        
        return x

class GrayscaleProcessorFactory:
    """Factory for creating grayscale processors with enterprise configurations."""
    
    @staticmethod
    def create_medical_processor(device_strategy: str = "auto") -> EnterpriseGrayscaleProcessor:
        """Create processor optimized for medical imaging (0.5 mean/std)."""
        return EnterpriseGrayscaleProcessor(mean=0.5, std=0.5, device_strategy=device_strategy)
    
    @staticmethod
    def create_imagenet_processor(device_strategy: str = "auto") -> EnterpriseBatchProcessor:
        """Create processor with ImageNet normalization."""
        return EnterpriseBatchProcessor(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            device_strategy=device_strategy
        )
    
    @staticmethod
    def create_custom_processor(mean: Union[float, Tuple[float, ...]], 
                              std: Union[float, Tuple[float, ...]], 
                              device_strategy: str = "auto") -> BaseGrayscaleProcessor:
        """Create processor with custom normalization parameters."""
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
            return EnterpriseGrayscaleProcessor(mean=mean, std=std, device_strategy=device_strategy)
        else:
            return EnterpriseBatchProcessor(mean=mean, std=std, device_strategy=device_strategy)

def grayscale_normalize_tensor(x: torch.Tensor, 
                              mean: float = 0.5, 
                              std: float = 0.5) -> torch.Tensor:
    """
    Functional interface for grayscale conversion with enterprise error handling.
    
    Args:
        x: Input RGB tensor [B, 3, H, W]
        mean: Normalization mean
        std: Normalization standard deviation
    
    Returns:
        Grayscale tensor [B, 1, H, W]
        
    Raises:
        TypeError: If input is not a tensor
        ValueError: If tensor dimensions are invalid
        RuntimeError: If processing fails
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError(f"Expected [B,3,H,W] tensor, got shape {x.shape}")
    
    registry = KernelRegistry()
    
    try:
        if registry.backend == KernelBackend.CUDA_CUSTOM and x.is_cuda:
            # Prepare tensor for CUDA operations
            if x.dtype != torch.float32:
                x = x.float()
            if not x.is_contiguous():
                x = x.contiguous()
            
            ops = registry.grayscale_ops
            return ops.grayscale_normalize(x, mean, std)
        else:
            # Fallback implementation
            rgb_weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1).to(x.device, x.dtype)
            gray = torch.sum(x * rgb_weights, dim=1, keepdim=True)
            return (gray - mean) / std
            
    except Exception as e:
        logger.error(f"Grayscale conversion failed: {e}")
        raise RuntimeError(f"Processing failed: {e}") from e

class EnterpriseBenchmark:
    """
    Enterprise-grade benchmarking suite for kernel performance analysis.
    
    Provides comprehensive performance metrics, memory analysis, and 
    comparative benchmarking across different backends.
    """
    
    def __init__(self):
        self.registry = KernelRegistry()
        self.results: List[PerformanceMetrics] = []
    
    def benchmark_kernels(self, 
                         batch_size: int = 32, 
                         height: int = 224, 
                         width: int = 224, 
                         num_runs: int = 100,
                         warmup_runs: int = 10) -> Dict[str, PerformanceMetrics]:
        """
        Comprehensive kernel benchmarking with statistical analysis.
        
        Args:
            batch_size: Batch size for testing
            height: Image height
            width: Image width  
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary of performance metrics by backend
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU benchmarks")
            return {}
        
        device = torch.device('cuda')
        tensor_shape = (batch_size, 3, height, width)
        
        # Create test data
        x = torch.randn(*tensor_shape, device=device, dtype=torch.float32)
        tensor_size_mb = x.numel() * x.element_size() / (1024 ** 2)
        
        logger.info(f"🔬 Starting benchmark: {tensor_shape}, {tensor_size_mb:.2f}MB, {num_runs} runs")
        
        results = {}
        
        # Benchmark custom CUDA kernel
        if self.registry.backend == KernelBackend.CUDA_CUSTOM:
            results['cuda_custom'] = self._benchmark_cuda_custom(x, num_runs, warmup_runs)
        
        # Benchmark PyTorch implementation
        results['pytorch'] = self._benchmark_pytorch(x, num_runs, warmup_runs)
        
        self._log_benchmark_results(results, tensor_size_mb)
        return results
    
    def _benchmark_cuda_custom(self, x: torch.Tensor, num_runs: int, warmup_runs: int) -> PerformanceMetrics:
        """Benchmark custom CUDA kernel."""
        processor = GrayscaleProcessorFactory.create_medical_processor()
        
        # Warmup
        for _ in range(warmup_runs):
            _ = processor(x)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_memory = torch.cuda.memory_allocated()
        start_event.record()
        
        for _ in range(num_runs):
            _ = processor(x)
        
        end_event.record()
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        
        execution_time = start_event.elapsed_time(end_event) / num_runs
        memory_usage = (end_memory - start_memory) / (1024 ** 2)
        throughput = (x.numel() * x.element_size() * num_runs) / (execution_time / 1000) / (1024 ** 3)
        
        return PerformanceMetrics(
            backend=KernelBackend.CUDA_CUSTOM,
            execution_time_ms=execution_time,
            throughput_gb_s=throughput,
            memory_usage_mb=memory_usage,
            batch_size=x.size(0),
            tensor_shape=tuple(x.shape)
        )
    
    def _benchmark_pytorch(self, x: torch.Tensor, num_runs: int, warmup_runs: int) -> PerformanceMetrics:
        """Benchmark PyTorch implementation."""
        rgb_weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1).to(x.device, x.dtype)
        
        # Warmup
        for _ in range(warmup_runs):
            gray = torch.sum(x * rgb_weights, dim=1, keepdim=True)
            _ = (gray - 0.5) / 0.5
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_memory = torch.cuda.memory_allocated()
        start_event.record()
        
        for _ in range(num_runs):
            gray = torch.sum(x * rgb_weights, dim=1, keepdim=True)
            _ = (gray - 0.5) / 0.5
        
        end_event.record()
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        
        execution_time = start_event.elapsed_time(end_event) / num_runs
        memory_usage = (end_memory - start_memory) / (1024 ** 2)
        throughput = (x.numel() * x.element_size() * num_runs) / (execution_time / 1000) / (1024 ** 3)
        
        return PerformanceMetrics(
            backend=KernelBackend.CUDA_PYTORCH,
            execution_time_ms=execution_time,
            throughput_gb_s=throughput,
            memory_usage_mb=memory_usage,
            batch_size=x.size(0),
            tensor_shape=tuple(x.shape)
        )
    
    def _log_benchmark_results(self, results: Dict[str, PerformanceMetrics], tensor_size_mb: float):
        """Log comprehensive benchmark results."""
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("=" * 50)
        
        for name, metrics in results.items():
            logger.info(f"{name.upper()}:")
            logger.info(f"  ⏱️  Execution: {metrics.execution_time_ms:.3f} ms")
            logger.info(f"  🚀 Throughput: {metrics.throughput_gb_s:.2f} GB/s")
            logger.info(f"  💾 Memory: {metrics.memory_usage_mb:.2f} MB")
        
        if len(results) > 1:
            times = [m.execution_time_ms for m in results.values()]
            speedup = max(times) / min(times)
            logger.info(f"  ⚡ Max Speedup: {speedup:.2f}x")

def run_enterprise_tests():
    """Run comprehensive enterprise-grade tests."""
    logger.info("🧪 Starting Enterprise Test Suite")
    
    registry = KernelRegistry()
    logger.info(f"Backend: {registry.backend.value}")
    
    try:
        # Test 1: Basic functionality
        logger.info("Test 1: Basic Functionality")
        processor = GrayscaleProcessorFactory.create_medical_processor()
        x = torch.randn(2, 3, 64, 64)
        
        if torch.cuda.is_available():
            x = x.cuda()
        
        result = processor(x)
        assert result.shape == (2, 1, 64, 64), f"Expected shape (2,1,64,64), got {result.shape}"
        logger.info("✅ Basic functionality test passed")
        
        # Test 2: Batch processing
        logger.info("Test 2: Batch Processing")
        batch_processor = GrayscaleProcessorFactory.create_imagenet_processor()
        result_batch = batch_processor(x)
        assert result_batch.shape == (2, 1, 64, 64), f"Expected shape (2,1,64,64), got {result_batch.shape}"
        logger.info("✅ Batch processing test passed")
        
        # Test 3: Functional interface
        logger.info("Test 3: Functional Interface")
        result_func = grayscale_normalize_tensor(x)
        assert result_func.shape == (2, 1, 64, 64), f"Expected shape (2,1,64,64), got {result_func.shape}"
        logger.info("✅ Functional interface test passed")
        
        # Test 4: Error handling
        logger.info("Test 4: Error Handling")
        try:
            _ = grayscale_normalize_tensor("invalid_input")
            assert False, "Should have raised TypeError"
        except TypeError:
            logger.info("✅ Error handling test passed")
        
        # Test 5: Performance benchmark
        if torch.cuda.is_available():
            logger.info("Test 5: Performance Benchmark")
            benchmark = EnterpriseBenchmark()
            benchmark.benchmark_kernels(batch_size=16, height=128, width=128, num_runs=50)
            logger.info("✅ Performance benchmark completed")
        
        logger.info("🎉 All enterprise tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enterprise test failed: {e}")
        return False

# Legacy compatibility aliases (deprecated)
class GrayscaleNormalizeGPU(EnterpriseGrayscaleProcessor):
    """Deprecated: Use EnterpriseGrayscaleProcessor instead."""
    def __init__(self, mean: float = 0.5, std: float = 0.5, use_cuda_kernel: bool = True):
        logger.warning("GrayscaleNormalizeGPU is deprecated. Use GrayscaleProcessorFactory.create_medical_processor() instead.")
        super().__init__(mean=mean, std=std)

class BatchGrayscaleNormalizeGPU(EnterpriseBatchProcessor):
    """Deprecated: Use EnterpriseBatchProcessor instead."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), use_cuda_kernel: bool = True):
        logger.warning("BatchGrayscaleNormalizeGPU is deprecated. Use GrayscaleProcessorFactory.create_imagenet_processor() instead.")
        super().__init__(mean=mean, std=std)

if __name__ == '__main__':
    # Run enterprise test suite
    success = run_enterprise_tests()
