"""
Core Processing Modules

Contains the enterprise-grade processing engines including:
- CUDA-accelerated grayscale conversion
- Medical image data utilities  
- Performance metrics and monitoring
- Configuration management
"""

from .grayscale_wrapper import (
    EnterpriseGrayscaleProcessor,
    EnterpriseBatchProcessor,
    GrayscaleProcessorFactory,
    grayscale_normalize_tensor,
    EnterpriseBenchmark,
    run_enterprise_tests
)

try:
    from .model import get_model
    from .data import create_data_loader
    from .metrics import MetricsTracker, compute_accuracy
except ImportError:
    # Optional modules - may not be available in all environments
    pass

__all__ = [
    'EnterpriseGrayscaleProcessor',
    'EnterpriseBatchProcessor', 
    'GrayscaleProcessorFactory',
    'grayscale_normalize_tensor',
    'EnterpriseBenchmark',
    'run_enterprise_tests'
]

from .grayscale_wrapper import (
    EnterpriseGrayscaleProcessor,
    EnterpriseBatchProcessor,
    GrayscaleProcessorFactory,
    grayscale_normalize_tensor,
    run_enterprise_tests
)

__all__ = [
    'EnterpriseGrayscaleProcessor',
    'EnterpriseBatchProcessor', 
    'GrayscaleProcessorFactory',
    'grayscale_normalize_tensor',
    'run_enterprise_tests'
]
