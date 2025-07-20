#!/usr/bin/env python3
"""
Enterprise Configuration Management System

FAANG-standard configuration management for production ML systems.
Supports environment-specific configurations, secrets management,
and comprehensive validation.

Author: Medical AI Platform Team
Version: 2.0.0
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Logging level configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class GPUConfig:
    """GPU configuration settings."""
    enabled: bool = True
    device_ids: List[int] = field(default_factory=list)
    memory_fraction: float = 0.95
    allow_growth: bool = True
    mixed_precision: bool = True

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "resnet50"
    num_classes: int = 10
    pretrained: bool = True
    dropout_rate: float = 0.2
    batch_norm: bool = True

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

@dataclass
class DataConfig:
    """Data pipeline configuration."""
    data_dir: str = "/data/medical_images"
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    image_size: int = 224
    use_dali: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "medical_training.log"
    max_file_size_mb: int = 100
    backup_count: int = 5

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    wandb_enabled: bool = True
    wandb_project: str = "medical-image-classification"
    tensorboard_enabled: bool = True
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    health_check_port: int = 8080

@dataclass
class SecurityConfig:
    """Security and compliance configuration."""
    encrypt_data_at_rest: bool = True
    encrypt_data_in_transit: bool = True
    audit_logging: bool = True
    secrets_backend: str = "aws_secrets_manager"
    data_retention_days: int = 365

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    replicas: int = 1
    max_replicas: int = 10
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "8Gi"
    memory_limit: str = "16Gi"
    gpu_request: int = 1
    node_selector: Dict[str, str] = field(default_factory=lambda: {"accelerator": "nvidia-tesla-v100"})

@dataclass
class EnterpriseConfig:
    """Master enterprise configuration."""
    environment: Environment = Environment.DEVELOPMENT
    project_name: str = "medical-image-training"
    version: str = "2.0.0"
    
    # Component configurations
    gpu: GPUConfig = field(default_factory=GPUConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Custom configurations
    custom: Dict[str, Any] = field(default_factory=dict)

class ConfigurationManager:
    """
    Enterprise configuration manager with validation and environment support.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[EnterpriseConfig] = None
    
    def load_config(self, 
                   environment: Environment = Environment.DEVELOPMENT,
                   config_file: Optional[str] = None) -> EnterpriseConfig:
        """
        Load configuration for specified environment.
        
        Args:
            environment: Target deployment environment
            config_file: Optional specific config file path
            
        Returns:
            Loaded and validated configuration
        """
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self.config_dir / f"{environment.value}.yaml"
        
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            config_dict = self._load_config_file(config_path)
            self._config = self._dict_to_config(config_dict)
        else:
            logger.info(f"No config file found at {config_path}, using defaults")
            self._config = EnterpriseConfig()
        
        self._config.environment = environment
        self._validate_config()
        self._apply_environment_overrides()
        
        logger.info(f"Configuration loaded for {environment.value} environment")
        return self._config
    
    def save_config(self, config: EnterpriseConfig, filename: Optional[str] = None):
        """Save configuration to file."""
        if not filename:
            filename = f"{config.environment.value}.yaml"
        
        config_path = self.config_dir / filename
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> EnterpriseConfig:
        """Convert dictionary to EnterpriseConfig object."""
        # This would typically use a more sophisticated serialization library
        # For now, using a simple approach
        config = EnterpriseConfig()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _config_to_dict(self, config: EnterpriseConfig) -> Dict[str, Any]:
        """Convert EnterpriseConfig to dictionary."""
        # Simple serialization - in production would use proper serialization
        return {
            'environment': config.environment.value,
            'project_name': config.project_name,
            'version': config.version,
            'gpu': {
                'enabled': config.gpu.enabled,
                'device_ids': config.gpu.device_ids,
                'memory_fraction': config.gpu.memory_fraction,
                'allow_growth': config.gpu.allow_growth,
                'mixed_precision': config.gpu.mixed_precision,
            },
            'model': {
                'architecture': config.model.architecture,
                'num_classes': config.model.num_classes,
                'pretrained': config.model.pretrained,
                'dropout_rate': config.model.dropout_rate,
                'batch_norm': config.model.batch_norm,
            },
            'training': {
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'epochs': config.training.epochs,
                'optimizer': config.training.optimizer,
                'scheduler': config.training.scheduler,
                'weight_decay': config.training.weight_decay,
                'gradient_clip_norm': config.training.gradient_clip_norm,
            },
            'custom': config.custom,
        }
    
    def _validate_config(self):
        """Validate configuration settings."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        
        # Validate GPU settings
        if self._config.gpu.memory_fraction <= 0 or self._config.gpu.memory_fraction > 1:
            raise ValueError("GPU memory fraction must be between 0 and 1")
        
        # Validate training settings
        if self._config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self._config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate data settings
        if self._config.data.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")
        
        logger.info("✅ Configuration validation passed")
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # GPU configuration
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            device_ids = [int(x) for x in os.getenv('CUDA_VISIBLE_DEVICES').split(',')]
            self._config.gpu.device_ids = device_ids
        
        # Training configuration
        if os.getenv('BATCH_SIZE'):
            self._config.training.batch_size = int(os.getenv('BATCH_SIZE'))
        
        if os.getenv('LEARNING_RATE'):
            self._config.training.learning_rate = float(os.getenv('LEARNING_RATE'))
        
        # Data configuration
        if os.getenv('DATA_DIR'):
            self._config.data.data_dir = os.getenv('DATA_DIR')
        
        # Monitoring configuration
        if os.getenv('WANDB_PROJECT'):
            self._config.monitoring.wandb_project = os.getenv('WANDB_PROJECT')
        
        logger.info("Environment variable overrides applied")
    
    @property
    def config(self) -> EnterpriseConfig:
        """Get current configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self._config

# Global configuration instance
_config_manager = ConfigurationManager()

def get_config() -> EnterpriseConfig:
    """Get the current global configuration."""
    return _config_manager.config

def load_config(environment: Environment = Environment.DEVELOPMENT) -> EnterpriseConfig:
    """Load configuration for the specified environment."""
    return _config_manager.load_config(environment)

def create_sample_configs():
    """Create sample configuration files for all environments."""
    manager = ConfigurationManager()
    
    for env in Environment:
        config = EnterpriseConfig()
        config.environment = env
        
        # Environment-specific overrides
        if env == Environment.PRODUCTION:
            config.training.batch_size = 64
            config.gpu.memory_fraction = 0.9
            config.security.encrypt_data_at_rest = True
            config.monitoring.wandb_enabled = True
            config.deployment.replicas = 3
        elif env == Environment.STAGING:
            config.training.batch_size = 32
            config.deployment.replicas = 2
        else:  # DEVELOPMENT/TESTING
            config.training.batch_size = 16
            config.monitoring.wandb_enabled = False
            config.deployment.replicas = 1
        
        manager.save_config(config)

if __name__ == '__main__':
    # Create sample configuration files
    create_sample_configs()
    print("✅ Sample configuration files created")
    
    # Test configuration loading
    config = load_config(Environment.DEVELOPMENT)
    print(f"✅ Configuration loaded: {config.project_name} v{config.version}")
