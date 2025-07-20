#!/usr/bin/env python3
"""
Enterprise Health Check and Monitoring System

Production-grade health monitoring, metrics collection, and observability
for distributed medical image training systems.

Author: Medical AI Platform Team
Version: 2.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import torch

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float

@dataclass
class HealthCheckResult:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: str
    response_time_ms: float

class EnterpriseHealthMonitor:
    """
    Enterprise-grade health monitoring system.
    
    Provides comprehensive system monitoring, health checks,
    and alerting capabilities for production ML systems.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.health_checks: Dict[str, callable] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.health_checks.update({
            'system_resources': self._check_system_resources,
            'gpu_availability': self._check_gpu_availability,
            'disk_space': self._check_disk_space,
            'memory_usage': self._check_memory_usage,
            'pytorch_functionality': self._check_pytorch_functionality,
        })
    
    async def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for name, check_func in self.health_checks.items():
            start_time = time.perf_counter()
            
            try:
                result = await self._run_single_check(name, check_func)
                result.response_time_ms = (time.perf_counter() - start_time) * 1000
                results.append(result)
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results.append(HealthCheckResult(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    details={'error': str(e), 'type': type(e).__name__},
                    timestamp=datetime.utcnow().isoformat(),
                    response_time_ms=(time.perf_counter() - start_time) * 1000
                ))
        
        return results
    
    async def _run_single_check(self, name: str, check_func: callable) -> HealthCheckResult:
        """Run a single health check with timeout."""
        try:
            # Run with timeout to prevent hanging
            result = await asyncio.wait_for(
                asyncio.create_task(check_func()), 
                timeout=30.0
            )
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=name,
                status=HealthStatus.CRITICAL,
                message="Health check timed out",
                details={'timeout_seconds': 30},
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=30000
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        message = "System resources normal"
        
        if cpu_percent > 90 or memory.percent > 90:
            status = HealthStatus.CRITICAL
            message = "System resources critically high"
        elif cpu_percent > 75 or memory.percent > 75:
            status = HealthStatus.DEGRADED
            message = "System resources elevated"
        
        return HealthCheckResult(
            component="system_resources",
            status=status,
            message=message,
            details={
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            },
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=0
        )
    
    async def _check_gpu_availability(self) -> HealthCheckResult:
        """Check GPU availability and status."""
        if not torch.cuda.is_available():
            return HealthCheckResult(
                component="gpu_availability",
                status=HealthStatus.DEGRADED,
                message="CUDA not available",
                details={'cuda_available': False},
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=0
            )
        
        gpu_count = torch.cuda.device_count()
        gpu_details = {}
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_used = torch.cuda.memory_allocated(i) / (1024**3)
            memory_total = props.total_memory / (1024**3)
            
            gpu_details[f'gpu_{i}'] = {
                'name': props.name,
                'memory_used_gb': round(memory_used, 2),
                'memory_total_gb': round(memory_total, 2),
                'memory_percent': round((memory_used / memory_total) * 100, 2)
            }
        
        return HealthCheckResult(
            component="gpu_availability",
            status=HealthStatus.HEALTHY,
            message=f"{gpu_count} GPU(s) available",
            details={
                'gpu_count': gpu_count,
                'gpus': gpu_details
            },
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=0
        )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        disk_usage = psutil.disk_usage('/')
        usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        status = HealthStatus.HEALTHY
        message = "Disk space sufficient"
        
        if usage_percent > 95:
            status = HealthStatus.CRITICAL
            message = "Disk space critically low"
        elif usage_percent > 85:
            status = HealthStatus.DEGRADED
            message = "Disk space getting low"
        
        return HealthCheckResult(
            component="disk_space",
            status=status,
            message=message,
            details={
                'usage_percent': round(usage_percent, 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'total_gb': round(disk_usage.total / (1024**3), 2)
            },
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=0
        )
    
    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage patterns."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        status = HealthStatus.HEALTHY
        message = "Memory usage normal"
        
        if memory.percent > 95 or swap.percent > 50:
            status = HealthStatus.CRITICAL
            message = "Memory usage critical"
        elif memory.percent > 85 or swap.percent > 25:
            status = HealthStatus.DEGRADED
            message = "Memory usage elevated"
        
        return HealthCheckResult(
            component="memory_usage",
            status=status,
            message=message,
            details={
                'memory_percent': memory.percent,
                'swap_percent': swap.percent,
                'available_gb': round(memory.available / (1024**3), 2),
                'cached_gb': round(memory.cached / (1024**3), 2)
            },
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=0
        )
    
    async def _check_pytorch_functionality(self) -> HealthCheckResult:
        """Check PyTorch functionality."""
        try:
            # Test basic tensor operations
            x = torch.randn(10, 10)
            y = torch.mm(x, x.t())
            
            # Test GPU if available
            gpu_test = False
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                y_gpu = torch.mm(x_gpu, x_gpu.t())
                gpu_test = True
            
            return HealthCheckResult(
                component="pytorch_functionality",
                status=HealthStatus.HEALTHY,
                message="PyTorch functioning normally",
                details={
                    'cpu_test': True,
                    'gpu_test': gpu_test,
                    'torch_version': torch.__version__
                },
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="pytorch_functionality",
                status=HealthStatus.CRITICAL,
                message=f"PyTorch functionality error: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=0
            )
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # GPU metrics
        gpu_utilization = []
        gpu_memory_used = []
        gpu_memory_total = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # Note: This would require nvidia-ml-py for actual GPU utilization
                # For now, using memory as a proxy
                memory_used = torch.cuda.memory_allocated(i)
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory
                
                gpu_memory_used.append(memory_used / (1024**3))  # GB
                gpu_memory_total.append(memory_total / (1024**3))  # GB
                gpu_utilization.append((memory_used / memory_total) * 100)
        
        metrics = SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            network_io=network_io,
            process_count=len(psutil.pids()),
            uptime_seconds=time.time() - self.start_time
        )
        
        # Store metrics history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_overall_status(self, health_results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system status from health check results."""
        if not health_results:
            return HealthStatus.UNHEALTHY
        
        status_counts = {}
        for result in health_results:
            status = result.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status based on worst individual status
        if HealthStatus.CRITICAL in status_counts:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in status_counts:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in status_counts:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        # Run health checks synchronously for report
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_results = loop.run_until_complete(self.run_health_checks())
        finally:
            loop.close()
        
        # Collect current metrics
        current_metrics = self.collect_metrics()
        
        # Calculate overall status
        overall_status = self.get_overall_status(health_results)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': overall_status.value,
            'uptime_seconds': current_metrics.uptime_seconds,
            'health_checks': [asdict(result) for result in health_results],
            'current_metrics': asdict(current_metrics),
            'summary': {
                'total_checks': len(health_results),
                'healthy_checks': sum(1 for r in health_results if r.status == HealthStatus.HEALTHY),
                'degraded_checks': sum(1 for r in health_results if r.status == HealthStatus.DEGRADED),
                'unhealthy_checks': sum(1 for r in health_results if r.status == HealthStatus.UNHEALTHY),
                'critical_checks': sum(1 for r in health_results if r.status == HealthStatus.CRITICAL),
            }
        }

# Global health monitor instance
_health_monitor = EnterpriseHealthMonitor()

def get_health_monitor() -> EnterpriseHealthMonitor:
    """Get the global health monitor instance."""
    return _health_monitor

async def health_check_endpoint() -> Dict[str, Any]:
    """Health check endpoint for load balancers and orchestrators."""
    monitor = get_health_monitor()
    health_results = await monitor.run_health_checks()
    overall_status = monitor.get_overall_status(health_results)
    
    return {
        'status': overall_status.value,
        'timestamp': datetime.utcnow().isoformat(),
        'checks': len(health_results),
        'healthy': overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    }

if __name__ == '__main__':
    # Test health monitoring system
    monitor = EnterpriseHealthMonitor()
    report = monitor.generate_health_report()
    
    print("🏥 Enterprise Health Report")
    print("=" * 40)
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"Uptime: {report['uptime_seconds']:.1f} seconds")
    print(f"Health Checks: {report['summary']['healthy_checks']}/{report['summary']['total_checks']} healthy")
    
    print("\n📊 Current Metrics:")
    metrics = report['current_metrics']
    print(f"CPU: {metrics['cpu_percent']:.1f}%")
    print(f"Memory: {metrics['memory_percent']:.1f}%")
    print(f"Disk: {metrics['disk_usage_percent']:.1f}%")
    
    if metrics['gpu_utilization']:
        print(f"GPU Utilization: {metrics['gpu_utilization']}")
    
    print("\n✅ Health monitoring system operational")
