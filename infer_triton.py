#!/usr/bin/env python3
"""
Triton Inference Server Client for Medical Image Classification
Demonstrates production inference with TLS and performance monitoring
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException, np_to_triton_dtype
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: tritonclient not available, please install with: pip install tritonclient[http]")

class TritonImageClassifier:
    """
    High-performance medical image classifier using NVIDIA Triton Inference Server
    """
    
    def __init__(self, 
                 server_url: str = "localhost:8000",
                 model_name: str = "resnet",
                 model_version: str = "1",
                 use_ssl: bool = False,
                 ssl_cert_path: Optional[str] = None,
                 timeout: float = 10.0):
        """
        Initialize Triton client
        
        Args:
            server_url: Triton server URL (e.g., "localhost:8000" or "https://example.com")
            model_name: Name of the deployed model
            model_version: Model version to use
            use_ssl: Whether to use HTTPS/SSL
            ssl_cert_path: Path to SSL certificate file
            timeout: Request timeout in seconds
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("tritonclient not available. Install with: pip install tritonclient[http]")
        
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version
        self.timeout = timeout
        
        # Initialize client
        self.client = httpclient.InferenceServerClient(
            url=server_url,
            ssl=use_ssl,
            ssl_cert=ssl_cert_path,
            verbose=False
        )
        
        # Get model metadata
        self.model_metadata = self._get_model_metadata()
        self.model_config = self._get_model_config()
        
        # Parse input/output specifications
        self.input_spec = self._parse_inputs()
        self.output_spec = self._parse_outputs()
        
        # Setup image preprocessing
        self.transform = self._setup_preprocessing()
        
        print(f"Connected to Triton server at {server_url}")
        print(f"Model: {model_name} (version {model_version})")
        print(f"Input shape: {self.input_spec['shape']}")
        print(f"Output shape: {self.output_spec['shape']}")
    
    def _get_model_metadata(self) -> Dict:
        """Get model metadata from Triton server"""
        try:
            metadata = self.client.get_model_metadata(self.model_name, self.model_version)
            return metadata
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to get model metadata: {e}")
    
    def _get_model_config(self) -> Dict:
        """Get model configuration from Triton server"""
        try:
            config = self.client.get_model_config(self.model_name, self.model_version)
            return config
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to get model config: {e}")
    
    def _parse_inputs(self) -> Dict:
        """Parse input specifications"""
        inputs = self.model_metadata['inputs']
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 input, got {len(inputs)}")
        
        input_spec = inputs[0]
        return {
            'name': input_spec['name'],
            'shape': input_spec['shape'],
            'datatype': input_spec['datatype']
        }
    
    def _parse_outputs(self) -> Dict:
        """Parse output specifications"""
        outputs = self.model_metadata['outputs']
        if len(outputs) != 1:
            raise ValueError(f"Expected 1 output, got {len(outputs)}")
        
        output_spec = outputs[0]
        return {
            'name': output_spec['name'],
            'shape': output_spec['shape'],
            'datatype': output_spec['datatype']
        }
    
    def _setup_preprocessing(self) -> transforms.Compose:
        """Setup image preprocessing pipeline"""
        # Extract expected input dimensions
        shape = self.input_spec['shape']
        if len(shape) == 4:  # [batch, channel, height, width]
            _, channels, height, width = shape
        elif len(shape) == 3:  # [channel, height, width]
            channels, height, width = shape
        else:
            raise ValueError(f"Unsupported input shape: {shape}")
        
        return transforms.Compose([
            transforms.Resize((int(height * 1.14), int(width * 1.14))),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess single image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        
        # Add batch dimension
        batch_tensor = tensor.unsqueeze(0)
        
        # Convert to numpy
        numpy_array = batch_tensor.numpy()
        
        return numpy_array
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess batch of images for inference
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch of preprocessed images
        """
        batch_list = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image)
            batch_list.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_list)
        return batch_tensor.numpy()
    
    def infer_single(self, image_path: str) -> Dict:
        """
        Run inference on single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Inference results with predictions and metadata
        """
        # Preprocess image
        start_time = time.time()
        input_data = self.preprocess_image(image_path)
        preprocess_time = time.time() - start_time
        
        # Create input object
        inputs = [
            httpclient.InferInput(
                self.input_spec['name'], 
                input_data.shape, 
                np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        
        # Create output object
        outputs = [
            httpclient.InferRequestedOutput(self.output_spec['name'])
        ]
        
        # Run inference
        start_time = time.time()
        response = self.client.infer(
            self.model_name,
            inputs,
            model_version=self.model_version,
            outputs=outputs
        )
        inference_time = time.time() - start_time
        
        # Extract results
        logits = response.as_numpy(self.output_spec['name'])
        probabilities = self._softmax(logits)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = np.max(probabilities, axis=1)[0]
        
        return {
            'image_path': image_path,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist(),
            'logits': logits[0].tolist(),
            'preprocess_time_ms': preprocess_time * 1000,
            'inference_time_ms': inference_time * 1000
        }
    
    def infer_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Run batch inference on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of inference results
        """
        # Preprocess batch
        start_time = time.time()
        input_data = self.preprocess_batch(image_paths)
        preprocess_time = time.time() - start_time
        
        # Create input object
        inputs = [
            httpclient.InferInput(
                self.input_spec['name'], 
                input_data.shape, 
                np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        
        # Create output object
        outputs = [
            httpclient.InferRequestedOutput(self.output_spec['name'])
        ]
        
        # Run inference
        start_time = time.time()
        response = self.client.infer(
            self.model_name,
            inputs,
            model_version=self.model_version,
            outputs=outputs
        )
        inference_time = time.time() - start_time
        
        # Extract results
        logits = response.as_numpy(self.output_spec['name'])
        probabilities = self._softmax(logits)
        predicted_classes = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        # Format results
        results = []
        for i, image_path in enumerate(image_paths):
            results.append({
                'image_path': image_path,
                'predicted_class': int(predicted_classes[i]),
                'confidence': float(confidences[i]),
                'probabilities': probabilities[i].tolist(),
                'logits': logits[i].tolist(),
                'preprocess_time_ms': preprocess_time * 1000 / len(image_paths),
                'inference_time_ms': inference_time * 1000 / len(image_paths)
            })
        
        return results
    
    def benchmark(self, image_path: str, num_runs: int = 100, batch_sizes: List[int] = None) -> Dict:
        """
        Benchmark inference performance
        
        Args:
            image_path: Path to test image
            num_runs: Number of runs per benchmark
            batch_sizes: List of batch sizes to test
            
        Returns:
            Benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")
            
            # Create batch of identical images
            image_paths = [image_path] * batch_size
            
            # Warm up
            for _ in range(5):
                _ = self.infer_batch(image_paths)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.infer_batch(image_paths)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            throughput = batch_size / (avg_time / 1000)  # Images per second
            
            results[f'batch_{batch_size}'] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'throughput_imgs_sec': throughput,
                'latency_per_image_ms': avg_time / batch_size
            }
        
        return results
    
    def health_check(self) -> Dict:
        """Check server health and model status"""
        try:
            # Check server health
            server_ready = self.client.is_server_ready()
            server_live = self.client.is_server_live()
            
            # Check model readiness
            model_ready = self.client.is_model_ready(self.model_name, self.model_version)
            
            # Get server metadata
            server_metadata = self.client.get_server_metadata()
            
            return {
                'server_ready': server_ready,
                'server_live': server_live,
                'model_ready': model_ready,
                'server_name': server_metadata.get('name', 'unknown'),
                'server_version': server_metadata.get('version', 'unknown'),
                'model_name': self.model_name,
                'model_version': self.model_version
            }
        except Exception as e:
            return {
                'error': str(e),
                'healthy': False
            }
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser(description='Triton Inference Client')
    parser.add_argument('--server-url', type=str, default='localhost:8000', 
                       help='Triton server URL')
    parser.add_argument('--model-name', type=str, default='resnet', 
                       help='Model name')
    parser.add_argument('--image-path', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--batch-size', type=int, default=1, 
                       help='Batch size for inference')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    parser.add_argument('--health-check', action='store_true', 
                       help='Perform health check')
    parser.add_argument('--ssl', action='store_true', 
                       help='Use HTTPS/SSL')
    parser.add_argument('--ssl-cert', type=str, 
                       help='Path to SSL certificate')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize client
    try:
        classifier = TritonImageClassifier(
            server_url=args.server_url,
            model_name=args.model_name,
            use_ssl=args.ssl,
            ssl_cert_path=args.ssl_cert
        )
    except Exception as e:
        print(f"Failed to initialize Triton client: {e}")
        sys.exit(1)
    
    # Health check
    if args.health_check:
        health_status = classifier.health_check()
        print("Health Check Results:")
        print(json.dumps(health_status, indent=2))
        return
    
    # Verify image exists
    if not Path(args.image_path).exists():
        print(f"Image not found: {args.image_path}")
        sys.exit(1)
    
    # Run inference
    if args.benchmark:
        print("Running performance benchmark...")
        benchmark_results = classifier.benchmark(args.image_path)
        print("\nBenchmark Results:")
        for batch_size, metrics in benchmark_results.items():
            print(f"{batch_size}: {metrics['throughput_imgs_sec']:.2f} imgs/sec, "
                  f"{metrics['latency_per_image_ms']:.2f} ms/img")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    else:
        if args.batch_size == 1:
            # Single image inference
            result = classifier.infer_single(args.image_path)
            print(f"\nInference Results:")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Preprocessing Time: {result['preprocess_time_ms']:.2f} ms")
            print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
        else:
            # Batch inference (duplicate same image)
            image_paths = [args.image_path] * args.batch_size
            results = classifier.infer_batch(image_paths)
            print(f"\nBatch Inference Results (batch size {args.batch_size}):")
            for i, result in enumerate(results):
                print(f"Image {i+1}: Class {result['predicted_class']}, "
                      f"Confidence {result['confidence']:.4f}")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(result if args.batch_size == 1 else results, f, indent=2)

if __name__ == '__main__':
    main()
