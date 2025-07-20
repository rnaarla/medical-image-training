import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports


from src.core.model import get_model
from src.core.data import create_data_loader
from src.core.metrics import MetricsTracker, compute_accuracy, top_k_accuracy
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports


# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from project_config import setup_imports
                    import time

#!/usr/bin/env python3
"""
Model Evaluation Script for Medical Image Classification
Comprehensive evaluation with metrics, visualizations, and reporting
"""

# Local imports
class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics and visualizations"""
    
    def __init__(self, 
                 model_path: str,
                 data_path: str,
                 num_classes: int = 10,
                 device: str = 'auto',
                 class_names: Optional[List[str]] = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to evaluation dataset
            num_classes: Number of classes
            device: Device to use ('cuda', 'cpu', or 'auto')
            class_names: List of class names for visualization
        """
        self.model_path = model_path
        self.data_path = data_path
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(num_classes=num_classes, class_names=self.class_names)
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint"""
        print(f"Loading model from {self.model_path}")
        
        # Create model instance
        model = get_model(num_classes=self.num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DDP training)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_state_dict[key[7:]] = value
            else:
                cleaned_state_dict[key] = value
        
        model.load_state_dict(cleaned_state_dict)
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def evaluate(self, 
                data_loader: DataLoader, 
                return_predictions: bool = False) -> Dict:
        """
        Evaluate model on dataset
        
        Args:
            data_loader: DataLoader for evaluation data
            return_predictions: Whether to return individual predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Starting evaluation...")
        
        self.metrics.reset()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Convert to probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                if return_predictions:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                
                # Progress update
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx + 1} batches...")
        
        # Compute final metrics
        results = self.metrics.get_summary()
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['probabilities'] = all_probabilities
            results['targets'] = all_targets
        
        print(f"Evaluation completed. Overall accuracy: {results['accuracy']:.4f}")
        return results
    
    def generate_report(self, results: Dict, output_dir: str = "./evaluation_results"):
        """Generate comprehensive evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating evaluation report in {output_dir}")
        
        # 1. Summary metrics
        self._save_summary_metrics(results, output_dir / "summary_metrics.json")
        
        # 2. Confusion matrix
        self._plot_confusion_matrix(output_dir / "confusion_matrix.png")
        
        # 3. Per-class metrics
        self._plot_per_class_metrics(results, output_dir / "per_class_metrics.png")
        
        # 4. ROC curves (if probabilities available)
        if 'probabilities' in results:
            self._plot_roc_curves(results, output_dir / "roc_curves.png")
        
        # 5. Classification report
        self._save_classification_report(results, output_dir / "classification_report.txt")
        
        # 6. Prediction examples
        if 'predictions' in results:
            self._save_prediction_examples(results, output_dir / "prediction_examples.json")
        
        print("Evaluation report generated successfully")
    
    def _save_summary_metrics(self, results: Dict, filepath: Path):
        """Save summary metrics to JSON"""
        summary = {
            'overall_accuracy': float(results['accuracy']),
            'average_loss': float(results['avg_loss']),
            'macro_precision': float(results['macro_precision']),
            'macro_recall': float(results['macro_recall']),
            'macro_f1': float(results['macro_f1']),
            'num_classes': self.num_classes,
            'model_path': str(self.model_path),
            'data_path': str(self.data_path)
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _plot_confusion_matrix(self, filepath: Path):
        """Plot and save confusion matrix"""
        self.metrics.plot_confusion_matrix(save_path=str(filepath), normalize=True)
    
    def _plot_per_class_metrics(self, results: Dict, filepath: Path):
        """Plot per-class precision, recall, F1 scores"""
        classes = self.class_names
        precision = [results['per_class_precision'][cls] for cls in classes]
        recall = [results['per_class_recall'][cls] for cls in classes]
        f1 = [results['per_class_f1'][cls] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, results: Dict, filepath: Path):
        """Plot ROC curves for each class"""
        targets = np.array(results['targets'])
        probabilities = np.array(results['probabilities'])
        
        # Binarize targets for multiclass ROC
        targets_binary = label_binarize(targets, classes=range(self.num_classes))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(targets_binary[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multiclass Classification')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_classification_report(self, results: Dict, filepath: Path):
        """Save sklearn classification report"""
        if 'predictions' in results and 'targets' in results:
            report = classification_report(
                results['targets'], 
                results['predictions'],
                target_names=self.class_names,
                digits=4
            )
            
            with open(filepath, 'w') as f:
                f.write("Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
        else:
            print("Skipping classification report - predictions not available")
    
    def _save_prediction_examples(self, results: Dict, filepath: Path):
        """Save examples of correct and incorrect predictions"""
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        probabilities = np.array(results['probabilities'])
        
        # Find correct and incorrect predictions
        correct_mask = predictions == targets
        incorrect_mask = ~correct_mask
        
        examples = {
            'correct_examples': [],
            'incorrect_examples': []
        }
        
        # Sample correct predictions
        correct_indices = np.where(correct_mask)[0]
        if len(correct_indices) > 0:
            sample_size = min(10, len(correct_indices))
            sample_indices = np.random.choice(correct_indices, sample_size, replace=False)
            
            for idx in sample_indices:
                examples['correct_examples'].append({
                    'index': int(idx),
                    'true_class': int(targets[idx]),
                    'predicted_class': int(predictions[idx]),
                    'confidence': float(probabilities[idx].max()),
                    'class_name': self.class_names[targets[idx]]
                })
        
        # Sample incorrect predictions
        incorrect_indices = np.where(incorrect_mask)[0]
        if len(incorrect_indices) > 0:
            sample_size = min(10, len(incorrect_indices))
            sample_indices = np.random.choice(incorrect_indices, sample_size, replace=False)
            
            for idx in sample_indices:
                examples['incorrect_examples'].append({
                    'index': int(idx),
                    'true_class': int(targets[idx]),
                    'predicted_class': int(predictions[idx]),
                    'confidence': float(probabilities[idx].max()),
                    'true_class_name': self.class_names[targets[idx]],
                    'predicted_class_name': self.class_names[predictions[idx]]
                })
        
        with open(filepath, 'w') as f:
            json.dump(examples, f, indent=2)
    
    def benchmark_performance(self, data_loader: DataLoader) -> Dict:
        """Benchmark inference performance"""
        print("Benchmarking inference performance...")
        
        # Warmup
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 5:  # 5 warmup iterations
                    break
                data = data.to(self.device)
                _ = self.model(data)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        times = []
        batch_sizes = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                batch_size = data.size(0)
                
                start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                
                if self.device.type == 'cuda':
                    start_time.record()
                else:
                    start_time = time.time()
                
                _ = self.model(data)
                
                if self.device.type == 'cuda':
                    end_time.record()
                    torch.cuda.synchronize()
                    batch_time = start_time.elapsed_time(end_time)  # milliseconds
                else:
                    batch_time = (time.time() - start_time) * 1000  # convert to milliseconds
                
                times.append(batch_time)
                batch_sizes.append(batch_size)
        
        # Calculate statistics
        total_samples = sum(batch_sizes)
        total_time = sum(times)
        avg_time_per_batch = np.mean(times)
        avg_time_per_sample = total_time / total_samples
        throughput = 1000 / avg_time_per_sample  # samples per second
        
        benchmark_results = {
            'total_samples': total_samples,
            'total_time_ms': total_time,
            'avg_time_per_batch_ms': avg_time_per_batch,
            'avg_time_per_sample_ms': avg_time_per_sample,
            'throughput_samples_per_sec': throughput,
            'device': str(self.device)
        }
        
        print(f"Benchmark completed:")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Avg time per sample: {avg_time_per_sample:.2f} ms")
        
        return benchmark_results

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to evaluation dataset')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--class-names', type=str, nargs='+',
                       help='List of class names')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.data_path).exists():
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        num_classes=args.num_classes,
        device=args.device,
        class_names=args.class_names
    )
    
    # Create data loader
    data_loader = create_data_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        image_size=224,
        is_training=False,
        num_workers=4,
        use_dali=False  # Use standard DataLoader for evaluation
    )
    
    # Run evaluation
    results = evaluator.evaluate(data_loader, return_predictions=True)
    
    # Generate report
    evaluator.generate_report(results, args.output_dir)
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_results = evaluator.benchmark_performance(data_loader)
        
        # Save benchmark results
        output_dir = Path(args.output_dir)
        with open(output_dir / "benchmark_results.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
