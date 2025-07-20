import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsTracker:
    """Track and compute training/validation metrics"""
    
    def __init__(self, num_classes=10, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds, targets, loss=None):
        """Update metrics with new batch"""
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        if preds.ndim > 1:  # Convert logits to predictions
            preds = np.argmax(preds, axis=1)
        
        self.predictions.extend(preds.flatten())
        self.targets.extend(targets.flatten())
        
        if loss is not None:
            if torch.is_tensor(loss):
                loss = loss.item()
            self.losses.append(loss)
    
    def compute_accuracy(self):
        """Compute overall accuracy"""
        if not self.predictions:
            return 0.0
        return accuracy_score(self.targets, self.predictions)
    
    def compute_class_metrics(self):
        """Compute per-class precision, recall, F1"""
        if not self.predictions:
            return {}, {}, {}
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.targets, self.predictions, average=None, zero_division=0
        )
        
        precision_dict = {self.class_names[i]: precision[i] for i in range(len(precision))}
        recall_dict = {self.class_names[i]: recall[i] for i in range(len(recall))}
        f1_dict = {self.class_names[i]: f1[i] for i in range(len(f1))}
        
        return precision_dict, recall_dict, f1_dict
    
    def compute_confusion_matrix(self):
        """Compute confusion matrix"""
        if not self.predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(self.targets, self.predictions, 
                              labels=list(range(self.num_classes)))
    
    def plot_confusion_matrix(self, save_path=None, normalize=True):
        """Plot confusion matrix"""
        cm = self.compute_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_summary(self):
        """Get comprehensive metrics summary"""
        accuracy = self.compute_accuracy()
        precision_dict, recall_dict, f1_dict = self.compute_class_metrics()
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        # Compute macro averages
        macro_precision = np.mean(list(precision_dict.values()))
        macro_recall = np.mean(list(recall_dict.values()))
        macro_f1 = np.mean(list(f1_dict.values()))
        
        return {
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision_dict,
            'per_class_recall': recall_dict,
            'per_class_f1': f1_dict
        }

def compute_accuracy(outputs, targets):
    """Simple accuracy computation for quick evaluation"""
    if torch.is_tensor(outputs):
        if outputs.dim() > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = outputs
    else:
        preds = outputs
    
    if torch.is_tensor(targets):
        targets = targets
    
    correct = (preds == targets).float().sum()
    total = targets.size(0)
    return (correct / total).item() * 100

def top_k_accuracy(outputs, targets, k=5):
    """Compute top-k accuracy"""
    with torch.no_grad():
        maxk = max(k, 1)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in range(1, maxk + 1):
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        
        return res[0] if len(res) == 1 else res
