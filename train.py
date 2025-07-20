#!/usr/bin/env python3
"""
Distributed Mixed-Precision Image Classification Training
Demonstrates full DL stack: PyTorch DDP, AMP, NCCL, CUDA kernels
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add project configuration
sys.path.insert(0, str(Path(__file__).parent))
from utils.project_config import setup_imports, PROJECT_ROOT, SRC_DIR

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import wandb
from tqdm import tqdm

# Local imports
from src.core.model import get_model
from src.core.data import create_data_loader
from src.core.metrics import MetricsTracker, compute_accuracy

class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        # Model parameters
        self.num_classes = 10
        self.image_size = 224
        self.model_name = "resnet50"
        
        # Training parameters
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        
        # Mixed precision
        self.use_amp = True
        self.grad_clip_norm = 1.0
        
        # Distributed training
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.distributed = False
        
        # Data
        self.data_path = "./data"
        self.num_workers = 8
        self.use_dali = True
        
        # Checkpointing
        self.save_dir = "./checkpoints"
        self.save_every = 10
        self.resume_from = None
        
        # Logging
        self.log_every = 100
        self.eval_every = 1
        self.use_wandb = True
        self.wandb_project = "medical-image-training"

def setup_distributed():
    """Initialize distributed training"""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        
        return world_size, rank, local_rank, True
    else:
        return 1, 0, 0, False

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

class MedicalImageTrainer:
    """Main trainer class with DDP and AMP support"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup distributed training
        self.config.world_size, self.config.rank, self.config.local_rank, self.config.distributed = setup_distributed()
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
            print("CUDA not available, using CPU")
        
        # Initialize model
        self.model = self._setup_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Metrics tracking
        self.train_metrics = MetricsTracker(num_classes=config.num_classes)
        self.val_metrics = MetricsTracker(num_classes=config.num_classes)
        
        # Checkpointing
        self.start_epoch = 0
        self.best_acc = 0.0
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_model(self):
        """Initialize and setup model"""
        model = get_model(
            num_classes=self.config.num_classes,
            pretrained=True
        ).to(self.device)
        
        if self.config.distributed:
            # Sync batch norm across processes
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            # Wrap with DDP
            model = DDP(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=False
            )
        
        return model
    
    def _setup_optimizer(self):
        """Setup optimizer with proper parameters"""
        # Separate weight decay for different parameter groups
        no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm.weight']
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(param_groups, lr=self.config.learning_rate, betas=(0.9, 0.999))
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = self.config.epochs * 1000  # Approximate steps per epoch
        
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_epochs / self.config.epochs
        )
    
    def _setup_logging(self):
        """Setup W&B logging"""
        if self.config.use_wandb and self.config.rank == 0:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
                name=f"medical-training-{int(time.time())}"
            )
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        if self.config.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_acc': self.best_acc,
            'config': vars(self.config)
        }
        
        # Save latest checkpoint
        checkpoint_path = Path(self.config.save_dir) / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.save_dir) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {self.best_acc:.4f}%")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.config.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint.get('best_acc', 0.0)
        
        print(f"Resumed training from epoch {self.start_epoch}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', 
                   disable=self.config.rank != 0)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.config.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            self.train_metrics.update(output, target, loss)
            
            # Update progress bar
            acc = compute_accuracy(output, target)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log metrics
            if batch_idx % self.config.log_every == 0 and self.config.use_wandb and self.config.rank == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': acc,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}', 
                       disable=self.config.rank != 0)
            
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.config.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.val_metrics.update(output, target, loss)
                
                acc = compute_accuracy(output, target)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%'})
        
        # Get validation metrics
        val_summary = self.val_metrics.get_summary()
        val_acc = val_summary['accuracy'] * 100
        val_loss = val_summary['avg_loss']
        
        # Log validation metrics
        if self.config.use_wandb and self.config.rank == 0:
            wandb.log({
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'val/macro_f1': val_summary['macro_f1'],
                'epoch': epoch
            })
        
        # Check if best model
        is_best = val_acc > self.best_acc
        if is_best:
            self.best_acc = val_acc
        
        if self.config.rank == 0:
            print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        return val_acc, is_best
    
    def train(self):
        """Main training loop"""
        # Load checkpoint if resuming
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        # Create data loaders
        train_loader = create_data_loader(
            data_path=os.path.join(self.config.data_path, 'train'),
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            is_training=True,
            num_workers=self.config.num_workers,
            device_id=self.config.local_rank,
            num_shards=self.config.world_size,
            shard_id=self.config.rank,
            use_dali=self.config.use_dali
        )
        
        val_loader = create_data_loader(
            data_path=os.path.join(self.config.data_path, 'val'),
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            is_training=False,
            num_workers=self.config.num_workers,
            device_id=self.config.local_rank,
            num_shards=self.config.world_size,
            shard_id=self.config.rank,
            use_dali=self.config.use_dali
        )
        
        # Training loop
        for epoch in range(self.start_epoch, self.config.epochs):
            start_time = time.time()
            
            # Train
            self.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % self.config.eval_every == 0:
                val_acc, is_best = self.validate(val_loader, epoch)
            else:
                val_acc, is_best = 0.0, False
            
            # Save checkpoint
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - start_time
            if self.config.rank == 0:
                print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        
        # Save final model
        self.save_checkpoint(self.config.epochs - 1, False)
        
        if self.config.use_wandb and self.config.rank == 0:
            wandb.finish()
        
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Medical Image Training')
    parser.add_argument('--data-path', type=str, default='./data', help='Dataset path')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.resume_from = args.resume
    config.use_amp = not args.no_amp
    config.use_wandb = not args.no_wandb
    
    # Create trainer and start training
    trainer = MedicalImageTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
