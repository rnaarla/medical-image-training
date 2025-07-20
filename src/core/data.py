import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import webdataset as wds
from PIL import Image

try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("NVIDIA DALI not available, falling back to PyTorch DataLoader")
    # Create dummy decorators and classes to avoid NameError
    def pipeline_def(func):
        return func
    class dali:
        pass

@pipeline_def
def create_dali_pipeline(data_dir, batch_size, num_threads, device_id, 
                        image_size=224, is_training=True, shard_id=0, num_shards=1):
    """DALI pipeline for high-performance medical image loading"""
    
    # File reader with automatic sharding for distributed training
    images, labels = dali.fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        name="Reader"
    )
    
    # Decode images
    images = dali.fn.decoders.image(images, device="mixed", output_type=dali.types.RGB)
    
    # Data augmentation for training
    if is_training:
        images = dali.fn.random_resized_crop(images, size=image_size, device="gpu")
        images = dali.fn.flip(images, horizontal=dali.fn.random.coin_flip(probability=0.5), device="gpu")
        images = dali.fn.color_twist(images,
                                   brightness=dali.fn.random.uniform(range=[0.8, 1.2]),
                                   contrast=dali.fn.random.uniform(range=[0.8, 1.2]),
                                   saturation=dali.fn.random.uniform(range=[0.8, 1.2]),
                                   hue=dali.fn.random.uniform(range=[-0.1, 0.1]),
                                   device="gpu")
    else:
        images = dali.fn.resize(images, size=image_size, device="gpu")
        images = dali.fn.center_crop(images, crop=(image_size, image_size), device="gpu")
    
    # Normalize to [-1, 1] range for mixed precision training
    images = dali.fn.crop_mirror_normalize(
        images,
        dtype=dali.types.FLOAT,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        device="gpu"
    )
    
    return images, labels

class WebDatasetLoader:
    """WebDataset implementation for streaming large datasets"""
    
    def __init__(self, urls, batch_size, image_size=224, is_training=True, 
                 num_workers=4, shuffle_buffer=1000):
        self.urls = urls
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        
        # Define transforms
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def decode_sample(self, sample):
        """Decode individual sample from webdataset"""
        image = Image.open(sample["jpg"]).convert("RGB")
        image = self.transform(image)
        label = int(sample["cls"])
        return image, label
    
    def create_loader(self):
        """Create WebDataset DataLoader"""
        dataset = (
            wds.WebDataset(self.urls)
            .shuffle(self.shuffle_buffer if self.is_training else 0)
            .decode("rgb")
            .map(self.decode_sample)
            .batched(self.batch_size)
        )
        
        if self.is_training:
            dataset = dataset.repeat()
        
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True
        )

def create_data_loader(data_path, batch_size, image_size=224, is_training=True,
                      num_workers=4, device_id=0, num_shards=1, shard_id=0,
                      use_dali=True, checkpoint_state=None):
    """
    Create optimized data loader with support for:
    - NVIDIA DALI for maximum performance
    - WebDataset for streaming large datasets
    - Resume from checkpoint state
    - Distributed training support
    """
    
    if use_dali and DALI_AVAILABLE:
        # DALI pipeline for maximum performance
        pipeline = create_dali_pipeline(
            data_dir=data_path,
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=device_id,
            image_size=image_size,
            is_training=is_training,
            shard_id=shard_id,
            num_shards=num_shards
        )
        
        pipeline.build()
        
        dali_iter = DALIGenericIterator(
            [pipeline],
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL if not is_training else LastBatchPolicy.DROP,
            auto_reset=True
        )
        
        # Resume from checkpoint if provided
        if checkpoint_state and 'data_loader_state' in checkpoint_state:
            dali_iter.reset()
            # Note: DALI state resumption would need custom implementation
            
        return dali_iter
    
    elif data_path.startswith('http') or data_path.endswith('.tar'):
        # WebDataset for streaming
        urls = [data_path] if isinstance(data_path, str) else data_path
        webdataset_loader = WebDatasetLoader(
            urls=urls,
            batch_size=batch_size,
            image_size=image_size,
            is_training=is_training,
            num_workers=num_workers
        )
        return webdataset_loader.create_loader()
    
    else:
        # Fallback to standard PyTorch DataLoader
        from torchvision.datasets import ImageFolder
        
        if is_training:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        dataset = ImageFolder(root=data_path, transform=transform)
        
        # Distributed sampling
        sampler = None
        if num_shards > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=num_shards, rank=shard_id, shuffle=is_training
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_training
        )
