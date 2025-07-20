# Medical Image Upload and Testing Guide

Complete guide for uploading medical images and testing the medical AI training platform.

## 🏥 Overview

This guide covers:
- Setting up medical image datasets
- Uploading images to AWS S3
- Testing image processing pipeline
- Validating model performance
- Running end-to-end workflows

## 🚀 Quick Start

### 1. Quick System Test
```bash
# Run comprehensive system validation
python quick_medical_test.py

# Save results for review
python quick_medical_test.py --save-results quick_test_results.json
```

### 2. Test with Sample Data
```bash
# Create and test sample medical images
python quick_medical_test.py --test-dir ./sample_medical_data
```

## 📊 Detailed Testing

### 1. Full Medical Pipeline Test
```bash
# Install additional dependencies for full testing
pip install pillow opencv-python pydicom pandas boto3 matplotlib seaborn

# Run comprehensive tests
python test_medical_pipeline.py

# Run with custom data directory
python test_medical_pipeline.py --data-dir /path/to/medical/data
```

### 2. Medical Data Pipeline
```bash
# Upload medical images to AWS S3
python medical_data_pipeline.py

# This will:
# - Validate DICOM, PNG, JPEG formats
# - Extract medical metadata
# - Upload to S3 with encryption
# - Generate dataset statistics
```

## 🔧 Supported Medical Image Formats

### DICOM Files (.dcm, .dicom)
- **Metadata Extraction**: Patient ID, Study Date, Modality, Body Part
- **Validation**: DICOM header validation, pixel array checks
- **Processing**: 16-bit depth support, medical normalization

### Standard Images (.png, .jpg, .tiff)
- **Formats**: PNG, JPEG, TIFF with medical adaptations
- **Validation**: Resolution, bit depth, quality metrics
- **Processing**: 8-bit normalization, medical preprocessing

## 📁 Data Organization

### Recommended Directory Structure
```
medical_data/
├── xrays/
│   ├── chest/
│   ├── bone/
│   └── dental/
├── ct_scans/
│   ├── brain/
│   ├── abdomen/
│   └── chest/
├── mri/
│   ├── brain/
│   ├── spine/
│   └── cardiac/
└── ultrasound/
    ├── cardiac/
    ├── obstetric/
    └── abdominal/
```

### S3 Organization (Auto-created)
```
s3://your-bucket/
└── datasets/
    ├── {dataset_name}/
    │   ├── images/
    │   │   ├── image001.dcm
    │   │   ├── image002.png
    │   │   └── ...
    │   ├── metadata.json
    │   └── stats.json
    └── ...
```

## 🧪 Test Scenarios

### 1. Basic Validation Tests
```python
# Test image format validation
from medical_data_pipeline import MedicalImageValidator

validator = MedicalImageValidator()
is_valid, errors, metadata = validator.validate_image(Path("image.dcm"))

print(f"Valid: {is_valid}")
print(f"Errors: {errors}")
print(f"Metadata: {metadata}")
```

### 2. Preprocessing Tests
```python
# Test grayscale conversion
from grayscale_wrapper import GrayscaleProcessorFactory

# Medical imaging processor (0.5 mean/std)
processor = GrayscaleProcessorFactory.create_medical_processor()

# ImageNet processor (standard normalization) 
processor = GrayscaleProcessorFactory.create_imagenet_processor()

# Custom normalization
processor = GrayscaleProcessorFactory.create_custom_processor(0.3, 0.2)
```

### 3. Performance Benchmarking
```python
# Benchmark different processing backends
from grayscale_wrapper import EnterpriseBenchmark

benchmark = EnterpriseBenchmark()
results = benchmark.benchmark_kernels(
    batch_size=32,
    height=512,
    width=512,
    num_runs=100
)

print(f"CUDA Custom: {results['cuda_custom'].execution_time_ms:.2f}ms")
print(f"PyTorch: {results['pytorch'].execution_time_ms:.2f}ms")
```

## 🔍 Quality Metrics

### Automated Quality Assessment
Each uploaded image is analyzed for:

- **Brightness**: Mean pixel intensity
- **Contrast**: Standard deviation of pixels
- **Sharpness**: Laplacian variance (edge detection)
- **Noise Level**: High-frequency content analysis

### Quality Thresholds
```python
# Typical medical image quality ranges
QUALITY_THRESHOLDS = {
    "min_brightness": 20,      # Avoid completely dark images
    "max_brightness": 240,     # Avoid overexposed images  
    "min_contrast": 10,        # Minimum contrast for detail
    "min_sharpness": 100,      # Minimum sharpness score
    "max_noise": 500           # Maximum acceptable noise
}
```

## ☁️ AWS Upload Process

### 1. Setup AWS Credentials
```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

### 2. Upload Dataset
```python
import asyncio
from medical_data_pipeline import MedicalDataUploader

async def upload_medical_data():
    uploader = MedicalDataUploader("your-s3-bucket-name")
    
    results = await uploader.upload_directory(
        directory_path=Path("/path/to/medical/images"),
        dataset_name="chest_xrays_2024",
        progress_callback=lambda current, total, filename: 
            print(f"Uploading {current}/{total}: {filename}")
    )
    
    print(f"Upload complete: {results['successful_uploads']} images")
    return results

# Run upload
results = asyncio.run(upload_medical_data())
```

### 3. Monitor Upload Progress
The uploader provides detailed progress information:
- File validation status
- Upload success/failure
- Dataset statistics
- Quality metrics summary

## 🏃‍♂️ Running Training Jobs

### 1. Start Training with Uploaded Data
```bash
# Deploy to AWS EKS
./deploy_aws.sh full

# Monitor training progress
kubectl logs -n medical-training -l app=medical-training -f
```

### 2. Training Configuration
```yaml
# Kubernetes training job
apiVersion: batch/v1
kind: Job
metadata:
  name: medical-training-job
spec:
  template:
    spec:
      containers:
      - name: training
        image: your-ecr-uri:latest
        args: [
          "--dataset", "s3://your-bucket/datasets/chest_xrays_2024",
          "--epochs", "100",
          "--batch-size", "32",
          "--mixed-precision"
        ]
        resources:
          requests:
            nvidia.com/gpu: 1
```

## 📈 Monitoring and Validation

### 1. Real-time Monitoring
```bash
# Access Grafana dashboard
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open http://localhost:3000 (admin/admin123)
```

### 2. Training Metrics
- **GPU Utilization**: NVIDIA DCGM metrics
- **Training Loss**: Per-epoch loss curves
- **Validation Accuracy**: Model performance metrics
- **Throughput**: Images processed per second

### 3. Model Validation
```python
# Test trained model inference
from infer_triton import TritonInferenceClient

client = TritonInferenceClient("localhost:8000")
result = client.infer("medical_model", input_data)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Image Validation Failures
```bash
# Check image format
file /path/to/image.dcm

# Validate DICOM manually
python -c "import pydicom; ds = pydicom.dcmread('image.dcm'); print(ds)"
```

#### 2. Upload Failures
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check S3 bucket access
aws s3 ls s3://your-bucket-name
```

#### 3. Processing Errors
```bash
# Test grayscale processing
python grayscale_wrapper.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Issues

#### 1. Slow Upload
- Use parallel uploads with `asyncio`
- Check network bandwidth
- Optimize image compression

#### 2. Memory Issues
- Reduce batch size
- Use mixed precision training
- Monitor GPU memory usage

#### 3. Training Speed
- Verify GPU utilization
- Check data loading pipeline
- Use distributed training

## 📋 Test Checklist

Before production deployment, verify:

- [ ] ✅ Image validation passes (>95% success rate)
- [ ] ✅ AWS credentials configured
- [ ] ✅ S3 bucket accessible
- [ ] ✅ Docker running
- [ ] ✅ Grayscale processing tests pass
- [ ] ✅ GPU processing available (if using CUDA)
- [ ] ✅ Kubernetes cluster accessible
- [ ] ✅ Model inference working
- [ ] ✅ Monitoring dashboards accessible

## 🔗 Advanced Features

### 1. Custom Preprocessing Pipeline
```python
# Create custom medical preprocessor
class CustomMedicalProcessor(BaseGrayscaleProcessor):
    def _process_tensor(self, x):
        # Custom medical image preprocessing
        # - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # - Noise reduction
        # - Standardization
        return processed_tensor
```

### 2. Multi-Modal Support
```python
# Handle different medical imaging modalities
MODALITY_CONFIGS = {
    "CT": {"window_center": 40, "window_width": 400},
    "MRI": {"normalization": "z_score"},
    "XRAY": {"enhancement": "clahe"},
    "US": {"denoising": "nlm"}
}
```

### 3. Automated Quality Control
```python
# Automated image quality assessment
def assess_medical_image_quality(image_path):
    # SNR, CNR, uniformity, artifacts
    quality_score = calculate_quality_metrics(image_path)
    return quality_score > QUALITY_THRESHOLD
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test results and error messages
3. Ensure all prerequisites are installed
4. Validate AWS and Docker configuration

## 🎯 Next Steps

1. **Run Quick Test**: `python quick_medical_test.py`
2. **Upload Medical Data**: Configure S3 and run upload script
3. **Deploy Platform**: `./deploy_aws.sh full` 
4. **Start Training**: Monitor via Kubernetes/Grafana
5. **Validate Results**: Check model performance metrics

Your medical AI training platform is ready for production workloads! 🚀
