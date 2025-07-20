#!/usr/bin/env python3
"""
Setup script for compiling custom CUDA kernels
"""

import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, Extension

def get_cuda_version():
    """Get CUDA version for compatibility"""
    if torch.cuda.is_available():
        return torch.version.cuda
    else:
        return None

def get_compute_capability():
    """Get GPU compute capability"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    else:
        return "8.0"  # Default to Ampere architecture

def main():
    """Setup and compile CUDA extensions"""
    
    # Check CUDA availability
    cuda_version = get_cuda_version()
    if cuda_version is None:
        print("Warning: CUDA not available, skipping CUDA kernel compilation")
        return
    
    print(f"Compiling CUDA kernels for CUDA {cuda_version}")
    
    # Get compute capability
    compute_cap = get_compute_capability()
    print(f"Target compute capability: {compute_cap}")
    
    # CUDA compilation flags
    cuda_flags = [
        '-O3',
        '--use_fast_math',
        '-Xcompiler', '-fPIC',
        f'-gencode=arch=compute_{compute_cap.replace(".", "")},code=sm_{compute_cap.replace(".", "")}',
    ]
    
    # C++ compilation flags
    cxx_flags = ['-O3', '-std=c++14']
    
    # Define extension
    ext_modules = [
        CUDAExtension(
            name='grayscale_ops',
            sources=[
                'custom_grayscale_kernel.cu',
            ],
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': cuda_flags
            },
            include_dirs=[
                # Add any additional include directories here
            ],
            libraries=[
                # Add any additional libraries here
            ],
        )
    ]
    
    # Setup
    setup(
        name='grayscale_ops',
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=False)
        },
        zip_safe=False,
    )

if __name__ == '__main__':
    main()
