#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for efficient grayscale conversion and normalization
__global__ void grayscale_normalize_kernel(
    float* __restrict__ input,     // Input RGB image [B, 3, H, W]
    float* __restrict__ output,    // Output grayscale [B, 1, H, W]
    const int batch_size,
    const int height,
    const int width,
    const float mean,
    const float std
) {
    // Calculate global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * height * width;
    
    if (idx < total_elements) {
        // Calculate batch, height, width indices
        const int b = idx / (height * width);
        const int hw = idx % (height * width);
        
        // Calculate input indices for R, G, B channels
        const int r_idx = b * 3 * height * width + 0 * height * width + hw;
        const int g_idx = b * 3 * height * width + 1 * height * width + hw;
        const int b_idx = b * 3 * height * width + 2 * height * width + hw;
        
        // Load RGB values
        const float r = input[r_idx];
        const float g = input[g_idx];
        const float b_val = input[b_idx];
        
        // Convert to grayscale using standard weights
        // Y = 0.299*R + 0.587*G + 0.114*B
        const float gray = 0.299f * r + 0.587f * g + 0.114f * b_val;
        
        // Normalize: (gray - mean) / std
        const float normalized = (gray - mean) / std;
        
        // Store result
        const int out_idx = b * height * width + hw;
        output[out_idx] = normalized;
    }
}

// CUDA kernel for batch grayscale conversion with per-channel mean/std
__global__ void batch_grayscale_normalize_kernel(
    float* __restrict__ input,     // Input RGB [B, 3, H, W]
    float* __restrict__ output,    // Output grayscale [B, 1, H, W]
    const float* __restrict__ mean, // Per-channel means [3]
    const float* __restrict__ std,  // Per-channel stds [3]
    const int batch_size,
    const int height,
    const int width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * height * width;
    
    if (idx < total_elements) {
        const int b = idx / (height * width);
        const int hw = idx % (height * width);
        
        // Input indices for RGB channels
        const int r_idx = b * 3 * height * width + 0 * height * width + hw;
        const int g_idx = b * 3 * height * width + 1 * height * width + hw;
        const int b_idx = b * 3 * height * width + 2 * height * width + hw;
        
        // Load and normalize RGB values
        const float r = (input[r_idx] - mean[0]) / std[0];
        const float g = (input[g_idx] - mean[1]) / std[1];
        const float b_val = (input[b_idx] - mean[2]) / std[2];
        
        // Convert normalized RGB to grayscale
        const float gray = 0.299f * r + 0.587f * g + 0.114f * b_val;
        
        // Store result
        const int out_idx = b * height * width + hw;
        output[out_idx] = gray;
    }
}

// Host function to launch grayscale conversion kernel
torch::Tensor grayscale_normalize_cuda(
    torch::Tensor input,
    float mean = 0.5f,
    float std = 0.5f
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    // Ensure input is on GPU and contiguous
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(channels == 3, "Input must have 3 channels (RGB)");
    
    // Create output tensor [B, 1, H, W]
    auto output = torch::empty({batch_size, 1, height, width}, input.options());
    
    // Launch parameters
    const int total_elements = batch_size * height * width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    grayscale_normalize_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        height,
        width,
        mean,
        std
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Host function for batch normalization with per-channel statistics
torch::Tensor batch_grayscale_normalize_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor std
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    // Validation
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(channels == 3, "Input must have 3 channels");
    TORCH_CHECK(mean.size(0) == 3, "Mean must have 3 elements");
    TORCH_CHECK(std.size(0) == 3, "Std must have 3 elements");
    
    // Create output tensor
    auto output = torch::empty({batch_size, 1, height, width}, input.options());
    
    // Launch parameters
    const int total_elements = batch_size * height * width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    batch_grayscale_normalize_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        std.data_ptr<float>(),
        batch_size,
        height,
        width
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grayscale_normalize", &grayscale_normalize_cuda, 
          "Grayscale conversion with normalization (CUDA)",
          py::arg("input"), py::arg("mean") = 0.5f, py::arg("std") = 0.5f);
    
    m.def("batch_grayscale_normalize", &batch_grayscale_normalize_cuda,
          "Batch grayscale conversion with per-channel normalization (CUDA)",
          py::arg("input"), py::arg("mean"), py::arg("std"));
}
