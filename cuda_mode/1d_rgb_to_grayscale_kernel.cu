#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__
void rgb_to_grayscale_kernel(unsigned char* input, unsigned char* out, int n) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= n) return;

    unsigned char r = input[offset + 0];   // red
    unsigned char g = input[offset + n];   // green
    unsigned char b = input[offset + 2*n];   // blue

    out[offset] = (unsigned char) (0.2989f * r + 0.5870f * g + 0.1140f * b);
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    int threads_per_block = 256;
    auto out = torch::empty({h, w}, input.options());

    rgb_to_grayscale_kernel<<<cdiv(w*h, threads_per_block), threads_per_block>>>(input.data_ptr<unsigned char>(), out.data_ptr<unsigned char>(), w*h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}