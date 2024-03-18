#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__
void my_empty_kernel(float* input, float* output, int n) {}

torch::Tensor my_empty_out(torch::Tensor& input, torch::Tensor output) {
    CHECK_INPUT(input);
    int n = input.numel();
    int threads_per_block = 256;
    int num_blocks = cdiv(n, threads_per_block);

    my_empty_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), n);


    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor my_empty(torch::Tensor& input) {
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    my_empty_out(input, output);
    return output;
}