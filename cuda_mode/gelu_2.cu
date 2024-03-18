/**
 python -m compile_cuda_kernel \
 --name=gelu_2 \
 --cu_file=gelu_2.cu \
 --cpp_source="torch::Tensor gelu(const torch::Tensor& input);\ntorch::Tensor gelu_out(const torch::Tensor& input, torch::Tensor output);" \
 --funcs=gelu,gelu_out
*/
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__
void gelu_kernel(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    output[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / 3.1415926535897932f) * (x + 0.044715f * x*x*x)));
}

torch::Tensor gelu_out(const torch::Tensor& input, torch::Tensor output) {
    CHECK_INPUT(input);

    TORCH_CHECK((output.sizes() == input.sizes())  || (output.device() == input.device())
                || (output.scalar_type() == input.scalar_type()));
    int threads_per_block = 256;
    int n = input.numel();
    int num_blocks = cdiv(n, threads_per_block);
    gelu_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gelu(const torch::Tensor& input) {
    CHECK_INPUT(input);

    auto output = torch::empty_like(input);
    gelu_out(input, output);

    return output;
}