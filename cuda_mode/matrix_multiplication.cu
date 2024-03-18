#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__
void matmul_kernel(float* m, float* n, float* output, int height, int width, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    float o = 0.0;
    for (int i = 0; i < k; ++i) {
        o += m[row * k +i] * n[i * width + col];
    }
    output[row * width + col] = o;
}


torch::Tensor matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); 
    CHECK_INPUT(n);

    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);

    TORCH_CHECK(k==n.size(0), "Size mismatch!");

    auto output = torch::zeros({h, w}, m.options());

    dim3 threads_per_block(16,16);
    dim3 num_blocks(cdiv(w, threads_per_block.x), cdiv(h, threads_per_block.y));

    matmul_kernel<<<num_blocks, threads_per_block>>>(
        m.data_ptr<float>(), 
        n.data_ptr<float>(), 
        output.data_ptr<float>(), 
        h, w, k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}