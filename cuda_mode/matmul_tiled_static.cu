#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__host__ __device__
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

constexpr int TILE_SIZE = 16;

__global__
void matmul_tiled_kernel(float* A, float* B, float* output, int height, int width, int k) {
    int tile_row = threadIdx.y, tile_col = threadIdx.x;
    int row = blockIdx.y * blockDim.y + tile_row;
    int col = blockIdx.x * blockDim.x + tile_col;

    __shared__ float A_shared_memory[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared_memory[TILE_SIZE][TILE_SIZE];

    float p = 0.0f;
    for (int ph = 0; ph < cdiv(k, TILE_SIZE); ++ph) {
        int idx = ph * TILE_SIZE;
        A_shared_memory[tile_row][tile_col] = (row < height && idx + tile_col < k) ? A[row * k + idx + tile_col] : 0.0f;
        B_shared_memory[tile_row][tile_col] = (idx + tile_row < k && col < width) ? B[(idx + tile_row) * width + col] : 0.0f;
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            p += A_shared_memory[tile_row][i] * B_shared_memory[i][tile_col];
        }
        __syncthreads();
    }

    if (row < height && col < width) {
        output[row * width + col] = p;
    }
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);

    TORCH_CHECK(k == B.size(0), "Size must match!");

    auto output = torch::zeros({h, w}, A.options());

    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks(cdiv(w, threads_per_block.x), cdiv(h, threads_per_block.y));

    matmul_tiled_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        h,
        w,
        k
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}