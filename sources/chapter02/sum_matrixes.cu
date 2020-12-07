#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void PrintThreadIndexKernel(int* A, const int nx, const int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ny;

    printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) "
           "global_index %d value %d",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

__global__ void SumMatrixOnGpu_1D_Grid_1D_Block_Kernel(
    float* d_mat_A, float* d_mat_B, float* d_mat_C, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < nx) {
        for (int i = 0; i < ny; ++i) {
            int idx = i * nx + ix;
            d_mat_C[idx] = d_mat_A[idx] + d_mat_B[idx];
        }
    }
}

__global__ void SumMatrixOnGpu_2D_Grid_1D_Block_Kernel(
    float* d_mat_A, float* d_mat_B, float* d_mat_C, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    if (ix < nx && iy < ny) {
        unsigned int idx = iy * nx + ix;
        d_mat_C[idx] = d_mat_A[idx] + d_mat_B[idx];
    }
}

__global__ void SumMatrixOnGpu_2D_Grid_2D_Block_Kernel(
    float* d_mat_A, float* d_mat_B, float* d_mat_C, int nx, int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        unsigned int idx = iy * nx + ix;
        d_mat_C[idx] = d_mat_A[idx] + d_mat_B[idx];
    }
}

extern "C" void CheckThreadIndex(
    int* d_A, const int nx, const int ny, const dim3 num_threads, const dim3 num_blocks)
{
    PrintThreadIndexKernel<<<num_blocks, num_threads>>>(d_A, nx, ny);
    return;
}

extern "C" void SumMatrixOnGpu_1D_Grid_1D_Block(float* d_mat_A, float* d_mat_B, float* d_mat_C,
    int nx, int ny, const dim3 num_threads, const dim3 num_blocks)
{
    SumMatrixOnGpu_1D_Grid_1D_Block_Kernel<<<num_blocks, num_threads>>>(
        d_mat_A, d_mat_B, d_mat_C, nx, ny);
    return;
}

extern "C" void SumMatrixOnGpu_2D_Grid_1D_Block(float* d_mat_A, float* d_mat_B, float* d_mat_C,
    int nx, int ny, const dim3 num_threads, const dim3 num_blocks)
{
    SumMatrixOnGpu_2D_Grid_1D_Block_Kernel<<<num_blocks, num_threads>>>(
        d_mat_A, d_mat_B, d_mat_C, nx, ny);
    return;
}

extern "C" void SumMatrixOnGpu_2D_Grid_2D_Block(float* d_mat_A, float* d_mat_B, float* d_mat_C,
    int nx, int ny, const dim3 num_threads, const dim3 num_blocks)
{
    SumMatrixOnGpu_2D_Grid_2D_Block_Kernel<<<num_blocks, num_threads>>>(
        d_mat_A, d_mat_B, d_mat_C, nx, ny);
    return;
}