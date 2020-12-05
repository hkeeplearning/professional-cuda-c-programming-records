#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void CheckDimensionKernel()
{
    printf("threadIdx: (%d, %d, %d); blockIdx: (%d, %d, %d); blockDim: (%d, %d, %d); "
           "gridDim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
            gridDim.x, gridDim.y, gridDim.z);
}

__global__ void SumArraysOnDeviceKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void CheckDimension(const dim3 num_threads, const dim3 num_blocks)
{
    CheckDimensionKernel<<<num_blocks, num_threads>>>();
}

extern "C" void SumArraysOnDevice(
    const float *d_A, 
    const float *d_B, 
    float *d_C, 
    int N,
    const dim3 num_threads,
    const dim3 num_blocks
)
{
    SumArraysOnDeviceKernel<<<num_blocks, num_threads>>>(
        d_A,
        d_B,
        d_C, 
        N
    );
}