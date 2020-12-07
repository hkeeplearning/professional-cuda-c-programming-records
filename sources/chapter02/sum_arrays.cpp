#include "sum_arrays.h"

#include "pccp_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void SumArraysOnHost(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int MainSumArraysOnHost(int argc, char* argv[])
{
    printf("Starting MainSumArraysOnHost...\n");
    const int kElementNumber = 1024;
    size_t bytes = kElementNumber * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    InitialData(h_A, kElementNumber);
    InitialData(h_B, kElementNumber);

    SumArraysOnHost(h_A, h_B, h_C, kElementNumber);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

int MainCheckDimension(int argc, char* argv[])
{
    printf("Starting MainCheckDimension...\n");
    const int kElementNumber = 6;
    dim3 block(3);
    dim3 grid((kElementNumber + block.x - 1) / block.x);

    printf("grid.x %d grid.y %d grid.z %d", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d", block.x, block.y, block.z);

    CheckDimension(block, grid);

    cudaDeviceSynchronize();

    return 0;
}

int MainDefineGridBlock(int argc, char* argv[])
{
    printf("Starting MainDefineGridBlock...\n");
    const int kElementNumber = 1024;

    dim3 block(1024);
    dim3 grid((kElementNumber + block.x - 1) / block.x);
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    block.x = 512;
    grid.x = (kElementNumber + block.x - 1) / block.x;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    block.x = 256;
    grid.x = (kElementNumber + block.x - 1) / block.x;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    block.x = 128;
    grid.x = (kElementNumber + block.x - 1) / block.x;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    cudaDeviceReset();

    return 0;
}

int MainSumArraysOnDevice(int argc, char* argv[])
{
    printf("Starting MainSumArraysOnDevice...\n");
    const int kElementNumber = 1024;
    size_t bytes = kElementNumber * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_h_ref = (float*)malloc(bytes);
    float* h_C_d_ref = (float*)malloc(bytes);

    InitialData(h_A, kElementNumber);
    InitialData(h_B, kElementNumber);

    memset(h_C_h_ref, 0, bytes);
    memset(h_C_d_ref, 0, bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)(&d_A), bytes);
    cudaMalloc((float**)(&d_B), bytes);
    cudaMalloc((float**)(&d_C), bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(kElementNumber);
    dim3 grid((kElementNumber + block.x - 1) / block.x);
    SumArraysOnDevice(d_A, d_B, d_C, kElementNumber, block, grid);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    cudaMemcpy(h_C_d_ref, d_C, bytes, cudaMemcpyDeviceToHost);

    SumArraysOnHost(h_A, h_B, h_C_h_ref, kElementNumber);

    CheckResult(h_C_h_ref, h_C_d_ref, kElementNumber);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_h_ref);
    free(h_C_d_ref);

    return 0;
}

int MainSumArraysOnDeviceTimer(int argc, char* argv[])
{
    printf("Starting MainSumArraysOnDeviceTimer...\n");
    const int kElementNumber = 1 << 24;
    size_t bytes = kElementNumber * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_h_ref = (float*)malloc(bytes);
    float* h_C_d_ref = (float*)malloc(bytes);

    InitialData(h_A, kElementNumber);
    InitialData(h_B, kElementNumber);

    memset(h_C_h_ref, 0, bytes);
    memset(h_C_d_ref, 0, bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)(&d_A), bytes);
    cudaMalloc((float**)(&d_B), bytes);
    cudaMalloc((float**)(&d_C), bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(1024);
    dim3 grid((kElementNumber + block.x - 1) / block.x);
    SumArraysOnDevice(d_A, d_B, d_C, kElementNumber, block, grid);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    cudaMemcpy(h_C_d_ref, d_C, bytes, cudaMemcpyDeviceToHost);

    SumArraysOnHost(h_A, h_B, h_C_h_ref, kElementNumber);

    CheckResult(h_C_h_ref, h_C_d_ref, kElementNumber);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_h_ref);
    free(h_C_d_ref);

    return 0;
}
