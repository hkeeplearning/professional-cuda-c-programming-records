#include "sum_arrays.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void InitialData(float *d, int size)
{
    srand(42);

    // C 库函数 int rand(void) 返回一个范围在 0 到 RAND_MAX 之间的伪随机数。
    // RAND_MAX 是一个常量，它的默认值在不同的实现中会有所不同，但是值至少是 32767。
    for(int i = 0; i < size; ++i) {
        d[i] = (float)(rand());
    }
}

void CheckResult(const float *host_ref, const float *device_ref, const int N)
{
    const double kEpsilon = 1.0E-8;
    int match = 1; // 是否相等，相等为1，否则为0
    for(int i = 0; i < N; ++i) {
        if(fabs((double)(host_ref[i]) - device_ref[i]) > kEpsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f device %5.2f at current %d\n", 
                host_ref[i], device_ref[i], i);
            break;
        }
    }
    if(match == 1) {
        printf("Arrays match.\n");
    }
    return;
}

void SumArraysOnHost(const float *A, const float *B, float *C, int N)
{
    for(int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int MainSumArraysOnHost(int argc, char *argv[])
{
    const int kElementNumber = 1024;
    size_t bytes = kElementNumber * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    InitialData(h_A, kElementNumber);
    InitialData(h_B, kElementNumber);

    SumArraysOnHost(h_A, h_B, h_C, kElementNumber);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

int MainCheckDimension(int argc, char *argv[])
{
    const int kElementNumber = 6;
    dim3 block(3);
    dim3 grid((kElementNumber + block.x - 1) / block.x);

    printf("grid.x %d grid.y %d grid.z %d", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d", block.x, block.y, block.z);

    CheckDimension(block, grid);

    cudaDeviceSynchronize();

    return 0;
}

int MainDefineGridBlock(int argc, char *argv[])
{
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

int MainSumArraysOnDevice(int argc, char *argv[])
{
    const int kElementNumber = 1024;
    size_t bytes = kElementNumber * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_h_ref = (float *)malloc(bytes);
    float *h_C_d_ref = (float *)malloc(bytes);

    InitialData(h_A, kElementNumber);
    InitialData(h_B, kElementNumber);

    memset(h_C_h_ref, 0, bytes);
    memset(h_C_d_ref, 0, bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)(&d_A), bytes);
    cudaMalloc((float **)(&d_B), bytes);
    cudaMalloc((float **)(&d_C), bytes);

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