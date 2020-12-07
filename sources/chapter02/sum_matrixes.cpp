#include "sum_matrixes.h"

#include "pccp_common.h"

#include <algorithm>
#include <stdio.h>
#include <vector>

void InitialInt(int* d, const int size)
{
    for (int i = 0; i < size; ++i) {
        d[i] = i;
    }
}

void PrintMatrix(int* C, const int nx, const int ny)
{
    int* ic = C;
    printf("\nMatrix: (%d, %d)\n", nx, ny);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
}

int MainCheckThreadIndex(int argc, char* argv[])
{
    printf("Starting MainCheckThreadIndex...");
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    PCCP_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    PCCP_CHECK(cudaSetDevice(dev));

    // set matrix
    int nx = 8;
    int ny = 8;
    int nxy = nx * ny;
    int bytes = nxy * sizeof(int);

    int* h_A = (int*)malloc(bytes);

    InitialInt(h_A, nxy);
    PrintMatrix(h_A, nx, ny);

    // malloc device matrix
    int* d_A;
    cudaMalloc((void**)(&d_A), bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    CheckThreadIndex(d_A, nx, ny, block, grid);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    free(h_A);

    cudaDeviceReset();

    return 0;
}

void SumMatrixOnHost(float* mat_A, float* mat_B, float* mat_C, int nx, int ny)
{
    float* ia = mat_A;
    float* ib = mat_B;
    float* ic = mat_C;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            ic[j] = ia[j] + ib[j];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
    return;
}

int MainSumMatrix(int argc, char* argv[])
{
    printf("Starting MainSumMatrix...\n");
    enum class TestCaseType { GRID_1D_BLOCK_1D, GRID_2D_BLOCK_1D, GRID_2D_BLOCK_2D };
    std::vector<TestCaseType> testCaseType { TestCaseType::GRID_1D_BLOCK_1D,
        TestCaseType::GRID_2D_BLOCK_1D, TestCaseType::GRID_2D_BLOCK_2D };

    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int bytes = nxy * sizeof(float);
    TimeWatcher timeWatcher;

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_h_ref = (float*)malloc(bytes);
    float* h_C_d_ref = (float*)malloc(bytes);

    InitialData(h_A, nxy);
    InitialData(h_B, nxy);

    timeWatcher.Reset();
    SumMatrixOnHost(h_A, h_B, h_C_h_ref, nx, ny);
    printf("SumMatrixOnHost elapsed time: %.4f\n", timeWatcher.GetElapsedMilliSeconds());

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    if (std::find(testCaseType.begin(), testCaseType.end(), TestCaseType::GRID_1D_BLOCK_1D)
        != testCaseType.end()) {
        cudaMemset(d_C, 0, bytes);
        dim3 num_threads(32);
        dim3 num_blocks((nx + num_threads.x - 1) / num_threads.x);
        timeWatcher.Reset();
        SumMatrixOnGpu_1D_Grid_1D_Block(d_A, d_B, d_C, nx, ny, num_threads, num_blocks);
        cudaDeviceSynchronize();
        printf("SumMatrixOnGpu_1D_Grid_1D_Block elapsed time: %.4f\n",
            timeWatcher.GetElapsedMilliSeconds());
        cudaMemcpy(h_C_d_ref, d_C, bytes, cudaMemcpyDeviceToHost);
        CheckResult(h_C_h_ref, h_C_d_ref, nxy);
    }

    if (std::find(testCaseType.begin(), testCaseType.end(), TestCaseType::GRID_2D_BLOCK_1D)
        != testCaseType.end()) {
        cudaMemset(d_C, 0, bytes);
        dim3 num_threads(32);
        dim3 num_blocks((nx + num_threads.x - 1) / num_threads.x, ny);
        timeWatcher.Reset();
        SumMatrixOnGpu_2D_Grid_1D_Block(d_A, d_B, d_C, nx, ny, num_threads, num_blocks);
        cudaDeviceSynchronize();
        printf("SumMatrixOnGpu_2D_Grid_1D_Block elapsed time: %.4f\n",
            timeWatcher.GetElapsedMilliSeconds());
        cudaMemcpy(h_C_d_ref, d_C, bytes, cudaMemcpyDeviceToHost);
        CheckResult(h_C_h_ref, h_C_d_ref, nxy);
    }

    if (std::find(testCaseType.begin(), testCaseType.end(), TestCaseType::GRID_2D_BLOCK_2D)
        != testCaseType.end()) {
        cudaMemset(d_C, 0, bytes);
        dim3 num_threads(32, 32);
        dim3 num_blocks(
            (nx + num_threads.x - 1) / num_threads.x, (ny + num_threads.y - 1) / num_threads.y);
        timeWatcher.Reset();
        SumMatrixOnGpu_2D_Grid_2D_Block(d_A, d_B, d_C, nx, ny, num_threads, num_blocks);
        cudaDeviceSynchronize();
        printf("SumMatrixOnGpu_2D_Grid_2D_Block elapsed time: %.4f\n",
            timeWatcher.GetElapsedMilliSeconds());
        cudaMemcpy(h_C_d_ref, d_C, bytes, cudaMemcpyDeviceToHost);
        CheckResult(h_C_h_ref, h_C_d_ref, nxy);
    }

    cudaDeviceReset();
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);

    free(h_C_d_ref);
    free(h_C_h_ref);
    free(h_B);
    free(h_A);

    return 0;
}