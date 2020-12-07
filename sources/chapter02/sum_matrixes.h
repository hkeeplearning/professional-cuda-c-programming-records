#ifndef SUM_MATRIXES_H_
#define SUM_MATRIXES_H_

#include "sum_common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

extern "C" void CheckThreadIndex(
    int* d_A, const int nx, const int ny, const dim3 num_threads, const dim3 num_blocks);
int MainCheckThreadIndex(int argc, char* argv[]);

void SumMatrixOnHost(float* mat_A, float* mat_B, float* mat_C, int nx, int ny);

extern "C" void SumMatrixOnGpu_1D_Grid_1D_Block(float* d_mat_A, float* d_mat_B, float* d_mat_C,
    int nx, int ny, const dim3 num_threads, const dim3 num_blocks);
extern "C" void SumMatrixOnGpu_2D_Grid_1D_Block(float* d_mat_A, float* d_mat_B, float* d_mat_C,
    int nx, int ny, const dim3 num_threads, const dim3 num_blocks);
extern "C" void SumMatrixOnGpu_2D_Grid_2D_Block(float* d_mat_A, float* d_mat_B, float* d_mat_C,
    int nx, int ny, const dim3 num_threads, const dim3 num_blocks);

int MainSumMatrix(int argc, char* argv[]);

#endif // SUM_MATRIXES_H_
