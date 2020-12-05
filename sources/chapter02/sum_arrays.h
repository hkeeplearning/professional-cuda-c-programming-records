#ifndef SUM_ARRAYS_H_
#define SUM_ARRAYS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

void InitialData(float *d, int size);

void CheckResult(const float *host_ref, const float *device_ref, const int N);

void SumArraysOnHost(const float *A, const float *B, float *C, int N);

int MainSumArraysOnHost(int argc, char *argv[]);

extern "C" void CheckDimension(const dim3 num_threads, const dim3 num_blocks);
int MainCheckDimension(int argc, char *argv[]);

int MainDefineGridBlock(int argc, char *argv[]);

extern "C" void SumArraysOnDevice(
    const float *d_A, 
    const float *d_B, 
    float *d_C, 
    int N,
    const dim3 num_threads,
    const dim3 num_blocks);
int MainSumArraysOnDevice(int argc, char *argv[]);
int MainSumArraysOnDeviceTimer(int argc, char* argv[]);

#endif // SUM_ARRAYS_H_
