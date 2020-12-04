#include <stdio.h>
#include "cuda_runtime.h"

__global__ void HelloWorldGpu()
{
    printf("Hello world from GPU!\n");
}

extern "C" void HelloWorld()
{
    HelloWorldGpu<<<1, 10>>>();
    cudaDeviceReset(); // 释放和清空资源，同时也起到同步的作用
    return;
}