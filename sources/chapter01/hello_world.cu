#include <stdio.h>

extern "C" void HelloWorld();

__global__ void HelloWorldGpu()
{
    printf("Hello world from GPU!\n");
}

void HelloWorld()
{
    HelloWorldGpu<<<1, 10>>>();
}