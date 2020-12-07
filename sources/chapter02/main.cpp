#include "sum_arrays.h"
#include "sum_matrixes.h"

#include <stdio.h>

int main(int argc, char* argv[])
{
    MainCheckDimension(argc, argv);
    MainDefineGridBlock(argc, argv);
    MainSumArraysOnHost(argc, argv);
    MainSumArraysOnDevice(argc, argv);
    MainSumArraysOnDeviceTimer(argc, argv);

    MainSumMatrix(argc, argv);

    return 0;
}