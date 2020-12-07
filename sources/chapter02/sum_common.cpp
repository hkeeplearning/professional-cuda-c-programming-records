#include "sum_common.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void InitialData(float* d, int size)
{
    srand(42);

    // C 库函数 int rand(void) 返回一个范围在 0 到 RAND_MAX 之间的伪随机数。
    // RAND_MAX 是一个常量，它的默认值在不同的实现中会有所不同，但是值至少是 32767。
    for (int i = 0; i < size; ++i) {
        d[i] = (float)(rand());
    }
}

void CheckResult(const float* host_ref, const float* device_ref, const int N)
{
    const double kEpsilon = 1.0E-8;
    int match = 1; // 是否相等，相等为1，否则为0
    for (int i = 0; i < N; ++i) {
        if (fabs((double)(host_ref[i]) - device_ref[i]) > kEpsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f device %5.2f at current %d\n",
                host_ref[i], device_ref[i], i);
            break;
        }
    }
    if (match == 1) {
        printf("Arrays match.\n");
    }
    return;
}