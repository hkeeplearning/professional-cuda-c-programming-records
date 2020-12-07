#include "sum_common.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void InitialData(float* d, int size)
{
    srand(42);

    // C �⺯�� int rand(void) ����һ����Χ�� 0 �� RAND_MAX ֮���α�������
    // RAND_MAX ��һ������������Ĭ��ֵ�ڲ�ͬ��ʵ���л�������ͬ������ֵ������ 32767��
    for (int i = 0; i < size; ++i) {
        d[i] = (float)(rand());
    }
}

void CheckResult(const float* host_ref, const float* device_ref, const int N)
{
    const double kEpsilon = 1.0E-8;
    int match = 1; // �Ƿ���ȣ����Ϊ1������Ϊ0
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