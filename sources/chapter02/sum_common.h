#ifndef SUM_COMMON_H_
#define SUM_COMMON_H_

void InitialData(float* d, int size);

void CheckResult(const float* host_ref, const float* device_ref, const int N);

#endif // SUM_COMMON_H_
