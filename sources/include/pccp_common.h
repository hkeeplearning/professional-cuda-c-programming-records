#ifndef PCCP_COMMON_H_
#define PCCP_COMMON_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define PCCP_DEBUG

#if defined(PCCP_DEBUG)
#define PCCP_CHECK(call)                                                                \
    do {                                                                                \
        const cudaError_t errorCode = call;                                             \
        if (errorCode != cudaSuccess) {                                                 \
            printf("Error: %s: %d, ", __FILE__, __LINE__);                              \
            printf("code: %d, reason: %s\n", errorCode, cudaGetErrorString(errorCode)); \
            exit(1);                                                                    \
        }                                                                               \
    } while (false)
#else
#define PCCP_CHECK(call) call
#endif

class TimeWatcher {
public:
    TimeWatcher()
    {
        start_time_ = clock();
    }
    void Reset()
    {
        start_time_ = clock();
    }
    double GetElapsedSeconds()
    {
        return (GetElapsedMilliSeconds() / CLOCKS_PER_SEC);
    }
    double GetElapsedMilliSeconds()
    {
        double interval = (double)(clock())- start_time_;
        return interval;
    }

private:
    clock_t start_time_;
};

#endif // PCCP_COMMON_H_
