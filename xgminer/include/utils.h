#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "log_manager.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "common.h"

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#define PRINT_CYAN(x) std::cout << "\033[1;36m" << x << "\033[0m" << std::endl

#define FULLMASK 0xffffffff
#define WARPSIZE 32

#define BLOCKSIZE 128

#define SHFL_DOWN_REDUCE(v)                                                    \
    v += __shfl_down_sync(FULLMASK, v, 16);                                      \
    v += __shfl_down_sync(FULLMASK, v, 8);                                       \
    v += __shfl_down_sync(FULLMASK, v, 4);                                       \
    v += __shfl_down_sync(FULLMASK, v, 2);                                       \
    v += __shfl_down_sync(FULLMASK, v, 1);

#define CEIL(a, b) ((a + b - 1) / b)

#define TIMERSTART(label)                                                    \
        cudaSetDevice(0);                                                    \
        cudaEvent_t start##label, stop##label;                               \
        float time##label;                                                   \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaSetDevice(0);                                                    \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
        std::cout << "kernel execution time: #" << time##label               \
                  << " ms (" << #label << ")" << std::endl;


#define UP_DIV(n, a)    (n + a - 1) / a
#define ROUND_UP(n, a)  UP_DIV(n, a) * a

#define zwu_error(...)                                                         \
  fprintf(stdout, "%s:line %d:\t", __FILE__, __LINE__);                        \
  fprintf(stdout, __VA_ARGS__);                                                \
  fprintf(stdout, "\n");

#if 0
#define CHECK_ERROR(err)                                                       \
  if (err != cudaSuccess) {                                                    \
    printf("%s:%d:\t", __FILE__, __LINE__);                               \
    std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;            \
    exit(-1);                                                                  \
  }
#endif

#define CHECK_LAST_ERROR                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << cudaGetErrorString(err) << std::endl;                       \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_ERROR(err)    CheckCudaError(err, __FILE__, __LINE__)
#define GET_LAST_ERR()      getCudaLastErr(__FILE__, __LINE__)

inline void getCudaLastErr(const char *file, const int line) {
    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess) {
        std::cerr << "CUDA error: " << file << "(" << line << "): " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

inline void CheckCudaError(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << file << "(" << line << "): " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

#define checkCudaError(a)                                                      \
    do {                                                                         \
        if (cudaSuccess != (a)) {                                                  \
            fprintf(stderr, "Cuda runTime error in line %d of file %s : %s \n",      \
                    __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));     \
            exit(EXIT_FAILURE);                                                      \
        }                                                                          \
    } while (0)

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }


__inline__ __device__ bool binarySearch(unsigned* arr, int target, int left, int right);
__inline__ __device__ int bin_search(int* arr, int target, int left, int right);

__inline__ __device__ bool binarySearch(unsigned* arr, int target, int left, int right) {
    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return true; // 找到目标元素
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return false; // 目标元素未找到
}

__inline__ __device__ int bin_search(int* arr, int target, int left, int right) {
    int start = left;
    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            // printf("mid: %d, left: %d\n", mid, left);
            return mid - start; // 找到目标元素
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1; // 目标元素未找到
}


inline std::string device_prop_string(cudaDeviceProp prop) {
    int ordinal;
    cudaGetDevice(&ordinal);

    size_t freeMem, totalMem;
    CHECK_ERROR(cudaMemGetInfo(&freeMem, &totalMem));

    double memBandwidth = (prop.memoryClockRate * 1000.0) *
      (prop.memoryBusWidth / 8 * 2) / 1.0e9;

    std::stringstream ss;
    ss << prop.name << " : " << prop.clockRate / 1000.0 << " Mhz   (Ordinal " << ordinal << ")\n";
    ss << prop.multiProcessorCount << " SMs enabled. Compute Capability sm_" << prop.major << prop.minor << "\n";
    ss << "FreeMem: " << (int)(freeMem / (1<< 20)) << "MB   TotalMem: " << (int)(totalMem / (1<< 20)) << "MB   " << 8 * sizeof(int*) << "-bit pointers.\n";
    ss << "Mem Clock: " << prop.memoryClockRate / 1000.0 << " Mhz x " << prop.memoryBusWidth << " bits   (" << memBandwidth << " GB/s)\n";
    // ss << "ECC " << (prop.ECCEnabled ? "Enabled" : "Disabled");
    std::string s = ss.str();
    return s;
}

class GPUTimer {
public:
    GPUTimer() {
        checkCudaError(cudaEventCreate(&t1));
        checkCudaError(cudaEventCreate(&t2));
        // cudaEventCreate(&t1);   
        // cudaEventCreate(&t2);
    }
    void start() { cudaEventRecord(t1, 0); }
    void end() {
        cudaEventRecord(t2, 0);
        // cudaEventSynchronize(t1);
        // cudaEventSynchronize(t2);
    }
    float elapsed() {
        cudaEventElapsedTime(&time, t1, t2);
        return time;
    }

    void end_with_sync() {
        cudaEventRecord(t2, 0);
        // cudaEventSynchronize(t1);
        cudaEventSynchronize(t2);
    }

private:
    float time;
    cudaEvent_t t1, t2;
};


/*********************************** GLUMIN *******************************/
namespace utils {
template <typename T>
bool search(const std::vector<T> &vlist, T key){
    return std::find(vlist.begin(), vlist.end(), key) != vlist.end();
}

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

inline std::vector<int> PrefixSum(const std::vector<int> &degrees) {
    std::vector<int> sums(degrees.size() + 1);
    int total = 0;
    for (size_t n=0; n < degrees.size(); n++) {
        sums[n] = total;
        total += degrees[n];
    }
    sums[degrees.size()] = total;
    return sums;
}

}



/*********************************** X-GMiner *******************************/
#include <iostream>
#include <chrono>

class CPUTimer {
public:
    // 构造函数：默认不启动计时器
    CPUTimer() : start_time_(std::chrono::high_resolution_clock::now()), is_running_(false) {}

    // 启动计时器
    void start() {
        if (!is_running_) {
            start_time_ = std::chrono::high_resolution_clock::now();
            is_running_ = true;
        }
    }

    // 停止计时器
    void stop() {
        if (is_running_) {
            elapsed_time_ += std::chrono::high_resolution_clock::now() - start_time_;
            is_running_ = false;
        }
    }

    // 重置计时器
    void reset() {
        elapsed_time_ = std::chrono::duration<double>::zero();
        is_running_ = false;
    }

    // 获取经过的时间（以秒为单位）
    double elapsed() const {
        if (is_running_) {
            return elapsed_time_.count() + 
                   std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time_).count();
        }
        return elapsed_time_.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_; // 记录启动时间
    std::chrono::duration<double> elapsed_time_ = std::chrono::duration<double>::zero(); // 累计经过时间
    bool is_running_; // 是否正在运行
};

