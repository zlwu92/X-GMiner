#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
// #include <device_launch_parameters.h>
// #include <device_functions.h>
// #include <device_atomic_functions.h>
// #include <cooperative_groups.h>

#define BLOCK_SIZE 128

#define WARP_SIZE 32

class GPM_Solver {
public:
    GPM_Solver();
    ~GPM_Solver() {}
    

private:

};