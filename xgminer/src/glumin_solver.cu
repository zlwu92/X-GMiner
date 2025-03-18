#include "glumin.h"

// __global__ void clear(AccType *accumulators);

__global__ void clear(AccType *accumulators) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[i] = 0;
}

void GLUMIN::PatternSolver_on_G2Miner() {

}


void GLUMIN::CliqueSolver_on_G2Miner() {

}

