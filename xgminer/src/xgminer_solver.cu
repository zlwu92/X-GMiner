#include "../../include/gpm_solver.cuh"

#include <cub/cub.cuh>

#include "../../include/graph_v2_gpu.h"
#include "../../include/glumin/operations.cuh"
#include "../../include/cuda_utils/cuda_launch_config.hpp"
#include "../../include/glumin/timer.h"
#include "../../include/glumin/codegen_LUT.cuh"
#include "../../include/glumin/codegen_utils.cuh"
#include "../../include/glumin/binary_encode.h"
// #define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

// #include "GM_LUT.cuh"
// #include "GM_build_LUT.cuh"
// #include "GM_LUT_deep.cuh"
// #include "GM_BS_vertex.cuh"
// #include "GM_BS_edge.cuh"

#define BLK_SZ BLOCK_SIZE



void XGMiner::clique_solver(Graph_V2& g) {

}


void XGMiner::motif_solver(Graph_V2& g) {
    LOG_INFO("Running motif solver");

    LOG_INFO("PatternSolver_on_G2Miner with LUT");
    int k = local_patternId;
    assert(k >= 1);
    size_t memsize = print_device_info(0);
    vidType nv = g.num_vertices();
    eidType ne = g.num_edges();
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
    std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
              << " GB, graph_mem = " << mem_graph/1024/1024 << " MB\n";
    if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";
    // CUDA_SAFE_CALL(cudaSetDevice(CUDA_SELECT_DEVICE));
    GraphGPU gg(g);
    gg.init_edgelist(g);

    size_t npatterns = 3;
    AccType *h_counts = (AccType *)malloc(sizeof(AccType) * npatterns);
    for (int i = 0; i < npatterns; i++) h_counts[i] = 0;
    AccType *d_counts;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_counts, sizeof(AccType) * npatterns));
    CUDA_SAFE_CALL(cudaMemset(d_counts, 0, sizeof(AccType) * npatterns));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    size_t n_lists;
    size_t n_bitmaps;
    n_lists = 7;
    n_bitmaps = 3;

    vidType switch_lut = 1;
    // switch_lut = Select_func(nv, ne, md);
    std::cout << "switch_lut = " << switch_lut << "\n";
    size_t nwarps = WARPS_PER_BLOCK;
    size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
    size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;


}
