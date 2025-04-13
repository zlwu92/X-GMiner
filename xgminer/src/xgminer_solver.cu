#include "../../include/gpm_solver.cuh"

#include <cub/cub.cuh>

#include "../include/graph_v2_gpu.h"
#include "../include/glumin/operations.cuh"
#include "../include/cuda_utils/cuda_launch_config.hpp"
#include "../include/glumin/timer.h"
#include "../include/glumin/codegen_LUT.cuh"
#include "../include/glumin/codegen_utils.cuh"
#include "../include/glumin/binary_encode.h"
// #define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

// #include "GM_LUT.cuh"
// #include "GM_build_LUT.cuh"
// #include "GM_LUT_deep.cuh"
// #include "GM_BS_vertex.cuh"
// #include "GM_BS_edge.cuh"

#define BLK_SZ BLOCK_SIZE
#include "../include/xgminer_bitmap.cuh"
#include "P2_kernel.cuh"




void XGMiner::clique_solver(Graph_V2& g) {

}


void XGMiner::motif_solver(Graph_V2& g) {
    LOG_INFO("Running motif solver");
    constexpr int BUCKET_NUM = calculate_value(TEMPLATE_BUCKET_NUM);
    LOG_INFO("Calculated BUCKET_NUM = " + std::to_string(BUCKET_NUM));
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
    
    std::cout << "bucket_k = " << TEMPLATE_BUCKET_NUM << "\n";
    XGMiner_BITMAP<> bitmap(TEMPLATE_BUCKET_NUM);
    bitmap.bigset_bitmap_processing(g);
    
    // for (int i = 0; i < nv; ++i) {
    //     if (g.get_degree(i) == 0) {
    //         std::cout << "vertex " << i << " degree 0\n";
    //     }
    // }
    
    GraphGPU gg(g);
    gg.init_edgelist(g);

    size_t npatterns = 1;
    AccType *h_counts = (AccType *)malloc(sizeof(AccType) * npatterns);
    for (int i = 0; i < npatterns; i++) h_counts[i] = 0;
    AccType *d_counts;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_counts, sizeof(AccType) * npatterns));
    CUDA_SAFE_CALL(cudaMemset(d_counts, 0, sizeof(AccType) * npatterns));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    size_t n_lists;
    size_t n_bitmaps;
    n_lists = 5;
    n_bitmaps = 3;

    // vidType switch_lut = 1;
    // switch_lut = Select_func(nv, ne, md);
    // std::cout << "switch_lut = " << switch_lut << "\n";
    size_t nwarps = WARPS_PER_BLOCK;
    size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
    size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;

    nblocks = std::min(nv, 640);
    nblocks = 640;
    nthreads = BLOCK_SIZE_DENSE;
    nwarps = nthreads / WARP_SIZE;
    // cudaDeviceProp deviceProp;
    // CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    // int max_blocks_per_SM;
    // if (k == 1){
    //   max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
    //   std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    // }
    // else {
    //   max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
    //   std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    // } 
    // size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
    // std::cout << "max_blocks = " << max_blocks << "\n";
    // nblocks = std::min(6*max_blocks, nblocks);
  
    // nblocks = 640;
    // std::cout << "CUDA pattern listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

    size_t list_size = nblocks * per_block_vlist_size;
    std::cout << "frontier list size: " << 1.0 * list_size / (1024*1024) << " MB\n";
    vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

    size_t bitmap_size = nblocks * nwarps * n_bitmaps * bitmap.bigset_bucket_num / BITMAP64_WIDTH * sizeof(bitmap64_Type);
    std::cout << "lut rows size: " << 1.0 * bitmap_size/(1024*1024) << " MB\n";
    bitmap64_Type *frontier_bitmap; // each thread has lut rows to store midresult of lut compute
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap, bitmap_size));
    CUDA_SAFE_CALL(cudaMemset(frontier_bitmap, 0x0, bitmap_size));

    // ideal case: transform all CSR neighbor list to adjacency bitmap
    double lut_gpu_mem = (double)(nv + 31) / 32 * nv * sizeof(vidType) / 1024.0 / 1024.0;
    std::cout << "lut_gpu_mem: " << lut_gpu_mem << " MB\n";

    GPUTimer gputimer;
    gputimer.start();
    if (k == 2) {
        if (algo == "bitmap_bigset_opt") {
            // xgminer_bitmap_bigset_opt_P2_edge_induced<BUCKET_NUM><<<nblocks, nthreads>>>(gg, 
            //                                                     frontier_list,
            //                                                     frontier_bitmap, 
            //                                                     bitmap, 
            //                                                     bitmap.bigset_bucket_num / BITMAP64_WIDTH,
            //                                                     md, 
            //                                                     d_counts);
            xgminer_bitmap_bigset_opt_P2_vertex_induced<BUCKET_NUM><<<nblocks, nthreads>>>(gg, 
                                                                    frontier_list,
                                                                    frontier_bitmap, 
                                                                    bitmap, 
                                                                    UP_DIV(bitmap.bigset_bucket_num, BITMAP64_WIDTH),
                                                                    md, 
                                                                    d_counts);
        }
        if (algo == "ideal_bitmap_test") {
            P2_GM_LUT_block_ideal_test<<<nblocks, nthreads>>>(gg, 
                                                            bitmap.d_bitmap_all);
        }
    }
    gputimer.end_with_sync();

    CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_counts, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < npatterns; i ++) total[i] = h_counts[i];
    
    total_time[0] += gputimer.elapsed();

    if (algo == "bitmap_bigset_opt") {
        std::cout << "P" << k << "[bitmap_bigset_opt] = " << gputimer.elapsed() / 1000 << " s\n";
    }
    CUDA_SAFE_CALL(cudaFree(d_counts));
}
