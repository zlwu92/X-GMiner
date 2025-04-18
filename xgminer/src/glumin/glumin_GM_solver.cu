#include "../../include/glumin.h"
#include <cub/cub.cuh>

#include "../../include/graph_v2_gpu.h"
#include "../../include/glumin/operations.cuh"
#include "../../include/cuda_utils/cuda_launch_config.hpp"
#include "../../include/glumin/codegen_LUT.cuh"
#include "../../include/glumin/codegen_utils.cuh"
#include "../../include/glumin/timer.h"
#include "../../include/glumin/binary_encode.h"
#define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "GM_LUT.cuh"
#include "GM_build_LUT.cuh"
#include "GM_LUT_deep.cuh"
#include "GM_BS_vertex.cuh"
#include "GM_BS_edge.cuh"

#define BLK_SZ BLOCK_SIZE
#include "GM_kernels/clique4_warp_edge.cuh"
#include "GM_kernels/clique5_warp_edge.cuh"
#include "GM_kernels/clique6_warp_edge.cuh"
#include "GM_kernels/clique7_warp_edge.cuh"


#include "GM_LUT_kernels/clique4_warp_vertex_subgraph.cuh"
#include "GM_LUT_kernels/clique5_warp_edge_subgraph.cuh"
#include "GM_LUT_kernels/clique6_warp_edge_subgraph.cuh"
#include "GM_LUT_kernels/clique7_warp_edge_subgraph.cuh"

#include "GM_LUT_kernels/P2_profile.cuh"
#include "GM_LUT_kernels/P3_profile.cuh"
#include "GM_LUT_kernels/P1_profile.cuh"
#include "GM_LUT_kernels/P6_profile.cuh"
#include "GM_kernels/P2_profile.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

__global__ __forceinline__ void clear(AccType *accumulators) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    accumulators[i] = 0;
}


void GLUMIN::PatternSolver_on_G2Miner(Graph_V2& g) {
    LOG_INFO("PatternSolver_on_G2Miner without LUT");
    // Graph_V2 g(prefix, use_dag);
    // g = Graph_V2(prefix, use_dag);
    // g.print_meta_data();
    // g = std::make_unique<Graph_V2>(prefix, use_dag);
    
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

#if 1
    size_t npatterns = 3;
    AccType *h_counts = (AccType *)malloc(sizeof(AccType) * npatterns);
    for (int i = 0; i < npatterns; i++) h_counts[i] = 0;
    AccType *d_counts;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_counts, sizeof(AccType) * npatterns));
    clear<<<1, npatterns>>>(d_counts);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  
    size_t nwarps = WARPS_PER_BLOCK;
    size_t n_lists;
    size_t n_bitmaps;

    n_lists = 5;
    n_bitmaps = 1;

    vidType switch_lut = 1;
    switch_lut = Select_func(nv, ne, md);

    size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
    size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM;
    if (k == 1){
      max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
      std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    }
    else {
      max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
      std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    } 
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

    nblocks = std::min(6*max_blocks, nblocks);

    nblocks = 640;
    std::cout << "CUDA pattern listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t list_size = nblocks * per_block_vlist_size;
    std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
    vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

    size_t bitmap_size = nblocks * per_block_bitmap_size;
    std::cout << "lut rows size: " << bitmap_size/(1024*1024) << " MB\n";
    bitmapType *frontier_bitmap; // each thread has lut rows to store midresult of lut compute
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap, bitmap_size));

    LUTManager<> lut_manager(nblocks * nwarps, WARP_LIMIT, WARP_LIMIT, true); 
    std::cout << "nblocks*nwarps = " << nblocks * nwarps << " WARP_LIMIT = " << WARP_LIMIT << "\n";
    // split vertex tasks
    std::vector<vidType> vid_warp, vid_block, vid_global;

    for (int vid = 0; vid < nv; ++vid) {
      auto degree = g.get_degree(vid);
      if (degree <= WARP_LIMIT) {
          vid_warp.push_back(vid);
      } else if (degree <= BLOCK_LIMIT) {
          vid_block.push_back(vid);
      } else {
          vid_global.push_back(vid);
      }
    }
    vidType vid_warp_size = vid_warp.size();
    vidType vid_block_size = vid_block.size();
    vidType vid_global_size = vid_global.size();
    // std::cout << "warp_task: " << vid_warp_size << " block_task: " << vid_block_size << " global_task: " << vid_global_size << "\n";
    vidType *d_vid_warp;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_warp, vid_warp_size * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_vid_warp, vid_warp.data(), vid_warp_size * sizeof(vidType), cudaMemcpyHostToDevice));

    vidType *d_vid_block;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_block, vid_block_size * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_vid_block, vid_block.data(), vid_block_size * sizeof(vidType), cudaMemcpyHostToDevice));

    Timer t;
    t.Start();
    // G2Miner
    if (k == 1){
      std::cout << "P1 Run G2Miner\n";
      P1_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 2){
      std::cout << "P2 Run G2Miner\n";
      P2_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 3){
      std::cout << "P3 Run G2Miner\n";
      std::cout << "nblocks = " << nblocks << " nthreads = " << nthreads << " ne = " << ne << " md = " << md << "\n";
      P3_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 6){
      std::cout << "P6 Run G2Miner\n";
      P6_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 7){
      std::cout << "P7 Run G2Miner\n";
      P7_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 7){
      std::cout << "P7 Run G2Miner\n";
      P7_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 8){
      std::cout << "P8 Run G2Miner\n";
      P8_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 9){
      std::cout << "P9 Run G2Miner\n";
      P9_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 10){
      std::cout << "P10 Run G2Miner\n";
      P10_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 11){
      std::cout << "P11 Run G2Miner\n";
      P11_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 12){
      std::cout << "P12 Run G2Miner\n";
      P12_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 13){
      std::cout << "P13 Run G2Miner\n";
      P13_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 14){
      std::cout << "P14 Run G2Miner\n";
      P14_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 15){
      std::cout << "P15 Run G2Miner\n";
      P15_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 16){
      std::cout << "P16 Run G2Miner\n";
      P16_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 17){
      std::cout << "P17 Run G2Miner\n";
      P17_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }              
    else if (k == 18){
      std::cout << "P18 Run G2Miner\n";
      P18_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 19){
      std::cout << "P19 Run G2Miner\n";
      P19_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 20){
      std::cout << "P20 Run G2Miner\n";
      P20_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 21){
      std::cout << "P21 Run G2Miner\n";
      P21_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else if (k == 22){
      std::cout << "P22 Run G2Miner\n";
      P22_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
    }
    else {
      LOG_ERROR("Unsupported pattern: " + std::to_string(k) + " for G2Miner without LUT.");
    }
    CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_counts, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < npatterns; i ++) total[i] = h_counts[i];
    t.Stop();

    total_time[0] += t.Seconds();
    std::cout << "runtime [G2Miner] = " << t.Seconds() << " sec\n";
    CUDA_SAFE_CALL(cudaFree(d_counts));
#endif
}

#if 1
void GLUMIN::PatternSolver_LUT_on_G2Miner(Graph_V2& g) {
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
    clear<<<1, npatterns>>>(d_counts);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  
    size_t nwarps = WARPS_PER_BLOCK;
    size_t n_lists;
    size_t n_bitmaps;
    n_lists = 7;
    n_bitmaps = 3;

    vidType switch_lut = use_lut;//1;
    // switch_lut = Select_func(nv, ne, md);
    std::cout << "switch_lut = " << switch_lut << "\n";
    size_t per_block_vlist_size = nwarps * n_lists * size_t(md) * sizeof(vidType);
    size_t per_block_bitmap_size = nwarps * n_bitmaps * ((size_t(md) + BITMAP_WIDTH-1)/BITMAP_WIDTH) * sizeof(vidType);

    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize*0.9 - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM;
    if (k == 1){
      max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
      std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    }
    else {
      max_blocks_per_SM = maximum_residency(GM_LUT_warp, nthreads, 0);
      std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    } 
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
    std::cout << "max_blocks = " << max_blocks << "\n";
    nblocks = std::min(6*max_blocks, nblocks);

    nblocks = 640;
    std::cout << "CUDA pattern listing (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t list_size = nblocks * per_block_vlist_size;
    std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
    vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

    size_t bitmap_size = nblocks * per_block_bitmap_size;
    std::cout << "lut rows size: " << bitmap_size/(1024*1024) << " MB\n";
    bitmapType *frontier_bitmap; // each thread has lut rows to store midresult of lut compute
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_bitmap, bitmap_size));

    LUTManager<> lut_manager(nblocks * nwarps, WARP_LIMIT, WARP_LIMIT, true); 
    std::cout << "WARP_LIMIT: " << WARP_LIMIT << ", BLOCK_LIMIT: " << BLOCK_LIMIT << "\n";
    // split vertex tasks
    std::vector<vidType> vid_warp, vid_block, vid_global;

    for (int vid = 0; vid < nv; ++vid) {
      auto degree = g.get_degree(vid);
      if (degree <= WARP_LIMIT) {
          vid_warp.push_back(vid);
      } else if (degree <= BLOCK_LIMIT) {
          vid_block.push_back(vid);
      } else {
          vid_global.push_back(vid);
      }
    }
    vidType vid_warp_size = vid_warp.size();
    vidType vid_block_size = vid_block.size();
    vidType vid_global_size = vid_global.size();
    std::cout << "warp_task: " << vid_warp_size << " block_task: " << vid_block_size << " global_task: " << vid_global_size << "\n";
    vidType *d_vid_warp;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_warp, vid_warp_size * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_vid_warp, vid_warp.data(), vid_warp_size * sizeof(vidType), cudaMemcpyHostToDevice));

    vidType *d_vid_block;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_vid_block, vid_block_size * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_vid_block, vid_block.data(), vid_block_size * sizeof(vidType), cudaMemcpyHostToDevice));

    std::ofstream out("/home/zlwu/workspace/2-graphmining/X-GMiner/results/g2miner_glumin_memory_profiling.csv", std::ios::app);
    // out << nv << "," << ne << "," << md << ",";
    // size_t graphsize = (nv+1)*sizeof(eidType) + 2*ne*sizeof(vidType);
    // out << (double)graphsize / 1024.0 / 1024.0 << ",";
    // out << (double)list_size / 1024.0 / 1024.0 + (double)bitmap_size / 1024.0 / 1024.0 + 
    //       (double)lut_manager.max_LUT_size_ * lut_manager.LUT_num_ * sizeof(vidType) / 1024.0 / 1024.0 << ",";
    // return;

    // vidType* d_work_depth_each_warp;
    // int num_warps = std::min((vidType)nblocks, vid_block_size) * nwarps;//nblocks * WARPS_PER_BLOCK;//
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_work_depth_each_warp, num_warps * sizeof(vidType)));
    // CUDA_SAFE_CALL(cudaMemset(d_work_depth_each_warp, 0, num_warps * sizeof(vidType)));
    // std::vector<vidType> work_depth_each_warp(num_warps);

    // ideal case: transform all CSR neighbor list to adjacency bitmap
    // double lut_gpu_mem = (double)(nv + 31) / 32 * nv * sizeof(vidType) / 1024.0 / 1024.0;
    // std::cout << "lut_gpu_mem: " << lut_gpu_mem << " MB\n";

    if (switch_lut) {
      lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
      // lut_manager.recreate(nblocks, md, md, true);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    std::vector<vidType> workload(nblocks * nthreads, 0);
    std::cout << "workload size:" << nblocks * nthreads << ", " << workload.size() << ", " << workload.capacity() << "\n";
    vidType *d_workload;
    if (prof_workload) {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_workload, nblocks * nthreads * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemset(d_workload, 0, nblocks * nthreads * sizeof(vidType)));
    }
    else {
      d_workload = nullptr;
    }
    std::vector<vidType> edgecheck(nblocks * nthreads);
    vidType *d_edgecheck, *d_edgecheck2; 
    AccType* d_edgecheck_cnt;
    uint64_t edgecheck_size = 1343125092;
    std::vector<vidType> edgecheck2;//
    if (prof_edgecheck) {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_edgecheck, nblocks * nthreads * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemset(d_edgecheck, 0, nblocks * nthreads * sizeof(vidType)));

      edgecheck2.resize(edgecheck_size);
      CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_edgecheck2, edgecheck_size * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemset(d_edgecheck2, 0, edgecheck_size * sizeof(vidType)));
      thrust::fill(thrust::device, d_edgecheck2, d_edgecheck2 + edgecheck_size, -1);
      std::cout << "edgecheck2 size: " << edgecheck_size * sizeof(vidType) / 1024.0 / 1024.0 << " MB\n";

      CUDA_SAFE_CALL(cudaMalloc((void **)&d_edgecheck_cnt, sizeof(AccType)));
      CUDA_SAFE_CALL(cudaMemset(d_edgecheck_cnt, 0, sizeof(AccType)));
    }
    else {
      d_edgecheck = nullptr;
    }

    std::cout << __LINE__ << "length = " << nblocks * nthreads << "\n";
    float time[3];
    Timer t;
    t.Start();
    cudaEvent_t e1, e2, e3, e4;
    GPUTimer t0;
    float elapsedTime;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);
    cudaEventRecord(e1, 0);
    // G2Miner + LUT
    if (k == 1) {
      std::cout << "P1 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P1_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          // lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          // t0.start();
          if (prof_workload) {
            PRINT_GREEN("P1_GM_LUT_block_workload_test");
            P1_GM_LUT_block_workload_test<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                  frontier_list, frontier_bitmap, md, d_counts, lut_manager, d_workload); 
          }
          else {
            P1_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                  frontier_list, frontier_bitmap, md, d_counts, lut_manager);
          }
          // t0.end_with_sync();
          // time[0] = t0.elapsed() / 1000;
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          // t0.start();
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P1_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
          // t0.end_with_sync();
          // time[2] = t0.elapsed() / 1000;
        }
      }
      else {
        P1_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 2) {
      PRINT_GREEN("P2 G2Miner + LUT");
      if (switch_lut){
        if (WARP_LIMIT != 0) {
          std::cout << __LINE__ << " vid_warp_size: " << vid_warp_size << ", nthreads: " << nthreads << "\n";
          P2_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_block_size) {
          std::cout << __LINE__ << "length = " << nblocks * nthreads << "\n";
          std::cout << __LINE__ << " vid_block_size: " << vid_block_size << ", nthreads: " << nthreads << "\n";
          // lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          // t0.start();
          if (prof_workload) {
            PRINT_GREEN("P2_GM_LUT_block_workload_test");
            P2_GM_LUT_block_workload_test<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                    frontier_list, frontier_bitmap, md, d_counts, lut_manager, 
                                                    d_workload
                                                  );
          } else if (prof_edgecheck) {
            PRINT_GREEN("P2_GM_LUT_block_profile_edgecheck_only");
            // P2_GM_LUT_block_edgecheck_test<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
            //                                         frontier_list, frontier_bitmap, md, d_counts, lut_manager, 
            //                                         d_edgecheck, d_edgecheck2, d_edgecheck_cnt
            //                                       );
            P2_GM_LUT_block_profile_edgecheck_only<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                      frontier_list, frontier_bitmap, md, d_counts, lut_manager, 
                                                      d_edgecheck2, d_edgecheck_cnt
                                                    );
          }
          else {
            P2_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                  frontier_list, frontier_bitmap, md, d_counts, lut_manager);
          }
          // t0.end_with_sync();
          // time[0] = t0.elapsed() / 1000;
        }
        if (vid_global_size){
          std::cout << __LINE__ << " vid_global_size: " << vid_global_size << "\n";
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          // t0.start();
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P2_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
          // t0.end_with_sync();
          // time[2] = t0.elapsed() / 1000;
        }
      
      }
      else {
        if (prof_workload) {
          PRINT_GREEN("P2_GM_workload_test");
          if (use_vert_para) {
            PRINT_GREEN("P2_GM_vert_parallel_test");
            P2_GM_vert_parallel_test<<<nblocks, nthreads>>>(ne, gg, 
                                    frontier_list, frontier_bitmap, md, d_counts, lut_manager, d_workload);
          } else {
            P2_GM_workload_test<<<nblocks, nthreads>>>(ne, gg, 
              frontier_list, frontier_bitmap, md, d_counts, lut_manager, d_workload);
          }
        }
        else {
          P2_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
      }
    }
    else if (k == 3) {
      std::cout << "P3 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P3_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          // lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          // t0.start();
          if (prof_workload) {
            PRINT_GREEN("P3_GM_LUT_block_workload_test");
            P3_GM_LUT_block_workload_test<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                  frontier_list, frontier_bitmap, md, d_counts, lut_manager, d_workload);
          }
          else{
            P3_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                                  frontier_list, frontier_bitmap, md, d_counts, lut_manager);
          }
          // t0.end_with_sync();
          // time[0] = t0.elapsed() / 1000;
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          // t0.start();
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P3_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
          // t0.end_with_sync();
          // time[2] = t0.elapsed() / 1000;
        }
      }
      else {
        P3_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 6) {
      std::cout << "P6 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P6_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          // lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          if (prof_workload) {
            PRINT_GREEN("P6_GM_LUT_block_workload_test");
            P6_GM_LUT_block_workload_test<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                            frontier_list, frontier_bitmap, md, d_counts, lut_manager, d_workload);
          }
          else {
            P6_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, 
                                            frontier_list, frontier_bitmap, md, d_counts, lut_manager);
            }
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P6_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P6_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 7) {
      std::cout << "P7 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P7_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P7_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P7_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P7_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 8) {
      std::cout << "P8 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P8_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P8_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P8_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P8_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 9) {
      std::cout << "P9 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P9_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P9_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P9_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P9_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 10) {
      std::cout << "P10 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P10_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P10_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P10_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P10_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 11) {
      std::cout << "P11 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P11_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P11_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P11_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P11_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 12) {
      std::cout << "P12 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P12_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P12_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P12_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P12_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 13) {
      std::cout << "P13 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P13_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P13_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P13_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P13_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 14) {
      std::cout << "P14 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P14_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P14_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P14_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P14_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 15) {
      std::cout << "P15 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P15_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P15_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P15_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P15_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 16) {
      std::cout << "P16 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P16_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P16_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P16_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P16_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 17) {
      std::cout << "P17 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P17_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P17_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P17_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P17_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 18) {
      std::cout << "P18 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P18_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P18_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P18_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P18_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 19) {
      std::cout << "P19 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P19_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P19_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P19_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P19_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 20) {
      std::cout << "P20 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P20_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P20_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P20_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P20_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 21) {
      std::cout << "P21 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P21_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P21_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P21_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P21_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else if (k == 22) {
      std::cout << "P22 G2Miner + LUT\n";
      if (switch_lut){
        if (WARP_LIMIT != 0) P22_GM_LUT_warp<<<nblocks, nthreads>>>(0, vid_warp_size, d_vid_warp, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        if (vid_block_size) {
          lut_manager.recreate(nblocks, BLOCK_LIMIT, BLOCK_LIMIT, true);
          P22_GM_LUT_block<<<nblocks, nthreads>>>(0, vid_block_size, d_vid_block, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
        }
        if (vid_global_size){
          lut_manager.recreate(1, md, md, true);
          nblocks = BLOCK_GROUP;
          for (vidType i = 0; i < vid_global_size; i++) {
            vidType task_id = vid_global[i];
            lut_manager.update_para(1, g.get_degree(task_id), g.get_degree(task_id), true);
            GM_build_LUT<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
            P22_GM_LUT_global<<<nblocks, nthreads>>>(0, nv, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager, task_id);
          }
        }
      }
      else {
        P22_GM<<<nblocks, nthreads>>>(ne, gg, frontier_list, frontier_bitmap, md, d_counts, lut_manager);
      }
    }
    else {
      std::cout << "Not supported right now\n";
    }
    cudaEventRecord(e2, 0);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&elapsedTime, e1, e2);
    t.Stop();
    CUDA_SAFE_CALL(cudaMemcpy(h_counts, d_counts, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < npatterns; i ++) total[i] = h_counts[i];

    // CUDA_SAFE_CALL(cudaMemcpy(work_depth_each_warp.data(), d_work_depth_each_warp, num_warps * sizeof(vidType), cudaMemcpyDeviceToHost));
    out.close();
    out.open("/data-ssd/home/zhenlin/workspace/graphmining/X-GMiner/results/work_depth_per_warp_glumin_g2miner_lut.csv", std::ios::app);
    // out << "P" << k << "_LUT,";
    // for (size_t i = 0; i < work_depth_each_warp.size(); i++) {
    //   out << work_depth_each_warp[i];
    //   if (i < work_depth_each_warp.size() - 1)  out << ",";
    // }
    // out << "\n";
    out.close();
    std::cout << __LINE__ << "length = " << nblocks * nthreads << "\n";
  
    out.open("../results/prof_glumin_LUT_kernel_time_percentage.csv", std::ios::app);
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, 0));
    std::string prop_name(prop.name);
    if (switch_lut) {
      if (prop_name.find("3090") != std::string::npos) {
        out << "P" << k << ",3090,";
        out << time[0] << "," << time[2] << "," << t.Seconds() << ",";
      } else if (prop_name.find("6000") != std::string::npos) {
        out << "P" << k << ",ada6000,";
        std::cout << "here!!!\n";
        out << time[0] << "," << time[2] << "," << t.Seconds() << ",";
      }
    } else {
      if (prop_name.find("3090") != std::string::npos) {
        // out << "3090,";
        out << elapsedTime / 1000.0 << "\n";
      } else if (prop_name.find("6000") != std::string::npos) {
        // out << "ada6000,";
        std::cout << "here~~~\n";
        out << elapsedTime / 1000.0 << "\n";
      }
    }

    if (switch_lut) {
      std::cout << "runtime [G2Miner + LUT] = " << t.Seconds() << " sec\n";
      total_time[0] += t.Seconds();
    }
    else {
      std::cout << "runtime [G2Miner] = " << elapsedTime / 1000.0 << " sec\n";
      total_time[0] += elapsedTime / 1000.0;
    }
    out.close();
    
    if (prof_workload) {
      std::string file = "../results/prof_glumin_kernel_workload_" + data_name + ".csv";
      out.open(file);
      std::cout << __LINE__ << "length = " << workload.size() << "\n";
      CUDA_SAFE_CALL(cudaMemcpy(workload.data(), d_workload, sizeof(vidType) * workload.size(), cudaMemcpyDeviceToHost));
      for (int i = 0; i < workload.size(); i++) {
        out << workload[i] << "\n";
        // if (workload[i] == 0) printf("zero workload!\n");
      }
      out.close();
      unsigned long total = std::accumulate(workload.begin(), workload.end(), 0LL);
      auto max = std::max_element(workload.begin(), workload.end());
      auto min = std::min_element(workload.begin(), workload.end());
      std::cout << workload.end() - workload.begin() << ", " << workload.size() << "\n";
      float max_min = (float) *max / *min;
      out.open("../results/prof_glumin_kernel_workload.csv", std::ios::app);
      std::cout << total << "," << *max << "," << *min << "," << (float) total / workload.size() << "\n";
      out << total << "," << max_min << "," << (float) total / workload.size();
    }
    if (prof_edgecheck) {
      std::string file = "../results/prof_glumin_kernel_edgecheck_" + data_name + ".csv";
      out.open(file);
      CUDA_SAFE_CALL(cudaMemcpy(edgecheck.data(), d_edgecheck, sizeof(vidType) * edgecheck.size(), cudaMemcpyDeviceToHost));
      for (int i = 0; i < nblocks * nthreads; i++) {
        out << edgecheck[i] << "\n";
      }
      out.close();
      // calculate max of edgecheck
      int max_edgecheck = *std::max_element(edgecheck.begin(), edgecheck.end());
      int total = std::accumulate(edgecheck.begin(), edgecheck.end(), 0);
      std::cout << "max_edgecheck = " << max_edgecheck << "\n";
      std::cout << "max memory = " << 1.0 * max_edgecheck * edgecheck.size() * sizeof(vidType) / 1024 / 1024 / 1024 << " GB\n";
      std::cout << "real memory = " << 1.0 * total * sizeof(vidType) / 1024 / 1024 / 1024 << " GB\n";
    
      AccType edgecheck_cnt;
      CUDA_SAFE_CALL(cudaMemcpy(&edgecheck_cnt, d_edgecheck_cnt, sizeof(AccType), cudaMemcpyDeviceToHost));
      std::cout << "edgecheck_cnt = " << edgecheck_cnt << "\n";

      CUDA_SAFE_CALL(cudaMemcpy(edgecheck2.data(), d_edgecheck2, sizeof(vidType) * edgecheck_cnt, cudaMemcpyDeviceToHost));
      out.open("../results/prof_glumin_kernel_edgecheck2_" + data_name + ".csv");
      std::map<std::pair<int, int>, int> pair_counts;
      for (int i = 0; i < edgecheck_cnt; i+=2) {
        // out << edgecheck2[i] << "," << edgecheck2[i+1] << "\n";
        pair_counts[{edgecheck2[i], edgecheck2[i + 1]}]++;
      }
      // 打印每对(pair)出现的次数
      // for (const auto& pair : pair_counts) {
      //   std::cout << "Pair {" << pair.first.first << ", " << pair.first.second << "}: " << pair.second << " times" << std::endl;
      // }
      int unique_pairs = pair_counts.size();
      std::cout << "Total number of unique pairs: " << unique_pairs << std::endl;
      // 统计总次数、最大次数和最小次数
      int total_count = 0;
      int max_count = std::numeric_limits<int>::min();
      int min_count = std::numeric_limits<int>::max();
      
      for (const auto& pair : pair_counts) {
          total_count += pair.second;
          max_count = std::max(max_count, pair.second);
          min_count = std::min(min_count, pair.second);
      }
      std::cout << "sum: " << total_count << ", max: " << max_count << ", min: " << min_count << "\n";
      out.close();
    }

    CUDA_SAFE_CALL(cudaFree(d_counts));
}

void GLUMIN::CliqueSolver_LUT_on_G2Miner(Graph_V2& g) {
    LOG_INFO("CliqueSolver_on_G2Miner + LUT");
    int k = k_num;
    assert(k > 3);
    size_t memsize = print_device_info(0);
    vidType nv = g.num_vertices();
    eidType ne = g.num_edges();
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
      std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
              << " GB, graph_mem = " << mem_graph/1024/1024 << " MB\n";
    if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

    GraphGPU gg(g);
    gg.init_edgelist(g);
    size_t nwarps = WARPS_PER_BLOCK;
    size_t nthreads = BLK_SZ;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;

    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM = maximum_residency(clique4_warp_vertex_subgraph, nthreads, 0);
    double clock_rate = deviceProp.clockRate;

    if (k==5) max_blocks_per_SM = maximum_residency(clique5_warp_edge_subgraph, nthreads, 0);
    if (k==6) max_blocks_per_SM = maximum_residency(clique6_warp_edge_subgraph, nthreads, 0);
    if (k==7) max_blocks_per_SM = maximum_residency(clique7_warp_edge_subgraph, nthreads, 0);
    std::cout << k << "-clique max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

    nblocks = std::min(max_blocks, nblocks);
    
    std::cout << "CUDA " << k << "-clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t list_size = nblocks * per_block_vlist_size;
    std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
    vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

    AccType h_total = 0, *d_total;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BinaryEncode<> sub_graph(nblocks * nwarps, md, md);
    printf("Sum Warp: %d!!!\n", nblocks * nwarps);

    Timer t;
    t.Start();
    if (k == 4) {
      std::cout << "P4 G2Miner + LUT\n";
      clique4_warp_vertex_subgraph<<<nblocks, nthreads>>>(nv, gg, frontier_list, md, d_total, sub_graph);
    } else if (k == 5) {
      std::cout << "P5 G2Miner + LUT\n";
      clique5_warp_edge_subgraph<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph);
    } else if (k == 6) {
      std::cout << "P23 G2Miner + LUT\n";
      clique6_warp_edge_subgraph<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph);
    } else if (k == 7) {
      std::cout << "P24 G2Miner + LUT\n";
      clique7_warp_edge_subgraph<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph);
    } else {
      LOG_ERROR("Unsupported pattern: " + std::to_string(k) + " for G2Miner with LUT.");
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();

    total_time[0] += t.Seconds();
    std::cout << "runtime [G2Miner + LUT] = " << t.Seconds() << " sec\n";
    CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
    total[0] = h_total;
    CUDA_SAFE_CALL(cudaFree(d_total));
}

#endif

#if 1
void GLUMIN::CliqueSolver_on_G2Miner(Graph_V2& g) {
    LOG_INFO("CliqueSolver_on_G2Miner without LUT");
    int k = k_num;
    assert(k > 3);
    size_t memsize = print_device_info(0);
    vidType nv = g.num_vertices();
    eidType ne = g.num_edges();
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
    std::cout << "GPU_total_mem = " << memsize/1024/1024/1024
              << " GB, graph_mem = " << mem_graph/1024/1024 << " MB\n";
    if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

    GraphGPU gg(g);
    gg.init_edgelist(g);
    size_t nwarps = WARPS_PER_BLOCK;
    size_t nthreads = BLK_SZ;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    size_t per_block_vlist_size = nwarps * size_t(k-3) * size_t(md) * sizeof(vidType);
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;

    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM = maximum_residency(clique4_warp_edge, nthreads, 0);

    if (k==5) max_blocks_per_SM = maximum_residency(clique5_warp_edge, nthreads, 0);  
    if (k==6) max_blocks_per_SM = maximum_residency(clique6_warp_edge, nthreads, 0);
    if (k==7) max_blocks_per_SM = maximum_residency(clique7_warp_edge, nthreads, 0);
    std::cout << k << "-clique max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

    nblocks = std::min(16*max_blocks, nblocks);  
    
    std::cout << "CUDA clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t list_size = nblocks * per_block_vlist_size;
    std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
    vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

    AccType h_total = 0, *d_total;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    Timer t;
    t.Start();
    if (k == 4) {
      std::cout << "P4 Run G2Miner\n";
      clique4_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else if (k == 5) {
      std::cout << "P5 Run G2Miner\n";
      clique5_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else if (k == 6) {
      std::cout << "P23 Run G2Miner\n";
      clique6_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else if (k == 7) {
      std::cout << "P24 Run G2Miner\n";
      clique7_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else {
      LOG_ERROR("Unsupported pattern: " + std::to_string(k) + " for G2Miner without LUT.");
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();

    total_time[0] = t.Seconds();
    std::cout << "runtime [G2Miner] = " << t.Seconds() << " sec\n";
    CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
    total[0] += h_total;
    CUDA_SAFE_CALL(cudaFree(d_total));
}
#endif
