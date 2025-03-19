#include "../../include/glumin.h"
#include <cub/cub.cuh>

#include "../../include/graph_v2_gpu.h"
#include "../../include/glumin/operations.cuh"
#include "../../include/cuda_utils/cuda_launch_config.hpp"
// #include "../../include/glumin/codegen_LUT.cuh"
// #include "../../include/glumin/codegen_utils.cuh"
#include "../../include/glumin/timer.h"
#include "../../include/glumin/binary_encode.h"
#define FISSION
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#define BLK_SZ BLOCK_SIZE
#include "../../include/glumin/expand_LUT.cuh"

#include "gpu_clique_kernels/clique4_warp_edge.cuh"
#include "gpu_clique_kernels/clique5_warp_edge.cuh"
#include "gpu_clique_kernels/clique6_warp_edge.cuh"
#include "gpu_clique_kernels/clique7_warp_edge.cuh"

#include "gpu_clique_kernels/clique4_warp_vertex_subgraph.cuh"
#include "gpu_clique_kernels/clique5_warp_edge_subgraph.cuh"
#include "gpu_clique_kernels/clique6_warp_edge_subgraph.cuh"
#include "gpu_clique_kernels/clique7_warp_edge_subgraph.cuh"

#include "gpu_clique_kernels/clique5_warp_edge_taskallocate.cuh"
#include "gpu_clique_kernels/clique5_GF.cuh"

#if 1
void GLUMIN::CliqueSolver_on_GM_Clique() {
    LOG_INFO("CliqueSolver_on_GM_Clique without LUT.");
    int k = local_patternId;
    assert(k > 3);
    size_t memsize = print_device_info(0);
    vidType nv = g.num_vertices();
    eidType ne = g.num_edges();
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
    std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
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
    
    std::cout << "CUDA " << k << "-clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
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
        clique4_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else if (k == 5) {
        clique5_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else if (k == 6) {
        clique6_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else if (k == 7) {
        clique7_warp_edge<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total);
    } else {
        LOG_ERROR("Not supported for P" + std::to_string(k) + " for clique solver without LUT on G2Miner.");
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();

    std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
    CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
    total[0] = h_total;
    CUDA_SAFE_CALL(cudaFree(d_total));
}


void GLUMIN::CliqueSolver_LUT_on_GM_Clique() {
    LOG_INFO("CliqueSolver_LUT_on_GM_Clique with LUT.");
    int k = local_patternId;
    assert(k > 3);
    size_t memsize = print_device_info(0);
    vidType nv = g.num_vertices();
    eidType ne = g.num_edges();
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
    std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
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
        clique4_warp_vertex_subgraph<<<nblocks, nthreads>>>(nv, gg, frontier_list, md, d_total, sub_graph);
    } else if (k == 5) {
        std::cout << "4-USE_SUBGRAPH_GPU!!!!!!\n";
        clique5_warp_edge_subgraph<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph);
    } else if (k == 6) {
        std::cout << "USE_SUBGRAPH_GPU_6Clique!!!!!!\n";
        clique6_warp_edge_subgraph<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph);
    } else if (k == 7) {
        std::cout << "USE_SUBGRAPH_GPU_7Clique!!!!!!\n";
        clique7_warp_edge_subgraph<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph);
    } else {
        LOG_ERROR("Not supported for P" + std::to_string(k) + " for clique solver with LUT on G2Miner.");
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();

    std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
    CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
    total[0] = h_total;
    CUDA_SAFE_CALL(cudaFree(d_total));
}


void GLUMIN::CliqueSolver_LUT_on_GF_Clique() {
    LOG_INFO("CliqueSolver_LUT_on_GF_Clique with LUT.");
    int k = local_patternId;
    // assert(k > 3);
    size_t memsize = print_device_info(0);
    vidType nv = g.num_vertices();
    eidType ne = g.num_edges();
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
    std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
    if (memsize < mem_graph) std::cout << "Graph too large. Unified Memory (UM) required\n";

    // CUDA_SAFE_CALL(cudaSetDevice(CUDA_SELECT_DEVICE));
    GraphGPU gg(g);
    gg.init_edgelist(g);
    size_t nwarps = WARPS_PER_BLOCK;
    size_t nthreads = BLK_SZ;
    size_t nblocks = (ne-1)/WARPS_PER_BLOCK+1;
    size_t per_block_vlist_size = nwarps * size_t(4) * size_t(md) * sizeof(vidType);
    if (nblocks > 65536) nblocks = 65536;
    size_t nb = (memsize - mem_graph) / per_block_vlist_size;
    if (nb < nblocks) nblocks = nb;

    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM = maximum_residency(clique5_warp_edge_taskallocate, nthreads, 0);
    double clock_rate = deviceProp.clockRate;

    std::cout << k << "-clique max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;

    nblocks = std::min(max_blocks, nblocks);
    // nblocks = 640;
    
    std::cout << "CUDA " << k << "-clique listing/counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t list_size = nblocks * per_block_vlist_size;
    std::cout << "frontier list size: " << list_size/(1024*1024) << " MB\n";
    vidType *frontier_list; // each warp has (k-3) vertex sets; each set has size of max_degree
    CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

    AccType h_total = 0, *d_total;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));

    int h_allocator, *d_allocator; // Set 0 for No pre allocate, atomicAdd from zero
    h_allocator = nblocks * nwarps;

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_allocator, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_allocator, &h_allocator, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BinaryEncode<> sub_graph(nblocks * nwarps, md, md);
    printf("Sum Warp: %d!!!\n", nblocks * nwarps);

    Timer t;
    t.Start();
    if (k == 1) {
        clique5_GF<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph, d_allocator);
    }
    else if (k == 2) {
        clique5_warp_edge_taskallocate<<<nblocks, nthreads>>>(ne, gg, frontier_list, md, d_total, sub_graph, d_allocator);
    }
    else {
        LOG_ERROR("Not supported for P" + std::to_string(k) + " for clique solver with LUT on GraphFold.");
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();

    std::cout << "runtime [cuda_base] = " << t.Seconds() << " sec\n";
    CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
    total[0] = h_total;
    CUDA_SAFE_CALL(cudaFree(d_total));
    CUDA_SAFE_CALL(cudaFree(d_allocator));
}

#endif