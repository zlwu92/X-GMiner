#define __N_LISTS3 2
#define __N_BITMAPS3 1

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_workload_test(eidType ne, 
                    GraphGPU g, 
                    vidType *vlists,
                    bitmapType* bitmaps,
                    vidType max_deg,
                    AccType *counter,
                    LUTManager<> LUTs,
                    vidType* workload){
    
    __shared__ vidType list_size[WARPS_PER_BLOCK * 2];
    __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int num_warps   = WARPS_PER_BLOCK * gridDim.x;
    int warp_id     = thread_id / WARP_SIZE;
    int warp_lane   = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    AccType count = 0;
    // meta
    StorageMeta meta;
    meta.lut = LUTs.getEmptyLUT(warp_id);
    meta.base = vlists;
    meta.base_size = list_size;
    meta.bitmap_base = bitmaps;
    meta.bitmap_base_size = bitmap_size;
    meta.slot_size = max_deg;
    meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
    meta.capacity = 2;
    meta.bitmap_capacity = 1;
    meta.global_warp_id = warp_id;
    meta.local_warp_id = warp_lane;
    __syncwarp();

    // BEGIN OF CODEGEN
    for(eidType eid = warp_id; eid < ne; eid += num_warps){
        auto v0 = g.get_src(eid);
        auto v1 = g.get_dst(eid);
        __intersect_test(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), 
                        __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0, workload);
        auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
        for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
            auto v2 = candidate_v2[v2_idx];
            count += __difference_num_test(
                            __get_vlist_from_heap(g, meta, /*slot_id=*/0), 
                            __get_vlist_from_graph(g, meta, /*vid=*/v2), 
                            /*upper_bound=*/v2, workload);
        }
    }
    
    // END OF CODEGEN
    workload[threadIdx.x + blockIdx.x * blockDim.x] += 1;
    atomicAdd(&counter[0], count);
}


__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_vert_parallel_test(eidType ne, 
                    GraphGPU g, 
                    vidType *vlists,
                    bitmapType* bitmaps,
                    vidType max_deg,
                    AccType *counter,
                    LUTManager<> LUTs,
                    vidType* workload){
    
    __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
    __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int num_warps   = WARPS_PER_BLOCK * gridDim.x;
    int warp_id     = thread_id / WARP_SIZE;
    int warp_lane   = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    
    int block_id    = blockIdx.x;
    int num_blocks  = gridDim.x;
    AccType count = 0;
    // meta
    StorageMeta meta;
    // meta.lut = LUTs.getEmptyLUT(warp_id);
    meta.lut = LUTs.getEmptyLUT(block_id);
    meta.base = vlists;
    meta.base_size = list_size;
    meta.bitmap_base = bitmaps;
    meta.bitmap_base_size = bitmap_size;
    meta.nv = g.V();
    meta.slot_size = max_deg;
    meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
    meta.capacity = 3;
    meta.bitmap_capacity = 1;
    meta.global_warp_id = warp_id;
    meta.local_warp_id = warp_lane;
    __syncwarp();

    // BEGIN OF CODEGEN
    auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
    for(vidType v0_idx = block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){
        auto v0 = v0_idx;
        auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);
        for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARPS_PER_BLOCK) {
            auto v1 = candidate_v1[v1_idx];
            // __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), 
            //                 __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
            __intersect_test(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), 
                            __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0, workload);
            auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
            for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx++){
                auto v2 = candidate_v2[v2_idx];
                // count += __difference_num(
                //                         __get_vlist_from_heap(g, meta, /*slot_id=*/0), 
                //                         __get_vlist_from_graph(g, meta, /*vid=*/v2), 
                //                         /*upper_bound=*/v2);
                count += __difference_num_test(
                                        __get_vlist_from_heap(g, meta, /*slot_id=*/0), 
                                        __get_vlist_from_graph(g, meta, /*vid=*/v2), 
                                        /*upper_bound=*/v2, workload);
            }
        }

        __syncthreads();
    }
    
    // END OF CODEGEN
    workload[threadIdx.x + blockIdx.x * blockDim.x] += 1;
    atomicAdd(&counter[0], count);
}