#define __N_LISTS1 2
#define __N_BITMAPS1 1


__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P1_GM_LUT_block_workload_test(vidType begin, vidType end, /*add begin, end!!!*/
                            vidType *vid_list,/*Add vid_list*/
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
    int block_id    = blockIdx.x; /*add blockid!!!*/
    int num_warps   = WARPS_PER_BLOCK * gridDim.x;
    int warp_id     = thread_id / WARP_SIZE;
    int warp_lane   = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int num_blocks  = gridDim.x;/*add num_blocks!!!*/
    AccType count = 0;
    // meta
    StorageMeta meta;
    meta.lut = LUTs.getEmptyLUT(block_id);/*warp_id->block_id!!!*/
    meta.base = vlists;
    meta.base_size = list_size;
    meta.bitmap_base = bitmaps;
    meta.bitmap_base_size = bitmap_size;
    meta.nv = end;
    meta.slot_size = max_deg;
    meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
    meta.capacity = 2;
    meta.bitmap_capacity = 1;
    meta.global_warp_id = warp_id;
    meta.local_warp_id = warp_lane;
    __syncwarp();

    // BEGIN OF CODEGEN
    auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
    for(vidType v0_idx = begin + block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){ /*add begin end, block_id!!!*/
        // auto v0 = candidate_v0[v0_idx];
        auto v0 = vid_list[v0_idx];
        auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0);/*Add vid_list*/     
        __build_LUT_block_test(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), workload); /*use block!!!*/
        __syncthreads(); /*syncthread!!!*/
        for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARPS_PER_BLOCK) { /*add warp_lane!!!*/
            auto v1 = __build_vid_from_vidx(g, meta, v1_idx);
            __build_index_from_vmap_test(g, meta, 
                        __get_vmap_from_lut_vid_limit_test(g, meta, /*idx_id=*/v1_idx, 
                                        /*connected=*/false, /*upper_bound=*/v1, workload), 
                        /*slot_id=*/1, workload);
            auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/1);
            for(vidType v2_idx_idx = thread_lane; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx += WARP_SIZE){
                auto v2_idx = candidate_v2_idx[v2_idx_idx];
                count += __difference_num(
                            __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/false, /*upper_bound=*/-1), 
                            __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), 
                            /*upper_bound=*/v2_idx);
                workload[threadIdx.x + block_id * blockDim.x] += (v2_idx / BITMAP_WIDTH + 1);
            }
        }
        __syncthreads(); /*syncthread!!!*/
    }
    // END OF CODEGEN
    workload[threadIdx.x + block_id * blockDim.x] += 1;
    atomicAdd(&counter[0], count);
}
