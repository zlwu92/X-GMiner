
template<vidType BUCKET_NUM>
__global__ void __launch_bounds__(BLOCK_SIZE_128, 4)
xgminer_bitmap_bigset_opt_P2_edge_induced(
                                        GraphGPU g, 
                                        vidType *vlists,
                                        bitmap64_Type* frontier_bitmap,
                                        XGMiner_BITMAP<> bitmaps,
                                        vidType bmap_size,
                                        vidType max_deg,
                                        AccType *counter
                                    ){
    
    // __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
    // __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
    // __shared__ typename BlockReduce::TempStorage temp_storage;
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id    = blockIdx.x;
    int num_warps   = WARPS_PER_BLOCK * gridDim.x;
    int warp_id     = thread_id / WARP_SIZE;
    int warp_lane   = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int num_blocks  = gridDim.x;
    AccType count = 0;

    __shared__ int nonzero_bucket_id[WARPS_PER_BLOCK * BUCKET_NUM];

    // meta
    StorageMeta meta;
    // meta.lut = LUTs.getEmptyLUT(block_id);
    meta.base = vlists;
    // meta.base_size = list_size;
    meta.bitmap64_base = frontier_bitmap;
    // meta.bitmap_base_size = bitmap_size;
    meta.nv = g.V();//end;
    meta.slot_size = max_deg;
    meta.bitmap_size = BUCKET_NUM;//(max_deg + WARP_SIZE - 1) / WARP_SIZE;
    meta.capacity = 3; // n_bitmaps
    meta.bitmap_capacity = 1;
    meta.global_warp_id = warp_id;
    meta.local_warp_id = warp_lane;
    __syncwarp();


    auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);

    for(vidType v0_idx = block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){
        auto v0 = v0_idx;//vid_list[v0_idx];
        auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0); // v1 is in N(v0)
        if (v0 == 0) {
        for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARPS_PER_BLOCK) { // 1 2 4 5
            auto v1 = candidate_v1[v1_idx];
            if (v1 > v0) {
                // perform intersection between N(v0) and N(v1)
                auto candidate_v2 = bitmaps.intersection(
                                                        bitmaps.d_bitmaps_ + v0 * bmap_size, 
                                                        bitmaps.d_bitmaps_ + v1 * bmap_size, 
                                                        bmap_size, 
                                                        bmap_size,
                                                        g, v0, v1, v1_idx,
                                                        meta,
                                                        0, 0,
                                                        nonzero_bucket_id,
                                                        BUCKET_NUM
                                                    );
                
                // auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
                if (warp_lane == 0 && thread_lane == 0) {
                    printf("v0: %d, v1: %d\n", v0, v1);
                    // print each bit of output_bucket_bmap[0]
                    // for (int i = 0; i < 64; ++i) {
                    //     printf("%lu", (output_bucket_bmap[0] >> (63-i)) & 1);
                    //     // printf("%lu", (bitmaps.d_bitmaps_[v0 * bmap_size] >> (63-i)) & 1);
                    // }
                    // for (int i = 0; i < bmap_size; i++) {
                    //     // printf("%ld ", output_bucket_bmap[i]);
                    //     // printf("%ld ", bitmaps.d_bitmaps_[v0 * bmap_size + i]);
                    //     // printf("%lu ", bitmaps.d_bitmaps_[v1 * bmap_size + i]);
                    // }
                    // for (int i = 0; i < BUCKET_NUM; i++) {
                    //     printf("%d ", nonzero_bucket_id[warp_lane * BUCKET_NUM + i]);
                    // }
                    // printf("\n");
                    // for (int i = 0; i < candidate_v2.size(); i++) {
                    //     printf("%d ", candidate_v2[i]);
                    // }
                    // printf("\n");
                }
                if (thread_lane == 0)   count += candidate_v2.size() * (candidate_v2.size() - 1) / 2;
                // for(vidType v2_idx = thread_lane; v2_idx < candidate_v2.size(); v2_idx += WARP_SIZE){
                //     auto v2 = candidate_v2[v2_idx];
                //     // if (warp_lane == 0 && thread_lane == 0) {
                //     //     printf("v2: %d\n", v2);
                //     // }
                //     count += __difference_num(__get_vlist_from_heap(g, meta, /*slot_id=*/0), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/v2);
                //     count += __difference_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/v2_idx);
                // }
            }
        }
        }
        __syncthreads();
    }

    atomicAdd(&counter[0], count);
}


template<vidType BUCKET_NUM>
__global__ void //__launch_bounds__(BLOCK_SIZE_128, 4)
xgminer_bitmap_bigset_opt_P2_vertex_induced(
                                        GraphGPU g, 
                                        vidType *vlists,
                                        bitmap64_Type* frontier_bitmap,
                                        XGMiner_BITMAP<> bitmaps,
                                        vidType bmap_size,
                                        vidType max_deg,
                                        AccType *counter
                                    ){
    
    // __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
    // __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
    // __shared__ typename BlockReduce::TempStorage temp_storage;
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id    = blockIdx.x;
    int num_warps   = WARP_PER_BLOCK_128 * gridDim.x;
    int warp_id     = thread_id / WARP_SIZE;
    int warp_lane   = threadIdx.x / WARP_SIZE;
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int num_blocks  = gridDim.x;
    AccType count = 0;

    __shared__ int nonzero_bucket_id[WARP_PER_BLOCK_128 * BUCKET_NUM];

    // meta
    StorageMeta meta;
    // meta.lut = LUTs.getEmptyLUT(block_id);
    meta.base = vlists;
    // meta.base_size = list_size;
    meta.bitmap64_base = frontier_bitmap;
    // meta.bitmap_base_size = bitmap_size;
    meta.nv = g.V();//end;
    meta.slot_size = max_deg;
    meta.bitmap_size = BUCKET_NUM;//(max_deg + WARP_SIZE - 1) / WARP_SIZE;
    meta.capacity = 5; // n_lists
    meta.bitmap_capacity = 3; // n_bitmaps
    meta.global_warp_id = warp_id;
    meta.local_warp_id = warp_lane;
    __syncwarp();


    auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);

    for(vidType v0_idx = block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){
        auto v0 = v0_idx;//vid_list[v0_idx];
        auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0); // v1 is in N(v0)
        // if (v0 == 18) 
        {
        for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARP_PER_BLOCK_128) { // 1 2 4 5
            auto v1 = candidate_v1[v1_idx];
            if (v1 > v0/* && v1_idx == 6*/) {
                // perform intersection between N(v0) and N(v1)
                auto candidate_v2 = bitmaps.intersection(
                                                        bitmaps.d_bitmaps_ + v0 * bmap_size, 
                                                        bitmaps.d_bitmaps_ + v1 * bmap_size, 
                                                        bmap_size, 
                                                        bmap_size,
                                                        g, v0, v1, v1_idx,
                                                        meta,
                                                        0, 0,
                                                        nonzero_bucket_id,
                                                        BUCKET_NUM
                                                    );
                
                // if (v1_idx == 6 && thread_lane == 0) {
                //     printf("warp_lane:%d, warp_id: %d, v0: %d, v1: %d\n", warp_lane, warp_id, v0, v1);
                //     // print each bit of output_bucket_bmap[0]
                //     // for (int i = 0; i < 64; ++i) {
                //     // //     printf("%lu", (output_bucket_bmap[0] >> (63-i)) & 1);
                //     //     printf("%lu", (bitmaps.d_bitmaps_[v1 * bmap_size] >> (63-i)) & 1);
                //     // }
                //     // printf("\n");
                //     // for (int i = 0; i < bmap_size; i++) {
                //     //     // printf("%ld ", output_bucket_bmap[i]);
                //     //     // printf("%ld ", bitmaps.d_bitmaps_[v0 * bmap_size + i]);
                //     //     // printf("%lu ", bitmaps.d_bitmaps_[v1 * bmap_size + i]);
                //     // }
                //     // for (int i = 0; i < BUCKET_NUM; i++) {
                //     //     printf("%d ", nonzero_bucket_id[warp_lane * BUCKET_NUM + i]);
                //     // }
                //     // printf("\n");
                //     for (int i = 0; i < candidate_v2.size(); i++) {
                //         printf("%d ", candidate_v2[i]);
                //     }
                //     printf("\n");
                // }
                #if 1
                // for(vidType v2_idx = thread_lane; v2_idx < candidate_v2.size(); v2_idx += WARP_SIZE){
                for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx++) {
                    auto v2 = candidate_v2[v2_idx];
                    // auto candidate_v3 = __get_vlist_from_graph(g, meta, /*vid=*/v2);
                    // if (warp_lane == 0 && thread_lane == 0) {
                    //     printf("candidate_v2: \n");
                    //     for (int i = 0; i < candidate_v2.size(); i++) {
                    //         printf("%d ", candidate_v2[i]);
                    //     }
                    //     printf("\n");
                    //     // printf("candidate_v3: \n");
                    //     // for (int i = 0; i < candidate_v3.size(); i++) {
                    //     //     printf("%d ", candidate_v3[i]);
                    //     // }
                    //     // printf("\n");
                    // }
                //     // count += __difference_num(candidate_v2, candidate_v3, /*upper_bound=*/v2);
                    // count += bitmaps.difference_num(candidate_v2, candidate_v3, /*upper_bound=*/v2);
                    // if (v2_idx == 0) 
                    {
                    // auto cnt = bitmaps.difference_num_warp_count_new(
                    //                                             // bitmaps.d_bitmaps_ + v2 * bmap_size, 
                    //                                             bitmaps.d_rbitmaps_ + v2 * bmap_size, 
                    //                                             bmap_size,
                    //                                             g, v1, v2, v2_idx,
                    //                                             meta,
                    //                                             0, 0,
                    //                                             nonzero_bucket_id,
                    //                                             BUCKET_NUM,
                    //                                             v2, v2
                    //                                         );
                    // if (thread_lane == 0)   count += cnt;
                    
                    auto neigh_v2 = __get_vlist_from_graph(g, meta, v2);
                    count += bitmaps.difference_num_bs(candidate_v2.ptr(), candidate_v2.size(), neigh_v2.ptr(), neigh_v2.size(), v2);
                    }
                }
                #endif
            }
        }
        }
        __syncthreads();
    }

    atomicAdd(&counter[0], count);
}


__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_LUT_block_ideal_test(GraphGPU g,
                            vidType* d_bitmap_all) {




}
