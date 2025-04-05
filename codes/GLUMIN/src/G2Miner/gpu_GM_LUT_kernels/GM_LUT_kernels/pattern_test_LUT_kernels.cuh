#define __N_LISTS1 3
#define __N_BITMAPS1 1


__device__ vidType count_AND_NOT_thread(Bitmap1DView<>& ope_u, Bitmap1DView<>& ope_v, int limit) {
    int valid_size = (ope_v.size_ < ope_u.size() ? ope_v.size_ : ope_u.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
        auto element = ope_v.load(i) & (~ope_u.load(i));
        num += __popc(element);
    }
    if (remain) {
        auto element = ope_v.load(countElement) & (~ope_u.load(countElement));
        num += __popc(element & ((1U << remain) - 1));
    }
    return num;
}

__device__ vidType count_NOT_AND_NOT_thread(Bitmap1DView<>& ope_u, Bitmap1DView<>& ope_v, int limit) {
    int valid_size = (ope_v.size_ < ope_u.size() ? ope_v.size_ : ope_u.size());
    limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
    int countElement = limit / BITMAP_WIDTH;
    int remain = limit % BITMAP_WIDTH;
    int num = 0;
    for (auto i = 0; i < countElement ; i ++) {
        auto element = (~ope_v.load(i)) & (~ope_u.load(i));
        num += __popc(element);
    }
    if (remain) {
        auto element = (~ope_v.load(countElement)) & (~ope_u.load(countElement));
        num += __popc(element & ((1U << remain) - 1));
    }
    return num;
}

__device__ vidType
__difference_num_test(VertexMapView v, VertexMapView u, vidType upper_bound) {
  if(v.use_one){
    return count_AND_NOT_thread(u.bitmap_, v.bitmap_, upper_bound);
  } else {
    return count_NOT_AND_NOT_thread(u.bitmap_, v.bitmap_, upper_bound);
  }
}


__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_LUT_block_test(vidType begin, vidType end, 
                  vidType *vid_list,
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs,
                  vidType* d_work_depth_each_warp
                ){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 3];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int block_id    = blockIdx.x;
  int num_warps   = WARPS_PER_BLOCK * gridDim.x;
  int warp_id     = thread_id / WARP_SIZE;
  int warp_lane   = threadIdx.x / WARP_SIZE;
  int thread_lane = threadIdx.x & (WARP_SIZE-1);
  int num_blocks  = gridDim.x;
  AccType count = 0;

  // meta
  StorageMeta meta;
  meta.lut = LUTs.getEmptyLUT(block_id);
  meta.base = vlists;
  meta.base_size = list_size;
  meta.bitmap_base = bitmaps;
  meta.bitmap_base_size = bitmap_size;
  meta.nv = end;
  meta.slot_size = max_deg;
  meta.bitmap_size = (max_deg + WARP_SIZE - 1) / WARP_SIZE;
  meta.capacity = 3;
  meta.bitmap_capacity = 1;
  meta.global_warp_id = warp_id;
  meta.local_warp_id = warp_lane;
  __syncwarp();
  int glb_warp_lane = thread_id & (WARP_SIZE-1);
  // BEGIN OF CODEGEN
  auto candidate_v0 = __get_vlist_from_heap(g, meta, /*slot_id=*/-1);
  // if (threadIdx.x == 0 && block_id < candidate_v0.size_) {
  //   printf("candidate_v0.size(): %d\n", candidate_v0.size_);
  //   for (int i = 0; i < candidate_v0.size_; i++) {
  //     printf("%d ", candidate_v0[i]);
  //   }
  //   printf("\n"); // 5 blocks
  // }
  // if (threadIdx.x == 0 && block_id == 0) {
  //   VertexMapView vmv = __get_vmap_from_lut(g, meta, /*idx_id=*/0, /*connected=*/true, /*upper_bound=*/-1);
  //   auto lut = meta.lut;
  //   printf("@@row bitmap: ");
  //   for (int i = 0; i < 5; i++) {
  //     printf("%d ", lut.row(0).ptr_[i]);
  //   }
  //   printf("\n");
  // }


  for(vidType v0_idx = begin + block_id; v0_idx < candidate_v0.size(); v0_idx += num_blocks){
    auto v0 = vid_list[v0_idx];
    auto candidate_v1 = __get_vlist_from_graph(g, meta, /*vid=*/v0); // v1 is in N(v0)
    // if (threadIdx.x == 0) {
    //     printf("blockid: %d, v0: %d, candidate_v1.size(): %d\n", block_id, v0, candidate_v1.size());
    // }
    if (threadIdx.x == 0 && block_id == 0) {
        // printf("v0: %d, candidate_v1.size(): %d\n", v0, candidate_v1.size());
    //   VertexMapView vmv = __get_vmap_from_lut(g, meta, /*idx_id=*/0, /*connected=*/true, /*upper_bound=*/-1);
    //   auto lut = meta.lut;
    //   printf("row bitmap: ");
    //   for (int i = 0; i < 5; i++) {
    //     printf("%d ", lut.row(0).ptr_[i]);
    //   }
    //   printf("\n");
    }
    __build_LUT_block(g, meta, __get_vlist_from_graph(g, meta, /*vid=*/v0));;
    __syncthreads();
    if (glb_warp_lane == 0) d_work_depth_each_warp[warp_id] = 1;
    // if (v0 == 0)
    {
    for (vidType v1_idx = warp_lane; v1_idx < candidate_v1.size(); v1_idx += WARPS_PER_BLOCK) { // 1 2 4 5
      auto v1 = candidate_v1[v1_idx];
      if (v1_idx == 1 && v0_idx == 0 && thread_lane == 0) {
        // printf("block_id: %d, v0: %d, v1_size: %d\n", block_id, v0, candidate_v1.size());
        // printf("current candidate_v1: %d\n", candidate_v1[v1_idx]);
        VertexMapView vmv = __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1);
        auto vlist = vmv.vlist_;
        auto bitmap = vmv.bitmap_;
        // printf("vlist.size(): %d\n", vlist.size_);
        // for (int i = 0; i < vlist.size_; i++) {
        //   printf("%d ", vlist.ptr_[i]);
        // }
        // printf("\n");
        // printf("bitmap.size(): %d\n", bitmap.size_);
        // for (int i = 0; i < bitmap.size_; i++) {
        //   printf("%d ", bitmap.ptr_[i]);
        //     // print each bit of ptr
        //     for (int j = 0; j < 32; j++) {
        //       printf("%d", (bitmap.ptr_[i] >> (31-j)) & 1);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
      }
      __build_index_from_vmap(g, meta, __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), /*slot_id=*/1);
      auto candidate_v2_idx = __get_vmap_from_heap(g, meta, /*bitmap_id=*/-1, /*slot_id=*/1);
      if (v1_idx == 1 && v0_idx == 0 && thread_lane == 0) {
        // printf("block_id: %d, v0: %d, v1_size: %d\n", block_id, v0, candidate_v1.size());
        // printf("\ncandidate_v2_idx.vlist_: ");
        // for (int i = 0; i < candidate_v2_idx.vlist_.size_; i++) {
        //   printf("%d ", candidate_v2_idx.vlist_.ptr_[i]);
        // }
        // printf("\n");
        // printf("candidate_v2_idx: "); // 1 3 ==> index of intersection of {1,2,4,5} and {0,2,3,5}
        // for (int i = 0; i < candidate_v2_idx.size(); i++) {
        //   printf("%d ", candidate_v2_idx[i]);
        // }
        // printf("\n");
      }
      if (glb_warp_lane == 0) d_work_depth_each_warp[warp_id] = 2;
      for(vidType v2_idx_idx = thread_lane; v2_idx_idx < candidate_v2_idx.size(); v2_idx_idx += WARP_SIZE){
        auto v2_idx = candidate_v2_idx[v2_idx_idx];
        // count += __difference_num(__get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), /*upper_bound=*/v2_idx);
        // auto cnt = __difference_num_test(
        //                 __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1), 
        //                 __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1), 
        //                 /*upper_bound=*/v2_idx
        //             );
        auto v2 = candidate_v2_idx.vlist_.ptr_[v2_idx];
        Bitmap1DView<> ope_u = __get_vmap_from_lut(g, meta, /*idx_id=*/v2_idx, /*connected=*/false, /*upper_bound=*/-1).bitmap_;
        Bitmap1DView<> ope_v = __get_vmap_from_lut(g, meta, /*idx_id=*/v1_idx, /*connected=*/true, /*upper_bound=*/-1).bitmap_;
        auto limit = v2_idx;
        int valid_size = (ope_v.size_ < ope_u.size() ? ope_v.size_ : ope_u.size());
        limit = limit < 0 ? valid_size : (limit < valid_size ? limit: valid_size);
        int countElement = limit / BITMAP_WIDTH;
        int remain = limit % BITMAP_WIDTH;
        int cnt = 0;
        for (auto i = 0; i < countElement ; i ++) {
            // printf("@@@@@\n");
            auto element = ope_v.load(i) & (~ope_u.load(i));
            cnt += __popc(element);
        }
        if (remain) {
            // printf("remain: %d\n", remain);
            auto element = ope_v.load(countElement) & (~ope_u.load(countElement));
            cnt += __popc(element & ((1U << remain) - 1)); // 低位有 remain 个1，高位为0。
            
            // if (v0 == 0 && v1 == 2 && v2_idx == 3) {
            //   printf("remain:%d ope_v.load(countElement): %d, ~ope_u.load(countElement): %d\n", 
            //           remain, ope_v.load(countElement), ~ope_u.load(countElement));
            //   // print each bit of ope_v.load(countElement)
            //   for (int j = 0; j < 32; j++) {
            //     printf("%d", (ope_v.load(countElement) >> (31-j)) & 1);
            //   }
            //   printf("\n");
            //   // print each bit of ope_u.load(countElement)
            //   for (int j = 0; j < 32; j++) {
            //     printf("%d", (~ope_u.load(countElement) >> (31-j)) & 1);
            //   }
            //   printf("\n");
            //   printf("element: %d result %d\n", element, element & ((1U << remain) - 1));
            // }
        }
        
        count += cnt;
        // if (cnt > 0 && warp_lane == 0) {
        // if (cnt > 0 && v0 < v1) {
        if (v0 == 0 && v1 == 2) {
          // printf("thread_lane: %d warp_lane %d, cnt: %d block_id: %d\n", thread_lane, warp_lane, cnt, block_id);
          // printf("v0: %d, v1: %d, v2_idx: %d:\n", v0, candidate_v1[v1_idx], v2_idx);
          printf("v2_idx: %d, v2: %d, thread_lane: %d, warp_lane %d, cnt: %d, limit: %d\n", 
                  v2_idx, v2, thread_lane, warp_lane, cnt, limit);
        }
        
        if (glb_warp_lane == 0) d_work_depth_each_warp[warp_id] = 3;
        // if (thread_lane == 0) printf("block_id:%d, warp_lane:%d\n", block_id, warp_lane);

      }
    }
    __syncthreads();
    }
  }
  // END OF CODEGEN

  atomicAdd(&counter[0], count);
}
