#define __N_LISTS3 2
#define __N_BITMAPS3 1

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P3_GM(eidType ne, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs){
  __shared__ vidType list_size[WARPS_PER_BLOCK * 2];
  __shared__ vidType bitmap_size[WARPS_PER_BLOCK * 1];
  // __shared__ typename BlockReduce::TempStorage temp_storage;
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
  // for(eidType eid = warp_id; eid < 1; eid += num_warps){
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    if (warp_id == 0 && threadIdx.x == 0) 
    printf("eid: %d, thread_id: %d, warp_id: %d, v0: %d, v1: %d\n", eid, thread_id, warp_id, v0, v1);
    // printf("eid: %d, v0: %d, v1: %d\n", eid, v0, v1);
    VertexArrayView vav = __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/v1, /*output_slot=*/0);
    // if (eid == 0) {
    //   printf("vav.size_: %d\n", vav.size_);
    // }
    __difference(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/1);
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
    for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
      auto v2 = candidate_v2[v2_idx];
      // printf("here!!\n");
      count += __difference_num(__get_vlist_from_heap(g, meta, /*slot_id=*/1), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/-1);
    }
  }
  // END OF CODEGEN
  // if (count > 0 && threadIdx.x == 0)
  if (count > 0)
  printf("warpId: %d, thread_id: %d, count: %d\n", warp_id, thread_id, count);
  atomicAdd(&counter[0], count);
}
