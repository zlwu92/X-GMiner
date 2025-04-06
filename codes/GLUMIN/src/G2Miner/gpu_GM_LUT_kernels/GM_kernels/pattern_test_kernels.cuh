#define __N_LISTS3 2
#define __N_BITMAPS3 1


template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs_test(T* a, T size_a, T* b, T size_b) {
  //if (size_a == 0) return 0;
  //assert(size_b != 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    if (!binary_search(b, key, size_b)) {
      num += 1;
        // printf("key: %d\n", key);
    }
  }
  return num;
}

template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs_test(T* a, T size_a, T* b, T size_b, T *c) {
  //if (size_a == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i]; // each thread picks a vertex as the key
    int found = 0;
    if (!binary_search(b, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs_test(T* a, T size_a, T* b, T size_b, T upper_bound) {
  //if (size_a == 0) return 0;
  //assert(size_b != 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(b, key, size_b))
      num += 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return num;
}


template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs_test(T* a, T size_a, T* b, T size_b, T upper_bound, T* c) {
  //if (size_a == 0) return 0;
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i]; // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search(b, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) c[count[warp_lane]+idx-1] = key;
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
    mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK) break;
  }
  return count[warp_lane];
}

__device__ vidType
__difference_num_test(VertexArrayView v, VertexArrayView u, vidType upper_bound) {
  if (upper_bound < 0) {
    // return difference_num_test(v.ptr(), v.size(), u.ptr(), u.size());
    return difference_num_bs_test(v.ptr(), v.size(), u.ptr(), u.size());
  } else {
    return difference_num_bs_test(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound);
  }
}

__device__ VertexArrayView
__difference_test(StorageMeta& meta, VertexArrayView v, VertexArrayView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if (upper_bound < 0) {
    cnt = difference_set(v.ptr(), v.size(), u.ptr(), u.size(), buffer);
  } else {
    cnt = difference_set(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  return VertexArrayView(buffer, cnt);
}

__device__ VertexArrayView
__intersect_test(StorageMeta& meta, VertexArrayView v, VertexArrayView u, vidType upper_bound, int slot_id) {
  vidType* buffer = meta.buffer(slot_id);
  vidType cnt;
  if(upper_bound < 0) {
    cnt = intersect(v.ptr(), v.size(), u.ptr(), u.size(), buffer);
  } else {
    cnt = intersect(v.ptr(), v.size(), u.ptr(), u.size(), upper_bound, buffer);
  }
  __syncwarp();
  if (0 == (threadIdx.x & (WARP_SIZE - 1))) {
    meta.base_size[WARPS_PER_BLOCK * slot_id + meta.local_warp_id] = cnt;
  }
  __syncwarp();
  return VertexArrayView(buffer, cnt);
}


__global__ void __launch_bounds__(BLOCK_SIZE, 8)
P2_GM_test(eidType ne, 
                  GraphGPU g, 
                  vidType *vlists,
                  bitmapType* bitmaps,
                  vidType max_deg,
                  AccType *counter,
                  LUTManager<> LUTs,
                  vidType* d_work_depth_each_warp
                ){
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
  int glb_warp_lane = thread_id & (WARP_SIZE-1);
  // if (thread_id == 0) printf("@@@ %d %d\n", g.d_src_list[0], g.d_dst_list[0]);
  // BEGIN OF CODEGEN
  for(eidType eid = warp_id; eid < ne; eid += num_warps){
  // for(eidType eid = warp_id; eid < 1; eid += num_warps){
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    // if (warp_id == 0 && threadIdx.x == 0)
    if (eid == 0 && thread_lane == 0) {
      // printf("eid: %ld, thread_id: %d, warp_id: %d, v0: %d, v1: %d\n", eid, thread_id, warp_id, v0, v1);
    // printf("eid: %d, v0: %d, v1: %d\n", eid, v0, v1);
      VertexArrayView vv = __get_vlist_from_graph(g, meta, /*vid=*/v0);
      // printf("v0: %d, size: %d\n", v0, vv.size());
      for (int i = 0; i < vv.size_; i++) {
        // printf("%d ", vv.ptr_[i]);
      }
      // printf("\n");
      VertexArrayView vv1 = __get_vlist_from_graph(g, meta, /*vid=*/v1);
      // printf("v1: %d, size: %d\n", v1, vv1.size_);
      for (int i = 0; i < vv1.size_; i++) {
        // printf("%d ", vv1.ptr_[i]);
      }
      // printf("\n");
    }
    VertexArrayView vav = __intersect(meta, __get_vlist_from_graph(g, meta, /*vid=*/v0), __get_vlist_from_graph(g, meta, /*vid=*/v1), /*upper_bound=*/-1, /*output_slot=*/0);
    if (eid == 0 && threadIdx.x == 0) {
      // printf("eid: %ld, thread_id: %d, warp_id: %d, v0: %d, v1: %d\n", eid, thread_id, warp_id, v0, v1);
    //   printf("vav.size_: %d\n", vav.size_);
      // print vav.ptr_
      for(vidType v2_idx = 0; v2_idx < vav.size_; v2_idx ++){
        // printf("%d ", vav.ptr_[v2_idx]);
      }
      // printf("\n");
    }
    if (glb_warp_lane == 0) d_work_depth_each_warp[warp_id] += 1;
    auto candidate_v2 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
    for(vidType v2_idx = 0; v2_idx < candidate_v2.size(); v2_idx ++){
      auto v2 = candidate_v2[v2_idx];
      if (eid == 0 && threadIdx.x == 0) {
        // printf("v2: %d\n", v2);
      }
    //   if (thread_lane == 0)
    //   printf("v0: %d, v1: %d, v2: %d: ", v0, v1, v2);
    //   count += __difference_num(__get_vlist_from_heap(g, meta, /*slot_id=*/0), __get_vlist_from_graph(g, meta, /*vid=*/v2), /*upper_bound=*/v2);
      auto cnt = __difference_num_test(
                  __get_vlist_from_heap(g, meta, /*slot_id=*/0), 
                  // vav,
                  __get_vlist_from_graph(g, meta, /*vid=*/v2), 
                  /*upper_bound=*/v2
              );
        count += cnt;
        // __syncwarp();
        // if (cnt > 0 && thread_lane == 1 && warp_lane == 0) {
        //   // if (cnt > 0) {
        //     printf("thread_lane: %d warp_lane %d, cnt: %d eid: %ld blockid: %d\n", thread_lane, warp_lane, cnt, eid, blockIdx.x);
        //     printf("v0: %d, v1: %d, v2: %d\n", v0, v1, v2);
        //     // print vlist
        //     auto vlist1 = __get_vlist_from_heap(g, meta, /*slot_id=*/0);
        //     printf("vlist1: ");
        //     for (int i = 0; i < vlist1.size_; i++) {
        //         printf("%d ", vlist1.ptr_[i]);
        //     }
        //     printf("vlist2: ");
        //     auto vlist2 = __get_vlist_from_graph(g, meta, /*vid=*/v2);
        //     for (int i = 0; i < vlist2.size_; i++) {
        //         printf("%d ", vlist2.ptr_[i]);
        //     }
        //     printf("\n");
        // }
        if (glb_warp_lane == 0) d_work_depth_each_warp[warp_id] += 1;
    }
  }
  // END OF CODEGEN
  // if (count > 0 && threadIdx.x == 0)
  // if (count > 0)
  // printf("warpId: %d, thread_id: %d, count: %d\n", warp_id, thread_id, count);
  atomicAdd(&counter[0], count);
}
