// edge-parallel warp-centric: each warp takes one edge
// this kernel is for the DAG version: no need to do symmetry breaking on-the-fly
__global__ void __launch_bounds__(BLK_SZ, 8)
clique4_warp_edge(eidType ne, GraphGPU g, vidType *vlists, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int num_warps   = (BLK_SZ / WARP_SIZE) * gridDim.x;       // total number of active warps
  vidType *vlist  = &vlists[int64_t(warp_id)*int64_t(max_deg)];
  AccType counter = 0;
  // printf("========\n");
  __shared__ vidType list_size[WARPS_PER_BLOCK];
  // __syncthreads();
  // eidType eid = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) 
  {
  #if 1
    if (eid == 7) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    // vidType v0 = g.d_src_list[eid];
    // vidType v1 = g.d_dst_list[eid];
    // printf("eid: %ld, v0: %d, v1: %d\n", eid, v0, v1);
    // printf("@@%d %d\n", g.d_src_list[eid], g.d_dst_list[eid]);
    // printf("v0: %d, v1: %d\n", v0, v1);
    // } 
    vidType v0_size = g.getOutDegree(v0);
    vidType v1_size = g.getOutDegree(v1);
    auto count = intersect(g.N(v0), v0_size, g.N(v1), v1_size, vlist);
    // if (eid == 0) 
    // printf("v0: %d, v1: %d, v0size: %d, v1size: %d, count: %d\n", v0, v1, v0_size, v1_size, count);
    if (thread_lane == 0) list_size[warp_lane] = count;
    __syncwarp();
    for (vidType i = 0; i < list_size[warp_lane]; i++) {
      vidType u = vlist[i];
      vidType u_size = g.getOutDegree(u);
      vidType v_size = list_size[warp_lane];
      if (eid == 0 && threadIdx.x == 0)
      printf("u: %d u_size: %d v_size: %d\n", u, u_size, v_size);
      counter += intersect_num(vlist, v_size, g.N(u), u_size);
    }
  #endif
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

