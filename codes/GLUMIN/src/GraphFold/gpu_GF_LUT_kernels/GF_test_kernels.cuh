

template <typename T = vidType>
__forceinline__ __device__ T difference_num_bs_test(T* a, T size_a, T* b, T size_b, T upper_bound, T parent) {
    //if (size_a == 0) return 0;
    //assert(size_b != 0);
    int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    T num = 0;
    for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
        auto key = a[i];
        int is_smaller = key < upper_bound ? 1 : 0;
        // if (parent == 4) {
        //     printf("thread_lane:%d, key:%d, upper_bound:%d, is_smaller:%d\n", thread_lane, key, upper_bound, is_smaller);
        // }
        if (is_smaller && !binary_search(b, key, size_b))
            num += 1;
        unsigned active = __activemask();
        unsigned mask = __ballot_sync(active, is_smaller);
        if (mask != FULL_MASK) break;
    }
    return num;
}


template <typename T = vidType>
__forceinline__ __device__ T difference_set_bs_test(T* a, T size_a, T* b, T size_b, T upper_bound, T* c, T parent) {
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


template <typename T = vidType>
__forceinline__ __device__ T intersect_bs_test(T* a, T size_a, T* b, T size_b, T upper_bound, T* c, T parent) {
    if (size_a == 0 || size_b == 0) return 0;
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
    __shared__ T count[WARPS_PER_BLOCK];
    T* lookup = a;
    T* search = b;
    T lookup_size = size_a;
    T search_size = size_b;
    if (size_a > size_b) {
        lookup = b;
        search = a;
        lookup_size = size_b;
        search_size = size_a;
    }
    if (thread_lane == 0) count[warp_lane] = 0;
    for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
        unsigned active = __activemask();
        __syncwarp(active);
        vidType key = lookup[i]; // each thread picks a vertex as the key
        int is_smaller = key < upper_bound ? 1 : 0;
        int found = 0;
        if (is_smaller && binary_search(search, key, search_size))
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


__global__ void P1_fused_matching_test(vidType nv, vidType ne, GraphGPU g,
                                   vidType *vlists, vidType max_deg, AccType *counters,
                                   AccType *INDEX1, AccType *INDEX2, int testv) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;                   // global warp index
    int thread_lane =
        threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
    int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
    vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
    __shared__ vidType list_size[WARPS_PER_BLOCK][2];
    AccType count = 0;
    AccType star3_count = 0;
    __syncthreads();
    // vidType vid = warp_id;
    for (vidType vid = warp_id; vid < nv;) {
    // for (; vid < nv;) {
        if (vid == 4) 
        {
        vidType v0 = vid;
        vidType v0_size = g.getOutDegree(v0);
        for (vidType i = thread_lane; i < v0_size; i += WARP_SIZE) {
            vlist[max_deg + i] = 0;
        }
        __syncwarp();
        // build lut merge
        for (int j = 0; j < v0_size; j++) {
            vidType v1 = g.N(v0)[j];
            vidType v1_size = g.getOutDegree(v1);
            for (auto i = thread_lane; i < v0_size; i += WARP_SIZE) {
                vidType key = g.N(v0)[i]; // each thread picks a vertex as the key
                int is_smaller = key < v1 ? 1 : 0;
                if (is_smaller && !binary_search(g.N(v1), key, v1_size)) {
                    atomicAdd(&vlist[max_deg + i], 1);
                    if (v0 == 4) {
                        printf("@@warp_id:%d, thread_lane = %d, v1:%d, key:%d, is_smaller:%d\n",
                                warp_id, thread_lane, v1, key, is_smaller);
                    }
                }
            }
        }
        __syncwarp();
        for (vidType v2_idx = 0; v2_idx < v0_size; v2_idx++) {
            vidType v2 = g.N(v0)[v2_idx];
            vidType v2_size = g.getOutDegree(v2);
            vidType tmp_cnt = difference_num_bs_test(g.N(v0), v0_size,
                                        g.N(v2), v2_size, v2, v0);
            vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
            __syncwarp();
            if (thread_lane == 0) {
                star3_count += (warp_cnt * vlist[max_deg + v2_idx]);
                if (v0 == 4) {
                    printf("v2:%d, v0_size:%d, v2_size:%d, tmp_cnt:%d, warp_cnt:%d, vlist[v2_idx]:%d, star3_count:%d\n",
                            v2, v0_size, v2_size, tmp_cnt, warp_cnt, vlist[max_deg + v2_idx], star3_count);
                    // for (int k = 0; k < v2_size; ++k) {
                    //     printf("%d ", g.N(v2)[k]);
                    // }
                    // printf("\n");
                }
            }
            __syncwarp();
            // if (thread_lane == 0 && v0 == 4) {
            //     printf("## ");
            //     for (int i = 0; i < v0_size; ++i) {
            //         printf("%d ", vlist[max_deg + i]);
            //     }
            //     printf("\n");
            // }

        #if 1   
            auto dif_cnt = difference_set_bs_test(g.N(v0), v0_size, g.N(v2),
                                        v2_size, v2, vlist, v0);
            auto int_cnt = intersect_bs_test(g.N(v0), v0_size, g.N(v2),
                                v2_size, v2, &vlist[max_deg], v0); // y0y1
            if (thread_lane == 0) {
                list_size[warp_lane][0] = dif_cnt;
                list_size[warp_lane][1] = int_cnt;
                if (v0 == 4) {
                    printf("## v0:%d, v2:%d, dif_cnt:%d, int_cnt:%d\n", v0, v2, dif_cnt, int_cnt);
                }
            }
            __syncwarp();
            if (thread_lane == 0 && v0 == 4) {
                // printf("warp_cnt:%d, dif_cnt:%d, int_cnt:%d\n",
                //         warp_cnt, list_size[warp_lane][0], list_size[warp_lane][1]);
                // if (v2 == 6) {
                //     printf("v0,v2 intersection: ");
                //     for (int k = 0; k < list_size[warp_lane][1]; ++k) {
                //         printf("%d ", vlist[max_deg + k]);
                //     }
                //     printf("\n");
                // }
            }

            for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
                vidType v3 = vlist[max_deg + i];
                vidType v3_size = g.getOutDegree(v3);
                for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
                    auto key = vlist[j];
                    vidType key_size = g.getOutDegree(key);
                    if (key > v3 && !binary_search(g.N(key), v3, key_size)) {
                        count += 1;
                    }
                    if (v0 == 4) {
                        printf("$$thread_lane:%d, v2:%d, v3:%d, key:%d, count:%lu\n",
                                thread_lane, v2, v3, key, count);
                    }
                }
            }
            // if (/*v0 == 4 && */count > 0) {
            // if (v0 == 4) {
            //     printf("$$v0:%d, thread_lane:%d, v2:%d, count:%lu\n", v0, thread_lane, v2, count);
            // }
        #endif
        // if (v0 == 4) {
        //     printf("thread_lane:%d, v0:%d, count:%lu\n", thread_lane, v0, count);
        // }
        }
        }
        if (thread_lane == 0) vid = atomicAdd(INDEX1, 1);
        __syncwarp();
        vid = __shfl_sync(0xffffffff, vid, 0);
    }

    if (count > 0 || star3_count > 0) {
        printf("testv:%d, warp_id:%d, thread_lane:%d, star3_count:%lu count:%lu\n", 
                testv, warp_id, thread_lane, star3_count, count);
    }
    atomicAdd(&counters[0], star3_count);
    atomicAdd(&counters[1], count);
}

// testv:0, warp_id:5, thread_lane:0, star3_count:3 count:0
// testv:0, warp_id:3, thread_lane:0, star3_count:1 count:0
// testv:0, warp_id:8, thread_lane:0, star3_count:1 count:1
// testv:0, warp_id:2, thread_lane:0, star3_count:1 count:0
// testv:0, warp_id:4, thread_lane:0, star3_count:6 count:1
// testv:0, warp_id:4, thread_lane:1, star3_count:0 count:2
// testv:0, warp_id:6, thread_lane:0, star3_count:3 count:1
// testv:0, warp_id:6, thread_lane:1, star3_count:0 count:1
// testv:0, warp_id:7, thread_lane:0, star3_count:2 count:1
// testv:0, warp_id:7, thread_lane:1, star3_count:0 count:1
// testv:0, warp_id:9, thread_lane:0, star3_count:4 count:1
// testv:0, warp_id:9, thread_lane:1, star3_count:0 count:1

__global__ void P1_frequency_count_test(vidType nv, GraphGPU g,
                                   vidType *vlists, vidType max_deg, AccType *counters,
                                   AccType *INDEX) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;                   // global warp index
    int thread_lane =
        threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
    int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
    vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
    __shared__ vidType list_size[WARPS_PER_BLOCK];
    AccType count = 0;
    AccType star3_count = 0;
    __syncthreads();
    for (vidType vid = warp_id; vid < nv;) {
        vidType v0 = vid;
        vidType v0_size = g.getOutDegree(v0);
        for (vidType i = thread_lane; i < v0_size; i += WARP_SIZE) {
            vlist[max_deg + i] = 0;
        }
        __syncwarp();
        // build lut merge
        for (int j = 0; j < v0_size; j++) {
            vidType v1 = g.N(v0)[j];
            vidType v1_size = g.getOutDegree(v1);
            for (auto i = thread_lane; i < v0_size; i += WARP_SIZE) {
                vidType key = g.N(v0)[i]; // each thread picks a vertex as the key
                int is_smaller = key < v1 ? 1 : 0;
                if (is_smaller && !binary_search(g.N(v1), key, v1_size)) {
                    atomicAdd(&vlist[max_deg + i], 1);
                    if (v0 == 4) {
                        printf("@@warp_id:%d, thread_lane = %d, v1:%d, key:%d, is_smaller:%d\n",
                                warp_id, thread_lane, v1, key, is_smaller);
                    }
                }
            }
        }
        __syncwarp();
        for (vidType v2_idx = 0; v2_idx < v0_size; v2_idx++) {
            vidType v2 = g.N(v0)[v2_idx];
            vidType v2_size = g.getOutDegree(v2);
            vidType tmp_cnt = difference_num_bs_test(g.N(v0), v0_size,
                                        g.N(v2), v2_size, v2, v0);
            vidType warp_cnt = warp_reduce<AccType>(tmp_cnt);
            __syncwarp();
            if (thread_lane == 0) {
                star3_count += (warp_cnt * vlist[max_deg + v2_idx]);
                
                if (v0 == 4) {
                    printf("v2:%d, v0_size:%d, v2_size:%d, tmp_cnt:%d, warp_cnt:%d, vlist[max_deg + v2_idx]:%d\n",
                            v2, v0_size, v2_size, tmp_cnt, warp_cnt, vlist[max_deg + v2_idx]);
                    for (int k = 0; k < v2_size; ++k) {
                        printf("%d ", g.N(v2)[k]);
                    }
                    printf("\n");
                }
            }
            __syncwarp();
        }
        if (thread_lane == 0) vid = atomicAdd(INDEX, 1);
        __syncwarp();
        vid = __shfl_sync(0xffffffff, vid, 0);
    }
    if (warp_id < nv && thread_lane == 0) {
        printf("warp_id:%d, star3_count:%lu count:%lu\n", warp_id, star3_count, count);
    }
    atomicAdd(&counters[0], star3_count);
}


__global__ void P1_count_correction_test(eidType ne, GraphGPU g,
                                    vidType *vlists, vidType max_deg, AccType *counters,
                                    AccType *INDEX) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;                   // global warp index
    int thread_lane =
        threadIdx.x & (WARP_SIZE - 1);           // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
    int num_warps = WARPS_PER_BLOCK * gridDim.x; // total number of active warps
    vidType *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
    __shared__ vidType list_size[WARPS_PER_BLOCK][2];
    AccType count = 0;
    __syncthreads();
    for (eidType eid = warp_id; eid < ne;) {
        vidType v0 = g.get_src(eid);
        vidType v1 = g.get_dst(eid);
        if (v1 == v0) {
            if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
            __syncwarp();
            eid = __shfl_sync(0xffffffff, eid, 0);
            continue;
        }
        vidType v0_size = g.getOutDegree(v0);
        vidType v1_size = g.getOutDegree(v1);
        auto dif_cnt = difference_set(g.N(v0), v0_size, g.N(v1),
                                    v1_size, v1, vlist);
        auto int_cnt = intersect(g.N(v0), v0_size, g.N(v1),
                                v1_size, v1, &vlist[max_deg]); // y0y1
        if (thread_lane == 0) {
            list_size[warp_lane][0] = dif_cnt;
            list_size[warp_lane][1] = int_cnt;
        }
        __syncwarp();
        for (vidType i = 0; i < list_size[warp_lane][1]; i++) {
            vidType v2 = vlist[max_deg + i];
            vidType v2_size = g.getOutDegree(v2);
            for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE) {
                auto key = vlist[j];
                vidType key_size = g.getOutDegree(key);
                if (key > v2 && !binary_search(g.N(key), v2, key_size))
                count += 1;
            }
        }
        __syncwarp();
        if (thread_lane == 0) eid = atomicAdd(INDEX, 1);
        __syncwarp();
        eid = __shfl_sync(0xffffffff, eid, 0);
    }
    atomicAdd(&counters[1], count);
}