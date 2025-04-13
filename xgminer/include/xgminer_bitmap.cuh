#pragma once
#include <iostream>
#include "utils.h"
#include "../include/graph_v2.h"
#include "../include/common.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <ranges> // C++20 ranges
#include <algorithm> // std::set_difference

__global__ void test_kernel(vidType* d_bucket_vlists_) {
    printf("test_kernel: bucket_vlists_[25638]: %d\n", d_bucket_vlists_[25638]);
}

// assumption: bitmapType == vidType
template <typename T = bitmap64_Type, int W = BITMAP64_WIDTH>
__device__ __host__ 
struct XGMiner_BITMAP {

    __host__ __device__
    XGMiner_BITMAP(int k) {
        // vlists.resize(0);
        // bitmaps_.resize(pow(2, k) / W, 0);
        bigset_bucket_num = pow(2, k);
        // bucket_vlists_.resize(bigset_bucket_num);
    }

    vidType bigset_bucket_num = 0;
    std::vector<vidType> bucket_vlists_;
    std::vector<vidType> bucket_sizelists_;
    std::vector<uint64_t> bitmaps_;
    vidType* d_bucket_vlists_ = nullptr;
    vidType* d_bucket_sizelists_ = nullptr;
    T* d_bitmaps_ = nullptr;


    __device__ __forceinline__ VertexArrayView intersection_old(
                                                T* bmap1, T* bmap2, int size1, int size2, 
                                                GraphGPU& g, int v1, int v2,
                                                StorageMeta& meta, 
                                                int bitmap_id, int slot_id,
                                                int* nonzero_bucket_id,
                                                vidType BUCKET_NUM
                                            ) {
        T* result = meta.bitmap64(bitmap_id);
        __shared__ vidType count[WARP_PER_BLOCK_128 * 2];
        int warp_lane   = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            count[warp_lane * 2] = 0;
            count[warp_lane * 2 + 1] = 0;
            // count[warp_lane * 3 + 2] = 0;
        }

        for (auto i = thread_lane; i < size1; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);
            result[i] = bmap1[i] & bmap2[i];
            bool found = result[i] != 0;
            unsigned mask = __ballot_sync(active, found);
            auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            // if (warp_lane == 0) {
            //     printf("@@ threadlane:%d, %lu, %d, idx:%d bmap1:%lu, bmap2:%lu\n", 
            //             thread_lane, result[i], static_cast<int>(found), idx, bmap1[i], bmap2[i]);
            // }
            if (found)  nonzero_bucket_id[warp_lane * BUCKET_NUM + count[warp_lane * 2]+idx-1] = i+1;
            if (thread_lane == 0) count[warp_lane * 2] += __popc(mask);
        }
        // __syncwarp();
        // __syncthreads();
        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("warp_lane:%d, count[0]: %d\n", warp_lane, count[warp_lane * 3]);
        //     for (int i = 0; i < BUCKET_NUM; i++) {
        //         printf("nonzero_bucket_id[%d]: %d\n", i, nonzero_bucket_id[warp_lane * BUCKET_NUM + i]);
        //     }
        //     // for (int i = 0; i < BUCKET_NUM; ++i) {
        //     //     printf("@@warp_lane:%d, result[%d]: %lu\n", warp_lane, i, result[i]);
        //     // }
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[0] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[1] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[2] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[3] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        // }
        vidType* buffer = meta.buffer(slot_id);
        for (int i = 0; i < count[warp_lane * 2]; i++) {
            int result_idx = nonzero_bucket_id[warp_lane * BUCKET_NUM + i] - 1;
            auto res = result[result_idx];
            #pragma unroll
            // if (thread_lane == 0) printf("res=%lu\n", res);
            for (int j = thread_lane; j < 64; j += WARP_SIZE) {
                unsigned active = __activemask();
                __syncwarp(active);
                // get each bit of res
                auto bit = (res >> (W - 1 - j)) & 1;
                // printf("j:%d, %lu\n", j, bit);
                unsigned mask = __ballot_sync(active, bit);
                auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
                // if (bit)    buffer[meta.slot_size + count[warp_lane * 2 + 1]+idx-1] = result_idx * 64 + j;
                if (bit)    buffer[count[warp_lane * 2 + 1]+idx-1] = result_idx * 64 + j;
                if (thread_lane == 0)   count[warp_lane * 2 + 1] += __popc(mask);
            }
        }
        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("warp_lane:%d, count[1]: %d\n", warp_lane, count[warp_lane * 3 + 1]);
        //     for (int i = 0; i < count[warp_lane * 3 + 1]; i++) {
        //         printf("%d ", buffer[meta.slot_size + i]);
        //     }
        //     printf("\n");
        // }
        #if 0
        // stage 3
        for (int i = 0; i < count[warp_lane * 3 + 1]; i++) {
            int bucket_idx = buffer[meta.slot_size + i];
            // intra-bucket binary search between 
            // (d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx], 
            // d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1])
            // and (d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            // d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx + 1])
            intersect_bs(d_bucket_vlists_, d_bucket_sizelists_, 
                        v1*bigset_bucket_num + bucket_idx,
                        v2*bigset_bucket_num + bucket_idx,
                        g.edge_begin(v1), g.edge_begin(v2), 
                        buffer, count[warp_lane * 3 + 2]);
            // if (thread_lane == 0) {
            //     // printf("bucket_idx:%d\n", bucket_idx);
            //     if (bucket_idx == 32) {
            //         // for (int k = d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx]; 
            //         //         k < d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1]; k++) {
            //         //     printf("d_bucket_vlists_[%d]: %d\n", k, d_bucket_vlists_[k]);
            //         // }
            //         printf("v1:%d, %d %d %lu %lu %d\n", v1, d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx],
            //                 d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1],
            //                 g.edge_begin(v1),
            //                 g.edge_begin(v1) + d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx],
            //                 d_bucket_vlists_[g.edge_begin(v1) + d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx]]);
            //         printf("v2:%d, %d %d %lu %lu %d\n", v2, d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            //                 d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx + 1],
            //                 g.edge_begin(v2),
            //                 g.edge_begin(v2) + d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            //                 d_bucket_vlists_[g.edge_begin(v2) + d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx]]);
            //     }
            // }
        }
        #endif
        return VertexArrayView(buffer, count[warp_lane * 2 + 1]);
    }


    __device__ __forceinline__ VertexArrayView intersection(
                                                T* bmap1, T* bmap2, int size1, int size2, 
                                                GraphGPU& g, int v1, int v2, int v2_idx,
                                                StorageMeta& meta, 
                                                int bitmap_id, int slot_id,
                                                int* nonzero_bucket_id,
                                                vidType BUCKET_NUM
                                            ) {
        T* result = meta.bitmap64(bitmap_id);
        __shared__ vidType count[WARP_PER_BLOCK_128 * 3];
        int warp_lane   = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            count[warp_lane * 3] = 0;
            count[warp_lane * 3 + 1] = 0;
            count[warp_lane * 3 + 2] = 0;
        }

        for (auto i = thread_lane; i < size1; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);
            result[i] = bmap1[i] & bmap2[i];
            bool found = result[i] != 0;
            unsigned mask = __ballot_sync(active, found);
            auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            // if (warp_lane == 0) {
            //     printf("@@ threadlane:%d, %lu, %d, idx:%d bmap1:%lu, bmap2:%lu\n", 
            //             thread_lane, result[i], static_cast<int>(found), idx, bmap1[i], bmap2[i]);
            // }
            if (found)  nonzero_bucket_id[warp_lane * BUCKET_NUM + count[warp_lane * 3]+idx-1] = i+1;
            if (thread_lane == 0) count[warp_lane * 3] += __popc(mask);
        }
        // __syncwarp();
        // __syncthreads();
        // if (v2_idx == 6 && thread_lane == 0) {
        //     printf("warp_lane:%d, count[0]: %d\n", warp_lane, count[warp_lane * 3]);
        //     for (int i = 0; i < BUCKET_NUM; i++) {
        //         printf("nonzero_bucket_id[%d]: %d\n", i, nonzero_bucket_id[warp_lane * BUCKET_NUM + i]);
        //     }
        //     // for (int i = 0; i < BUCKET_NUM; ++i) {
        //     //     printf("@@warp_lane:%d, result[%d]: %lu\n", warp_lane, i, result[i]);
        //     // }
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[0] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[1] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[2] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (result[3] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        // }
        vidType* buffer = meta.buffer(slot_id);
        for (int i = 0; i < count[warp_lane * 3]; i++) {
            int result_idx = nonzero_bucket_id[warp_lane * BUCKET_NUM + i] - 1;
            auto res = result[result_idx];
            #pragma unroll
            // if (thread_lane == 0) printf("res=%lu\n", res);
            for (int j = thread_lane; j < 64; j += WARP_SIZE) {
                unsigned active = __activemask();
                __syncwarp(active);
                // get each bit of res
                auto bit = (res >> (W - 1 - j)) & 1;
                // printf("j:%d, %lu\n", j, bit);
                unsigned mask = __ballot_sync(active, bit);
                auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
                if (bit)    buffer[meta.slot_size + count[warp_lane * 3 + 1]+idx-1] = result_idx * 64 + j;
                if (thread_lane == 0)   count[warp_lane * 3 + 1] += __popc(mask);
            }
        }
        // if (v2_idx == 6 && thread_lane == 0) {
        //     printf("warp_lane:%d, count[1]: %d\n", warp_lane, count[warp_lane * 3 + 1]);
        //     for (int i = 0; i < count[warp_lane * 3 + 1]; i++) {
        //         printf("%d ", buffer[meta.slot_size + i]);
        //     }
        //     printf("\n");
        // }
        
        // stage 3
        for (int i = 0; i < count[warp_lane * 3 + 1]; i++) {
            int bucket_idx = buffer[meta.slot_size + i];
            // intra-bucket binary search between 
            intersect_bs(d_bucket_vlists_, d_bucket_sizelists_, 
                        v1*bigset_bucket_num + bucket_idx,
                        v2*bigset_bucket_num + bucket_idx,
                        bucket_idx,
                        g.edge_begin(v1), g.edge_begin(v2), 
                        g.edge_end(v1), g.edge_end(v2), 
                        buffer, count[warp_lane * 3 + 2]);
            // if (thread_lane == 0) {
            //     // printf("bucket_idx:%d\n", bucket_idx);
            //     if (bucket_idx == 255) {
            //         // for (int k = d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx]; 
            //         //         k < d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1]; k++) {
            //         //     printf("d_bucket_vlists_[%d]: %d\n", k, d_bucket_vlists_[k]);
            //         // }
            //         printf("v1:%d, %d %d %lu %lu %d\n", v1, d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx],
            //                 d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1],
            //                 g.edge_begin(v1),
            //                 g.edge_begin(v1) + d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx],
            //                 d_bucket_vlists_[g.edge_begin(v1) + d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx]]);
            //         printf("v2:%d, %d %d %lu %lu %d\n", v2, d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            //                 d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx + 1],
            //                 g.edge_begin(v2),
            //                 g.edge_begin(v2) + d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            //                 d_bucket_vlists_[g.edge_begin(v2) + d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx]]);
            //     }
            // }
        }
        return VertexArrayView(buffer, count[warp_lane * 3 + 2]);
    }

    __device__ __forceinline__ void intersect_bs(vidType* d_bucket_vlists_, vidType* d_bucket_sizelists_,
                                        int v1_start_idx, int v2_start_idx, 
                                        int bucket_idx,
                                        int v1_edge_off, int v2_edge_off,
                                        int v1_edge_end, int v2_edge_end, 
                                        vidType* buffer, vidType& count) {

        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        vidType* lookup = d_bucket_vlists_ + v1_edge_off + d_bucket_sizelists_[v1_start_idx];
        vidType* search = d_bucket_vlists_ + v2_edge_off + d_bucket_sizelists_[v2_start_idx];
        vidType lookup_size = bucket_idx == bigset_bucket_num - 1 ? v1_edge_end - d_bucket_sizelists_[v1_start_idx] - v1_edge_off:
                            d_bucket_sizelists_[v1_start_idx + 1] - d_bucket_sizelists_[v1_start_idx];
        vidType search_size = bucket_idx == bigset_bucket_num - 1 ? v2_edge_end - d_bucket_sizelists_[v2_start_idx] - v2_edge_off:
                            d_bucket_sizelists_[v2_start_idx + 1] - d_bucket_sizelists_[v2_start_idx];
        if (lookup_size > search_size) {
            auto tmp = lookup;
            lookup = search;
            search = tmp;
            auto tmp_size = lookup_size;
            lookup_size = search_size;
            search_size = tmp_size;
        }
        // if (thread_lane == 0 && bucket_idx == 255) {
        //     printf("lookup_size:%d, search_size:%d\n", lookup_size, search_size);
        // }
        for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);

            int found = 0;
            vidType key = lookup[i]; // each thread picks a vertex as the key
            // printf("%d key:%d\n", bucket_idx, key);
            if (binary_search(search, key, search_size))    found = 1;
            unsigned mask = __ballot_sync(active, found);
            auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            
            if (found)  buffer[count+idx-1] = key;
            if (thread_lane == 0)   count += __popc(mask);
        }
    }


    __device__ __forceinline__ vidType difference_num_warp_count(
                                                // T* bmap1, int size1, 
                                                T* bmap2, int size2, 
                                                StorageMeta& meta, 
                                                int bitmap_id, int slot_id,
                                                int* nonzero_bucket_id,
                                                vidType BUCKET_NUM, vidType curr_v, vidType lower_bound
                                            ) {
        T* last_level_result = meta.bitmap64(bitmap_id);
        T* curr_level_result = meta.bitmap64(bitmap_id + 1);
        __shared__ vidType count[WARP_PER_BLOCK_128 * 2];
        int warp_lane   = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            count[warp_lane * 2] = 0;
            count[warp_lane * 2 + 1] = 0;
            // count[warp_lane * 3 + 2] = 0;
        }
        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("bmap2:\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[0] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[1] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[2] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[3] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        // }
        for (auto i = thread_lane; i < size2; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);
            // if (warp_lane <= 1) {
            //     printf("warp_lane:%d, threadlane:%d, %lu\n", warp_lane, thread_lane, last_level_result[thread_lane]);
            // }
            curr_level_result[i] = (last_level_result[i] ^ bmap2[i]) & last_level_result[i];
            // curr_level_result[i] = last_level_result[i] & bmap2[i];
            bool found = curr_level_result[i] != 0;
            unsigned mask = __ballot_sync(active, found);
            auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            // if (warp_lane <= 1) {
            //     printf("warp_lane:%d, threadlane:%d, %d, idx:%d\n", warp_lane, thread_lane, found, idx);
            // }
            // if (i == 0) {
            //     printf("@:");
            //     for (int s = 0; s < 64; ++s) {
            //         printf("%lu", (curr_level_result[i] >> (W-1-s)) & 1);
            //         // printf("%lu", (bmap2[i] >> (W-1-s)) & 1);
            //     }
            //     printf("\n");
            // }
            if (found)  nonzero_bucket_id[warp_lane * BUCKET_NUM + count[warp_lane * 2]+idx-1] = i+1;
            if (thread_lane == 0)   count[warp_lane * 2] += __popc(mask);
        }
        // __syncwarp();
        // __syncthreads();
        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("curr_v:%d, count[0]: %d\n", curr_v, count[warp_lane * 3]);
        //     for (int i = 0; i < BUCKET_NUM; i++) {
        //         printf("nonzero_bucket_id[%d]: %d\n", i, nonzero_bucket_id[warp_lane * BUCKET_NUM + i]);
        //     }
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[0] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[1] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[2] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[3] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        // }

        for (int i = 0; i < count[warp_lane * 2]; i++) {
            int result_idx = nonzero_bucket_id[warp_lane * BUCKET_NUM + i] - 1;
            auto res = curr_level_result[result_idx];
            #pragma unroll
            for (int j = thread_lane; j < 64; j += WARP_SIZE) {
                unsigned active = __activemask();
                __syncwarp(active);
                // get each bit of res
                bool condition = lower_bound >=0 ? j+result_idx*64 > lower_bound : j+result_idx*64 != curr_v;
                unsigned bit1 = (res >> (W - 1 - j)) & 1;
                unsigned bit = ((res >> (W - 1 - j)) & 1) && condition;
                // if (i == 0 && warp_lane == 0) {
                //     printf("j:%d, res:%u, %u\n", j, bit1, bit);
                // }
                unsigned mask = __ballot_sync(active, bit);

                if (thread_lane == 0)   count[warp_lane * 2 + 1] += __popc(mask);
            }
            // if (warp_lane == 1 && thread_lane == 0) printf("@count[0]: %d\n", count[warp_lane]);
        }

        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("warp_lane:%d, curr_v:%d, count[1]:%d\n", warp_lane, curr_v, count[warp_lane * 3 + 1]);
        // }
        return count[warp_lane * 2 + 1];
    }


    __device__ __forceinline__ vidType difference_num_warp_count_new(
                                                // T* bmap1, int size1, 
                                                T* bmap2, int size2, 
                                                GraphGPU& g, int v1, int v2, int v2_idx,
                                                StorageMeta& meta, 
                                                int bitmap_id, int slot_id,
                                                int* nonzero_bucket_id,
                                                vidType BUCKET_NUM, vidType curr_v, vidType lower_bound
                                            ) {
        T* last_level_result = meta.bitmap64(bitmap_id);
        T* curr_level_result = meta.bitmap64(bitmap_id + 1);
        __shared__ vidType count[WARP_PER_BLOCK_128 * 3];
        int warp_lane   = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            count[warp_lane * 3] = 0;
            count[warp_lane * 3 + 1] = 0;
            count[warp_lane * 3 + 2] = 0;
        }
        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("bmap2:\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[0] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[1] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[2] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (bmap2[3] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        // }
        for (auto i = thread_lane; i < size2; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);
            // if (warp_lane <= 1) {
            //     printf("warp_lane:%d, threadlane:%d, %lu\n", warp_lane, thread_lane, last_level_result[thread_lane]);
            // }
            // curr_level_result[i] = (last_level_result[i] ^ bmap2[i]) & last_level_result[i];
            curr_level_result[i] = last_level_result[i] & bmap2[i];
            bool found = curr_level_result[i] != 0;
            unsigned mask = __ballot_sync(active, found);
            auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            // if (warp_lane <= 1) {
            //     printf("warp_lane:%d, threadlane:%d, %d, idx:%d\n", warp_lane, thread_lane, found, idx);
            // }
            // if (i == 0) {
            //     printf("@:");
            //     for (int s = 0; s < 64; ++s) {
            //         printf("%lu", (curr_level_result[i] >> (W-1-s)) & 1);
            //         // printf("%lu", (bmap2[i] >> (W-1-s)) & 1);
            //     }
            //     printf("\n");
            // }
            if (found)  nonzero_bucket_id[warp_lane * BUCKET_NUM + count[warp_lane * 3]+idx-1] = i+1;
            if (thread_lane == 0)   count[warp_lane * 3] += __popc(mask);
        }
        // __syncwarp();
        // __syncthreads();
        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("curr_v:%d, count[0]: %d\n", curr_v, count[warp_lane * 3]);
        //     for (int i = 0; i < BUCKET_NUM; i++) {
        //         printf("nonzero_bucket_id[%d]: %d\n", i, nonzero_bucket_id[warp_lane * BUCKET_NUM + i]);
        //     }
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[0] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[1] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[2] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        //     for (int i = 0; i < 64; ++i) {
        //         printf("%lu", (curr_level_result[3] >> (W-1-i)) & 1);
        //     }
        //     printf("\n");
        // }
        vidType* buffer = meta.buffer(slot_id);
        for (int i = 0; i < count[warp_lane * 3]; i++) {
            int result_idx = nonzero_bucket_id[warp_lane * BUCKET_NUM + i] - 1;
            auto res = curr_level_result[result_idx];
            #pragma unroll
            for (int j = thread_lane; j < 64; j += WARP_SIZE) {
                unsigned active = __activemask();
                __syncwarp(active);
                // get each bit of res
                // bool condition = lower_bound >=0 ? j+result_idx*64 > lower_bound : j+result_idx*64 != curr_v;
                // unsigned bit1 = (res >> (W - 1 - j)) & 1;
                // unsigned bit = ((res >> (W - 1 - j)) & 1) && condition;
                auto bit = (res >> (W - 1 - j)) & 1;
                // printf("j:%d, %lu\n", j, bit);
                unsigned mask = __ballot_sync(active, bit);
                auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
                if (bit)    buffer[meta.slot_size + count[warp_lane * 3 + 1]+idx-1] = result_idx * 64 + j;
                if (thread_lane == 0)   count[warp_lane * 3 + 1] += __popc(mask);
            }
            // if (warp_lane == 1 && thread_lane == 0) printf("@count[0]: %d\n", count[warp_lane]);
        }

        // if (warp_lane <= 1 && thread_lane == 0) {
        //     printf("warp_lane:%d, curr_v:%d, count[1]:%d\n", warp_lane, curr_v, count[warp_lane * 3 + 1]);
        // }
        // return count[warp_lane * 3 + 1];
        // stage 3
        for (int i = 0; i < count[warp_lane * 3 + 1]; i++) {
            int bucket_idx = buffer[meta.slot_size + i];
            // intra-bucket binary search between 
            intersect_bs_num(
                        d_bucket_vlists_, d_rbucket_vlists_,
                        d_bucket_sizelists_, d_rbucket_sizelists_,
                        v1*bigset_bucket_num + bucket_idx,
                        v2*bigset_bucket_num + bucket_idx,
                        bucket_idx,
                        g.edge_begin(v1), g.edge_begin(v2), 
                        g.edge_end(v1), g.edge_end(v2), 
                        // buffer, 
                        count[warp_lane * 3 + 2], lower_bound);
            // if (thread_lane == 0) {
            //     // printf("bucket_idx:%d\n", bucket_idx);
            //     if (bucket_idx == 255) {
            //         // for (int k = d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx]; 
            //         //         k < d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1]; k++) {
            //         //     printf("d_bucket_vlists_[%d]: %d\n", k, d_bucket_vlists_[k]);
            //         // }
            //         printf("v1:%d, %d %d %lu %lu %d\n", v1, d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx],
            //                 d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx + 1],
            //                 g.edge_begin(v1),
            //                 g.edge_begin(v1) + d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx],
            //                 d_bucket_vlists_[g.edge_begin(v1) + d_bucket_sizelists_[v1*bigset_bucket_num + bucket_idx]]);
            //         printf("v2:%d, %d %d %lu %lu %d\n", v2, d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            //                 d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx + 1],
            //                 g.edge_begin(v2),
            //                 g.edge_begin(v2) + d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx],
            //                 d_bucket_vlists_[g.edge_begin(v2) + d_bucket_sizelists_[v2*bigset_bucket_num + bucket_idx]]);
            //     }
            // }
        }
        return count[warp_lane * 3 + 2];
    }


    __device__ __forceinline__ void intersect_bs_num(
                                        vidType* d_bucket_vlists1_, vidType* d_bucket_vlists2_, 
                                        vidType* d_bucket_sizelists1_, vidType* d_bucket_sizelists2_,
                                        int v1_start_idx, int v2_start_idx, 
                                        int bucket_idx,
                                        int v1_edge_off, int v2_edge_off,
                                        int v1_edge_end, int v2_edge_end, 
                                        // vidType* buffer, 
                                        vidType& count,
                                        vidType lower_bound) {

        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        vidType* lookup = d_bucket_vlists1_ + v1_edge_off + d_bucket_sizelists1_[v1_start_idx];
        vidType* search = d_bucket_vlists2_ + v2_edge_off + d_bucket_sizelists2_[v2_start_idx];
        vidType lookup_size = bucket_idx == bigset_bucket_num - 1 ? v1_edge_end - d_bucket_sizelists1_[v1_start_idx] - v1_edge_off:
                            d_bucket_sizelists1_[v1_start_idx + 1] - d_bucket_sizelists1_[v1_start_idx];
        vidType search_size = bucket_idx == bigset_bucket_num - 1 ? v2_edge_end - d_bucket_sizelists2_[v2_start_idx] - v2_edge_off:
                            d_bucket_sizelists2_[v2_start_idx + 1] - d_bucket_sizelists2_[v2_start_idx];
        if (lookup_size > search_size) {
            auto tmp = lookup;
            lookup = search;
            search = tmp;
            auto tmp_size = lookup_size;
            lookup_size = search_size;
            search_size = tmp_size;
        }
        // if (thread_lane == 0 && bucket_idx == 255) {
        //     printf("lookup_size:%d, search_size:%d\n", lookup_size, search_size);
        // }
        for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);

            int found = 0;
            vidType key = lookup[i]; // each thread picks a vertex as the key
            // printf("%d key:%d\n", bucket_idx, key);
            if (key > lower_bound) {
                if (binary_search(search, key, search_size))    found = 1;
            }
            unsigned mask = __ballot_sync(active, found);
            // auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            
            // if (found)  buffer[count+idx-1] = key;
            if (thread_lane == 0)   count += __popc(mask);
        }
    }



    // template <typename D = vidType>
    __device__ __forceinline__ vidType difference_num_bs(vidType* a, vidType size_a, 
                                        vidType* b, vidType size_b, vidType lower_bound) {
        //if (size_a == 0) return 0;
        //assert(size_b != 0);
        int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
        vidType num = 0;
        for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
            auto key = a[i];
            int is_larger = key > lower_bound ? 1 : 0;
            // printf("%d %d %d %d\n", key, is_smaller, upper_bound, size_b);
            if (is_larger && !binary_search(b, key, size_b)) {
                // printf("thread_lane:%d, key:%d\n", thread_lane, key);
                num += 1;
            }
            // unsigned active = __activemask();
            // unsigned mask = __ballot_sync(active, is_larger);
            // if (mask != FULL_MASK) break;
        }
        return num;
    }



    __host__ void bigset_bitmap_processing(Graph_V2& g) {
        LOG_INFO("Running bigset bitmap processing");
        std::cout << "W: " << W << " " << UP_DIV(bigset_bucket_num, W) << "\n";
        vidType nv = g.num_vertices();
        eidType ne = g.num_edges();
        // bucket_vlists_.resize(nv);
        bucket_sizelists_.resize(nv * bigset_bucket_num, 0);
        // bitmaps_.resize(nv);
        int bmap_size = UP_DIV(bigset_bucket_num, W);
        // std::set<int> fullSet;
        // for (int i = 0; i < nv; ++i) fullSet.insert(i);
        std::vector<std::vector<int>> r_edgeList(nv);
        bitmaps_.resize(nv * bmap_size, 0x0);
        std::vector<bool> temp_bitmap(nv, 0);
        for (int i = 0; i < nv; i++) {
            // auto vlist = g.N(i);
            auto edge_beg = g.edge_begin(i);
            auto edge_end = g.edge_end(i);
            // bucket_vlists_[i].resize(bigset_bucket_num);
            if (i == 338 || i == 18)
            std::cout << "vertex " << i << ": \n";
            // std::set<int> edgeSet;
            std::fill(temp_bitmap.begin(), temp_bitmap.end(), 0);
            for (int j = edge_beg; j < edge_end; j++) {
                auto neigh = g.out_colidx()[j];
                // edgeSet.insert(neigh);
                temp_bitmap[neigh] = 1;
                int bucket_id = neigh & (bigset_bucket_num - 1);
                // bucket_vlists_.push_back(neigh);
                bucket_sizelists_[i * bigset_bucket_num + bucket_id]++;
                
                int bucket_ele = bucket_id / W;
                int bucket_bit = bucket_id % W;
                if (i == 338 || i == 18)
                std::cout << neigh << "[" << bucket_id << " " << bucket_ele << " " << bucket_bit << "] ";
                bitmaps_[i * bmap_size + bucket_ele] |= (1ULL << (W - 1 - bucket_bit));
                // for (int s = 0; s < 64; ++s) {
                //     printf("%lu", (bitmaps_[i * bmap_size + bucket_ele] >> (W-1-s)) & 1);
                // }
                // std::cout << "\n";
            }

            if (i == 338 || i == 18)
            std::cout << "\n";
            if (i == 18) {
                for (int t = 0; t < bmap_size; ++t) {
                    for (int s = 0; s < 64; ++s) {
                        printf("%lu", (bitmaps_[i * bmap_size + t] >> (W-1-s)) & 1);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            if (i == 338) {
                for (int t = 0; t < bmap_size; ++t) {
                    for (int s = 0; s < 64; ++s) {
                        printf("%lu", (bitmaps_[i * bmap_size + t] >> (W-1-s)) & 1);
                    }
                    printf("\n");
                }
            }
            
            // std::set<int> difference;
            // compute complement of edgeSet
            // std::set_difference(fullSet.begin(), fullSet.end(),
            //                     edgeSet.begin(), edgeSet.end(),
            //                     std::inserter(difference, difference.begin()));
            // r_edgeList[i].assign(difference.begin(), difference.end());
            for (int j = 0; j < nv; ++j) {
                if (temp_bitmap[j] == 0) {
                    r_edgeList[i].push_back(j);
                }
            }
            // if (i == 0) {
            //     for (auto val : r_edgeList[i]) {
            //         printf("r_edgeList[%d]: %d\n", i, val);
            //     }
            // }
        }
        // std::cout << "bitmaps_[0][0]: " << bitmaps_[0][0] << "\n";
        // for (int i = 0; i < 64; ++i) {
        //     printf("%lu", (bitmaps_[0][0] >> (63-i)) & 1);
        // }
        // std::cout << "\n";
        // for (auto val : bitmaps_[0]) {
        //     std::cout << val << " ";
        // }
        // std::cout << "\n";

        // perform prefix sum on bucket_sizelists_
        auto nnz = std::accumulate(bucket_sizelists_.begin(), bucket_sizelists_.end(), 0);
        std::cout << nnz << " " << g.E() << "\n";
        bucket_vlists_.resize(g.E());
        // edge list reordering
        for (int i = 0; i < nv; i++) {
            // auto vlist = g.N(i);
            auto edge_beg = g.edge_begin(i);
            auto edge_end = g.edge_end(i);
            // need to maintain a temp bucket size list
            std::vector<int> temp_bucket_size(bigset_bucket_num, 0);
            // copy bucket_sizelists_ to temp_bucket_size1
            // std::vector<int> temp_bucket_size1(bigset_bucket_num, 0);
            // std::copy(bucket_sizelists_.begin() + i * bigset_bucket_num, 
            //         bucket_sizelists_.begin() + (i + 1) * bigset_bucket_num, temp_bucket_size1.begin());
            // perform prefix sum on temp_bucket_size
            // thrust::exclusive_scan(temp_bucket_size1.begin(), temp_bucket_size1.end(), temp_bucket_size1.begin());
            thrust::exclusive_scan(bucket_sizelists_.begin() + i * bigset_bucket_num, 
                            bucket_sizelists_.begin() + (i + 1) * bigset_bucket_num, 
                            bucket_sizelists_.begin() + i * bigset_bucket_num);
            for (int j = edge_beg; j < edge_end; j++) {
                auto neigh = g.out_colidx()[j];
                int bucket_id = neigh & (bigset_bucket_num - 1);
                if ((i == 18 || i == 338) && neigh == 767) {
                    printf("bucket_id: %d, %d %d %lu %lu %d\n", 
                            bucket_id, i, g.edge_begin(i), 
                            g.edge_begin(i) + bucket_sizelists_[i * bigset_bucket_num + bucket_id] + temp_bucket_size[bucket_id], 
                            g.edge_end(i),
                            neigh);
                    // printf("%d %d \n", bucket_sizelists_[i * bigset_bucket_num + bucket_id],
                    //         bucket_sizelists_[i * bigset_bucket_num + bucket_id + 1]);
                }
                bucket_vlists_[g.edge_begin(i) + bucket_sizelists_[i * bigset_bucket_num + bucket_id] + temp_bucket_size[bucket_id]++] = neigh;
            }
            // if (i == 0) {
            //     for (int i = 0; i < bigset_bucket_num; ++i) {
            //         printf("bucket_sizelists_[%d]: %d\n", i, bucket_sizelists_[i]);
            //     }
            // }
        }

        // copy bitmaps_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bitmaps_, sizeof(T) * nv * bmap_size));
        // for (int i = 0; i < nv; i++) {
        //     CUDA_SAFE_CALL(cudaMemcpy(d_bitmaps_ + i * (bigset_bucket_num / W), 
        //                     bitmaps_[i].data(), sizeof(T) * (bigset_bucket_num / W), cudaMemcpyHostToDevice));
        // }
        CUDA_SAFE_CALL(cudaMemcpy(d_bitmaps_, bitmaps_.data(), sizeof(T) * nv * bmap_size, cudaMemcpyHostToDevice));

        // printf("host-side: bucket_vlists_[25638]: %d\n", bucket_vlists_[25638]);
        // copy bucket_vlists_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bucket_vlists_, sizeof(vidType) * g.E()));
        CUDA_SAFE_CALL(cudaMemcpy(d_bucket_vlists_, bucket_vlists_.data(), sizeof(vidType) * g.E(), cudaMemcpyHostToDevice));
        
        // test_kernel<<<1, 1>>>(d_bucket_vlists_);
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
        // copy bucket_sizelists_ to device
        // thrust::exclusive_scan(bucket_sizelists_.begin(), bucket_sizelists_.end(), bucket_sizelists_.begin());
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bucket_sizelists_, sizeof(vidType) * bucket_sizelists_.size()));
        CUDA_SAFE_CALL(cudaMemcpy(d_bucket_sizelists_, bucket_sizelists_.data(), 
                            sizeof(vidType) * bucket_sizelists_.size(), cudaMemcpyHostToDevice));
    

        // generate_reverse_bitmap(g, r_edgeList);
    
    }

    std::vector<vidType> rbucket_vlists_;
    std::vector<vidType> rbucket_sizelists_;
    std::vector<uint64_t> rbitmaps_;
    vidType* d_rbucket_vlists_ = nullptr;
    vidType* d_rbucket_sizelists_ = nullptr;
    T* d_rbitmaps_ = nullptr;

    __host__ void generate_reverse_bitmap(Graph_V2& g, std::vector<std::vector<int>>& r_edgeList) {
        LOG_INFO("Running reverse bitmap processing");

        vidType nv = g.num_vertices();
        eidType ne = g.num_edges();
        rbucket_sizelists_.resize(nv * bigset_bucket_num, 0);
        int bmap_size = UP_DIV(bigset_bucket_num, W);
        rbitmaps_.resize(nv * bmap_size, 0x0);
        int count = 0;
        std::vector<eidType> edge_beg(nv+1, 0); 
        for (int i = 0; i < nv; i++) {
            for (int j = 0; j < r_edgeList[i].size(); j++) {
                auto neigh = r_edgeList[i][j];
                int bucket_id = neigh & (bigset_bucket_num - 1);
                rbucket_sizelists_[i * bigset_bucket_num + bucket_id]++;
                
                int bucket_ele = bucket_id / W;
                int bucket_bit = bucket_id % W;
                rbitmaps_[i * bmap_size + bucket_ele] |= (1ULL << (W - 1 - bucket_bit));
            }
            count += r_edgeList[i].size();
            edge_beg[i+1] = count;

            if (i == 288) {
                for (int t = 0; t < bmap_size; ++t) {
                    for (int s = 0; s < 64; ++s) {
                        printf("%lu", (rbitmaps_[i * bmap_size + t] >> (W-1-s)) & 1);
                    }
                    printf("\n");
                }
            }
        }

        rbucket_vlists_.resize(count);

        for (int i = 0; i < nv; i++) {
            std::vector<int> temp_bucket_size(bigset_bucket_num, 0);
            thrust::exclusive_scan(rbucket_sizelists_.begin() + i * bigset_bucket_num, 
                                rbucket_sizelists_.begin() + (i + 1) * bigset_bucket_num, 
                                rbucket_sizelists_.begin() + i * bigset_bucket_num);
            for (int j = 0; j < r_edgeList[i].size(); j++) {
                auto neigh = r_edgeList[i][j];
                int bucket_id = neigh & (bigset_bucket_num - 1);
                rbucket_vlists_[edge_beg[i+1] + rbucket_sizelists_[i * bigset_bucket_num + bucket_id] + temp_bucket_size[bucket_id]++] = neigh;
            }
        }

        // copy rbitmaps_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_rbitmaps_, sizeof(T) * nv * bmap_size));
        CUDA_SAFE_CALL(cudaMemcpy(d_rbitmaps_, bitmaps_.data(), sizeof(T) * nv * bmap_size, cudaMemcpyHostToDevice));

        // copy rbucket_vlists_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_rbucket_vlists_, sizeof(vidType) * count));
        CUDA_SAFE_CALL(cudaMemcpy(d_rbucket_vlists_, rbucket_vlists_.data(), sizeof(vidType) * count, cudaMemcpyHostToDevice));

        // copy rbucket_sizelists_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_rbucket_sizelists_, sizeof(vidType) * rbucket_sizelists_.size()));
        CUDA_SAFE_CALL(cudaMemcpy(d_rbucket_sizelists_, rbucket_sizelists_.data(), 
                            sizeof(vidType) * rbucket_sizelists_.size(), cudaMemcpyHostToDevice));
    }


    __host__ void transform_all_neighlist_to_bitmap(Graph_V2& g) {
        LOG_INFO("Running transform all neighlist to bitmap");
        vidType nv = g.num_vertices();
        bitmap_all.resize(nv * UP_DIV(nv, 32), 0x0);

        for (int i = 0; i < nv; i++) {
            auto edge_beg = g.edge_begin(i);
            auto edge_end = g.edge_end(i);
            for (int j = edge_beg; j < edge_end; j++) {
                auto neigh = g.out_colidx()[j];
                auto index = neigh / 32;
                auto bit = neigh & 31;
                bitmap_all[i * UP_DIV(nv, 32) + index] |= (1 << (32 - 1 - bit));
            }
        }

        // copy bitmap_all to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bitmap_all, sizeof(vidType) * nv * UP_DIV(nv, 32)));
        CUDA_SAFE_CALL(cudaMemcpy(d_bitmap_all, bitmap_all.data(), 
                                sizeof(vidType) * nv * UP_DIV(nv, 32), cudaMemcpyHostToDevice));
    }

    std::vector<vidType> bitmap_all;
    vidType* d_bitmap_all = nullptr;
};

