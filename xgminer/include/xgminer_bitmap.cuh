#pragma once
#include <iostream>
#include "utils.h"
#include "../include/graph_v2.h"
#include "../include/common.h"
// #include <memory> // std::unique_ptr


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

    // XGMiner_BITMAP() : 

    vidType bigset_bucket_num = 0;
    std::vector<vidType> bucket_vlists_;
    std::vector<vidType> bucket_sizelists_;
    std::vector<std::vector<uint64_t>> bitmaps_;
    vidType* d_bucket_vlists_ = nullptr;
    vidType* d_bucket_sizelists_ = nullptr;
    T* d_bitmaps_ = nullptr;

    __device__ __forceinline__ VertexArrayView intersection(T* bmap1, T* bmap2, int size1, int size2, 
                                                StorageMeta& meta, int bitmap_id, int slot_id,
                                                int* nonzero_bucket_id,
                                                vidType BUCKET_NUM) {
        T* result = meta.bitmap64(bitmap_id);
        __shared__ vidType count[WARPS_PER_BLOCK * 2];
        int warp_lane   = threadIdx.x / WARP_SIZE;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            count[warp_lane * 2] = 0;
            count[warp_lane * 2 + 1] = 0;
        }

        for (auto i = thread_lane; i < size1; i += WARP_SIZE) {
            unsigned active = __activemask();
            __syncwarp(active);
            result[i] = bmap1[i] & bmap2[i];
            // if (result[i])  count[warp_lane * 4 + i] = 1;
            bool found = result[i] != 0;
            unsigned mask = __ballot_sync(active, found);
            auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
            if (found)  nonzero_bucket_id[warp_lane * BUCKET_NUM + count[warp_lane * 2]+idx-1] = i+1;
            if (thread_lane == 0) count[warp_lane * 2] += __popc(mask);
        }
        // __syncwarp();
        // if (warp_lane == 0 && thread_lane == 0) {
        //     printf("count[0]: %d\n", count[warp_lane]);
        // }
        vidType* buffer = meta.buffer(slot_id);
        for (int i = 0; i < count[warp_lane]; i++) {
            auto res = result[nonzero_bucket_id[i] - 1];
            for (int j = thread_lane; j < 64; j += WARP_SIZE) {
                unsigned active = __activemask();
                __syncwarp(active);
                // get each bit of res
                auto bit = (res >> (W - 1 - j)) & 1;
                unsigned mask = __ballot_sync(active, bit);
                auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
                if (bit)    buffer[count[warp_lane * 2 + 1]+idx-1] = j;
                if (thread_lane == 0) count[warp_lane * 2 + 1] += __popc(mask);
            }
        }
        if (warp_lane == 1 && thread_lane == 0) {
            printf("count[1]: %d; %d %d\n", count[warp_lane * 2 + 1], buffer[0], buffer[1]);
        }
        // return result;
        // return buffer;
        return VertexArrayView(buffer, count[warp_lane * 2 + 1]);
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
                num += 1;
            }
            unsigned active = __activemask();
            unsigned mask = __ballot_sync(active, is_larger);
            if (mask != FULL_MASK) break;
        }
        return num;
    }

    // template <typename D = vidType>
    __forceinline__ __device__ vidType difference_num(VertexArrayView a, VertexArrayView b, vidType lower_bound) {
        return difference_num_bs(a.ptr(), a.size(), b.ptr(), b.size(), lower_bound);
    }



    __host__ void bigset_bitmap_processing(Graph_V2& g) {
        LOG_INFO("Running bigset bitmap processing");
        std::cout << "W: " << W << " " << bigset_bucket_num / W << "\n";
        vidType nv = g.num_vertices();
        eidType ne = g.num_edges();
        // bucket_vlists_.resize(nv);
        bucket_sizelists_.resize(nv * bigset_bucket_num, 0);
        bitmaps_.resize(nv);
        int bmap_size = bigset_bucket_num / W;
        for (int i = 0; i < nv; i++) {
            // auto vlist = g.N(i);
            auto edge_beg = g.edge_begin(i);
            auto edge_end = g.edge_end(i);
            // bucket_vlists_[i].resize(bigset_bucket_num);
            std::cout << "vertex " << i << ": ";
            bitmaps_[i].resize(bigset_bucket_num / W, 0x0);
            for (int j = edge_beg; j < edge_end; j++) {
                auto neigh = g.out_colidx()[j];
                int bucket_id = neigh & (bigset_bucket_num - 1);
                bucket_vlists_.push_back(neigh);
                bucket_sizelists_[i * bucket_id]++;
                
                int bucket_ele = bucket_id / W;
                int bucket_bit = bucket_id % W;
                std::cout << neigh << "[" << bucket_id << " " << bucket_ele << " " << bucket_bit << "] ";
                bitmaps_[i][bucket_ele] |= (1ULL << (W - 1 - bucket_bit));
                // for (int s = 0; s < 64; ++s) {
                //     printf("%lu", (bitmaps_[i][bucket_ele] >> (W-1-s)) & 1);
                // }
                // std::cout << "\n";
            }
            std::cout << "\n";
        }
        // std::cout << "bitmaps_[0][0]: " << bitmaps_[0][0] << "\n";
        // for (int i = 0; i < 64; ++i) {
        //     printf("%lu", (bitmaps_[0][0] >> (63-i)) & 1);
        // }
        // std::cout << "\n";
        for (auto val : bitmaps_[0]) {
            std::cout << val << " ";
        }
        std::cout << "\n";

        // copy bitmaps_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bitmaps_, sizeof(T) * nv * (bigset_bucket_num / W)));
        for (int i = 0; i < nv; i++) {
            CUDA_SAFE_CALL(cudaMemcpy(d_bitmaps_ + i * (bigset_bucket_num / W), 
                            bitmaps_[i].data(), sizeof(T) * (bigset_bucket_num / W), cudaMemcpyHostToDevice));
        }

        // copy bucket_vlists_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bucket_vlists_, sizeof(vidType) * bucket_vlists_.size()));
        CUDA_SAFE_CALL(cudaMemcpy(d_bucket_vlists_, bucket_vlists_.data(), 
                            sizeof(vidType) * bucket_vlists_.size(), cudaMemcpyHostToDevice));
        // copy bucket_sizelists_ to device
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_bucket_sizelists_, sizeof(vidType) * bucket_sizelists_.size()));
        CUDA_SAFE_CALL(cudaMemcpy(d_bucket_sizelists_, bucket_sizelists_.data(), 
                            sizeof(vidType) * bucket_sizelists_.size(), cudaMemcpyHostToDevice));
    }
};
// 00000000000000000000000000000000
