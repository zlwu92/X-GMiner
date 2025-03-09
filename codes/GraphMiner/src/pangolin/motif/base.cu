// Copyright (c) MIT 2021
// Author: Xuhao Chen
#include "motif.h"
#include "timer.h"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define USE_PID
//#define USE_WEDGE
#define USE_SIMPLE
#define VERTEX_INDUCED
#include "pangolin_gpu/miner.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

__global__ void extend_alloc(unsigned m, unsigned level, GraphGPU graph, EmbeddingList emb_list, IndexT *num_new_emb) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
  if(pos < m) {
    IndexT num = 0;
    emb_list.get_embedding(level, pos, emb[tid]);
    for (unsigned i = 0; i < level+1; ++i) {
      IndexT src = emb[tid][i];
      IndexT row_begin = graph.edge_begin(src);
      IndexT row_end = graph.edge_end(src);
      for (IndexT e = row_begin; e < row_end; e++) {
        IndexT dst = graph.getEdgeDst(e);
        if (!is_vertexInduced_automorphism(level+1, emb[tid], i, src, dst, graph))
          num ++;
      }
    }
    num_new_emb[pos] = num;
  }
}

__global__ void extend_insert(unsigned m, unsigned max_size, unsigned level, GraphGPU graph, EmbeddingList emb_list, IndexT *indices) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
  if(pos < m) {
    emb_list.get_embedding(level, pos, emb[tid]);
    IndexT start = indices[pos];
    for (unsigned i = 0; i < level+1; ++i) {
      IndexT src = emb[tid][i];
      IndexT row_begin = graph.edge_begin(src);
      IndexT row_end = graph.edge_end(src);
      for (IndexT e = row_begin; e < row_end; e++) {
        IndexT dst = graph.getEdgeDst(e);
        if (!is_vertexInduced_automorphism(level+1, emb[tid], i, src, dst, graph)) {
          if (level == 1 && max_size == 4)
            emb_list.set_pid(start, find_3motif_pattern_id(i, dst, emb[tid], graph, start));
          emb_list.set_idx(level+1, start, pos);
          emb_list.set_vid(level+1, start++, dst);
        }
      }
    }
  }
}

__global__ void aggregate(unsigned m, unsigned level, unsigned npatterns, GraphGPU graph, EmbeddingList emb_list, AccType *accumulators) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
  AccType local_num[6];
  for (int i = 0; i < npatterns; i++) local_num[i] = 0;
  if(pos < m) {
    unsigned pattern = 0;
    emb_list.get_embedding(level, pos, emb[tid]);
    unsigned n = level+1;
    assert(n < 4);
    if (n == 3) pattern = emb_list.get_pid(pos);
    for (unsigned i = 0; i < n; ++i) {
      IndexT src = emb[tid][i];
      IndexT row_begin = graph.edge_begin(src);
      IndexT row_end = graph.edge_end(src);
      for (IndexT e = row_begin; e < row_end; e++) {
        IndexT dst = graph.getEdgeDst(e);
        if (!is_vertexInduced_automorphism(n, emb[tid], i, src, dst, graph)) {
          unsigned pid = 1; // 3-chain
          if (n == 2) pid = find_3motif_pattern_id(i, dst, emb[tid], graph, pos);
          else pid = find_4motif_pattern_id(n, i, dst, emb[tid], pattern, graph, pos);
          local_num[pid] += 1;
        }
      }
    }
  }
  AccType block_num;
  for (int i = 0; i < npatterns; i++) {
    atomicAdd(&accumulators[i], local_num[i]);
  }
}

__global__ void clear(AccType *accumulators) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  accumulators[i] = 0;
}

void parallel_prefix_sum(int n, IndexT *in, IndexT *out) {
  IndexT total = 0;
  for (size_t i = 0; i < n; i++) {
    out[i] = total;
    total += in[i];
  }
  out[n] = total;
}

void MotifSolver(Graph &g, unsigned k, std::vector<uint64_t> &counts) {
  print_device_info(0);
  size_t npatterns = counts.size();
  AccType *h_accumulators = (AccType *)malloc(sizeof(AccType) * npatterns);
  for (int i = 0; i < npatterns; i++) h_accumulators[i] = 0;
  AccType *d_accumulators;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_accumulators, sizeof(AccType) * npatterns));
  clear<<<1, npatterns>>>(d_accumulators);
  int m = g.num_vertices();
  int nnz = g.num_edges();
  int nthreads = BLOCK_SIZE;
  int nblocks = DIVIDE_INTO(m, nthreads);
  std::cout << "CUDA " << k << "-motifs counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  GraphGPU gg(g);
  EmbeddingList emb_list;
  emb_list.init(nnz, k, false);
  emb_list.init_cpu(g);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  unsigned level = 1;
  unsigned num_emb = emb_list.size();
  while (level < k-2) {
    //std::cout << "number of embeddings in level " << level << ": " << num_emb << "\n";
    IndexT *num_new_emb, *indices;
    CUDA_SAFE_CALL(cudaMalloc((void **)&num_new_emb, sizeof(IndexT) * (num_emb+1)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&indices, sizeof(IndexT) * (num_emb+1)));
    nblocks = (num_emb-1)/nthreads+1;
    extend_alloc<<<nblocks, nthreads>>>(num_emb, level, gg, emb_list, num_new_emb);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    thrust::exclusive_scan(thrust::device, num_new_emb, num_new_emb+num_emb+1, indices);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    IndexT new_size;
    CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[num_emb], sizeof(IndexT), cudaMemcpyDeviceToHost));
    //std::cout << "number of new embeddings: " << new_size << "\n";
    emb_list.add_level(new_size);
#ifdef USE_WEDGE
    //if (level == 1 && max_size == 4) {
    //	is_wedge.resize(emb_list.size());
    //	std::fill(is_wedge.begin(), is_wedge.end(), 0);
    //}
#endif
    extend_insert<<<nblocks, nthreads>>>(num_emb, k, level, gg, emb_list, indices);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //std::cout << "Extend_insert Done\n";
    num_emb = emb_list.size();
    CUDA_SAFE_CALL(cudaFree(num_new_emb));
    CUDA_SAFE_CALL(cudaFree(indices));
    level ++;
  }
  if (k < 5) {
    nblocks = (num_emb-1)/nthreads+1;
    aggregate<<<nblocks, nthreads>>>(num_emb, level, npatterns, gg, emb_list, d_accumulators);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  } else {
    std::cout << "Not supported\n";
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [gpu_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(h_accumulators, d_accumulators, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_accumulators));
  for (int i = 0; i < npatterns; i++) counts[i] = h_accumulators[i];
}

