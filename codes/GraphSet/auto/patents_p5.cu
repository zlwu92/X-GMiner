__global__ void gpu_pattern_matching_generated(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context) {
    __shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK];
    extern __shared__ GPUVertexSet block_vertex_set[];
    GPUSchedule *schedule = context->dev_schedule;
    uint32_t *tmp = context->dev_tmp;
    uint32_t *edge = (uint32_t *)context->dev_edge;
    e_index_t *vertex = context->dev_vertex;
    uint32_t *edge_from = (uint32_t *)context->dev_edge_from;
    int wid = threadIdx.x / THREADS_PER_WARP, lid = threadIdx.x % THREADS_PER_WARP, global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;
    e_index_t &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * 6;
    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * 6;
        for (int i = 0; i < 6; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset);
            offset += buffer_size;
        }
    }
    GPUVertexSet& subtraction_set = vertex_set[4];
    __threadfence_block();
    uint32_t v0, v1;
    e_index_t l, r;
    unsigned long long sum = 0;
    while (true) {
        if (lid == 0) {
            edge_idx = atomicAdd(context->dev_cur_edge, 1);
            unsigned int i = edge_idx;
            if (i < edge_num) {
                subtraction_set.init();
                subtraction_set.push_back(edge_from[i]);
                subtraction_set.push_back(edge[i]);
            }
        }
        __threadfence_block();
        e_index_t i = edge_idx;
        if(i >= edge_num) break;
        v0 = edge_from[i];
        v1 = edge[i];
        get_edge_index(v0, l, r);
        if (threadIdx.x % THREADS_PER_WARP == 0)
            vertex_set[0].init(r - l, &edge[l]);
        __threadfence_block();
        if(v0 <= v1) continue;
        get_edge_index(v1, l, r);
        GPUVertexSet* tmp_vset;
        intersection2(vertex_set[1].get_data_ptr(), vertex_set[0].get_data_ptr(), &edge[l], vertex_set[0].get_size(), r - l, &vertex_set[1].size);
        if (vertex_set[1].get_size() == 0) continue;
        extern __shared__ char ans_array[];
        int* ans = ((int*) (ans_array + 704)) + 3 * (threadIdx.x / THREADS_PER_WARP);
        int loop_size_depth2 = vertex_set[1].get_size();
        if( loop_size_depth2 <= 0) continue;
        uint32_t* loop_data_ptr_depth2 = vertex_set[1].get_data_ptr();
        uint32_t min_vertex_depth2 = 0xffffffff;
        for(int i_depth2 = 0; i_depth2 < loop_size_depth2; ++i_depth2) {
            uint32_t v_depth2 = loop_data_ptr_depth2[i_depth2];
            if(subtraction_set.has_data(v_depth2)) continue;
            unsigned int l_depth2, r_depth2;
            get_edge_index(v_depth2, l_depth2, r_depth2);
            intersection2(vertex_set[2].get_data_ptr(), vertex_set[1].get_data_ptr(), &edge[l_depth2], vertex_set[1].get_size(), r_depth2 - l_depth2, &vertex_set[2].size);
            if (vertex_set[2].get_size() == 0) continue;
            if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.push_back(v_depth2);
            __threadfence_block();
            int loop_size_depth3 = vertex_set[2].get_size();
            if( loop_size_depth3 <= 0) continue;
            uint32_t* loop_data_ptr_depth3 = vertex_set[2].get_data_ptr();
            uint32_t min_vertex_depth3 = 0xffffffff;
            for(int i_depth3 = 0; i_depth3 < loop_size_depth3; ++i_depth3) {
                uint32_t v_depth3 = loop_data_ptr_depth3[i_depth3];
                if(subtraction_set.has_data(v_depth3)) continue;
                unsigned int l_depth3, r_depth3;
                get_edge_index(v_depth3, l_depth3, r_depth3);
                {
                tmp_vset = &vertex_set[3];
                if (threadIdx.x % THREADS_PER_WARP == 0)
                    tmp_vset->init(r_depth3 - l_depth3, &edge[l_depth3]);
                __threadfence_block();
                if (r_depth3 - l_depth3 > vertex_set[2].get_size())
                    tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[2], -1);
                else
                    tmp_vset->size = vertex_set[2].get_size() - unordered_subtraction_size(vertex_set[2], *tmp_vset, -1);
                }
                if (vertex_set[3].get_size() == 0) continue;
                if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.push_back(v_depth3);
                __threadfence_block();
                ans[0] = vertex_set[3].get_size() - 0;
                ans[1] = vertex_set[2].get_size() - 1;
                ans[2] = vertex_set[1].get_size() - 2;
                long long val;
                val = ans[0];
                val = val * ans[1];
                val = val * ans[2];
                sum += val * 1;
                val = ans[0];
                val = val * ans[1];
                sum += val * -1;
                val = ans[0];
                val = val * ans[1];
                sum += val * -1;
                val = ans[0];
                val = val * ans[2];
                sum += val * -1;
                val = ans[0];
                sum += val * 2;
                if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.pop_back();
                __threadfence_block();
            }
            if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.pop_back();
            __threadfence_block();
        }
    }
    if (lid == 0) atomicAdd(context->dev_sum, sum);
}
