#include "../include/graph.h"
#include "../include/graphmpi.h"
#include "../include/vertex_set.h"
#include "../include/common.h"
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <atomic>
#include <queue>
#include <iostream>

int Graph::intersection_size(int v1,int v2) {
    unsigned int l1, r1;
    get_edge_index(v1, l1, r1);
    unsigned int l2, r2;
    get_edge_index(v2, l2, r2);
    int ans = 0;
    while(l1 < r1 && l2 < r2) {
        if(edge[l1] < edge[l2]) {
            ++l1;
        }
        else {
            if(edge[l2] < edge[l1]) {
                ++l2;
            }
            else {
                ++l1;
                ++l2;
                ++ans;
            }
        }
    }
    return ans;
}

/*int Graph::intersection_size_mpi(int v1, int v2) {
    Graphmpi &gm = Graphmpi::getinstance();
    int ans = 0;
    if (gm.include(v2))
        return intersection_size(v1, v2);
    unsigned int l1, r1;
    get_edge_index(v1, l1, r1);
    int *data = gm.getneighbor(v2);
    for (int l2 = 0; l1 < r1 && ~data[l2];) {
        if(edge[l1] < data[l2]) {
            ++l1;
        }
        else if(edge[l1] > data[l2]) {
            ++l2;
        }
        else {
            ++l1;
            ++l2;
            ++ans;
        }
    }
    return ans;
}
*/

int Graph::intersection_size_clique(int v1,int v2) {
    unsigned int l1, r1;
    get_edge_index(v1, l1, r1);
    unsigned int l2, r2;
    get_edge_index(v2, l2, r2);
    int min_vertex = v2;
    int ans = 0;
    if (edge[l1] >= min_vertex || edge[l2] >= min_vertex)
        return 0;
    while(l1 < r1 && l2 < r2) {
        if(edge[l1] < edge[l2]) {
            if (edge[++l1] >= min_vertex)
                break;
        }
        else {
            if(edge[l2] < edge[l1]) {
                if (edge[++l2] >= min_vertex)
                    break;
            }
            else {
                ++ans;
                if (edge[++l1] >= min_vertex)
                    break;
                if (edge[++l2] >= min_vertex)
                    break;
            }
        }
    }
    return ans;
}

long long Graph::triangle_counting() {
    long long ans = 0;
    for(int v = 0; v < v_cnt; ++v) {
        // for v in G
        unsigned int l, r;
        get_edge_index(v, l, r);
        for(unsigned int v1 = l; v1 < r; ++v1) {
            //for v1 in N(v)
            ans += intersection_size(v,edge[v1]);
        }
    }
    ans /= 6;
    return ans;
}

long long Graph::triangle_counting_mt(int thread_count) {
    long long ans = 0;
#pragma omp parallel num_threads(thread_count)
    {
        tc_mt(&ans);
    }
    return ans;
}

void Graph::tc_mt(long long *global_ans) {
    long long my_ans = 0;
    #pragma omp for schedule(dynamic)
    for(int v = 0; v < v_cnt; ++v) {
        // for v in G
        unsigned int l, r;
        get_edge_index(v, l, r);
        for(unsigned int v1 = l; v1 < r; ++v1) {
            if (v <= edge[v1])
                break;
            //for v1 in N(v)
            my_ans += intersection_size_clique(v,edge[v1]);
        }
    }
    #pragma omp critical
    {
        *global_ans += my_ans;
    }
}

void Graph::get_edge_index(int v, unsigned int& l, unsigned int& r) const
{
    l = vertex[v];
    r = vertex[v + 1];
}

void Graph::pattern_matching_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;
    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    /*if (clique == true)
      {
      int last_vertex = subtraction_set.get_last();
    // The number of this vertex must be greater than the number of last vertex.
    loop_start = std::upper_bound(loop_data_ptr, loop_data_ptr + loop_size, last_vertex) - loop_data_ptr;
    }*/
    if (depth == schedule.get_size() - 1)
    {
        // TODO : try more kinds of calculation.
        // For example, we can maintain an ordered set, but it will cost more to maintain itself when entering or exiting recursion.
        if (clique == true)
            local_ans += loop_size;
        else if (loop_size > 0)
            local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
        return;
    }

    int last_vertex = subtraction_set.get_last();
    for (int i = 0; i < loop_size; ++i)
    {
        if (last_vertex <= loop_data_ptr[i] && clique == true)
            break;
        int vertex = loop_data_ptr[i];
        if (!clique)
            if (subtraction_set.has_data(vertex))
                continue;
        unsigned int l, r;
        get_edge_index(vertex, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex, clique);
            if( vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if( is_zero ) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        pattern_matching_func(schedule, vertex_set, subtraction_set, local_ans, depth + 1, clique);
        subtraction_set.pop_back();
    }
}

long long Graph::pattern_matching(const Schedule& schedule, int thread_count, bool clique, bool print)
{
    long long global_ans = 0;
    // printf("in_exclusion_optimize_redundancy = %ld\n", schedule.get_in_exclusion_optimize_redundancy());
#pragma omp parallel num_threads(thread_count) reduction(+: global_ans)
    {
        double start_time = get_wall_time();
        double current_time;
        VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
        if (print) {
            int total_prefix_num = schedule.get_total_prefix_num();
            printf("total_prefix_num = %d\n", schedule.get_total_prefix_num());
            printf("in_exclusion_optimize_redundancy = %ld\n", schedule.get_in_exclusion_optimize_redundancy());
            printf("VertexSet::max_intersection_size = %d\n", VertexSet::max_intersection_size);

            int max_prefix_num = schedule.get_size() * (schedule.get_size() - 1) / 2;
            printf("max_prefix_num = %d\n", max_prefix_num);
            // print schedule.last and schedule.next and father_prefix_id
            for (int i = 0; i < schedule.get_size(); ++i) { // pattern size: 4
                printf("last[%d] = %d ", i, schedule.get_last(i));
            }
            puts("");
            for (int i = 0; i < max_prefix_num; ++i) {
                printf("next[%d] = %d ", i, schedule.get_next(i));
            }
            puts("");
            for (int i = 0; i < max_prefix_num; ++i) {
                printf("father_prefix_id[%d] = %d ", i, schedule.get_father_prefix_id(i));
            }
            puts("");
            // print schedule.loop_set_prefix_id
            for (int i = 0; i < schedule.get_size(); ++i) {
                printf("loop_set_prefix_id[%d]=%d ", i, schedule.get_loop_set_prefix_id(i));
            }
            puts("");
            // print schedule.prefix
            for (int i = 0; i < total_prefix_num; ++i) {
                printf("prefix[%d].size = %d: ", i, schedule.prefix[i].get_size());
                for (int j = 0; j < schedule.prefix[i].get_size(); ++j) {
                    printf("%d ", schedule.prefix[i].get_data(j));
                }
                puts("");
            }
            puts("\n===========");
            // print schedule.adj_mat
            // for (int i = 0; i < schedule.get_size(); ++i) {
            //     for (int j = 0; j < schedule.get_size(); ++j) {
            //         printf("%d", schedule.get_adj_mat_ptr()[INDEX(i, j, schedule.get_size())]);
            //     }
            //     puts("");
            // }
        }
        
        VertexSet subtraction_set;
        VertexSet tmp_set;
        subtraction_set.init();
        long long local_ans = 0;
        // TODO : try different chunksize
#pragma omp for schedule(dynamic) nowait
        for (int vertex = 0; vertex < v_cnt; ++vertex)
        {
            unsigned int l, r;
            get_edge_index(vertex, l, r);
            if (print) {
                printf("vertex=%d, l=%d, r=%d, edge[l]=%d\n", vertex, l, r, edge[l]);
                printf("before build_vertex_set\n");
                // print all data in vertex_set
                for (int i = 0; i < schedule.get_total_prefix_num(); ++i) {
                    printf("vertex_set[%d].size = %d: ", i, vertex_set[i].get_size());
                    for (int j = 0; j < vertex_set[i].get_size(); ++j) {
                        printf("%d ", vertex_set[i].get_data(j));
                    }
                    puts("");
                }
                // print edge
                for (int i = l; i < r; ++i) {
                    printf("edge[%d]=%d ", i, edge[i]);
                }
                puts("");
            }
            for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
            {
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id);
                if (print)  printf("vertex_set[prefix_id=%d].size = %d\n", prefix_id, vertex_set[prefix_id].get_size());
            }
            //subtraction_set.insert_ans_sort(vertex);
            subtraction_set.push_back(vertex);
            if (print) {
                printf("after build_vertex_set\n");
                for (int i = 0; i < schedule.get_total_prefix_num(); ++i) {
                    printf("vertex_set[%d].size = %d: ", i, vertex_set[i].get_size());
                    for (int j = 0; j < vertex_set[i].get_size(); ++j) {
                        printf("%d ", vertex_set[i].get_data(j));
                    }
                    puts("");
                }
                // print subtraction_set
                // printf("subtraction_set.size = %d: ", subtraction_set.get_size());
                // for (int i = 0; i < subtraction_set.get_size(); ++i) {
                //     printf("%d ", subtraction_set.get_data(i));
                // }
                // puts("");
            }
            //if (schedule.get_total_restrict_num() > 0 && clique == false)
            if(true) {
                // printf("pattern_matching_aggressive_func\n");
                pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, 1, print);
            }
            else
                pattern_matching_func(schedule, vertex_set, subtraction_set, local_ans, 1, clique);
            subtraction_set.pop_back();
            /*
            if( (vertex & (-vertex)) == (1<<15) ) {
                current_time = get_wall_time();
                if( current_time - start_time > max_running_time) {
                    printf("TIMEOUT!\n");
                    fflush(stdout);
                    assert(0);
                }
            }*/
            if (print) {
                printf("vertex = %d local_ans = %ld\n", vertex, local_ans);
                printf("depth=0, subtraction_set.size = %d: ", subtraction_set.get_size());
                for (int i = 0; i < subtraction_set.get_size(); ++i) {
                    printf("%d ", subtraction_set.get_data(i));
                }
                puts("");
                puts("=========");
            }
        }
        delete[] vertex_set;
        // TODO : Computing multiplicty for a pattern
        global_ans += local_ans;
        // printf("local_ans = %ld global_ans = %ld\n", local_ans, global_ans);
    }
    return global_ans / schedule.get_in_exclusion_optimize_redundancy();
}


void Graph::pattern_matching_aggressive_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, 
                        VertexSet& tmp_set, long long& local_ans, int depth, bool print) // 3 same # or @ in comment are useful in code generation ###
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);// @@@
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;
    if (print) {
        printf("##in_exclusion_optimize_num = %d\n", schedule.get_in_exclusion_optimize_num());
        printf("##total_restrict_num = %d\n", schedule.get_total_restrict_num());
        printf("##depth = %d local_ans = %ld\n", depth, local_ans);
        printf("depth=%d, subtraction_set.size = %d: ", depth, subtraction_set.get_size());
        for (int i = 0; i < subtraction_set.get_size(); ++i) {
            printf("%d ", subtraction_set.get_data(i));
        }
        puts("");
    }
    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
/* @@@ 
    //Case: in_exclusion_optimize_num = 2
    if (depth == schedule.get_size() - 2 && schedule.get_in_exclusion_optimize_num() == 2) { 
        int loop_set_prefix_id_nxt = schedule.get_loop_set_prefix_id( depth + 1);
        int loop_size_nxt = vertex_set[loop_set_prefix_id_nxt].get_size();
        int size1 = VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
        int size2 = VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id_nxt], subtraction_set);
        VertexSet tmp_set;
        tmp_set.init();
        tmp_set.intersection(vertex_set[loop_set_prefix_id], vertex_set[loop_set_prefix_id_nxt]);
        int size3 = VertexSet::unorderd_subtraction_size(tmp_set, subtraction_set);
        local_ans += 1ll * size1 * size2 - size3;
        return;
    }
*/
/*
    //Case: in_exclusion_optimize_num = 3
    if( depth == schedule.get_size() - 3 && schedule.get_in_exclusion_optimize_num() == 3) { 
        int in_exclusion_optimize_num = 3;
        int loop_set_prefix_ids[ in_exclusion_optimize_num];
        for(int i = 0; i < in_exclusion_optimize_num; ++i)
            loop_set_prefix_ids[i] = schedule.get_loop_set_prefix_id( depth + i );
        
        int loop_sizes[ in_exclusion_optimize_num ];
        for(int i = 0; i < in_exclusion_optimize_num; ++i)
            loop_sizes[i] = VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_ids[i]], subtraction_set);
        
        local_ans += 1ll * loop_sizes[0] * loop_sizes[1] * loop_sizes[2];

        for(int i = 1; i < 3; ++i) 
            for(int j = 0; j < i; ++j){
                VertexSet tmp_set;
                tmp_set.init();
                tmp_set.intersection(vertex_set[loop_set_prefix_ids[i]], vertex_set[loop_set_prefix_ids[j]]);
                int tmp_size = VertexSet::unorderd_subtraction_size(tmp_set, subtraction_set);
                int size2;
                for(int k = 0; k < 3; ++k)
                    if( i != k && j != k) size2 = loop_sizes[k];
                local_ans -= 1ll * tmp_size * size2;
            }
        VertexSet tmp1;
        tmp1.init();
        tmp1.intersection(vertex_set[loop_set_prefix_ids[0]], vertex_set[loop_set_prefix_ids[1]]);
        VertexSet tmp2;
        tmp2.init();
        tmp2.intersection(vertex_set[loop_set_prefix_ids[2]], tmp1);
        local_ans += 1ll * 2 * VertexSet::unorderd_subtraction_size(tmp2, subtraction_set);
        return;
    }
*/
    //Case: in_exclusion_optimize_num > 1
    if( depth == schedule.get_size() - schedule.get_in_exclusion_optimize_num() ) {
        if (print) {
            printf("!!in_exclusion_optimize_num = %d\n", schedule.get_in_exclusion_optimize_num());
            printf("!!depth = %d\n", depth);
        }
        int in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();// @@@
        int loop_set_prefix_ids[ in_exclusion_optimize_num ];
        loop_set_prefix_ids[0] = loop_set_prefix_id;
        for(int i = 1; i < in_exclusion_optimize_num; ++i)
            loop_set_prefix_ids[i] = schedule.get_loop_set_prefix_id( depth + i );
        for(int optimize_rank = 0; optimize_rank < schedule.in_exclusion_optimize_group.size(); ++optimize_rank) {
            const std::vector< std::vector<int> >& cur_graph = schedule.in_exclusion_optimize_group[optimize_rank];
            long long val = schedule.in_exclusion_optimize_val[optimize_rank];
            for(int cur_graph_rank = 0; cur_graph_rank < cur_graph.size(); ++ cur_graph_rank) {
                //                VertexSet tmp_set;
                
                //if size == 1 , we will not call intersection(...)
                //so we will not allocate memory for data
                //otherwise, we need to copy the data to do intersection(...)
                if(cur_graph[cur_graph_rank].size() == 1) {
                    int id = loop_set_prefix_ids[cur_graph[cur_graph_rank][0]];
                    val = val * VertexSet::unorderd_subtraction_size(vertex_set[id], subtraction_set);
                }
                else {
                    int id0 = loop_set_prefix_ids[cur_graph[cur_graph_rank][0]];
                    int id1 = loop_set_prefix_ids[cur_graph[cur_graph_rank][1]];
                    tmp_set.init(this->max_degree);
                    tmp_set.intersection(vertex_set[id0], vertex_set[id1]);

                    for(int i = 2; i < cur_graph[cur_graph_rank].size(); ++i) {
                        int id = loop_set_prefix_ids[cur_graph[cur_graph_rank][i]];
                        tmp_set.intersection_with(vertex_set[id]);
                    }
                    val = val * VertexSet::unorderd_subtraction_size(tmp_set, subtraction_set);
                }
                if( val == 0 ) break;

            }
            local_ans += val;
        }
        return;// @@@
            
    }
    //Case: in_exclusion_optimize_num <= 1
    if (depth == schedule.get_size() - 1)
    {
        if (print) {
            printf("@@depth = %d\n", depth);
            printf("@@loop_set_prefix_id = %d\n", loop_set_prefix_id);
            printf("@@loop_size = %d\n", loop_size);
        }
        // TODO : try more kinds of calculation. @@@
        // For example, we can maintain an ordered set, but it will cost more to maintain itself when entering or exiting recursion.
        if (schedule.get_total_restrict_num() > 0)
        {
            int min_vertex = v_cnt;
            for (int i = schedule.get_restrict_last(depth); i != -1; i = schedule.get_restrict_next(i))
                if (min_vertex > subtraction_set.get_data(schedule.get_restrict_index(i)))
                    min_vertex = subtraction_set.get_data(schedule.get_restrict_index(i));
            const VertexSet& vset = vertex_set[loop_set_prefix_id];
            int size_after_restrict = std::lower_bound(vset.get_data_ptr(), vset.get_data_ptr() + vset.get_size(), min_vertex) - vset.get_data_ptr();
            int tmp_result = 0;
            if (size_after_restrict > 0) {
                if (print) {
                    printf("\033[33mvertex_set[loop_set_prefix_id].size() = %d: ", vertex_set[loop_set_prefix_id].get_size());
                    for (int i = 0; i < vertex_set[loop_set_prefix_id].get_size(); ++i) {
                        printf("%d ", vertex_set[loop_set_prefix_id].get_data(i));
                    }
                    puts("\033[0m");
                }
                // local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set, size_after_restrict);
                tmp_result = VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set, size_after_restrict);
                local_ans += tmp_result;
            }
            if (print) {
                printf("\033[33mmin_vertex = %d size_after_restrict = %d\033[0m\n", min_vertex, size_after_restrict);
                // print vset
                printf("vset.size = %d: ", vset.get_size());
                for (int i = 0; i < vset.get_size(); ++i) {
                    printf("%d ", vset.get_data(i));
                }
                puts("");
                printf("\033[33munorderd_subtraction_size = %d\033[0m\n", tmp_result);
            }
        }
        else
            local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set); 
        return;// @@@
    }
  
    // TODO : min_vertex is also a loop invariant @@@
    int min_vertex = v_cnt;
    for (int i = schedule.get_restrict_last(depth); i != -1; i = schedule.get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule.get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule.get_restrict_index(i));
    if (depth == 1) Graphmpi::getinstance().get_loop(loop_data_ptr, loop_size);
    if (print) {
        printf("min_vertex = %d loop_size = %d\n", min_vertex, loop_size);
        for (int i = 0; i < loop_size; ++i) {
            printf("loop_data_ptr[%d] = %d ", i, loop_data_ptr[i]);
        }
        puts("");
    }
    int ii = 0;
    for (int &i = ii; i < loop_size; ++i)
    {
        if (min_vertex <= loop_data_ptr[i])
            break;
        int vertex = loop_data_ptr[i];
        if (subtraction_set.has_data(vertex))
            continue;
        unsigned int l, r;
        get_edge_index(vertex, l, r);
        bool is_zero = false;
        if (print) {
            printf("\033[32mvertex=%d, l=%d, r=%d, edge[l]=%d\033[0m\n", vertex, l, r, edge[l]);
            printf("\033[32mbefore build_vertex_set\033[0m\n");
            for (int i = 0; i < schedule.get_total_prefix_num(); ++i) {
                printf("vertex_set[%d].size = %d: ", i, vertex_set[i].get_size());
                for (int j = 0; j < vertex_set[i].get_size(); ++j) {
                    printf("%d ", vertex_set[i].get_data(j));
                }
                puts("");
            }
            // print edge
            for (int i = l; i < r; ++i) {
                printf("edge[%d]=%d ", i, edge[i]);
            }
            puts("");
        }
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex);
            if (print)  printf("\033[32mdepth=%d, vertex_set[prefix_id=%d].size = %d\033[0m\n", depth, prefix_id, vertex_set[prefix_id].get_size());
            if( vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (print) {
            printf("\033[32mafter build_vertex_set\033[0m\n");
            for (int i = 0; i < schedule.get_total_prefix_num(); ++i) {
                printf("vertex_set[%d].size = %d: ", i, vertex_set[i].get_size());
                for (int j = 0; j < vertex_set[i].get_size(); ++j) {
                    printf("%d ", vertex_set[i].get_data(j));
                }
                puts("");
            }
            printf("\033[32mis_zero = %d\033[0m\n", is_zero);
            printf("before recursion\n");
            printf("depth=%d, subtraction_set.size = %d: ", depth, subtraction_set.get_size());
            for (int i = 0; i < subtraction_set.get_size(); ++i) {
                printf("%d ", subtraction_set.get_data(i));
            }
            puts("");
        }
        if( is_zero ) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1, print);// @@@
        subtraction_set.pop_back(); // @@@
        if (print) {
            printf("after recursion\n");
            printf("depth=%d, subtraction_set.size = %d: ", depth, subtraction_set.get_size());
            for (int i = 0; i < subtraction_set.get_size(); ++i) {
                printf("%d ", subtraction_set.get_data(i));
            }
            puts("");
        }
    }
    if (print) {
        printf("$$ depth=%d, subtraction_set.size = %d: ", depth, subtraction_set.get_size());
        for (int i = 0; i < subtraction_set.get_size(); ++i) {
            printf("%d ", subtraction_set.get_data(i));
        }
        puts("");
    } 
    //if (depth == 1 && ii < loop_size) Graphmpi::getinstance().set_cur(subtraction_set.get_data(0));// @@@
} 
// ###
long long Graph::pattern_matching_mpi(const Schedule& schedule, int thread_count, bool clique)
{
    Graphmpi &gm = Graphmpi::getinstance();
    long long global_ans = 0;
#pragma omp parallel num_threads(thread_count)
    {
#pragma omp master
        {
            gm.init(thread_count, this, schedule);
        }
#pragma omp barrier //mynodel have to be calculated before running other threads
#pragma omp master
        {
            global_ans = gm.runmajor();
        }
        if (omp_get_thread_num()) {
            VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
            long long local_ans = 0;
            VertexSet subtraction_set;
            VertexSet tmp_set;
            subtraction_set.init();
            int last = -1;
            gm.set_loop_flag();
            auto match_edge = [&](int vertex, int *data, int size) {
                if (vertex != last) {
                    if (~last) subtraction_set.pop_back();
                    unsigned int l, r;
                    get_edge_index(vertex, l, r);
                    for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id)) {
                        vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, edge + l, r - l, prefix_id);
                    }
                    subtraction_set.push_back(vertex);
                    last = vertex;
                }
                gm.set_loop(data, size);
                pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, 1);
            };
            for (unsigned int *data; data = gm.get_edge_range();) {
                match_edge(data[1], edge + data[2], data[3] - data[2]);
                /*for (int i = 1; i <= data[4]; i++) {
                    int l, r;
                    get_edge_index(data[1] + i, l, r);
                    match_edge(data[1] + i, edge + l, r - l);
                }*/
            }
            delete[] vertex_set;
            gm.report(local_ans);
        }
    }
    return global_ans;
}
