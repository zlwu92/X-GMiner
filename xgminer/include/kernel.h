#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "graph.h"
#include "../include/dataloader.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "cmd_option.h"
// #include "graph_v2_gpu.h"

class Kernel {
public:
    Kernel() {}
    ~Kernel() {}

    void rectangle4_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                        long long& total_count, std::vector<int>& embedding);

    void rectangle4_baseline_cpu_kernel_with_graphpi_sched(int vertices, std::vector<std::set<int>>& edgeLists,
                                        long long& total_count, std::vector<int>& embedding, 
                                        std::vector<std::vector<int>>& p_edgeList, 
                                        std::vector< std::pair<int,int>>& restrict_pair);

    void rectangle4_baseline_cpu_kernel_with_graphpi_sched_v2(int vertices, std::vector<std::set<int>>& edgeLists,
                                        long long& total_count, std::vector<int>& embedding, 
                                        std::vector<std::vector<int>>& p_edgeList, 
                                        std::vector< std::pair<int,int>>& restrict_pair);

    void motif4_glumin_p1_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding, int vert_induced);

    void motif4_glumin_p2_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding, int vert_induced);


    void motif4_glumin_p3_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding);


    // __global__ void __launch_bounds__(BLOCK_SIZE, 8)
    // GM_LUT_warp(vidType begin, vidType end, /*add begin, end!!!*/
    //                 vidType *vid_list, /*Add vid_list*/
    //                 GraphGPU g, 
    //                 vidType *vlists,
    //                 bitmapType* bitmaps,
    //                 vidType max_deg,
    //                 AccType *counter,
    //                 LUTManager<> LUTs);

};
