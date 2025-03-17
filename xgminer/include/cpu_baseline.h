#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "../include/graph.h"
#include "../include/dataloader.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "cmd_option.h"
#include "kernel.h"

class CPU_Baseline {
public:
    CPU_Baseline(Command_Option &opts) {
        data_name = opts.data_name;
        data_path = opts.datagraph_file;
        // g = new Graph();
        use_graphpi_sched = opts.use_graphpi_sched;
        run_graphpi = opts.run_graphpi;
        run_our_baseline = opts.run_our_baseline;
        pattern_size = opts.pattern_size;
        adj_mat = opts.pattern_adj_mat;
        patternID = opts.patternID;
    }

    ~CPU_Baseline() {
        // std::cout << "CPU_Baseline destructor" << std::endl;
    }

    void run_baseline_with_graphpi_sched();

    void run_graphpi_test();

    void run_our_baseline_test();

    void run() {
        // if (use_graphpi_sched) {
        //     LOG_INFO("Using GraphPi schedule.");

        //     DataType my_type;
        //     D.GetDataType(my_type, data_name);
        //     assert(D.load_data(g, my_type, data_path.c_str())==true); 
        //     run_baseline_with_graphpi();
        // }
        if (run_graphpi) {
            LOG_INFO("Running GraphPi algorithm.");
            run_graphpi_test();
        }
        if (run_our_baseline) {
            LOG_INFO("Running our baseline implementation.");
            if (use_graphpi_sched) {
                run_baseline_with_graphpi_sched();
            } else {
                run_our_baseline_test();
            }
        }
    }

private:
    int use_graphpi_sched = 1;
    int run_graphpi = 0;
    int run_our_baseline = 1;
    
    std::string data_name = "Wiki-Vote";
    std::string data_path = "";
    char* adj_mat;
    int pattern_size = 3;
    int patternID = 1;
    const int* p_adj_mat;
    
    int vertices, edges;
    std::vector<std::set<int>> edgeLists;
    int total_count = 0;
    std::vector<int> embedding;
    

    Kernel kernel;
};
