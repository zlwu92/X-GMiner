#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "../include/graph.h"
#include "../include/dataloader.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"

class CPU_Baseline {
public:
    CPU_Baseline(int use_graphpi_sched_, std::string data_name_, std::string data_path_) {
        data_name = data_name_;
        data_path = data_path_;
        g = new Graph();
        // std::cout << "CPU_Baseline constructor" << std::endl;
        use_graphpi_sched = use_graphpi_sched_;
        
    }

    ~CPU_Baseline() {
        // std::cout << "CPU_Baseline destructor" << std::endl;
    }

    void run_baseline_with_graphpi();

    void run_graphpi_test();

    void run_our_baseline_test();

    void run() {
        if (use_graphpi_sched) {
            LOG_INFO("Using GraphPi schedule.");

            DataType my_type;
            D.GetDataType(my_type, data_name);
            assert(D.load_data(g, my_type, data_path.c_str())==true); 
            run_baseline_with_graphpi();
        }
        if (run_graphpi) {
            LOG_INFO("Running GraphPi algorithm.");
            run_graphpi_test();
        }
        if (run_our_baseline) {
            LOG_INFO("Running our baseline implementation.");
            run_our_baseline_test();
        }
    }

private:
    int use_graphpi_sched = 1;
    int run_graphpi = 0;
    int run_our_baseline = 1;
    Graph *g;
    DataLoader D;
    std::string data_name = "Wiki-Vote";
    std::string data_path = "";
};
