#pragma once
#include <iostream>
#include "utils.h"
#include "../include/graph.h"
#include "../include/dataloader.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"

class CPU_Baseline {
public:
    CPU_Baseline(int use_graphpi_sched_) {
        // std::cout << "CPU_Baseline constructor" << std::endl;
        use_graphpi_sched = use_graphpi_sched_;
        if (use_graphpi_sched) {
            LOG_INFO("Using GraphPi scheduler.");
        }
    }

    ~CPU_Baseline() {
        // std::cout << "CPU_Baseline destructor" << std::endl;
    }

private:
    int use_graphpi_sched = 1;
};
