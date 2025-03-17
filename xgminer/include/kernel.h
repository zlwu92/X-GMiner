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


class Kernel {
public:
    Kernel() {}
    ~Kernel() {}

    void rectangle4_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                        int& total_count, std::vector<int>& embedding);
};
