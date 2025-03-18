#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "../include/graph_v2.h"
#include "../include/dataloader.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "cmd_option.h"
#include "kernel.h"

class GLUMIN {
public:
    GLUMIN(Command_Option &opts) {
        data_name = opts.data_name;
        data_path = opts.datagraph_file;
        use_graphpi_sched = opts.use_graphpi_sched;
        pattern_size = opts.pattern_size;
        adj_mat = opts.pattern_adj_mat;
        patternID = opts.patternID;
        file_format = getFileFormat(data_path);
        do_validation = opts.do_validation;
        algo = opts.algo;
        total.resize(num_patterns, 0);
    }

    ~GLUMIN() {
        // std::cout << "CPU_Baseline destructor" << std::endl;
    }

    void load_graph_data_from_file() {
        local_patternId = patternID - 16;
        if (file_format == Input_FileFormat::BINARY) { // stored in CSR format
            if (local_patternId == 4 || local_patternId == 5 || local_patternId == 23 || local_patternId == 24) {
                use_dag = true;
            }
            std::string prefix = data_path + "/graph";
            Graph_V2 g(prefix, use_dag);
        }
    }

    void run_glumin_g2miner();

    void run_glumin_g2miner_lut();

    void run_glumin_graphfold();

    void run_glumin_graphfold_lut();

    void run_glumin_automine();

    void run_glumin_automine_lut();

    void run() {
        load_graph_data_from_file();
        if (algo == "glumin_g2miner") {
            run_glumin_g2miner();
        } else if (algo == "glumin_g2miner_lut") {
            run_glumin_g2miner_lut();
        } else if (algo == "glumin_gf") {
            run_glumin_graphfold();
        } else if (algo == "glumin_gf_lut") {
            run_glumin_graphfold_lut();
        } else if (algo == "glumin_automine") {
            run_glumin_automine();
        } else if (algo == "glumin_automine_lut") {
            run_glumin_automine_lut();
        } else {
            LOG_ERROR("Unsupported algorithm: " + algo);
            return;
        }
    }

    void PatternSolver_on_G2Miner();
    void CliqueSolver_on_G2Miner();

private:
    int use_graphpi_sched = 1;
    
    Graph_V2 g;
    std::string data_name = "mico";
    std::string data_path = "";
    char* adj_mat;
    int pattern_size = 3;
    int patternID = 1;
    int num_patterns = 1;
    std::vector<uint64_t> total;
    const int* p_adj_mat;
    std::string algo = "";
    int k_num;

    int local_patternId = 0;
    bool use_dag = false;
    int n_devices = 1;
    int chunk_size = 1024;
    int select_device = 3;
    // if (argc > 3) select_device = atoi(argv[3]);
    // if (argc > 4) n_devices = atoi(argv[4]);
    // if (argc > 5) chunk_size = atoi(argv[5]);
    
    int vertices, edges;
    std::vector<std::set<int>> edgeLists;
    long long total_count = 0;
    std::vector<int> embedding;
    Input_FileFormat file_format = Input_FileFormat::BINARY;

    bool do_validation = true;

    CPUTimer timer;
    Kernel kernel;

};
