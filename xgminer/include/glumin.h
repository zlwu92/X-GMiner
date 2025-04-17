#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "../include/graph_v2.h"
#include "../include/common.h"
#include "cmd_option.h"
#include "kernel.h"
#include <memory> // std::unique_ptr

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
        prof_workload = opts.prof_workload;
        use_vert_para = opts.use_vert_para;
        prof_edgecheck = opts.prof_edgecheck;
        total.resize(num_patterns, 0);
        total_time.resize(num_patterns, 0);
    }

    ~GLUMIN() {
        std::cout << "GLUMIN destructor" << std::endl;
    }

    void load_graph_data_from_file() {
        // patternID = 17;
        local_patternId = patternID - 16;
        if (file_format == Input_FileFormat::BINARY) { // stored in CSR format
            if (local_patternId == 4 || local_patternId == 5 || local_patternId == 23 || local_patternId == 24) {
                use_dag = true;
            }
            if (algo == "glumin_cliques" || algo == "glumin_cliques_lut")   use_dag = true;
            prefix = data_path + "/graph";
            // Graph_V2 g(prefix, use_dag);
            // g = Graph_V2(prefix, use_dag);
        }
    }

    void run_glumin_g2miner();

    void run_glumin_graphfold();

    void run_glumin_automine();

    void run_glumin_cliques();

    void run() {
        load_graph_data_from_file();
        if (algo == "glumin_g2miner" || algo == "glumin_g2miner_lut") {
            std::cout << __LINE__ << " algo: " << algo << std::endl;
            if (algo == "glumin_g2miner_lut") {
                use_lut = 1;
            }
            run_glumin_g2miner();
        }
        else if (algo == "glumin_gf" || algo == "glumin_gf_lut") {
            if (algo == "glumin_gf_lut")   use_lut = 1;
            run_glumin_graphfold();
        }
        else if (algo == "glumin_automine" || algo == "glumin_automine_lut") {
            if (algo == "glumin_automine_lut")   use_lut = 1;
            run_glumin_automine();
        }
        else if (algo == "glumin_cliques" || algo == "glumin_cliques_lut") {
            run_glumin_cliques();
        }
        else {
            LOG_ERROR("Unsupported algorithm: " + algo);
            return;
        }
    }

    void PatternSolver_on_G2Miner(Graph_V2& g);
    void CliqueSolver_on_G2Miner(Graph_V2& g);

    void PatternSolver_LUT_on_G2Miner(Graph_V2& g);
    void CliqueSolver_LUT_on_G2Miner(Graph_V2& g);

    void PatternSolver_on_GraphFold();
    void CliqueSolver_on_GraphFold();

    void PatternSolver_on_AutoMine();

    void CliqueSolver_on_GM_Clique();
    void CliqueSolver_LUT_on_GM_Clique();
    void CliqueSolver_LUT_on_GF_Clique();

private:
    int use_graphpi_sched = 1;
    
    Graph_V2 g;
    // std::unique_ptr<Graph_V2> g2;
    std::string data_name = "mico";
    std::string data_path = "";
    char* adj_mat;
    int pattern_size = 3;
    int patternID = 17;
    int num_patterns = 1;
    std::vector<uint64_t> total;
    const int* p_adj_mat;
    std::string algo = "";
    std::string prefix = "";
    int k_num;

    int local_patternId = 0;
    bool use_dag = false;
    int n_devices = 1;
    int chunk_size = 1024;
    int select_device = 3;
    int use_lut = 0;
    std::vector<double> total_time;
    int repeated = 1;
    std::string output_path = "/home/wuzhenlin/workspace/2-graphmining/X-GMiner/scripts/";
    // if (argc > 3) select_device = atoi(argv[3]);
    // if (argc > 4) n_devices = atoi(argv[4]);
    // if (argc > 5) chunk_size = atoi(argv[5]);
    
    int vertices, edges;
    std::vector<std::set<int>> edgeLists;
    long long total_count = 0;
    std::vector<int> embedding;
    Input_FileFormat file_format = Input_FileFormat::BINARY;
    int prof_workload = 0;
    int use_vert_para = 0;
    int prof_edgecheck = 0;

    bool do_validation = true;

    CPUTimer timer;
    Kernel kernel;

};
