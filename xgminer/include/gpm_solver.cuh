#pragma once
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "../include/graph_v2.h"
#include "../include/common.h"
#include "cmd_option.h"
#include "kernel.h"
#include <memory> // std::unique_ptr
#include "cpu_baseline.h"

// 编译期计算 pow(2, N) / 64
constexpr int calculate_value(int N) {
    return ((1 << N) + BITMAP64_WIDTH - 1) / BITMAP64_WIDTH; // 使用位移代替 pow(2, N)
}

class XGMiner {
public:
    XGMiner(Command_Option &opts) {
        data_name = opts.data_name;
        data_path = opts.datagraph_file;
        use_graphpi_sched = opts.use_graphpi_sched;
        pattern_size = opts.pattern_size;
        adj_mat = opts.pattern_adj_mat;
        patternID = opts.patternID;
        file_format = getFileFormat(data_path);
        do_validation = opts.do_validation;
        algo = opts.algo;
        tune_k = opts.tune_k;
        bucket_k = opts.set_k;
        run_xgminer = opts.run_xgminer;
        total.resize(num_patterns, 0);
        total_time.resize(num_patterns, 0);
        if (do_validation) {
            opts.run_our_baseline = 1;
            // std::filesystem::path absolutePath = data_path;
            // size_t found = absolutePath.string().find(benchmarkset_name);
            // if (found != std::string::npos) {
            //     // 截取 "graphmine_bench" 前面的部分
            //     std::string commonPrefix = absolutePath.string().substr(0, found);  
            //     std::cout << "commonPrefix: " << commonPrefix << std::endl;
            //     opts.datagraph_file = commonPrefix + "graphpi_data/datasets/";
            // }
            cpu_base = new CPU_Baseline(opts);
        }
    }

    ~XGMiner() {
        LOG_INFO("XGMiner destructor");
    }

    void load_graph_data_from_file() {
        // patternID = 17;
        local_patternId = patternID - 16;
        if (file_format == Input_FileFormat::BINARY) { // stored in CSR format
            if (local_patternId == 4 || local_patternId == 5 || local_patternId == 23 || local_patternId == 24) {
                use_dag = true;
            }
            // if (algo == "glumin_cliques" || algo == "glumin_cliques_lut")   use_dag = true;
            prefix = data_path + "/graph";
            // Graph_V2 g(prefix, use_dag);
            // g = Graph_V2(prefix, use_dag);
        }
    }

    void run_bitmap_bigset_opt();

    void run() {
        load_graph_data_from_file();
        if (algo == "bitmap_bigset_opt") {
            run_bitmap_bigset_opt();
        }
        else if (algo == "ideal_bitmap_test") {
            run_bitmap_bigset_opt();
        }
        else {
            LOG_ERROR("Unsupported algorithm: " + algo);
            return;
        }
    }

    void clique_solver(Graph_V2& g);
    void motif_solver(Graph_V2& g);

    // void PatternSolver_on_G2Miner(Graph_V2& g);
    // void CliqueSolver_on_G2Miner(Graph_V2& g);

    // void PatternSolver_LUT_on_G2Miner(Graph_V2& g);
    // void CliqueSolver_LUT_on_G2Miner(Graph_V2& g);

    // void PatternSolver_on_GraphFold();
    // void CliqueSolver_on_GraphFold();

    // void PatternSolver_on_AutoMine();

    // void CliqueSolver_on_GM_Clique();
    // void CliqueSolver_LUT_on_GM_Clique();
    // void CliqueSolver_LUT_on_GF_Clique();

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

    bool do_validation = true;
    int bucket_k = 8; // for bitmap bucket number
    bool tune_k = false;
    CPU_Baseline* cpu_base;
    std::string benchmarkset_name = "graphmine_bench";
    int run_xgminer = 0;

    CPUTimer timer;
    Kernel kernel;

};
