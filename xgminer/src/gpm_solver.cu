#include "gpm_solver.cuh"

#if 0
void XGMiner::run_glumin_g2miner() {
    if (local_patternId == 4 || local_patternId == 5 || local_patternId == 23 || local_patternId == 24) {
        k_num = local_patternId;
        if (local_patternId == 23) k_num = 6;
        if (local_patternId == 24) k_num = 7;
        // int n_devices = 1;
        // int chunk_size = 1024;
        // int select_device = 3;
        // int choose = 0;
        // if (argc > 3) select_device = atoi(argv[3]);
        // if (argc > 4) n_devices = atoi(argv[4]);
        // if (argc > 5) chunk_size = atoi(argv[5]);
        Graph_V2 g(prefix, use_dag);
        g.print_meta_data();
    
        for (int r = 0; r < repeated; r++) {
            if (!use_lut)   CliqueSolver_on_G2Miner(g);
            else            CliqueSolver_LUT_on_G2Miner(g);
            uint64_t total_ = total[0];
            std::cout << "Pattern P" << local_patternId << " count: " << total_ << "\n";
        }
    }
    else {
        std::cout << __LINE__ << " local_patternId:" << local_patternId << " use_lut:" << use_lut << "\n";
        // int n_devices = 1;
        // int chunk_size = 1024;
        // if (argc > 3) n_devices = atoi(argv[3]);
        // if (argc > 4) chunk_size = atoi(argv[4]);
        Graph_V2 g(prefix, use_dag);
        g.print_meta_data();

        // g.init_edgelist();
    
        // int num_patterns = 1;
        // std::cout << "num_patterns: " << num_patterns << "\n";
        // std::vector<uint64_t> total(num_patterns, 0);
        for (int r = 0; r < repeated; r++) {
            if (!use_lut)   PatternSolver_on_G2Miner(g);
            else            PatternSolver_LUT_on_G2Miner(g);
            for (int i = 0; i < num_patterns; i++)
            std::cout << "Pattern P" << local_patternId << " count: " << total[i] << "\n";
        }
    }

    std::ofstream out(output_path + "overall_performance.csv", std::ios::app);
    out << data_name << ",P" << local_patternId << "," << algo << "," << total[0] << "," << total_time[0] / repeated << "\n";
}



void XGMiner::run_glumin_graphfold() {
    
    k_num = local_patternId;
    if (!use_lut)   k_num = k_num + 1;

    if (k_num == 5 || k_num == 6) {
        // Graph g(argv[1], USE_DAG); // use DAG
        // int n_devices = 8;
        // int chunk_size = 1024;
        // int select_device = 3;
        // int choose = 0;
        // if (argc > 4) select_device = atoi(argv[4]);
        // if (argc > 5) n_devices = atoi(argv[5]);
        // if (argc > 6) chunk_size = atoi(argv[6]);
        g.print_meta_data();
    
        // if (!use_lut)   
        CliqueSolver_on_GraphFold();
        // else            CliqueSolver_LUT_on_GraphFold();
        uint64_t total_ = total[0];
        std::cout << "Pattern P" << local_patternId << " count: " << total_ << "\n";
    }
    else {
        // Graph g(argv[1]);
        // int n_devices = 1;
        // int chunk_size = 1024;
        // if (argc > 4) n_devices = atoi(argv[4]);
        // if (argc > 5) chunk_size = atoi(argv[5]);
        g.print_meta_data();

        // int num_patterns = 1;
        // std::cout << "num_patterns: " << num_patterns << "\n";
        // std::vector<uint64_t> total(num_patterns, 0);
        // if (!use_lut)   
        PatternSolver_on_GraphFold();
        // else            PatternSolver_LUT_on_GraphFold();
        for (int i = 0; i < num_patterns; i++)
        std::cout << "Pattern P" << local_patternId << " count: " << total[i] << "\n";
    }
}


void XGMiner::run_glumin_automine() {

    k_num = local_patternId;
    if (!use_lut)   k_num = k_num + 1;
    // int n_devices = 1;
    // int chunk_size = 1024;
    // if (argc > 4) n_devices = atoi(argv[4]);
    // if (argc > 5) chunk_size = atoi(argv[5]);
    g.print_meta_data();
    
    // int num_patterns = 1;
    // std::cout << "num_patterns: " << num_patterns << "\n";
    // std::vector<uint64_t> total(num_patterns, 0);
    PatternSolver_on_AutoMine();
    for (int i = 0; i < num_patterns; i++)
        std::cout << "Pattern P" << local_patternId << " count: " << total[i] << "\n";
}


void XGMiner::run_glumin_cliques() {
    
    if (local_patternId == 4 || local_patternId == 5 || local_patternId == 6 || local_patternId == 7) {
        // int k = atoi(argv[2]);
        // int n_devices = 8;
        // int chunk_size = 1024;
        // int select_device = 3;
        // int choose = 0;
        // if (argc > 3) select_device = atoi(argv[3]);
        // if (argc > 4) n_devices = atoi(argv[4]);
        // if (argc > 5) chunk_size = atoi(argv[5]);
        g.print_meta_data();
        
        if (use_lut) {
            CliqueSolver_LUT_on_GM_Clique();
        }
        else {
            CliqueSolver_on_GM_Clique();
        }
        uint64_t total_ = total[0];
        std::cout << "num_" << local_patternId << "-cliques = " << total_ << "\n";
    }
    if (local_patternId == 1 || local_patternId == 2) {
        if (use_lut) {
            CliqueSolver_LUT_on_GF_Clique();
            uint64_t total_ = total[0];
            std::cout << "num_" << local_patternId << "-cliques = " << total_ << "\n";
        }
    }
}
#endif

void XGMiner::run_bitmap_bigset_opt() {
    if (local_patternId == 4 || local_patternId == 5 || local_patternId == 23 || local_patternId == 24) {
        k_num = local_patternId;
        if (local_patternId == 23) k_num = 6;
        if (local_patternId == 24) k_num = 7;

        use_dag = true;
        Graph_V2 g(prefix, use_dag);
        g.print_meta_data();

        for (int r = 0; r < repeated; r++) {

            clique_solver(g);
            uint64_t total_ = total[0];
            std::cout << "Pattern P" << local_patternId << " count: " << total_ << "\n";
        }
    } else {
        Graph_V2 g(prefix, use_dag);
        g.print_meta_data();

        for (int r = 0; r < repeated; r++) {
            
            motif_solver(g);
            for (int i = 0; i < num_patterns; i++)
            std::cout << "Pattern P" << local_patternId << " count: " << total[i] << "\n";
        }
    }
}
