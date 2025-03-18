#include "glumin.h"

void GLUMIN::run_glumin_g2miner() {
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
        g.print_meta_data();
    
        uint64_t total = 0;
        CliqueSolver_on_G2Miner();
        std::cout << "Pattern P" << local_patternId << " count: " << total << "\n";
    }
    else {
        // int n_devices = 1;
        // int chunk_size = 1024;
        // if (argc > 3) n_devices = atoi(argv[3]);
        // if (argc > 4) chunk_size = atoi(argv[4]);
        g.print_meta_data();
    
        // int num_patterns = 1;
        // std::cout << "num_patterns: " << num_patterns << "\n";
        // std::vector<uint64_t> total(num_patterns, 0);
        PatternSolver_on_G2Miner();
        for (int i = 0; i < num_patterns; i++)
        std::cout << "Pattern P" << local_patternId << " count: " << total[i] << "\n";
    }
}

void GLUMIN::run_glumin_g2miner_lut() {

}

void GLUMIN::run_glumin_graphfold() {

}


void GLUMIN::run_glumin_graphfold_lut() {

}


void GLUMIN::run_glumin_automine() {

}


void GLUMIN::run_glumin_automine_lut() {

}

