#include "../include/kernel.h"


void Kernel::motif4_glumin_p1_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding) {
    LOG_INFO("Running motif4_glumin_p1_baseline_cpu_kernel.");

    for (int i = 0; i < vertices; i++) { // level 1
        int candidate_0 = i;
        embedding.push_back(candidate_0);
        for (int candidate_1 : edgeLists[i]) { // level 2
            embedding.push_back(candidate_1);
            for (int candidate_2 : edgeLists[candidate_0]) { // level 3
                if (candidate_2 > candidate_1) {
                    embedding.push_back(candidate_2);
                    for (int candidate_3 : edgeLists[candidate_0]) { // level 4
                        if (candidate_3 > candidate_2) {
                            embedding.push_back(candidate_3);
                            total_count++;
                            // print out the embedding
                            // for (int v : embedding) {
                            //     std::cout << v << " ";
                            // }
                            // std::cout << std::endl;
                            embedding.pop_back();
                        }
                    }
                    embedding.pop_back();
                }
            }
            embedding.pop_back();
        }
        embedding.pop_back();
    }

}



