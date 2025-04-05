#include "../include/kernel.h"


void Kernel::motif4_glumin_p2_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding, int vert_induced) {
    LOG_INFO("Running motif4_glumin_p2_baseline_cpu_kernel.");

    for (int i = 0; i < vertices; i++) { // level 1
        int candidate_0 = i;
        embedding.push_back(candidate_0);
        for (int candidate_1 : edgeLists[i]) { // level 2
            if (candidate_1 > candidate_0) {
                embedding.push_back(candidate_1);
                std::set<int> intersect;
                std::set_intersection(edgeLists[candidate_0].begin(), edgeLists[candidate_0].end(), 
                                                edgeLists[candidate_1].begin(), edgeLists[candidate_1].end(), 
                                                std::inserter(intersect, intersect.begin()));
                for (int candidate_2 : intersect) { // level 3
                    embedding.push_back(candidate_2);
                    for (int candidate_3 : intersect) {
                        if (candidate_3 > candidate_2) { // level 4
                            if (!vert_induced) {
                                embedding.push_back(candidate_3);
                                total_count++;
                                // print out the embedding
                                for (int v : embedding) {
                                    std::cout << v << " ";
                                }
                                std::cout << std::endl;
                                embedding.pop_back();
                            } else {
                                // candidate_3 should not be neighbors of candidate_2
                                if (edgeLists[candidate_2].find(candidate_3) == edgeLists[candidate_2].end()) {
                                    embedding.push_back(candidate_3);
                                    total_count++;
                                    // print out the embedding
                                    for (int v : embedding) {
                                        std::cout << v << " ";
                                    }
                                    std::cout << std::endl;
                                    embedding.pop_back();
                                }
                            }
                        }
                    }
                    embedding.pop_back();
                }
                embedding.pop_back();
            }
        }
        embedding.pop_back();
    }
}
