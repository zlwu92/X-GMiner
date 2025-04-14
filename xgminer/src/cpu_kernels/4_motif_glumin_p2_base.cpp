#include "../include/kernel.h"


void Kernel::motif4_glumin_p2_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding, int vert_induced) {
    LOG_INFO("Running motif4_glumin_p2_baseline_cpu_kernel.");
    LOG_INFO("vert_induced: " + std::to_string(vert_induced));
    std::vector<std::vector<int>> edge_vecList(vertices);
    for (int i = 0; i < vertices; i++) {
        edge_vecList[i].assign(edgeLists[i].begin(), edgeLists[i].end());
    }
    for (int i = 0; i < vertices; i++) { // level 1
        int candidate_0 = i;
        // if (i == 0) 
        {
        embedding.push_back(candidate_0);
        // for (int candidate_1 : edge_vecList[i]) { // level 2
        for (int c1 = 0; c1 < edge_vecList[i].size(); c1++) {
            int candidate_1 = edge_vecList[i][c1];
            if (candidate_1 > candidate_0 
                // && c1 <= 8 
                // && c1 >= 7
            ) {
                embedding.push_back(candidate_1);
                std::set<int> intersect;
                std::set_intersection(edge_vecList[candidate_0].begin(), edge_vecList[candidate_0].end(), 
                                                edge_vecList[candidate_1].begin(), edge_vecList[candidate_1].end(), 
                                                std::inserter(intersect, intersect.begin()));
                for (int candidate_2 : intersect) { // level 3
                    embedding.push_back(candidate_2);
                    // std::cout << "partial embedding: ";
                    // for (int v : embedding) {
                    //     std::cout << v << " ";
                    // }
                    // std::cout << std::endl;
                    for (int candidate_3 : intersect) {
                        if (candidate_3 > candidate_2) { // level 4
                            if (!vert_induced) {
                                embedding.push_back(candidate_3);
                                total_count++;
                                // print out the embedding
                                // for (int v : embedding) {
                                //     std::cout << v << " ";
                                // }
                                // std::cout << std::endl;
                                embedding.pop_back();
                            } else {
                                // candidate_3 should not be neighbors of candidate_2
                                // if (edge_vecList[candidate_2].find(candidate_3) == edge_vecList[candidate_2].end()) {
                                if (std::find(edge_vecList[candidate_2].begin(), edge_vecList[candidate_2].end(), 
                                    candidate_3) == edge_vecList[candidate_2].end()) {
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
}
