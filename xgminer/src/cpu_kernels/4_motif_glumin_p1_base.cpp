#include "../include/kernel.h"


void Kernel::motif4_glumin_p1_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                            long long& total_count, std::vector<int>& embedding, int vert_induced) {
    LOG_INFO("Running motif4_glumin_p1_baseline_cpu_kernel.");
    LOG_INFO("vert_induced: " + std::to_string(vert_induced));
    std::vector<std::vector<int>> edge_vecList(vertices);
    for (int i = 0; i < vertices; i++) {
        edge_vecList[i].assign(edgeLists[i].begin(), edgeLists[i].end());
    }

    for (int i = 0; i < vertices; i++) { // level 1
        int candidate_0 = i;
        embedding.push_back(candidate_0);
        // for (int candidate_1 : edgeLists[i]) { // level 2
        for (int c1 = 0; c1 < edge_vecList[i].size(); c1++) {
            // embedding.push_back(candidate_1);
            int candidate_1 = edge_vecList[i][c1];
            embedding.push_back(candidate_1);
            // for (int candidate_2 : edgeLists[candidate_0]) { // level 3
            for (int c2 = 0; c2 < edge_vecList[candidate_0].size(); c2++) {
                int candidate_2 = edge_vecList[candidate_0][c2];
                if (candidate_2 < candidate_1) {
                    if (!vert_induced) {
                        embedding.push_back(candidate_2);
                        // for (int candidate_3 : edgeLists[candidate_0]) { // level 4
                        for (int c3 = 0; c3 < edge_vecList[candidate_0].size(); c3++) {
                            int candidate_3 = edge_vecList[candidate_0][c3];
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
                    } else {
                        // candidate_2 should not be neighbors of candidate_1
                        if (std::find(edge_vecList[candidate_1].begin(), edge_vecList[candidate_1].end(), 
                            candidate_2) == edge_vecList[candidate_1].end()) {
                            embedding.push_back(candidate_2);
                            for (int c3 = 0; c3 < edge_vecList[candidate_0].size(); c3++) {
                                int candidate_3 = edge_vecList[candidate_0][c3];
                                if (candidate_3 < candidate_2) {
                                    // candidate_3 should not be neighbors of candidate_1 and candidate_2
                                    if (std::find(edge_vecList[candidate_1].begin(), edge_vecList[candidate_1].end(), 
                                        candidate_3) == edge_vecList[candidate_1].end() &&
                                        std::find(edge_vecList[candidate_2].begin(), edge_vecList[candidate_2].end(), 
                                        candidate_3) == edge_vecList[candidate_2].end()) {
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



