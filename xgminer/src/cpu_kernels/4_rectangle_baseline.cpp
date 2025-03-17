#include "../include/kernel.h"

void Kernel::rectangle4_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                        int& total_count, std::vector<int>& embedding) {
// void rectangle4_baseline_cpu_kernel() {
    std::set<std::set<int>> uniques;

    for (int i = 0; i < vertices; i++) { // level 1
        int candidate_1 = i;
        embedding.push_back(i);
        // if (i == 0) 
        {
            for (int candidate_2 : edgeLists[i]) { // level 2
                if (candidate_2 > candidate_1) {
                    embedding.push_back(candidate_2);
                    for (int candidate_3 : edgeLists[candidate_1]) { // level 3
                        if (
                            // std::find(embedding.begin(), embedding.end(), candidate_3) == embedding.end()
                            candidate_3 > candidate_1
                            && candidate_3 > candidate_2) {
                            embedding.push_back(candidate_3);
                            // should be intersection of edgeLists[candidate_1] and edgeLists[candidate_3]
                            std::set<int> candidate_4;
                            std::set_intersection(edgeLists[candidate_2].begin(), edgeLists[candidate_2].end(), 
                                            edgeLists[candidate_3].begin(), edgeLists[candidate_3].end(), 
                                            std::inserter(candidate_4, candidate_4.begin()));
                            for (int candidate_4 : candidate_4) { // level 4
                                if (
                                    // std::find(embedding.begin(), embedding.end(), candidate_4) == embedding.end()
                                    candidate_4 > candidate_1
                                    ) {
                                    embedding.push_back(candidate_4);
                                    total_count++;
                                    // print out the embedding
                                    // for (int v : embedding) {
                                    //     std::cout << v << " ";
                                    // }
                                    // std::cout << std::endl;
                                    if (uniques.find(std::set<int>(embedding.begin(), embedding.end())) == uniques.end()) {
                                        uniques.insert(std::set<int>(embedding.begin(), embedding.end()));
                                    }
                                    embedding.pop_back();
                                }
                            }
                            // }
                            embedding.pop_back();
                        }
                    }
                    embedding.pop_back();
                }
            }
        }
        embedding.pop_back();
    }

}