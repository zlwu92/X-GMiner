#include "../include/kernel.h"

void Kernel::rectangle4_baseline_cpu_kernel(int vertices, std::vector<std::set<int>> edgeLists,
                                        long long& total_count, std::vector<int>& embedding) {
    LOG_INFO("Running rectangle4_baseline_cpu_kernel.");
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
                                    // if (uniques.find(std::set<int>(embedding.begin(), embedding.end())) == uniques.end()) {
                                    //     uniques.insert(std::set<int>(embedding.begin(), embedding.end()));
                                    // }
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

bool get_restrict_prefix(const std::vector<std::pair<int, int>>& pairs, 
                        int uID, int b,
                        std::vector<int>& embedding) {
    // if (embedding[0]==0 && embedding[1]==2 && uID==2) {
    //     printf("uID=%d, b=%d\n", uID, b);
    // }
    for (const auto& pair : pairs) {
        if (pair.second == uID) {
            // if (embedding[0]==0 && embedding[1]==2 && uID==2) {
            //     printf("pair.first=%d, pair.second=%d\n", pair.first, pair.second);
            //     printf("embedding[pair.first]=%d, b=%d\n", embedding[pair.first], b);
            // }
            if (pair.first < uID && embedding[pair.first] >= b) {
                return false;
            }
        } 
    }
    return true;
}

void Kernel::rectangle4_baseline_cpu_kernel_with_graphpi_sched(
                                        int vertices, std::vector<std::set<int>>& edgeLists,
                                        long long& total_count, std::vector<int>& embedding, 
                                        std::vector<std::vector<int>>& p_edgeList, 
                                        std::vector< std::pair<int,int>>& restrict_pair) {
    LOG_INFO("Running rectangle4_baseline_cpu_kernel_with_graphpi_sched.");
    int uID = 0;
    for (int i = 0; i < vertices; i++) { // level 0, search candidates of u0
        int candidate_0 = i;
        // if (i == 0)
        {
        embedding.push_back(candidate_0);
        uID = 1;
        for (int candidate_1 : edgeLists[candidate_0]) { // level 1, search candidates of u1 in N(v0)
            // search if exists restrict pair for u1
            bool isValid = get_restrict_prefix(restrict_pair, uID, candidate_1, embedding);
            if (isValid) {
                embedding.push_back(candidate_1);
                uID = 2;
                for (int candidate_2 : edgeLists[candidate_0]) { // level 2, search candidates of u2 in N(v0)
                    // search if exists restrict pair for u2
                    bool isValid = get_restrict_prefix(restrict_pair, uID, candidate_2, embedding);
                    // if (embedding[0]==0 && embedding[1]==2 && uID==2) {
                    //     printf("candidate_2=%d isValid=%d\n", candidate_2, isValid);
                    // }
                    if (isValid) {
                        embedding.push_back(candidate_2);
                        uID = 3;
                        // should be intersection of edgeLists[candidate_1] and edgeLists[candidate_2]
                        // printf("candidate_1=%d, candidate_2=%d\n", candidate_1, candidate_2);
                        std::set<int> candidate_3;
                        std::set_intersection(edgeLists[candidate_1].begin(), edgeLists[candidate_1].end(), 
                                        edgeLists[candidate_2].begin(), edgeLists[candidate_2].end(), 
                                        std::inserter(candidate_3, candidate_3.begin()));
                        for (int candidate_3 : candidate_3) { // level 3, search candidates of u3 in N(v1) and N(v2)
                            // search if exists restrict pair for u3
                            bool isValid = get_restrict_prefix(restrict_pair, uID, candidate_3, embedding);
                            
                            if (isValid) {
                                // if (embedding[0]==0 && embedding[1]==1 && embedding[2]==4) 
                                // printf("candidate_3=%d\n", candidate_3);
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
        }
        embedding.pop_back();
        }
    }
}



std::vector<int> get_neighbor_prefix(std::vector<std::vector<int>>& p_edgeList,
                        std::vector<int>& embedding, int uID, std::vector<int>& search_path) {
    std::vector<int> neigh_belong_to;
    int curr_pattern_vertex = uID;
    for (auto v : p_edgeList[curr_pattern_vertex]) {
        if (search_path[v] == 1) {
            neigh_belong_to.push_back(v);
        }
    }

    return neigh_belong_to;
}


void Kernel::rectangle4_baseline_cpu_kernel_with_graphpi_sched_v2(
                                        int vertices, std::vector<std::set<int>>& edgeLists,
                                        long long& total_count, std::vector<int>& embedding, 
                                        std::vector<std::vector<int>>& p_edgeList, 
                                        std::vector< std::pair<int,int>>& restrict_pair) {
    LOG_INFO("Running rectangle4_baseline_cpu_kernel_with_graphpi_sched_v2.");
    int uID = 0;
    std::vector<int> p_search_path(p_edgeList.size(), 0);
    p_search_path[uID] = 1;
    // std::vector<int> search_path(vertices, 0);
    for (int i = 0; i < vertices; i++) { // level 0, search candidates of u0
        int candidate_0 = i;
        // if (i == 0)
        {
        embedding.push_back(candidate_0);
        uID = 1;
        p_search_path[uID] = 1;
        for (int candidate_1 : edgeLists[candidate_0]) { // level 1, search candidates of u1 in N(v0)
            // search if exists restrict pair for u1
            bool isValid = get_restrict_prefix(restrict_pair, uID, candidate_1, embedding);
            if (isValid) {
                embedding.push_back(candidate_1);
                uID = 2;
                auto neigh_belong_to = get_neighbor_prefix(p_edgeList, embedding, uID, p_search_path);
                // std::cout << __LINE__ << " embedding: ";
                // for (int v : embedding) {
                //     std::cout << v << " ";
                // }
                // std::cout << "; neigh_belong_to: ";
                // for (int v : neigh_belong_to) {
                //     std::cout << v << " ";
                // }
                // puts("");
                if (neigh_belong_to.size() == 0) {
                    embedding.pop_back();
                    continue;
                }
                p_search_path[uID] = 1;
                int father = neigh_belong_to[0];
                int cand_father = embedding[father];
                // std::cout << "neigh_belong_to.size()=" << neigh_belong_to.size() << " father=" << father << std::endl;
                for (int candidate_2 : edgeLists[cand_father]) { // level 2, search candidates of u2 in N(v0)
                    // search if exists restrict pair for u2
                    bool isValid = get_restrict_prefix(restrict_pair, uID, candidate_2, embedding);
                    // if (embedding[0]==0 && embedding[1]==2 && uID==2) {
                    //     printf("candidate_2=%d isValid=%d\n", candidate_2, isValid);
                    // }
                    // printf("## cand_father=%d, candidate_2=%d isValid=%d\n", cand_father, candidate_2, isValid);
                    if (isValid) {
                        embedding.push_back(candidate_2);
                        uID = 3;
                        auto neigh_belong_to = get_neighbor_prefix(p_edgeList, embedding, uID, p_search_path);
                        // std::cout << __LINE__ << " embedding: ";
                        // for (int v : embedding) {
                        //     std::cout << v << " ";
                        // }
                        // std::cout << "; neigh_belong_to: ";
                        // for (int v : neigh_belong_to) {
                        //     std::cout << v << " ";
                        // }
                        // puts("");
                        // should be intersection of edgeLists[candidate_1] and edgeLists[candidate_2]
                        if (neigh_belong_to.size() < 2) {
                            embedding.pop_back();
                            continue;
                        }
                        p_search_path[uID] = 1;
                        int father1 = neigh_belong_to[0];
                        int father2 = neigh_belong_to[1];
                        int cand_father1 = embedding[father1];
                        int cand_father2 = embedding[father2];
                        // printf("father1=%d, father2=%d\n", father1, father2);
                        std::set<int> candidate_3;
                        std::set_intersection(edgeLists[cand_father1].begin(), edgeLists[cand_father1].end(), 
                                        edgeLists[cand_father2].begin(), edgeLists[cand_father2].end(), 
                                        std::inserter(candidate_3, candidate_3.begin()));
                        for (int candidate_3 : candidate_3) { // level 3, search candidates of u3 in N(v1) and N(v2)
                            // search if exists restrict pair for u3
                            bool isValid = get_restrict_prefix(restrict_pair, uID, candidate_3, embedding);
                            
                            if (isValid) {
                                // if (embedding[0]==0 && embedding[1]==1 && embedding[2]==4) 
                                // printf("candidate_3=%d\n", candidate_3);
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
                        p_search_path[3] = 0;
                    }
                } 
                embedding.pop_back();
                p_search_path[2] = 0;
            }
        }
        embedding.pop_back();
        }
        p_search_path[1] = 0;
    }
}
