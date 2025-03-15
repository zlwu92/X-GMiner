#include "cpu_baseline.h"


void CPU_Baseline::run_baseline_with_graphpi() {
    // Step 1 : Define the pattern
    Pattern pattern(PatternType::Rectangle);
    const int* adj_mat = pattern.get_adj_mat_ptr();
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j< 4; ++j)
            printf("%d", adj_mat[INDEX(i,j,4)]);
    puts("");

    // Step 2 : Define the schedule
    bool is_pattern_valid = false;
    int performance_modeling_type = 1;
    bool use_in_exclusion_optimize = true;
    int restricts_type = 1;
    Schedule schedule(pattern, is_pattern_valid, 
                        performance_modeling_type, restricts_type, use_in_exclusion_optimize, 
                        g->v_cnt, g->e_cnt, g->tri_cnt);
    // ASSERT_EQ(is_pattern_valid, true);

    // 0110
    // 1001
    // 1001
    // 0110
}

void CPU_Baseline::run_our_baseline_test() {
    std::ifstream file(data_path); // 打开文本文件
    int vertices, edges;
    file >> vertices >> edges; // 读取顶点数和边数
    std::cout << "Vertices: " << vertices << ", Edges: " << edges << std::endl;
    // std::vector<std::vector<int>> edgeLists; // 用于存储每个顶点的边列表
    std::vector<std::set<int>> edgeLists; // 用于存储每个顶点的边列表
    edgeLists.resize(vertices);
    int source, target;
    while (file >> source >> target) {
        // edgeLists[source].push_back(target); // 添加边到源顶点的边列表
        // edgeLists[target].push_back(source); // 无向图需要双向添加
        edgeLists[source].insert(target); // 添加边到源顶点的边列表
        edgeLists[target].insert(source); // 无向图需要双向添加
    }

    // 输出每个顶点的边列表
    // for (const auto& entry : edgeLists) {
    //     std::cout << "Vertex " << entry.first << " edges: ";
    //     for (int edge : entry.second) {
    //         std::cout << edge << " ";
    //     }
    //     std::cout << std::endl;
    // }
    for (int i = 0; i < vertices; i++) {
        std::cout << "Vertex " << i << " edges: ";
        for (int edge : edgeLists[i]) {
            std::cout << edge << " ";
        }
        std::cout << std::endl;
    }

    file.close(); // 关闭文件

    std::vector<int> embedding;
    std::set<std::set<int>> uniques;
    int count = 0;
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
                                count++;
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

    LOG_INFO("total embeddings: " + std::to_string(count));
    LOG_INFO("unique embeddings: " + std::to_string(uniques.size()));
}