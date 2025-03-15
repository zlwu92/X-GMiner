#include <iostream>
#include <numeric>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <limits>
#include <cstdio>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <climits>
#include <cassert>
#include <cstdint>
#include <regex>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <stack>
#include "cmd_option.h"
#include "cpu_baseline.h"
#include "gpm_solver.cuh"


void test_pattern(Graph* g, const Pattern &pattern, int performance_modeling_type, int restricts_type, bool use_in_exclusion_optimize = false) {
    int thread_num = 24;
    double t1,t2;
    
    bool is_pattern_valid;
    Schedule schedule(pattern, is_pattern_valid, performance_modeling_type, restricts_type, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    assert(is_pattern_valid);

    t1 = get_wall_time();
    long long ans = g->pattern_matching(schedule, thread_num);
    t2 = get_wall_time();

    printf("ans %lld\n", ans);
    printf("time %.6lf\n", t2 - t1);
    schedule.print_schedule();
    const auto& pairs = schedule.restrict_pair;
    printf("%d ",pairs.size());
    for(auto& p : pairs)
        printf("(%d,%d)",p.first,p.second);
    puts("");
    fflush(stdout);

}


int main(int argc, char *argv[]) {
   
    Command_Option opts(argc, argv);
    opts.parseByArgParse();

    Logger& logger = Logger::getInstance();
    logger.setConsoleEnabled(true);

    // 第一次设置文件：截断（覆盖）
    // logger.setFile("app.log");
    // LOG_INFO("First message (truncated)");

    // // 第二次设置同一个文件：追加
    // logger.setFile("app.log");
    // LOG_INFO("Second message (appended)");

    // // 切换到新文件：截断
    // logger.setFile("another.log");
    // LOG_INFO("New file message (truncated)");

    // std::regex pattern("^P(\\d+)$");
    // std::smatch match;
    // int k;
    // if (std::regex_match(opts.pattern_name, match, pattern)) {
    //     k = std::stoi(match[1].str());
    //     std::cout << "Pattern P" << k << std::endl;
    // } else {
    //     std::cerr << "Invalid input format. Expected format: Px, where x is an integer." << std::endl;
    //     return 1;
    // }

    // if (opts.algo == "cpu_baseline") {
    //     LOG_INFO("Running CPU baseline algorithm.");
    //     CPU_Baseline cpu_base(opts.use_graphpi_sched, opts.data_name, opts.datagraph_file);
        
    // } else {
    //     LOG_ERROR("Unsupported algorithm: " + opts.algo);
    //     return 1;
    // }
#if 0
    Graph *g;
    DataLoader D;

    const std::string type = argv[1];
    const std::string path = argv[2];
    
    int size = atoi(argv[3]);
    char* adj_mat = argv[4];

    // comments in include/schedule.h explain the meaning of these parameters.
    int test_type = 1; // performance_modeling_type = restricts_type = use_in_exclusion_optimize = 1

    DataType my_type;
    
    D.GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    assert(D.load_data(g,my_type,path.c_str())==true); 

    printf("Load data success!\n");
    fflush(stdout);

    Pattern p(size, adj_mat);
    // Pattern p(PatternType::Rectangle);
    test_pattern(g, p, test_type, test_type, test_type);
#endif

    std::ifstream file(opts.datagraph_file); // 打开文本文件
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

    LOG_INFO("Finish processing.");
    return 0;
}