#include "cpu_baseline.h"


void CPU_Baseline::run_graphpi_test() {
    printf("run_graphpi_test\n");
    // step 0: load data
    DataType my_type;
    Graph *g;
    DataLoader D;
    D.GetDataType(my_type, data_name);
    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return ;
    }

    // assert(D.load_data(g, my_type, data_path.c_str())==true);
    int ret = D.load_data(g, my_type, data_path.c_str());
    assert(ret);
    printf("Load data success!\n");
    // fflush(stdout);

    while (*adj_mat != '\0') {
        printf("%c", *adj_mat);
        adj_mat++;
    }
    puts("");
    adj_mat -= pattern_size * pattern_size;


    // Step 1 : Define the pattern
    Pattern pattern(pattern_size, adj_mat);
    // Pattern pattern(pattern_size, adj_mat_buffer);
    // Pattern pattern(PatternType::Rectangle);
    p_adj_mat = pattern.get_adj_mat_ptr();
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j< 4; ++j)
            printf("%d", p_adj_mat[INDEX(i,j,4)]);
    puts("");

    double t1,t2;
    
    printf("g->v_cnt %d g->e_cnt %d g->tri_cnt %lld\n", g->v_cnt,g->e_cnt,g->tri_cnt);

    // Step 2 : Define the schedule
    bool is_pattern_valid = false;
    int performance_modeling_type = 1;
    bool use_in_exclusion_optimize = true;
    int restricts_type = 1;
    Schedule schedule(pattern, is_pattern_valid, 
                        performance_modeling_type, restricts_type, use_in_exclusion_optimize, 
                        g->v_cnt, g->e_cnt, g->tri_cnt);

    std::cout << __LINE__ << " " << __FILE__ << std::endl;
    // 0110
    // 1001
    // 1001
    // 0110

    int thread_num = 1;//24;//
    t1 = get_wall_time();
    long long ans = g->pattern_matching(schedule, thread_num);
    t2 = get_wall_time();

    printf("ans %lld\n", ans);
    printf("time %.6lf\n", t2 - t1);
    
    // fflush(stdout);
}

void CPU_Baseline::run_our_baseline_test() {
    std::ifstream file(data_path); // 打开文本文件
    int vertices, edges;
    file >> vertices >> edges; // 读取顶点数和边数
    std::cout << "Vertices: " << vertices << ", Edges: " << edges << std::endl;
    // std::vector<std::vector<int>> edgeLists; // 用于存储每个顶点的边列表
    // std::vector<std::set<int>> edgeLists; // 用于存储每个顶点的边列表
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

    file.close();

    // std::vector<int> embedding;
    // std::set<std::set<int>> uniques;
    // int count = 0;
    
    // identify the pattern
    // 4_rectangle_baseline_cpu_kernel();
    kernel.rectangle4_baseline_cpu_kernel(vertices, edgeLists, total_count, embedding);

    LOG_INFO("total embeddings: " + std::to_string(total_count));
    // LOG_INFO("unique embeddings: " + std::to_string(uniques.size()));
}


void CPU_Baseline::run_baseline_with_graphpi_sched() {
    
    // step 0: load data
    DataType my_type;
    Graph *g;
    DataLoader D;
    D.GetDataType(my_type, data_name);
    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return ;
    }

    // assert(D.load_data(g, my_type, data_path.c_str())==true);
    int ret = D.load_data(g, my_type, data_path.c_str());
    assert(ret);
    printf("Load data success!\n");
    // fflush(stdout);

    printf("Input schedule: ");
    while (*adj_mat != '\0') {
        printf("%c", *adj_mat);
        adj_mat++;
    }
    puts("");
    adj_mat -= pattern_size * pattern_size;


    // Step 1 : Define the pattern
    Pattern pattern(pattern_size, adj_mat);
    // Pattern pattern(pattern_size, adj_mat_buffer);
    // Pattern pattern(PatternType::Rectangle);
    p_adj_mat = pattern.get_adj_mat_ptr();
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j< 4; ++j)
            printf("%d", p_adj_mat[INDEX(i,j,4)]);
    puts("");

    double t1,t2;
    
    printf("g->v_cnt %d g->e_cnt %d g->tri_cnt %lld\n", g->v_cnt,g->e_cnt,g->tri_cnt);

    // Step 2 : Define the schedule
    bool is_pattern_valid = false;
    int performance_modeling_type = 1;
    bool use_in_exclusion_optimize = true;
    int restricts_type = 1;
    Schedule schedule(pattern, is_pattern_valid, 
                        performance_modeling_type, restricts_type, use_in_exclusion_optimize, 
                        g->v_cnt, g->e_cnt, g->tri_cnt);

    printf("Current schedule: ");
    const int* sched_adj_mat = schedule.get_adj_mat_ptr();
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j< 4; ++j)
            printf("%d", sched_adj_mat[INDEX(i,j,4)]);
    puts("");

    auto& pairs = schedule.restrict_pair;
    printf("%ld ",pairs.size());
    for(auto& p : pairs)
        printf("(%d,%d)",p.first,p.second);
    puts("");

    // construct edgeList
    std::ifstream file(data_path); // 打开文本文件
    int vertices, edges;
    file >> vertices >> edges; // 读取顶点数和边数
    std::cout << "Vertices: " << vertices << ", Edges: " << edges << std::endl;
    edgeLists.resize(vertices);
    int source, target;
    while (file >> source >> target) {
        edgeLists[source].insert(target);
        edgeLists[target].insert(source);
    }
    file.close();
    // print edgeLists
    for (int i = 0; i < vertices; i++) {
        std::cout << "Vertex " << i << " edges: ";
        for (int edge : edgeLists[i]) {
            std::cout << edge << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<int>> p_edgeList(pattern_size);
    for (int i = 0; i < pattern_size * pattern_size; ++i) {
        if (sched_adj_mat[i] == 1) {
            int row = i / pattern_size;
            int col = i % pattern_size;
            p_edgeList[row].push_back(col);
        } 
    }

    // print p_edgeList
    for (int i = 0; i < pattern_size; i++) {
        std::cout << "Pattern Vertex " << i << " edges: ";
        for (int edge : p_edgeList[i]) {
            std::cout << edge << " ";
        }
        std::cout << std::endl;
    }

    // kernel.rectangle4_baseline_cpu_kernel_with_graphpi_sched(vertices, edgeLists, total_count, embedding,
    //                                                         p_edgeList, pairs);
    kernel.rectangle4_baseline_cpu_kernel_with_graphpi_sched_v2(vertices, edgeLists, total_count, embedding,
                                                            p_edgeList, pairs);
    LOG_INFO("total embeddings: " + std::to_string(total_count));
}
