#include "cpu_baseline.h"


long long CPU_Baseline::run_graphpi_test() {
    printf("run_graphpi_test\n");
    // step 0: load data
    DataType my_type;
    Graph *g;
    DataLoader D;
    D.GetDataType(my_type, data_name);
    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return -1;
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
    
    std::ofstream out(output_path + "overall_performance.csv", std::ios::app);
    out << data_name << ",P" << local_patternId << ",graphpi" << "," << ans << "," << t2 - t1 << "\n";
    // fflush(stdout);
    return ans;
}


void CPU_Baseline::run_our_baseline_test() {
    
    timer.start();
    if (patternID == XGMinerPatternType::RECTANGLE) {
        kernel.rectangle4_baseline_cpu_kernel(vertices, edgeLists, total_count, embedding);
    } else if (patternID == XGMinerPatternType::P1_GLUMIN) {
        kernel.motif4_glumin_p1_baseline_cpu_kernel(vertices, edgeLists, total_count, embedding);
    } else if (patternID == XGMinerPatternType::P3_GLUMIN) {
        kernel.motif4_glumin_p3_baseline_cpu_kernel(vertices, edgeLists, total_count, embedding);
    } else {
        LOG_ERROR("Invalid pattern ID.");
    }
    timer.stop();
    LOG_INFO("Elapsed time: " + std::to_string(timer.elapsed()) + " seconds.");
    LOG_INFO("total embeddings: " + std::to_string(total_count));
    // LOG_INFO("unique embeddings: " + std::to_string(uniques.size()));

    std::ofstream out(output_path + "overall_performance.csv", std::ios::app);
    out << data_name << ",P" << local_patternId << ",our_baseline" << "," << total_count << "," << timer.elapsed() << "\n";
    if (do_validation) {
        validation();
    }
    
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
    // int vertices, edges;
    // if (file_format == Input_FileFormat::SNAP_TXT) {
    //     std::ifstream file(data_path);
    //     file >> vertices >> edges;
    //     std::cout << "Vertices: " << vertices << ", Edges: " << edges << std::endl;
    //     edgeLists.resize(vertices);
    //     int source, target;
    //     while (file >> source >> target) {
    //         edgeLists[source].insert(target);
    //         edgeLists[target].insert(source);
    //     }
    //     file.close();
    // }
    
    // print edgeLists
    // for (int i = 0; i < vertices; i++) {
    //     std::cout << "Vertex " << i << " edges: ";
    //     for (int edge : edgeLists[i]) {
    //         std::cout << edge << " ";
    //     }
    //     std::cout << std::endl;
    // }

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

    timer.start();
    if (patternID == XGMinerPatternType::RECTANGLE) {
        // kernel.rectangle4_baseline_cpu_kernel_with_graphpi_sched(vertices, edgeLists, total_count, embedding,
        //                                                         p_edgeList, pairs);
        kernel.rectangle4_baseline_cpu_kernel_with_graphpi_sched_v2(vertices, edgeLists, total_count, embedding,
                                                                p_edgeList, pairs);
    } else {
        LOG_ERROR("Invalid pattern ID.");
    }
    timer.stop();
    LOG_INFO("Elapsed time: " + std::to_string(timer.elapsed()) + " seconds.");
    LOG_INFO("total embeddings: " + std::to_string(total_count));

    std::ofstream out(output_path + "overall_performance.csv", std::ios::app);
    out << data_name << ",P" << local_patternId << ",our_baseline_graphpi_sched" << "," << total_count << "," << timer.elapsed() << "\n";
    if (do_validation) {
        validation();
    }

}
