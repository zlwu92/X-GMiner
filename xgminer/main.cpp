#include <iostream>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cassert>
#include <regex>
#include <algorithm>
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
    printf("%ld ",pairs.size());
    for(auto& p : pairs)
        printf("(%d,%d)",p.first,p.second);
    puts("");
    fflush(stdout);

}


int main(int argc, char *argv[]) {
   
    Command_Option opts(argc, argv);
    // opts.parseByArgParse();
    opts.parseByCxxOpts();

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

    if (opts.algo == "cpu_baseline") {
        LOG_INFO("Running CPU baseline algorithm.");
        CPU_Baseline cpu_base(opts);
        cpu_base.run();
    } else {
        LOG_ERROR("Unsupported algorithm: " + opts.algo);
        return 1;
    }
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

    LOG_INFO("Finish processing.");
    return 0;
}