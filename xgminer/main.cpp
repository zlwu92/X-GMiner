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
#include <unordered_map>
#include "cmd_option.h"
#include "cpu_baseline.h"
#include "log_manager.h"
#include "gpm_solver.cuh"

int main(int argc, char *argv[]) {
   
    Command_Option opts(argc, argv);
    opts.parseByArgParse();

    Logger& logger = Logger::getInstance();
    logger.setConsoleEnabled(true);

    // 第一次设置文件：截断（覆盖）
    // logger.setFile("log/app.log");
    // LOG_INFO("First message (truncated)");

    // // 第二次设置同一个文件：追加
    // logger.setFile("log/app.log");
    // LOG_INFO("Second message (appended)");

    // // 切换到新文件：截断
    // logger.setFile("log/another.log");
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




    std::cout << "Finish processing." << std::endl;
    return 0;
}