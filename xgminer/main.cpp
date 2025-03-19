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
#include "glumin.h"


int main(int argc, char *argv[]) {
   
    Command_Option opts(argc, argv);
    // opts.parseByArgParse();
    opts.parseByCxxOpts();

    Logger& logger = Logger::getInstance();
    logger.setConsoleEnabled(true);

    // 第一次设置文件：截断（覆盖）
    // logger.setFile("app.log");
    // LOG_INFO("First message (truncated)");

    // // 切换到新文件：截断
    // logger.setFile("another.log");
    // LOG_INFO("New file message (truncated)");

    if (opts.algo == "cpu_baseline") {
        LOG_INFO("Running CPU baseline algorithm.");
        CPU_Baseline cpu_base(opts);
        cpu_base.run();
    } else if (opts.algo == "glumin_g2miner") {
        LOG_INFO("Running GLUMIN+G2Miner algorithm.");
        GLUMIN glumin(opts);
        glumin.run();
    } else if (opts.algo == "glumin_g2miner_lut") {
        LOG_INFO("Running GLUMIN+G2Miner+LUT algorithm.");
        GLUMIN glumin(opts);
        glumin.run();
    } else {
        LOG_ERROR("Unsupported algorithm: " + opts.algo);
        return 1;
    }


    LOG_INFO("Finish processing.");
    return 0;
}