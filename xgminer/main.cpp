#include <iostream>
#include <numeric>
#include "cmd_option.h"
#include "cpu_baseline.h"
#include "log_manager.h"

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

    std::cout << "Finish processing." << std::endl;
    return 0;
}